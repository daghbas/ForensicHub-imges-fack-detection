"""FastAPI application for interactive image forgery inspection.

The goal of this module is to provide a ready-to-run web demo that
allows a user to upload an image, run a pretrained ForensicHub model
without any further training, and visualise both the predicted class
(probability of manipulation) and a localisation heatmap highlighting
forged regions.

Example
-------
```
uvicorn ForensicHub.applications.forgery_web_app:create_app --factory --port 8000 \
    --log-level info --reload
```

The application expects the caller to provide minimal configuration
through environment variables or keyword arguments when constructing the
:class:`ForgeryDetectionService`.  This keeps the dependency surface
small while making the example practical for real-world usage.
"""

from __future__ import annotations

import base64
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

from ForensicHub.registry import MODELS, build_from_registry


def _infer_device(device: Optional[str] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_checkpoint(path: str | os.PathLike[str]) -> Path:
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return checkpoint_path


@dataclass
class ServiceConfig:
    """Configuration required to bootstrap the web service."""

    model_name: str
    checkpoint_path: Path
    image_size: int = 512
    threshold: float = 0.5
    device: Optional[str] = None
    mean: Iterable[float] = (0.485, 0.456, 0.406)
    std: Iterable[float] = (0.229, 0.224, 0.225)
    additional_inputs: Optional[Mapping[str, Any]] = None

    @classmethod
    def from_environment(cls, prefix: str = "FORHUB_") -> "ServiceConfig":
        """Create configuration from environment variables."""

        env = os.environ
        try:
            model_name = env[f"{prefix}MODEL"]
            checkpoint_path = _resolve_checkpoint(env[f"{prefix}CHECKPOINT"])
        except KeyError as exc:  # pragma: no cover - explicit error path
            raise RuntimeError(
                "Environment variables FORHUB_MODEL and FORHUB_CHECKPOINT must be set"
            ) from exc

        image_size = int(env.get(f"{prefix}IMAGE_SIZE", 512))
        threshold = float(env.get(f"{prefix}THRESHOLD", 0.5))
        device = env.get(f"{prefix}DEVICE")

        mean = tuple(
            float(x) for x in env.get(f"{prefix}MEAN", "0.485,0.456,0.406").split(",")
        )
        std = tuple(
            float(x) for x in env.get(f"{prefix}STD", "0.229,0.224,0.225").split(",")
        )

        extra_inputs = env.get(f"{prefix}STATIC_INPUTS")
        additional_inputs = None
        if extra_inputs:
            try:
                parsed = json.loads(extra_inputs)
                if not isinstance(parsed, MutableMapping):
                    raise ValueError("STATIC_INPUTS must be a JSON object")
                additional_inputs = parsed
            except json.JSONDecodeError as exc:
                raise ValueError("Unable to parse STATIC_INPUTS as JSON") from exc

        return cls(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            image_size=image_size,
            threshold=threshold,
            device=device,
            mean=mean,
            std=std,
            additional_inputs=additional_inputs,
        )


class ForgeryDetectionService:
    """Utility responsible for running inference and visualisation."""

    def __init__(self, config: ServiceConfig):
        self.config = config
        self.device = _infer_device(config.device)
        self.model = self._load_model()
        self.transform = transforms.Compose(
            [
                transforms.Resize((config.image_size, config.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=config.mean, std=config.std),
            ]
        )

    def _load_model(self) -> torch.nn.Module:
        model_kwargs = {
            "name": self.config.model_name,
            "init_path": str(self.config.checkpoint_path),
            "init_config": {"image_size": self.config.image_size},
        }
        model = build_from_registry(MODELS, model_kwargs)
        checkpoint = torch.load(self.config.checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("model", checkpoint)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def _prepare_inputs(self, image_tensor: torch.Tensor) -> Dict[str, Any]:
        inputs: Dict[str, Any] = {"image": image_tensor}
        if self.config.additional_inputs:
            for key, value in self.config.additional_inputs.items():
                tensor = torch.as_tensor(value)
                if tensor.ndim == 0:
                    tensor = tensor.unsqueeze(0)
                inputs[key] = tensor.to(self.device)
        return inputs

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        inputs = self._prepare_inputs(tensor)
        with torch.no_grad():
            outputs = self.model(**inputs)

        probability = self._extract_probability(outputs)
        mask = self._extract_mask(outputs)
        overlay = None
        if mask is not None:
            overlay = self._build_overlay(image, mask)

        return {
            "probability": probability,
            "mask": mask,
            "overlay": overlay,
        }

    def _extract_probability(self, outputs: Mapping[str, Any]) -> Optional[float]:
        prediction = None
        if "pred_label" in outputs:
            prediction = outputs["pred_label"]
        elif "pred" in outputs:
            prediction = outputs["pred"]

        if prediction is None:
            return None

        tensor = torch.as_tensor(prediction).detach().float()
        if tensor.ndim == 0:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim > 1:
            tensor = tensor.flatten()
        prob = torch.sigmoid(tensor[0])
        return float(prob.cpu().item())

    def _extract_mask(self, outputs: Mapping[str, Any]) -> Optional[torch.Tensor]:
        if "pred_mask" not in outputs:
            return None
        mask = torch.as_tensor(outputs["pred_mask"]).detach().float()
        while mask.ndim < 4:
            mask = mask.unsqueeze(0)
        mask = mask[0]
        if mask.shape[0] > 1:
            mask = mask[0]
        mask = mask.sigmoid() if mask.min() < 0 or mask.max() > 1 else mask
        mask = mask.cpu()
        return mask

    def _build_overlay(self, image: Image.Image, mask: torch.Tensor) -> str:
        resized_mask = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0),
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        normalised = resized_mask
        mask_min = float(normalised.min())
        mask_max = float(normalised.max())
        if mask_max > mask_min:
            normalised = (normalised - mask_min) / (mask_max - mask_min)
        normalised = normalised.clamp(0, 1).cpu().numpy()

        image_np = np.array(image).astype(np.float32) / 255.0
        heat_map = np.zeros_like(image_np)
        heat_map[..., 0] = normalised
        overlay = (1 - self.config.threshold) * image_np + self.config.threshold * heat_map
        overlay = np.clip(overlay, 0, 1)
        overlay_img = Image.fromarray((overlay * 255).astype(np.uint8))
        return self._encode_image(overlay_img)

    @staticmethod
    def _encode_image(image: Image.Image) -> str:
        with io.BytesIO() as buffer:
            image.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _render_html(probability: Optional[float], overlay: Optional[str]) -> str:
    probability_text = (
        f"<p><strong>Manipulation probability:</strong> {probability:.2%}</p>"
        if probability is not None
        else "<p><em>No probability output available for this model.</em></p>"
    )

    overlay_img = (
        f'<img src="data:image/png;base64,{overlay}" alt="Manipulation heatmap" />'
        if overlay is not None
        else "<p><em>Model did not return a localisation mask.</em></p>"
    )

    return f"""
    <html>
        <head>
            <title>ForensicHub Forgery Inspector</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 2rem; }}
                .container {{ max-width: 720px; margin: auto; }}
                .result {{ margin-top: 2rem; }}
                img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Image Forgery Inspector</h1>
                <form action="/analyze" method="post" enctype="multipart/form-data">
                    <input type="file" name="file" accept="image/*" required>
                    <button type="submit">Analyze</button>
                </form>
                <div class="result">
                    {probability_text}
                    {overlay_img}
                </div>
            </div>
        </body>
    </html>
    """


def create_app(config: Optional[ServiceConfig] = None) -> FastAPI:
    """Factory that creates and configures the FastAPI app."""

    if config is None:
        config = ServiceConfig.from_environment()
    service = ForgeryDetectionService(config)
    app = FastAPI(title="ForensicHub Forgery Inspector")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse(_render_html(None, None))

    @app.post("/analyze")
    async def analyze(file: UploadFile = File(...)) -> HTMLResponse:
        try:
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:  # pragma: no cover - protective clause
            raise HTTPException(status_code=400, detail="Invalid image file") from exc

        results = service.predict(image)
        return HTMLResponse(
            _render_html(results.get("probability"), results.get("overlay"))
        )

    @app.post("/api/analyze")
    async def api_analyze(
        file: UploadFile = File(...),
        return_mask: bool = Form(False),
    ) -> JSONResponse:
        image_bytes = await file.read()
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:  # pragma: no cover - protective clause
            raise HTTPException(status_code=400, detail="Invalid image file") from exc

        results = service.predict(image)
        payload: Dict[str, Any] = {"probability": results.get("probability")}
        if return_mask and results.get("overlay"):
            payload["overlay"] = results["overlay"]
        return JSONResponse(payload)

    return app


__all__ = ["ServiceConfig", "ForgeryDetectionService", "create_app"]
