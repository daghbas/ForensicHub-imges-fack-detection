"""Inference helpers for the Flask forgery detection demo."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

from ForensicHub.registry import MODELS, build_from_registry


@dataclass
class ServiceConfig:
    """Configuration for the :class:`ForgeryDetectionService`."""

    model_name: str
    checkpoint_path: Path
    image_size: int = 512
    threshold: float = 0.5
    device: Optional[str] = None
    mean: Iterable[float] = (0.485, 0.456, 0.406)
    std: Iterable[float] = (0.229, 0.224, 0.225)
    additional_inputs: Optional[Mapping[str, Any]] = None

    @classmethod
    def from_kwargs(
        cls,
        *,
        model_name: str,
        checkpoint_path: str | Path,
        image_size: int = 512,
        threshold: float = 0.5,
        device: Optional[str] = None,
        mean: Iterable[float] = (0.485, 0.456, 0.406),
        std: Iterable[float] = (0.229, 0.224, 0.225),
        additional_inputs: Optional[Mapping[str, Any]] = None,
    ) -> "ServiceConfig":
        return cls(
            model_name=model_name,
            checkpoint_path=Path(checkpoint_path).expanduser().resolve(),
            image_size=image_size,
            threshold=threshold,
            device=device,
            mean=mean,
            std=std,
            additional_inputs=additional_inputs,
        )


class ForgeryDetectionService:
    """Run inference and localisation using a pretrained ForensicHub model."""

    def __init__(self, config: ServiceConfig):
        self.config = config
        self.device = torch.device(
            config.device if config.device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
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
        return mask.cpu()

    def _build_overlay(self, image: Image.Image, mask: torch.Tensor) -> str:
        resized_mask = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0),
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        mask_min = float(resized_mask.min())
        mask_max = float(resized_mask.max())
        normalised = resized_mask
        if mask_max > mask_min:
            normalised = (resized_mask - mask_min) / (mask_max - mask_min)
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
