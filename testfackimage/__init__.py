"""Flask application factory for the forgery detection demo."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Optional

from flask import Flask, render_template, request
from PIL import Image

from .service import ForgeryDetectionService, ServiceConfig

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.json"
DEFAULT_WEIGHTS_PATH = PROJECT_ROOT / "best_model_1.pth"


def _load_config(config_path: Optional[Path] = None) -> ServiceConfig:
    path = config_path or DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found at {path}. Please create config.json before running."
        )
    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)

    model_name = payload["model_name"]
    image_size = int(payload.get("image_size", 512))
    threshold = float(payload.get("threshold", 0.5))
    device = payload.get("device")
    mean = payload.get("mean", [0.485, 0.456, 0.406])
    std = payload.get("std", [0.229, 0.224, 0.225])
    additional_inputs = payload.get("additional_inputs")

    checkpoint_path = Path(payload.get("checkpoint_path", DEFAULT_WEIGHTS_PATH))
    if not checkpoint_path.is_absolute():
        checkpoint_path = (PROJECT_ROOT / checkpoint_path).resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Model checkpoint not found. Expected to locate best_model_1.pth in the project root."
        )

    return ServiceConfig.from_kwargs(
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        image_size=image_size,
        threshold=threshold,
        device=device,
        mean=mean,
        std=std,
        additional_inputs=additional_inputs,
    )


def _encode_image(image: Image.Image) -> str:
    with io.BytesIO() as buffer:
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


service_instance: Optional[ForgeryDetectionService] = None


def create_app(config_path: Optional[Path] = None) -> Flask:
    global service_instance
    if service_instance is None:
        config = _load_config(config_path)
        service_instance = ForgeryDetectionService(config)

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB uploads
    app.secret_key = "forensic-hub-demo"

    @app.route("/", methods=["GET", "POST"])
    def index():
        error = None
        probability = None
        overlay_data = None
        original_data = None

        if request.method == "POST":
            file = request.files.get("image")
            if file is None or file.filename == "":
                error = "يرجى اختيار صورة لتحميلها."
            else:
                try:
                    image = Image.open(file.stream).convert("RGB")
                    original_data = _encode_image(image)
                    results = service_instance.predict(image)
                    probability = results.get("probability")
                    overlay_data = results.get("overlay")
                    if overlay_data is None:
                        error = "هذا النموذج لا ينتج خريطة مناطق التلاعب."  # informative message
                except Exception as exc:  # pragma: no cover - runtime guard
                    error = f"تعذر تحليل الصورة: {exc}"

        return render_template(
            "index.html",
            error=error,
            probability=probability,
            overlay_data=overlay_data,
            original_data=original_data,
        )

    return app


__all__ = ["create_app"]
