from __future__ import annotations

from pathlib import Path

from flask import Flask

from . import create_app


def build_app() -> Flask:
    config_path = Path(__file__).resolve().parent / "config.json"
    return create_app(config_path)


app = build_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
