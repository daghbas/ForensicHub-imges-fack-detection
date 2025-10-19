"""High-level application entrypoints built on top of ForensicHub models."""

from .forgery_web_app import create_app, ForgeryDetectionService

__all__ = ["create_app", "ForgeryDetectionService"]
