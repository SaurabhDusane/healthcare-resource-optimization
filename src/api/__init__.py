"""FastAPI service exposing forecasts, predictions and early-warning alerts."""

from .app import create_app

__all__ = ["create_app"]
