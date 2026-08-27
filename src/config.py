"""
Centralized Runtime Configuration
=================================

A single source of truth for environment-driven settings that were previously
scattered across modules (artifact paths, API authentication, rate limiting,
server host/port). Everything is read from environment variables with sensible
defaults, so the app runs out of the box and is configured for production purely
through the environment.

Usage::

    from src.config import get_settings
    settings = get_settings()
    settings.models_dir           # -> "models"
    settings.api_key              # -> None unless API_KEY is set
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    """Immutable, environment-driven application settings."""

    # Artifact locations (shared by the pipeline and the API).
    models_dir: str = os.environ.get("MODELS_DIR", "models")
    processed_dir: str = os.environ.get("PROCESSED_DIR", "data/processed")
    reports_dir: str = os.environ.get("REPORTS_DIR", "reports")

    # API security. When api_key is None the API is open (local/demo mode);
    # set API_KEY to require an ``X-API-Key`` header on protected endpoints.
    api_key: Optional[str] = os.environ.get("API_KEY") or None

    # Rate limiting: max requests per client per rolling window (0 disables).
    rate_limit_per_minute: int = _get_int("RATE_LIMIT_PER_MINUTE", 120)

    # Server bind (used by tooling / docs; uvicorn CLI can override).
    host: str = os.environ.get("HOST", "0.0.0.0")
    port: int = _get_int("PORT", 8000)

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_key)

    @property
    def rate_limiting_enabled(self) -> bool:
        return self.rate_limit_per_minute > 0


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton (cached)."""
    return Settings()
