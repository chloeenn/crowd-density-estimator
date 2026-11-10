"""FastAPI dependencies: access to app-wide state with a clean failure mode."""

from __future__ import annotations

from fastapi import Request

from app.config import Settings
from app.errors import ApiError
from app.inference import InferenceService


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_service(request: Request) -> InferenceService:
    """The loaded model registry, or 503 while models are not loaded."""
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise ApiError(503, "not_ready", "Models are not loaded yet.", headers={"Retry-After": "5"})
    return service
