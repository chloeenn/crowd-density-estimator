"""API settings, read from environment variables (or a local `.env`). See `.env.example`."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ALLOWED_CONTENT_TYPES = {"image/jpeg": "JPEG", "image/png": "PNG", "image/webp": "WEBP"}


class Settings(BaseSettings):
    # `model_` is pydantic's protected prefix; these names come from the plan's env contract.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", protected_namespaces=())

    # Models: the default part is served unless a request asks for the other one.
    model_path: Path = Path("models/csrnet-partA.pth")
    model_part: Literal["A", "B"] = "A"
    alt_model_path: Path | None = None  # other part's model; enables auto routing

    auto_route: bool = True  # choose the model per photo when both are loaded (see crowd.routing)

    # Upload limits.
    max_upload_mb: int = 10
    max_long_side: int = 1024  # larger images are downscaled to this (training max)
    max_image_pixels: int = 60_000_000  # decompression-bomb guard (width x height)

    # Concurrency: one model instance, bounded queue, 503 + Retry-After beyond it.
    device: str = "cpu"
    torch_threads: int = 1
    max_concurrent_inference: int = 1
    max_queue: int = 4
    retry_after_s: int = 2

    cors_origins: str = "http://localhost:5173"  # comma-separated; never "*"
    log_level: str = "INFO"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]
