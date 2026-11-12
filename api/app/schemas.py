"""Response models (also the OpenAPI contract the frontend codes against)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class DensityBands(BaseModel):
    medium: float
    high: float
    critical: float


class Thresholds(BaseModel):
    low: int
    medium: int
    high: int
    critical: int


class DensityResult(BaseModel):
    """People per m^2, present only when the request gave the ground area."""

    people_per_m2: float
    level: Literal["Low", "Medium", "High", "Critical"]
    area_m2: float
    count: float  # people inside the marked region (the whole photo when none was marked)
    in_region: bool
    bands: DensityBands
    units: str


class ModelInfo(BaseModel):
    version: str
    part: str
    git_sha: str


class PredictResponse(BaseModel):
    count: float
    level: Literal["Low", "Medium", "High", "Critical"]
    thresholds: Thresholds
    density: DensityResult | None
    routed: bool  # the scene model was chosen automatically
    model: ModelInfo
    heatmap: str  # data:image/png;base64,...
    inference_ms: int


class HealthResponse(BaseModel):
    status: Literal["ok", "unavailable"]
    models: list[str]


class ModelDetail(ModelInfo):
    run_id: str
    epoch: int
    val_mae: float | None
    val_rmse: float | None
    thresholds_match: bool | None  # checkpoint was trained against these level bands


class Limits(BaseModel):
    max_upload_mb: int
    max_long_side: int
    max_image_pixels: int
    allowed_content_types: list[str]


class MetadataResponse(BaseModel):
    default_part: str
    models: list[ModelDetail]
    thresholds: Thresholds
    thresholds_info: dict
    density_bands: DensityBands
    routing_threshold: float | None
    limits: Limits
