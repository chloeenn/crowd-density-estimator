"""Response models (also the OpenAPI contract the frontend codes against)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class ConcentrationBasis(BaseModel):
    cell_input_px: int
    processed_width: int
    processed_height: int
    units: str


class Thresholds(BaseModel):
    low: int
    medium: int
    high: int
    critical: int


class ModelInfo(BaseModel):
    version: str
    part: str
    git_sha: str


class PredictResponse(BaseModel):
    count: float
    level: Literal["Low", "Medium", "High", "Critical"]
    local_concentration: float
    max_concentration: float
    concentration_basis: ConcentrationBasis
    thresholds: Thresholds
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
    level_caveat: str
    limits: Limits
