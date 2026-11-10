"""GET /api/health and GET /api/metadata"""

from __future__ import annotations

from typing import Annotated

from crowd.levels import LEVEL_CAVEAT, load_thresholds, thresholds_sha
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from app.config import ALLOWED_CONTENT_TYPES, Settings
from app.deps import get_service, get_settings
from app.inference import InferenceService
from app.schemas import HealthResponse, Limits, MetadataResponse, ModelDetail

router = APIRouter()


@router.get("/health", response_model=HealthResponse, responses={503: {"model": HealthResponse}})
async def health(request: Request):
    """200 only when every configured model is loaded; 503 otherwise (a Docker healthcheck
    can rely on this)."""
    service = getattr(request.app.state, "service", None)
    if service is None:
        body = HealthResponse(status="unavailable", models=[])
        return JSONResponse(body.model_dump(), status_code=503)
    return HealthResponse(status="ok", models=sorted(service.predictors))


@router.get("/metadata", response_model=MetadataResponse)
async def metadata(
    service: Annotated[InferenceService, Depends(get_service)],
    settings: Annotated[Settings, Depends(get_settings)],
):
    thresholds = load_thresholds()
    current = thresholds_sha()
    models = []
    for part in sorted(service.predictors):
        meta = service.predictors[part].meta
        sha = meta.get("thresholds_sha")
        models.append(
            ModelDetail(
                version=meta["model_version"],
                part=meta["part"],
                git_sha=meta["git_sha"],
                run_id=meta["run_id"],
                epoch=meta["epoch"],
                val_mae=meta.get("val_mae"),
                val_rmse=meta.get("val_rmse"),
                thresholds_match=None if sha is None else sha == current,
            )
        )
    info_keys = ("split", "n_images", "parts_pooled", "validation_seed", "calibrated")
    return MetadataResponse(
        default_part=service.default_part,
        models=models,
        thresholds=thresholds["bands"],
        thresholds_info={k: thresholds[k] for k in info_keys},
        level_caveat=LEVEL_CAVEAT,
        limits=Limits(
            max_upload_mb=settings.max_upload_mb,
            max_long_side=settings.max_long_side,
            max_image_pixels=settings.max_image_pixels,
            allowed_content_types=sorted(ALLOWED_CONTENT_TYPES),
        ),
    )
