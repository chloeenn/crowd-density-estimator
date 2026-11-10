"""POST /api/predict"""

from __future__ import annotations

import asyncio
from time import perf_counter
from typing import Annotated, Literal

from crowd.levels import concentration, concentration_basis, level_for_count, load_thresholds
from crowd.viz import overlay_heatmap, to_png_data_uri
from fastapi import APIRouter, Depends, File, Query, Request, UploadFile

from app.config import Settings
from app.deps import get_service, get_settings
from app.inference import InferenceService
from app.schemas import PredictResponse
from app.uploads import check_content_type, decode_image, read_bytes

router = APIRouter()

ERROR_RESPONSES = {
    400: {"description": "Not a valid or complete image"},
    413: {"description": "File or image too large"},
    415: {"description": "Not a JPEG, PNG or WebP image"},
    422: {"description": "Bad parameter, or requested model not loaded"},
    503: {"description": "Busy (see Retry-After) or model not ready"},
}


@router.post("/predict", response_model=PredictResponse, responses=ERROR_RESPONSES)
async def predict(
    request: Request,
    service: Annotated[InferenceService, Depends(get_service)],
    settings: Annotated[Settings, Depends(get_settings)],
    file: Annotated[UploadFile, File(description="JPEG, PNG or WebP crowd photo")],
    part: Annotated[Literal["A", "B"] | None, Query(description="Model to use")] = None,
    scale_max: Annotated[
        float | None,
        Query(
            gt=0, le=1e6, description="Fixed heatmap scale (people per cell) for comparing images"
        ),
    ] = None,
) -> PredictResponse:
    predictor = service.get(part)
    check_content_type(file)
    data = await read_bytes(file, settings)

    async with service.slot():
        image = await asyncio.to_thread(decode_image, data, settings)
        start = perf_counter()
        prediction = await asyncio.to_thread(predictor.predict, image)
        inference_ms = round((perf_counter() - start) * 1000)
    request.state.inference_ms = inference_ms
    request.state.part = predictor.part

    return await asyncio.to_thread(build_response, predictor, prediction, scale_max, inference_ms)


def build_response(
    predictor, prediction, scale_max: float | None, inference_ms: int
) -> PredictResponse:
    """Turn a `Prediction` into the response body (levels, concentration, heatmap)."""
    thresholds = load_thresholds()
    local, peak = concentration(prediction.density)
    overlay, _ = overlay_heatmap(prediction.image, prediction.density, alpha=0.4, vmax=scale_max)
    return PredictResponse(
        count=round(prediction.count, 1),
        level=level_for_count(prediction.count, thresholds["bands"]),
        local_concentration=round(local, 4),
        max_concentration=round(peak, 4),
        concentration_basis=concentration_basis(prediction.image.shape[:2]),
        thresholds=thresholds["bands"],
        model={
            "version": predictor.version,
            "part": predictor.part,
            "git_sha": predictor.meta["git_sha"],
        },
        heatmap=to_png_data_uri(overlay),
        inference_ms=inference_ms,
    )
