"""POST /api/predict"""

from __future__ import annotations

import asyncio
import json
from time import perf_counter
from typing import Annotated, Literal

from crowd.area import (
    DENSITY_BANDS,
    DENSITY_UNITS,
    count_in_region,
    level_for_density,
    parse_region,
    people_per_m2,
)
from crowd.levels import level_for_count, load_thresholds
from crowd.viz import overlay_heatmap, to_png_data_uri
from fastapi import APIRouter, Depends, File, Query, Request, UploadFile

from app.config import Settings
from app.deps import get_service, get_settings
from app.errors import ApiError
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
    part: Annotated[
        Literal["A", "B"] | None,
        Query(description="Force one model. Omit to choose by scene automatically"),
    ] = None,
    area_m2: Annotated[
        float | None,
        Query(gt=0, le=1e7, description="Ground area in m^2 (of the region, if one is given)"),
    ] = None,
    region: Annotated[
        str | None,
        Query(description="Polygon [[x, y], ...] with x, y in 0..1 marking where that area is"),
    ] = None,
) -> PredictResponse:
    if part is not None:
        service.get(part)  # reject an unloaded model before reading the upload
    points = _parse_region(region, area_m2)
    check_content_type(file)
    data = await read_bytes(file, settings)

    async with service.slot():
        image = await asyncio.to_thread(decode_image, data, settings)
        start = perf_counter()
        predictor, prediction, routed = await asyncio.to_thread(service.predict, image, part)
        inference_ms = round((perf_counter() - start) * 1000)
    request.state.inference_ms = inference_ms
    request.state.part = predictor.part

    return await asyncio.to_thread(
        build_response, predictor, prediction, inference_ms, routed, area_m2, points
    )


def _parse_region(region: str | None, area_m2: float | None):
    if region is None:
        return None
    if area_m2 is None:
        raise ApiError(422, "area_required", "A marked region needs its area in m^2.")
    try:
        return parse_region(json.loads(region))
    except (ValueError, TypeError) as e:  # bad JSON, shape or values
        raise ApiError(422, "bad_region", f"Invalid region: {e}") from e


def build_response(
    predictor,
    prediction,
    inference_ms: int,
    routed: bool = False,
    area_m2: float | None = None,
    points=None,
) -> PredictResponse:
    """Turn a `Prediction` into the response body (levels, density, heatmap)."""
    thresholds = load_thresholds()
    overlay, _ = overlay_heatmap(prediction.image, prediction.density, alpha=0.4)
    density = None
    if area_m2 is not None:
        in_area = count_in_region(prediction.density, points)
        per_m2 = people_per_m2(in_area, area_m2)
        density = {
            "people_per_m2": round(per_m2, 2),
            "level": level_for_density(per_m2),
            "area_m2": area_m2,
            "count": round(in_area, 1),
            "in_region": points is not None,
            "bands": DENSITY_BANDS,
            "units": DENSITY_UNITS,
        }
    return PredictResponse(
        density=density,
        routed=routed,
        count=round(prediction.count, 1),
        level=level_for_count(prediction.count, thresholds["bands"]),
        thresholds=thresholds["bands"],
        model={
            "version": predictor.version,
            "part": predictor.part,
            "git_sha": predictor.meta["git_sha"],
        },
        heatmap=to_png_data_uri(overlay),
        inference_ms=inference_ms,
    )
