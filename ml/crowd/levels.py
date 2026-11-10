"""Density levels and image-space concentration.

Count describes crowd *size*; framing changes how dense a scene *looks* (200 people in a
stadium shot is sparse, in a stairwell it is an emergency). So a result carries both a count
level and a concentration value, and neither is a measure of people per square metre.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from crowd.config import OUTPUT_STRIDE

LEVELS = ("Low", "Medium", "High", "Critical")
# Ships next to this module as package data (see [tool.setuptools.package-data]).
THRESHOLDS_PATH = Path(__file__).with_name("thresholds.json")
CONCENTRATION_QUANTILE = 95

CONCENTRATION_UNITS = "people per 8x8-input-pixel cell (image-space, not people/m^2)"
LEVEL_CAVEAT = (
    "Levels are quantiles of ShanghaiTech training counts, a default for demo use. They depend "
    "on camera distance and framing, so compare images of similar scale only."
)


def load_thresholds() -> dict:
    """The calibrated thresholds shipped with the package (see `crowd.calibrate`)."""
    return json.loads(THRESHOLDS_PATH.read_text())


def thresholds_sha() -> str | None:
    """SHA-256 of the shipped thresholds file, recorded in checkpoints; None if absent."""
    if not THRESHOLDS_PATH.is_file():
        return None
    return hashlib.sha256(THRESHOLDS_PATH.read_bytes()).hexdigest()


def level_for_count(count: float, bands: dict[str, float]) -> str:
    """Map a head count to a level. Each band edge is where that band *starts*: a count at or
    above `critical` is Critical, at or above `high` is High, at or above `medium` is Medium,
    and anything lower is Low (`bands["low"]` is the lowest quantile, kept for display)."""
    if count >= bands["critical"]:
        return "Critical"
    if count >= bands["high"]:
        return "High"
    if count >= bands["medium"]:
        return "Medium"
    return "Low"


def concentration(density: np.ndarray) -> tuple[float, float]:
    """(95th percentile, maximum) of per-cell density of a predicted map.

    Each cell covers `OUTPUT_STRIDE` x `OUTPUT_STRIDE` pixels at the scale the model saw, so
    the values are image-space: they change with camera distance, resolution and cropping.
    `density` must already be cropped to the image (no padding cells).
    """
    cells = np.clip(density, 0, None)
    return float(np.percentile(cells, CONCENTRATION_QUANTILE)), float(cells.max())


def concentration_basis(processed_hw: tuple[int, int]) -> dict:
    """Describes the scale `concentration` was measured at, for the API response."""
    h, w = processed_hw
    return {
        "cell_input_px": OUTPUT_STRIDE,
        "processed_width": w,
        "processed_height": h,
        "units": CONCENTRATION_UNITS,
    }
