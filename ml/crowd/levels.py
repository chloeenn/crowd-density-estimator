"""Crowd-size levels from the head count.

A count says how big a crowd is, not how tightly it is packed: 200 people are sparse in a
stadium and dangerous in a stairwell. People per square metre needs the ground area, so that
lives in `crowd.area`. These levels compare a count with ShanghaiTech training scenes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

LEVELS = ("Low", "Medium", "High", "Critical")
# Ships next to this module as package data (see [tool.setuptools.package-data]).
THRESHOLDS_PATH = Path(__file__).with_name("thresholds.json")


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
