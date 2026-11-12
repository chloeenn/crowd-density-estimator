"""People per square metre: the one density measure that does not depend on camera framing.

The model counts heads. A count becomes a density only when the ground area it stands on is
known, so the caller supplies that area (in m^2) and, optionally, a polygon that marks where
on the photo that area is. The count is then taken inside the polygon.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

# Crowd-safety bands in people per m^2 (after Fruin's levels of service). A band starts at its edge.
DENSITY_BANDS = {"medium": 1.0, "high": 2.0, "critical": 4.0}
DENSITY_UNITS = "people per m^2"

MIN_POLYGON_POINTS = 3
MAX_POLYGON_POINTS = 64


def parse_region(raw: list[list[float]]) -> list[tuple[float, float]]:
    """Validate a polygon given as [[x, y], ...] with x and y in 0..1 (fractions of the image).

    Raises ValueError for fewer than three points, points outside the image, or no area.
    """
    if not MIN_POLYGON_POINTS <= len(raw) <= MAX_POLYGON_POINTS:
        raise ValueError(
            f"region needs {MIN_POLYGON_POINTS} to {MAX_POLYGON_POINTS} points, got {len(raw)}"
        )
    pts = []
    for p in raw:
        if len(p) != 2 or not all(0.0 <= float(v) <= 1.0 for v in p):
            raise ValueError("region points must be [x, y] with both values between 0 and 1")
        pts.append((float(p[0]), float(p[1])))
    xs, ys = zip(*pts, strict=True)
    twice_area = abs(
        sum(
            xs[i] * ys[(i + 1) % len(pts)] - xs[(i + 1) % len(pts)] * ys[i] for i in range(len(pts))
        )
    )
    if twice_area < 1e-6:
        raise ValueError("region has no area")
    return pts


def region_mask(points: list[tuple[float, float]], shape: tuple[int, int], scale: int = 4):
    """Fraction (0..1) of each density cell covered by the polygon, shape (h, w).

    Drawn at `scale` times the cell grid and averaged down, so a polygon edge splits a cell's
    count in proportion instead of snapping to whole cells.
    """
    h, w = shape
    big = Image.new("L", (w * scale, h * scale), 0)
    # PIL fills edge pixels inclusively and pixel i is centred at i + 0.5, hence the -0.5.
    corners = [(x * w * scale - 0.5, y * h * scale - 0.5) for x, y in points]
    ImageDraw.Draw(big).polygon(corners, fill=255)
    arr = np.asarray(big, dtype=np.float32) / 255.0
    return arr.reshape(h, scale, w, scale).mean(axis=(1, 3))


def count_in_region(density: np.ndarray, points: list[tuple[float, float]] | None) -> float:
    """Predicted people inside the polygon, or in the whole image when there is none."""
    cells = np.clip(density, 0, None)
    if points is None:
        return float(cells.sum())
    return float((cells * region_mask(points, cells.shape)).sum())


def people_per_m2(count: float, area_m2: float) -> float:
    if area_m2 <= 0:
        raise ValueError("area must be positive")
    return count / area_m2


def level_for_density(per_m2: float, bands: dict[str, float] | None = None) -> str:
    """Low / Medium / High / Critical from people per m^2."""
    bands = bands or DENSITY_BANDS
    if per_m2 >= bands["critical"]:
        return "Critical"
    if per_m2 >= bands["high"]:
        return "High"
    if per_m2 >= bands["medium"]:
        return "Medium"
    return "Low"
