"""Density-map heatmaps with OpenCV (no matplotlib: this module is also used by the API,
where matplotlib's global figure state is a thread-safety hazard and ~10x slower)."""

from __future__ import annotations

import cv2
import numpy as np

from crowd.config import OUTPUT_STRIDE


def crop_to_image(density: np.ndarray, image_hw: tuple[int, int]) -> np.ndarray:
    """Drop the padding that `pad_to_multiple` added, from a 1/8-resolution density map."""
    h, w = image_hw
    return density[: -(-h // OUTPUT_STRIDE), : -(-w // OUTPUT_STRIDE)]


def density_to_heatmap(
    density: np.ndarray, size_hw: tuple[int, int], vmax: float | None = None
) -> tuple[np.ndarray, float]:
    """Colour a density map (blue = sparse, red = dense) at `size_hw` resolution.

    Normalised by `vmax` (default: the map's own maximum, for contrast). Returns the RGB
    uint8 heatmap and the `vmax` used, so a caller can label the colour scale or reuse the
    same scale to compare two maps.
    """
    d = np.clip(density, 0, None).astype(np.float32)
    vmax = float(d.max()) if vmax is None else float(vmax)
    norm = d / vmax if vmax > 0 else np.zeros_like(d)
    h, w = size_hw
    up = cv2.resize(norm, (w, h), interpolation=cv2.INTER_CUBIC)  # cubic can overshoot
    heat = cv2.applyColorMap((np.clip(up, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB), vmax


def overlay_heatmap(
    image: np.ndarray, density: np.ndarray, alpha: float = 0.4, vmax: float | None = None
) -> tuple[np.ndarray, float]:
    """Blend the heatmap over an RGB uint8 image. `density` must already be cropped to the
    image (see `crop_to_image`). Returns (RGB uint8 overlay, vmax used)."""
    heat, vmax = density_to_heatmap(density, image.shape[:2], vmax)
    return cv2.addWeighted(image, 1.0 - alpha, heat, alpha, 0.0), vmax
