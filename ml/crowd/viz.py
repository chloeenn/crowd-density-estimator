"""Density-map heatmaps with OpenCV (no matplotlib: this module is also used by the API,
where matplotlib's global figure state is a thread-safety hazard and ~10x slower)."""

from __future__ import annotations

import base64

import cv2
import numpy as np

from crowd.config import OUTPUT_STRIDE

# Viridis: perceptually uniform, readable with the common forms of colour-blindness, and its
# dense end (yellow) cannot be mistaken for the "Critical" red used for the density level in the
# web app.
# The web legend (web/src/components/ScaleLegend.tsx) draws these same stops; test_viz.py pins them.
COLORMAP = cv2.COLORMAP_VIRIDIS
LEGEND_STOPS = ("#440154", "#414487", "#2a788e", "#22a884", "#7ad151", "#fde725")  # sparse -> dense


def crop_to_image(density: np.ndarray, image_hw: tuple[int, int]) -> np.ndarray:
    """Drop the padding that `pad_to_multiple` added, from a 1/8-resolution density map."""
    h, w = image_hw
    return density[: -(-h // OUTPUT_STRIDE), : -(-w // OUTPUT_STRIDE)]


def density_to_heatmap(
    density: np.ndarray, size_hw: tuple[int, int], vmax: float | None = None
) -> tuple[np.ndarray, float]:
    """Colour a density map (dark purple = sparse, yellow = dense) at `size_hw` resolution.

    Normalised by `vmax` (default: the map's own maximum, for contrast). Returns the RGB
    uint8 heatmap and the `vmax` used, so a caller can label the colour scale or reuse the
    same scale to compare two maps.
    """
    d = np.clip(density, 0, None).astype(np.float32)
    vmax = float(d.max()) if vmax is None else float(vmax)
    norm = d / vmax if vmax > 0 else np.zeros_like(d)
    h, w = size_hw
    up = cv2.resize(norm, (w, h), interpolation=cv2.INTER_CUBIC)  # cubic can overshoot
    heat = cv2.applyColorMap((np.clip(up, 0, 1) * 255).astype(np.uint8), COLORMAP)
    return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB), vmax


def overlay_heatmap(
    image: np.ndarray, density: np.ndarray, alpha: float = 0.4, vmax: float | None = None
) -> tuple[np.ndarray, float]:
    """Blend the heatmap over an RGB uint8 image. `density` must already be cropped to the
    image (see `crop_to_image`). Returns (RGB uint8 overlay, vmax used)."""
    heat, vmax = density_to_heatmap(density, image.shape[:2], vmax)
    return cv2.addWeighted(image, 1.0 - alpha, heat, alpha, 0.0), vmax


def draw_label(image: np.ndarray, lines: list[str]) -> np.ndarray:
    """Burn a text label (one string per line) onto the top-left of an RGB uint8 image.

    White text on a dark translucent panel so it stays readable on any background. The font
    scales with the image so small and large photos both get a legible label. Returns a copy.
    """
    out = image.copy()
    if not lines:
        return out
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.4, min(h, w) / 700)
    thick = max(1, round(scale * 2))
    pad = max(4, round(10 * scale))
    sizes = [cv2.getTextSize(t, font, scale, thick)[0] for t in lines]
    line_h = max(sz[1] for sz in sizes) + pad
    box_w = min(w, max(sz[0] for sz in sizes) + 2 * pad)
    box_h = min(h, line_h * len(lines) + pad)
    panel = out[:box_h, :box_w]
    out[:box_h, :box_w] = cv2.addWeighted(panel, 0.4, np.zeros_like(panel), 0.6, 0.0)
    for i, text in enumerate(lines):
        y = pad + line_h * i + sizes[i][1]
        cv2.putText(out, text, (pad, y), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return out


def to_png_data_uri(rgb: np.ndarray, compression: int = 3) -> str:
    """Encode an RGB uint8 image as an inline `data:image/png;base64,...` URI.

    Low PNG compression: encoding time matters more than size here, and the result is
    embedded in a JSON response (plan: inline base64 in v1).
    """
    ok, buf = cv2.imencode(
        ".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, compression]
    )
    if not ok:
        raise ValueError("PNG encoding failed")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
