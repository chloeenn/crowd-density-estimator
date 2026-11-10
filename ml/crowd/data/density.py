"""Turn head-point annotations into density maps whose sum equals the head count."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from crowd.config import DensityConfig
from crowd.data.annotations import clip_points


def compute_sigmas(points: np.ndarray, cfg: DensityConfig) -> np.ndarray:
    """Per-head Gaussian sigma, shape (N,).

    Adaptive: sigma_i = beta * mean distance to the k nearest other heads (tight kernels
    in crowds, wide ones in sparse regions). Falls back to `fixed_sigma` when there are
    too few heads to measure a neighbour distance.
    """
    n = len(points)
    if not cfg.adaptive or n < 2:
        return np.full(n, cfg.fixed_sigma, dtype=np.float32)
    k = min(cfg.k, n - 1)
    dists, _ = cKDTree(points).query(points, k=k + 1)  # column 0 is the point itself
    sigmas = cfg.beta * dists[:, 1:].mean(axis=1)
    return np.maximum(sigmas, cfg.min_sigma).astype(np.float32)


def density_map(points: np.ndarray, height: int, width: int, cfg: DensityConfig) -> np.ndarray:
    """Build a float32 (H, W) density map from (N, 2) (x, y) points.

    Each head contributes a Gaussian that is renormalised after truncation to the image,
    so every head adds exactly 1.0 and `density.sum() == N` even for heads at the border
    (a plain gaussian_filter would leak mass off the edge there).
    """
    dmap = np.zeros((height, width), dtype=np.float32)
    if len(points) == 0:
        return dmap
    pts = clip_points(points, height, width)
    sigmas = compute_sigmas(pts, cfg)
    for (x, y), sigma in zip(pts, sigmas):
        radius = int(cfg.truncate * float(sigma) + 0.5)
        cx, cy = int(round(float(x))), int(round(float(y)))
        x0, x1 = max(cx - radius, 0), min(cx + radius + 1, width)
        y0, y1 = max(cy - radius, 0), min(cy + radius + 1, height)
        gx = np.exp(-0.5 * ((np.arange(x0, x1) - float(x)) / sigma) ** 2)
        gy = np.exp(-0.5 * ((np.arange(y0, y1) - float(y)) / sigma) ** 2)
        patch = np.outer(gy, gx)
        dmap[y0:y1, x0:x1] += (patch / patch.sum()).astype(np.float32)
    return dmap
