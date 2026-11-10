"""Read ShanghaiTech ground-truth `.mat` files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io as sio


def gt_path_for(image_path: Path) -> Path:
    """IMG_12.jpg -> ../ground-truth/GT_IMG_12.mat"""
    return image_path.parent.parent / "ground-truth" / f"GT_{image_path.stem}.mat"


def load_points(mat_path) -> np.ndarray:
    """Return head annotations as float32 (N, 2) of (x, y) pixel coordinates."""
    mat = sio.loadmat(str(mat_path))
    pts = mat["image_info"][0, 0][0, 0]["location"]
    return np.asarray(pts, dtype=np.float32).reshape(-1, 2)


def clip_points(points: np.ndarray, height: int, width: int) -> np.ndarray:
    """Clip points into the image so none is silently dropped.

    A few annotations sit a fraction of a pixel outside the frame; indexing with them
    would wrap or lose a head.
    """
    out = np.asarray(points, dtype=np.float32).reshape(-1, 2).copy()
    out[:, 0] = np.clip(out[:, 0], 0, width - 1)
    out[:, 1] = np.clip(out[:, 1], 0, height - 1)
    return out
