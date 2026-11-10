"""Shared test helpers."""

from pathlib import Path

import pytest

RAW = Path(__file__).resolve().parents[2] / "data" / "ShanghaiTech"

needs_data = pytest.mark.skipif(not RAW.exists(), reason="ShanghaiTech dataset not present")


def make_samples(root: Path, n: int, size: tuple[int, int] = (64, 64), seed: int = 0):
    """Write `n` synthetic (jpg, density .h5) pairs under `root` and return Samples.

    Each image is noise with a handful of 'heads' at known points, with a matching density
    map, so training has a real (if trivial) signal to fit.
    """
    import h5py
    import numpy as np
    from PIL import Image

    from crowd.config import DENSITY_BY_PART
    from crowd.data.dataset import Sample

    rng = np.random.default_rng(seed)
    h, w = size
    samples = []
    for i in range(n):
        pts = rng.uniform([4, 4], [w - 4, h - 4], size=(int(rng.integers(3, 8)), 2))
        img = rng.integers(0, 60, (h, w, 3), dtype=np.uint8)
        for x, y in pts.astype(int):
            img[max(y - 2, 0) : y + 3, max(x - 2, 0) : x + 3] = 255  # bright blob per head
        from crowd.data.density import density_map

        dmap = density_map(pts.astype(np.float32), h, w, DENSITY_BY_PART["part_B"])
        image_path, density_path = root / f"{i}.jpg", root / f"{i}.h5"
        Image.fromarray(img).save(image_path, quality=95)
        with h5py.File(density_path, "w") as f:
            f.create_dataset("density", data=dmap)
        samples.append(Sample(image_path, density_path))
    return samples
