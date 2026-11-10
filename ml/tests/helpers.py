"""Shared test helpers."""

from pathlib import Path

import pytest

RAW = Path(__file__).resolve().parents[2] / "data" / "ShanghaiTech"

needs_data = pytest.mark.skipif(not RAW.exists(), reason="ShanghaiTech dataset not present")


def make_samples(
    root: Path,
    n: int,
    size: tuple[int, int] = (64, 64),
    seed: int = 0,
    density_root: Path | None = None,
):
    """Write `n` synthetic (jpg, density .h5) pairs and return Samples.

    Both go under `root`, unless `density_root` is given (to mimic the raw/processed layout).

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
        root.mkdir(parents=True, exist_ok=True)
        (density_root or root).mkdir(parents=True, exist_ok=True)
        image_path = root / f"IMG_{i}.jpg"
        density_path = (density_root or root) / f"IMG_{i}.h5"
        Image.fromarray(img).save(image_path, quality=95)
        with h5py.File(density_path, "w") as f:
            f.create_dataset("density", data=dmap)
        samples.append(Sample(image_path, density_path))
    return samples


def make_dataset_tree(tmp: Path, part: str, n_train: int = 3, n_test: int = 4, seed: int = 0):
    """A miniature ShanghaiTech layout: `<tmp>/raw/<part>/{train,test}_data/images` plus the
    matching `<tmp>/processed/...` density maps. Returns (raw, processed, train_s, test_s)."""
    raw, processed = tmp / "raw", tmp / "processed"
    out = []
    for split, n, sd in (("train_data", n_train, seed), ("test_data", n_test, seed + 1)):
        out.append(
            make_samples(
                raw / part / split / "images",
                n,
                seed=sd,
                density_root=processed / part / split / "density",
            )
        )
    return raw, processed, out[0], out[1]


RUN = "t"


def tiny_cfg(tmp_path: Path, **kw):
    """A TrainConfig small enough to train in seconds on CPU."""
    from dataclasses import replace

    from crowd.config import DatasetConfig, TrainConfig

    base = TrainConfig(
        runs_dir=tmp_path / "runs",
        run_id=RUN,
        publish_dir=None,
        data=DatasetConfig(crop_size=(64, 64)),
        epochs=2,
        batch_size=2,
        lr=1e-5,
        num_workers=0,
        device="cpu",
        pretrained=False,
    )
    return replace(base, **kw)
