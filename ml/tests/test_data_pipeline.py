"""Guards for the silent failure modes of density-map generation (see docs/PLAN.md §4.5).

None of these bugs raise an error; they produce a model that trains smoothly to a
useless MAE. Run these before any training.
"""
import numpy as np
import pytest
import torch
from conftest import RAW, needs_data
from PIL import Image

from crowd.config import DENSITY_BY_PART, DatasetConfig
from crowd.data.annotations import clip_points, load_points
from crowd.data.build import build_one, gt_path_for
from crowd.data.dataset import (
    CrowdDataset,
    Sample,
    discover_samples,
    random_crop_flip,
    train_val_split,
)
from crowd.data.density import density_map
from crowd.transforms import normalize, pad_to_multiple, sum_pool

A, B = DENSITY_BY_PART["part_A"], DENSITY_BY_PART["part_B"]


def _first_images(part, n):
    return sorted((RAW / part / "train_data" / "images").glob("*.jpg"))[:n]


# 1. The premise of the whole method: density sums to the head count.
@needs_data
@pytest.mark.parametrize("part", ["part_A", "part_B"])
def test_density_sum_equals_count_real_samples(part):
    for img in _first_images(part, 10):
        w, h = Image.open(img).size
        pts = load_points(gt_path_for(img))
        total = density_map(pts, h, w, DENSITY_BY_PART[part]).sum()
        assert total == pytest.approx(len(pts), rel=0.01), img.name


# 2. Count survives the 1/8 downsample (guards the missing x64 bug).
@needs_data
def test_count_preserved_after_downsample():
    for img in _first_images("part_A", 5):
        w, h = Image.open(img).size
        pts = load_points(gt_path_for(img))
        d = pad_to_multiple(torch.from_numpy(density_map(pts, h, w, A)))
        pooled = sum_pool(d)
        assert pooled.shape == (d.shape[0] // 8, d.shape[1] // 8)
        assert pooled.sum().item() == pytest.approx(len(pts), rel=0.02)
        assert pooled.min().item() >= 0


# 3. Grayscale Part A images become (3, H, W) tensors through the real dataset.
@needs_data
def test_grayscale_image_yields_three_channels(tmp_path):
    gray = next(p for p in sorted((RAW / "part_A" / "train_data" / "images").glob("*.jpg"))
                if Image.open(p).mode != "RGB")
    out = tmp_path / "d.h5"
    build_one(gray, out, A)
    for train in (True, False):
        img, target = CrowdDataset([Sample(gray, out)], train=train)[0]
        assert img.shape[0] == 3 and target.shape[0] == 1
        assert torch.isfinite(img).all()


# 4. Out-of-bounds points are clipped, not dropped.
def test_out_of_bounds_points_are_clipped_not_dropped():
    pts = np.array([[-0.4, 5.0], [100.3, 20.0], [10.0, 50.7], [30.0, -2.0]], np.float32)
    clipped = clip_points(pts, height=50, width=100)
    assert clipped[:, 0].min() >= 0 and clipped[:, 0].max() <= 99
    assert clipped[:, 1].min() >= 0 and clipped[:, 1].max() <= 49
    assert density_map(pts, 50, 100, B).sum() == pytest.approx(4, rel=1e-4)


# 5. Image and density are cropped/flipped identically.
def test_crop_flip_moves_image_and_density_together():
    h, w = 64, 80
    for seed in range(50):
        rng = np.random.default_rng(seed)
        y, x = int(rng.integers(0, h)), int(rng.integers(0, w))
        image = np.zeros((h, w, 3), np.uint8)
        density = np.zeros((h, w), np.float32)
        image[y, x] = 255
        density[y, x] = 1.0
        # Crop sized so the marker is not always inside; only check when it survives.
        im2, d2 = random_crop_flip(image, density, (32, 32), np.random.default_rng(seed + 1000))
        assert im2.shape[:2] == d2.shape == (32, 32)
        if d2.sum() > 0:
            assert np.argwhere(im2[..., 0] == 255).tolist() == np.argwhere(d2 == 1.0).tolist()
        else:
            assert im2.sum() == 0


def test_crop_pads_images_smaller_than_crop():
    im, d = random_crop_flip(np.ones((100, 120, 3), np.uint8), np.ones((100, 120), np.float32),
                             (256, 256), np.random.default_rng(0))
    assert im.shape == (256, 256, 3) and d.shape == (256, 256)
    assert d.sum() == 100 * 120  # nothing lost to padding


# 6. Degenerate point counts produce finite maps.
@pytest.mark.parametrize("cfg", [A, B], ids=["adaptive", "fixed"])
@pytest.mark.parametrize("n", [0, 1, 2, 3])
def test_degenerate_point_counts_are_finite(cfg, n):
    pts = np.array([[20.0, 20.0], [25.0, 22.0], [60.0, 40.0]][:n], np.float32).reshape(-1, 2)
    d = density_map(pts, 64, 96, cfg)
    assert np.isfinite(d).all() and d.sum() == pytest.approx(n, abs=1e-3)


def test_duplicate_points_do_not_break_adaptive_kernel():
    pts = np.array([[10.0, 10.0]] * 4, np.float32)  # identical -> zero neighbour distance
    d = density_map(pts, 40, 40, A)
    assert np.isfinite(d).all() and d.sum() == pytest.approx(4, abs=1e-3)


# Extras: dataset behaviour end to end.
@needs_data
def test_train_batches_have_fixed_shape_despite_variable_image_sizes(tmp_path):
    imgs = _first_images("part_A", 6)
    samples = []
    for i in imgs:
        out = tmp_path / f"{i.stem}.h5"
        build_one(i, out, A)
        samples.append(Sample(i, out))
    ds = CrowdDataset(samples, train=True)
    batch = [ds[k] for k in range(len(ds))]
    x = torch.stack([b[0] for b in batch])
    y = torch.stack([b[1] for b in batch])
    assert x.shape == (6, 3, 256, 256) and y.shape == (6, 1, 32, 32)


@needs_data
def test_eval_target_sums_to_ground_truth_count(tmp_path):
    img = _first_images("part_B", 1)[0]
    out = tmp_path / "d.h5"
    n = build_one(img, out, B)
    _, target = CrowdDataset([Sample(img, out)], train=False)[0]
    assert target.sum().item() == pytest.approx(n, rel=0.01)


def test_normalize_shape_and_range():
    t = normalize(np.full((8, 8, 3), 255, np.uint8))
    assert t.shape == (3, 8, 8) and t.dtype == torch.float32


@needs_data
def test_val_split_is_deterministic_disjoint_and_sized():
    s = discover_samples(RAW, RAW.parent / "processed", "part_A", "train_data")
    tr1, va1 = train_val_split(s, DatasetConfig())
    tr2, va2 = train_val_split(s, DatasetConfig())
    assert va1 == va2 and tr1 == tr2
    assert not set(tr1) & set(va1)
    assert len(tr1) + len(va1) == len(s) == 300 and len(va1) == 30
