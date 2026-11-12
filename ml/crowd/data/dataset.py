"""PyTorch dataset over precomputed density maps."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

from crowd.config import OUTPUT_STRIDE, DatasetConfig
from crowd.transforms import load_rgb, normalize, pad_to_multiple, sum_pool

T = TypeVar("T")


@dataclass(frozen=True)
class Sample:
    image: Path
    density: Path


def discover_samples(raw: Path, processed: Path, part: str, split: str) -> list[Sample]:
    """All samples of a raw split ('train_data' / 'test_data'), sorted for determinism."""
    images = sorted((raw / part / split / "images").glob("*.jpg"))
    return [Sample(i, processed / part / split / "density" / f"{i.stem}.h5") for i in images]


def train_val_split(samples: Sequence[T], cfg: DatasetConfig | None = None):
    """Deterministically hold out `val_fraction` of train for checkpoint selection.

    Selecting checkpoints on the test set leaks; test is touched once, at the end. Works on
    any sequence (Samples here, plain image paths in `crowd.calibrate`), so both exclude the
    same validation images.
    """
    cfg = cfg or DatasetConfig()
    idx = list(range(len(samples)))
    random.Random(cfg.split_seed).shuffle(idx)
    n_val = max(1, round(len(samples) * cfg.val_fraction))
    val = set(idx[:n_val])
    train_s = [s for i, s in enumerate(samples) if i not in val]
    val_s = [s for i, s in enumerate(samples) if i in val]
    return train_s, val_s


def random_crop_flip(image: np.ndarray, density: np.ndarray, crop: tuple[int, int], rng):
    """Random crop to `crop` (H, W) plus a random horizontal flip.

    The *same* crop and flip are applied to image and density; otherwise supervision is
    spatially wrong. Inputs smaller than the crop are zero-padded (6 Part A images are).
    """
    ch, cw = crop
    h, w = density.shape
    if h < ch or w < cw:
        ph, pw = max(ch - h, 0), max(cw - w, 0)
        image = np.pad(image, ((0, ph), (0, pw), (0, 0)))
        density = np.pad(density, ((0, ph), (0, pw)))
        h, w = density.shape
    y = int(rng.integers(0, h - ch + 1))
    x = int(rng.integers(0, w - cw + 1))
    image, density = image[y : y + ch, x : x + cw], density[y : y + ch, x : x + cw]
    if rng.random() < 0.5:
        image, density = image[:, ::-1], density[:, ::-1]
    return np.ascontiguousarray(image), np.ascontiguousarray(density)


def random_scale_crop(
    image: np.ndarray, density: np.ndarray, crop: tuple[int, int], rng, scale_range
):
    """Like `random_crop_flip`, but the crop is first drawn at a random zoom.

    A window of `crop / s` pixels is cut out and resized to `crop`, with `s` log-uniform in
    `scale_range`. The density window is resized the same way and rescaled so its sum (the
    head count in the window) is unchanged: zooming changes head size, never the count.
    """
    lo, hi = scale_range
    s = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    ch, cw = crop
    wh, ww = max(round(ch / s), OUTPUT_STRIDE), max(round(cw / s), OUTPUT_STRIDE)
    image, density = random_crop_flip(image, density, (wh, ww), rng)
    if (wh, ww) == (ch, cw):
        return image, density
    shrinking = wh > ch
    image = cv2.resize(
        image, (cw, ch), interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
    )
    total = float(density.sum())
    density = cv2.resize(density, (cw, ch), interpolation=cv2.INTER_LINEAR)
    got = float(density.sum())
    if got > 0:
        density = density * (total / got)
    return np.ascontiguousarray(image), np.ascontiguousarray(density.astype(np.float32))


def photometric_jitter(image: np.ndarray, rng) -> np.ndarray:
    """Brightness and contrast in +-20%, and a 10% chance of grayscale (some scenes are)."""
    x = image.astype(np.float32)
    if rng.random() < 0.1:
        x = np.repeat(x.mean(axis=2, keepdims=True), 3, axis=2)
    brightness, contrast = rng.uniform(0.8, 1.2, size=2)
    mean = x.mean()
    x = (x - mean) * contrast + mean * brightness
    return np.clip(x, 0, 255).astype(np.uint8)


class CrowdDataset(Dataset):
    """Yields (image (3,H,W) normalised, target (1,H/8,W/8) density).

    train=True : random crop + flip -> fixed size, so batches work despite Part A's
                 189 different resolutions.
    train=False: full image padded to a multiple of 8 (use batch size 1).
    `target.sum()` equals the ground-truth head count.
    """

    def __init__(
        self, samples: list[Sample], train: bool, cfg: DatasetConfig | None = None, seed: int = 0
    ):
        cfg = cfg or DatasetConfig()
        if any(c % OUTPUT_STRIDE for c in cfg.crop_size):
            raise ValueError(f"crop_size must be multiples of {OUTPUT_STRIDE}")
        self.samples, self.train, self.cfg, self.seed = samples, train, cfg, seed
        self._rng = None

    def __len__(self) -> int:
        return len(self.samples) * (self.cfg.crops_per_image if self.train else 1)

    def _get_rng(self):
        if self._rng is None:  # created lazily so each DataLoader worker gets its own stream
            info = get_worker_info()
            self._rng = np.random.default_rng([self.seed, info.id if info else 0])
        return self._rng

    def __getitem__(self, i: int):
        if not 0 <= i < len(self):
            raise IndexError(i)  # keeps `for x in dataset` finite
        s = self.samples[i % len(self.samples)]
        image = load_rgb(s.image)
        with h5py.File(s.density, "r") as f:
            density = f["density"][()]
        if self.train:
            rng = self._get_rng()
            if self.cfg.scale_range:
                image, density = random_scale_crop(
                    image, density, self.cfg.crop_size, rng, self.cfg.scale_range
                )
            else:
                image, density = random_crop_flip(image, density, self.cfg.crop_size, rng)
            if self.cfg.photometric:
                image = photometric_jitter(image, rng)
        img_t = pad_to_multiple(normalize(image))
        den_t = pad_to_multiple(torch.from_numpy(density))
        return img_t, sum_pool(den_t)[None]
