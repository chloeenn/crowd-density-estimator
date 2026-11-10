"""Central, frozen configuration for the data pipeline.

Keeping every tunable here (instead of inline literals) makes runs reproducible and
lets docs point at one place.
"""
from __future__ import annotations

from dataclasses import dataclass

# VGG-16 features[:23] (first 10 convs) downsamples by 8, so CSRNet outputs a
# density map at 1/8 of the input resolution.
OUTPUT_STRIDE = 8

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class DensityConfig:
    """How annotated head points are turned into a density map.

    Part A (dense crowds) uses a geometry-adaptive kernel: sigma = beta * mean distance
    to the k nearest neighbouring heads. Part B (sparse) uses a fixed sigma.
    """

    adaptive: bool
    beta: float = 0.3
    k: int = 3
    fixed_sigma: float = 15.0
    min_sigma: float = 1.0  # floor so a degenerate (duplicate-point) kernel stays valid
    truncate: float = 4.0  # kernel radius in sigmas


DENSITY_BY_PART = {
    "part_A": DensityConfig(adaptive=True),
    "part_B": DensityConfig(adaptive=False, fixed_sigma=15.0),
}


@dataclass(frozen=True)
class DatasetConfig:
    crop_size: tuple[int, int] = (256, 256)  # (H, W); multiples of OUTPUT_STRIDE
    val_fraction: float = 0.1  # held out of *train*; test is never used for selection
    split_seed: int = 0
