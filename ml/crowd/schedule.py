"""Learning-rate schedule as a pure function of the epoch.

Stateless on purpose: there is no scheduler object to checkpoint, so `--resume` cannot get
out of sync, and changing `--epochs` simply defines a new curve.
"""

from __future__ import annotations

import math

from crowd.config import TrainConfig


def lr_for_epoch(cfg: TrainConfig, epoch: int) -> float:
    """Learning rate to use during `epoch` (0-based).

    "constant": cfg.lr throughout.
    "cosine"  : cfg.lr at epoch 0, decaying to cfg.lr * cfg.lr_min_factor at the last epoch.
    """
    if cfg.lr_schedule == "constant" or cfg.epochs <= 1:
        return cfg.lr
    if cfg.lr_schedule == "cosine":
        progress = min(epoch / (cfg.epochs - 1), 1.0)
        floor = cfg.lr * cfg.lr_min_factor
        return floor + 0.5 * (cfg.lr - floor) * (1.0 + math.cos(math.pi * progress))
    raise ValueError(f"unknown lr_schedule {cfg.lr_schedule!r}")
