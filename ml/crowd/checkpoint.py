"""Checkpoint IO. Writes are atomic, so a Colab disconnect mid-save leaves `last.pth` intact."""

from __future__ import annotations

import os
from pathlib import Path

import torch


def save_checkpoint(path: Path, *, model, meta: dict, **state) -> None:
    """Save `{"model": weights, "meta": meta, **state}` to `path` atomically (write a temp
    file, then rename over the target).

    `meta` is the provenance dict stored next to the weights (model_version, part, run_id,
    git_sha, seed, epoch, val_mae, val_rmse, preprocess, dataset_manifest_sha,
    thresholds_sha). `state` carries training state for `last.pth` (optimizer, epoch, best,
    rng, elapsed_s); `best.pth` omits it and stays lean.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({"model": model.state_dict(), "meta": meta, **state}, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path, map_location="cpu") -> dict:
    """Load a checkpoint dict (tensors and plain python only, so `weights_only` is safe)."""
    return torch.load(path, map_location=map_location, weights_only=True)
