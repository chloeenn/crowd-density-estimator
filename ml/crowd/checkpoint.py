"""Checkpoint IO. Writes are atomic, so a Colab disconnect mid-save leaves `last.pth` intact."""

from __future__ import annotations

import os
from pathlib import Path

import torch

from crowd.config import preprocess_signature
from crowd.models.csrnet import CSRNet


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


def load_model(path: Path, device="cpu") -> tuple[CSRNet, dict]:
    """Load a trained CSRNet in eval mode and return it with its checkpoint metadata.

    Raises ValueError if the checkpoint was trained with different preprocessing than the
    live code (normalisation, stride, density kernels): predictions would be silently wrong.
    """
    meta = load_checkpoint(path)["meta"]
    expected = preprocess_signature(f"part_{meta['part']}")
    if meta["preprocess"] != expected:
        diff = sorted(k for k in expected if meta["preprocess"].get(k) != expected[k])
        raise ValueError(f"{path.name}: preprocess differs from the live config in {diff}")
    model = CSRNet(pretrained=False)
    model.load_state_dict(load_checkpoint(path, map_location="cpu")["model"])
    return model.to(device).eval(), meta
