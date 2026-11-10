"""Seeding and RNG-state capture for reproducible, resumable runs.

GPU kernels (cuDNN, MPS) are not bit-deterministic, so what is reproducible from the seed is
the validation split, the weight initialisation and the shuffle order - not the exact loss.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> dict:
    """Seed python, numpy and torch (all devices). Returns a record for `seed.json`."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds CUDA and MPS generators
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return {
        "seed": seed,
        "seeded": ["random", "numpy", "torch"]
        + (["cuda"] if torch.cuda.is_available() else [])
        + (["mps"] if torch.backends.mps.is_available() else []),
        "deterministic_kernels": False,
    }


def capture_rng() -> dict:
    """Snapshot every RNG as plain python/tensors, safe for `torch.load(weights_only=True)`."""
    kind, keys, pos, has_gauss, cached = np.random.get_state()
    state = {
        "python": random.getstate(),
        "numpy": [kind, keys.tolist(), pos, has_gauss, cached],
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng(state: dict) -> None:
    """Inverse of `capture_rng`. Backends missing on this machine are skipped."""
    random.setstate(state["python"])
    kind, keys, pos, has_gauss, cached = state["numpy"]
    np.random.set_state((kind, np.asarray(keys, dtype=np.uint32), pos, has_gauss, cached))
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"].cpu())
