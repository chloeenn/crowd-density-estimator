"""Pick the compute device (shared by training, evaluation and the API)."""

from __future__ import annotations

import torch


def get_device(name: str = "auto") -> torch.device:
    """'auto' prefers CUDA (Colab), then Apple MPS, then CPU."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
