"""Print the toolchain and prove the selected device can run a backward pass.

Why: MPS and CUDA fail in different ways (missing ops, NaN gradients). Catching that in
seconds beats discovering it an hour into training. Run: `python -m crowd.env_check`.
"""

from __future__ import annotations

import platform
import sys

import torch

from crowd.device import get_device


def versions() -> dict[str, str]:
    """Versions of the libraries the pipeline depends on."""
    import cv2
    import h5py
    import numpy
    import scipy
    import torchvision

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "cv2": cv2.__version__,
        "h5py": h5py.__version__,
    }


def backward_check(device: torch.device) -> float:
    """3x3 conv forward and backward; returns the gradient norm (finite and non-zero)."""
    torch.manual_seed(0)
    conv = torch.nn.Conv2d(3, 4, 3, padding=1).to(device)
    out = conv(torch.randn(2, 3, 16, 16, device=device))
    out.square().mean().backward()
    grad = conv.weight.grad
    assert grad is not None, "no gradient was produced"
    norm = float(grad.norm().cpu())
    assert torch.isfinite(grad).all(), "gradient contains NaN or Inf"
    assert norm > 0, "gradient is exactly zero"
    return norm


def main() -> int:
    for name, version in versions().items():
        print(f"{name:12s} {version}")
    print(f"{'mps':12s} {torch.backends.mps.is_available()}")
    print(f"{'cuda':12s} {torch.cuda.is_available()}")
    device = get_device()
    print(f"{'device':12s} {device}")
    print(f"{'grad norm':12s} {backward_check(device):.4f}")
    print("env_check: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
