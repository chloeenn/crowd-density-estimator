"""Serving inference: one RGB image in, count and density map out.

This is the only inference path outside training/evaluation, and it shares their
preprocessing (`crowd.transforms`), so the API never carries its own normalisation code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from crowd.checkpoint import load_model
from crowd.config import OUTPUT_STRIDE
from crowd.transforms import downscale_long_side, normalize, pad_to_multiple
from crowd.viz import crop_to_image


@dataclass(frozen=True)
class Prediction:
    count: float  # estimated number of people
    density: np.ndarray  # (h/8, w/8) float32 >= 0, cropped to the processed image
    image: np.ndarray  # the RGB uint8 image the model actually saw (possibly downscaled)


class Predictor:
    """A trained model plus the metadata needed to serve it."""

    def __init__(self, model, meta: dict, device, max_long_side: int = 1600):
        self.model, self.meta, self.device = model, meta, torch.device(device)
        self.max_long_side = max_long_side

    @classmethod
    def from_checkpoint(cls, path: Path, device="cpu", max_long_side: int = 1600) -> Predictor:
        """Load a checkpoint; raises ValueError if its preprocessing differs from the live
        config (see `load_model`)."""
        model, meta = load_model(path, device)
        return cls(model, meta, device, max_long_side)

    @property
    def part(self) -> str:
        return self.meta["part"]  # "A" or "B"

    @property
    def version(self) -> str:
        return self.meta["model_version"]

    def warmup(self) -> None:
        """Run a tiny forward pass so the first real request doesn't pay lazy-init costs."""
        self.predict(np.zeros((OUTPUT_STRIDE * 8, OUTPUT_STRIDE * 8, 3), np.uint8))

    @torch.inference_mode()
    def predict(self, image: np.ndarray) -> Prediction:
        """Estimate the crowd in an RGB uint8 (H, W, 3) image.

        Steps: downscale the long side, normalise, pad to a multiple of 8, run the network,
        clamp negative density to zero. The count sums the whole output exactly as
        `crowd.evaluate` does; only the returned density map is cropped to the image.
        """
        image = downscale_long_side(image, self.max_long_side)
        x = pad_to_multiple(normalize(image))[None].to(self.device)
        out = self.model(x).clamp(min=0)[0, 0].float().cpu().numpy()
        return Prediction(
            count=float(out.sum()),
            density=crop_to_image(out, image.shape[:2]),
            image=image,
        )
