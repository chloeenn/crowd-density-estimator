"""Image preprocessing shared by training, evaluation and the API.

There must be exactly one definition of "how an image becomes a network input";
a second copy in the web backend is the classic source of train/serve skew.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image

from crowd.config import IMAGENET_MEAN, IMAGENET_STD, OUTPUT_STRIDE

_MEAN = np.asarray(IMAGENET_MEAN, dtype=np.float32)
_STD = np.asarray(IMAGENET_STD, dtype=np.float32)


def load_rgb(path) -> np.ndarray:
    """Load an image as uint8 (H, W, 3).

    ShanghaiTech Part A contains grayscale JPEGs; the pretrained VGG frontend needs
    three channels, so we always convert explicitly.
    """
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def downscale_long_side(image: np.ndarray, max_long_side: int) -> np.ndarray:
    """Shrink a uint8 (H, W, 3) image so its longer side is at most `max_long_side`, keeping
    the aspect ratio. Smaller images are returned unchanged (never upscaled): the model was
    trained on images up to 1024 px, and upscaling only adds blur."""
    h, w = image.shape[:2]
    scale = max_long_side / max(h, w)
    if scale >= 1.0:
        return image
    size = (max(round(w * scale), 1), max(round(h * scale), 1))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def normalize(image: np.ndarray) -> torch.Tensor:
    """uint8 (H, W, 3) -> float32 (3, H, W), ImageNet-normalised."""
    x = (image.astype(np.float32) / 255.0 - _MEAN) / _STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


def pad_amount(size: int, multiple: int = OUTPUT_STRIDE) -> int:
    """Pixels to add to make `size` a multiple of `multiple`."""
    return (-size) % multiple


def pad_to_multiple(x: torch.Tensor, multiple: int = OUTPUT_STRIDE) -> torch.Tensor:
    """Zero-pad the bottom/right of a (..., H, W) tensor to multiples of `multiple`.

    Zero in normalised space is the dataset mean colour, so padding is neutral.
    """
    ph, pw = pad_amount(x.shape[-2], multiple), pad_amount(x.shape[-1], multiple)
    if ph == 0 and pw == 0:
        return x
    return torch.nn.functional.pad(x, (0, pw, 0, ph))


def sum_pool(density: torch.Tensor, stride: int = OUTPUT_STRIDE) -> torch.Tensor:
    """Downsample a (H, W) density map by summing stride x stride blocks.

    Summing (rather than interpolating and re-scaling by stride**2) preserves the total
    count *exactly* and cannot produce negative values. H and W must be multiples of stride.
    """
    h, w = density.shape[-2:]
    if h % stride or w % stride:
        raise ValueError(f"density shape {(h, w)} not divisible by stride {stride}")
    return density.reshape(*density.shape[:-2], h // stride, stride, w // stride, stride).sum(
        dim=(-3, -1)
    )
