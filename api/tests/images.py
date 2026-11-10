"""Image bytes for tests."""

from io import BytesIO

import numpy as np
from PIL import Image


def noise(w: int = 120, h: int = 80, mode: str = "RGB", seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    channels = {"RGB": 3, "RGBA": 4, "L": None, "P": 3}[mode]
    shape = (h, w) if channels is None else (h, w, channels)
    arr = rng.integers(0, 255, shape, dtype=np.uint8)
    im = Image.fromarray(arr)
    return im.convert("P") if mode == "P" else im


def encode(im: Image.Image, fmt: str, **kw) -> bytes:
    buf = BytesIO()
    im.save(buf, fmt, **kw)
    return buf.getvalue()


def jpeg(w=120, h=80, **kw) -> bytes:
    return encode(noise(w, h), "JPEG", **kw)


def png(w=120, h=80, mode="RGB") -> bytes:
    return encode(noise(w, h, mode), "PNG")


def webp(w=120, h=80) -> bytes:
    return encode(noise(w, h), "WEBP")
