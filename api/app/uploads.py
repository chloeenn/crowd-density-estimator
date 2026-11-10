"""Upload validation and decoding. Everything stays in memory; nothing is written to disk."""

from __future__ import annotations

from io import BytesIO

import numpy as np
from fastapi import UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError

from app.config import ALLOWED_CONTENT_TYPES, Settings
from app.errors import ApiError


def check_content_type(upload: UploadFile) -> None:
    declared = (upload.content_type or "").split(";")[0].strip().lower()
    if declared not in ALLOWED_CONTENT_TYPES:
        raise ApiError(
            415,
            "unsupported_media_type",
            f"Unsupported file type {declared or 'unknown'!r}. Upload a JPEG, PNG or WebP image.",
        )


async def read_bytes(upload: UploadFile, settings: Settings) -> bytes:
    """Read the upload, refusing anything over the size limit without reading the rest."""
    data = await upload.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise ApiError(
            413,
            "payload_too_large",
            f"File is larger than {settings.max_upload_mb} MB.",
        )
    if not data:
        raise ApiError(400, "invalid_image", "The uploaded file is empty.")
    return data


def _too_large(settings: Settings) -> ApiError:
    """One message for both oversize paths (our pixel check, and Pillow's bomb guard)."""
    return ApiError(
        413,
        "payload_too_large",
        f"Image is too large: more than {settings.max_image_pixels:,} pixels.",
    )


def decode_image(data: bytes, settings: Settings) -> np.ndarray:
    """Decode to RGB uint8 (H, W, 3), honouring EXIF orientation (phone photos).

    Decodes fully (`load`), which is what catches truncated files; header checks alone don't.
    """
    try:
        with Image.open(BytesIO(data)) as im:
            if im.format not in ALLOWED_CONTENT_TYPES.values():
                raise ApiError(
                    415,
                    "unsupported_media_type",
                    "Unsupported image format. Upload a JPEG, PNG or WebP image.",
                )
            if im.width * im.height > settings.max_image_pixels:
                raise _too_large(settings)
            im.load()
            return np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
    except ApiError:
        raise
    except Image.DecompressionBombError as e:
        raise _too_large(settings) from e
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as e:
        raise ApiError(
            400,
            "invalid_image",
            "The file is not a valid, complete image (it may be corrupt or truncated).",
        ) from e
