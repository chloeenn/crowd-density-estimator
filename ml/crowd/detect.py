"""Labelled bounding boxes (object detection) drawn on a photo.

Separate from the density models: CSRNet gives a count, not object locations. This wraps an
optional pretrained YOLO model (`pip install ultralytics`, or `pip install -e ml[detect]`),
imported lazily so the rest of the package does not need it.
"""

from __future__ import annotations

import colorsys
import zlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 in pixels of the image that was analysed


class Detector:
    """A pretrained YOLO model that returns `Detection`s for an RGB uint8 image."""

    def __init__(self, weights: str | Path, conf: float = 0.3, device: str = "cpu"):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise RuntimeError("object detection needs ultralytics: pip install ultralytics") from e
        self.model, self.conf, self.device = YOLO(str(weights)), conf, device
        self.names: dict[int, str] = self.model.names

    def detect(self, image: np.ndarray) -> list[Detection]:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # ultralytics expects BGR arrays
        result = self.model.predict(bgr, conf=self.conf, device=self.device, verbose=False)[0]
        out = []
        for xyxy, conf, cls in zip(
            result.boxes.xyxy.tolist(),
            result.boxes.conf.tolist(),
            result.boxes.cls.tolist(),
            strict=True,
        ):
            x1, y1, x2, y2 = (round(v) for v in xyxy)
            out.append(Detection(self.names[int(cls)], round(conf, 3), (x1, y1, x2, y2)))
        return out


def label_color(label: str) -> tuple[int, int, int]:
    """A stable, saturated RGB colour per class name, so 'car' is always the same colour."""
    hue = (zlib.crc32(label.encode()) % 360) / 360
    r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 1.0)
    return round(r * 255), round(g * 255), round(b * 255)


def draw_detections(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Draw each box with a filled class-name tag above it. Returns a copy (RGB uint8)."""
    out = image.copy()
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.4, min(h, w) / 800)
    thick = max(1, round(scale * 2))
    for det in detections:
        color = label_color(det.label)
        x1, y1, x2, y2 = det.box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
        (tw, th), base = cv2.getTextSize(det.label, font, scale, thick)
        top = max(0, y1 - th - base - 4)  # keep the tag on-image for boxes at the top edge
        cv2.rectangle(out, (x1, top), (x1 + tw + 6, top + th + base + 4), color, -1)
        cv2.putText(out, det.label, (x1 + 3, top + th + 1), font, scale, (0, 0, 0), thick,
                    cv2.LINE_AA)  # fmt: skip
    return out
