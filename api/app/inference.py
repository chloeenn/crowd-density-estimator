"""Model registry and admission control around `crowd.predict`.

Holds one `Predictor` per part. Inference runs one-at-a-time per instance (CPU), with a small
bounded queue; beyond it requests get 503 + Retry-After instead of piling up unboundedly.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

import numpy as np
import torch
from crowd.levels import thresholds_sha
from crowd.predict import Prediction, Predictor
from crowd.routing import load_routing, use_sparse

from app.config import Settings
from app.errors import ApiError
from app.logging_setup import LOGGER_NAME

log = logging.getLogger(LOGGER_NAME)


class InferenceService:
    def __init__(
        self,
        predictors: dict[str, Predictor],
        default_part: str,
        *,
        max_concurrent: int = 1,
        max_queue: int = 4,
        retry_after_s: int = 2,
        routing_threshold: float | None = None,
    ):
        if default_part not in predictors:
            raise ValueError(f"default part {default_part!r} is not among {sorted(predictors)}")
        self.predictors, self.default_part = predictors, default_part
        # Automatic routing needs both models and a threshold; without one the default serves.
        self.routing_threshold = routing_threshold if len(predictors) == 2 else None
        self.capacity = max_concurrent + max_queue  # running + waiting
        self.retry_after_s = retry_after_s
        self._pending = 0
        self._sem = asyncio.Semaphore(max_concurrent)

    @classmethod
    def from_settings(cls, settings: Settings) -> InferenceService:
        """Load and warm up the configured models. Raises (so the server refuses to start) if
        a checkpoint is missing, has mismatching preprocessing, or is for the wrong part."""
        torch.set_num_threads(settings.torch_threads)
        paths = {settings.model_part: settings.model_path}
        if settings.alt_model_path:
            paths[_other(settings.model_part)] = settings.alt_model_path
        predictors = {}
        for part, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError(
                    f"model checkpoint not found: {path}. Set MODEL_PATH to a checkpoint, "
                    "or run `make download-weights` once checkpoints are published."
                )
            predictor = Predictor.from_checkpoint(path, settings.device, settings.max_long_side)
            if predictor.part != part:
                raise ValueError(f"{path} holds Part {predictor.part}, expected Part {part}")
            sha = predictor.meta.get("thresholds_sha")
            if sha is not None and sha != thresholds_sha():
                log.warning(
                    "model %s was trained with different level thresholds",
                    predictor.version,
                )
            predictor.warmup()
            predictors[part] = predictor
            log.info("loaded %s from %s", predictor.version, path.name)
        routing = load_routing()
        return cls(
            predictors,
            settings.model_part,
            routing_threshold=(
                float(routing["threshold"]) if routing and settings.auto_route else None
            ),
            max_concurrent=settings.max_concurrent_inference,
            max_queue=settings.max_queue,
            retry_after_s=settings.retry_after_s,
        )

    def predict(self, image: np.ndarray, part: str | None) -> tuple[Predictor, Prediction, bool]:
        """Run the requested model, or route by scene when `part` is None.

        Returns (predictor used, prediction, routed). The dense model runs first; the sparse
        model takes over when the dense one counts fewer people than the routing threshold.
        """
        if part is not None or self.routing_threshold is None:
            predictor = self.get(part)
            return predictor, predictor.predict(image), False
        dense = self.predictors["A"].predict(image)
        if not use_sparse(dense.count, self.routing_threshold):
            return self.predictors["A"], dense, True
        return self.predictors["B"], self.predictors["B"].predict(image), True

    def get(self, part: str | None) -> Predictor:
        part = part or self.default_part
        if part not in self.predictors:
            raise ApiError(
                422,
                "model_unavailable",
                f"Model for Part {part} is not loaded. "
                f"Available: {', '.join(sorted(self.predictors))}.",
            )
        return self.predictors[part]

    @asynccontextmanager
    async def slot(self):
        """Admission control: wait for the model if the queue has room, else 503."""
        if self._pending >= self.capacity:
            raise ApiError(
                503,
                "busy",
                "The server is busy. Try again shortly.",
                headers={"Retry-After": str(self.retry_after_s)},
            )
        self._pending += 1
        try:
            async with self._sem:
                yield
        finally:
            self._pending -= 1


def _other(part: str) -> str:
    return "B" if part == "A" else "A"
