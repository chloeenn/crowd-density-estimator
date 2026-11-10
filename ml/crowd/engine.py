"""One-epoch training and validation loops, plus the count metrics (MAE / RMSE)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from crowd.metrics import Metrics, count_metrics


def train_epoch(model, loader: DataLoader, optimizer, device) -> float:
    """Train for one epoch; returns the mean per-sample loss.

    Loss is MSE with reduction="sum" divided by batch size (see TrainConfig for why).
    """
    model.train()
    total = 0.0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        loss = F.mse_loss(model(images), targets, reduction="sum") / images.shape[0]
        if not torch.isfinite(loss):
            raise FloatingPointError(
                "loss is not finite - the learning rate is too high for the sum-reduced loss"
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.inference_mode()
def predict_counts(model, loader: DataLoader, device) -> tuple[list[float], list[float]]:
    """Predicted and ground-truth head counts per image, in loader order.

    The raw output has no final ReLU, so it is clamped at zero before summing (a negative
    density has no meaning).
    """
    model.eval()
    pred: list[float] = []
    truth: list[float] = []
    for images, targets in loader:
        counts = model(images.to(device)).clamp(min=0).sum(dim=(1, 2, 3))
        pred += counts.tolist()
        truth += targets.sum(dim=(1, 2, 3)).tolist()
    return pred, truth


def validate(model, loader: DataLoader, device) -> Metrics:
    """Count MAE / RMSE over full images."""
    return count_metrics(*predict_counts(model, loader, device))
