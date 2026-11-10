"""The LR schedule is a pure function of the epoch."""

import itertools
from dataclasses import replace

import pytest

from crowd.config import TrainConfig
from crowd.schedule import lr_for_epoch

CFG = TrainConfig(epochs=11, lr=1e-3, lr_schedule="cosine", lr_min_factor=0.01)


def test_cosine_starts_at_lr_ends_at_floor_and_is_monotonic():
    lrs = [lr_for_epoch(CFG, e) for e in range(CFG.epochs)]
    assert lrs[0] == pytest.approx(1e-3)
    assert lrs[-1] == pytest.approx(1e-5)
    assert all(a > b for a, b in itertools.pairwise(lrs))
    assert lrs[5] == pytest.approx((1e-3 + 1e-5) / 2)  # halfway point of a cosine


def test_constant_schedule_never_changes():
    cfg = replace(CFG, lr_schedule="constant")
    assert {lr_for_epoch(cfg, e) for e in range(cfg.epochs)} == {1e-3}


def test_single_epoch_run_uses_the_base_lr():
    assert lr_for_epoch(replace(CFG, epochs=1), 0) == 1e-3


def test_epochs_past_the_end_are_clamped_not_periodic():
    assert lr_for_epoch(CFG, 500) == pytest.approx(1e-5)  # CosineAnnealingLR would climb again


def test_unknown_schedule_is_rejected():
    with pytest.raises(ValueError, match="lr_schedule"):
        lr_for_epoch(replace(CFG, lr_schedule="linear"), 1)
