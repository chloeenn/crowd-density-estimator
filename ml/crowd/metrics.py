"""Count-error metrics and their analysis. Pure python: no torch, no I/O."""

from __future__ import annotations

import math
from dataclasses import dataclass

from crowd.config import ACCEPTANCE

# Ground-truth count buckets: [0,100) [100,500) [500,1000) [1000,inf)
BUCKET_EDGES = (0, 100, 500, 1000)


@dataclass(frozen=True)
class Metrics:
    mae: float  # mean absolute count error
    rmse: float  # root mean squared count error; exposes catastrophic outliers MAE hides
    n: int


@dataclass(frozen=True)
class BucketRow:
    label: str
    n: int
    mae: float | None  # None for an empty bucket
    rmse: float | None
    bias: float | None  # mean signed error (pred - truth): >0 overcounts, <0 undercounts


@dataclass(frozen=True)
class Verdict:
    mae_bar: float
    rmse_bar: float
    mae_ok: bool
    rmse_ok: bool

    @property
    def passed(self) -> bool:
        return self.mae_ok and self.rmse_ok


def count_metrics(pred: list[float], truth: list[float]) -> Metrics:
    """MAE and RMSE between predicted and ground-truth counts."""
    if len(pred) != len(truth) or not pred:
        raise ValueError("need equal-length, non-empty count lists")
    errs = [p - t for p, t in zip(pred, truth, strict=True)]
    return Metrics(
        mae=sum(abs(e) for e in errs) / len(errs),
        rmse=math.sqrt(sum(e * e for e in errs) / len(errs)),
        n=len(errs),
    )


def bucket_label(lo: int, hi: int | None) -> str:
    return f"{lo}+" if hi is None else f"{lo}-{hi - 1}"


def bucket_metrics(
    pred: list[float], truth: list[float], edges: tuple[int, ...] = BUCKET_EDGES
) -> list[BucketRow]:
    """Errors grouped by ground-truth count, which shows *where* the model fails (a single
    average hides that dense scenes dominate the error). Empty buckets are kept, with n=0."""
    if len(pred) != len(truth):
        raise ValueError("need equal-length count lists")
    uppers = [*edges[1:], None]
    rows = []
    for lo, hi in zip(edges, uppers, strict=True):
        # Counts are integers up to float noise from summing a density map, so round first.
        idx = [i for i, t in enumerate(truth) if round(t) >= lo and (hi is None or round(t) < hi)]
        label = bucket_label(lo, hi)
        if not idx:
            rows.append(BucketRow(label, 0, None, None, None))
            continue
        m = count_metrics([pred[i] for i in idx], [truth[i] for i in idx])
        bias = sum(pred[i] - truth[i] for i in idx) / len(idx)
        rows.append(BucketRow(label, m.n, m.mae, m.rmse, bias))
    return rows


def check_acceptance(part: str, m: Metrics) -> Verdict:
    """Compare against the project's acceptance bar (inclusive: MAE <= bar)."""
    mae_bar, rmse_bar = ACCEPTANCE[part]
    return Verdict(mae_bar, rmse_bar, m.mae <= mae_bar, m.rmse <= rmse_bar)


def diagnose(
    part: str, pred: list[float], truth: list[float], buckets: list[BucketRow], verdict: Verdict
) -> list[str]:
    """Data-driven observations about where the error comes from (no guesswork)."""
    n = len(pred)
    errs = [p - t for p, t in zip(pred, truth, strict=True)]
    mean_truth = sum(truth) / n
    bias = sum(errs) / n
    lines = [
        f"On average the model {'overcounts' if bias > 0 else 'undercounts'} by "
        f"{abs(bias):.1f} people per image ({abs(bias) / mean_truth:.1%} of the mean "
        f"ground-truth count of {mean_truth:.0f})."
    ]
    total_abs = sum(abs(e) for e in errs)
    if total_abs > 0:
        top = max((b for b in buckets if b.n), key=lambda b: b.mae * b.n)
        lines.append(
            f"The {top.label} count bucket ({top.n} images) accounts for "
            f"{top.mae * top.n / total_abs:.0%} of the total absolute error."
        )
        worst10 = sum(sorted((abs(e) for e in errs), reverse=True)[:10])
        lines.append(f"The 10 worst images account for {worst10 / total_abs:.0%} of it.")
    if not verdict.passed and part == "part_A" and count_metrics(pred, truth).mae > 120:
        lines.append(
            "Part A MAE above 120 points to wrong density maps; re-check them "
            "(density sums and the 1/8 downsample) before tuning anything."
        )
    return lines
