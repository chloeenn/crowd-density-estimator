"""Choose between the dense-crowd model (Part A) and the sparse-scene model (Part B).

A sparse model undercounts a packed crowd and a dense model overcounts an open street, so
the right one depends on the scene. The dense model runs first; when it sees fewer people
than `threshold` the sparse model is used instead. The threshold is fitted on held-out
validation images from both parts and shipped as `routing.json`. It is capped at the size of the
crowds the sparse model was trained on, because above that it undercounts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROUTING_PATH = Path(__file__).with_name("routing.json")
COUNT_FLOOR = 20.0  # relative error is measured against max(truth, floor)
SPARSE_TRAIN_QUANTILE = 75  # the threshold never exceeds this quantile of Part B training counts


def load_routing() -> dict | None:
    """The shipped routing record, or None if it has not been fitted."""
    return json.loads(ROUTING_PATH.read_text()) if ROUTING_PATH.is_file() else None


def use_sparse(dense_count: float, threshold: float) -> bool:
    return dense_count < threshold


def routed_counts(dense, sparse, threshold: float) -> np.ndarray:
    dense, sparse = np.asarray(dense, float), np.asarray(sparse, float)
    return np.where(dense < threshold, sparse, dense)


def relative_error(pred, truth) -> float:
    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    return float(np.mean(np.abs(pred - truth) / np.maximum(truth, COUNT_FLOOR)))


def fit_threshold(dense, sparse, truth, max_threshold: float | None = None) -> dict:
    """Threshold on the dense model's count that minimises mean relative error.

    Candidates are midpoints between neighbouring dense counts, plus "never" (0) and "always"
    (above the maximum), none above `max_threshold`. Ties go to the lower threshold.
    """
    dense = np.asarray(dense, float)
    pts = np.unique(dense)
    cands = [0.0, *((pts[:-1] + pts[1:]) / 2).tolist(), float(pts[-1]) + 1.0]
    if max_threshold is not None:
        cands = [c for c in cands if c <= max_threshold]
    errs = [relative_error(routed_counts(dense, sparse, t), truth) for t in cands]
    best = int(np.argmin(errs))
    return {
        "threshold": round(cands[best], 1),
        "relative_error": errs[best],
        "relative_error_dense_only": relative_error(dense, truth),
        "relative_error_sparse_only": relative_error(sparse, truth),
    }


def collect_predictions(dense, sparse, samples_by_split, cache: Path, key: str) -> list[dict]:
    """Count every image with both models, appending to `cache` so an interrupted run resumes.

    `samples_by_split` maps (part, split) to samples. Rows hold the model counts and the true
    count; rows written for other checkpoints (a different `key`) are ignored.
    """
    import h5py

    from crowd.transforms import load_rgb

    rows = [json.loads(line) for line in cache.read_text().splitlines()] if cache.exists() else []
    rows = [r for r in rows if r["key"] == key]
    done = {(r["part"], r["split"], r["name"]) for r in rows}
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("a") as f:
        for (part, split), samples in samples_by_split.items():
            for s in samples:
                if (part, split, s.image.name) in done:
                    continue
                image = load_rgb(s.image)
                with h5py.File(s.density, "r") as h:
                    truth = float(h["density"][()].sum())
                row = {
                    "key": key, "part": part, "split": split, "name": s.image.name,
                    "truth": truth,
                    "dense": dense.predict(image).count,
                    "sparse": sparse.predict(image).count,
                }  # fmt: skip
                f.write(json.dumps(row) + "\n")
                f.flush()
                rows.append(row)
    return rows


def _metrics(pred, truth) -> dict:
    p, t = np.asarray(pred, float), np.asarray(truth, float)
    return {
        "mae": float(np.mean(np.abs(p - t))),
        "rmse": float(np.sqrt(np.mean((p - t) ** 2))),
        "relative_error": relative_error(p, t),
    }


def main(argv=None) -> None:
    """Fit the threshold on validation images, then report all three options on the test sets.

    python -m crowd.routing runs/part_A-full runs/part_B-full
    """
    import argparse
    from datetime import date

    from crowd.calibrate import training_counts
    from crowd.config import DatasetConfig
    from crowd.data.dataset import discover_samples, train_val_split
    from crowd.device import get_device
    from crowd.predict import Predictor
    from crowd.runs import git_info

    ap = argparse.ArgumentParser(description=main.__doc__.split("\n")[0])
    ap.add_argument("dense_run", type=Path, help="Part A run directory")
    ap.add_argument("sparse_run", type=Path, help="Part B run directory")
    ap.add_argument("--raw", type=Path, default=Path("data/ShanghaiTech"))
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--out", type=Path, default=ROUTING_PATH)
    ap.add_argument("--report", type=Path, default=Path("docs/results/routing.json"))
    ap.add_argument("--cache", type=Path, default=Path("runs/routing-predictions.jsonl"))
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    device = get_device(a.device)
    dense = Predictor.from_checkpoint(a.dense_run / "best.pth", device, flip_average=False)
    sparse = Predictor.from_checkpoint(a.sparse_run / "best.pth", device, flip_average=False)

    groups = {}
    for part in ("part_A", "part_B"):
        train = discover_samples(a.raw, a.processed, part, "train_data")
        groups[(part, "val")] = train_val_split(train, DatasetConfig())[1]
        groups[(part, "test")] = discover_samples(a.raw, a.processed, part, "test_data")
    key = f"{a.dense_run.name}/{a.sparse_run.name}"
    rows = collect_predictions(dense, sparse, groups, a.cache, key)

    def column(split, name, part=None):
        return [r[name] for r in rows if r["split"] == split and part in (None, r["part"])]

    cap_counts, _ = training_counts(a.raw, DatasetConfig(), parts=("part_B",))
    cap = float(np.percentile(cap_counts, SPARSE_TRAIN_QUANTILE))
    fit = fit_threshold(
        column("val", "dense"), column("val", "sparse"), column("val", "truth"), max_threshold=cap
    )
    git = git_info()
    record = {
        "threshold": fit["threshold"],
        "rule": "use the sparse model when the dense model counts fewer people than `threshold`",
        "fitted_on": "validation images of Part A and Part B",
        "n_images": len(column("val", "truth")),
        "max_threshold": round(cap, 1),
        "max_threshold_rule": f"p{SPARSE_TRAIN_QUANTILE} of the sparse model's training counts",
        "relative_error_floor": COUNT_FLOOR,
        "dense_run": a.dense_run.name,
        "sparse_run": a.sparse_run.name,
        "git_sha": git["sha"],
        "fitted": date.today().isoformat(),
    }
    a.out.write_text(json.dumps(record, indent=2) + "\n")

    report = {"threshold": fit["threshold"], "validation": fit, "test": {}}
    for part in ("part_A", "part_B", None):
        d, s, t = (column("test", n, part) for n in ("dense", "sparse", "truth"))
        routed = routed_counts(d, s, fit["threshold"])
        report["test"][part or "both"] = {
            name: _metrics(p, t)
            for name, p in (("dense_only", d), ("sparse_only", s), ("auto", routed))
        }
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
