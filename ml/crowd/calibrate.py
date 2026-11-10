"""Derive the Low / Medium / High / Critical count bands from the *training* data.

    python -m crowd.calibrate --raw data/ShanghaiTech

Pools the Part A and Part B training splits, excludes exactly the validation images that
training holds out (same split function and seed), and takes the 25th/50th/75th/90th
percentiles of the head counts. Test images are never read. Writes
`ml/crowd/thresholds.json`, which ships with the package.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np

from crowd.config import DatasetConfig
from crowd.data.annotations import gt_path_for, load_points
from crowd.data.dataset import train_val_split
from crowd.levels import THRESHOLDS_PATH
from crowd.runs import git_info

PARTS = ("part_A", "part_B")
QUANTILES = (25, 50, 75, 90)
BAND_NAMES = ("low", "medium", "high", "critical")  # band edge = where the band starts


def training_counts(raw: Path, cfg: DatasetConfig, parts=PARTS) -> tuple[list[int], int]:
    """Head counts of the pooled training images (validation excluded) and how many
    validation images were excluded."""
    counts: list[int] = []
    excluded = 0
    for part in parts:
        images = sorted((raw / part / "train_data" / "images").glob("*.jpg"))
        if not images:
            raise FileNotFoundError(f"no training images under {raw / part}")
        train, val = train_val_split(images, cfg)
        counts += [len(load_points(gt_path_for(i))) for i in train]
        excluded += len(val)
    return counts, excluded


def calibrate(raw: Path, cfg: DatasetConfig | None = None, parts=PARTS) -> dict:
    """Compute the thresholds record (see `main` for where it is written)."""
    cfg = cfg or DatasetConfig()
    counts, excluded = training_counts(raw, cfg, parts)
    quantiles = [float(np.percentile(counts, q)) for q in QUANTILES]
    git = git_info()
    return {
        "bands": {name: round(q) for name, q in zip(BAND_NAMES, quantiles, strict=True)},
        "quantiles": {f"p{q}": v for q, v in zip(QUANTILES, quantiles, strict=True)},
        "rule": "a band starts at its edge; counts below `medium` are Low",
        "split": "train_data minus validation",
        "parts_pooled": list(parts),
        "n_images": len(counts),
        "n_validation_excluded": excluded,
        "validation_seed": cfg.split_seed,
        "validation_fraction": cfg.val_fraction,
        "count_min": min(counts),
        "count_max": max(counts),
        "git_sha": git["sha"],
        "git_dirty": git["dirty"],
        "calibrated": date.today().isoformat(),
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw", type=Path, default=Path("data/ShanghaiTech"))
    ap.add_argument("--out", type=Path, default=None, help="default: the packaged thresholds.json")
    a = ap.parse_args(argv)
    result = calibrate(a.raw)
    out = a.out or THRESHOLDS_PATH
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {out}")
    print(json.dumps({k: result[k] for k in ("bands", "n_images", "parts_pooled")}))


if __name__ == "__main__":
    main()
