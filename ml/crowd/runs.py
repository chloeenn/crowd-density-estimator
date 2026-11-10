"""Run directories and provenance: every training run is reproducible and traceable.

runs/<run_id>/   config.yaml env.json dataset.json split.json seed.json
                 metrics.jsonl best.pth last.pth summary.json
docs/runs/<run_id>.json   tracked copy of summary + config + env (runs/ is git-ignored)
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path

import yaml

from crowd.config import TrainConfig


def git_info(cwd: Path | None = None) -> dict:
    """Short commit sha and dirty flag; ('nogit', None) outside a repo (e.g. on Colab)."""

    def run(*args):
        return subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return {
            "sha": run("rev-parse", "--short", "HEAD"),
            "dirty": bool(run("status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"sha": "nogit", "dirty": None}


def make_run_id(part: str, git_sha: str, now: datetime | None = None) -> str:
    """`<part>-<YYYYMMDD-HHMM>-<git_sha>`"""
    return f"{part}-{(now or datetime.now()):%Y%m%d-%H%M}-{git_sha}"


def filelist_sha256(root: Path) -> str:
    """Hash of the sorted relative paths and byte sizes under `root` - cheap, and it changes
    whenever a file is added, removed or replaced with different content length."""
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name != ".DS_Store":
            h.update(f"{p.relative_to(root).as_posix()}\t{p.stat().st_size}\n".encode())
    return h.hexdigest()


def collect_env(device) -> dict:
    import cv2
    import h5py
    import numpy
    import scipy
    import torch
    import torchvision

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": str(device),
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "opencv": cv2.__version__,
        "h5py": h5py.__version__,
        "git": git_info(),
    }


def _dump_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")


def write_provenance(
    run_dir: Path,
    cfg: TrainConfig,
    *,
    env: dict,
    seed: dict,
    train_names,
    val_names,
    manifest_sha: str,
) -> None:
    """Write the static, written-once files of a run directory."""
    write_config(run_dir, cfg)
    _dump_json(run_dir / "env.json", env)
    _dump_json(run_dir / "seed.json", seed)
    _dump_json(run_dir / "dataset.json", {"filelist_sha256": manifest_sha, "root": str(cfg.raw)})
    _dump_json(
        run_dir / "split.json",
        {"split_seed": cfg.data.split_seed, "val_fraction": cfg.data.val_fraction,
         "train": sorted(train_names), "val": sorted(val_names)},
    )  # fmt: skip


def write_config(run_dir: Path, cfg: TrainConfig) -> None:
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))


def write_summary(run_dir: Path, summary: dict) -> None:
    _dump_json(run_dir / "summary.json", summary)


def publish_run(run_dir: Path, publish_dir: Path) -> Path:
    """Copy summary + config + env to the tracked `docs/runs/<run_id>.json`."""
    out = publish_dir / f"{run_dir.name}.json"
    publish_dir.mkdir(parents=True, exist_ok=True)
    _dump_json(
        out,
        {
            "summary": json.loads((run_dir / "summary.json").read_text()),
            "config": yaml.safe_load((run_dir / "config.yaml").read_text()),
            "env": json.loads((run_dir / "env.json").read_text()),
        },
    )
    return out


def find_latest_run(runs_dir: Path, part: str) -> Path | None:
    """Most recently modified run of `part` that has a resumable checkpoint."""
    runs = [p for p in runs_dir.glob(f"{part}-*") if (p / "last.pth").exists()]
    return max(runs, key=lambda p: (p / "last.pth").stat().st_mtime, default=None)
