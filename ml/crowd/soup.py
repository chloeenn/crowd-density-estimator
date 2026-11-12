"""Average the weights of several Part A checkpoints into one.

    python -m crowd.soup runs/part_A-soup runs/part_A-full runs/part_A-aug

Runs that start from the same pretrained network stay close enough in weight space for their
average to work as one model, and it costs nothing extra to serve. The result is written as a
new run directory (`best.pth` plus the provenance files of the first run), so it is evaluated
like any other run. Choose the inputs on validation MAE, never on the test split.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from crowd.checkpoint import load_checkpoint, save_checkpoint

COPIED = ("split.json", "config.yaml", "dataset.json", "env.json", "seed.json")


def average_states(states: list[dict]) -> dict:
    """Element-wise mean of several state dicts with identical keys."""
    keys = states[0].keys()
    if any(s.keys() != keys for s in states):
        raise ValueError("checkpoints have different parameters")
    return {k: sum(s[k].float() for s in states) / len(states) for k in keys}


def make_soup(out: Path, sources: list[Path], val_mae: float, val_rmse: float) -> Path:
    """Write `out/best.pth` from the `best.pth` of each source run directory."""
    ckpts = [load_checkpoint(s / "best.pth") for s in sources]
    parts = {c["meta"]["part"] for c in ckpts}
    if len(parts) != 1:
        raise ValueError(f"cannot average different parts: {sorted(parts)}")
    meta = ckpts[0]["meta"] | {
        "run_id": out.name,
        "val_mae": val_mae,
        "val_rmse": val_rmse,
        "soup_of": [s.name for s in sources],
    }

    class _Holder:  # save_checkpoint wants an object with .state_dict()
        def state_dict(self):
            return average_states([c["model"] for c in ckpts])

    out.mkdir(parents=True, exist_ok=False)
    save_checkpoint(out / "best.pth", model=_Holder(), meta=meta)
    for name in COPIED:
        if (sources[0] / name).exists():
            shutil.copy(sources[0] / name, out / name)
    summary = json.loads((sources[0] / "summary.json").read_text())
    summary |= {
        "run_id": out.name,
        "status": "finished",
        "best": {"epoch": None, "val_mae": val_mae, "val_rmse": val_rmse},
        "soup_of": [s.name for s in sources],
        "test": None,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out", type=Path, help="new run directory, e.g. runs/part_A-soup")
    ap.add_argument("sources", type=Path, nargs="+", help="run directories to average")
    ap.add_argument("--val-mae", type=float, required=True)
    ap.add_argument("--val-rmse", type=float, required=True)
    a = ap.parse_args(argv)
    print(make_soup(a.out, a.sources, a.val_mae, a.val_rmse))


if __name__ == "__main__":
    main()
