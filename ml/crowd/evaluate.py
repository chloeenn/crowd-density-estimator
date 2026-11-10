"""Evaluate a trained run on a held-out split and write the results.

    python -m crowd.evaluate runs/<run_id>               # test split, read ONCE (gate G5)
    python -m crowd.evaluate runs/<run_id> --split val   # dry run on the validation split

On the test split this writes `docs/results/<part>.{json,-scatter.png,-worst10.png}`,
regenerates `docs/results.md`, records the numbers in the run's `summary.json` and re-publishes
`docs/runs/<run_id>.json`. The test split is read once per run: a second evaluation is refused
unless `--force` is given. A validation dry run writes only inside the run directory.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from crowd.checkpoint import load_model
from crowd.config import PUBLISHED
from crowd.data.dataset import CrowdDataset, discover_samples
from crowd.device import get_device
from crowd.engine import predict_counts
from crowd.metrics import bucket_metrics, check_acceptance, count_metrics, diagnose
from crowd.report import plot_scatter, plot_worst_gallery, render_results_md
from crowd.runs import publish_run, write_summary
from crowd.transforms import load_rgb
from crowd.viz import crop_to_image

WORST_N = 10


class AlreadyEvaluatedError(RuntimeError):
    """The test split was already read for this run."""


def load_split_samples(run_dir: Path, raw: Path, processed: Path, part: str, split: str):
    """Samples of the requested split: the part's test set, or this run's validation images."""
    if split == "test":
        return discover_samples(raw, processed, part, "test_data")
    val_names = set(json.loads((run_dir / "split.json").read_text())["val"])
    train = discover_samples(raw, processed, part, "train_data")
    return [s for s in train if s.image.name in val_names]


def worst_indices(pred: list[float], truth: list[float], k: int = WORST_N) -> list[int]:
    """Indices of the k images with the largest absolute count error, worst first."""
    return sorted(range(len(pred)), key=lambda i: -abs(pred[i] - truth[i]))[:k]


def build_result(
    *,
    part: str,
    split: str,
    names: list[str],
    pred: list[float],
    truth: list[float],
    meta: dict,
    checkpoint: str,
) -> dict:
    """Everything `results.md` and `summary.json` need, as a JSON-ready dict."""
    m = count_metrics(pred, truth)
    verdict = check_acceptance(part, m)
    buckets = bucket_metrics(pred, truth)
    return {
        "part": part,
        "split": split,
        "run_id": meta["run_id"],
        "checkpoint": checkpoint,
        "epoch": meta["epoch"],
        "model_version": meta["model_version"],
        "git_sha": meta["git_sha"],
        "dataset_manifest_sha": meta["dataset_manifest_sha"],
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "n": m.n,
        "mae": m.mae,
        "rmse": m.rmse,
        "bias": sum(p - t for p, t in zip(pred, truth, strict=True)) / m.n,
        "acceptance": {**asdict(verdict), "passed": verdict.passed},
        "published": dict(zip(("mae", "rmse"), PUBLISHED[part], strict=True)),
        "buckets": [asdict(b) for b in buckets],
        "worst": [
            {"image": names[i], "truth": truth[i], "pred": pred[i], "error": pred[i] - truth[i]}
            for i in worst_indices(pred, truth)
        ],
        "diagnosis": diagnose(part, pred, truth, buckets, verdict),
    }


@torch.inference_mode()
def gallery_rows(
    model, dataset: CrowdDataset, indices: list[int], pred, truth, device
) -> list[dict]:
    """Re-run the model on just the worst images to get their density maps for the gallery
    (cheaper than keeping every map in memory during the full pass)."""
    rows = []
    for i in indices:
        image_t, target = dataset[i]  # same preprocessing path as the metric pass
        density = model(image_t[None].to(device)).clamp(min=0)[0, 0].cpu().numpy()
        image = load_rgb(dataset.samples[i].image)
        hw = image.shape[:2]
        rows.append(
            {
                "name": dataset.samples[i].image.name,
                "image": image,
                "gt": crop_to_image(target[0].numpy(), hw),
                "pred": crop_to_image(density, hw),
                "gt_count": truth[i],
                "pred_count": pred[i],
            }
        )
    return rows


def record_test_result(run_dir: Path, result: dict, docs_dir: Path) -> None:
    """Persist a test result: run summary, tracked docs/runs copy, and docs/results.md."""
    summary = json.loads((run_dir / "summary.json").read_text())
    summary["test"] = {
        k: result[k] for k in ("mae", "rmse", "bias", "n", "checkpoint", "epoch", "evaluated_at")
    } | {"acceptance_passed": result["acceptance"]["passed"]}
    write_summary(run_dir, summary)
    publish_run(run_dir, docs_dir / "runs")
    results = [
        json.loads(p.read_text()) for p in sorted((docs_dir / "results").glob("part_*.json"))
    ]
    (docs_dir / "results.md").write_text(render_results_md(results))


def evaluate_run(
    run_dir: Path,
    *,
    raw: Path,
    processed: Path,
    docs_dir: Path,
    checkpoint: str = "best",
    split: str = "test",
    device: str = "auto",
    workers: int = 2,
    force: bool = False,
) -> dict:
    """Evaluate `runs/<run_id>/<checkpoint>.pth` on `split` and write the outputs."""
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text())
    if split == "test" and summary["test"] is not None and not force:
        raise AlreadyEvaluatedError(
            f"{run_dir.name}: the test split was already evaluated; it is read once "
            "(use --force to overwrite)"
        )
    dev = get_device(device)
    model, meta = load_model(run_dir / f"{checkpoint}.pth", dev)
    if meta["run_id"] != run_dir.name:
        raise ValueError(f"checkpoint belongs to run {meta['run_id']!r}, not {run_dir.name!r}")

    part = f"part_{meta['part']}"
    samples = load_split_samples(run_dir, raw, processed, part, split)
    if not samples:
        raise FileNotFoundError(f"no {split} images found for {part} under {raw}")
    if not samples[0].density.exists():
        raise FileNotFoundError("density maps missing; run: python -m crowd.data.build")

    dataset = CrowdDataset(samples, train=False)
    loader = DataLoader(dataset, batch_size=1, num_workers=workers)
    pred, truth = predict_counts(model, loader, dev)
    result = build_result(
        part=part,
        split=split,
        names=[s.image.name for s in samples],
        pred=pred,
        truth=truth,
        meta=meta,
        checkpoint=f"{checkpoint}.pth",
    )

    out_dir, tag = (
        (docs_dir / "results", part) if split == "test" else (run_dir / "eval_val", "val")
    )
    title = f"{part} {split} (n={result['n']}): MAE {result['mae']:.1f}, RMSE {result['rmse']:.1f}"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_scatter(pred, truth, title, out_dir / f"{tag}-scatter.png")
    rows = gallery_rows(model, dataset, worst_indices(pred, truth), pred, truth, dev)
    plot_worst_gallery(
        rows, f"{part} {split}: worst {len(rows)} images", out_dir / f"{tag}-worst10.png"
    )
    (out_dir / f"{tag}.json").write_text(json.dumps(result, indent=2) + "\n")
    if split == "test":
        record_test_result(run_dir, result, docs_dir)
    return result


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir", type=Path, help="runs/<run_id>")
    ap.add_argument("--checkpoint", choices=["best", "last"], default="best")
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument("--raw", type=Path, default=Path("data/ShanghaiTech"))
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--docs", type=Path, default=Path("docs"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--force", action="store_true", help="re-evaluate an already-evaluated run")
    a = ap.parse_args(argv)
    try:
        r = evaluate_run(
            a.run_dir,
            raw=a.raw,
            processed=a.processed,
            docs_dir=a.docs,
            checkpoint=a.checkpoint,
            split=a.split,
            device=a.device,
            workers=a.workers,
            force=a.force,
        )
    except AlreadyEvaluatedError as e:
        raise SystemExit(str(e)) from e
    acc = r["acceptance"]
    verdict = "MET" if acc["passed"] else "MISSED"
    print(f"{r['part']} {r['split']} n={r['n']}")
    print(f"  MAE {r['mae']:.2f}  RMSE {r['rmse']:.2f}  bias {r['bias']:+.2f}")
    print(f"  bar MAE<={acc['mae_bar']:.0f} RMSE<={acc['rmse_bar']:.0f}: {verdict}")
    for line in r["diagnosis"]:
        print(f"  - {line}")


if __name__ == "__main__":
    main()
