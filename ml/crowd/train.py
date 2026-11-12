"""Train CSRNet on one ShanghaiTech part.

    python -m crowd.train --part part_A --epochs 150
    python -m crowd.train --part part_A --resume          # continue the latest part_A run
    python -m crowd.train --part part_B --overfit 10      # pipeline sanity check (gate G3)

Each run writes `runs/<run_id>/` (see `crowd.runs`). The test split is never read here:
checkpoints are selected on a validation split held out of train.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import time
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from crowd.checkpoint import load_checkpoint, save_checkpoint
from crowd.config import DatasetConfig, TrainConfig, preprocess_signature
from crowd.data.dataset import CrowdDataset, Sample, discover_samples, train_val_split
from crowd.device import get_device
from crowd.engine import train_epoch, validate
from crowd.levels import thresholds_sha
from crowd.metrics import Metrics
from crowd.models.csrnet import CSRNet
from crowd.runs import (
    collect_env,
    filelist_sha256,
    find_latest_run,
    git_info,
    make_run_id,
    publish_run,
    write_config,
    write_provenance,
    write_summary,
)
from crowd.schedule import lr_for_epoch
from crowd.seeding import capture_rng, restore_rng, set_seed

log = logging.getLogger("crowd.train")


def make_optimizer(cfg: TrainConfig, model: CSRNet) -> torch.optim.Optimizer:
    # The optimiser skips frozen params (no grad). One optimiser serves both the warmup
    # and the full fine-tuning phase.
    params = list(model.parameters())
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(
            params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay
        )
    if cfg.optimizer == "adam":
        return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    raise ValueError(f"unknown optimizer {cfg.optimizer!r}")


def make_loaders(cfg: TrainConfig, train_s: list[Sample], val_s: list[Sample], device):
    common = {
        "num_workers": cfg.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": cfg.num_workers > 0,
    }
    train_loader = DataLoader(
        CrowdDataset(train_s, train=True, cfg=cfg.data, seed=cfg.seed),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=len(train_s) > cfg.batch_size,
        generator=torch.Generator(),  # reseeded every epoch: a resumed run shuffles identically
        **common,
    )
    # Validation uses full images of varying size. Use batch size 1.
    val_loader = DataLoader(
        CrowdDataset(val_s, train=False, cfg=cfg.data), batch_size=1, shuffle=False, **common
    )
    return train_loader, val_loader


def open_run(cfg: TrainConfig) -> tuple[str, Path, bool]:
    """Return (run_id, run_dir, resumed). Creates the directory for a new run."""
    if cfg.resume:
        found = cfg.runs_dir / cfg.run_id if cfg.run_id else find_latest_run(cfg.runs_dir, cfg.part)
        if found and (found / "last.pth").exists():
            return found.name, found, True
        log.warning("--resume given but no resumable %s run found; starting fresh", cfg.part)
    run_id = cfg.run_id or make_run_id(cfg.part, git_info()["sha"])
    run_dir = cfg.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_id, run_dir, False


def fit(cfg: TrainConfig, train_s: list[Sample], val_s: list[Sample]) -> dict:
    """Run (or resume) training and return the run summary."""
    seed_record = set_seed(cfg.seed)
    device = get_device(cfg.device)
    run_id, run_dir, resumed = open_run(cfg)
    manifest_sha = filelist_sha256(cfg.raw) if cfg.raw.exists() else "unavailable"

    model = CSRNet(pretrained=cfg.pretrained).to(device)  # seeded above -> reproducible init
    if cfg.init_from and not resumed:
        model.load_state_dict(load_checkpoint(cfg.init_from, map_location=device)["model"])
        log.info("initialised from %s", cfg.init_from)
    optimizer = make_optimizer(cfg, model)
    ema_model = copy.deepcopy(model).eval() if cfg.ema_decay > 0 else None
    start, elapsed = 0, 0.0
    best = {"epoch": -1, "val_mae": float("inf"), "val_rmse": None}
    if resumed:
        ckpt = load_checkpoint(run_dir / "last.pth", map_location=device)
        model.load_state_dict(ckpt["model"])
        if ema_model is not None:
            ema_model.load_state_dict(ckpt.get("ema", ckpt["model"]))
        optimizer.load_state_dict(ckpt["optimizer"])
        restore_rng(ckpt["rng"])
        start, elapsed, best = ckpt["epoch"] + 1, ckpt["elapsed_s"], ckpt["best"]
        write_config(run_dir, cfg)  # config.yaml reflects the latest invocation
        log.info("resumed %s at epoch %d (best val MAE %.2f)", run_id, start, best["val_mae"])
    else:
        write_provenance(
            run_dir,
            cfg,
            env=collect_env(device),
            seed=seed_record,
            manifest_sha=manifest_sha,
            train_names=[s.image.name for s in train_s],
            val_names=[s.image.name for s in val_s],
        )

    def meta(epoch: int, val: Metrics | None) -> dict:
        return {
            "model_version": f"csrnet-{cfg.part.replace('_', '')}-v1",
            "part": cfg.part[-1],
            "run_id": run_id,
            "git_sha": git_info()["sha"],
            "seed": cfg.seed,
            "epoch": epoch,
            "val_mae": val.mae if val else None,
            "val_rmse": val.rmse if val else None,
            "preprocess": preprocess_signature(cfg.part),
            "dataset_manifest_sha": manifest_sha,
            "thresholds_sha": thresholds_sha(),
        }

    train_loader, val_loader = make_loaders(cfg, train_s, val_s, device)
    log.info("run=%s device=%s train=%d val=%d", run_id, device, len(train_s), len(val_s))
    status, last_val, completed = "finished", None, start - 1

    def on_step() -> None:
        """Move the averaged weights a little towards the live ones after every update."""
        if ema_model is None:
            return
        with torch.no_grad():
            for avg, live in zip(ema_model.parameters(), model.parameters(), strict=True):
                avg.mul_(cfg.ema_decay).add_(live.detach(), alpha=1 - cfg.ema_decay)

    evaluated = ema_model if ema_model is not None else model  # what validation and best.pth use
    for epoch in range(start, cfg.epochs):
        t0 = time.time()
        train_loader.generator.manual_seed(cfg.seed + epoch)
        model.set_frontend_trainable(epoch >= cfg.freeze_frontend_epochs)
        for group in optimizer.param_groups:
            group["lr"] = lr_for_epoch(cfg, epoch)
        loss = train_epoch(model, train_loader, optimizer, device, on_step)

        record = {"epoch": epoch, "train_loss": loss, "lr": optimizer.param_groups[0]["lr"]}
        if (epoch + 1) % cfg.val_every == 0 or epoch == cfg.epochs - 1:
            last_val = validate(evaluated, val_loader, device)
            record |= {"val_mae": last_val.mae, "val_rmse": last_val.rmse}
            if last_val.mae < best["val_mae"]:
                best = {"epoch": epoch, "val_mae": last_val.mae, "val_rmse": last_val.rmse}
                save_checkpoint(run_dir / "best.pth", model=evaluated, meta=meta(epoch, last_val))
        record["seconds"] = round(time.time() - t0, 1)
        elapsed += time.time() - t0
        completed = epoch

        save_checkpoint(
            run_dir / "last.pth",
            model=model,
            meta=meta(epoch, last_val),
            optimizer=optimizer.state_dict(),
            **({"ema": ema_model.state_dict()} if ema_model is not None else {}),
            epoch=epoch,
            best=best,
            rng=capture_rng(),
            elapsed_s=elapsed,
        )
        with (run_dir / "metrics.jsonl").open("a") as f:  # append: a resumed run continues the log
            f.write(json.dumps(record) + "\n")
        write_summary(run_dir, _summary(run_id, cfg, completed, best, elapsed, "running"))
        log.info(
            " ".join(
                f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in record.items()
            )
        )

        if cfg.patience and best["epoch"] >= 0 and epoch - best["epoch"] >= cfg.patience:
            log.info("early stop: no val-MAE improvement for %d epochs", cfg.patience)
            status = "stopped_early"
            break

    summary = _summary(run_id, cfg, completed, best, elapsed, status)
    write_summary(run_dir, summary)
    if cfg.publish_dir:
        log.info("published %s", publish_run(run_dir, cfg.publish_dir))
    return summary


def _summary(run_id, cfg, epoch, best, elapsed, status) -> dict:
    return {
        "run_id": run_id,
        "part": cfg.part,
        "status": status,
        "epochs_completed": epoch + 1,
        "best": best,
        "total_seconds": round(elapsed, 1),
        "test": None,  # filled in by crowd.evaluate; never touched here
    }


def parse_args(argv=None) -> tuple[TrainConfig, int]:
    d = TrainConfig()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--part", choices=["part_A", "part_B"], default=d.part)
    ap.add_argument("--raw", type=Path, default=d.raw)
    ap.add_argument("--processed", type=Path, default=d.processed)
    ap.add_argument("--runs-dir", type=Path, default=d.runs_dir)
    ap.add_argument("--run-id", default=None, help="default: <part>-<YYYYMMDD-HHMM>-<git_sha>")
    ap.add_argument("--publish-dir", type=Path, default=d.publish_dir)
    ap.add_argument("--no-publish", action="store_true", help="skip the docs/runs summary copy")
    ap.add_argument("--epochs", type=int, default=d.epochs)
    ap.add_argument("--batch-size", type=int, default=d.batch_size)
    ap.add_argument("--crop", type=int, default=d.data.crop_size[0], help="square crop size")
    ap.add_argument("--optimizer", choices=["adam", "sgd"], default=d.optimizer)
    ap.add_argument("--lr", type=float, default=d.lr)
    ap.add_argument("--lr-schedule", choices=["constant", "cosine"], default=d.lr_schedule)
    ap.add_argument("--lr-min-factor", type=float, default=d.lr_min_factor)
    ap.add_argument("--momentum", type=float, default=d.momentum)
    ap.add_argument("--weight-decay", type=float, default=d.weight_decay)
    ap.add_argument("--freeze-frontend-epochs", type=int, default=d.freeze_frontend_epochs)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument(
        "--augment", action="store_true", help="random zoom (0.8-1.25x) and photometric jitter"
    )
    ap.add_argument("--ema", type=float, default=d.ema_decay, help="weight average decay, 0=off")
    ap.add_argument(
        "--init-from", type=Path, default=None, help="start from a checkpoint's weights"
    )
    ap.add_argument("--crops-per-image", type=int, default=d.data.crops_per_image)
    ap.add_argument("--val-every", type=int, default=d.val_every)
    ap.add_argument("--patience", type=int, default=d.patience)
    ap.add_argument("--workers", type=int, default=d.num_workers)
    ap.add_argument("--device", default=d.device)
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--overfit",
        type=int,
        default=0,
        metavar="N",
        help="train AND validate on N train images (gate G3 sanity check)",
    )
    a = ap.parse_args(argv)
    cfg = TrainConfig(
        part=a.part,
        raw=a.raw,
        processed=a.processed,
        runs_dir=a.runs_dir,
        run_id=a.run_id,
        publish_dir=None if a.no_publish else a.publish_dir,
        data=DatasetConfig(
            crop_size=(a.crop, a.crop),
            crops_per_image=a.crops_per_image,
            scale_range=(0.8, 1.25) if a.augment else None,
            photometric=a.augment,
        ),
        ema_decay=a.ema,
        init_from=a.init_from,
        epochs=a.epochs,
        batch_size=a.batch_size,
        optimizer=a.optimizer,
        lr=a.lr,
        lr_schedule=a.lr_schedule,
        lr_min_factor=a.lr_min_factor,
        momentum=a.momentum,
        weight_decay=a.weight_decay,
        freeze_frontend_epochs=a.freeze_frontend_epochs,
        pretrained=not a.no_pretrained,
        val_every=a.val_every,
        patience=a.patience,
        num_workers=a.workers,
        device=a.device,
        seed=a.seed,
        resume=a.resume,
    )
    return cfg, a.overfit


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    cfg, overfit = parse_args(argv)
    samples = discover_samples(cfg.raw, cfg.processed, cfg.part, "train_data")
    if not samples:
        raise SystemExit(f"no images found under {cfg.raw / cfg.part}; check --raw")
    if not samples[0].density.exists():
        raise SystemExit("density maps missing; run: python -m crowd.data.build")
    if overfit:
        # The run id skips the `part_*-` prefix, so --resume never picks it. Not published.
        train_s = val_s = samples[:overfit]
        run_id = cfg.run_id or make_run_id(f"overfit{overfit}-{cfg.part}", git_info()["sha"])
        cfg = replace(cfg, run_id=run_id, publish_dir=None)
    else:
        train_s, val_s = train_val_split(samples, cfg.data)
    print(json.dumps(fit(cfg, train_s, val_s)))


if __name__ == "__main__":
    main()
