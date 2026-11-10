"""Training machinery on tiny synthetic data (CPU, no dataset or GPU needed)."""

import json
from dataclasses import replace

import pytest
import torch
import yaml
from helpers import make_samples
from torch import nn

from crowd.checkpoint import load_checkpoint, save_checkpoint
from crowd.config import DatasetConfig, TrainConfig
from crowd.data.dataset import CrowdDataset
from crowd.engine import count_metrics, validate
from crowd.models.csrnet import CSRNet
from crowd.runs import preprocess_signature
from crowd.train import fit, parse_args

RUN = "t"


def tiny_cfg(tmp_path, **kw) -> TrainConfig:
    base = TrainConfig(
        runs_dir=tmp_path / "runs",
        run_id=RUN,
        publish_dir=None,
        data=DatasetConfig(crop_size=(64, 64)),
        epochs=2,
        batch_size=2,
        lr=1e-5,
        num_workers=0,
        device="cpu",
        pretrained=False,
    )
    return replace(base, **kw)


def run_dir(cfg):
    return cfg.runs_dir / cfg.run_id


def log_lines(cfg):
    return [json.loads(x) for x in (run_dir(cfg) / "metrics.jsonl").read_text().splitlines()]


def test_count_metrics_math():
    m = count_metrics([10, 20, 30], [12, 20, 26])  # errors: -2, 0, 4
    assert m.mae == pytest.approx(2.0)
    assert m.rmse == pytest.approx((20 / 3) ** 0.5)
    assert m.n == 3


def test_count_metrics_rejects_bad_input():
    with pytest.raises(ValueError, match="non-empty"):
        count_metrics([], [])
    with pytest.raises(ValueError, match="equal-length"):
        count_metrics([1], [1, 2])


class _Constant(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, x):
        return torch.full((x.shape[0], 1, x.shape[2] // 8, x.shape[3] // 8), self.value)


def test_validate_clamps_negative_density_to_zero(tmp_path):
    ds = CrowdDataset(make_samples(tmp_path, 3), train=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=1)
    truth_mean = sum(t.sum().item() for _, t in ds) / 3
    # A model predicting negative density everywhere must count as predicting zero people.
    assert validate(_Constant(-5.0), loader, "cpu").mae == pytest.approx(truth_mean)
    assert validate(_Constant(0.0), loader, "cpu").mae == pytest.approx(truth_mean)


def test_checkpoint_roundtrip_is_atomic(tmp_path):
    model = CSRNet(pretrained=False)
    path = tmp_path / "sub" / "c.pth"
    save_checkpoint(path, model=model, meta={"run_id": "x"}, epoch=3)
    assert [p.name for p in path.parent.iterdir()] == ["c.pth"]  # no leftover .tmp
    ck = load_checkpoint(path)
    assert (ck["epoch"], ck["meta"]) == (3, {"run_id": "x"})
    other = CSRNet(pretrained=False)
    other.load_state_dict(ck["model"])
    same = [torch.equal(a, b) for a, b in zip(model.parameters(), other.parameters(), strict=True)]
    assert all(same)


def test_fit_writes_the_full_run_directory(tmp_path):
    cfg = tiny_cfg(tmp_path)
    s = make_samples(tmp_path, 4)
    summary = fit(cfg, s[:3], s[3:])
    assert {p.name for p in run_dir(cfg).iterdir()} == {
        "best.pth", "last.pth", "metrics.jsonl", "summary.json", "config.yaml",
        "env.json", "dataset.json", "split.json", "seed.json",
    }  # fmt: skip
    rows = log_lines(cfg)
    assert [r["epoch"] for r in rows] == [0, 1]
    assert all({"train_loss", "lr", "seconds", "val_mae", "val_rmse"} <= r.keys() for r in rows)
    assert all(torch.isfinite(torch.tensor(r["train_loss"])) for r in rows)
    on_disk = json.loads((run_dir(cfg) / "summary.json").read_text())
    assert on_disk == summary
    assert summary["status"] == "finished" and summary["epochs_completed"] == 2
    assert summary["best"]["val_mae"] == min(r["val_mae"] for r in rows)
    assert summary["test"] is None  # the test split is never touched by training


def test_run_files_record_provenance(tmp_path):
    cfg = tiny_cfg(tmp_path, seed=7)
    s = make_samples(tmp_path, 4)
    fit(cfg, s[:3], s[3:])
    d = run_dir(cfg)
    split = json.loads((d / "split.json").read_text())
    assert split["train"] == sorted(x.image.name for x in s[:3])
    assert split["val"] == [s[3].image.name] and not set(split["train"]) & set(split["val"])
    assert json.loads((d / "seed.json").read_text())["seed"] == 7
    assert yaml.safe_load((d / "config.yaml").read_text())["lr"] == 1e-5
    env = json.loads((d / "env.json").read_text())
    assert {"python", "torch", "device", "git"} <= env.keys()
    assert "filelist_sha256" in json.loads((d / "dataset.json").read_text())


def test_checkpoints_carry_metadata(tmp_path):
    cfg = tiny_cfg(tmp_path, part="part_B")
    s = make_samples(tmp_path, 3)
    fit(cfg, s[:2], s[2:])
    for name in ("best.pth", "last.pth"):
        meta = load_checkpoint(run_dir(cfg) / name)["meta"]
        assert meta["model_version"] == "csrnet-partB-v1" and meta["part"] == "B"
        assert meta["run_id"] == RUN and meta["seed"] == 0
        assert meta["preprocess"] == preprocess_signature("part_B")
        assert meta["thresholds_sha"] is None  # calibration is Phase 5
        assert meta["val_mae"] is not None and meta["val_rmse"] is not None
    assert "optimizer" not in load_checkpoint(run_dir(cfg) / "best.pth")  # lean
    assert {"optimizer", "rng", "best", "elapsed_s"} <= load_checkpoint(
        run_dir(cfg) / "last.pth"
    ).keys()


def test_finished_run_is_published_to_docs_runs(tmp_path):
    cfg = tiny_cfg(tmp_path, publish_dir=tmp_path / "docs" / "runs")
    s = make_samples(tmp_path, 3)
    fit(cfg, s[:2], s[2:])
    published = json.loads((tmp_path / "docs" / "runs" / f"{RUN}.json").read_text())
    assert set(published) == {"summary", "config", "env"}
    assert published["summary"]["run_id"] == RUN


def test_new_run_refuses_to_overwrite_existing_run_id(tmp_path):
    s = make_samples(tmp_path, 3)
    fit(tiny_cfg(tmp_path, epochs=1), s[:2], s[2:])
    with pytest.raises(FileExistsError):
        fit(tiny_cfg(tmp_path, epochs=1), s[:2], s[2:])


def test_resume_continues_the_same_run(tmp_path):
    s = make_samples(tmp_path, 4)
    fit(tiny_cfg(tmp_path, epochs=1), s[:3], s[3:])
    cfg = tiny_cfg(tmp_path, epochs=3, resume=True)
    summary = fit(cfg, s[:3], s[3:])
    assert [r["epoch"] for r in log_lines(cfg)] == [0, 1, 2]  # appended; epoch 0 not repeated
    assert summary["epochs_completed"] == 3
    assert load_checkpoint(run_dir(cfg) / "last.pth")["epoch"] == 2


def test_resume_without_run_id_picks_latest_run_of_that_part(tmp_path):
    s = make_samples(tmp_path, 3)
    fit(tiny_cfg(tmp_path, epochs=1, run_id="part_A-old"), s[:2], s[2:])
    fit(tiny_cfg(tmp_path, epochs=1, run_id="part_A-new"), s[:2], s[2:])
    cfg = tiny_cfg(tmp_path, epochs=2, run_id=None, resume=True)
    assert fit(cfg, s[:2], s[2:])["run_id"] == "part_A-new"


def test_resuming_a_finished_run_is_a_no_op(tmp_path):
    s = make_samples(tmp_path, 3)
    first = fit(tiny_cfg(tmp_path, epochs=1), s[:2], s[2:])
    again = fit(tiny_cfg(tmp_path, epochs=1, resume=True), s[:2], s[2:])
    assert again["epochs_completed"] == first["epochs_completed"] == 1
    assert len(log_lines(tiny_cfg(tmp_path))) == 1


def test_resume_with_no_checkpoint_starts_fresh(tmp_path):
    s = make_samples(tmp_path, 3)
    cfg = tiny_cfg(tmp_path, epochs=1, resume=True)
    fit(cfg, s[:2], s[2:])
    assert [r["epoch"] for r in log_lines(cfg)] == [0]


def test_frontend_warmup_freezes_frontend_only(tmp_path):
    s = make_samples(tmp_path, 3)
    cfg = tiny_cfg(tmp_path, epochs=1, freeze_frontend_epochs=1, lr=1e-3, weight_decay=5e-4)
    fit(cfg, s[:2], s[2:])
    torch.manual_seed(cfg.seed)
    ref = CSRNet(pretrained=False)  # same seed -> same initial weights as inside fit()
    trained = CSRNet(pretrained=False)
    trained.load_state_dict(load_checkpoint(run_dir(cfg) / "last.pth")["model"])
    frozen_same = [
        torch.equal(a, b)
        for a, b in zip(ref.frontend.parameters(), trained.frontend.parameters(), strict=True)
    ]
    assert all(frozen_same)
    assert not torch.equal(ref.backend[0].weight, trained.backend[0].weight)


def test_diverging_lr_raises_a_helpful_error(tmp_path):
    s = make_samples(tmp_path, 3)
    cfg = tiny_cfg(tmp_path, epochs=2, lr=1e12)  # step 1 explodes the weights; step 2 sees it
    with pytest.raises(FloatingPointError, match="learning rate"):
        fit(cfg, s[:2], s[2:])


def test_patience_stops_early(tmp_path):
    s = make_samples(tmp_path, 3)
    cfg = tiny_cfg(tmp_path, epochs=50, lr=0.0, patience=2)  # frozen weights: MAE never improves
    summary = fit(cfg, s[:2], s[2:])
    assert len(log_lines(cfg)) == 3  # epoch 0 sets the best; epochs 1-2 exhaust patience
    assert summary["status"] == "stopped_early"


def test_training_reduces_loss_on_a_tiny_set(tmp_path):
    s = make_samples(tmp_path, 2)
    cfg = tiny_cfg(tmp_path, epochs=25, batch_size=2, optimizer="adam", lr=1e-4)
    fit(cfg, s, s)
    rows = log_lines(cfg)
    assert rows[-1]["train_loss"] < 0.5 * rows[0]["train_loss"]


def test_cli_defaults_and_overrides():
    cfg, overfit = parse_args([])
    assert (cfg.lr, cfg.optimizer, cfg.weight_decay, overfit) == (1e-5, "adam", 0.0, 0)
    assert str(cfg.runs_dir) == "runs" and str(cfg.publish_dir) == "docs/runs"
    cfg, overfit = parse_args(
        [
            "--part",
            "part_B",
            "--crop",
            "128",
            "--overfit",
            "10",
            "--no-publish",
            "--optimizer",
            "sgd",
        ]
    )
    assert (cfg.part, cfg.data.crop_size, overfit) == ("part_B", (128, 128), 10)
    assert cfg.publish_dir is None and cfg.optimizer == "sgd"
