import json

import pytest
import torch
from helpers import make_samples, tiny_cfg

from crowd.checkpoint import load_checkpoint
from crowd.soup import average_states, make_soup
from crowd.train import fit


def test_average_is_the_elementwise_mean():
    a = {"w": torch.tensor([1.0, 3.0])}
    b = {"w": torch.tensor([3.0, 5.0])}
    assert average_states([a, b])["w"].tolist() == [2.0, 4.0]


def test_mismatched_parameters_are_rejected():
    with pytest.raises(ValueError, match="different parameters"):
        average_states([{"w": torch.zeros(1)}, {"v": torch.zeros(1)}])


def test_soup_run_is_a_loadable_run_directory(tmp_path):
    s = make_samples(tmp_path, 4)
    runs = []
    for i, name in enumerate(("one", "two")):
        cfg = tiny_cfg(tmp_path, run_id=name, seed=i)
        fit(cfg, s[:3], s[3:])
        runs.append(cfg.runs_dir / name)
    out = make_soup(tmp_path / "runs" / "soup", runs, val_mae=1.0, val_rmse=2.0)
    ck = load_checkpoint(out / "best.pth")
    assert ck["meta"]["run_id"] == "soup" and ck["meta"]["soup_of"] == ["one", "two"]
    a, b = (load_checkpoint(r / "best.pth")["model"] for r in runs)
    k = next(iter(a))
    assert torch.allclose(ck["model"][k], (a[k] + b[k]) / 2)
    assert json.loads((out / "summary.json").read_text())["status"] == "finished"
    assert (out / "split.json").exists()
