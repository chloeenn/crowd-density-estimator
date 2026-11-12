"""Dataset fingerprint (synthetic tree) and the device backward check."""

import json

import torch

from crowd.data import manifest
from crowd.env_check import backward_check


def _tree(root):
    for name in ("part_A/train_data/images/a.jpg", "part_A/test_data/images/b.jpg"):
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x" * 10)
    return root


def test_manifest_round_trip_and_detects_change(tmp_path):
    raw = _tree(tmp_path / "raw")
    mf = tmp_path / "MANIFEST.json"
    assert manifest.main(["write", "--raw", str(raw), "--manifest", str(mf)]) == 0
    assert manifest.main(["verify", "--raw", str(raw), "--manifest", str(mf)]) == 0
    assert json.loads(mf.read_text())["n_files"] == 2
    (raw / "part_A/train_data/images/a.jpg").write_bytes(b"y" * 11)  # same count, new size
    assert manifest.main(["verify", "--raw", str(raw), "--manifest", str(mf)]) == 1
    (raw / "part_A/train_data/images/c.jpg").write_bytes(b"z")
    assert manifest.main(["verify", "--raw", str(raw), "--manifest", str(mf)]) == 1


def test_backward_check_on_cpu():
    assert backward_check(torch.device("cpu")) > 0
