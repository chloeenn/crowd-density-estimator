"""Run provenance (run ids, dataset hash, git info) and seeding / RNG restore."""

import random
from datetime import datetime

import numpy as np
import torch

from crowd.runs import filelist_sha256, find_latest_run, git_info, make_run_id, preprocess_signature
from crowd.seeding import capture_rng, restore_rng, set_seed


def test_run_id_format():
    assert (
        make_run_id("part_A", "7b74a46", datetime(2026, 11, 9, 21, 5))
        == "part_A-20261109-2105-7b74a46"
    )


def test_filelist_hash_tracks_paths_and_sizes_but_ignores_ds_store(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    base = filelist_sha256(tmp_path)
    assert filelist_sha256(tmp_path) == base  # stable
    (tmp_path / ".DS_Store").write_text("junk")
    assert filelist_sha256(tmp_path) == base
    (tmp_path / "a.txt").write_text("hello!")  # size changed
    changed = filelist_sha256(tmp_path)
    assert changed != base
    (tmp_path / "b.txt").write_text("x")  # file added
    assert filelist_sha256(tmp_path) != changed


def test_git_info_outside_a_repo_degrades_gracefully(tmp_path):
    assert git_info(tmp_path) == {"sha": "nogit", "dirty": None}


def test_preprocess_signature_distinguishes_parts():
    a, b = preprocess_signature("part_A"), preprocess_signature("part_B")
    assert a["sigma_strategy"] == "adaptive" and b["sigma_strategy"] == "fixed"
    assert a["stride"] == a["pad_multiple"] == 8 and len(a["mean"]) == len(a["std"]) == 3


def test_find_latest_run_ignores_runs_without_checkpoints(tmp_path):
    for name in ("part_A-1", "part_A-2", "part_B-1", "overfit10-part_A-3"):
        (tmp_path / name).mkdir()
    (tmp_path / "part_A-1" / "last.pth").write_bytes(b"x")
    (tmp_path / "overfit10-part_A-3" / "last.pth").write_bytes(b"x")  # newest, but an experiment
    assert (
        find_latest_run(tmp_path, "part_A").name == "part_A-1"
    )  # not part_A-2 (no checkpoint) or the overfit run
    assert find_latest_run(tmp_path, "part_B") is None


def _draw():
    return random.random(), float(np.random.rand()), torch.rand(3).tolist()


def test_set_seed_is_reproducible_and_recorded():
    record = set_seed(123)
    first = _draw()
    set_seed(123)
    assert _draw() == first
    set_seed(124)
    assert _draw() != first
    assert record["seed"] == 123 and {"random", "numpy", "torch"} <= set(record["seeded"])


def test_rng_capture_and_restore_resumes_the_stream(tmp_path):
    set_seed(5)
    _draw()
    state = capture_rng()
    # The snapshot must survive the safe checkpoint format used for last.pth.
    torch.save({"rng": state}, tmp_path / "s.pt")
    state = torch.load(tmp_path / "s.pt", weights_only=True)["rng"]
    expected = _draw()
    set_seed(999)  # scramble everything
    restore_rng(state)
    assert _draw() == expected
