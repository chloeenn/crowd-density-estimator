"""scripts/download_weights.py: hash verification, with a local file:// 'release'."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "download_weights.py"
spec = importlib.util.spec_from_file_location("download_weights", SCRIPT)
dw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dw)


def _manifest(tmp_path, payload, sha):
    src = tmp_path / "src.bin"
    src.write_bytes(payload)
    mf = tmp_path / "weights.json"
    mf.write_text(json.dumps({"files": {"m.pth": {"url": src.as_uri(), "sha256": sha}}}))
    return mf


def test_good_hash_installs_file(tmp_path):
    data = b"weights"
    mf = _manifest(tmp_path, data, hashlib.sha256(data).hexdigest())
    assert dw.main(["--manifest", str(mf), "--out", str(tmp_path / "out")]) == 0
    assert (tmp_path / "out" / "m.pth").read_bytes() == data


def test_bad_hash_discards_download(tmp_path):
    mf = _manifest(tmp_path, b"weights", "0" * 64)
    with pytest.raises(SystemExit, match="mismatch"):
        dw.main(["--manifest", str(mf), "--out", str(tmp_path / "out")])
    assert not list((tmp_path / "out").glob("m.pth*"))


def test_unpublished_manifest_refuses(tmp_path):
    mf = tmp_path / "w.json"
    mf.write_text(json.dumps({"files": {"m.pth": {"url": None, "sha256": None}}}))
    with pytest.raises(SystemExit, match="not been published"):
        dw.main(["--manifest", str(mf), "--out", str(tmp_path)])


def test_local_file_with_matching_hash_needs_no_url(tmp_path):
    data = b"weights"
    (tmp_path / "m.pth").write_bytes(data)
    mf = tmp_path / "w.json"
    mf.write_text(
        json.dumps({"files": {"m.pth": {"url": None, "sha256": hashlib.sha256(data).hexdigest()}}})
    )
    assert dw.main(["--manifest", str(mf), "--out", str(tmp_path)]) == 0
