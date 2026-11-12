"""Download the model checkpoints listed in models/weights.json and verify their SHA-256.

Why: checkpoints (~62 MB each) are never committed or baked into images. The file is written
to a temporary name and renamed only after the hash matches, so a failed download never
leaves a corrupt checkpoint where the API would load it.
Usage: python scripts/download_weights.py [--manifest models/weights.json] [--out models]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path


def sha256_of(path: Path) -> str:
    """Hex SHA-256 of a file, read in chunks."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str, entry: dict, out: Path) -> None:
    """Download one file unless it is already present with the right hash."""
    url, want = entry.get("url"), entry.get("sha256")
    dest = out / name
    if want and dest.exists() and sha256_of(dest) == want:
        print(f"{name}: already present, hash OK")
        return
    if not url or not want:
        raise SystemExit(
            f"{name}: no url/sha256 in the manifest. The checkpoints have not been published yet."
        )
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"{name}: downloading {url}")
    urllib.request.urlretrieve(url, tmp)
    got = sha256_of(tmp)
    if got != want:
        tmp.unlink()
        raise SystemExit(
            f"{name}: SHA-256 mismatch (expected {want}, got {got}); file discarded"
        )
    tmp.replace(dest)
    print(f"{name}: OK")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=Path("models/weights.json"))
    ap.add_argument("--out", type=Path, default=Path("models"))
    args = ap.parse_args(argv)
    files = json.loads(args.manifest.read_text())["files"]
    args.out.mkdir(parents=True, exist_ok=True)
    for name, entry in files.items():
        fetch(name, entry, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
