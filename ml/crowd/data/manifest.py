"""Fingerprint the raw dataset so a changed or partial copy fails loudly.

Counts files per part and split and hashes the sorted list of (relative path, byte size).
`write` creates `data/MANIFEST.json`; `verify` compares the tree against it.
CLI: `python -m crowd.data.manifest [write|verify]`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

DEFAULT_RAW = Path("data/ShanghaiTech")
DEFAULT_MANIFEST = Path("data/MANIFEST.json")


def fingerprint(raw: Path) -> dict:
    """Per-split file counts plus a SHA-256 of the sorted (path, size) listing."""
    files = sorted(p for p in raw.rglob("*") if p.is_file() and p.name != ".DS_Store")
    if not files:
        raise FileNotFoundError(f"no dataset files under {raw}")
    counts: dict[str, int] = {}
    digest = hashlib.sha256()
    for p in files:
        rel = p.relative_to(raw)
        key = "/".join(rel.parts[:3]) if len(rel.parts) > 3 else "/".join(rel.parts[:-1])
        counts[key] = counts.get(key, 0) + 1
        digest.update(f"{rel.as_posix()}\0{p.stat().st_size}\n".encode())
    return {
        "n_files": len(files),
        "counts": dict(sorted(counts.items())),
        "sha256": digest.hexdigest(),
    }


def differences(expected: dict, actual: dict) -> list[str]:
    """Human-readable differences between two fingerprints (empty when identical)."""
    out = []
    for key in sorted(set(expected["counts"]) | set(actual["counts"])):
        a, b = expected["counts"].get(key), actual["counts"].get(key)
        if a != b:
            out.append(f"{key}: manifest has {a}, found {b}")
    if not out and expected["sha256"] != actual["sha256"]:
        out.append("file names or sizes differ (same counts, different hash)")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["write", "verify"], nargs="?", default="verify")
    ap.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = ap.parse_args(argv)
    actual = fingerprint(args.raw)
    if args.action == "write":
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(actual, indent=2) + "\n")
        print(f"wrote {args.manifest} ({actual['n_files']} files)")
        return 0
    diffs = differences(json.loads(args.manifest.read_text()), actual)
    if diffs:
        print("verify-data: FAILED\n  " + "\n  ".join(diffs))
        return 1
    print(f"verify-data: OK ({actual['n_files']} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
