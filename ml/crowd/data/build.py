"""Precompute density maps: raw ShanghaiTech -> data/processed/*.h5.

    python -m crowd.data.build --raw data/ShanghaiTech --out data/processed

Precomputing matters: building a KDTree and thousands of kernels per epoch would
dominate training time. Output mirrors the raw layout, one .h5 per image.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
from PIL import Image

from crowd.config import DENSITY_BY_PART, DensityConfig
from crowd.data.annotations import load_points
from crowd.data.density import density_map

PARTS = ("part_A", "part_B")
SPLITS = ("train_data", "test_data")


def gt_path_for(image_path: Path) -> Path:
    """IMG_12.jpg -> ../ground-truth/GT_IMG_12.mat"""
    return image_path.parent.parent / "ground-truth" / f"GT_{image_path.stem}.mat"


def build_one(image_path: Path, out_path: Path, cfg: DensityConfig) -> int:
    """Write one density map; returns the annotated head count."""
    with Image.open(image_path) as im:
        width, height = im.size
    points = load_points(gt_path_for(image_path))
    dmap = density_map(points, height, width, cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("density", data=dmap, compression="gzip")
        f.attrs["count"] = len(points)
    return len(points)


def build_all(raw: Path, out: Path, parts=PARTS, overwrite: bool = False) -> None:
    for part in parts:
        for split in SPLITS:
            images = sorted((raw / part / split / "images").glob("*.jpg"))
            done = 0
            for img in images:
                target = out / part / split / "density" / f"{img.stem}.h5"
                if target.exists() and not overwrite:
                    continue
                build_one(img, target, DENSITY_BY_PART[part])
                done += 1
            print(f"{part}/{split}: {done} written, {len(images) - done} skipped")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw", type=Path, default=Path("data/ShanghaiTech"))
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    ap.add_argument("--parts", nargs="+", default=list(PARTS), choices=PARTS)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    build_all(args.raw, args.out, args.parts, args.overwrite)


if __name__ == "__main__":
    main()
