"""Central, frozen configuration for the data pipeline.

Keeping every tunable here (instead of inline literals) makes runs reproducible and
lets docs point at one place.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

# VGG-16 features[:23] (first 10 convs) downsamples by 8, so CSRNet outputs a
# density map at 1/8 of the input resolution.
OUTPUT_STRIDE = 8

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class DensityConfig:
    """How annotated head points are turned into a density map.

    Part A (dense crowds) uses a geometry-adaptive kernel: sigma = beta * mean distance
    to the k nearest neighbouring heads. Part B (sparse) uses a fixed sigma.
    """

    adaptive: bool
    beta: float = 0.3
    k: int = 3
    fixed_sigma: float = 15.0
    min_sigma: float = 1.0  # floor so a degenerate (duplicate-point) kernel stays valid
    truncate: float = 4.0  # kernel radius in sigmas


DENSITY_BY_PART = {
    "part_A": DensityConfig(adaptive=True),
    "part_B": DensityConfig(adaptive=False, fixed_sigma=15.0),
}


# (MAE, RMSE) acceptance bar for this project, from docs/PLAN.md section 4.
ACCEPTANCE = {"part_A": (85.0, 135.0), "part_B": (16.0, 24.0)}
# Published CSRNet results (Li et al., CVPR 2018), for reference only.
PUBLISHED = {"part_A": (68.2, 115.0), "part_B": (10.6, 16.0)}


def preprocess_signature(part: str) -> dict:
    """Everything that must match between training and serving. Stored in each checkpoint;
    `load_model` compares it with the live config and refuses to run on a mismatch."""
    d = DENSITY_BY_PART[part]
    return {
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
        "stride": OUTPUT_STRIDE,
        "pad_multiple": OUTPUT_STRIDE,
        "sigma_strategy": "adaptive" if d.adaptive else "fixed",
        "beta": d.beta,
        "k": d.k,
        "fixed_sigma": d.fixed_sigma,
    }


@dataclass(frozen=True)
class DatasetConfig:
    crop_size: tuple[int, int] = (256, 256)  # (H, W); multiples of OUTPUT_STRIDE
    val_fraction: float = 0.1  # held out of *train*; test is never used for selection
    split_seed: int = 0


@dataclass(frozen=True)
class TrainConfig:
    """Everything that defines a training run; saved next to every checkpoint.

    OPTIMISER CHOICE. The default is Adam at lr 1e-5. The CSRNet paper's recipe (SGD,
    lr 1e-7, momentum 0.95) needs ~100k+ single-image steps to converge; with fixed-size
    crops in batches of 8 we take far fewer steps, and in a 10-image overfit check that recipe
    barely left the predict-nothing baseline (MAE 134 -> 90 in 150 steps; SGD at 1e-6 stalled
    at 56) while Adam at 1e-5 fit it (correlation with ground truth 0.86). The paper recipe
    stays available: --optimizer sgd --lr 1e-7 --weight-decay 5e-4.

    LOSS AND LEARNING RATE ARE COUPLED. The loss is MSE with reduction="sum" (divided by
    batch size), which adds up every output pixel, so gradients are far larger than with
    reduction="mean". Learning rates stay small for this reason. Change both together.
    Changing one makes training diverge or stall.
    """

    part: str = "part_A"
    raw: Path = Path("data/ShanghaiTech")
    processed: Path = Path("data/processed")
    runs_dir: Path = Path("runs")  # each run gets runs/<run_id>/
    run_id: str | None = None  # default: <part>-<YYYYMMDD-HHMM>-<git_sha>
    publish_dir: Path | None = Path("docs/runs")  # tracked copy of the summary; None = skip
    data: DatasetConfig = DatasetConfig()

    epochs: int = 150
    batch_size: int = 8  # crops are fixed-size, so batching works; validation uses 1
    optimizer: str = "adam"  # "adam" | "sgd"
    lr: float = 1e-5
    lr_schedule: str = "constant"  # "constant" | "cosine" (see crowd.schedule)
    lr_min_factor: float = 0.01  # cosine floor = lr * lr_min_factor
    momentum: float = 0.95  # sgd only
    weight_decay: float = 0.0
    freeze_frontend_epochs: int = 0  # optional warmup: train only the backend at first
    pretrained: bool = True

    val_every: int = 1
    patience: int = 0  # stop after this many epochs without val-MAE improvement; 0 = off
    num_workers: int = 4
    device: str = "auto"  # "auto" | "cuda" | "mps" | "cpu"
    seed: int = 0
    resume: bool = False  # continue run_id, or the latest run of `part` if run_id is None

    def to_dict(self) -> dict:
        """Plain-python dict (Paths as str) so it can live inside a safe checkpoint."""

        def plain(v):
            if isinstance(v, Path):
                return str(v)
            if isinstance(v, dict):
                return {k: plain(x) for k, x in v.items()}
            if isinstance(v, tuple):
                return list(v)
            return v

        return plain(asdict(self))
