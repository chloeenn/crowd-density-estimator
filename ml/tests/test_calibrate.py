"""Calibration uses training annotations only, and excludes exactly the validation images
that training holds out."""

import numpy as np
import pytest
import scipy.io as sio
from PIL import Image

from crowd.calibrate import calibrate, training_counts
from crowd.config import DatasetConfig
from crowd.data.annotations import gt_path_for, load_points
from crowd.data.dataset import train_val_split


def write_part(raw, part, counts):
    """Synthetic ShanghaiTech part: images and .mat files for train_data, and a *test_data*
    split with absurd counts that must never influence the result."""
    for split, cs in (("train_data", counts), ("test_data", [9999] * 3)):
        (raw / part / split / "images").mkdir(parents=True)
        (raw / part / split / "ground-truth").mkdir(parents=True)
        for i, c in enumerate(cs):
            Image.new("RGB", (8, 8)).save(raw / part / split / "images" / f"IMG_{i}.jpg")
            info = np.empty((1, 1), dtype=object)
            info[0, 0] = np.array(
                [(np.zeros((c, 2)), c)], dtype=[("location", "O"), ("number", "O")]
            )
            sio.savemat(
                raw / part / split / "ground-truth" / f"GT_IMG_{i}.mat", {"image_info": info}
            )


@pytest.fixture
def raw(tmp_path):
    write_part(tmp_path, "part_A", list(range(10, 110, 5)))  # 20 images
    write_part(tmp_path, "part_B", list(range(1, 21)))  # 20 images
    return tmp_path


def test_thresholds_come_from_training_counts_minus_validation(raw):
    result = calibrate(raw)
    counts, excluded = training_counts(raw, DatasetConfig())
    assert result["n_images"] == len(counts) == 36  # 40 training images minus 4 validation
    assert excluded == result["n_validation_excluded"] == 4
    assert result["bands"]["critical"] == round(np.percentile(counts, 90))
    assert result["bands"]["medium"] == round(np.percentile(counts, 50))


def test_the_test_split_is_never_read(raw):
    # The synthetic test_data counts are 9999; they would show up as the maximum if read.
    assert calibrate(raw)["count_max"] < 9999


def test_calibration_excludes_exactly_the_images_training_holds_out(raw):
    cfg = DatasetConfig()
    expected = []
    for part in ("part_A", "part_B"):
        images = sorted((raw / part / "train_data" / "images").glob("*.jpg"))
        train, _ = train_val_split(images, cfg)  # the very function the trainer uses
        expected += [len(load_points(gt_path_for(i))) for i in train]
    counts, _ = training_counts(raw, cfg)
    assert counts == expected


def test_records_the_split_seed_and_pooled_parts(raw):
    r = calibrate(raw, DatasetConfig(split_seed=7))
    assert r["validation_seed"] == 7 and r["parts_pooled"] == ["part_A", "part_B"]
    assert r["split"] == "train_data minus validation"


def test_missing_part_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="no training images"):
        calibrate(tmp_path)
