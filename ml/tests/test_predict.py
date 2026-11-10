"""Serving inference must match the evaluation path exactly (train/serve parity)."""

import numpy as np
import pytest
import torch
from helpers import make_dataset_tree, tiny_cfg

from crowd.checkpoint import save_checkpoint
from crowd.config import preprocess_signature
from crowd.data.dataset import CrowdDataset
from crowd.engine import predict_counts
from crowd.models.csrnet import CSRNet
from crowd.predict import Predictor
from crowd.train import fit
from crowd.transforms import downscale_long_side, load_rgb


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    base = tmp_path_factory.mktemp("pred")
    raw, processed, train_s, test_s = make_dataset_tree(base, "part_A", n_train=3, n_test=3)
    fit(
        tiny_cfg(base, part="part_A", run_id="part_A-t", raw=raw, processed=processed),
        train_s[:2],
        train_s[2:],
    )
    return base / "runs" / "part_A-t" / "best.pth", test_s


def test_predictor_matches_the_evaluation_path(trained):
    ckpt, samples = trained
    predictor = Predictor.from_checkpoint(ckpt)
    loader = torch.utils.data.DataLoader(CrowdDataset(samples, train=False), batch_size=1)
    eval_counts, _ = predict_counts(predictor.model, loader, "cpu")
    served = [predictor.predict(load_rgb(s.image)).count for s in samples]
    assert served == pytest.approx(eval_counts, rel=1e-5)


def test_prediction_shapes_and_properties(trained):
    ckpt, _ = trained
    p = Predictor.from_checkpoint(ckpt)
    image = np.random.default_rng(0).integers(0, 255, (90, 121, 3), dtype=np.uint8)
    out = p.predict(image)
    assert out.density.shape == (12, 16)  # ceil(90/8), ceil(121/8): padding cropped away
    assert out.density.min() >= 0 and out.count >= 0 and np.isfinite(out.count)
    assert out.image.shape == image.shape
    assert (p.part, p.version) == ("A", "csrnet-partA-v1")


def test_long_side_is_downscaled_before_inference(trained):
    p = Predictor.from_checkpoint(trained[0], max_long_side=64)
    out = p.predict(np.zeros((100, 200, 3), np.uint8))
    assert out.image.shape == (32, 64, 3)
    assert out.density.shape == (4, 8)


def test_downscale_keeps_aspect_ratio_and_never_upscales():
    big = np.zeros((300, 600, 3), np.uint8)
    assert downscale_long_side(big, 100).shape == (50, 100, 3)
    small = np.zeros((30, 40, 3), np.uint8)
    assert downscale_long_side(small, 100) is small
    assert downscale_long_side(np.zeros((1, 5000, 3), np.uint8), 100).shape == (1, 100, 3)


def test_predictor_refuses_a_checkpoint_with_different_preprocessing(tmp_path):
    meta = {
        "part": "A",
        "model_version": "x",
        "preprocess": {**preprocess_signature("part_A"), "stride": 16},
    }
    save_checkpoint(tmp_path / "bad.pth", model=CSRNet(pretrained=False), meta=meta)
    with pytest.raises(ValueError, match="stride"):
        Predictor.from_checkpoint(tmp_path / "bad.pth")


def test_warmup_runs_a_forward_pass(trained):
    Predictor.from_checkpoint(trained[0]).warmup()
