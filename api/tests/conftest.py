"""Fixtures: real checkpoint files holding *randomly initialised* models, so tests exercise
the genuine load path (preprocess check included) without needing a trained model."""

from __future__ import annotations

import io
import logging

import pytest
import torch
from crowd.checkpoint import save_checkpoint
from crowd.config import preprocess_signature
from crowd.levels import thresholds_sha
from crowd.models.csrnet import CSRNet
from fastapi.testclient import TestClient

from app.config import Settings
from app.logging_setup import LOGGER_NAME, JsonFormatter
from app.main import create_app


def write_checkpoint(path, part: str, **meta_overrides) -> None:
    """Save a random-weight CSRNet with valid metadata for `part` ('A' or 'B')."""
    torch.manual_seed(0)
    meta = {
        "model_version": f"csrnet-part{part}-v1",
        "part": part,
        "run_id": f"part_{part}-test",
        "git_sha": "abc1234",
        "epoch": 3,
        "val_mae": None,
        "val_rmse": None,
        "preprocess": preprocess_signature(f"part_{part}"),
        "thresholds_sha": thresholds_sha(),
    }
    save_checkpoint(path, model=CSRNet(pretrained=False), meta=meta | meta_overrides)


@pytest.fixture(scope="session")
def checkpoints(tmp_path_factory) -> dict[str, object]:
    base = tmp_path_factory.mktemp("models")
    paths = {"A": base / "partA.pth", "B": base / "partB.pth"}
    for part, path in paths.items():
        write_checkpoint(path, part)
    return paths


@pytest.fixture(scope="session")
def make_settings(checkpoints):
    """Settings factory that ignores any local .env and defaults to both models loaded."""

    def factory(**overrides) -> Settings:
        values = {
            "model_path": checkpoints["A"],
            "model_part": "A",
            "alt_model_path": checkpoints["B"],
            "max_long_side": 256,  # keep inference fast in tests
            "auto_route": False,  # tests pick a model explicitly; routing has its own tests
        }
        return Settings(_env_file=None, **(values | overrides))

    return factory


@pytest.fixture
def make_client(make_settings):
    """Build and start an app (runs the lifespan, i.e. loads the models); closes on teardown."""
    clients: list[TestClient] = []

    def factory(**overrides) -> TestClient:
        client = TestClient(create_app(make_settings(**overrides)))
        client.__enter__()
        clients.append(client)
        return client

    yield factory
    for c in clients:
        c.__exit__(None, None, None)


@pytest.fixture(scope="module")
def client(make_settings) -> TestClient:
    """A default client shared by a module's read-only tests."""
    with TestClient(create_app(make_settings())) as c:
        yield c


@pytest.fixture
def log_lines():
    """Capture everything the API logger writes, as formatted JSON lines."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger(LOGGER_NAME)
    logger.addHandler(handler)  # added after create_app's setup; removed on teardown
    yield lambda: [line for line in stream.getvalue().splitlines() if line]
    logger.removeHandler(handler)
