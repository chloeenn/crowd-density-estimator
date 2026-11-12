"""Health, metadata, startup refusal, CORS, request ids, logging."""

import json

import pytest
from conftest import write_checkpoint
from crowd.config import preprocess_signature
from crowd.levels import load_thresholds
from fastapi.testclient import TestClient
from images import jpeg

from app.main import create_app


def test_health_is_503_until_models_are_loaded(make_settings):
    cold = TestClient(create_app(make_settings()))  # lifespan not run: nothing loaded yet
    r = cold.get("/api/health")
    assert r.status_code == 503 and r.json() == {"status": "unavailable", "models": []}
    assert (
        cold.post("/api/predict", files={"file": ("a.jpg", jpeg(), "image/jpeg")}).status_code
        == 503
    )
    assert cold.post("/api/predict", files={"file": ("a.jpg", jpeg(), "image/jpeg")}).headers[
        "retry-after"
    ]


def test_health_is_200_once_loaded(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json() == {"status": "ok", "models": ["A", "B"]}


def test_metadata_describes_models_thresholds_and_limits(client):
    body = client.get("/api/metadata").json()
    assert body["default_part"] == "A" and [m["part"] for m in body["models"]] == ["A", "B"]
    a = body["models"][0]
    assert (
        a["version"] == "csrnet-partA-v1"
        and a["run_id"] == "part_A-test"
        and a["thresholds_match"] is True
    )
    assert body["thresholds"] == load_thresholds()["bands"]
    assert body["thresholds_info"]["split"] == "train_data minus validation"
    limits = body["limits"]
    assert limits["max_upload_mb"] == 10 and limits["allowed_content_types"] == [
        "image/jpeg", "image/png", "image/webp",
    ]  # fmt: skip


def test_metadata_flags_a_model_trained_with_other_thresholds(make_client, tmp_path):
    other = tmp_path / "a.pth"
    write_checkpoint(other, "A", thresholds_sha="0" * 64)
    c = make_client(model_path=other, alt_model_path=None)
    assert c.get("/api/metadata").json()["models"][0]["thresholds_match"] is False


def test_metadata_leaves_thresholds_match_unknown_for_older_checkpoints(make_client, tmp_path):
    older = tmp_path / "a.pth"
    write_checkpoint(older, "A", thresholds_sha=None)
    c = make_client(model_path=older, alt_model_path=None)
    assert c.get("/api/metadata").json()["models"][0]["thresholds_match"] is None


# --- startup refusal -------------------------------------------------------------------


def test_startup_refuses_a_checkpoint_with_different_preprocessing(make_settings, tmp_path):
    bad = tmp_path / "bad.pth"
    write_checkpoint(bad, "A", preprocess={**preprocess_signature("part_A"), "stride": 16})
    with (
        pytest.raises(ValueError, match="stride"),
        TestClient(create_app(make_settings(model_path=bad))),
    ):
        pass


def test_startup_refuses_a_missing_checkpoint(make_settings, tmp_path):
    with (
        pytest.raises(FileNotFoundError, match="not found"),
        TestClient(create_app(make_settings(model_path=tmp_path / "nope.pth"))),
    ):
        pass


def test_startup_refuses_a_checkpoint_for_the_wrong_part(make_settings, checkpoints):
    with (
        pytest.raises(ValueError, match="holds Part A, expected Part B"),
        TestClient(
            create_app(
                make_settings(model_part="B", model_path=checkpoints["A"], alt_model_path=None)
            )
        ),
    ):
        pass


def test_alt_model_must_be_the_other_part(make_settings, checkpoints):
    with (
        pytest.raises(ValueError, match="expected Part B"),
        TestClient(create_app(make_settings(alt_model_path=checkpoints["A"]))),
    ):
        pass


# --- CORS, request ids, logs ------------------------------------------------------------


def test_cors_allows_only_configured_origins(make_client):
    c = make_client(cors_origins="http://good.example, http://also.example")
    ok = c.get("/api/health", headers={"Origin": "http://good.example"})
    assert ok.headers["access-control-allow-origin"] == "http://good.example"
    assert (
        "access-control-allow-origin"
        not in c.get("/api/health", headers={"Origin": "http://evil.example"}).headers
    )
    pre = c.options(
        "/api/predict",
        headers={"Origin": "http://good.example", "Access-Control-Request-Method": "POST"},
    )
    assert (
        pre.status_code == 200
        and pre.headers["access-control-allow-origin"] == "http://good.example"
    )


def test_cors_headers_also_cover_error_responses(make_client):
    c = make_client(cors_origins="http://good.example")
    r = c.post(
        "/api/predict",
        files={"file": ("a.txt", b"x", "text/plain")},
        headers={"Origin": "http://good.example"},
    )
    assert (
        r.status_code == 415 and r.headers["access-control-allow-origin"] == "http://good.example"
    )


def test_valid_request_id_is_echoed_and_bad_ones_replaced(client):
    assert (
        client.get("/api/health", headers={"X-Request-ID": "abc-123_x"}).headers["x-request-id"]
        == "abc-123_x"
    )
    for bad in ("has space", "x" * 100, "semi;colon"):
        assert (
            client.get("/api/health", headers={"X-Request-ID": bad}).headers["x-request-id"] != bad
        )


def test_each_request_logs_one_structured_line_without_filenames(client, log_lines):
    secret = "my_private_holiday_photo.jpg"
    client.post(
        "/api/predict",
        files={"file": (secret, jpeg(), "image/jpeg")},
        headers={"X-Request-ID": "req-1"},
    )
    client.post(
        "/api/predict",
        files={"file": (secret, b"junk", "text/plain")},
        headers={"X-Request-ID": "req-2"},
    )
    text = "\n".join(log_lines())
    assert secret not in text and "my_private" not in text
    rows = {r["request_id"]: r for r in map(json.loads, log_lines()) if r["msg"] == "request"}
    assert rows["req-1"]["status"] == 200 and rows["req-1"]["path"] == "/api/predict"
    assert (
        rows["req-1"]["inference_ms"] >= 0
        and rows["req-1"]["part"] == "A"
        and "duration_ms" in rows["req-1"]
    )
    assert rows["req-2"]["status"] == 415 and rows["req-2"]["inference_ms"] is None


def test_unhandled_errors_are_logged_server_side_with_the_request_id(
    client, log_lines, monkeypatch
):
    from crowd.predict import Predictor

    monkeypatch.setattr(Predictor, "predict", lambda self, image: 1 / 0)
    r = client.post(
        "/api/predict",
        files={"file": ("a.jpg", jpeg(), "image/jpeg")},
        headers={"X-Request-ID": "boom-1"},
    )
    assert r.status_code == 500 and "ZeroDivision" not in r.text
    logged = [json.loads(line) for line in log_lines()]
    assert any(row.get("request_id") == "boom-1" and "exc" in row for row in logged)
