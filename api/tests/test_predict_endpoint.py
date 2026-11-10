"""POST /api/predict: the response contract."""

import base64
from io import BytesIO

import cv2
import numpy as np
import pytest
from crowd.levels import LEVELS, load_thresholds
from crowd.predict import Predictor
from images import encode, jpeg, noise, png, webp
from PIL import Image

from app.schemas import PredictResponse


def post(client, data: bytes, content_type="image/jpeg", name="a.jpg", **params):
    return client.post("/api/predict", files={"file": (name, data, content_type)}, params=params)


def heatmap_size(body) -> tuple[int, int]:
    raw = base64.b64decode(body["heatmap"].split(",", 1)[1])
    h, w = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR).shape[:2]
    return w, h


def test_valid_image_returns_a_well_formed_body(client):
    r = post(client, jpeg(120, 80))
    assert r.status_code == 200
    body = r.json()
    parsed = PredictResponse.model_validate(body)  # the schema is the contract
    assert parsed.count >= 0 and body["level"] in LEVELS
    assert body["thresholds"] == load_thresholds()["bands"]
    assert body["model"] == {"version": "csrnet-partA-v1", "part": "A", "git_sha": "abc1234"}
    assert body["heatmap"].startswith("data:image/png;base64,")
    assert heatmap_size(body) == (120, 80)  # overlay is at the processed image size
    basis = body["concentration_basis"]
    assert (basis["processed_width"], basis["processed_height"], basis["cell_input_px"]) == (
        120,
        80,
        8,
    )
    assert body["local_concentration"] <= body["max_concentration"]
    assert isinstance(body["inference_ms"], int) and body["inference_ms"] >= 0
    assert r.headers["x-request-id"] and r.headers["cache-control"] == "no-store"


@pytest.mark.parametrize(
    ("data", "content_type"),
    [
        (png(), "image/png"),
        (png(mode="RGBA"), "image/png"),
        (png(mode="L"), "image/png"),
        (png(mode="P"), "image/png"),
        (webp(), "image/webp"),
    ],
    ids=["png", "rgba", "grayscale", "palette", "webp"],
)
def test_other_formats_and_colour_modes_are_accepted(client, data, content_type):
    assert post(client, data, content_type).status_code == 200


def test_exif_orientation_is_applied(client):
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90: a 120x80 file displays as 80x120
    r = post(client, encode(noise(120, 80), "JPEG", exif=exif))
    basis = r.json()["concentration_basis"]
    assert (basis["processed_width"], basis["processed_height"]) == (80, 120)


def test_large_images_are_downscaled_to_the_long_side_limit(make_client):
    c = make_client(max_long_side=64)
    body = post(c, jpeg(400, 200)).json()
    assert heatmap_size(body) == (64, 32)
    assert body["concentration_basis"]["processed_width"] == 64


def test_model_switch_selects_the_requested_part(client):
    assert post(client, jpeg()).json()["model"]["part"] == "A"  # default
    assert post(client, jpeg(), part="B").json()["model"]["part"] == "B"


def test_unknown_part_is_a_readable_422(client):
    r = post(client, jpeg(), part="C")
    assert (
        r.status_code == 422
        and r.json()["code"] == "invalid_request"
        and "part" in r.json()["detail"]
    )


def test_part_that_is_not_loaded_is_a_422_listing_what_is(make_client):
    c = make_client(alt_model_path=None)
    r = post(c, jpeg(), part="B")
    assert r.status_code == 422 and r.json()["code"] == "model_unavailable"
    assert "Available: A" in r.json()["detail"]


def test_fixed_scale_changes_the_heatmap_but_not_the_measurements(client):
    data = jpeg()
    auto, fixed = post(client, data).json(), post(client, data, scale_max=50.0).json()
    assert auto["heatmap"] != fixed["heatmap"]
    assert auto["max_concentration"] == fixed["max_concentration"]
    assert auto["count"] == fixed["count"]


@pytest.mark.parametrize("scale", [0, -1, 2_000_000, "abc"])
def test_invalid_fixed_scale_is_rejected(client, scale):
    r = post(client, jpeg(), scale_max=scale)
    assert r.status_code == 422 and "scale_max" in r.json()["detail"]


def test_prediction_is_deterministic(client):
    data = jpeg()
    assert post(client, data).json()["count"] == post(client, data).json()["count"]


def test_api_count_equals_the_library_prediction(client, checkpoints):
    """Train/serve parity at the HTTP boundary: same bytes, same number."""
    data = jpeg(120, 80)
    served = post(client, data).json()["count"]
    direct = Predictor.from_checkpoint(checkpoints["A"], max_long_side=256).predict(
        np.asarray(Image.open(BytesIO(data)).convert("RGB"))
    )
    assert served == pytest.approx(round(direct.count, 1))


def test_missing_file_field_is_a_readable_422(client):
    r = client.post("/api/predict", data={"x": "y"})
    assert r.status_code == 422 and "'file'" in r.json()["detail"]


def test_unexpected_failure_returns_a_generic_500_without_internals(client, monkeypatch):
    def boom(self, image):
        raise RuntimeError("secret /internal/path")

    monkeypatch.setattr(Predictor, "predict", boom)
    r = post(client, jpeg())
    assert r.status_code == 500 and r.json() == {
        "detail": "Internal server error.",
        "code": "internal_error",
    }
    assert "secret" not in r.text and "Traceback" not in r.text


def test_unknown_route_uses_the_same_error_shape(client):
    r = client.get("/api/nope")
    assert r.status_code == 404 and set(r.json()) == {"detail", "code"}
