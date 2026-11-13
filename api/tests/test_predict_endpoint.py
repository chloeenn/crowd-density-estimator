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
    assert heatmap_size(r.json()) == (80, 120)  # the overlay has the processed (rotated) size


def test_large_images_are_downscaled_to_the_long_side_limit(make_client):
    c = make_client(max_long_side=64)
    body = post(c, jpeg(400, 200)).json()
    assert heatmap_size(body) == (64, 32)


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


REGION_LEFT = "[[0,0],[0.5,0],[0.5,1],[0,1]]"


def test_no_density_without_an_area(client):
    body = post(client, jpeg(120, 80)).json()
    assert body["density"] is None and body["routed"] is False


def test_area_gives_people_per_m2_and_a_level(client):
    body = post(client, jpeg(120, 80), area_m2=25).json()
    d = body["density"]
    assert d["area_m2"] == 25 and d["in_region"] is False and d["units"] == "people per m^2"
    assert d["people_per_m2"] == pytest.approx(body["count"] / 25, abs=0.02)
    assert d["level"] in LEVELS and d["bands"] == {"medium": 1.0, "high": 2.0, "critical": 4.0}


def test_region_counts_inside_the_polygon_only(client):
    whole = post(client, jpeg(120, 80), area_m2=10).json()["density"]
    left = post(client, jpeg(120, 80), area_m2=10, region=REGION_LEFT).json()["density"]
    assert left["in_region"] is True and left["count"] <= whole["count"]


@pytest.mark.parametrize(
    ("params", "code"),
    [
        ({"area_m2": 0}, None),
        ({"area_m2": -5}, None),
        ({"region": REGION_LEFT}, "area_required"),
        ({"area_m2": 10, "region": "not json"}, "bad_region"),
        ({"area_m2": 10, "region": "[[0,0],[1,1]]"}, "bad_region"),
        ({"area_m2": 10, "region": "[[0,0],[2,0],[1,1]]"}, "bad_region"),
    ],
)
def test_bad_area_or_region_is_a_422(client, params, code):
    r = post(client, jpeg(120, 80), **params)
    assert r.status_code == 422
    if code:
        assert r.json()["code"] == code


def test_auto_routing_uses_dense_model_above_threshold_and_sparse_below(make_client):
    c = make_client()
    service = c.app.state.service
    service.routing_threshold = 1e9  # every count is below it -> sparse model
    body = post(c, jpeg(120, 80)).json()
    assert (body["routed"], body["model"]["part"]) == (True, "B")
    service.routing_threshold = 0.0  # no count is below it -> dense model
    body = post(c, jpeg(120, 80)).json()
    assert (body["routed"], body["model"]["part"]) == (True, "A")
    forced = post(c, jpeg(120, 80), part="B").json()  # an explicit part always wins
    assert (forced["routed"], forced["model"]["part"]) == (False, "B")


def test_metadata_reports_density_bands_and_routing(make_client):
    c = make_client()
    c.app.state.service.routing_threshold = 150.0
    meta = c.get("/api/metadata").json()
    assert meta["density_bands"] == {"medium": 1.0, "high": 2.0, "critical": 4.0}
    assert meta["routing_threshold"] == 150.0


def test_label_burns_text_onto_the_image_without_changing_the_numbers(client):
    data = jpeg(300, 200)
    plain = post(client, data).json()
    labelled = post(client, data, label="true").json()
    assert labelled["count"] == plain["count"] and labelled["level"] == plain["level"]
    assert labelled["heatmap"] != plain["heatmap"]
    assert heatmap_size(labelled) == heatmap_size(plain)


class FakeDetector:
    def detect(self, image):
        from crowd.detect import Detection

        h, w = image.shape[:2]
        return [
            Detection("person", 0.9, (1, 1, w // 2, h // 2)),
            Detection("car", 0.8, (5, 5, 20, 20)),
        ]


def test_detect_returns_labelled_boxes_and_an_image(make_client):
    c = make_client()
    c.app.state.detector = FakeDetector()
    body = post(c, jpeg(120, 80), detect="true").json()
    assert [d["label"] for d in body["detections"]] == ["person", "car"]
    assert body["detections"][0]["box"] == [1, 1, 60, 40]
    assert body["detections_image"].startswith("data:image/png;base64,")
    assert "detections" in PredictResponse.model_validate(body).model_fields_set


def test_detect_is_absent_by_default_and_422_when_not_enabled(client):
    assert post(client, jpeg(120, 80)).json()["detections"] is None
    r = post(client, jpeg(120, 80), detect="true")
    assert r.status_code == 422 and r.json()["code"] == "detector_unavailable"
