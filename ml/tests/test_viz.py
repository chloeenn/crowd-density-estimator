import numpy as np
import pytest

from crowd.viz import crop_to_image, density_to_heatmap, overlay_heatmap


def test_crop_removes_padding_rows_and_columns():
    padded = np.zeros((12, 16))  # image 90x121 -> ceil(90/8)=12, ceil(121/8)=16
    assert crop_to_image(padded, (90, 121)).shape == (12, 16)
    assert crop_to_image(np.zeros((13, 17)), (90, 121)).shape == (12, 16)  # one extra pad row/col
    assert crop_to_image(np.zeros((12, 16)), (96, 128)).shape == (12, 16)  # already aligned


def test_heatmap_is_red_where_dense_and_blue_where_sparse():
    d = np.zeros((8, 8), np.float32)
    d[4:, 4:] = 1.0
    heat, vmax = density_to_heatmap(d, (64, 64))
    assert vmax == 1.0 and heat.shape == (64, 64, 3) and heat.dtype == np.uint8
    dense, sparse = heat[60, 60], heat[2, 2]  # RGB
    assert dense[0] > dense[2] and sparse[2] > sparse[0]


def test_all_zero_density_does_not_divide_by_zero():
    heat, vmax = density_to_heatmap(np.zeros((4, 4)), (32, 32))
    assert vmax == 0.0 and np.isfinite(heat).all()


def test_negative_density_is_clipped():
    _, vmax = density_to_heatmap(np.full((4, 4), -3.0), (16, 16))
    assert vmax == 0.0


def test_shared_vmax_makes_two_maps_comparable():
    weak, strong = np.full((4, 4), 1.0), np.full((4, 4), 4.0)
    a, _ = density_to_heatmap(weak, (8, 8), vmax=4.0)
    b, _ = density_to_heatmap(strong, (8, 8), vmax=4.0)
    assert not np.array_equal(a, b)
    own, _ = density_to_heatmap(weak, (8, 8))  # own max -> same colour as the strong one
    assert np.array_equal(own, b)


def test_overlay_blends_with_alpha_and_keeps_shape():
    image = np.full((32, 48, 3), 200, np.uint8)
    out, vmax = overlay_heatmap(image, np.ones((4, 6)), alpha=0.0)
    assert np.array_equal(out, image) and vmax == 1.0
    out, _ = overlay_heatmap(image, np.ones((4, 6)), alpha=0.4)
    assert out.shape == image.shape and out.dtype == np.uint8 and not np.array_equal(out, image)


@pytest.mark.parametrize("alpha", [-0.1, 1.5])
def test_overlay_alpha_outside_unit_interval_still_returns_uint8(alpha):
    out, _ = overlay_heatmap(np.full((8, 8, 3), 100, np.uint8), np.ones((1, 1)), alpha=alpha)
    assert out.dtype == np.uint8


def test_png_data_uri_roundtrips_pixels_exactly():
    import base64

    import cv2

    from crowd.viz import to_png_data_uri

    rgb = np.random.default_rng(0).integers(0, 255, (6, 9, 3), dtype=np.uint8)
    uri = to_png_data_uri(rgb)
    assert uri.startswith("data:image/png;base64,")
    raw = np.frombuffer(base64.b64decode(uri.split(",", 1)[1]), np.uint8)
    decoded = cv2.cvtColor(cv2.imdecode(raw, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    assert np.array_equal(decoded, rgb)  # PNG is lossless; channel order preserved
