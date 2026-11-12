import numpy as np
import pytest

from crowd.area import (
    count_in_region,
    level_for_density,
    parse_region,
    people_per_m2,
    region_mask,
)

SQUARE = [[0, 0], [1, 0], [1, 1], [0, 1]]
LEFT_HALF = [[0, 0], [0.5, 0], [0.5, 1], [0, 1]]


@pytest.mark.parametrize(
    ("per_m2", "level"),
    [(0, "Low"), (0.99, "Low"), (1, "Medium"), (1.99, "Medium"), (2, "High"), (3.99, "High"),
     (4, "Critical"), (9, "Critical")],
)  # fmt: skip
def test_density_levels_start_at_their_edge(per_m2, level):
    assert level_for_density(per_m2) == level


def test_people_per_m2():
    assert people_per_m2(300, 100) == 3
    with pytest.raises(ValueError, match="positive"):
        people_per_m2(10, 0)


def test_whole_image_count_without_a_region():
    d = np.full((8, 8), 0.5, np.float32)
    assert count_in_region(d, None) == pytest.approx(32)


def test_region_counts_only_inside_the_polygon():
    d = np.ones((8, 8), np.float32)
    assert count_in_region(d, parse_region(LEFT_HALF)) == pytest.approx(32, abs=0.5)
    assert count_in_region(d, parse_region(SQUARE)) == pytest.approx(64, abs=0.5)


def test_polygon_edge_splits_a_cell_in_proportion():
    d = np.ones((4, 4), np.float32)
    quarter_off = parse_region([[0, 0], [0.625, 0], [0.625, 1], [0, 1]])  # ends mid-cell
    assert count_in_region(d, quarter_off) == pytest.approx(10, abs=0.2)


def test_mask_is_a_fraction_per_cell():
    m = region_mask(parse_region(LEFT_HALF), (4, 4))
    assert m.shape == (4, 4) and m.min() >= 0 and m.max() <= 1


@pytest.mark.parametrize(
    "bad",
    [
        [[0, 0], [1, 1]],
        [[0, 0], [1, 0], [2, 1]],
        [[0, 0], [0.5, 0.5], [1, 1]],
        [[0, 0], [1], [1, 1]],
    ],
    ids=["two-points", "outside-image", "no-area", "malformed-point"],
)
def test_bad_regions_are_rejected(bad):
    with pytest.raises(ValueError, match="region"):
        parse_region(bad)
