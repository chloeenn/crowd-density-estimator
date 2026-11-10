import numpy as np
import pytest

from crowd.levels import (
    LEVELS,
    concentration,
    concentration_basis,
    level_for_count,
    load_thresholds,
    thresholds_sha,
)

BANDS = {"low": 85, "medium": 166, "high": 351, "critical": 687}


@pytest.mark.parametrize(
    ("count", "level"),
    [
        (0, "Low"),
        (84.9, "Low"),
        (165.9, "Low"),  # below `medium` is Low, including below the `low` edge
        (166, "Medium"),  # edges are inclusive lower bounds
        (350.9, "Medium"),
        (351, "High"),
        (686.9, "High"),
        (687, "Critical"),
        (5000, "Critical"),
    ],
)
def test_level_boundaries(count, level):
    assert level_for_count(count, BANDS) == level


def test_levels_are_monotonic_in_count():
    ranks = [LEVELS.index(level_for_count(c, BANDS)) for c in range(0, 2000, 7)]
    assert ranks == sorted(ranks)


def test_shipped_thresholds_are_ordered_and_record_their_training_split():
    t = load_thresholds()
    b = t["bands"]
    assert b["low"] < b["medium"] < b["high"] < b["critical"]
    assert t["split"] == "train_data minus validation"
    assert t["parts_pooled"] == ["part_A", "part_B"] and t["n_validation_excluded"] > 0
    assert len(thresholds_sha()) == 64


def test_concentration_is_p95_and_max_of_the_cells():
    d = np.zeros((10, 10), np.float32)
    d[0, 0] = 5.0  # a single hot cell: 1% of cells, so it does not move the p95
    p95, peak = concentration(d)
    assert (p95, peak) == (0.0, 5.0)
    d[:, :] = np.arange(100).reshape(10, 10) / 100
    p95, peak = concentration(d)
    assert p95 == pytest.approx(np.percentile(d, 95)) and peak == pytest.approx(0.99)


def test_concentration_ignores_negative_values():
    assert concentration(np.full((4, 4), -2.0)) == (0.0, 0.0)


def test_concentration_basis_reports_the_measurement_scale():
    b = concentration_basis((1067, 1600))
    assert (b["cell_input_px"], b["processed_width"], b["processed_height"]) == (8, 1600, 1067)
    assert "not people/m^2" in b["units"]
