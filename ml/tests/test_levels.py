import pytest

from crowd.levels import (
    LEVELS,
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
