"""Pure metric logic: buckets, acceptance bar, diagnosis."""

import pytest

from crowd.metrics import (
    BUCKET_EDGES,
    bucket_label,
    bucket_metrics,
    check_acceptance,
    count_metrics,
    diagnose,
)


def test_bucket_labels():
    assert [bucket_label(lo, hi) for lo, hi in [(0, 100), (100, 500), (1000, None)]] == [
        "0-99",
        "100-499",
        "1000+",
    ]


def test_buckets_assign_by_ground_truth_with_inclusive_lower_edges():
    truth = [0, 99, 100, 499, 500, 999, 1000, 3000]
    rows = bucket_metrics([0.0] * 8, truth)
    assert [r.label for r in rows] == ["0-99", "100-499", "500-999", "1000+"]
    assert [r.n for r in rows] == [2, 2, 2, 2]


def test_bucket_edges_tolerate_float_noise_from_density_sums():
    # Both are a true count of 100 plus/minus float noise, so both belong in 100-499,
    # not split across the 99/100 boundary.
    rows = bucket_metrics([5.0, 5.0], [99.9999999, 100.0000001])
    assert [r.n for r in rows] == [0, 2, 0, 0]


def test_bucket_stats_and_signed_bias():
    rows = bucket_metrics([110.0, 130.0], [100.0, 150.0])  # errors +10, -20 in 100-499
    b = rows[1]
    assert (b.n, b.mae, b.bias) == (2, 15.0, -5.0)
    assert b.rmse == pytest.approx((250) ** 0.5)


def test_empty_buckets_are_kept_with_none_stats():
    rows = bucket_metrics([50.0], [50.0])
    assert len(rows) == len(BUCKET_EDGES)
    assert rows[3].n == 0 and rows[3].mae is None and rows[3].bias is None


def test_bucket_metrics_rejects_length_mismatch():
    with pytest.raises(ValueError, match="equal-length"):
        bucket_metrics([1.0], [1.0, 2.0])


@pytest.mark.parametrize(
    ("part", "mae", "rmse", "expected"),
    [
        ("part_A", 85.0, 135.0, (True, True)),  # bar is inclusive
        ("part_A", 85.1, 120.0, (False, True)),
        ("part_A", 60.0, 135.5, (True, False)),
        ("part_B", 16.0, 24.0, (True, True)),
        ("part_B", 17.0, 25.0, (False, False)),
    ],
)
def test_acceptance_bar(part, mae, rmse, expected):
    from crowd.metrics import Metrics

    v = check_acceptance(part, Metrics(mae, rmse, 10))
    assert (v.mae_ok, v.rmse_ok) == expected and v.passed == all(expected)


def _diagnose(part, pred, truth):
    return diagnose(
        part,
        pred,
        truth,
        bucket_metrics(pred, truth),
        check_acceptance(part, count_metrics(pred, truth)),
    )


def test_diagnosis_reports_undercounting_and_error_concentration():
    truth = [50.0, 60.0, 2000.0]
    pred = [50.0, 60.0, 1000.0]  # all the error is in the dense image
    lines = _diagnose("part_B", pred, truth)
    assert "undercounts" in lines[0]
    assert "1000+" in lines[1] and "100%" in lines[1]


def test_diagnosis_flags_overcounting():
    assert "overcounts" in _diagnose("part_B", [120.0], [100.0])[0]


def test_diagnosis_of_a_perfect_model_does_not_divide_by_zero():
    lines = _diagnose("part_B", [10.0, 20.0], [10.0, 20.0])
    assert len(lines) == 1


def test_part_a_mae_above_120_points_back_to_the_density_maps():
    lines = _diagnose("part_A", [0.0, 0.0], [300.0, 400.0])
    assert any("Phase 1" in line for line in lines)
    assert not any("Phase 1" in line for line in _diagnose("part_B", [0.0], [300.0]))
