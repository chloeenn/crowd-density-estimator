import json

import numpy as np
import pytest

from crowd.routing import fit_threshold, relative_error, routed_counts, use_sparse


def test_sparse_model_takes_over_below_the_threshold():
    assert use_sparse(40, 100) and not use_sparse(100, 100) and not use_sparse(500, 100)
    out = routed_counts([40, 500], [55, 90], 100)
    assert out.tolist() == [55, 500]


def test_fit_picks_the_threshold_that_separates_the_scenes():
    # Sparse scenes (truth 30..120): the dense model overcounts them, the sparse one is close.
    # Dense scenes (truth 400..2000): the dense model is close, the sparse one undercounts.
    truth = [30, 60, 120, 400, 900, 2000]
    dense = [80, 120, 200, 410, 880, 2050]
    sparse = [32, 58, 118, 150, 200, 300]
    fit = fit_threshold(dense, sparse, truth)
    assert 200 < fit["threshold"] <= 410
    assert fit["relative_error"] < fit["relative_error_dense_only"]
    assert fit["relative_error"] < fit["relative_error_sparse_only"]
    assert fit["relative_error"] == pytest.approx(
        relative_error(routed_counts(dense, sparse, fit["threshold"]), truth)
    )


def test_fit_never_picks_a_threshold_above_the_cap():
    truth = [30, 60, 120, 400, 900, 2000]
    dense = [80, 120, 200, 410, 880, 2050]
    sparse = [32, 58, 118, 150, 200, 300]
    free = fit_threshold(dense, sparse, truth)
    capped = fit_threshold(dense, sparse, truth, max_threshold=150)
    assert free["threshold"] > 150 >= capped["threshold"]


def test_predictions_are_cached_and_resumed(tmp_path):
    from helpers import make_samples

    from crowd.routing import collect_predictions

    class Fixed:
        def __init__(self, n):
            self.n, self.calls = n, 0

        def predict(self, image):
            self.calls += 1
            return type("P", (), {"count": self.n})()

    samples = make_samples(tmp_path, 3)
    groups = {("part_A", "val"): samples}
    cache = tmp_path / "c.jsonl"
    dense, sparse = Fixed(100.0), Fixed(50.0)
    rows = collect_predictions(dense, sparse, groups, cache, "k")
    assert [r["dense"] for r in rows] == [100.0] * 3 and dense.calls == 3
    again = collect_predictions(Fixed(1.0), Fixed(1.0), groups, cache, "k")
    assert len(again) == 3 and again[0]["dense"] == 100.0  # reused, not recomputed
    other = collect_predictions(Fixed(1.0), Fixed(1.0), groups, cache, "other-run")
    assert [r["dense"] for r in other] == [1.0] * 3  # a different checkpoint pair recomputes


def test_fit_can_choose_to_never_route():
    truth = [100, 200, 300]
    fit = fit_threshold(truth, [1, 1, 1], truth)  # dense is perfect, sparse is useless
    assert fit["threshold"] == 0 and fit["relative_error"] == 0


def test_relative_error_floors_small_truths():
    assert relative_error([30], [0]) == pytest.approx(30 / 20)
    assert relative_error(np.array([110]), np.array([100])) == pytest.approx(0.1)


def test_shipped_routing_file_is_well_formed_if_present():
    from crowd.routing import ROUTING_PATH

    if ROUTING_PATH.is_file():
        assert json.loads(ROUTING_PATH.read_text())["threshold"] > 0
