from __future__ import annotations

import pandas as pd
import pytest

from demand.evaluation.metrics import (
    MetricError,
    bias,
    evaluate,
    mape,
    paired_bootstrap_wape_delta,
    rmsse_per_series,
    wape,
)


def test_wape_hand_calculated() -> None:
    assert wape([10, 0, 5], [8, 1, 5]) == pytest.approx(3 / 15)


def test_mape_ignores_zero_actuals() -> None:
    # errors on non-zero days: 2/10 and 0/5
    assert mape([10, 0, 5], [8, 1, 5]) == pytest.approx(0.1)


def test_bias_sign_and_magnitude() -> None:
    assert bias([10, 0, 5], [8, 1, 5]) == pytest.approx(-1 / 15)


def test_zero_denominator_raises() -> None:
    with pytest.raises(MetricError, match="undefined"):
        wape([0, 0], [1, 1])
    with pytest.raises(MetricError, match="undefined"):
        mape([0, 0], [1, 1])


def test_evaluate_reports_mape_coverage() -> None:
    metrics = evaluate([10, 0, 5], [8, 1, 5])
    assert metrics["mape_nonzero_n"] == 2
    assert metrics["zero_actual_share"] == pytest.approx(1 / 3)
    assert metrics["n"] == 3


def test_rmsse_matches_hand_calculation() -> None:
    training = pd.DataFrame(
        {
            "series_id": ["A"] * 4,
            "date": pd.date_range("2020-01-01", periods=4),
            "sales": [0, 1, 2, 3],
        }
    )
    predictions = pd.DataFrame({"series_id": ["A", "A"], "y": [4, 5], "yhat": [3, 3]})
    rmsse = rmsse_per_series(predictions, training, min_history=2)
    assert rmsse["A"] == pytest.approx(2.5**0.5)


def test_bootstrap_interval_contains_delta() -> None:
    rows = []
    for index in range(10):
        actual = 10 + index
        rows.append({"model": "base", "series_id": f"s{index}", "y": actual, "yhat": actual + 3})
        rows.append({"model": "cand", "series_id": f"s{index}", "y": actual, "yhat": actual + 1})
    result = paired_bootstrap_wape_delta(
        pd.DataFrame(rows), baseline="base", candidate="cand", reps=200, seed=7
    )
    assert result["delta"] < 0
    assert result["ci_low"] <= result["delta"] <= result["ci_high"]
    assert result["n_series"] == 10


def test_bootstrap_with_zero_reps_returns_point_estimate() -> None:
    frame = pd.DataFrame(
        {
            "model": ["base", "base", "cand", "cand"],
            "series_id": ["a", "b", "a", "b"],
            "y": [10, 20, 10, 20],
            "yhat": [11, 22, 10, 20],
        }
    )
    result = paired_bootstrap_wape_delta(frame, "base", "cand", reps=0)
    assert result["delta"] == pytest.approx(-3 / 30)
    assert result["ci_low"] == result["ci_high"] == result["delta"]
