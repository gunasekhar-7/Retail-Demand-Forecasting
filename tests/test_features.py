from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from demand.config import FeatureConfig
from demand.features.calendar import calendar_frame
from demand.features.lags import add_series_features
from demand.features.matrix import (
    FeatureError,
    build_inference_features,
    build_training_matrix,
    numeric_feature_names,
)
from demand.features.spec import FeatureSpec, FeatureSpecError
from demand.registry import ensure_serving_history


def _cfg(**overrides: object) -> FeatureConfig:
    payload = {
        "horizon": 7,
        "lags": [7, 14, 21],
        "rolling_windows": [7],
        "origin_stride_days": 7,
        "train_window_days": 240,
    }
    payload.update(overrides)
    return FeatureConfig.model_validate(payload)


def test_lags_ignore_rows_after_the_origin(tiny_panel: pd.DataFrame) -> None:
    cfg = _cfg()
    cutoff = pd.Timestamp("2020-01-20")
    baseline = add_series_features(tiny_panel, cfg)
    mutated = tiny_panel.copy()
    mutated.loc[mutated["date"] > cutoff, "sales"] = 9999
    changed = add_series_features(mutated, cfg)
    left = baseline[(baseline["series_id"] == "ITEM_1_CA_1") & (baseline["date"] == cutoff)]
    right = changed[(changed["series_id"] == "ITEM_1_CA_1") & (changed["date"] == cutoff)]
    pd.testing.assert_frame_equal(left.reset_index(drop=True), right.reset_index(drop=True))


def test_rolling_windows_need_a_full_window(tiny_panel: pd.DataFrame) -> None:
    features = add_series_features(tiny_panel, _cfg())
    series = features[features["series_id"] == "ITEM_1_CA_1"].reset_index(drop=True)
    assert series["sales_mean_7"].iloc[:7].isna().all()
    assert not np.isnan(series["sales_mean_7"].iloc[7])


def test_future_rows_do_not_change_a_lag(tiny_panel: pd.DataFrame) -> None:
    features = add_series_features(tiny_panel, _cfg())
    series = features[features["series_id"] == "ITEM_1_CA_1"].reset_index(drop=True)
    origin_index = 30
    actual = tiny_panel[tiny_panel["series_id"] == "ITEM_1_CA_1"]["sales"].to_numpy()
    assert series["sales_lag_7"].iloc[origin_index] == pytest.approx(actual[origin_index - 7])


def test_training_and_serving_features_are_identical(tiny_panel: pd.DataFrame) -> None:
    cfg = _cfg()
    origin = pd.Timestamp("2020-02-05")
    cutoff = origin + pd.Timedelta(days=cfg.horizon)
    matrix = build_training_matrix(tiny_panel, cfg, cutoff)
    history = tiny_panel[tiny_panel["date"] <= origin]
    inference = build_inference_features(
        history, calendar_frame(tiny_panel), matrix.spec, origin, cfg.horizon
    )
    merged = matrix.X.assign(target_date=matrix.meta["target_date"].to_numpy())
    selected = merged[merged["target_date"] == origin + pd.Timedelta(days=3)].drop(
        columns=["target_date"]
    )
    expected = inference.X[inference.meta["horizon"] == 3].reset_index(drop=True)
    np.testing.assert_allclose(
        selected.to_numpy(dtype="float64"),
        expected.to_numpy(dtype="float64"),
        rtol=1e-6,
        atol=1e-6,
        equal_nan=True,
    )


def test_stale_sale_feature_parity_uses_bounded_serving_history() -> None:
    """A sale outside the serving tail must not change the feature value."""
    dates = pd.date_range("2018-01-01", periods=800, freq="D")
    panel = pd.DataFrame(
        {
            "series_id": "ITEM_STALE_CA_1",
            "item_id": "ITEM_STALE",
            "dept_id": "FOODS_3",
            "cat_id": "FOODS",
            "store_id": "CA_1",
            "state_id": "CA",
            "date": dates,
            "sales": [1.0] + [0.0] * (len(dates) - 1),
            "sell_price": 1.0,
            "snap": 0,
            "snap_CA": 0,
            "snap_TX": 0,
            "snap_WI": 0,
            "wday": dates.day_of_week + 1,
            "weekday": dates.day_name(),
            "month": dates.month,
            "year": dates.year,
            "d_index": np.arange(1, len(dates) + 1),
        }
    )
    cfg = _cfg(lags=[7], rolling_windows=[7], train_window_days=1095)
    origin = dates[-30]
    cutoff = origin + pd.Timedelta(days=cfg.horizon)
    panel = panel[panel["date"] <= cutoff]
    matrix = build_training_matrix(panel, cfg, cutoff)

    # Use a short serving window: the implementation must extend it enough for
    # the bounded days-since-last-sale feature, not just the longest lag.
    history = ensure_serving_history(panel, days=cfg.max_lag_days + cfg.horizon + 5)
    inference = build_inference_features(
        history, calendar_frame(panel), matrix.spec, origin, cfg.horizon
    )

    horizon = 3
    target_date = origin + pd.Timedelta(days=horizon)
    training_rows = np.flatnonzero(
        (matrix.meta["target_date"] == target_date).to_numpy()
        & (matrix.meta["horizon"] == horizon).to_numpy()
    )
    inference_rows = np.flatnonzero((inference.meta["horizon"] == horizon).to_numpy())
    assert len(training_rows) == 1
    assert len(inference_rows) == 1

    training_row = matrix.X.iloc[training_rows[0]]
    inference_row = inference.X.iloc[inference_rows[0]]
    assert training_row["days_since_last_sale"] == pytest.approx(365.0)
    assert training_row["days_since_last_sale"] == inference_row["days_since_last_sale"]
    np.testing.assert_allclose(
        training_row.to_numpy(dtype="float64"),
        inference_row.to_numpy(dtype="float64"),
        rtol=1e-6,
        atol=1e-6,
        equal_nan=True,
    )


def test_spec_round_trip_and_unknown_category(tiny_panel: pd.DataFrame, tmp_path: Path) -> None:
    matrix = build_training_matrix(tiny_panel, _cfg(), pd.Timestamp("2020-02-10"))
    path = tmp_path / "spec.json"
    matrix.spec.save(path)
    loaded = FeatureSpec.load(path)
    assert loaded.columns == matrix.spec.columns
    assert loaded.categories == matrix.spec.categories
    frame = pd.DataFrame({column: ["UNSEEN"] for column in loaded.categories})
    for column in loaded.columns:
        if column not in frame:
            frame[column] = 0.0
    encoded = loaded.encode(frame)
    assert (encoded[list(loaded.categories)] == -1).all().all()


def test_spec_rejects_missing_columns(tiny_panel: pd.DataFrame) -> None:
    matrix = build_training_matrix(tiny_panel, _cfg(), pd.Timestamp("2020-02-10"))
    broken = pd.DataFrame({c: [0.0] for c in matrix.spec.columns[:-1]})
    with pytest.raises(FeatureSpecError, match="missing columns"):
        matrix.spec.encode(broken)


def test_matrix_respects_row_guard(tiny_panel: pd.DataFrame) -> None:
    cfg = _cfg(max_training_rows=10)
    with pytest.raises(FeatureError, match="training matrix would have"):
        build_training_matrix(tiny_panel, cfg, pd.Timestamp("2020-02-10"))


def test_numeric_features_follow_switches() -> None:
    with_events = numeric_feature_names(_cfg(include_events=True))
    without_events = numeric_feature_names(_cfg(include_events=False))
    assert "has_event" in with_events
    assert "has_event" not in without_events
