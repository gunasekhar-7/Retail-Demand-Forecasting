"""Training and inference matrices.

Both paths share `date_features` + `add_series_features`, then use the same
cross-join over horizons, so the only difference between a training row and a
serving row is where the target came from. A test asserts parity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from demand.config import FeatureConfig
from demand.features.calendar import date_features
from demand.features.lags import add_series_features
from demand.features.spec import FeatureSpec

STATE_SNAP = {"CA": "snap_CA", "TX": "snap_TX", "WI": "snap_WI"}


class FeatureError(RuntimeError):
    """Raised when a matrix cannot be built without breaking temporal integrity."""


@dataclass(frozen=True)
class TrainingMatrix:
    X: pd.DataFrame
    y: pd.Series
    meta: pd.DataFrame
    spec: FeatureSpec


@dataclass(frozen=True)
class InferenceMatrix:
    X: pd.DataFrame
    meta: pd.DataFrame


def numeric_feature_names(cfg: FeatureConfig) -> list[str]:
    names = [
        "dow",
        "day",
        "dayofyear",
        "month",
        "weekofyear",
        "is_weekend",
        "sin_doy",
        "cos_doy",
        "days_since_last_sale",
    ]
    if cfg.include_events:
        names += ["has_event", "days_to_event", "days_since_event"]
    if cfg.include_snap:
        names += ["snap"]
    if cfg.include_price:
        names += ["sell_price", "price_change_7d", "price_zscore_28"]
    names += [f"sales_lag_{lag}" for lag in cfg.lags]
    for window in cfg.rolling_windows:
        names += [f"sales_mean_{window}", f"sales_std_{window}", f"zero_count_{window}"]
    return names


def categorical_feature_names(cfg: FeatureConfig) -> list[str]:
    names = ["item_id", "dept_id", "cat_id", "store_id"]
    if cfg.include_events:
        names += ["event_type_1"]
    return names


def _calendar_lookup(panel: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    calendar_columns = [
        "date",
        "wday",
        "month",
        "year",
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2",
        "snap_CA",
        "snap_TX",
        "snap_WI",
    ]
    available = [c for c in calendar_columns if c in panel.columns]
    calendar = panel[available].drop_duplicates("date")
    features = date_features(calendar, include_events=cfg.include_events)
    if cfg.include_snap:
        missing = set(STATE_SNAP.values()) - set(features.columns)
        if missing:
            raise FeatureError(f"panel is missing SNAP columns: {sorted(missing)}")
    return features.rename(columns={"date": "target_date"})


def _attach_snap(grid: pd.DataFrame) -> pd.Series:
    values = np.select(
        [
            grid["state_id"].eq("CA").to_numpy(),
            grid["state_id"].eq("TX").to_numpy(),
            grid["state_id"].eq("WI").to_numpy(),
        ],
        [
            grid["snap_CA"].to_numpy(),
            grid["snap_TX"].to_numpy(),
            grid["snap_WI"].to_numpy(),
        ],
        default=0,
    )
    return pd.Series(values, index=grid.index, dtype="int8")


def _horizon_grid(snapshot: pd.DataFrame, horizon: int) -> pd.DataFrame:
    horizons = pd.DataFrame({"horizon": np.arange(1, horizon + 1, dtype="int32")})
    grid = snapshot.merge(horizons, how="cross")
    grid["target_date"] = grid["date"] + pd.to_timedelta(grid["horizon"], unit="D")
    return grid


def _assemble(
    panel: pd.DataFrame,
    cfg: FeatureConfig,
    origins: pd.DatetimeIndex | None,
    horizon: int,
) -> pd.DataFrame:
    series = add_series_features(panel, cfg)
    if origins is not None:
        series = series[series["date"].isin(origins)]
    grid = _horizon_grid(series, horizon)
    calendar = _calendar_lookup(panel, cfg)
    overlap = (set(grid.columns) & set(calendar.columns)) - {"target_date"}
    if overlap:
        grid = grid.drop(columns=sorted(overlap))
    grid = grid.merge(calendar, on="target_date", how="left", validate="many_to_one")
    if cfg.include_snap:
        grid["snap"] = _attach_snap(grid)
    return grid


def training_origins(
    panel: pd.DataFrame, cfg: FeatureConfig, cutoff: pd.Timestamp
) -> pd.DatetimeIndex:
    cutoff = pd.Timestamp(cutoff).normalize()
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    earliest = dates.min() + pd.Timedelta(days=cfg.max_lag_days)
    window_start = max(earliest, cutoff - pd.Timedelta(days=cfg.train_window_days))
    last_origin = cutoff - pd.Timedelta(days=cfg.horizon)
    usable = dates[(dates >= window_start) & (dates <= last_origin)]
    if len(usable) == 0:
        raise FeatureError(
            f"no usable origins before {cutoff.date()}: need {cfg.max_lag_days} days of history "
            f"and a {cfg.horizon}-day horizon"
        )
    # Anchor the stride on the most recent origin so every model sees the freshest data.
    selected = usable[::-1][:: cfg.origin_stride_days][::-1]
    return pd.DatetimeIndex(selected)


def build_training_matrix(
    panel: pd.DataFrame, cfg: FeatureConfig, cutoff: pd.Timestamp
) -> TrainingMatrix:
    cutoff = pd.Timestamp(cutoff).normalize()
    origins = training_origins(panel, cfg, cutoff)
    grid = _assemble(panel, cfg, origins, cfg.horizon)
    if len(grid) > cfg.max_training_rows:
        raise FeatureError(
            f"training matrix would have {len(grid):,} rows (limit {cfg.max_training_rows:,}); "
            f"increase features.origin_stride_days or shrink features.train_window_days"
        )
    targets = panel[["series_id", "date", "sales"]].rename(
        columns={"date": "target_date", "sales": "y"}
    )
    grid = grid.merge(targets, on=["series_id", "target_date"], how="left", validate="many_to_one")
    grid = grid[grid["target_date"] <= cutoff]
    if grid["y"].isna().any():
        raise FeatureError("target merge produced missing values; the panel has date gaps")
    numeric = [*numeric_feature_names(cfg), "horizon"]
    categorical = categorical_feature_names(cfg)
    categories = {
        column: tuple(sorted(grid[column].dropna().astype("string").unique().tolist()))
        for column in categorical
    }
    spec = FeatureSpec(
        columns=tuple(numeric + categorical),
        numeric=tuple(numeric),
        categorical=tuple(categorical),
        categories=categories,
        feature_config=cfg.model_dump(mode="json"),
        max_lag_days=cfg.max_lag_days,
        horizon=cfg.horizon,
    )
    meta = grid[["series_id", "target_date", "horizon", "y"]].reset_index(drop=True)
    return TrainingMatrix(X=spec.encode(grid), y=grid["y"].astype("float32"), meta=meta, spec=spec)


def build_inference_features(
    history: pd.DataFrame,
    calendar: pd.DataFrame,
    spec: FeatureSpec,
    origin: pd.Timestamp,
    horizon: int | None = None,
) -> InferenceMatrix:
    cfg = spec.config
    origin = pd.Timestamp(origin).normalize()
    horizon = horizon or cfg.horizon
    if horizon > cfg.horizon:
        raise FeatureError(f"horizon {horizon} exceeds the trained horizon {cfg.horizon}")
    required_history = cfg.max_lag_days + 1
    days = history["date"].nunique() if "date" in history.columns else 0
    if days < required_history:
        raise FeatureError(f"serving history has {days} days but the spec needs {required_history}")
    series = add_series_features(history, cfg)
    snapshot = series[series["date"] == origin]
    if snapshot.empty:
        raise FeatureError(f"origin {origin.date()} not present in serving history")
    grid = _horizon_grid(snapshot, horizon)
    calendar_features = (
        calendar
        if "dow" in calendar.columns
        else date_features(calendar, include_events=cfg.include_events)
    )
    calendar_features = calendar_features.rename(columns={"date": "target_date"})
    overlap = (set(grid.columns) & set(calendar_features.columns)) - {"target_date"}
    if overlap:
        grid = grid.drop(columns=sorted(overlap))
    grid = grid.merge(calendar_features, on="target_date", how="left", validate="many_to_one")
    if grid["dow"].isna().any():
        raise FeatureError("calendar does not cover the requested forecast horizon")
    if cfg.include_snap:
        grid["snap"] = _attach_snap(grid)
    meta = (
        grid[["series_id", "target_date", "horizon"]]
        .rename(columns={"target_date": "date"})
        .reset_index(drop=True)
    )
    return InferenceMatrix(X=spec.encode(grid), meta=meta)
