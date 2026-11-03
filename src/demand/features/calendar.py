"""Calendar features for the *target* date.

Holidays, weekends and SNAP days are known arbitrarily far in advance, so a
direct multi-horizon model may look at the calendar of the day it is predicting.
Lags and rolling statistics stay as-of-origin (see `lags.py`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EVENT_COLUMNS = ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]
SNAP_COLUMNS = ["snap_CA", "snap_TX", "snap_WI"]


def calendar_frame(panel: pd.DataFrame) -> pd.DataFrame:
    """Unique calendar rows from a panel, including the per-state SNAP columns."""
    columns = [
        "date",
        "wday",
        "month",
        "year",
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2",
        *SNAP_COLUMNS,
    ]
    available = [c for c in columns if c in panel.columns]
    return panel[available].drop_duplicates("date").sort_values("date").reset_index(drop=True)


def _distance_to_events(
    day_numbers: np.ndarray, event_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    event_days = day_numbers[event_mask]
    if event_days.size == 0:
        return (
            np.full(day_numbers.shape, 999, dtype="int32"),
            np.full(day_numbers.shape, 999, dtype="int32"),
        )
    upper = np.searchsorted(event_days, day_numbers)
    lower = upper - 1
    next_event = np.where(
        upper < event_days.size,
        event_days[np.clip(upper, 0, event_days.size - 1)] - day_numbers,
        999,
    )
    previous_event = np.where(
        lower >= 0,
        day_numbers - event_days[np.clip(lower, 0, event_days.size - 1)],
        999,
    )
    return np.minimum(next_event, 999).astype("int32"), np.minimum(previous_event, 999).astype(
        "int32"
    )


def date_features(calendar: pd.DataFrame, include_events: bool = True) -> pd.DataFrame:
    """One row per unique date with deterministic, future-known features."""
    required = {"date", "wday", "month", "year"}
    missing = required - set(calendar.columns)
    if missing:
        raise ValueError(f"calendar missing columns: {sorted(missing)}")
    frame = calendar.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame = frame.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)

    dates = frame["date"]
    day_of_year = dates.dt.day_of_year.to_numpy()
    features = pd.DataFrame(
        {
            "date": dates,
            "dow": frame["wday"].astype("int8"),
            "day": dates.dt.day.astype("int8"),
            "dayofyear": day_of_year.astype("int16"),
            "month": frame["month"].astype("int8"),
            "weekofyear": dates.dt.isocalendar().week.to_numpy().astype("int16"),
            "is_weekend": (frame["wday"].to_numpy() >= 6).astype("int8"),
            "sin_doy": np.sin(2 * np.pi * day_of_year / 365.25).astype("float32"),
            "cos_doy": np.cos(2 * np.pi * day_of_year / 365.25).astype("float32"),
        }
    )
    if include_events:
        event_name_1 = frame.get("event_name_1", pd.Series(index=frame.index, dtype="object"))
        event_name_2 = frame.get("event_name_2", pd.Series(index=frame.index, dtype="object"))
        has_event = (event_name_1.notna() | event_name_2.notna()).to_numpy()
        day_numbers = (dates - dates.min()).dt.days.to_numpy()
        days_to_event, days_since_event = _distance_to_events(day_numbers, has_event)
        features["has_event"] = has_event.astype("int8")
        features["days_to_event"] = days_to_event
        features["days_since_event"] = days_since_event
        features["event_type_1"] = frame.get(
            "event_type_1", pd.Series(index=frame.index, dtype="object")
        )
        features["event_type_2"] = frame.get(
            "event_type_2", pd.Series(index=frame.index, dtype="object")
        )
    for column in SNAP_COLUMNS:
        features[column] = (
            frame[column].fillna(0).astype("int8") if column in frame.columns else np.int8(0)
        )
    return features
