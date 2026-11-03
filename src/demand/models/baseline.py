"""Seasonal-naive baseline: next week looks like last week.

It needs no training, cannot leak, and is the number the README compares against.
If a gradient-boosted model cannot beat this, the feature set is not earning its
complexity.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from demand.models.base import ModelError, validate_forecast


class SeasonalNaiveForecaster:
    name = "seasonal_naive"

    def __init__(self, season_length: int = 7) -> None:
        if season_length < 1:
            raise ValueError("season_length must be positive")
        self.season_length = season_length
        self._cutoff: pd.Timestamp | None = None
        self._series: list[str] = []

    def fit(self, panel: pd.DataFrame, cutoff: pd.Timestamp) -> None:
        cutoff = pd.Timestamp(cutoff).normalize()
        self._cutoff = cutoff
        self._series = sorted(panel.loc[panel["date"] <= cutoff, "series_id"].unique().tolist())

    def forecast(
        self,
        history: pd.DataFrame,
        calendar: pd.DataFrame,
        origin: pd.Timestamp,
        horizon: int,
    ) -> pd.DataFrame:
        origin = pd.Timestamp(origin).normalize()
        season = self.season_length
        window_start = origin - pd.Timedelta(days=season - 1)
        tail = history[(history["date"] >= window_start) & (history["date"] <= origin)]
        if tail.empty:
            raise ModelError("seasonal naive needs at least one full season of history")
        pivot = tail.pivot_table(index="series_id", columns="date", values="sales", aggfunc="last")
        dates = pd.DatetimeIndex(sorted(pivot.columns))
        if len(dates) < season:
            raise ModelError(f"seasonal naive needs {season} days of history")

        rows: list[pd.DataFrame] = []
        for step in range(1, horizon + 1):
            offset = ((step - 1) % season) - (season - 1)
            source = origin + pd.Timedelta(days=offset)
            if source not in pivot.columns:
                raise ModelError(f"history is missing {source.date()} needed for horizon {step}")
            values = pivot[source].to_numpy(dtype="float64")
            rows.append(
                pd.DataFrame(
                    {
                        "series_id": pivot.index.to_numpy(),
                        "date": origin + pd.Timedelta(days=step),
                        "yhat": np.clip(values, 0.0, None),
                    }
                )
            )
        frame = pd.concat(rows, ignore_index=True)
        return validate_forecast(frame, horizon, expected_series=len(self._series))

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        payload = {"kind": "seasonal_naive", "season_length": self.season_length}
        (path / "model.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")

    def metadata(self) -> dict[str, Any]:
        return {
            "kind": "seasonal_naive",
            "season_length": self.season_length,
            "series": len(self._series),
            "cutoff": self._cutoff.strftime("%Y-%m-%d") if self._cutoff is not None else None,
        }

    @staticmethod
    def required_history_days(season_length: int = 7) -> int:
        return math.ceil(season_length)
