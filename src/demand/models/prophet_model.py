"""Prophet challenger, deliberately capped to the top-revenue series.

Prophet fits one model per series and needs a Stan backend, so scoring 30,490
series is not interactive. Benchmarking it on the highest-value series answers
"would a classical decomposition model win here?" without pretending it is a
serving candidate.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from demand.config import ProphetConfig
from demand.models.base import ModelError, validate_forecast


class ProphetForecaster:
    name = "prophet"

    def __init__(self, cfg: ProphetConfig | None = None, horizon: int = 28) -> None:
        self.cfg = cfg or ProphetConfig()
        self.horizon = horizon
        self._cutoff: pd.Timestamp | None = None
        self._series: list[str] = []
        self._models: dict[str, Any] = {}
        self._fitted = False

    def _holidays(self, calendar: pd.DataFrame) -> pd.DataFrame | None:
        if not self.cfg.include_events or "event_name_1" not in calendar.columns:
            return None
        frames = []
        for name_column in ("event_name_1", "event_name_2"):
            if name_column not in calendar.columns:
                continue
            subset = calendar.loc[calendar[name_column].notna(), ["date", name_column]].copy()
            subset = subset.rename(columns={"date": "ds", name_column: "holiday"})
            frames.append(subset)
        if not frames:
            return None
        holidays = pd.concat(frames, ignore_index=True).dropna()
        holidays["ds"] = pd.to_datetime(holidays["ds"])
        holidays = holidays.drop_duplicates(["ds", "holiday"]).reset_index(drop=True)
        holidays["lower_window"] = -1
        holidays["upper_window"] = 1
        return holidays if not holidays.empty else None

    def fit(self, panel: pd.DataFrame, cutoff: pd.Timestamp) -> None:
        from prophet import Prophet

        cutoff = pd.Timestamp(cutoff).normalize()
        self._cutoff = None
        self._series = []
        self._models = {}
        self._fitted = False
        history = panel[panel["date"] <= cutoff]
        revenue = (
            history.assign(
                revenue=history["sales"].astype("float64") * history["sell_price"].fillna(0.0)
            )
            .groupby("series_id", observed=True)["revenue"]
            .sum()
        )
        self._series = revenue.nlargest(self.cfg.n_series).index.tolist()
        holidays = self._holidays(panel)
        logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
        for series_id in self._series:
            frame = (
                history.loc[history["series_id"] == series_id, ["date", "sales"]]
                .rename(columns={"date": "ds", "sales": "y"})
                .sort_values("ds")
            )
            if len(frame) < 2 * self.horizon or frame["y"].sum() <= 0:
                continue
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                seasonality_mode=self.cfg.seasonality_mode,
                holidays=holidays,
                interval_width=0.8,
            )
            if self.cfg.country:
                model.add_country_holidays(country_name=self.cfg.country)
            model.fit(frame)
            self._models[series_id] = model
        if not self._models:
            raise ModelError("Prophet fitted no series (insufficient history)")
        self._cutoff = cutoff
        self._fitted = True

    def forecast(
        self,
        history: pd.DataFrame,
        calendar: pd.DataFrame,
        origin: pd.Timestamp,
        horizon: int,
    ) -> pd.DataFrame:
        if not self._fitted:
            raise ModelError("Prophet model is not fitted")
        origin = pd.Timestamp(origin).normalize()
        rows = []
        for series_id, model in self._models.items():
            future = pd.DataFrame(
                {"ds": pd.date_range(origin + pd.Timedelta(days=1), periods=horizon, freq="D")}
            )
            predicted = model.predict(future)
            rows.append(
                pd.DataFrame(
                    {
                        "series_id": series_id,
                        "date": predicted["ds"].to_numpy(),
                        "yhat": np.clip(predicted["yhat"].to_numpy(dtype="float64"), 0.0, None),
                    }
                )
            )
        frame = pd.concat(rows, ignore_index=True)
        return validate_forecast(frame, horizon, expected_series=len(self._models))

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        payload = {
            "kind": self.name,
            "series": self._series,
            "fitted_series": sorted(self._models),
            "cutoff": self._cutoff.strftime("%Y-%m-%d") if self._cutoff else None,
            "config": self.cfg.model_dump(mode="json"),
        }
        (path / "model.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def metadata(self) -> dict[str, Any]:
        return {
            "kind": self.name,
            "requested_series": self.cfg.n_series,
            "fitted_series": len(self._models),
            "series": self._series if len(self._series) <= 50 else self._series[:50],
            "cutoff": self._cutoff.strftime("%Y-%m-%d") if self._cutoff else None,
        }
