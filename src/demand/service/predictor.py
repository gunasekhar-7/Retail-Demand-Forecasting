"""Artifact-backed inference.

Retail forecasts are produced as a nightly batch across every series; the HTTP
service therefore scores the whole horizon once at startup and serves slices.
That keeps p95 latency independent of model complexity and mirrors how demand
planning systems actually run (batch score, low-latency read).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, TypedDict

from demand.registry import RunArtifact, load_latest


class UnknownSeriesError(KeyError):
    """Requested series is not covered by the loaded artifact."""


class HorizonError(ValueError):
    """Requested horizon is outside the trained range."""


class ForecastRow(TypedDict):
    date: date
    yhat: float


class Predictor:
    def __init__(self, artifact: RunArtifact) -> None:
        self.artifact = artifact
        self._scored = artifact.model.forecast(
            artifact.history, artifact.calendar, artifact.cutoff, artifact.horizon
        )
        self._by_series = {
            str(series_id): frame.sort_values("date").reset_index(drop=True)
            for series_id, frame in self._scored.groupby("series_id", observed=True)
        }

    @classmethod
    def load(cls, artifacts_dir: Path) -> Predictor:
        return cls(load_latest(artifacts_dir))

    @property
    def metadata(self) -> dict[str, Any]:
        return self.artifact.metadata

    @property
    def horizon(self) -> int:
        return self.artifact.horizon

    @property
    def series(self) -> list[str]:
        return sorted(self._by_series)

    def forecast(self, series_id: str, horizon_days: int) -> list[ForecastRow]:
        if horizon_days < 1:
            raise HorizonError("horizon_days must be at least 1")
        if horizon_days > self.horizon:
            raise HorizonError(
                f"horizon_days={horizon_days} exceeds the trained horizon {self.horizon}"
            )
        frame = self._by_series.get(series_id)
        if frame is None:
            raise UnknownSeriesError(series_id)
        window = frame.head(horizon_days)
        return [
            ForecastRow(date=row.date, yhat=float(row.yhat))
            for row in window.itertuples(index=False)
        ]

    def forecast_dynamic(
        self,
        series_id: str,
        horizon_days: int,
        price_multiplier: float = 1.0,
        event_override: str | None = None,
    ) -> list[ForecastRow]:
        """Compute counterfactual or scenario-adjusted forecasts on demand."""
        if horizon_days < 1:
            raise HorizonError("horizon_days must be at least 1")
        if horizon_days > self.horizon:
            raise HorizonError(
                f"horizon_days={horizon_days} exceeds the trained horizon {self.horizon}"
            )
        if series_id not in self._by_series:
            raise UnknownSeriesError(series_id)

        if price_multiplier == 1.0 and event_override is None:
            return self.forecast(series_id, horizon_days)

        history = self.artifact.history[self.artifact.history["series_id"] == series_id].copy()
        calendar = self.artifact.calendar.copy()

        if price_multiplier != 1.0 and "sell_price" in history.columns:
            history["sell_price"] = (history["sell_price"] * price_multiplier).astype("float32")

        if event_override is not None and "event_name_1" in calendar.columns:
            calendar["event_name_1"] = event_override

        scored = self.artifact.model.forecast(
            history, calendar, self.artifact.cutoff, horizon_days
        )
        series_frame = (
            scored[scored["series_id"] == series_id].sort_values("date").reset_index(drop=True)
        )
        window = series_frame.head(horizon_days)
        return [
            ForecastRow(date=row.date, yhat=float(row.yhat))
            for row in window.itertuples(index=False)
        ]
