from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest

from demand.config import ProphetConfig
from demand.features.calendar import calendar_frame
from demand.models.base import ModelError
from demand.models.prophet_model import ProphetForecaster


class FakeProphet:
    """Minimal stand-in that exercises the forecaster without Stan."""

    instances: ClassVar[list[FakeProphet]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.country: str | None = None
        self.history: pd.DataFrame | None = None
        FakeProphet.instances.append(self)

    def add_country_holidays(self, country_name: str) -> None:
        self.country = country_name

    def fit(self, frame: pd.DataFrame) -> None:
        self.history = frame

    def make_future_dataframe(self, periods: int, include_history: bool = False) -> pd.DataFrame:
        assert self.history is not None
        start = self.history["ds"].max() + pd.Timedelta(days=1)
        return pd.DataFrame({"ds": pd.date_range(start, periods=periods, freq="D")})

    def predict(self, future: pd.DataFrame) -> pd.DataFrame:
        return future.assign(yhat=np.arange(1, len(future) + 1, dtype="float64"))


@pytest.fixture(autouse=True)
def _clear_fake_instances() -> None:
    FakeProphet.instances.clear()


def test_holidays_are_built_from_both_event_columns() -> None:
    calendar = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "event_name_1": ["New Year", None],
            "event_name_2": [None, "Local"],
        }
    )
    model = ProphetForecaster(ProphetConfig())
    holidays = model._holidays(calendar)

    assert holidays is not None
    assert sorted(holidays["holiday"]) == ["Local", "New Year"]
    assert list(holidays["lower_window"]) == [-1, -1]
    assert list(holidays["upper_window"]) == [1, 1]


def test_holidays_can_be_disabled() -> None:
    calendar = pd.DataFrame({"date": pd.to_datetime(["2020-01-01"]), "event_name_1": ["New Year"]})
    assert ProphetForecaster(ProphetConfig(include_events=False))._holidays(calendar) is None


def test_fit_selects_the_top_revenue_series(
    tiny_panel: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("prophet.Prophet", FakeProphet)
    cutoff = pd.Timestamp(tiny_panel["date"].max())
    model = ProphetForecaster(ProphetConfig(n_series=1), horizon=7)
    model.fit(tiny_panel, cutoff)

    assert len(FakeProphet.instances) == 1
    assert FakeProphet.instances[0].country == "US"
    assert model.metadata()["fitted_series"] == 1
    forecast = model.forecast(tiny_panel, calendar_frame(tiny_panel), cutoff, 7)
    assert set(forecast.columns) == {"series_id", "date", "yhat"}
    assert len(forecast) == 7
    assert (forecast["yhat"] >= 0).all()


def test_fit_rejects_panels_with_too_little_history(
    tiny_panel: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("prophet.Prophet", FakeProphet)
    short = tiny_panel.groupby("series_id", observed=True).tail(10)
    model = ProphetForecaster(ProphetConfig(n_series=1), horizon=7)
    with pytest.raises(ModelError, match="fitted no series"):
        model.fit(short, pd.Timestamp(short["date"].max()))


def test_forecast_requires_fit() -> None:
    model = ProphetForecaster(ProphetConfig())
    with pytest.raises(ModelError, match="not fitted"):
        model.forecast(pd.DataFrame(), pd.DataFrame(), pd.Timestamp("2020-01-01"), 7)


def test_save_writes_inspectable_metadata(
    tiny_panel: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("prophet.Prophet", FakeProphet)
    cutoff = pd.Timestamp(tiny_panel["date"].max())
    model = ProphetForecaster(ProphetConfig(n_series=1), horizon=7)
    model.fit(tiny_panel, cutoff)
    model.save(tmp_path)

    payload = json.loads((tmp_path / "model.json").read_text(encoding="utf-8"))
    assert payload["kind"] == "prophet"
    assert payload["cutoff"] == cutoff.strftime("%Y-%m-%d")
    assert len(payload["fitted_series"]) == 1


def test_refit_drops_stale_models_and_honors_origin(monkeypatch: pytest.MonkeyPatch) -> None:
    dates = pd.date_range("2020-01-01", periods=90, freq="D")
    rows = []
    for series_id, start, end in (("A", 0, 30), ("B", 30, 90)):
        for index, date in enumerate(dates):
            rows.append(
                {
                    "series_id": series_id,
                    "date": date,
                    "sales": 10.0 if start <= index < end else 0.0,
                    "sell_price": 1.0,
                }
            )
    panel = pd.DataFrame(rows)
    monkeypatch.setattr("prophet.Prophet", FakeProphet)
    model = ProphetForecaster(ProphetConfig(n_series=1), horizon=7)

    first_cutoff = dates[49]
    model.fit(panel[panel["date"] <= first_cutoff], first_cutoff)
    first_origin = first_cutoff + pd.Timedelta(days=3)
    first = model.forecast(panel, calendar_frame(panel), first_origin, 7)
    assert set(first["series_id"]) == {"A"}
    assert first["date"].min() == first_origin + pd.Timedelta(days=1)

    second_cutoff = dates[-1]
    model.fit(panel, second_cutoff)
    second = model.forecast(panel, calendar_frame(panel), second_cutoff, 7)
    assert set(second["series_id"]) == {"B"}
    assert second["date"].min() == second_cutoff + pd.Timedelta(days=1)
