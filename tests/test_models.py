from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from demand.config import Config
from demand.features.calendar import calendar_frame
from demand.features.matrix import FeatureError
from demand.models.base import ModelError
from demand.models.baseline import SeasonalNaiveForecaster
from demand.models.gbm import LgbmForecaster


def test_seasonal_naive_repeats_the_last_week(tiny_panel: pd.DataFrame) -> None:
    model = SeasonalNaiveForecaster(season_length=7)
    cutoff = pd.Timestamp(tiny_panel["date"].max())
    model.fit(tiny_panel, cutoff)
    forecast = model.forecast(tiny_panel, calendar_frame(tiny_panel), cutoff, 7)
    actual = (
        tiny_panel[tiny_panel["series_id"] == "ITEM_1_CA_1"].sort_values("date")["sales"].to_numpy()
    )
    predicted = (
        forecast[forecast["series_id"] == "ITEM_1_CA_1"].sort_values("date")["yhat"].to_numpy()
    )
    np.testing.assert_allclose(predicted, actual[-7:])


def test_lightgbm_forecast_contract(
    demo_config: Config, panel: pd.DataFrame, calendar: pd.DataFrame
) -> None:
    cutoff = pd.Timestamp(panel["date"].max())
    model = LgbmForecaster(demo_config.model, demo_config.features, seed=demo_config.seed)
    model.fit(panel, cutoff)
    forecast = model.forecast(panel, calendar, cutoff, demo_config.features.horizon)
    assert set(forecast.columns) == {"series_id", "date", "yhat"}
    assert forecast["series_id"].nunique() == panel["series_id"].nunique()
    assert (
        forecast.groupby("series_id", observed=True).size() == demo_config.features.horizon
    ).all()
    assert (forecast["yhat"] >= 0).all()
    assert np.isfinite(forecast["yhat"]).all()
    assert model.spec.horizon == demo_config.features.horizon
    assert len(model.spec.columns) == len(model.spec.numeric) + len(model.spec.categorical)


def test_lightgbm_rejects_too_short_history(
    demo_config: Config, panel: pd.DataFrame, calendar: pd.DataFrame
) -> None:
    cutoff = pd.Timestamp(panel["date"].max())
    model = LgbmForecaster(demo_config.model, demo_config.features, seed=demo_config.seed)
    model.fit(panel, cutoff)
    with pytest.raises(FeatureError, match="serving history"):
        model.forecast(panel.tail(10), calendar, cutoff, demo_config.features.horizon)


def test_lightgbm_requires_fit_before_forecast(demo_config: Config) -> None:
    model = LgbmForecaster(demo_config.model, demo_config.features)
    with pytest.raises(ModelError):
        model.forecast(pd.DataFrame(), pd.DataFrame(), pd.Timestamp("2020-01-01"), 7)


def test_prophet_is_exported_lazily() -> None:
    import demand.models as models

    assert models.ProphetForecaster.name == "prophet"
    with pytest.raises(AttributeError):
        _ = models.NoSuchForecaster
