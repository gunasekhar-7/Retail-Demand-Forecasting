"""Forecasters sharing one fit/forecast contract."""

from typing import TYPE_CHECKING, Any

from demand.models.base import FORECAST_COLUMNS, Forecaster, ModelError, validate_forecast
from demand.models.baseline import SeasonalNaiveForecaster
from demand.models.gbm import LgbmForecaster

if TYPE_CHECKING:
    from demand.models.prophet_model import ProphetForecaster

__all__ = [
    "FORECAST_COLUMNS",
    "Forecaster",
    "LgbmForecaster",
    "ModelError",
    "ProphetForecaster",
    "SeasonalNaiveForecaster",
    "validate_forecast",
]


def __getattr__(name: str) -> Any:
    """Keep the training-only Prophet forecaster out of the serving import path."""
    if name == "ProphetForecaster":
        from demand.models.prophet_model import ProphetForecaster

        return ProphetForecaster
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
