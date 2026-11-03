from __future__ import annotations

import pandas as pd
import pytest

from demand.config import InventoryConfig
from demand.evaluation.inventory import (
    InventoryError,
    simulate_order_up_to,
    z_for_service_level,
)


def _frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    dates = pd.date_range("2024-01-01", periods=3)
    actual = pd.DataFrame({"series_id": ["A"] * 3, "date": dates, "y": [10.0, 20.0, 5.0]})
    forecast = pd.DataFrame({"series_id": ["A"] * 3, "date": dates, "yhat": [10.0, 10.0, 10.0]})
    return actual, forecast, pd.Series({"A": 2.0})


def test_order_up_to_hand_calculated() -> None:
    actual, forecast, price = _frames()
    cfg = InventoryConfig(lead_time_days=1, service_level=0.9)
    result = simulate_order_up_to(actual, forecast, cfg, price)
    summary = result.summary
    assert summary["demand_units"] == pytest.approx(35.0)
    assert summary["sold_units"] == pytest.approx(25.0)
    assert summary["lost_units"] == pytest.approx(10.0)
    assert summary["fill_rate"] == pytest.approx(25 / 35)
    assert summary["avg_on_hand_units"] == pytest.approx(5 / 3)
    assert summary["stockout_day_share"] == pytest.approx(1 / 3)


def test_safety_stock_increases_orders() -> None:
    actual, forecast, price = _frames()
    cfg = InventoryConfig(lead_time_days=1, service_level=0.9)
    without_buffer = simulate_order_up_to(actual, forecast, cfg, price)
    with_buffer = simulate_order_up_to(
        actual, forecast, cfg, price, residual_sigma=pd.Series({"A": 5.0})
    )
    assert with_buffer.summary["ordered_units"] > without_buffer.summary["ordered_units"]


def test_missing_forecast_is_rejected() -> None:
    actual, _, price = _frames()
    with pytest.raises(InventoryError):
        simulate_order_up_to(
            actual, actual.drop(columns=["y"]).assign(wrong=0.0), InventoryConfig(), price
        )


def test_service_level_bounds() -> None:
    assert z_for_service_level(0.9) == pytest.approx(1.2816, abs=1e-3)
    with pytest.raises(InventoryError):
        z_for_service_level(1.0)
