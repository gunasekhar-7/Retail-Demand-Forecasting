"""Inventory policy simulation.

Forecast error is only interesting to a business if it changes stockouts and
on-hand inventory. This module simulates a daily order-up-to policy with a
configurable lead time and safety buffer, and reports fill rate, stockout days,
average inventory and cost. Every cost parameter is an explicit assumption in
config, and the report includes a sensitivity sweep.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd

from demand.config import InventoryConfig


class InventoryError(ValueError):
    """Raised when the simulation inputs cannot describe a policy."""


@dataclass(frozen=True)
class InventoryResult:
    per_series: pd.DataFrame
    summary: dict[str, float]

    def to_dict(self) -> dict[str, float]:
        return self.summary


def z_for_service_level(service_level: float) -> float:
    if not 0.5 < service_level < 1:
        raise InventoryError("service_level must be between 0.5 and 1")
    return NormalDist().inv_cdf(service_level)


def simulate_order_up_to(
    actual: pd.DataFrame,
    forecast: pd.DataFrame,
    cfg: InventoryConfig,
    unit_price: pd.Series,
    residual_sigma: pd.Series | None = None,
    days_per_year: int = 365,
) -> InventoryResult:
    required_actual = {"series_id", "date", "y"}
    required_forecast = {"series_id", "date", "yhat"}
    if not required_actual.issubset(actual.columns):
        raise InventoryError(f"actual needs {sorted(required_actual)}")
    if not required_forecast.issubset(forecast.columns):
        raise InventoryError(f"forecast needs {sorted(required_forecast)}")
    if actual.empty:
        raise InventoryError("no actuals to simulate")

    z_value = z_for_service_level(cfg.service_level)
    records: list[dict[str, object]] = []

    for series_id, actual_group in actual.groupby("series_id", observed=True):
        series_forecast = forecast[forecast["series_id"] == series_id]
        if series_forecast.empty:
            continue
        demand = (
            actual_group.assign(date=pd.to_datetime(actual_group["date"]))
            .set_index("date")["y"]
            .sort_index()
        )
        prediction = (
            series_forecast.assign(date=pd.to_datetime(series_forecast["date"]))
            .groupby("date")["yhat"]
            .mean()
            .sort_index()
        )
        days = demand.index
        if len(days) == 0:
            continue
        sigma = float(residual_sigma.get(series_id, 0.0)) if residual_sigma is not None else 0.0
        safety = z_value * sigma * np.sqrt(cfg.lead_time_days + 1)
        on_hand = float(prediction.iloc[: cfg.lead_time_days].sum() + safety)
        pipeline: dict[int, float] = {}
        for offset, day in enumerate(days):
            received = pipeline.pop(offset, 0.0)
            on_hand += received
            requested = float(max(demand.loc[day], 0.0))
            sold = min(requested, on_hand)
            lost = requested - sold
            on_hand -= sold
            position = on_hand + sum(pipeline.values())
            lookahead = prediction.iloc[offset : offset + cfg.lead_time_days].sum()
            order_up_to = float(lookahead + safety)
            order = max(0.0, order_up_to - position)
            if order > 0:
                pipeline[offset + cfg.lead_time_days] = (
                    pipeline.get(offset + cfg.lead_time_days, 0.0) + order
                )
            records.append(
                {
                    "series_id": series_id,
                    "date": day,
                    "demand": requested,
                    "sold": sold,
                    "lost_units": lost,
                    "on_hand": on_hand,
                    "order_units": order,
                }
            )

    per_series = pd.DataFrame.from_records(records)
    if per_series.empty:
        raise InventoryError("simulation produced no rows")
    per_series["stockout_day"] = (per_series["lost_units"] > 0).astype(int)
    prices = unit_price.reindex(per_series["series_id"]).to_numpy(dtype="float64")
    per_series["holding_cost"] = (
        per_series["on_hand"].to_numpy(dtype="float64")
        * prices
        * (1.0 - cfg.unit_margin_pct)
        * cfg.annual_holding_rate
        / days_per_year
    )
    per_series["lost_margin"] = (
        per_series["lost_units"].to_numpy(dtype="float64") * prices * cfg.unit_margin_pct
    )
    total_demand = float(per_series["demand"].sum())
    total_sold = float(per_series["sold"].sum())
    summary = {
        "days": float(per_series["date"].nunique()),
        "series": float(per_series["series_id"].nunique()),
        "demand_units": total_demand,
        "sold_units": total_sold,
        "lost_units": float(per_series["lost_units"].sum()),
        "fill_rate": total_sold / total_demand if total_demand > 0 else float("nan"),
        "stockout_day_share": float(
            per_series.groupby("series_id", observed=True)["stockout_day"].mean().mean()
        ),
        "avg_on_hand_units": float(per_series["on_hand"].mean()),
        "ordered_units": float(per_series["order_units"].sum()),
        "holding_cost": float(per_series["holding_cost"].sum()),
        "lost_margin_cost": float(per_series["lost_margin"].sum()),
        "total_cost": float(per_series["holding_cost"].sum() + per_series["lost_margin"].sum()),
    }
    return InventoryResult(per_series=per_series, summary=summary)


def compare_policies(
    actual: pd.DataFrame,
    forecasts: dict[str, pd.DataFrame],
    cfg: InventoryConfig,
    unit_price: pd.Series,
    residual_sigma: pd.Series | None = None,
) -> pd.DataFrame:
    rows = []
    for name, forecast in forecasts.items():
        result = simulate_order_up_to(actual, forecast, cfg, unit_price, residual_sigma)
        rows.append({"model": name, **result.summary})
    return pd.DataFrame(rows)
