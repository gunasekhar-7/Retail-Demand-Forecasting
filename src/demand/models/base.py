"""The single contract every model implements.

`fit` sees data up to `cutoff`; `forecast` sees history up to `origin` plus a
calendar that may extend past it (holidays are known in advance). Backtesting,
serving and tests all call the same two methods.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd

FORECAST_COLUMNS = ("series_id", "date", "yhat")


class ModelError(RuntimeError):
    """Raised when a model is used before fitting or produces an invalid frame."""


@runtime_checkable
class Forecaster(Protocol):
    name: str

    def fit(self, panel: pd.DataFrame, cutoff: pd.Timestamp) -> None: ...

    def forecast(
        self,
        history: pd.DataFrame,
        calendar: pd.DataFrame,
        origin: pd.Timestamp,
        horizon: int,
    ) -> pd.DataFrame: ...

    def save(self, path: Path) -> None: ...

    def metadata(self) -> dict[str, Any]: ...


def validate_forecast(
    frame: pd.DataFrame, horizon: int, expected_series: int | None = None
) -> pd.DataFrame:
    missing = [c for c in FORECAST_COLUMNS if c not in frame.columns]
    if missing:
        raise ModelError(f"forecast missing columns: {missing}")
    result = frame[list(FORECAST_COLUMNS)].copy()
    result["date"] = pd.to_datetime(result["date"]).dt.normalize()
    if result["yhat"].isna().any() or not np.isfinite(result["yhat"]).all():
        raise ModelError("forecast contains NaN or infinite predictions")
    if (result["yhat"] < 0).any():
        raise ModelError("forecast contains negative demand")
    counts = result.groupby("series_id", observed=True)["date"].nunique()
    if (counts != horizon).any():
        raise ModelError(f"every series must have exactly {horizon} forecast days")
    if expected_series is not None and len(counts) != expected_series:
        raise ModelError(f"expected {expected_series} series, got {len(counts)}")
    return result.sort_values(["series_id", "date"], kind="stable").reset_index(drop=True)
