"""Dashboard data access.

Loading and aggregation are plain functions so they can be unit-tested and timed
without a browser. The Streamlit layer only caches them and renders.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class DashboardDataError(FileNotFoundError):
    """Raised when the dashboard artifacts have not been generated yet."""


@dataclass(frozen=True)
class DashboardData:
    predictions: pd.DataFrame
    metrics: dict[str, Any]
    folds: pd.DataFrame
    run_meta: dict[str, Any]
    inventory: pd.DataFrame | None

    @property
    def models(self) -> list[str]:
        return sorted(self.predictions["model"].unique().tolist())

    @property
    def series(self) -> list[str]:
        return sorted(self.predictions["series_id"].unique().tolist())

    @property
    def synthetic(self) -> bool:
        return bool(self.metrics.get("dataset", {}).get("synthetic", False))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def load_data(artifacts_dir: Path) -> DashboardData:
    backtest_dir = artifacts_dir / "backtest"
    predictions_path = backtest_dir / "predictions.parquet"
    if not predictions_path.exists():
        raise DashboardDataError(
            f"missing {predictions_path}; run `demand backtest` before the dashboard"
        )
    predictions = pd.read_parquet(predictions_path)
    predictions["date"] = pd.to_datetime(predictions["date"]).dt.normalize()
    metrics = _read_json(backtest_dir / "metrics.json")
    folds = pd.DataFrame(metrics.get("folds", []))
    run_meta: dict[str, Any] = {}
    latest = artifacts_dir / "latest.json"
    if latest.exists():
        pointer = json.loads(latest.read_text(encoding="utf-8"))
        metadata_path = Path(pointer["path"]) / "metadata.json"
        if not metadata_path.exists():
            metadata_path = artifacts_dir / "runs" / str(pointer["run_id"]) / "metadata.json"
        if metadata_path.exists():
            run_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    inventory_path = artifacts_dir / "inventory" / "inventory.parquet"
    inventory = pd.read_parquet(inventory_path) if inventory_path.exists() else None
    return DashboardData(
        predictions=predictions,
        metrics=metrics,
        folds=folds,
        run_meta=run_meta,
        inventory=inventory,
    )


def kpi_table(metrics: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for model, report in dict(metrics.get("models", {})).items():
        overall = dict(report.get("overall", {}))
        rows.append(
            {
                "model": model,
                "wape": overall.get("wape"),
                "mape": overall.get("mape"),
                "bias": overall.get("bias"),
                "mean_rmsse": report.get("mean_rmsse"),
                "fit_seconds": report.get("fit_seconds"),
                "series": report.get("series"),
            }
        )
    return pd.DataFrame(rows).sort_values("wape").reset_index(drop=True) if rows else pd.DataFrame()


def horizon_curve(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.assign(abs_error=(predictions["y"] - predictions["yhat"]).abs())
    grouped = (
        frame.groupby(["model", "horizon"], observed=True)
        .agg(abs_error=("abs_error", "sum"), actual=("y", lambda s: s.abs().sum()))
        .reset_index()
    )
    grouped["wape"] = grouped["abs_error"] / grouped["actual"].replace(0, np.nan)
    return grouped.pivot(index="horizon", columns="model", values="wape").sort_index()


def series_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.assign(
        abs_error=(predictions["y"] - predictions["yhat"]).abs(),
        actual=predictions["y"].abs(),
        bias=(predictions["yhat"] - predictions["y"]),
    )
    grouped = (
        frame.groupby(["model", "series_id"], observed=True)
        .agg(abs_error=("abs_error", "sum"), actual=("actual", "sum"), bias=("bias", "sum"))
        .reset_index()
    )
    grouped["wape"] = grouped["abs_error"] / grouped["actual"].replace(0, np.nan)
    grouped["bias_pct"] = grouped["bias"] / grouped["actual"].replace(0, np.nan)
    return grouped


def worst_series(predictions: pd.DataFrame, model: str, n: int = 10) -> pd.DataFrame:
    summary = series_summary(predictions)
    subset = summary[(summary["model"] == model) & summary["wape"].notna()]
    return subset.sort_values("wape", ascending=False).head(n).reset_index(drop=True)


def series_frame(
    predictions: pd.DataFrame, series_id: str, model: str | None = None
) -> pd.DataFrame:
    subset = predictions[predictions["series_id"] == series_id]
    if model is not None:
        subset = subset[subset["model"] == model]
    return subset.sort_values(["model", "date"]).reset_index(drop=True)


def fold_table(metrics: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for model, report in dict(metrics.get("models", {})).items():
        for fold in report.get("per_fold", []):
            rows.append({"model": model, **fold})
    return pd.DataFrame(rows)


def error_histogram(predictions: pd.DataFrame, model: str, bins: int = 30) -> pd.DataFrame:
    subset = predictions[predictions["model"] == model]
    errors = (subset["y"] - subset["yhat"]).to_numpy(dtype="float64")
    counts, edges = np.histogram(errors, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    return pd.DataFrame({"error": centers, "count": counts})
