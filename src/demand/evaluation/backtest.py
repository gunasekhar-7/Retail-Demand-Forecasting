"""Walk-forward backtesting.

Each fold refits every model on data up to `train_end` and scores the following
`horizon` days. Predictions are written once, and every metric in the report is
derived from that file, so no number can be typed by hand.
"""

from __future__ import annotations

import contextlib
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from demand.evaluation.metrics import (
    MetricError,
    evaluate,
    paired_bootstrap_wape_delta,
    rmsse_per_series,
    hierarchical_rmsse,
)
from demand.evaluation.splits import Fold
from demand.features.calendar import calendar_frame
from demand.models.base import Forecaster, validate_forecast


class BacktestError(RuntimeError):
    """Raised when a fold cannot be scored."""


@dataclass(frozen=True)
class BacktestResult:
    predictions: pd.DataFrame
    metrics: dict[str, Any]
    output_dir: Path

    @property
    def models(self) -> list[str]:
        return sorted(self.predictions["model"].unique().tolist())


def _horizon_metrics(predictions: pd.DataFrame) -> list[dict[str, float]]:
    rows = []
    for horizon, subset in predictions.groupby("horizon", observed=True):
        try:
            metrics = evaluate(subset["y"], subset["yhat"])
        except MetricError:
            continue
        rows.append({"horizon": int(horizon), **metrics})
    return rows


def run_backtest(
    models: list[Forecaster],
    panel: pd.DataFrame,
    folds: list[Fold],
    output_dir: Path,
    bootstrap_reps: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
    synthetic: bool = False,
) -> BacktestResult:
    if not models:
        raise BacktestError("no models to evaluate")
    if not folds:
        raise BacktestError("no folds to evaluate")
    calendar = calendar_frame(panel)
    frames: list[pd.DataFrame] = []
    fit_seconds: dict[str, float] = {}
    predict_seconds: dict[str, float] = {}
    rmsse_by_fold: dict[str, list[float]] = {}

    for fold in folds:
        training = panel[panel["date"] <= fold.train_end]
        actual = panel[(panel["date"] >= fold.test_start) & (panel["date"] <= fold.test_end)][
            ["series_id", "date", "sales"]
        ].rename(columns={"sales": "y"})
        if actual.empty:
            raise BacktestError(f"fold {fold.index} has no actuals in the panel")
        for model in models:
            started = time.perf_counter()
            model.fit(training, fold.train_end)
            fit_seconds[model.name] = (
                fit_seconds.get(model.name, 0.0) + time.perf_counter() - started
            )
            started = time.perf_counter()
            forecast = model.forecast(training, calendar, fold.train_end, fold.horizon)
            predict_seconds[model.name] = (
                predict_seconds.get(model.name, 0.0) + time.perf_counter() - started
            )
            forecast = validate_forecast(forecast, fold.horizon)
            scored = forecast.merge(
                actual, on=["series_id", "date"], how="inner", validate="one_to_one"
            )
            if scored.empty:
                raise BacktestError(f"fold {fold.index} / {model.name} produced no scorable rows")
            scored = scored.assign(
                model=model.name,
                fold=fold.index,
                horizon=(scored["date"] - fold.test_start).dt.days + 1,
            )
            frames.append(scored)
            if model.name != "prophet":
                try:
                    rmsse = rmsse_per_series(scored, training)
                    rmsse_by_fold.setdefault(model.name, []).append(float(rmsse.mean()))
                except (MetricError, ZeroDivisionError):
                    pass

    predictions = pd.concat(frames, ignore_index=True)
    predictions = predictions[["model", "fold", "series_id", "date", "horizon", "y", "yhat"]]

    coverage = {
        name: frozenset(group["series_id"].unique())
        for name, group in predictions.groupby("model", observed=True)
    }
    common = frozenset.intersection(*coverage.values()) if coverage else frozenset()

    model_reports: dict[str, Any] = {}
    baseline = predictions[predictions["model"] == "seasonal_naive"]
    for name in sorted(predictions["model"].unique()):
        subset = predictions[predictions["model"] == name]
        per_fold = []
        for fold_index, fold_rows in subset.groupby("fold", observed=True):
            metrics = evaluate(fold_rows["y"], fold_rows["yhat"])
            per_fold.append({"fold": int(fold_index), **metrics})
        report: dict[str, Any] = {
            "overall": evaluate(subset["y"], subset["yhat"]),
            "overall_common": evaluate(
                subset.loc[subset["series_id"].isin(common), "y"],
                subset.loc[subset["series_id"].isin(common), "yhat"],
            )
            if common
            else None,
            "per_fold": per_fold,
            "per_horizon": _horizon_metrics(subset),
            "fit_seconds": round(fit_seconds.get(name, 0.0), 3),
            "predict_seconds": round(predict_seconds.get(name, 0.0), 3),
            "series": int(subset["series_id"].nunique()),
            "coverage_series": len(coverage[name]),
            "common_series": len(common),
        }
        if name in rmsse_by_fold:
            report["mean_rmsse"] = float(np.mean(rmsse_by_fold[name]))
        # Compute hierarchical RMSSE (store-level -> dept-level -> overall)
        try:
            from demand.evaluation.metrics import hierarchical_rmsse

            report["hierarchical_rmsse"] = hierarchical_rmsse(
                subset,
                training,
            )
        except (MetricError, ZeroDivisionError):
            report["hierarchical_rmsse"] = {
                "level_0_rmsse": float("nan"),
                "level_1_rmsse": float("nan"),
                "level_2_rmsse": float("nan"),
            }
        if not baseline.empty and name != "seasonal_naive" and bootstrap_reps > 0:
            with contextlib.suppress(MetricError):
                report["vs_baseline"] = paired_bootstrap_wape_delta(
                    predictions,
                    baseline="seasonal_naive",
                    candidate=name,
                    reps=bootstrap_reps,
                    confidence=confidence,
                    seed=seed,
                )
        model_reports[name] = report

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output_dir / "predictions.parquet", index=False)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "folds": [fold.to_dict() for fold in folds],
        "dataset": {
            "n_series": int(panel["series_id"].nunique()),
            "n_days": int(panel["date"].nunique()),
            "last_date": pd.Timestamp(panel["date"].max()).strftime("%Y-%m-%d"),
            "synthetic": synthetic,
        },
        "models": model_reports,
    }
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (output_dir / "folds.json").write_text(
        json.dumps([fold.to_dict() for fold in folds], indent=2) + "\n", encoding="utf-8"
    )
    return BacktestResult(predictions=predictions, metrics=payload, output_dir=output_dir)
