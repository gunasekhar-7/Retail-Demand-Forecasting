"""Forecast accuracy metrics.

WAPE is primary because retail demand is zero-inflated and volume-weighted; MAPE
is reported only over non-zero actuals and the excluded count travels with it so
a small denominator cannot quietly inflate the headline number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class MetricError(ValueError):
    """Raised when a metric is undefined for the supplied series."""


def _arrays(y_true: object, y_pred: object) -> tuple[np.ndarray, np.ndarray]:
    actual = np.asarray(y_true, dtype="float64").reshape(-1)
    predicted = np.asarray(y_pred, dtype="float64").reshape(-1)
    if actual.shape != predicted.shape:
        raise MetricError(f"shape mismatch: {actual.shape} vs {predicted.shape}")
    if actual.size == 0:
        raise MetricError("no observations")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise MetricError("metrics require finite actuals and predictions")
    return actual, predicted


def wape(y_true: object, y_pred: object) -> float:
    actual, predicted = _arrays(y_true, y_pred)
    denominator = np.abs(actual).sum()
    if denominator <= 0:
        raise MetricError("WAPE is undefined when total actual demand is zero")
    return float(np.abs(actual - predicted).sum() / denominator)


def mape(y_true: object, y_pred: object) -> float:
    actual, predicted = _arrays(y_true, y_pred)
    mask = actual != 0
    if not mask.any():
        raise MetricError("MAPE is undefined when every actual is zero")
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])))


def mae(y_true: object, y_pred: object) -> float:
    actual, predicted = _arrays(y_true, y_pred)
    return float(np.mean(np.abs(actual - predicted)))


def rmse(y_true: object, y_pred: object) -> float:
    actual, predicted = _arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def bias(y_true: object, y_pred: object) -> float:
    actual, predicted = _arrays(y_true, y_pred)
    denominator = np.abs(actual).sum()
    if denominator <= 0:
        raise MetricError("bias is undefined when total actual demand is zero")
    return float((predicted - actual).sum() / denominator)


def evaluate(y_true: object, y_pred: object) -> dict[str, float]:
    actual, _ = _arrays(y_true, y_pred)
    metrics = {
        "n": float(actual.size),
        "wape": wape(actual, y_pred),
        "mae": mae(actual, y_pred),
        "rmse": rmse(actual, y_pred),
        "bias": bias(actual, y_pred),
        "zero_actual_share": float(np.mean(actual == 0)),
    }
    if (actual != 0).any():
        metrics["mape"] = mape(actual, y_pred)
        metrics["mape_nonzero_n"] = float(np.count_nonzero(actual))
    else:
        metrics["mape"] = float("nan")
        metrics["mape_nonzero_n"] = 0.0
    return metrics


def rmsse_per_series(
    predictions: pd.DataFrame, training: pd.DataFrame, min_history: int = 14
) -> pd.Series:
    """RMSSE per series against the in-sample one-step seasonal-naive scale.

    `predictions` needs columns [series_id, y, yhat]; `training` needs
    [series_id, date, sales]. The denominator is the mean squared first
    difference of the training series (M5 uses the same scaling idea).
    """
    required_predictions = {"series_id", "y", "yhat"}
    required_training = {"series_id", "date", "sales"}
    if not required_predictions.issubset(predictions.columns):
        raise MetricError(f"predictions need columns {sorted(required_predictions)}")
    if not required_training.issubset(training.columns):
        raise MetricError(f"training needs columns {sorted(required_training)}")

    ordered = training.sort_values(["series_id", "date"], kind="stable")
    scales = (
        ordered.groupby("series_id", observed=True)["sales"]
        .apply(
            lambda values: (
                float(np.mean(np.diff(values.to_numpy(dtype="float64")) ** 2))
                if len(values) >= min_history
                else np.nan
            )
        )
        .rename("scale")
    )
    errors = (
        predictions.assign(squared=(predictions["y"] - predictions["yhat"]) ** 2)
        .groupby("series_id", observed=True)["squared"]
        .mean()
        .rename("mse")
    )
    combined = pd.concat([errors, scales], axis=1).dropna()
    combined = combined[combined["scale"] > 0]
    if combined.empty:
        raise MetricError("no series with a positive naive scale")
    return np.sqrt(combined["mse"] / combined["scale"]).rename("rmsse")


def hierarchical_rmsse(
    predictions: pd.DataFrame, training: pd.DataFrame, hierarchy: list[list[str]] | None = None
) -> dict[str, float]:
    """RMSSE computed at each hierarchy level.

    `hierarchy` is a list of level lists, e.g. [["CA_1","TX_1"], ["FOODS_3"], ["ALL"]].
    If None, defaults to store-level -> department -> overall.
    """
    # Default hierarchy: store -> department -> overall
    if hierarchy is None:
        hierarchy = [
            ["store_id"],  # store level
            ["dept_id"],  # department level
            [],  # overall (all series)
        ]

    results: dict[str, float] = {}
    # Build series_id -> level mapping
    pred_by_series = predictions.groupby("series_id", observed=True)

    for level_idx, level_cols in enumerate(hierarchy):
        if not level_cols:  # overall level
            # All series aggregated
            overall_rmsse = rmsse_per_series(predictions, training).mean()
            results[f"level_{level_idx}_rmsse"] = float(overall_rmsse) if not pd.isna(overall_rmsse) else float("nan")
            continue

        # Compute RMSSE per series in this level, then average within each group
        level_errors: dict[str, list[float]] = {}
        for series_id, group in pred_by_series:
            # Extract the relevant portion of the hierarchy for this series
            # Group by the columns at this level
            series_panel = training[training["series_id"] == series_id]
            if series_panel.empty:
                continue

            # Get the group key for this series at this level
            group_keys = {}
            for col in level_cols:
                if col in series_panel.columns:
                    val = series_panel[col].iloc[0]
                    group_keys[col] = val
                else:
                    group_keys[col] = "unknown"

            # Find or create the group entry
            key_tuple = tuple(group_keys[c] for c in level_cols)
            if key_tuple not in level_errors:
                level_errors[key_tuple] = []
            level_errors[key_tuple].append(
                rmsse_per_series(group, series_panel).iloc[0]
            )

        # Average RMSSE within each group at this level
        level_averages: list[float] = []
        for key, rmsse_vals in level_errors.items():
            if rmsse_vals:
                level_averages.append(float(np.mean(rmsse_vals)))

        if level_averages:
            results[f"level_{level_idx}_rmsse"] = float(np.mean(level_averages))
        else:
            results[f"level_{level_idx}_rmsse"] = float("nan")

    return results


def paired_bootstrap_wape_delta(
    predictions: pd.DataFrame,
    baseline: str,
    candidate: str,
    reps: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap the WAPE difference over series, paired by series id.

    Resampling series (not rows) respects the dependence structure: a SKU's 28
    daily errors are not independent draws.
    """
    required = {"model", "series_id", "y", "yhat"}
    if not required.issubset(predictions.columns):
        raise MetricError(f"predictions need columns {sorted(required)}")
    frames = {}
    for name in (baseline, candidate):
        subset = predictions[predictions["model"] == name]
        if subset.empty:
            raise MetricError(f"no predictions for model {name!r}")
        frames[name] = (
            subset.assign(abs_error=(subset["y"] - subset["yhat"]).abs(), actual=subset["y"].abs())
            .groupby("series_id", observed=True)[["abs_error", "actual"]]
            .sum()
        )
    joined = frames[baseline].join(frames[candidate], suffixes=("_base", "_cand"), how="inner")
    if joined.empty:
        raise MetricError("baseline and candidate share no series")
    point = float(
        joined["abs_error_cand"].sum() / joined["actual_cand"].sum()
        - joined["abs_error_base"].sum() / joined["actual_base"].sum()
    )
    if reps <= 0:
        return {
            "delta": point,
            "ci_low": point,
            "ci_high": point,
            "reps": 0.0,
            "n_series": float(len(joined)),
        }
    rng = np.random.default_rng(seed)
    base_error = joined["abs_error_base"].to_numpy()
    base_actual = joined["actual_base"].to_numpy()
    cand_error = joined["abs_error_cand"].to_numpy()
    cand_actual = joined["actual_cand"].to_numpy()
    deltas = np.empty(reps, dtype="float64")
    n = len(joined)
    for i in range(reps):
        idx = rng.integers(0, n, n)
        base_den = base_actual[idx].sum()
        cand_den = cand_actual[idx].sum()
        deltas[i] = (
            cand_error[idx].sum() / cand_den - base_error[idx].sum() / base_den
            if base_den > 0 and cand_den > 0
            else np.nan
        )
    deltas = deltas[np.isfinite(deltas)]
    if deltas.size == 0:
        raise MetricError("bootstrap produced no valid resamples")
    alpha = (1 - confidence) / 2
    return {
        "delta": point,
        "ci_low": float(np.quantile(deltas, alpha)),
        "ci_high": float(np.quantile(deltas, 1 - alpha)),
        "reps": float(deltas.size),
        "n_series": float(n),
    }
