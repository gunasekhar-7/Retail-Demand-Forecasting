"""Production data and feature drift monitoring.

Computes Population Stability Index (PSI) and Wasserstein Distance to detect
covariate shift and target distribution drift between training reference data
and current serving batches.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from demand.service.observability import DRIFT_ALERT, DRIFT_PSI


def calculate_psi(
    reference: np.ndarray,
    current: np.ndarray,
    num_buckets: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """Calculate Population Stability Index (PSI) between two empirical distributions.

    Rules of thumb:
      PSI < 0.10: No significant drift (stable)
      0.10 <= PSI < 0.25: Moderate drift (warning)
      PSI >= 0.25: Significant drift (alert, model retraining recommended)
    """
    ref = np.asarray(reference, dtype="float64")
    curr = np.asarray(current, dtype="float64")
    ref = ref[np.isfinite(ref)]
    curr = curr[np.isfinite(curr)]

    if len(ref) == 0 or len(curr) == 0:
        return 0.0

    quantiles = np.linspace(0, 100, num_buckets + 1)
    bins = np.percentile(ref, quantiles)
    bins[0] = -np.inf
    bins[-1] = np.inf
    bins = np.unique(bins)

    if len(bins) < 2:
        return 0.0

    ref_counts, _ = np.histogram(ref, bins=bins)
    curr_counts, _ = np.histogram(curr, bins=bins)

    ref_pct = (ref_counts + epsilon) / (len(ref) + epsilon * len(ref_counts))
    curr_pct = (curr_counts + epsilon) / (len(curr) + epsilon * len(curr_counts))

    psi = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
    return float(max(0.0, psi))


def calculate_wasserstein(reference: np.ndarray, current: np.ndarray) -> float:
    """Calculate normalized 1D Wasserstein (Earth Mover's) distance."""
    ref = np.asarray(reference, dtype="float64")
    curr = np.asarray(current, dtype="float64")
    ref = ref[np.isfinite(ref)]
    curr = curr[np.isfinite(curr)]

    if len(ref) == 0 or len(curr) == 0:
        return 0.0

    ref_std = float(np.std(ref))
    scale = ref_std if ref_std > 1e-6 else 1.0

    n_eval = min(500, max(len(ref), len(curr)))
    percentiles = np.linspace(0, 1, n_eval)
    q_ref = np.quantile(ref, percentiles)
    q_curr = np.quantile(curr, percentiles)

    dist = float(np.mean(np.abs(q_ref - q_curr)))
    return dist / scale


def compute_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features: list[str],
    psi_warning: float = 0.10,
    psi_alert: float = 0.25,
) -> dict[str, Any]:
    """Compare reference and current frames across specified numerical features."""
    feature_metrics: dict[str, dict[str, Any]] = {}
    has_warning = False
    has_alert = False

    for col in features:
        if col not in reference_df.columns or col not in current_df.columns:
            continue

        ref_vals = reference_df[col].to_numpy()
        curr_vals = current_df[col].to_numpy()

        psi = calculate_psi(ref_vals, curr_vals)
        wass = calculate_wasserstein(ref_vals, curr_vals)

        if psi >= psi_alert:
            status = "alert"
            has_alert = True
        elif psi >= psi_warning:
            status = "warning"
            has_warning = True
        else:
            status = "stable"

        feature_metrics[col] = {
            "psi": round(psi, 4),
            "normalized_wasserstein": round(wass, 4),
            "status": status,
            "ref_mean": round(float(np.nanmean(ref_vals)), 4),
            "curr_mean": round(float(np.nanmean(curr_vals)), 4),
        }

    overall_status = "alert" if has_alert else ("warning" if has_warning else "ok")

    return {
        "status": overall_status,
        "generated_at": datetime.now(UTC).isoformat(),
        "reference_rows": len(reference_df),
        "current_rows": len(current_df),
        "psi_warning_threshold": psi_warning,
        "psi_alert_threshold": psi_alert,
        "features": feature_metrics,
    }


def run_drift_check(
    panel: pd.DataFrame,
    cutoff: pd.Timestamp,
    window_days: int = 28,
    features: list[str] | None = None,
    out_dir: Path | None = None,
    psi_warning: float = 0.10,
    psi_alert: float = 0.25,
) -> dict[str, Any]:
    """Execute drift detection between the reference baseline and recent observation window."""
    target_features = features or ["sales", "sell_price"]
    cutoff = pd.Timestamp(cutoff).normalize()
    split_date = cutoff - pd.Timedelta(days=window_days)

    ref_df = panel[panel["date"] < split_date]
    curr_df = panel[(panel["date"] >= split_date) & (panel["date"] <= cutoff)]

    report = compute_drift_report(
        ref_df, curr_df, target_features, psi_warning=psi_warning, psi_alert=psi_alert
    )

    # Update Prometheus gauges
    for feat_name, metrics in report["features"].items():
        DRIFT_PSI.labels(feat_name).set(metrics["psi"])
    DRIFT_ALERT.set(1 if report["status"] == "alert" else 0)

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / "drift.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    return report
