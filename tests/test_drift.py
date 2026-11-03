from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from demand.config import Config
from demand.monitoring.drift import (
    calculate_psi,
    calculate_wasserstein,
    run_drift_check,
)
from demand.service.app import create_app


def test_psi_and_wasserstein_metrics() -> None:
    rng = np.random.default_rng(42)
    ref = rng.normal(10.0, 1.0, 1000)
    stable = rng.normal(10.0, 1.0, 500)
    drifted = rng.normal(15.0, 3.0, 500)

    psi_stable = calculate_psi(ref, stable)
    psi_drifted = calculate_psi(ref, drifted)
    assert psi_stable < 0.10
    assert psi_drifted > 0.25

    wass_stable = calculate_wasserstein(ref, stable)
    wass_drifted = calculate_wasserstein(ref, drifted)
    assert wass_drifted > wass_stable


def test_drift_report_generation(tiny_panel: pd.DataFrame) -> None:
    cutoff = pd.Timestamp("2020-02-15")
    report = run_drift_check(tiny_panel, cutoff=cutoff, window_days=14, features=["sales", "sell_price"])
    assert "status" in report
    assert "features" in report
    assert "sales" in report["features"]
    assert "psi" in report["features"]["sales"]


def test_drift_api_endpoint(demo_config: Config) -> None:
    app = create_app(demo_config.artifacts_dir)
    with TestClient(app) as client:
        resp = client.get("/monitoring/drift")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("ok", "warning", "alert", "unavailable")
