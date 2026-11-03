from __future__ import annotations

from fastapi.testclient import TestClient

from demand.config import Config
from demand.registry import RunArtifact
from demand.service.app import create_app


def test_what_if_endpoint_scenario_analysis(
    trained_artifact: RunArtifact,
    demo_config: Config,
) -> None:
    app = create_app(demo_config.artifacts_dir)
    with TestClient(app) as client:
        series_id = trained_artifact.series[0]
        h = trained_artifact.horizon

        # Normal price
        base_resp = client.post(
            "/forecast/what-if",
            json={"series_id": series_id, "horizon_days": h, "price_multiplier": 1.0},
        )
        assert base_resp.status_code == 200
        base_data = base_resp.json()
        assert base_data["series_id"] == series_id
        assert len(base_data["points"]) == h
        assert base_data["delta_units"] == 0.0

        # Increased price (+50%)
        expensive_resp = client.post(
            "/forecast/what-if",
            json={"series_id": series_id, "horizon_days": h, "price_multiplier": 1.5},
        )
        assert expensive_resp.status_code == 200
        exp_data = expensive_resp.json()
        assert len(exp_data["points"]) == h
        assert exp_data["price_multiplier"] == 1.5
        # Dynamic scoring reflects the counterfactual change
        assert exp_data["delta_units"] != 0.0
        assert exp_data["delta_pct"] != 0.0


def test_hot_reload_and_model_version_header(
    trained_artifact: RunArtifact,
    demo_config: Config,
) -> None:
    app = create_app(demo_config.artifacts_dir)
    with TestClient(app) as client:
        # Check X-Model-Version header on forecast
        resp = client.post(
            "/forecast",
            json={"series_id": trained_artifact.series[0], "horizon_days": 1},
        )
        assert resp.status_code == 200
        assert "x-model-version" in resp.headers
        assert resp.headers["x-model-version"] == trained_artifact.run_id

        # Reload model
        reload_resp = client.post("/reload")
        assert reload_resp.status_code == 200
        data = reload_resp.json()
        assert data["status"] == "reloaded"
        assert data["current_version"] == trained_artifact.run_id
        assert data["n_series"] > 0
