from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from demand.config import Config
from demand.service.app import create_app


@pytest.fixture(scope="module")
def client(trained_artifact, demo_config: Config):
    app = create_app(demo_config.artifacts_dir)
    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_loaded_model(client: TestClient, trained_artifact) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_version"] == trained_artifact.run_id
    assert payload["horizon"] == trained_artifact.horizon
    assert payload["series"] == 2


def test_model_info_exposes_metadata(client: TestClient) -> None:
    response = client.get("/model")
    assert response.status_code == 200
    payload = response.json()
    assert payload["model"] == "lightgbm"
    assert payload["n_series"] == 2
    assert payload["dataset"]["stores"] == ["CA_1"]


def test_forecast_returns_contiguous_points(client: TestClient, trained_artifact) -> None:
    series_id = trained_artifact.series[0]
    response = client.post("/forecast", json={"series_id": series_id, "horizon_days": 5})
    assert response.status_code == 200
    payload = response.json()
    assert payload["series_id"] == series_id
    assert payload["model_version"] == trained_artifact.run_id
    assert len(payload["points"]) == 5
    dates = [point["date"] for point in payload["points"]]
    assert dates == sorted(dates)
    assert all(point["yhat"] >= 0 for point in payload["points"])


def test_unknown_series_is_404(client: TestClient) -> None:
    response = client.post("/forecast", json={"series_id": "nope", "horizon_days": 3})
    assert response.status_code == 404


def test_horizon_beyond_training_is_422(client: TestClient, trained_artifact) -> None:
    response = client.post(
        "/forecast",
        json={
            "series_id": trained_artifact.series[0],
            "horizon_days": trained_artifact.horizon + 1,
        },
    )
    assert response.status_code == 422


def test_invalid_horizon_is_422(client: TestClient, trained_artifact) -> None:
    response = client.post(
        "/forecast", json={"series_id": trained_artifact.series[0], "horizon_days": 0}
    )
    assert response.status_code == 422


def test_missing_artifact_degrades_instead_of_crashing(tmp_path: Path) -> None:
    app = create_app(tmp_path / "empty")
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "degraded"
        forecast = client.post("/forecast", json={"series_id": "x", "horizon_days": 1})
        assert forecast.status_code == 503
