from __future__ import annotations

import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from demand.config import Config
from demand.registry import RunArtifact
from demand.service.app import create_app


@pytest.fixture
def keyed_client(
    trained_artifact: RunArtifact,
    demo_config: Config,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("DEMAND_API_KEY", "test-key")
    app = create_app(demo_config.artifacts_dir)
    with TestClient(app) as client:
        yield client, trained_artifact


def _body(artifact: RunArtifact) -> dict[str, object]:
    return {"series_id": artifact.series[0], "horizon_days": 1}


def test_forecast_and_model_require_the_api_key(keyed_client) -> None:
    client, artifact = keyed_client
    body = _body(artifact)

    missing = client.post("/forecast", json=body)
    assert missing.status_code == 401
    assert missing.json()["detail"] == "invalid or missing API key"
    assert missing.headers["www-authenticate"] == "ApiKey"

    assert client.post("/forecast", json=body, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/model").status_code == 401

    allowed = client.post("/forecast", json=body, headers={"X-API-Key": "test-key"})
    assert allowed.status_code == 200
    assert len(allowed.json()["points"]) == 1
    assert client.get("/model", headers={"X-API-Key": "test-key"}).status_code == 200


def test_health_and_metrics_stay_open_when_auth_is_on(keyed_client) -> None:
    client, _ = keyed_client
    for path in ("/health", "/health/live", "/health/ready", "/metrics"):
        assert client.get(path).status_code == 200


def test_request_id_is_echoed_or_generated(keyed_client) -> None:
    client, _ = keyed_client

    supplied = client.get("/health", headers={"X-Request-ID": "req-123"})
    assert supplied.headers["x-request-id"] == "req-123"
    assert float(supplied.headers["x-response-time-ms"]) >= 0

    generated = client.get("/health").headers["x-request-id"]
    assert len(generated) == 32
    assert generated != "req-123"


def test_metrics_endpoint_reports_request_counters(keyed_client) -> None:
    client, artifact = keyed_client
    client.post(
        "/forecast",
        json=_body(artifact),
        headers={"X-API-Key": "test-key"},
    )
    text = client.get("/metrics").text
    assert "demand_http_requests_total" in text
    assert 'route="/forecast"' in text
    assert "demand_model_loaded 1.0" in text


def test_missing_artifact_returns_generic_detail(tmp_path: Path) -> None:
    app = create_app(tmp_path / "missing")
    with TestClient(app) as client:
        assert client.get("/health/live").status_code == 200

        ready = client.get("/health/ready")
        assert ready.status_code == 503
        assert ready.json()["detail"] == "model unavailable"
        assert "artifacts" not in ready.json()["detail"]

        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "degraded"
        assert health.json()["detail"] == "model unavailable"


def test_rate_limit_returns_429_after_the_window_budget(
    trained_artifact: RunArtifact,
    demo_config: Config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEMAND_RATE_LIMIT_PER_MINUTE", "2")
    app = create_app(demo_config.artifacts_dir)
    body = _body(trained_artifact)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.post("/forecast", json=body).status_code == 200
        assert client.post("/forecast", json=body).status_code == 200
        limited = client.post("/forecast", json=body)
        assert limited.status_code == 429
        assert limited.headers["retry-after"] == "60"
        assert limited.json()["detail"] == "rate limit exceeded"


def test_oversized_body_is_rejected_before_parsing(
    demo_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DEMAND_MAX_BODY_BYTES", "64")
    app = create_app(demo_config.artifacts_dir)
    with TestClient(app) as client:
        response = client.post(
            "/forecast",
            content=b"{" + b" " * 200 + b"}",
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 413
        assert response.json()["detail"] == "request body too large"
        assert response.headers["x-request-id"]


def test_security_headers_are_present(keyed_client) -> None:
    client, _ = keyed_client
    response = client.get("/health")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["cache-control"] == "no-store"


def test_raising_route_preserves_original_exception_and_records_500(
    demo_config: Config, caplog: pytest.LogCaptureFixture
) -> None:
    app = create_app(demo_config.artifacts_dir)

    @app.get("/boom")
    def boom() -> None:
        raise RuntimeError("boom-original")

    with (
        TestClient(app, raise_server_exceptions=False) as client,
        caplog.at_level(logging.INFO, logger="demand.service"),
    ):
        response = client.get("/boom")

    assert response.status_code == 500
    assert "RuntimeError: boom-original" in caplog.text
    assert '"event":"http_request"' in caplog.text
    assert '"status":500' in caplog.text
    assert "UnboundLocalError" not in caplog.text

    with (
        TestClient(app, raise_server_exceptions=True) as client,
        pytest.raises(RuntimeError, match="boom-original"),
    ):
        client.get("/boom")
