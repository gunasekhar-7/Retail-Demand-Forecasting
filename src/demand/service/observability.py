"""Process-local Prometheus metrics and structured request logs.

Route labels are allowlisted so an arbitrary 404 path cannot explode metric
cardinality. Logs are one JSON object per request and never include headers or
bodies, which keeps the shared API key and caller payloads out of stdout.
"""

from __future__ import annotations

import json
import logging

from prometheus_client import Counter, Gauge, Histogram

LOGGER = logging.getLogger("demand.service")

REQUEST_COUNT = Counter(
    "demand_http_requests_total",
    "HTTP requests by method, route and status code.",
    ("method", "route", "status"),
)
REQUEST_LATENCY = Histogram(
    "demand_http_request_seconds",
    "Request latency by method and route.",
    ("method", "route"),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
IN_FLIGHT = Gauge("demand_http_in_flight", "Requests currently being served.")
MODEL_LOADED = Gauge("demand_model_loaded", "1 when a model artifact is loaded, 0 otherwise.")
MODEL_RELOAD_COUNT = Counter("demand_model_reloads_total", "Total successful model hot reloads.")
DRIFT_PSI = Gauge("demand_drift_psi", "Population Stability Index by feature.", ("feature",))
DRIFT_ALERT = Gauge("demand_drift_alert", "1 if any feature exceeds alert threshold, 0 otherwise.")

ROUTE_LABELS = frozenset(
    {
        "/health",
        "/health/live",
        "/health/ready",
        "/metrics",
        "/model",
        "/forecast",
        "/forecast/what-if",
        "/reload",
        "/monitoring/drift",
        "/docs",
        "/openapi.json",
    }
)


def route_label(path: str) -> str:
    return path if path in ROUTE_LABELS else "other"


def log_request(
    *,
    request_id: str,
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
) -> None:
    LOGGER.info(
        json.dumps(
            {
                "event": "http_request",
                "request_id": request_id,
                "method": method,
                "path": path[:200],
                "status": status_code,
                "duration_ms": round(duration_ms, 2),
            },
            separators=(",", ":"),
        )
    )
