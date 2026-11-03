"""FastAPI application factory.

The model loads in `lifespan`, not at import time, so importing the module for
tests or tooling stays free. A missing artifact degrades the service instead of
crashing the container: `/health/ready` reports 503 and every forecast returns
503. Full startup errors go to the log; clients only see a generic detail.

Every request gets a request id, one structured log line, and a Prometheus
sample. `/health/live` answers whether the process is up; `/health/ready`
answers whether a model is loaded.
"""

from __future__ import annotations

import datetime as dt
import os
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from demand import __version__
from demand.registry import RegistryError
from demand.service.guard import (
    RateLimiter,
    max_body_bytes,
    resolve_request_id,
    verify_api_key,
    verify_api_keys,
)
from demand.service.observability import (
    IN_FLIGHT,
    LOGGER,
    MODEL_LOADED,
    MODEL_RELOAD_COUNT,
    REQUEST_COUNT,
    REQUEST_LATENCY,
    log_request,
    route_label,
)
from demand.service.predictor import HorizonError, Predictor, UnknownSeriesError
from demand.service.schemas import (
    DriftReportResponse,
    ForecastPoint,
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    ModelInfo,
    ReloadResponse,
    WhatIfRequest,
    WhatIfResponse,
)

PUBLIC_PATHS = frozenset({"/health", "/health/live", "/health/ready", "/metrics"})
STARTUP_DETAIL = "model unavailable"


def _artifacts_dir() -> Path:
    return Path(os.environ.get("DEMAND_ARTIFACTS", "artifacts"))


def create_app(artifacts_dir: Path | str | None = None) -> FastAPI:
    directory = Path(artifacts_dir) if artifacts_dir is not None else _artifacts_dir()
    limiter = RateLimiter.from_env()
    body_limit = max_body_bytes()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.predictor = None
        app.state.startup_error = None
        MODEL_LOADED.set(0)
        try:
            app.state.predictor = Predictor.load(directory)
        except (RegistryError, FileNotFoundError, OSError):
            LOGGER.exception("model load failed from %s", directory)
            app.state.startup_error = STARTUP_DETAIL
        else:
            MODEL_LOADED.set(1)
        yield
        app.state.predictor = None
        MODEL_LOADED.set(0)

    app = FastAPI(
        title="Retail Demand Forecasting API",
        version=__version__,
        description="28-day SKU-store demand forecasts with a frozen feature contract.",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def observe(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = resolve_request_id(request.headers.get("x-request-id"))
        path = request.url.path
        route = route_label(path)
        started = time.perf_counter()
        duration_ms = 0.0
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

        def rejected(
            status_code: int, detail: str, headers: dict[str, str] | None = None
        ) -> JSONResponse:
            payload = JSONResponse({"detail": detail}, status_code=status_code, headers=headers)
            payload.headers["X-Request-ID"] = request_id
            payload.headers["X-Content-Type-Options"] = "nosniff"
            REQUEST_COUNT.labels(request.method, route, str(status_code)).inc()
            log_request(
                request_id=request_id,
                method=request.method,
                path=path,
                status_code=status_code,
                duration_ms=0.0,
            )
            return payload

        raw_length = request.headers.get("content-length")
        if body_limit and raw_length is not None:
            try:
                too_large = int(raw_length) > body_limit
            except ValueError:
                too_large = True
            if too_large:
                return rejected(status.HTTP_413_CONTENT_TOO_LARGE, "request body too large")

        client = request.client.host if request.client is not None else "unknown"
        if limiter.limit_per_minute > 0 and path not in PUBLIC_PATHS and not limiter.allow(client):
            return rejected(
                status.HTTP_429_TOO_MANY_REQUESTS,
                "rate limit exceeded",
                {"Retry-After": "60"},
            )

        IN_FLIGHT.inc()
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception:
            LOGGER.exception("unhandled error request_id=%s path=%s", request_id, path[:200])
            raise
        finally:
            duration_ms = (time.perf_counter() - started) * 1000.0
            IN_FLIGHT.dec()
            REQUEST_COUNT.labels(request.method, route, str(status_code)).inc()
            REQUEST_LATENCY.labels(request.method, route).observe(duration_ms / 1000.0)
            log_request(
                request_id=request_id,
                method=request.method,
                path=path,
                status_code=status_code,
                duration_ms=duration_ms,
            )

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-ms"] = f"{duration_ms:.1f}"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        predictor_state = getattr(request.app.state, "predictor", None)
        if predictor_state is not None:
            response.headers["X-Model-Version"] = str(
                predictor_state.metadata.get("run_id", "unknown")
            )
        return response

    def predictor_or_503(request: Request) -> Predictor:
        predictor = getattr(request.app.state, "predictor", None)
        if predictor is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=getattr(request.app.state, "startup_error", STARTUP_DETAIL),
            )
        return predictor

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    def health(request: Request) -> HealthResponse:
        predictor = getattr(request.app.state, "predictor", None)
        if predictor is None:
            return HealthResponse(
                status="degraded",
                model_version=None,
                series=0,
                horizon=None,
                detail=getattr(request.app.state, "startup_error", STARTUP_DETAIL),
            )
        return HealthResponse(
            status="ok",
            model_version=predictor.metadata["run_id"],
            series=len(predictor.series),
            horizon=predictor.horizon,
        )

    @app.get("/health/live", include_in_schema=False)
    def live() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready", response_model=HealthResponse, include_in_schema=False)
    def ready(request: Request, response: Response) -> HealthResponse:
        payload = health(request)
        if payload.status != "ok":
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return payload

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get(
        "/model",
        response_model=ModelInfo,
        tags=["forecast"],
        dependencies=[Depends(verify_api_keys)],
    )
    def model_info(request: Request) -> ModelInfo:
        predictor = predictor_or_503(request)
        meta = predictor.metadata
        return ModelInfo(
            model_version=str(meta["run_id"]),
            model=str(meta["model"]),
            cutoff=dt.date.fromisoformat(str(meta["cutoff"])),
            horizon=int(meta["horizon"]),
            created_at=dt.datetime.fromisoformat(str(meta["created_at"])),
            git_sha=str(meta.get("git_sha", "unknown")),
            n_series=int(meta.get("dataset", {}).get("n_series", len(predictor.series))),
            dataset=dict(meta.get("dataset", {})),
            validation=meta.get("model_metadata", {}).get("validation"),
        )

    @app.post(
        "/forecast",
        response_model=ForecastResponse,
        tags=["forecast"],
        dependencies=[Depends(verify_api_keys)],
    )
    def forecast(payload: ForecastRequest, request: Request) -> ForecastResponse:
        predictor = predictor_or_503(request)
        try:
            points = predictor.forecast(payload.series_id, payload.horizon_days)
        except UnknownSeriesError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"unknown series_id: {payload.series_id}",
            ) from exc
        except HorizonError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
            ) from exc
        return ForecastResponse(
            series_id=payload.series_id,
            model_version=str(predictor.metadata["run_id"]),
            generated_at=dt.datetime.now(dt.UTC),
            points=[ForecastPoint(date=point["date"], yhat=point["yhat"]) for point in points],
        )

    @app.post(
        "/forecast/what-if",
        response_model=WhatIfResponse,
        tags=["forecast"],
        dependencies=[Depends(verify_api_keys)],
    )
    def what_if(payload: WhatIfRequest, request: Request) -> WhatIfResponse:
        predictor = predictor_or_503(request)
        try:
            scenario_points = predictor.forecast_dynamic(
                payload.series_id,
                payload.horizon_days,
                price_multiplier=payload.price_multiplier,
                event_override=payload.event_override,
            )
            baseline_points = predictor.forecast(payload.series_id, payload.horizon_days)
        except UnknownSeriesError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"unknown series_id: {payload.series_id}",
            ) from exc
        except HorizonError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
            ) from exc

        base_total = sum(p["yhat"] for p in baseline_points)
        scen_total = sum(p["yhat"] for p in scenario_points)
        delta_units = scen_total - base_total
        delta_pct = (delta_units / base_total * 100.0) if base_total > 0 else 0.0

        return WhatIfResponse(
            series_id=payload.series_id,
            model_version=str(predictor.metadata["run_id"]),
            price_multiplier=payload.price_multiplier,
            baseline_units=round(base_total, 2),
            scenario_units=round(scen_total, 2),
            delta_units=round(delta_units, 2),
            delta_pct=round(delta_pct, 2),
            points=[ForecastPoint(date=p["date"], yhat=p["yhat"]) for p in scenario_points],
        )

    @app.post(
        "/reload",
        response_model=ReloadResponse,
        tags=["ops"],
        dependencies=[Depends(verify_api_keys)],
    )
    def reload_model(request: Request) -> ReloadResponse:
        try:
            new_predictor = Predictor.load(directory)
        except Exception as exc:
            LOGGER.exception("model reload failed from %s", directory)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"model reload failed: {exc}",
            ) from exc

        old_version = (
            str(request.app.state.predictor.metadata["run_id"])
            if getattr(request.app.state, "predictor", None) is not None
            else "none"
        )
        request.app.state.predictor = new_predictor
        request.app.state.startup_error = None
        MODEL_LOADED.set(1)
        MODEL_RELOAD_COUNT.inc()
        new_version = str(new_predictor.metadata["run_id"])
        LOGGER.info("model hot-reloaded successfully: %s -> %s", old_version, new_version)
        return ReloadResponse(
            status="reloaded",
            previous_version=old_version,
            current_version=new_version,
            n_series=len(new_predictor.series),
        )

    @app.get(
        "/monitoring/drift",
        response_model=DriftReportResponse,
        tags=["ops"],
        dependencies=[Depends(verify_api_keys)],
    )
    def get_drift() -> DriftReportResponse:
        drift_file = directory / "monitoring" / "drift.json"
        if not drift_file.exists():
            return DriftReportResponse(status="unavailable")
        import json

        payload = json.loads(drift_file.read_text(encoding="utf-8"))
        return DriftReportResponse(
            status=payload.get("status", "unavailable"),
            generated_at=payload.get("generated_at"),
            reference_rows=payload.get("reference_rows", 0),
            current_rows=payload.get("current_rows", 0),
            features=payload.get("features", {}),
        )

    return app
