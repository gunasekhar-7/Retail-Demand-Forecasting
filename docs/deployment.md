# Deploying the service

The service is a read-only, artifact-backed API. It is designed to sit behind a reverse
proxy that owns TLS, user authentication and shared rate limiting. Do not publish port
8000 directly.

```
client ──TLS──► reverse proxy ──HTTP──► 127.0.0.1:8000 ──► uvicorn (2 workers)
                 │                                          │
                 ├─ identity / basic auth                   ├─ artifacts/latest.json (ro)
                 └─ shared rate limit                       └─ /metrics scraped by Prometheus
```

## Local container deployment

```bash
DEMAND_API_KEY="$(openssl rand -hex 32)" docker compose up --build
```

Compose binds both services to `127.0.0.1`. The API runs read-only, without Linux
capabilities, with `no-new-privileges`, a 1 GB memory limit and a PID limit. Artifacts
are mounted read-only; the container never writes to them.

The API image is built from the `api` target and installs only base runtime dependencies:
pandas, numpy, pyarrow, LightGBM, FastAPI, uvicorn, pydantic, PyYAML and
prometheus-client. It contains no Prophet, Streamlit, DuckDB or httpx. `libgomp1` is still
required because LightGBM's OpenMP runtime is not part of `python:3.12-slim`. The
dashboard target adds the `dashboard` extra (Streamlit); local pipeline, dashboard and
Prophet work uses `uv sync --extra dev`.

Measured image sizes: API **1.43 GB** vs **3.14 GB** for the previous all-dependencies
image (about 55% smaller); dashboard **1.72 GB**. Build them directly with:

```bash
docker build --target api -t demand-api .
docker build --target dashboard -t demand-dashboard .
```

When the key is set, health and metrics stay open for the orchestrator, while model and
forecast routes require it:

```bash
curl -s localhost:8000/forecast \
  -H "content-type: application/json" \
  -H "X-API-Key: $DEMAND_API_KEY" \
  -d '{"series_id": "FOODS_3_090_CA_1", "horizon_days": 28}'
```

## Reverse proxy

Caddy example (automatic TLS, basic auth for humans, injected service key):

```caddyfile
demand.example.com {
    encode zstd gzip
    basicauth {
        planner {$BASICAUTH_HASH}
    }
    reverse_proxy 127.0.0.1:8000 {
        header_up X-API-Key {$DEMAND_API_KEY}
    }
    header {
        Strict-Transport-Security "max-age=31536000; includeSubDomains"
        X-Content-Type-Options "nosniff"
    }
}
```

Basic auth is appropriate for a small internal audience. For per-user identity or
per-tenant data, replace it with OIDC and enforce authorization at the proxy or in a
dedicated gateway. `DEMAND_API_KEY` is a shared service credential, not a user session.

The dashboard is a separate Streamlit process on `127.0.0.1:8501`. Streamlit has no
built-in authentication; either put it behind the same proxy policy or keep it internal.

## Metrics and probes

```yaml
scrape_configs:
  - job_name: demand-api
    metrics_path: /metrics
    static_configs:
      - targets: ["127.0.0.1:8000"]
```

- `/health/live` is process liveness; never fail it because of artifacts.
- `/health/ready` returns `503` until a model is loaded. Route traffic only when it is `200`.
- Alerts worth having: readiness failing for 5 minutes, p95 latency above 150 ms,
  `demand_http_requests_total{status=~"5.."}` increasing, `demand_model_loaded == 0`,
  and `demand_drift_alert == 1`.

Each worker exposes its own registry; scrape every worker or run one worker per pod.

Example alerts live in [`deploy/prometheus/alerts.yml`](../deploy/prometheus/alerts.yml):
readiness, a 5xx ratio above 5%, p95 above the 150 ms budget, readiness flapping, and drift alerts.

## Rollout and rollback

Training writes a self-describing run directory and updates `artifacts/latest.json`.
A deployment is: train, verify `/model`, then either:
- **Zero-downtime hot reload:** Send `POST /reload` with `X-API-Key: $DEMAND_API_KEY`.
  The active worker atomically loads the new artifact, pre-scores the horizon, and swaps
  in-memory predictors without dropping active requests or cycling pods.
- **Pod restart:** Swap the mounted artifact directory and perform a rolling container restart.

Rollback is restoring the previous `latest.json` pointer and triggering `POST /reload`
(or restarting the container); there is no database migration.
run directory) and restarting; there is no database migration.

The API loads and batch-scores the full horizon at startup, so a rollout is not complete
until `/health/ready` is `200`. Keep the previous run directory until the new one has
served traffic.

## Runbook

| Symptom | Check | Action |
|---|---|---|
| `/health/ready` 503 | container logs for `model load failed` | Verify `artifacts/latest.json` and the run directory exist and are readable |
| `401` on forecast | `X-API-Key` matches `DEMAND_API_KEY` | Rotate the key in the proxy and service together |
| `429` under load | proxy rate limits and the per-worker limiter | Raise the proxy budget; do not rely on the in-process limiter for fairness |
| p95 regression | `/metrics` histogram and `reports/perf.md` | Check artifact size, worker count and CPU limits |
| Stale forecasts | `/model` cutoff and `run_id` | Retrain and redeploy; the scheduler is intentionally out of scope |

## What is intentionally not included

A scheduler, a feature store, per-user authorization and distributed tracing. The
pipeline is a documented command sequence and the README states the scaling limits; those
are the next architectural steps, not hidden behavior.
