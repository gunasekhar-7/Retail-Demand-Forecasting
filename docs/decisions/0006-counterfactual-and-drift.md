# ADR 0006: Counterfactual what-if inference, hot-reload, and drift monitoring

Status: accepted (2026-11-02)

## Context

In an enterprise retail deployment:
1. Merchandising and replenishment planners frequently require "what-if" counterfactual
   simulations (e.g. assessing demand lift under price promotions or holiday overrides)
   without retraining models.
2. In-process batch pre-scoring at startup yields sub-15ms reads, but in production, new
   model versions must be deployed without dropping active HTTP connections or restarting
   containers (zero-downtime hot reloading).
3. Production covariate and target shift (e.g. price inflation or holiday shopping spikes)
   must be detected and alerted before forecast accuracy degrades.

## Decision

1. **Counterfactual What-If Endpoint:** Expose `POST /forecast/what-if` backed by
   `Predictor.forecast_dynamic()`. It allows callers to provide scenario multipliers
   (such as `price_multiplier: 0.85` for a 15% promotional discount) or event overrides,
   dynamically recalculating price and calendar features while computing baseline vs. scenario
   demand lift.
2. **Zero-Downtime Hot-Reload:** Expose `POST /reload` (authenticated via `X-API-Key`) that
   atomically loads the latest published artifact into an in-memory `Predictor`, swaps
   `app.state.predictor` atomically, and echoes `X-Model-Version` in response headers.
3. **Data Drift Monitoring:** Implement Population Stability Index (PSI) and 1D Wasserstein
   Distance in `demand.monitoring.drift`. Expose a CLI command `demand drift` and HTTP endpoint
   `GET /monitoring/drift`, reporting green/yellow/red status and streaming Prometheus metrics
   (`demand_drift_psi`, `demand_drift_alert`).

## Consequences

- Business users can interactively evaluate pricing scenarios through the API.
- Continuous deployment pipelines can trigger `POST /reload` after nightly training without
  restarting Kubernetes pods or draining connections.
- Prometheus alerting rules can trigger automated notifications when PSI exceeds 0.25 on
  core features (`sales`, `sell_price`).

