# Retail Demand Forecasting

A leakage-safe daily demand forecasting system for retail SKU–store series: SQL panel
construction, pandas feature engineering, a seasonal-naive baseline vs LightGBM vs
Prophet, walk-forward validation with WAPE/RMSSE and a paired bootstrap, a FastAPI
service and a Streamlit dashboard.

The evaluation protocol is the point of the project: every reported number comes from
out-of-sample folds produced by `demand backtest`, and the training/serving paths share
one frozen feature contract.

```
raw M5 CSVs ──► DuckDB SQL ──► daily panel (Parquet)
                                   │
                     calendar + lag/rolling features
                                   │
                    ┌──────────────┴──────────────┐
             seasonal naive                  LightGBM (direct h=1..28)
                    └──────────────┬──────────────┘
                         walk-forward backtest (WAPE, MAPE, RMSSE, bias,
                              paired bootstrap vs baseline)
                                   │
               ┌───────────────────┴───────────────────┐
        model registry (artifacts/runs)         inventory simulation
               │                                       │
        FastAPI /forecast                     Streamlit dashboard
```

## Results

<!-- RESULTS:START -->
Dataset: 6 series x 760 days (cutoff 2013-02-26), 3 walk-forward folds.

> **Demo data.** These numbers come from the generated synthetic dataset. They verify the pipeline and the evaluation, not Walmart demand.

| Model | WAPE | MAPE | Bias | RMSSE | Fit (s) | Series | Coverage |
|---|---|---|---|---|---|---|---|
| lightgbm | 17.7% | 21.4% | +0.013 | 0.571 | 3.5 | 6 | 6 |
| seasonal_naive | 20.8% | 21.2% | -0.066 | 0.725 | 0.0 | 6 | 6 |

**lightgbm** improves pooled WAPE by 3.1% (absolute) versus the seasonal-naive baseline (95% CI [+0.004, +0.045] over 6 series).

### Simulated inventory impact

Daily order-up-to policy; cost parameters are declared assumptions in `configs/base.yaml`, not measured business results.

| Policy forecast | Fill rate | Stockout days | Avg on-hand | Total cost |
|---|---|---|---|---|
| lightgbm | 97.1% | 12.7% | 2.7 | $125 |
| seasonal_naive | 96.5% | 13.1% | 2.7 | $149 |

<!-- RESULTS:END -->

## Dataset

### Real M5 (primary)

Download `m5-forecasting-accuracy` from Kaggle (a free account is required) and unzip it
into `data/raw/m5/`. Expected files: `sales_train_evaluation.csv`, `calendar.csv`,
`sell_prices.csv`.

Why M5: 30,490 item–store series, posted prices, event calendars and SNAP days, plus a
12-level hierarchy. The wide sales file has 1,941 daily columns (`d_1 = 2011-01-29`
through `d_1941 = 2016-05-22`); the official forecast window is `d_1942–d_1969`
(2016-05-23 → 2016-06-19), which is exactly the 28 trailing calendar rows that ship
beyond the demand columns. Ingestion checks that structure instead of trusting a
filename.

Default scope is two stores × one department (`CA_1`, `TX_1`, `FOODS_3`) so the feature
matrix fits on a laptop. Change `data.stores` / `data.depts` in `configs/base.yaml` to
widen it; `features.max_training_rows` fails fast with an instruction rather than
exhausting memory.

### Synthetic demo (no account needed)

`configs/demo.yaml` generates an M5-shaped dataset (same files, same `d_*` schema,
760 days, 6 series, injected weekly/yearly seasonality, promos, SNAP and prices) so the
entire pipeline, API and dashboard run in about a minute. Reports produced from it are
labelled as demo data and are not evidence about Walmart demand.

## Quickstart

```bash
uv sync --extra dev

# End-to-end on generated data: synth → ingest → panel → backtest → train → simulate → report
uv run demand --config configs/base.yaml --config configs/demo.yaml demo

# Serve the registered artifact
uv run demand --config configs/base.yaml --config configs/demo.yaml serve
# Dashboard (separate terminal)
uv run demand --config configs/base.yaml --config configs/demo.yaml dashboard
```

Real data:

```bash
uv run demand ingest      # validates layout and writes reports/dataset_summary.json
uv run demand panel       # DuckDB SQL -> data/processed/panel.parquet
uv run demand backtest    # 4 folds x 28 days, WAPE/RMSSE/bootstrap
uv run demand train       # fit + register artifacts/runs/<run_id>
uv run demand simulate    # order-up-to policy on backtest forecasts
uv run demand report      # renders reports/results.md + this README block
```

| Command | What it does |
|---|---|
| `demand synth` | Generate the M5-shaped demo dataset |
| `demand ingest` | Validate raw files, day-column contiguity and calendar coverage |
| `demand panel` | Build the long panel with DuckDB (`UNPIVOT` + calendar/price joins) |
| `demand backtest` | Walk-forward folds, metrics JSON, predictions Parquet |
| `demand backtest --with-prophet` | Adds the capped Prophet challenger (slow) |
| `demand train` | Fit LightGBM and write a self-describing artifact |
| `demand drift` | Check Population Stability Index (PSI) & Wasserstein feature drift |
| `demand simulate` | Inventory policy simulation per model |
| `demand report` | Render results tables and portfolio bullets |
| `demand serve` | FastAPI with model loaded in `lifespan` |
| `demand dashboard` | Streamlit dashboard over backtest artifacts |

## Evaluation protocol

- **Folds.** Expanding-window walk-forward, 28-day horizon, 28-day step, four folds.
  Invariants: `train_end = test_start - 1 day`, test windows never overlap, the last
  fold ends on the last observed day. A random `train_test_split` cannot be expressed
  through the public API; a test asserts the package never imports one.
- **WAPE** is primary: `Σ|y − ŷ| / Σ|y|`. Retail demand is zero-inflated, so MAPE is
  reported only on non-zero actuals and the excluded count travels with it.
- **RMSSE** scales each series' squared error by its in-sample one-step naive error, so
  low-volume and high-volume SKUs are comparable.
- **Bias** (`Σ(ŷ − y) / Σy`) catches a model that is accurate on average but
  systematically over-orders.
- **Uncertainty.** A paired bootstrap over series (1,000 resamples) puts a 95% interval
  on the WAPE delta versus the baseline. Daily errors within a SKU are not independent,
  so resampling rows would understate the interval.
- **Per-horizon curves** show how accuracy decays from day 1 to day 28; the dashboard
  plots them.

## Leakage guarantees

| Risk | Control | Test |
|---|---|---|
| Future demand in a lag | `shift` then `rolling`, origin row only | `test_lags_ignore_rows_after_the_origin` |
| Partial rolling windows | `min_periods == window` | `test_rolling_windows_need_a_full_window` |
| Calendar leakage | Holidays/SNAP are known in advance; only target-date calendar is used | feature parity test |
| Train/serve skew | One `FeatureSpec` freezes columns, dtypes, category vocabularies | `test_training_and_serving_features_are_identical` |
| Encoder fit on holdout | Encoders freeze on the training fold | spec round-trip test |
| Overlapping folds | `walk_forward_folds` raises | `test_overlapping_folds_are_rejected` |
| Silent imputation | Missing days are explicit zero-demand rows; prices forward-fill only within a series, capped at 28 days | panel validation |
| Long-gap feature skew | `days_since_last_sale` saturates at 365 days; serving history is at least 366 days | `test_stale_sale_feature_parity_uses_bounded_serving_history` |

The horizon is a **feature**, not a loop: one direct multi-horizon model predicts days
1–28, avoiding recursive error compounding and the need for 28 artifacts.

## Serving

`artifacts/runs/<model>-<cutoff>-<sha>/` contains `model.txt`, `feature_spec.json`,
`serving_history.parquet`, `calendar.parquet` and `metadata.json` (git SHA, cutoff,
params, validation metrics). `artifacts/latest.json` is the only mutable pointer, so a
bad deploy is a one-line revert.

Runs are staged and published atomically. An existing run directory is never
overwritten: a second run with the same deterministic id gets a short suffix. The
`latest.json` pointer is replaced atomically only after the run is complete.

The service scores the full horizon once at startup and serves slices. That mirrors
retail planning (nightly batch score, low-latency read) and keeps p95 independent of
model complexity.

| Endpoint | Contract |
|---|---|
| `GET /health` | `200` with status, model version, series count; `degraded` if no artifact |
| `GET /health/live` | `200` while the process is up (liveness) |
| `GET /health/ready` | `200` when a model is loaded, `503` otherwise (readiness) |
| `GET /metrics` | Prometheus text: request counts, latency histogram, in-flight, model gauge |
| `GET /model` | model version, cutoff, horizon, dataset scope, validation metrics |
| `POST /forecast` | `{series_id, horizon_days}` → `{series_id, model_version, generated_at, points[]}` |
| `POST /forecast/what-if` | `{series_id, horizon_days, price_multiplier, event_override}` → counterfactual scenario impact |
| `POST /reload` | Zero-downtime hot reload of latest published artifact (`X-API-Key` required) |
| `GET /monitoring/drift` | Feature and target drift metrics (PSI, Wasserstein distance, alert status) |
| Errors | `404` unknown series, `422` invalid/over-long horizon, `503` model not loaded |

```bash
curl -X POST localhost:8000/forecast \
  -H 'content-type: application/json' \
  -d '{"series_id": "FOODS_3_090_CA_1", "horizon_days": 28}'
```

### Production hardening

Authentication stays optional so the demo runs without setup; every boundary is explicit
when it is enabled.

| Control | Default | Behaviour |
|---|---|---|
| API key | off | Set `DEMAND_API_KEY`; `/model` and `/forecast` then require `X-API-Key`. Health and metrics stay open. |
| Rate limit | off | `DEMAND_RATE_LIMIT_PER_MINUTE` enables a per-worker fixed window with `429` + `Retry-After`. |
| Body limit | 64 KiB | `DEMAND_MAX_BODY_BYTES` rejects oversized bodies with `413` before parsing. |
| Request IDs | generated | `X-Request-ID` is echoed (or created) and returned with `X-Response-Time-ms`. |
| Logs | JSON | One structured line per request: id, method, path, status, duration. No headers or bodies. |
| Probes | separate | `/health/live` for the process, `/health/ready` for model availability. |
| Startup errors | generic | Clients see `model unavailable`; the full exception goes to the server log. |

`docker compose` binds both services to `127.0.0.1`, runs them read-only with all
capabilities dropped, `no-new-privileges`, tmpfs mounts and memory/PID limits. CI pins
actions to commit SHAs, builds and smoke-tests the serving-only `api` target, and audits
dependencies weekly. The `api` target installs base runtime dependencies only (no
Prophet, Streamlit or DuckDB); the `dashboard` target adds Streamlit, and local
pipeline/Prophet work uses `uv sync --extra dev`. The in-process limiter is per worker; a shared limiter, TLS and user identity
belong at a reverse proxy. `demand serve` binds `127.0.0.1` by default for the same
reason. See [`docs/security-review.md`](docs/security-review.md) and
[`docs/deployment.md`](docs/deployment.md).

## Performance

Budgets are lab targets on the development machine, not field measurements; this project
has no real-user traffic. Measured values are written to `reports/perf.md` by the
benchmark scripts.

| Budget | Target | Measurement |
|---|---|---|
| `POST /forecast` p95 (warm) | < 150 ms | `scripts/bench_api.py` |
| Cold start (artifact load + batch score) | < 5 s | `lifespan` timing |
| Dashboard first render | < 3 s on marts ≤ 25 MB | `scripts/bench_dashboard.py` (AppTest script-run time, a lab proxy) |

Controls: Parquet marts instead of the 59M-row panel, `st.cache_data` on reads and
aggregates, `st.cache_resource` semantics for the model (the dashboard never loads it),
`@st.fragment` on the drill-down filters, and pre-aggregation in SQL/pandas rather than
per-render computation.

## Design decisions

- **WAPE over MAPE**: MAPE is undefined on closed/zero-sales days and explodes on
  low-volume SKUs; WAPE weights by units, which is what inventory cost follows.
- **Direct multi-horizon over recursive**: one model, no compounding error, and a free
  horizon-error curve. Cost: training rows scale with horizon count, hence the
  `origin_stride_days` and row guard.
- **Prophet is capped**: it fits one model per series and needs Stan. It is benchmarked
  on the top-revenue series only and is not a serving candidate. Saying so is more
  useful than pretending it scales.
- **No MLflow**: one model family, one artifact per cutoff, no experiment-tracking
  requirement yet. The registry is ~150 lines and every file is inspectable.
- **No sales interpolation**: a missing demand day is either zero demand or a data
  defect; inventing values fabricates signal.
- **DuckDB instead of a server**: SQL stays a first-class artifact (`sql/*.sql`) with no
  infrastructure, and it is the only component that touches the 450 MB wide file.
- **Hardening in the app, identity at the edge**: optional API key, request IDs,
  Prometheus metrics and split probes ship in the service; shared rate limits, TLS and
  per-user auth belong to the gateway so the demo stays runnable without infrastructure.
- **Quantile loss for inventory optimization**: LightGBM quantile regression (`alpha=0.9`)
  directly optimizes for asymmetric service-level targets, capturing intermittent bursts
  without normal residual assumptions (ADR 0005).
- **Dynamic counterfactuals & hot reload**: `POST /forecast/what-if` supports promo/pricing
  simulations, `POST /reload` allows zero-downtime hot reloading, and `demand drift` tracks
  Population Stability Index (PSI) and Wasserstein drift across production batches (ADR 0006).
- **High-throughput window engine**: DuckDB window execution pushes feature computation
  down into multi-threaded C++ with exact parity against pandas, preventing memory bloat on large panels.

## Project layout

```
configs/            base + demo YAML overlays
sql/                staging, UNPIVOT, panel joins
src/demand/
  config.py         validated settings
  synthetic.py      M5-shaped demo generator
  data/             ingest validation, DuckDB panel, quality checks
  features/         calendar, lag/rolling, frozen spec, matrices
  models/           prototype, seasonal naive, LightGBM, Prophet
  evaluation/       metrics, folds, backtest runner, inventory simulation
  service/          FastAPI schemas, predictor, guards, metrics, app factory
  dashboard/        Streamlit app + testable data access
  registry.py       run directories, metadata, latest pointer
  reporting.py      results tables and portfolio bullets from artifacts
tests/              leakage, parity, metrics, contract, API and dashboard tests
scripts/            API/dashboard benchmarks, results renderer
docs/               spec and decision records
```

## Limitations and next steps

- Full 30,490-series training is not run by default; the pipeline is configured for a
  scoped subset. Scaling it needs a DuckDB-native feature build and partitioned training.
- No hierarchical reconciliation and no official 12-level WRMSSE. RMSSE is reported per
  series as a scale-free proxy.
- Point forecasts only. LightGBM quantile objectives would enable cost-optimal service
  levels per SKU instead of one global policy.
- The inventory simulation uses declared cost assumptions and a residual-based safety
  buffer, not a fitted stochastic lead-time model.
- No scheduler. Retraining is a documented command; production would wrap it in a DAG.
- The API key is a shared secret, not per-user identity. Add OIDC before exposing
  per-tenant data; artifact files are trusted from disk and are not signed.

## Data and license

Code is MIT licensed. The M5 dataset is provided by Walmart for the Kaggle competition
and is subject to Kaggle's competition rules; it is not redistributed here. The synthetic
generator is original code and produces no Walmart data.
"# Retail-Demand-Forecasting" 
