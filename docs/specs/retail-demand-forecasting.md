# Spec: Retail Demand Forecasting portfolio system

## Goal

A production-shaped demand forecasting system on M5-shaped retail data that demonstrates
end-to-end data science: SQL and pandas engineering, leakage-safe features, model
comparison, rigorous temporal validation, serving, a dashboard and quantified business
impact.

## Scope

- **Data**: M5 Walmart (`sales_train_evaluation.csv`, `calendar.csv`, `sell_prices.csv`),
  scoped by store and department via config. A synthetic generator with the same schema
  makes the project reproducible without Kaggle credentials.
- **Grain**: one row per item × store × day.
- **Horizon**: 28 days.
- **Models**: seasonal-naive baseline, LightGBM direct multi-horizon, capped Prophet
  challenger.
- **Validation**: expanding-window walk-forward folds, non-overlapping 28-day tests,
  WAPE primary, MAPE/RMSSE/bias secondary, paired series bootstrap.
- **Serving**: FastAPI with artifact loaded in `lifespan`, registry with git SHA and
  metadata, Streamlit dashboard over backtest artifacts.
- **Impact**: order-up-to inventory simulation with declared cost assumptions and a
  sensitivity sweep.

## Acceptance criteria

1. `demand demo` completes from an empty checkout and writes predictions, metrics,
   an inventory report and rendered results.
2. No feature at an origin depends on data after that origin; a test proves it by
   mutating future rows.
3. Training and serving features are identical for the same origin; a test proves it.
4. Folds never overlap and every test window starts the day after its training cutoff.
5. Every number in the README results block comes from `metrics.json`; missing metrics
   raise instead of rendering a placeholder.
6. The API returns 404/422/503 for unknown series, invalid horizon and missing model.
7. `ruff`, `mypy` and the fast test suite pass in CI.

## Out of scope

Full 30,490-series training by default, hierarchical reconciliation and official WRMSSE,
probabilistic forecasts, workflow orchestration, cloud deployment, Rossmann as a second
dataset.
