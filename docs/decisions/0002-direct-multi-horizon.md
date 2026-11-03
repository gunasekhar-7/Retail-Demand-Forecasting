# ADR 0002: Direct multi-horizon LightGBM

Status: accepted (2026-11-02)

## Context

A 28-day forecast can be produced by recursion (predict day 1, feed it back), by one
model per horizon, or by a single direct model with the horizon as a feature.

## Decision

One LightGBM model trained on `(series, origin, horizon)` rows, with `horizon` as a
numeric feature and the target being demand at `origin + horizon`. Features are
as-of-origin lag/rolling statistics plus target-date calendar facts (weekday, holiday,
SNAP), which are known in advance.

## Consequences

- No recursive error compounding and one artifact to register and serve.
- Training rows multiply by 28; `features.origin_stride_days` (14 days) and
  `features.max_training_rows` bound the cost, and the guard fails with an actionable
  message rather than an OOM.
- The model learns horizon-specific noise implicitly, which shows up as the
  per-horizon WAPE curve.
- Recursive models could use a longer history per row; this design trades that for
  simpler serving and no future-regressor feedback loop.
