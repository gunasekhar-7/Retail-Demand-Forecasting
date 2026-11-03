# ADR 0001: WAPE as the primary metric

Status: accepted (2026-11-02)

## Context

Retail demand is zero-inflated: closed days, discontinued SKUs and slow movers produce
many zero actuals. MAPE divides by the actual, so it is undefined on those days and
unstable on very low volume. The M5 competition uses a weighted, scaled error
(WRMSSE); Rossmann uses RMSPE while ignoring zero-sales days.

## Decision

Use **WAPE** = Σ|y − ŷ| / Σ|y| as the primary metric. Report MAPE on non-zero actuals
only, with the excluded count. Report RMSSE per series (scaled by the in-sample
one-step naive error) as a scale-free secondary metric, and bias to catch systematic
over- or under-ordering.

## Consequences

- WAPE is volume-weighted, so a large error on a high-volume SKU dominates. That is
  intentional for inventory cost, but it means a model can look good while being poor
  on the long tail. The dashboard ranks worst series separately.
- WAPE is undefined when total demand is zero; the metric raises instead of returning
  `inf` or `nan`, so a degenerate fold fails loudly.
- The official M5 WRMSSE is not reproduced (it needs 12-level aggregation and dollar
  weights). RMSSE is the honest subset.
