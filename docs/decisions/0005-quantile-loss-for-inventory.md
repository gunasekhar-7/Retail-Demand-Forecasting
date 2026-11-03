# ADR 0005: Quantile objectives for asymmetric inventory optimization

Status: accepted (2026-11-02)

## Context

Retail demand distributions are non-Gaussian, intermittent, and heavily right-skewed.
Traditional point forecasting models (trained with L1/MAE or L2/MSE) optimize for the
conditional median or mean. When driving inventory replenishment policies, safety buffers
are historically approximated using symmetric Gaussian residual assumptions:

$$SS = z \cdot \sigma \cdot \sqrt{L + 1}$$

Under intermittent or zero-inflated demand, Gaussian assumptions either over-buffer slow
movers or under-protect high-margin items during demand surges. Furthermore, stockout
penalties (lost customer lifetime value) are often far more costly than inventory carrying
costs.

## Decision

Support LightGBM quantile regression (`objective="quantile"`, with parameter `alpha` $\in (0, 1)$),
directly minimizing asymmetric pinball loss:

$$L_\alpha(y, \hat{y}) = \max(\alpha (y - \hat{y}), (1 - \alpha) (\hat{y} - y))$$

For a target cycle service level (e.g. 90%), the model directly predicts the 90th percentile
demand ($\hat{y}_{p90}$), integrating safety stock directly into the machine-learned quantile
rather than relying on post-hoc parametric formulas.

## Consequences

- Direct quantile predictions capture heteroskedasticity and seasonal variance changes
  without assuming normal residual distributions.
- The model family names itself deterministically (e.g. `lightgbm_p90`), allowing side-by-side
  backtesting and policy simulation against L1 median forecasts.
- Training requires specifying target quantiles; for multi-service-level operations, multiple
  quantile models can be trained or evaluated.

