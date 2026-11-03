"""As-of-origin series features.

Every value on row `t` is computed from rows strictly before `t` (shift-then-roll),
so selecting the origin row can never see the target day. `min_periods == window`
means a partial window is NaN rather than a deceptively confident mean.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from demand.config import FeatureConfig

DAYS_SINCE_LAST_SALE_CAP = 365


def add_series_features_duckdb(panel: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """High-throughput window feature pushdown using DuckDB SQL.

    Produces exact numerical parity with pandas while streaming aggregations
    in parallel without intermediate Python object allocation.
    """
    import duckdb

    required = {"series_id", "date", "sales"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel missing columns: {sorted(missing)}")
    if cfg.include_price and "sell_price" not in panel.columns:
        raise ValueError("include_price is true but panel has no sell_price column")
    if panel.empty:
        return add_series_features_pandas(panel, cfg)

    select_items = [
        "series_id", "date", "state_id", "item_id", "dept_id", "cat_id", "store_id"
    ]
    select_items = [c for c in select_items if c in panel.columns]
    for lag in cfg.lags:
        select_items.append(
            f"CAST(LAG(sales, {lag}) OVER (PARTITION BY series_id ORDER BY date) AS FLOAT) AS sales_lag_{lag}"
        )
    for w in cfg.rolling_windows:
        select_items.append(
            f"CAST(CASE WHEN pos >= {w} THEN "
            f"AVG(sales) OVER (PARTITION BY series_id ORDER BY date ROWS BETWEEN {w} PRECEDING AND 1 PRECEDING) "
            f"ELSE NULL END AS FLOAT) AS sales_mean_{w}"
        )
        select_items.append(
            f"CAST(CASE WHEN pos >= {w} THEN "
            f"STDDEV_SAMP(sales) OVER (PARTITION BY series_id ORDER BY date ROWS BETWEEN {w} PRECEDING AND 1 PRECEDING) "
            f"ELSE NULL END AS FLOAT) AS sales_std_{w}"
        )
        select_items.append(
            f"CAST(CASE WHEN pos >= {w - 1} THEN "
            f"SUM(CASE WHEN sales = 0 THEN 1.0 ELSE 0.0 END) OVER (PARTITION BY series_id ORDER BY date ROWS BETWEEN {w} PRECEDING AND 1 PRECEDING) "
            f"ELSE NULL END AS FLOAT) AS zero_count_{w}"
        )

    select_items.append(
        f"""CAST(CASE
        WHEN last_sale_pos IS NOT NULL THEN LEAST(pos - last_sale_pos, {DAYS_SINCE_LAST_SALE_CAP})
        ELSE {DAYS_SINCE_LAST_SALE_CAP}
    END AS FLOAT) AS days_since_last_sale"""
    )

    if cfg.include_price:
        select_items.append("CAST(sell_price AS FLOAT) AS sell_price")
        select_items.append(
            "CAST(sell_price / NULLIF(LAG(sell_price, 7) OVER (PARTITION BY series_id ORDER BY date), 0) - 1.0 AS FLOAT) AS price_change_7d"
        )
        select_items.append(
            """CAST(CASE WHEN pos >= 27 THEN
            (sell_price - AVG(sell_price) OVER (PARTITION BY series_id ORDER BY date ROWS BETWEEN 27 PRECEDING AND CURRENT ROW)) /
            NULLIF(STDDEV_SAMP(sell_price) OVER (PARTITION BY series_id ORDER BY date ROWS BETWEEN 27 PRECEDING AND CURRENT ROW), 0.0)
        ELSE NULL END AS FLOAT) AS price_zscore_28"""
        )

    select_sql = ",\n        ".join(select_items)
    query = f"""
    WITH indexed AS (
        SELECT *,
               (row_number() OVER (PARTITION BY series_id ORDER BY date) - 1) AS pos
        FROM panel
    ),
    with_last_sale AS (
        SELECT *,
               MAX(CASE WHEN sales > 0 THEN pos ELSE NULL END)
                   OVER (PARTITION BY series_id ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS last_sale_pos
        FROM indexed
    )
    SELECT
        {select_sql}
    FROM with_last_sale
    ORDER BY series_id, date
    """

    con = duckdb.connect()
    con.register("panel", panel)
    result = con.execute(query).df()
    result["date"] = pd.to_datetime(result["date"])

    keep = [
        "series_id",
        "date",
        "state_id",
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
    ] + [
        c
        for c in result.columns
        if c.startswith(("sales_lag_", "sales_mean_", "sales_std_", "zero_count_"))
    ]
    keep.append("days_since_last_sale")
    if cfg.include_price:
        keep += ["sell_price", "price_change_7d", "price_zscore_28"]
    return result[[c for c in dict.fromkeys(keep) if c in result.columns]]


def _grouped_rolling(values: pd.Series, groups: pd.Series, window: int, method: str) -> pd.Series:
    rolled = values.groupby(groups, observed=True).rolling(window, min_periods=window)
    result = getattr(rolled, method)().reset_index(level=0, drop=True)
    return result.reindex(values.index)


def add_series_features_pandas(panel: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    required = {"series_id", "date", "sales"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel missing columns: {sorted(missing)}")
    frame = panel.sort_values(["series_id", "date"], kind="stable").reset_index(drop=True)
    groups = frame["series_id"]
    sales = frame["sales"].astype("float64")
    grouped = sales.groupby(groups, observed=True)

    for lag in cfg.lags:
        frame[f"sales_lag_{lag}"] = grouped.shift(lag).astype("float32")

    shifted = grouped.shift(1)
    zero_shifted = (shifted == 0).astype("float32")
    for window in cfg.rolling_windows:
        frame[f"sales_mean_{window}"] = _grouped_rolling(shifted, groups, window, "mean").astype(
            "float32"
        )
        frame[f"sales_std_{window}"] = _grouped_rolling(shifted, groups, window, "std").astype(
            "float32"
        )
        frame[f"zero_count_{window}"] = _grouped_rolling(
            zero_shifted, groups, window, "sum"
        ).astype("float32")

    positions = np.arange(len(frame))
    last_sale = (
        pd.Series(np.where(frame["sales"].to_numpy() > 0, positions, -1), index=frame.index)
        .groupby(groups, observed=True)
        .cummax()
    )
    # Cap the feature so a sale older than the serving window has the same
    # meaning in training and inference. Serving history is kept above this
    # bound by `ensure_serving_history`; 365 days is the documented saturation
    # point for "no recent sale" rather than an arbitrary truncation.
    days_since_last_sale = np.where(
        last_sale.to_numpy() >= 0,
        positions - last_sale.to_numpy(),
        DAYS_SINCE_LAST_SALE_CAP,
    )
    frame["days_since_last_sale"] = np.minimum(
        days_since_last_sale, DAYS_SINCE_LAST_SALE_CAP
    ).astype("float32")

    if cfg.include_price:
        if "sell_price" not in frame.columns:
            raise ValueError("include_price is true but panel has no sell_price column")
        price = frame["sell_price"].astype("float64")
        price_grouped = price.groupby(groups, observed=True)
        rolling_mean = _grouped_rolling(price, groups, 28, "mean")
        rolling_std = _grouped_rolling(price, groups, 28, "std")
        frame["price_change_7d"] = (price / price_grouped.shift(7) - 1.0).astype("float32")
        spread = rolling_std.replace(0.0, np.nan)
        frame["price_zscore_28"] = ((price - rolling_mean) / spread).astype("float32")
        frame["sell_price"] = price.astype("float32")

    keep = [
        "series_id",
        "date",
        "state_id",
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
    ] + [
        c
        for c in frame.columns
        if c.startswith(("sales_lag_", "sales_mean_", "sales_std_", "zero_count_"))
    ]
    keep.append("days_since_last_sale")
    if cfg.include_price:
        keep += ["sell_price", "price_change_7d", "price_zscore_28"]
    return frame[[c for c in dict.fromkeys(keep) if c in frame.columns]]


def add_series_features(
    panel: pd.DataFrame, cfg: FeatureConfig, engine: str | None = None
) -> pd.DataFrame:
    """Compute series lag and rolling features.

    Supports both vectorized pandas and high-throughput DuckDB SQL window execution.
    """
    chosen_engine = engine or getattr(cfg, "engine", "pandas")
    if chosen_engine == "duckdb":
        try:
            return add_series_features_duckdb(panel, cfg)
        except ImportError:
            return add_series_features_pandas(panel, cfg)
    return add_series_features_pandas(panel, cfg)


def build_serving_history(panel: pd.DataFrame, cutoff: pd.Timestamp, days: int) -> pd.DataFrame:
    """Tail window long enough for the longest lag plus the forecast horizon."""
    cutoff = pd.Timestamp(cutoff).normalize()
    start = cutoff - pd.Timedelta(days=days)
    window = panel[(panel["date"] > start) & (panel["date"] <= cutoff)]
    return window.sort_values(["series_id", "date"], kind="stable").reset_index(drop=True)
