from __future__ import annotations

import pandas as pd
import pytest

from demand.config import FeatureConfig
from demand.features.lags import (
    add_series_features,
    add_series_features_duckdb,
    add_series_features_pandas,
)


def test_duckdb_and_pandas_feature_engines_have_exact_parity(tiny_panel: pd.DataFrame) -> None:
    cfg = FeatureConfig(lags=[7, 14], rolling_windows=[7, 28], include_price=True)
    pandas_df = add_series_features_pandas(tiny_panel, cfg)
    duckdb_df = add_series_features_duckdb(tiny_panel, cfg)

    assert list(pandas_df.columns) == list(duckdb_df.columns)
    for col in pandas_df.columns:
        duckdb_df[col] = duckdb_df[col].astype(pandas_df[col].dtype)

    pd.testing.assert_frame_equal(pandas_df, duckdb_df[pandas_df.columns])


def test_engine_dispatcher_uses_duckdb_when_configured(tiny_panel: pd.DataFrame) -> None:
    cfg = FeatureConfig(lags=[7], rolling_windows=[7], engine="duckdb")
    df = add_series_features(tiny_panel, cfg)
    assert "sales_lag_7" in df.columns
    assert "sales_mean_7" in df.columns


def test_duckdb_missing_price_column_raises_error(tiny_panel: pd.DataFrame) -> None:
    cfg = FeatureConfig(lags=[7], rolling_windows=[7], include_price=True)
    no_price = tiny_panel.drop(columns=["sell_price"])
    with pytest.raises(ValueError, match="sell_price"):
        add_series_features_duckdb(no_price, cfg)
