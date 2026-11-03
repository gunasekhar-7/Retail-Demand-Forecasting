from __future__ import annotations

from pathlib import Path

import pandas as pd

from demand.config import Config, ModelConfig
from demand.models.gbm import LgbmForecaster


def test_lgbm_quantile_objective_and_naming(
    demo_config: Config, panel: pd.DataFrame, calendar: pd.DataFrame
) -> None:
    model_cfg = ModelConfig(
        objective="quantile",
        alpha=0.9,
        num_boost_round=20,
        early_stopping_rounds=5,
        min_data_in_leaf=5,
        num_leaves=10,
    )
    forecaster = LgbmForecaster(model_cfg, demo_config.features, seed=42)
    assert forecaster.name == "lightgbm_p90"

    cutoff = pd.Timestamp(panel["date"].max())
    forecaster.fit(panel, cutoff)

    preds = forecaster.forecast(panel, calendar, cutoff, horizon=demo_config.features.horizon)

    assert "yhat" in preds.columns
    assert (preds["yhat"] >= 0).all()
    assert len(preds) == panel["series_id"].nunique() * demo_config.features.horizon


def test_lgbm_quantile_save_and_load(
    tmp_path: Path, demo_config: Config, panel: pd.DataFrame
) -> None:
    model_cfg = ModelConfig(
        objective="quantile",
        alpha=0.75,
        num_boost_round=20,
        early_stopping_rounds=5,
        min_data_in_leaf=5,
        num_leaves=10,
    )
    forecaster = LgbmForecaster(model_cfg, demo_config.features, seed=42)
    assert forecaster.name == "lightgbm_p75"

    cutoff = pd.Timestamp(panel["date"].max())
    forecaster.fit(panel, cutoff)
    forecaster.save(tmp_path)

    loaded = LgbmForecaster.load(tmp_path, forecaster.spec)
    assert loaded.name == "lightgbm_p75"
    assert loaded.model_cfg.objective == "quantile"
    assert loaded.model_cfg.alpha == 0.75
