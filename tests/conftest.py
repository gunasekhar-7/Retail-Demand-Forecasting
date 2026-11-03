from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from demand.config import Config
from demand.data.panel import build_panel, read_calendar, read_panel
from demand.evaluation.backtest import run_backtest
from demand.evaluation.splits import walk_forward_folds
from demand.models.baseline import SeasonalNaiveForecaster
from demand.models.gbm import LgbmForecaster
from demand.registry import ensure_serving_history, save_run
from demand.synthetic import generate_synthetic_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def demo_config(tmp_path_factory: pytest.TempPathFactory) -> Config:
    root = tmp_path_factory.mktemp("demand")
    config = Config.model_validate(
        {
            "seed": 3,
            "data": {
                "raw_dir": str(root / "raw"),
                "processed_dir": str(root / "processed"),
                "reports_dir": str(root / "reports"),
                "stores": ["CA_1"],
                "depts": ["FOODS_3"],
                "synthetic": True,
            },
            "features": {
                "horizon": 7,
                "lags": [7, 14, 21],
                "rolling_windows": [7, 14],
                "origin_stride_days": 7,
                "train_window_days": 240,
                "max_training_rows": 500_000,
            },
            "model": {
                "num_boost_round": 120,
                "early_stopping_rounds": 10,
                "min_data_in_leaf": 10,
                "num_leaves": 15,
            },
            "eval": {"horizon": 7, "n_folds": 2, "step_days": 7, "bootstrap_reps": 50},
            "prophet": {"n_series": 1},
            "serve": {"artifacts_dir": str(root / "artifacts")},
            "inventory": {"lead_time_days": 1},
            "synthetic": {
                "stores": ["CA_1"],
                "items_per_store": 2,
                "days": 260,
                "start_date": "2011-01-29",
                "noise": 0.1,
            },
        }
    )
    generate_synthetic_dataset(config.synthetic, config.data.raw_dir, seed=config.seed)
    return config


@pytest.fixture(scope="session")
def panel(demo_config: Config) -> pd.DataFrame:
    build = build_panel(demo_config)
    return read_panel(build.path)


@pytest.fixture(scope="session")
def calendar(demo_config: Config) -> pd.DataFrame:
    return read_calendar(demo_config)


@pytest.fixture(scope="session")
def trained_artifact(demo_config: Config, panel: pd.DataFrame, calendar: pd.DataFrame):
    cutoff = pd.Timestamp(panel["date"].max())
    model = LgbmForecaster(demo_config.model, demo_config.features, seed=demo_config.seed)
    model.fit(panel, cutoff)
    history_days = demo_config.features.max_lag_days + demo_config.features.horizon + 10
    history = ensure_serving_history(panel, history_days)
    return save_run(model, model.spec, history, calendar, demo_config)


@pytest.fixture(scope="session")
def backtest_dir(demo_config: Config, panel: pd.DataFrame) -> Path:
    folds = walk_forward_folds(
        panel["date"].max(),
        horizon=demo_config.eval.horizon,
        n_folds=demo_config.eval.n_folds,
        step_days=demo_config.eval.step_days,
    )
    result = run_backtest(
        [
            SeasonalNaiveForecaster(),
            LgbmForecaster(demo_config.model, demo_config.features, seed=demo_config.seed),
        ],
        panel,
        folds,
        demo_config.artifacts_dir / "backtest",
        bootstrap_reps=0,
        synthetic=demo_config.data.synthetic,
    )
    return result.output_dir


@pytest.fixture
def tiny_panel() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=45, freq="D")
    rows = []
    for series_id, item_id in (("ITEM_1_CA_1", "ITEM_1"), ("ITEM_2_CA_1", "ITEM_2")):
        for index, date in enumerate(dates):
            rows.append(
                {
                    "series_id": series_id,
                    "item_id": item_id,
                    "dept_id": "FOODS_3",
                    "cat_id": "FOODS",
                    "store_id": "CA_1",
                    "state_id": "CA",
                    "date": date,
                    "sales": 10 + (index % 7),
                    "sell_price": 3.5,
                    "event_name_1": None,
                    "event_type_1": None,
                    "event_name_2": None,
                    "event_type_2": None,
                    "snap": int(date.day <= 10),
                    "snap_CA": int(date.day <= 10),
                    "snap_TX": int(date.day <= 15),
                    "snap_WI": int(date.day <= 10),
                    "wday": date.day_of_week + 1,
                    "weekday": date.day_name(),
                    "month": date.month,
                    "year": date.year,
                    "d_index": index + 1,
                }
            )
    return pd.DataFrame(rows)
