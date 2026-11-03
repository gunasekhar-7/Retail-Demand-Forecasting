from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from demand.config import Config, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_demo_overlay_merges_and_keeps_base_values() -> None:
    cfg = load_config(ROOT / "configs" / "base.yaml", ROOT / "configs" / "demo.yaml")
    assert cfg.seed == 7
    assert str(cfg.data.raw_dir).endswith("synthetic")
    assert cfg.features.horizon == 28  # inherited from base
    assert cfg.features.train_window_days == 700  # overridden by demo
    assert cfg.data.stores == ["CA_1", "TX_1"]


def test_default_config_is_valid() -> None:
    cfg = Config()
    assert cfg.features.max_lag_days == 364
    assert cfg.data.panel_path.name == "panel.parquet"


def test_unknown_key_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Config.model_validate({"features": {"unknown_flag": True}})


def test_zero_horizon_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Config.model_validate({"features": {"horizon": 0}})


def test_missing_config_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_config(ROOT / "configs" / "does-not-exist.yaml")
