"""Typed configuration with YAML overlays.

Every stage receives a `Config` instance; nothing reads environment variables or
global state directly. Paths are resolved against the current working directory,
which the CLI fixes to the repository root.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(_Model):
    raw_dir: Path = Path("data/raw/m5")
    processed_dir: Path = Path("data/processed")
    reports_dir: Path = Path("reports")
    sales_file: str = "sales_train_evaluation.csv"
    calendar_file: str = "calendar.csv"
    prices_file: str = "sell_prices.csv"
    stores: list[str] = Field(default_factory=lambda: ["CA_1", "TX_1"])
    depts: list[str] = Field(default_factory=lambda: ["FOODS_3"])
    panel_filename: str = "panel.parquet"
    synthetic: bool = False

    @field_validator("stores", "depts")
    @classmethod
    def _non_empty(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("must contain at least one id")
        return value

    @property
    def panel_path(self) -> Path:
        return self.processed_dir / self.panel_filename

    @property
    def sales_path(self) -> Path:
        return self.raw_dir / self.sales_file

    @property
    def calendar_path(self) -> Path:
        return self.raw_dir / self.calendar_file

    @property
    def prices_path(self) -> Path:
        return self.raw_dir / self.prices_file


class FeatureConfig(_Model):
    horizon: int = Field(default=28, ge=1, le=56)
    lags: list[int] = Field(default_factory=lambda: [7, 14, 28, 364])
    rolling_windows: list[int] = Field(default_factory=lambda: [7, 28, 56])
    origin_stride_days: int = Field(default=14, ge=1)
    train_window_days: int = Field(default=1095, ge=56)
    max_training_rows: int = Field(default=6_000_000, ge=1)
    include_events: bool = True
    include_snap: bool = True
    include_price: bool = True
    engine: Literal["pandas", "duckdb"] = "pandas"

    @field_validator("lags", "rolling_windows")
    @classmethod
    def _positive_and_sorted(cls, value: list[int]) -> list[int]:
        if not value or min(value) < 1:
            raise ValueError("windows must be positive day counts")
        return sorted(set(value))

    @property
    def max_lag_days(self) -> int:
        return max(self.lags)


class ModelConfig(_Model):
    objective: Literal["regression_l1", "quantile"] = "regression_l1"
    alpha: float = Field(default=0.5, gt=0, lt=1)
    learning_rate: float = Field(default=0.05, gt=0, le=1)
    num_leaves: int = Field(default=63, ge=2)
    min_data_in_leaf: int = Field(default=100, ge=1)
    feature_fraction: float = Field(default=0.8, gt=0, le=1)
    bagging_fraction: float = Field(default=0.8, gt=0, le=1)
    bagging_freq: int = Field(default=1, ge=1)
    num_boost_round: int = Field(default=2000, ge=1)
    early_stopping_rounds: int = Field(default=50, ge=1)
    seed: int = 42


class ProphetConfig(_Model):
    n_series: int = Field(default=24, ge=1)
    seasonality_mode: Literal["additive", "multiplicative"] = "multiplicative"
    country: str = "US"
    include_events: bool = True


class EvalConfig(_Model):
    horizon: int = Field(default=28, ge=1, le=56)
    n_folds: int = Field(default=4, ge=1)
    step_days: int = Field(default=28, ge=1)
    bootstrap_reps: int = Field(default=1000, ge=0)
    confidence: float = Field(default=0.95, gt=0, lt=1)


class ServeConfig(_Model):
    artifacts_dir: Path = Path("artifacts")
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)


class InventoryConfig(_Model):
    lead_time_days: int = Field(default=2, ge=1)
    review_period_days: int = Field(default=1, ge=1)
    service_level: float = Field(default=0.9, gt=0.5, lt=1)
    unit_margin_pct: float = Field(default=0.25, gt=0, lt=1)
    annual_holding_rate: float = Field(default=0.20, gt=0, lt=1)


class SyntheticConfig(_Model):
    stores: list[str] = Field(default_factory=lambda: ["CA_1", "TX_1"])
    items_per_store: int = Field(default=3, ge=1)
    days: int = Field(default=760, ge=120)
    calendar_extra_days: int = Field(default=28, ge=0)
    start_date: str = "2011-01-29"
    noise: float = Field(default=0.18, ge=0, le=1)
    promo_lift: float = 0.35
    trend_per_day: float = 0.0002


class MonitoringConfig(_Model):
    psi_warning_threshold: float = Field(default=0.1, gt=0)
    psi_alert_threshold: float = Field(default=0.25, gt=0)
    window_days: int = Field(default=28, ge=7)
    features: list[str] = Field(default_factory=lambda: ["sales", "sell_price"])


class Config(_Model):
    seed: int = 42
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    prophet: ProphetConfig = Field(default_factory=ProphetConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    serve: ServeConfig = Field(default_factory=ServeConfig)
    inventory: InventoryConfig = Field(default_factory=InventoryConfig)
    synthetic: SyntheticConfig = Field(default_factory=SyntheticConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)

    @property
    def artifacts_dir(self) -> Path:
        return self.serve.artifacts_dir


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(*paths: str | Path) -> Config:
    """Load and validate one or more YAML overlays, left to right."""
    payload: dict[str, Any] = {}
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"config not found: {path}")
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"config root must be a mapping: {path}")
        payload = _deep_merge(payload, loaded)
    return Config.model_validate(payload)
