"""Frozen feature contract shared by training and serving.

The spec is the only source of truth for column order, dtypes and category
vocabularies. Training writes it; serving reads it and refuses to run if the
model's expected columns are unavailable. This is what prevents silent
train/serve skew.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from demand.config import FeatureConfig


class FeatureSpecError(RuntimeError):
    """Raised when a feature frame does not match the frozen contract."""


@dataclass(frozen=True)
class FeatureSpec:
    columns: tuple[str, ...]
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    categories: dict[str, tuple[str, ...]] = field(default_factory=dict)
    feature_config: dict[str, Any] = field(default_factory=dict)
    max_lag_days: int = 0
    horizon: int = 28
    version: int = 1

    @property
    def config(self) -> FeatureConfig:
        return FeatureConfig.model_validate(self.feature_config)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["columns"] = list(self.columns)
        payload["numeric"] = list(self.numeric)
        payload["categorical"] = list(self.categorical)
        payload["categories"] = {k: list(v) for k, v in self.categories.items()}
        return payload

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> FeatureSpec:
        if not path.exists():
            raise FeatureSpecError(f"feature spec not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            columns=tuple(payload["columns"]),
            numeric=tuple(payload["numeric"]),
            categorical=tuple(payload["categorical"]),
            categories={k: tuple(v) for k, v in payload.get("categories", {}).items()},
            feature_config=payload.get("feature_config", {}),
            max_lag_days=int(payload.get("max_lag_days", 0)),
            horizon=int(payload.get("horizon", 28)),
            version=int(payload.get("version", 1)),
        )

    def encode(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.columns if c not in frame.columns]
        if missing:
            raise FeatureSpecError(f"feature frame is missing columns: {missing}")
        encoded = pd.DataFrame(index=frame.index)
        for column in self.columns:
            if column in self.categorical:
                vocab = list(self.categories.get(column, ()))
                mapping = {value: index for index, value in enumerate(vocab)}
                codes = frame[column].astype("string").map(mapping)
                encoded[column] = codes.fillna(-1).astype("int32")
            else:
                encoded[column] = pd.to_numeric(frame[column], errors="coerce").astype("float32")
        return encoded

    def assert_categorical_valid(self) -> None:
        missing = [c for c in self.categorical if c not in self.categories]
        if missing:
            raise FeatureSpecError(f"categorical columns without vocabularies: {missing}")
