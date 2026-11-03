"""LightGBM direct multi-horizon model.

One model predicts every horizon 1..28 by consuming `horizon` as a feature. The
training frame stacks (series, origin, horizon) rows, so the model learns
"28 days out is noisier" instead of compounding a one-step recursion. Early
stopping uses an inner temporal holdout carved from the training origins.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from demand.config import FeatureConfig, ModelConfig
from demand.features.matrix import build_inference_features, build_training_matrix
from demand.features.spec import FeatureSpec
from demand.models.base import ModelError, validate_forecast


class LgbmForecaster:
    def __init__(
        self,
        model_cfg: ModelConfig | None = None,
        feature_cfg: FeatureConfig | None = None,
        seed: int = 42,
    ) -> None:
        self.model_cfg = model_cfg or ModelConfig()
        self.feature_cfg = feature_cfg or FeatureConfig()
        self.seed = seed
        self._booster: lgb.Booster | None = None
        self._spec: FeatureSpec | None = None
        self._best_iteration = 0
        self._train_rows = 0
        self._validation: dict[str, float] = {}

        if getattr(self.model_cfg, "objective", "regression_l1") == "quantile":
            pct = round(self.model_cfg.alpha * 100)
            self.name = f"lightgbm_p{pct}"
        else:
            self.name = "lightgbm"

    @property
    def spec(self) -> FeatureSpec:
        if self._spec is None:
            raise ModelError("model has no feature spec; call fit() or load() first")
        return self._spec

    def _params(self) -> dict[str, Any]:
        cfg = self.model_cfg
        objective = getattr(cfg, "objective", "regression_l1")
        metric = "quantile" if objective == "quantile" else "mae"
        params: dict[str, Any] = {
            "objective": objective,
            "metric": metric,
            "learning_rate": cfg.learning_rate,
            "num_leaves": cfg.num_leaves,
            "min_data_in_leaf": cfg.min_data_in_leaf,
            "feature_fraction": cfg.feature_fraction,
            "bagging_fraction": cfg.bagging_fraction,
            "bagging_freq": cfg.bagging_freq,
            "verbosity": -1,
            "seed": self.seed,
            "deterministic": True,
            "force_row_wise": True,
            "num_threads": 0,
        }
        if objective == "quantile":
            params["alpha"] = getattr(cfg, "alpha", 0.5)
        return params

    def fit(self, panel: pd.DataFrame, cutoff: pd.Timestamp) -> None:
        cutoff = pd.Timestamp(cutoff).normalize()
        matrix = build_training_matrix(panel, self.feature_cfg, cutoff)
        meta = matrix.meta
        split_date = cutoff - pd.Timedelta(days=self.feature_cfg.horizon)
        train_mask = (meta["target_date"] <= split_date).to_numpy()
        valid_mask = ~train_mask
        if train_mask.sum() == 0:
            raise ModelError("no training rows before the inner validation split")

        categorical = list(matrix.spec.categorical)
        train_set = lgb.Dataset(
            matrix.X[train_mask],
            label=matrix.y[train_mask],
            categorical_feature=categorical,
            free_raw_data=False,
        )
        valid_sets: list[lgb.Dataset] = [train_set]
        valid_names = ["train"]
        callbacks: list[Any] = [lgb.log_evaluation(period=0)]
        if valid_mask.sum() > 0:
            valid_set = lgb.Dataset(
                matrix.X[valid_mask],
                label=matrix.y[valid_mask],
                reference=train_set,
                free_raw_data=False,
            )
            valid_sets.append(valid_set)
            valid_names.append("valid")
            callbacks.append(
                lgb.early_stopping(self.model_cfg.early_stopping_rounds, verbose=False)
            )

        booster = lgb.train(
            self._params(),
            train_set,
            num_boost_round=self.model_cfg.num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
        self._booster = booster
        self._spec = matrix.spec
        self._best_iteration = int(booster.best_iteration or booster.current_iteration() or 0)
        self._train_rows = len(matrix.X)
        self._validation = {}
        for name, scores in booster.best_score.items():
            self._validation[name] = float(next(iter(scores.values())))
        self._validation["valid_rows"] = float(valid_mask.sum())

    def forecast(
        self,
        history: pd.DataFrame,
        calendar: pd.DataFrame,
        origin: pd.Timestamp,
        horizon: int,
    ) -> pd.DataFrame:
        if self._booster is None or self._spec is None:
            raise ModelError("LightGBM model is not fitted")
        inference = build_inference_features(
            history, calendar, self._spec, pd.Timestamp(origin), horizon
        )
        raw = self._booster.predict(
            inference.X, num_iteration=self._best_iteration if self._best_iteration > 0 else None
        )
        frame = inference.meta.assign(yhat=np.clip(np.asarray(raw, dtype="float64"), 0.0, None))
        return validate_forecast(frame, horizon)

    def predict_matrix(self, features: pd.DataFrame) -> np.ndarray:
        if self._booster is None:
            raise ModelError("LightGBM model is not fitted")
        return np.asarray(self._booster.predict(features), dtype="float64")

    def save(self, path: Path) -> None:
        if self._booster is None:
            raise ModelError("cannot save an unfitted model")
        path.mkdir(parents=True, exist_ok=True)
        self._booster.save_model(str(path / "model.txt"))
        payload = {
            "kind": self.name,
            "best_iteration": self._best_iteration,
            "train_rows": self._train_rows,
            "validation": self._validation,
            "model_config": self.model_cfg.model_dump(mode="json"),
            "feature_config": self.feature_cfg.model_dump(mode="json"),
            "seed": self.seed,
        }
        (path / "model.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path, spec: FeatureSpec) -> LgbmForecaster:
        payload = json.loads((path / "model.json").read_text(encoding="utf-8"))
        model = cls(
            model_cfg=ModelConfig.model_validate(payload["model_config"]),
            feature_cfg=FeatureConfig.model_validate(payload["feature_config"]),
            seed=int(payload.get("seed", 42)),
        )
        model._booster = lgb.Booster(model_file=str(path / "model.txt"))
        model._spec = spec
        model._best_iteration = int(payload.get("best_iteration", 0))
        model._train_rows = int(payload.get("train_rows", 0))
        model._validation = dict(payload.get("validation", {}))
        return model

    def metadata(self) -> dict[str, Any]:
        return {
            "kind": self.name,
            "best_iteration": self._best_iteration,
            "train_rows": self._train_rows,
            "validation": self._validation,
            "n_features": len(self._spec.columns) if self._spec else 0,
            "params": self._params(),
        }
