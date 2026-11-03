from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from demand.config import Config
from demand.registry import load_latest, save_run


def test_backtest_writes_predictions_and_metrics(backtest_dir: Path, demo_config: Config) -> None:
    predictions = pd.read_parquet(backtest_dir / "predictions.parquet")
    metrics = json.loads((backtest_dir / "metrics.json").read_text(encoding="utf-8"))
    expected_rows = (
        len(metrics["models"])
        * 2  # series
        * demo_config.eval.horizon
        * demo_config.eval.n_folds
    )
    assert len(predictions) == expected_rows
    assert set(predictions.columns) == {
        "model",
        "fold",
        "series_id",
        "date",
        "horizon",
        "y",
        "yhat",
    }
    assert "seasonal_naive" in metrics["models"]
    assert "lightgbm" in metrics["models"]
    assert metrics["models"]["seasonal_naive"]["overall"]["wape"] >= 0
    assert len(metrics["folds"]) == demo_config.eval.n_folds
    assert (backtest_dir / "folds.json").exists()


def test_forecasts_never_include_future_rows(backtest_dir: Path) -> None:
    predictions = pd.read_parquet(backtest_dir / "predictions.parquet")
    folds = json.loads((backtest_dir / "folds.json").read_text(encoding="utf-8"))
    for fold in folds:
        subset = predictions[predictions["fold"] == fold["fold"]]
        assert subset["date"].min() >= pd.Timestamp(fold["test_start"])
        assert subset["date"].max() <= pd.Timestamp(fold["test_end"])


def test_registry_round_trip(trained_artifact, demo_config: Config) -> None:
    loaded = load_latest(demo_config.artifacts_dir)
    assert loaded.run_id == trained_artifact.run_id
    assert loaded.cutoff == trained_artifact.cutoff
    assert loaded.metadata["git_sha"]
    assert loaded.metadata["model"] == "lightgbm"
    assert (loaded.path / "feature_spec.json").exists()
    assert (loaded.path / "serving_history.parquet").exists()

    first = trained_artifact.model.forecast(
        loaded.history, loaded.calendar, loaded.cutoff, loaded.horizon
    )
    second = loaded.model.forecast(loaded.history, loaded.calendar, loaded.cutoff, loaded.horizon)
    np.testing.assert_allclose(first["yhat"].to_numpy(), second["yhat"].to_numpy(), rtol=1e-6)


def test_failed_publish_keeps_latest_on_previous_complete_run(
    trained_artifact, demo_config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    latest_path = tmp_path / "latest.json"
    published = save_run(
        trained_artifact.model,
        trained_artifact.spec,
        trained_artifact.history,
        trained_artifact.calendar,
        demo_config,
        out_dir=tmp_path,
    )
    pointer_before = latest_path.read_text(encoding="utf-8")
    loaded_before = load_latest(tmp_path)
    assert loaded_before.run_id == published.run_id
    assert (loaded_before.path / "metadata.json").exists()
    assert (loaded_before.path / "serving_history.parquet").exists()

    def fail_after_partial_write(path: Path) -> None:
        (path / "partial.txt").write_text("incomplete", encoding="utf-8")
        raise RuntimeError("simulated disk failure")

    monkeypatch.setattr(published.model, "save", fail_after_partial_write)
    with pytest.raises(RuntimeError, match="simulated disk failure"):
        save_run(
            trained_artifact.model,
            trained_artifact.spec,
            trained_artifact.history,
            trained_artifact.calendar,
            demo_config,
            out_dir=tmp_path,
        )

    assert latest_path.read_text(encoding="utf-8") == pointer_before
    loaded_after = load_latest(tmp_path)
    assert loaded_after.run_id == loaded_before.run_id
    assert (loaded_after.path / "metadata.json").exists()
