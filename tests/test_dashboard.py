from __future__ import annotations

from pathlib import Path

import pytest

from demand.dashboard.data_access import (
    error_histogram,
    fold_table,
    horizon_curve,
    kpi_table,
    load_data,
    series_frame,
    series_summary,
    worst_series,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def dashboard_data(backtest_dir: Path):
    return load_data(backtest_dir.parent)


def test_load_data_reads_predictions(dashboard_data) -> None:
    assert not dashboard_data.predictions.empty
    assert dashboard_data.models == ["lightgbm", "seasonal_naive"]
    assert dashboard_data.synthetic is True


def test_kpi_table_orders_by_wape(dashboard_data) -> None:
    table = kpi_table(dashboard_data.metrics)
    assert list(table.columns)[:4] == ["model", "wape", "mape", "bias"]
    assert table["wape"].is_monotonic_increasing


def test_horizon_curve_is_per_horizon(dashboard_data) -> None:
    curve = horizon_curve(dashboard_data.predictions)
    assert list(curve.index) == sorted(curve.index)
    assert "seasonal_naive" in curve.columns
    assert not curve["seasonal_naive"].isna().any()


def test_series_helpers(dashboard_data) -> None:
    summary = series_summary(dashboard_data.predictions)
    assert len(summary) == 4  # two series x two models
    assert summary["wape"].between(0, 5).all()
    frame = series_frame(dashboard_data.predictions, summary["series_id"].iloc[0])
    assert len(frame) == 28  # 7 days x 2 models
    ranked = worst_series(dashboard_data.predictions, "seasonal_naive", n=1)
    assert len(ranked) == 1
    histogram = error_histogram(dashboard_data.predictions, "seasonal_naive", bins=10)
    naive_rows = (dashboard_data.predictions["model"] == "seasonal_naive").sum()
    assert histogram["count"].sum() == naive_rows


def test_fold_table(dashboard_data) -> None:
    folds = fold_table(dashboard_data.metrics)
    assert len(folds) == 4  # two models x two folds
    assert {"model", "fold", "wape"}.issubset(folds.columns)


def test_dashboard_renders_without_exception(
    backtest_dir: Path, demo_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("DEMAND_ARTIFACTS", str(demo_config.artifacts_dir))
    app = AppTest.from_file(
        str(ROOT / "src" / "demand" / "dashboard" / "app.py"), default_timeout=90
    )
    app.run()
    assert not app.exception
    assert len(app.title) == 1
    labels = [metric.label for metric in app.metric]
    assert "Best deployable WAPE" in labels
