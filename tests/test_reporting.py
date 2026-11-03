from __future__ import annotations

import json
from pathlib import Path

import pytest

from demand.reporting import ReportingError, build_results_markdown, portfolio_bullets, render


def _metrics() -> dict:
    return {
        "dataset": {"n_series": 6, "n_days": 760, "last_date": "2013-02-26", "synthetic": True},
        "folds": [{"fold": 1}, {"fold": 2}],
        "models": {
            "seasonal_naive": {
                "overall": {"wape": 0.20, "mape": 0.19, "bias": -0.02},
                "mean_rmsse": 0.72,
                "fit_seconds": 0.01,
                "series": 6,
            },
            "lightgbm": {
                "overall": {"wape": 0.18, "mape": 0.21, "bias": 0.01},
                "mean_rmsse": 0.57,
                "fit_seconds": 8.0,
                "series": 6,
                "vs_baseline": {"delta": -0.02, "ci_low": -0.04, "ci_high": -0.001, "n_series": 6},
            },
        },
    }


def test_results_markdown_labels_synthetic_data() -> None:
    body = build_results_markdown(_metrics(), None)
    assert "Demo data" in body
    assert "| lightgbm |" in body
    assert "18.0%" in body


def test_portfolio_bullets_use_measured_values() -> None:
    bullets = portfolio_bullets(_metrics(), None)
    assert "20.0%" in bullets and "18.0%" in bullets
    assert "2-fold" in bullets


def test_portfolio_bullets_require_a_baseline() -> None:
    metrics = _metrics()
    del metrics["models"]["seasonal_naive"]
    with pytest.raises(ReportingError, match="seasonal_naive"):
        portfolio_bullets(metrics, None)


def test_deployable_champion_is_named_when_prophet_wins() -> None:
    metrics = _metrics()
    metrics["models"]["prophet"] = {
        "overall": {"wape": 0.17, "mape": 0.18, "bias": 0.0},
        "coverage_series": 6,
        "series": 6,
        "fit_seconds": 30.0,
    }
    body = build_results_markdown(metrics, None)
    assert "Deployable champion" in body
    assert "lightgbm" in body.split("Deployable champion")[1]
    bullets = portfolio_bullets(metrics, None)
    assert "Prophet reached 17.0%" in bullets
    assert "18.0%" in bullets


def test_render_fails_without_metrics(tmp_path: Path) -> None:
    with pytest.raises(ReportingError, match="missing artifact"):
        render(tmp_path, tmp_path / "reports")


def test_render_writes_results_and_readme_block(backtest_dir: Path, tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(
        "# Demo\n\n<!-- RESULTS:START -->\nold\n<!-- RESULTS:END -->\n", encoding="utf-8"
    )
    written = render(backtest_dir.parent, tmp_path / "reports", readme)
    assert "results" in written
    text = readme.read_text(encoding="utf-8")
    assert "seasonal_naive" in text
    assert "\nold\n" not in text


def test_inventory_block_requires_all_metrics() -> None:
    inventory = {
        "models": {
            "lightgbm": {
                "fill_rate": 0.97,
                "stockout_day_share": 0.12,
                "avg_on_hand_units": 2.7,
                "total_cost": 124.6,
            }
        }
    }
    body = build_results_markdown(_metrics(), inventory)
    assert "97.0%" in body
    broken = {"models": {"lightgbm": {"fill_rate": 0.97}}}
    with pytest.raises(ReportingError, match="stockout_day_share"):
        build_results_markdown(_metrics(), broken)


def test_json_metrics_are_serializable(backtest_dir: Path) -> None:
    payload = json.loads((backtest_dir / "metrics.json").read_text(encoding="utf-8"))
    assert payload["models"]["seasonal_naive"]["overall"]["wape"] >= 0
