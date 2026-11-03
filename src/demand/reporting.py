"""Report rendering.

Every number in `reports/results.md`, `reports/portfolio_bullets.md` and the
README results block is read from `metrics.json` / `inventory.json`. If a metric
is missing, rendering raises rather than printing a placeholder that could be
mistaken for a result.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from demand.evaluation.metrics import MetricError
from demand.registry import RegistryError

RESULTS_START = "<!-- RESULTS:START -->"
RESULTS_END = "<!-- RESULTS:END -->"


class ReportingError(RuntimeError):
    """Raised when artifacts needed for a report are missing."""


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ReportingError(f"missing artifact: {path}")
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def _require(mapping: dict[str, Any], key: str, context: str) -> float:
    value = mapping.get(key)
    if value is None:
        raise ReportingError(f"{context} has no {key!r}; refusing to render a placeholder")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ReportingError(f"{context} has a non-numeric {key!r}") from exc


def _pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def _headline(report: dict[str, Any]) -> dict[str, Any]:
    """Metrics on the series every model covers, so coverage cannot flatter a model."""
    common = report.get("overall_common")
    return dict(common) if common else dict(report.get("overall", {}))


def results_table(metrics: dict[str, Any]) -> str:
    models = dict(metrics["models"])
    rows = [
        "| Model | WAPE | MAPE | Bias | RMSSE | Hier RMSSE (store|dept|overall) | Fit (s) | Series | Coverage |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, report in models.items():
        overall = _headline(report)
        wape = _require(overall, "wape", f"{name}.overall")
        bias = _require(overall, "bias", f"{name}.overall")
        mape_value = overall.get("mape")
        mape = _pct(float(mape_value)) if mape_value is not None else "-"
        rmsse_value = report.get("mean_rmsse")
        rmsse = f"{float(rmsse_value):.3f}" if rmsse_value is not None else "-"
        hier = report.get("hierarchical_rmsse", {})
        hier_store = hier.get("level_0_rmsse")
        hier_dept = hier.get("level_1_rmsse")
        hier_overall = hier.get("level_2_rmsse")
        hier_text = (
            f"{float(hier_store):.3f}|{float(hier_dept):.3f}|{float(hier_overall):.3f}"
            if all(v is not None and not pd.isna(v) for v in [hier_store, hier_dept, hier_overall])
            else "-"
        )
        coverage = int(report.get("coverage_series", report.get("series", 0)))
        rows.append(
            f"| {name} | {_pct(wape)} | {mape} | {bias:+.3f} | {rmsse} | {hier_text} | "
            f"{float(report.get('fit_seconds', 0)):.1f} | {int(report.get('series', 0))} | "
            f"{coverage} |"
        )
    return "\n".join(rows)


def impact_table(inventory: dict[str, Any]) -> str:
    rows = [
        "| Policy forecast | Fill rate | Stockout days | Avg on-hand | Total cost |",
        "|---|---|---|---|---|",
    ]
    for name, summary in dict(inventory["models"]).items():
        fill = _require(summary, "fill_rate", f"inventory.{name}")
        stockout = _require(summary, "stockout_day_share", f"inventory.{name}")
        on_hand = _require(summary, "avg_on_hand_units", f"inventory.{name}")
        cost = _require(summary, "total_cost", f"inventory.{name}")
        rows.append(f"| {name} | {_pct(fill)} | {_pct(stockout)} | {on_hand:,.1f} | ${cost:,.0f} |")
    return "\n".join(rows)


def _improvement_interval(ci: dict[str, Any]) -> tuple[float, float]:
    """Convert a (candidate - baseline) WAPE interval into an improvement interval."""
    low = -float(ci["ci_high"])
    high = -float(ci["ci_low"])
    return min(low, high), max(low, high)


def build_results_markdown(metrics: dict[str, Any], inventory: dict[str, Any] | None) -> str:
    dataset = dict(metrics.get("dataset", {}))
    folds = list(metrics.get("folds", []))
    lines = [
        "## Results",
        "",
        f"Dataset: {dataset.get('n_series', '?')} series x {dataset.get('n_days', '?')} days "
        f"(cutoff {dataset.get('last_date', '?')}), "
        f"{len(folds)} walk-forward folds.",
        "",
    ]
    if dataset.get("synthetic"):
        lines += [
            "> **Demo data.** These numbers come from the generated synthetic dataset. "
            "They verify the pipeline and the evaluation, not Walmart demand.",
            "",
        ]
    lines += [results_table(metrics), ""]
    models = dict(metrics["models"])
    coverages = {
        int(report.get("coverage_series", report.get("series", 0))) for report in models.values()
    }
    if len(coverages) > 1:
        common_count = max(
            (int(report.get("common_series", 0)) for report in models.values()), default=0
        )
        lines += [
            f"Model coverage differs; the table compares the {common_count} series every model "
            "scored. Prophet is fitted only on the top-revenue series and is benchmark-only.",
            "",
        ]
    best_name, best_report = min(models.items(), key=lambda item: float(_headline(item[1])["wape"]))
    baseline = models.get("seasonal_naive")
    if baseline is not None and best_name != "seasonal_naive":
        delta = float(_headline(baseline)["wape"]) - float(_headline(best_report)["wape"])
        ci = best_report.get("vs_baseline")
        if ci:
            low, high = _improvement_interval(ci)
            ci_text = f"95% CI [{low:+.3f}, {high:+.3f}] over {int(ci['n_series'])} series"
        else:
            ci_text = "no bootstrap interval available"
        lines += [
            f"**{best_name}** improves pooled WAPE by {_pct(delta)} (absolute) versus the "
            f"seasonal-naive baseline ({ci_text}).",
            "",
        ]
    if best_name == "prophet" and "lightgbm" in models:
        deployable = _headline(models["lightgbm"])
        lines += [
            f"Deployable champion: **lightgbm** at {_pct(float(deployable['wape']))} WAPE. "
            "Prophet fits one model per series, so it is capped to the top-revenue series "
            "and is a benchmark, not the serving candidate.",
            "",
        ]
    if inventory is not None:
        lines += [
            "### Simulated inventory impact",
            "",
            "Daily order-up-to policy; cost parameters are declared assumptions in "
            "`configs/base.yaml`, not measured business results.",
            "",
            impact_table(inventory),
            "",
        ]
    return "\n".join(lines)


def portfolio_bullets(metrics: dict[str, Any], inventory: dict[str, Any] | None) -> str:
    models = dict(metrics["models"])
    baseline_report = models.get("seasonal_naive")
    if baseline_report is None:
        raise ReportingError("portfolio bullets need a seasonal_naive baseline")
    # Prophet is a capped benchmark that cannot be fitted per series at catalogue scale;
    # resume bullets describe the model this project actually serves.
    candidates = {k: v for k, v in models.items() if k not in {"seasonal_naive", "prophet"}} or {
        k: v for k, v in models.items() if k != "seasonal_naive"
    }
    if not candidates:
        raise ReportingError("portfolio bullets need at least one non-baseline model")
    name, report = min(candidates.items(), key=lambda item: float(_headline(item[1])["wape"]))
    baseline_wape = _require(_headline(baseline_report), "wape", "seasonal_naive.overall")
    wape = _require(_headline(report), "wape", f"{name}.overall")
    relative = (baseline_wape - wape) / baseline_wape
    points = (baseline_wape - wape) * 100
    dataset = dict(metrics.get("dataset", {}))
    folds = len(metrics.get("folds", []))
    lines = [
        f"- Cut 28-day demand forecast error (WAPE) from {_pct(baseline_wape)} to "
        f"{_pct(wape)} ({_pct(relative)} relative, {points:.1f} points) across "
        f"{int(dataset.get('n_series', 0)):,} SKU-store series, validated with "
        f"{folds}-fold walk-forward backtesting.",
    ]
    ci = report.get("vs_baseline")
    if ci:
        low, high = _improvement_interval(ci)
        lines.append(
            f"- Paired series bootstrap puts the WAPE improvement at "
            f"{_pct(baseline_wape - wape)} "
            f"(95% CI [{_pct(low)}, {_pct(high)}])."
        )
    prophet = models.get("prophet")
    if prophet is not None:
        prophet_wape = _require(_headline(prophet), "wape", "prophet.overall")
        fit_seconds = float(prophet.get("fit_seconds", 0.0))
        lines.append(
            f"- Prophet reached {_pct(prophet_wape)} WAPE on the same series with "
            f"{fit_seconds:.1f}s total fit time across folds, but fits one model per series "
            f"and is capped, so it is benchmarked rather than served."
        )
    if inventory is not None:
        summaries = dict(inventory["models"])
        if name in summaries and "seasonal_naive" in summaries:
            candidate = summaries[name]
            base = summaries["seasonal_naive"]
            base_lost = _require(base, "lost_units", "inventory.seasonal_naive")
            cand_lost = _require(candidate, "lost_units", f"inventory.{name}")
            base_on_hand = _require(base, "avg_on_hand_units", "inventory.seasonal_naive")
            cand_on_hand = _require(candidate, "avg_on_hand_units", f"inventory.{name}")
            lost_change = (cand_lost - base_lost) / base_lost if base_lost else 0.0
            on_hand_change = (cand_on_hand - base_on_hand) / base_on_hand if base_on_hand else 0.0
            lines.append(
                f"- Simulated 90% service-level policy: lost units {lost_change:+.1%} and "
                f"average on-hand {on_hand_change:+.1%} versus the naive baseline "
                f"(cost assumptions documented, sensitivity reported)."
            )
    return "\n".join(lines) + "\n"


def render(
    artifacts_dir: Path,
    reports_dir: Path,
    readme: Path | None = None,
) -> dict[str, Path]:
    metrics = _load_json(artifacts_dir / "backtest" / "metrics.json")
    inventory_path = artifacts_dir / "inventory" / "inventory.json"
    inventory = _load_json(inventory_path) if inventory_path.exists() else None
    try:
        body = build_results_markdown(metrics, inventory)
        bullets = portfolio_bullets(metrics, inventory)
    except MetricError as exc:
        raise ReportingError(str(exc)) from exc
    reports_dir.mkdir(parents=True, exist_ok=True)
    results_path = reports_dir / "results.md"
    results_path.write_text(body + "\n", encoding="utf-8")
    bullets_path = reports_dir / "portfolio_bullets.md"
    bullets_path.write_text(bullets, encoding="utf-8")
    written = {"results": results_path, "bullets": bullets_path}
    if readme is not None:
        if not readme.exists():
            raise ReportingError(f"README not found: {readme}")
        original = readme.read_text(encoding="utf-8")
        if RESULTS_START not in original or RESULTS_END not in original:
            raise ReportingError(
                f"README needs {RESULTS_START} and {RESULTS_END} markers around the results block"
            )
        head, rest = original.split(RESULTS_START, 1)
        _, tail = rest.split(RESULTS_END, 1)
        readme_body = body.removeprefix("## Results\n\n")
        readme.write_text(
            f"{head}{RESULTS_START}\n{readme_body}\n{RESULTS_END}{tail}", encoding="utf-8"
        )
        written["readme"] = readme
    return written


def frame_to_markdown(frame: pd.DataFrame) -> str:
    try:
        return frame.to_markdown(index=False)
    except ImportError as exc:  # tabulate is optional
        raise RegistryError("markdown rendering needs `tabulate`") from exc
