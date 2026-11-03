"""Command line entry point.

`demand demo` runs the whole pipeline on generated M5-shaped data; the same
commands with `--config configs/base.yaml` run against downloaded competition
files.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from demand.config import Config, load_config
from demand.data.ingest import IngestReport, validate_layout
from demand.data.panel import (
    PanelBuild,
    PanelStats,
    build_panel,
    read_calendar,
    read_panel,
    validate_panel,
)
from demand.evaluation.backtest import BacktestResult, run_backtest
from demand.evaluation.inventory import compare_policies
from demand.evaluation.splits import walk_forward_folds
from demand.models.baseline import SeasonalNaiveForecaster
from demand.models.gbm import LgbmForecaster
from demand.models.prophet_model import ProphetForecaster
from demand.registry import ensure_serving_history, save_run
from demand.reporting import render
from demand.synthetic import generate_synthetic_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIGS = ["configs/base.yaml"]


def _print(message: str) -> None:
    print(message, flush=True)


def _report_ingest(report: IngestReport) -> None:
    _print(
        f"ingest ok: {report.n_series:,} series ({report.n_series_in_scope:,} in scope), "
        f"{report.n_days:,} days {report.start_date} -> {report.end_date}"
    )


def _report_panel(build: PanelBuild, stats: PanelStats) -> None:
    _print(
        f"panel ok: {build.n_rows:,} rows, {build.n_series:,} series x {build.n_days:,} days "
        f"in {build.seconds:.1f}s -> {build.path}"
    )
    _print(
        f"  zero-sales share {float(stats['zero_sales_share']):.1%}, "
        f"missing prices {float(stats['price_null_share']):.1%}"
    )


def cmd_synth(cfg: Config) -> int:
    dataset = generate_synthetic_dataset(cfg.synthetic, cfg.data.raw_dir, seed=cfg.seed)
    _print(
        f"synthetic data: {dataset.n_series} series x {dataset.n_days} days -> {dataset.raw_dir}"
    )
    return 0


def cmd_ingest(cfg: Config) -> int:
    try:
        report = validate_layout(cfg)
    except Exception as exc:
        _print(f"ingest failed: {exc}")
        _print()
        _print("To use real M5 data:")
        _print("  1. Get the M5 dataset from Kaggle: ")
        _print("     https://www.kaggle.com/c/m5-forecasting-accuracy")
        _print("  2. Accept competition rules and download")
        _print("  3. Unzip into data/raw/m5/")
        _print("     Expected files: sales_train_evaluation.csv, calendar.csv, sell_prices.csv")
        _print("  4. Adjust stores/depts in configs/base.yaml if needed")
        _print("  5. Run: uv run demand ingest")
        return 1
    _report_ingest(report)
    return 0


def cmd_panel(cfg: Config) -> int:
    _report_ingest(validate_layout(cfg))
    build = build_panel(cfg)
    stats = validate_panel(read_panel(build.path))
    _report_panel(build, stats)
    return 0


def _models(cfg: Config, with_prophet: bool = False) -> list:
    models: list = [
        SeasonalNaiveForecaster(),
        LgbmForecaster(cfg.model, cfg.features, seed=cfg.seed),
    ]
    if with_prophet:
        models.append(ProphetForecaster(cfg.prophet, horizon=cfg.eval.horizon))
    return models


def cmd_backtest(cfg: Config, with_prophet: bool = False) -> int:
    panel = read_panel(cfg.data.panel_path)
    validate_panel(panel)
    folds = walk_forward_folds(
        panel["date"].max(),
        horizon=cfg.eval.horizon,
        n_folds=cfg.eval.n_folds,
        step_days=cfg.eval.step_days,
    )
    _print(
        f"backtest: {len(folds)} folds x {cfg.eval.horizon} days, "
        f"{panel['series_id'].nunique():,} series"
    )
    result = run_backtest(
        _models(cfg, with_prophet),
        panel,
        folds,
        cfg.artifacts_dir / "backtest",
        bootstrap_reps=cfg.eval.bootstrap_reps,
        confidence=cfg.eval.confidence,
        seed=cfg.seed,
        synthetic=cfg.data.synthetic,
    )
    _print_results(result)
    return 0


def _print_results(result: BacktestResult) -> None:
    rows = []
    for name, report in result.metrics["models"].items():
        overall = report["overall"]
        h_rmsse = report.get("hierarchical_rmsse", {})
        rows.append(
            {
                "model": name,
                "wape": round(float(overall["wape"]), 4),
                "mape": round(float(overall.get("mape", float("nan"))), 4),
                "bias": round(float(overall["bias"]), 4),
                "mean_rmsse": round(float(report.get("mean_rmsse", float("nan"))), 4),
                "hier_store": round(float(h_rmsse.get("level_0_rmsse", float("nan"))), 4),
                "hier_dept": round(float(h_rmsse.get("level_1_rmsse", float("nan"))), 4),
                "hier_overall": round(float(h_rmsse.get("level_2_rmsse", float("nan"))), 4),
                "fit_s": report.get("fit_seconds"),
            }
        )
    _print(pd.DataFrame(rows).to_string(index=False))
    _print(f"artifacts -> {result.output_dir}")


def cmd_train(cfg: Config, seed: int | None = None) -> int:
    panel = read_panel(cfg.data.panel_path)
    validate_panel(panel)
    cutoff = pd.Timestamp(panel["date"].max())
    model = LgbmForecaster(cfg.model, cfg.features, seed=seed if seed is not None else cfg.seed)
    _print(f"training LightGBM up to {cutoff.date()}...")
    model.fit(panel, cutoff)
    history_days = cfg.features.max_lag_days + cfg.features.horizon + 30
    history = ensure_serving_history(panel, history_days)
    artifact = save_run(
        model,
        model.spec,
        history,
        read_calendar(cfg),
        cfg,
        extra={"config": cfg.model_dump(mode="json")},
    )
    metrics_path = cfg.artifacts_dir / "backtest" / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        model_report = metrics.get("models", {}).get("lightgbm")
        if model_report:
            artifact.metadata["backtest"] = {
                "overall": model_report.get("overall"),
                "mean_rmsse": model_report.get("mean_rmsse"),
                "vs_baseline": model_report.get("vs_baseline"),
            }
            (artifact.path / "metadata.json").write_text(
                json.dumps(artifact.metadata, indent=2) + "\n", encoding="utf-8"
            )
    _print(f"registered run {artifact.run_id} -> {artifact.path}")
    _print(f"  {len(artifact.series):,} series, horizon {artifact.horizon}, cutoff {cutoff.date()}")
    return 0


def cmd_simulate(cfg: Config) -> int:
    panel = read_panel(cfg.data.panel_path, columns=["series_id", "date", "sales", "sell_price"])
    backtest_dir = cfg.artifacts_dir / "backtest"
    predictions_path = backtest_dir / "predictions.parquet"
    if not predictions_path.exists():
        _print("no predictions; run `demand backtest` first")
        return 2
    predictions = pd.read_parquet(predictions_path)
    model_names = sorted(predictions["model"].unique().tolist())
    actual_source = predictions[predictions["model"] == model_names[0]]
    actual = actual_source[["series_id", "date", "y"]].rename(columns={"y": "sales"})
    unit_price = (
        panel.assign(revenue=panel["sales"].astype("float64") * panel["sell_price"].fillna(0.0))
        .groupby("series_id", observed=True)
        .apply(lambda d: d["revenue"].sum() / max(d["sales"].sum(), 1))
        .clip(lower=0.01)
    )
    forecasts = {}
    residuals = {}
    for name in model_names:
        frame = predictions[predictions["model"] == name][["series_id", "date", "y", "yhat"]]
        forecasts[name] = frame[["series_id", "date", "yhat"]]
        residuals[name] = (
            frame.assign(error=frame["y"] - frame["yhat"])
            .groupby("series_id", observed=True)["error"]
            .std()
            .fillna(0.0)
        )
    rows = []
    detail = []
    for name, forecast in forecasts.items():
        result = compare_policies(
            actual.rename(columns={"sales": "y"}),
            {name: forecast},
            cfg.inventory,
            unit_price,
            residuals[name],
        )
        rows.append(result)
        per_series = result.assign(model=name)
        detail.append(per_series)
    summary = pd.concat(rows, ignore_index=True)
    out_dir = cfg.artifacts_dir / "inventory"
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_frame = pd.concat(detail, ignore_index=True)
    detail_frame.to_parquet(out_dir / "inventory.parquet", index=False)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "synthetic": cfg.data.synthetic,
        "assumptions": cfg.inventory.model_dump(mode="json"),
        "models": {
            row["model"]: {k: float(v) for k, v in row.items() if k != "model"}
            for row in summary.to_dict(orient="records")
        },
    }
    (out_dir / "inventory.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _print(summary.round(3).to_string(index=False))
    _print(f"inventory simulation -> {out_dir}")
    return 0


def cmd_report(cfg: Config) -> int:
    written = render(
        cfg.artifacts_dir,
        cfg.data.reports_dir,
        readme=PROJECT_ROOT / "README.md" if (PROJECT_ROOT / "README.md").exists() else None,
    )
    for name, path in written.items():
        _print(f"{name} -> {path}")
    return 0


def cmd_drift(cfg: Config) -> int:
    from demand.monitoring.drift import run_drift_check

    panel = read_panel(cfg.data.panel_path)
    validate_panel(panel)
    cutoff = pd.Timestamp(panel["date"].max())
    out_dir = cfg.artifacts_dir / "monitoring"
    report = run_drift_check(
        panel,
        cutoff=cutoff,
        window_days=cfg.monitoring.window_days,
        features=cfg.monitoring.features,
        out_dir=out_dir,
        psi_warning=cfg.monitoring.psi_warning_threshold,
        psi_alert=cfg.monitoring.psi_alert_threshold,
    )
    status_label = report["status"].upper()
    _print(f"drift check: {status_label} (status={report['status']})")
    rows = []
    for feat, metrics in report["features"].items():
        rows.append(
            {
                "feature": feat,
                "psi": metrics["psi"],
                "wasserstein": metrics["normalized_wasserstein"],
                "status": metrics["status"],
            }
        )
    if rows:
        _print(pd.DataFrame(rows).to_string(index=False))
    _print(f"drift report -> {out_dir / 'drift.json'}")
    return 1 if report["status"] == "alert" else 0


def cmd_serve(cfg: Config, workers: int) -> int:
    import uvicorn

    os.environ["DEMAND_ARTIFACTS"] = str(Path(cfg.artifacts_dir).resolve())
    _print(
        f"serving {cfg.artifacts_dir} on http://{cfg.serve.host}:{cfg.serve.port} "
        f"({workers} worker(s))"
    )
    uvicorn.run(
        "demand.service.app:create_app",
        factory=True,
        host=cfg.serve.host,
        port=cfg.serve.port,
        workers=workers,
        log_level="info",
    )
    return 0


def cmd_dashboard(cfg: Config) -> int:
    app_path = PROJECT_ROOT / "src" / "demand" / "dashboard" / "app.py"
    env = dict(os.environ, DEMAND_ARTIFACTS=str(Path(cfg.artifacts_dir).resolve()))
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(cfg.serve.port + 1),
        "--server.address",
        cfg.serve.host,
    ]
    _print("dashboard: " + " ".join(command))
    return subprocess.call(command, env=env)


def cmd_demo(cfg: Config) -> int:
    steps = (
        ("synth", lambda: cmd_synth(cfg)),
        ("ingest", lambda: cmd_ingest(cfg)),
        ("panel", lambda: cmd_panel(cfg)),
        ("backtest", lambda: cmd_backtest(cfg, with_prophet=False)),
        ("train", lambda: cmd_train(cfg)),
        ("drift", lambda: cmd_drift(cfg)),
        ("simulate", lambda: cmd_simulate(cfg)),
        ("report", lambda: cmd_report(cfg)),
    )
    for name, step in steps:
        _print(f"\n== {name} ==")
        code = step()
        if code != 0:
            _print(f"step {name} failed with exit code {code}")
            return code
    _print(
        "\ndemo complete. Start the API with `demand --config configs/base.yaml "
        "--config configs/demo.yaml serve` and the dashboard with `... dashboard`."
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="demand", description=__doc__)
    parser.add_argument(
        "--config",
        action="append",
        default=None,
        help="YAML config path; repeat to overlay (e.g. base.yaml then demo.yaml)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("synth", help="generate the synthetic M5-shaped dataset")
    sub.add_parser("ingest", help="validate the raw dataset layout")
    sub.add_parser("panel", help="build the DuckDB long panel")
    backtest = sub.add_parser("backtest", help="run walk-forward validation")
    backtest.add_argument(
        "--with-prophet", action="store_true", help="include the Prophet challenger"
    )
    train = sub.add_parser("train", help="fit and register the LightGBM artifact")
    train.add_argument("--seed", type=int, default=None)
    sub.add_parser("drift", help="check feature and target drift (PSI and Wasserstein)")
    sub.add_parser("simulate", help="simulate the inventory policy on backtest output")
    sub.add_parser("report", help="render results tables and portfolio bullets")
    serve = sub.add_parser("serve", help="run the FastAPI service")
    serve.add_argument("--workers", type=int, default=1)
    sub.add_parser("dashboard", help="run the Streamlit dashboard")
    sub.add_parser(
        "demo",
        help="synth -> ingest -> panel -> backtest -> train -> drift -> simulate -> report",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configs = args.config if args.config else DEFAULT_CONFIGS
    cfg = load_config(*configs)
    commands = {
        "synth": cmd_synth,
        "ingest": cmd_ingest,
        "panel": cmd_panel,
        "backtest": lambda c: cmd_backtest(c, with_prophet=getattr(args, "with_prophet", False)),
        "train": lambda c: cmd_train(c, seed=getattr(args, "seed", None)),
        "drift": cmd_drift,
        "simulate": cmd_simulate,
        "report": cmd_report,
        "serve": lambda c: cmd_serve(c, workers=getattr(args, "workers", 1)),
        "dashboard": cmd_dashboard,
        "demo": cmd_demo,
    }
    return commands[args.command](cfg)


if __name__ == "__main__":
    raise SystemExit(main())
