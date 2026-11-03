#!/usr/bin/env python
"""Measure serving latency and dashboard render time, then write reports/perf.md.

This starts the real API in a subprocess, waits for /health, measures warm
requests and process RSS, stops it, and runs the dashboard headlessly. Output is
lab evidence on one machine, not field data.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _first_series(artifacts: Path) -> str:
    pointer = json.loads((artifacts / "latest.json").read_text(encoding="utf-8"))
    run_dir = Path(pointer["path"])
    if not run_dir.is_absolute():
        run_dir = artifacts / "runs" / pointer["run_id"]
    history = pd.read_parquet(run_dir / "serving_history.parquet", columns=["series_id"])
    return str(history["series_id"].iloc[0])


def _wait_for_health(url: str, timeout: float = 60.0) -> float:
    started = time.perf_counter()
    with httpx.Client(timeout=1.0) as client:
        while time.perf_counter() - started < timeout:
            try:
                response = client.get(f"{url}/health")
                if response.status_code == 200:
                    return time.perf_counter() - started
            except httpx.HTTPError:
                pass
            time.sleep(0.2)
    raise RuntimeError(f"API did not become healthy within {timeout:.0f}s")


def _percentile(ordered: list[float], fraction: float) -> float:
    index = min(len(ordered) - 1, round(fraction * (len(ordered) - 1)))
    return ordered[index]


def measure_api(artifacts: Path, port: int, requests: int) -> dict[str, float]:
    series_id = _first_series(artifacts)
    env = dict(os.environ, DEMAND_ARTIFACTS=str(artifacts.resolve()))
    url = f"http://127.0.0.1:{port}"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "demand.service.app:create_app",
            "--factory",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        cold_start = _wait_for_health(url)
        samples = []
        with httpx.Client(base_url=url, timeout=10.0) as client:
            client.post(
                "/forecast", json={"series_id": series_id, "horizon_days": 28}
            ).raise_for_status()
            for _ in range(requests):
                started = time.perf_counter()
                response = client.post(
                    "/forecast", json={"series_id": series_id, "horizon_days": 28}
                )
                response.raise_for_status()
                samples.append((time.perf_counter() - started) * 1000)
            health = client.get("/health").json()
        rss_mb = 0.0
        try:
            import psutil

            root = psutil.Process(process.pid)
            rss_mb = sum(p.memory_info().rss for p in [root, *root.children(recursive=True)]) / (
                1024 * 1024
            )
        except Exception:  # pragma: no cover - psutil is a dev extra
            pass
        ordered = sorted(samples)
        return {
            "cold_start_s": cold_start,
            "p50_ms": statistics.median(samples),
            "p95_ms": _percentile(ordered, 0.95),
            "p99_ms": _percentile(ordered, 0.99),
            "mean_ms": statistics.fmean(samples),
            "requests": float(requests),
            "rss_mb": rss_mb,
            "series": float(health.get("series", 0)),
            "horizon": float(health.get("horizon", 0)),
        }
    finally:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:  # pragma: no cover
            process.kill()


def measure_dashboard(artifacts: Path, runs: int) -> dict[str, float]:
    from streamlit.testing.v1 import AppTest

    os.environ["DEMAND_ARTIFACTS"] = str(artifacts.resolve())
    app_path = ROOT / "src" / "demand" / "dashboard" / "app.py"
    timings = []
    for _ in range(runs):
        started = time.perf_counter()
        app = AppTest.from_file(str(app_path), default_timeout=120)
        app.run()
        timings.append((time.perf_counter() - started) * 1000)
        if app.exception:
            raise RuntimeError(f"dashboard raised: {app.exception}")
    return {"first_ms": timings[0], "median_ms": statistics.median(timings), "runs": float(runs)}


def _status(measured: float, budget: float) -> str:
    return "pass" if measured <= budget else "miss"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    api = measure_api(args.artifacts, args.port, args.requests)
    dashboard = measure_dashboard(args.artifacts, args.runs)
    args.reports.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Performance (lab measurement)",
        "",
        f"Measured {datetime.now(UTC):%Y-%m-%d %H:%M UTC} on {platform.system()} "
        f"{platform.release()}, Python {platform.python_version()}, "
        f"{os.cpu_count()} logical CPUs.",
        "",
        "No field/RUM data exists for a locally-run demo, so these are lab numbers only. "
        "They are not reported as real-user performance.",
        "",
        "| Metric | Budget | Measured | Status |",
        "|---|---|---|---|",
        f"| API cold start (import + load + batch score) | < 5 s | "
        f"{api['cold_start_s']:.2f} s | {_status(api['cold_start_s'], 5.0)} |",
        f"| `POST /forecast` p95, warm ({int(api['requests'])} requests) | < 150 ms | "
        f"{api['p95_ms']:.1f} ms | {_status(api['p95_ms'], 150.0)} |",
        f"| `POST /forecast` p50, warm | - | {api['p50_ms']:.1f} ms | - |",
        f"| `POST /forecast` p99, warm | - | {api['p99_ms']:.1f} ms | - |",
        *(
            [
                f"| API RSS after benchmark | < 1024 MB | {api['rss_mb']:.0f} MB | "
                f"{_status(api['rss_mb'], 1024.0)} |"
            ]
            if api["rss_mb"] > 20
            else ["| API RSS after benchmark | < 1024 MB | not measured reliably | - |"]
        ),
        f"| Dashboard first render (AppTest) | < 3 s | {dashboard['first_ms'] / 1000:.2f} s | "
        f"{_status(dashboard['first_ms'], 3000.0)} |",
        f"| Dashboard warm render (median of {int(dashboard['runs'])}) | - | "
        f"{dashboard['median_ms'] / 1000:.2f} s | - |",
        "",
        f"Served series: {int(api['series'])}; horizon: {int(api['horizon'])} days.",
        "",
        "Method: the API is started as a real uvicorn subprocess with the registered "
        "artifact, polled until `/health` is 200, warmed with one request, then measured "
        "over sequential requests. Dashboard time is Streamlit AppTest script-run time, a "
        "lab proxy for first render rather than browser LCP.",
        "",
    ]
    path = args.reports / "perf.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"api": api, "dashboard": dashboard, "report": str(path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
