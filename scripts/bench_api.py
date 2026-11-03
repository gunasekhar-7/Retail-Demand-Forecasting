#!/usr/bin/env python
"""Latency benchmark for POST /forecast.

Lab measurement on one machine: start the API, then run
`python scripts/bench_api.py --series <id> --requests 200`.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import httpx


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--series", required=True)
    parser.add_argument("--horizon", type=int, default=28)
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--api-key", default=None, help="sent as X-API-Key when set")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    with httpx.Client(base_url=args.url, timeout=10.0) as client:
        health = client.get("/health")
        health.raise_for_status()
        warmup = client.post(
            "/forecast",
            json={"series_id": args.series, "horizon_days": args.horizon},
            headers=headers,
        )
        warmup.raise_for_status()

        samples: list[float] = []
        for _ in range(args.requests):
            started = time.perf_counter()
            response = client.post(
                "/forecast",
                json={"series_id": args.series, "horizon_days": args.horizon},
                headers=headers,
            )
            samples.append((time.perf_counter() - started) * 1000)
            response.raise_for_status()

    ordered = sorted(samples)

    def percentile(p: float) -> float:
        index = min(len(ordered) - 1, round(p * (len(ordered) - 1)))
        return ordered[index]

    payload = {
        "url": args.url,
        "series": args.series,
        "requests": args.requests,
        "p50_ms": round(statistics.median(samples), 2),
        "p95_ms": round(percentile(0.95), 2),
        "p99_ms": round(percentile(0.99), 2),
        "mean_ms": round(statistics.fmean(samples), 2),
        "health": health.json(),
    }
    print(json.dumps(payload, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
