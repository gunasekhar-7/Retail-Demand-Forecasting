#!/usr/bin/env python
"""Dashboard render benchmark using Streamlit's AppTest.

This is a lab proxy (script run time), not a browser LCP measurement: there is no
field data for a locally-run dashboard, so nothing here is reported as a
real-user metric.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parents[1] / "src" / "demand" / "dashboard" / "app.py"
    os.environ["DEMAND_ARTIFACTS"] = str(args.artifacts.resolve())
    timings = []
    for _ in range(args.runs):
        started = time.perf_counter()
        app = AppTest.from_file(str(app_path), default_timeout=120)
        app.run()
        timings.append((time.perf_counter() - started) * 1000)
        if app.exception:
            raise SystemExit(f"dashboard raised: {app.exception}")
    payload = {
        "runs": args.runs,
        "first_run_ms": round(timings[0], 1),
        "best_ms": round(min(timings), 1),
        "median_ms": round(sorted(timings)[len(timings) // 2], 1),
    }
    print(json.dumps(payload, indent=2))
    if args.out:
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
