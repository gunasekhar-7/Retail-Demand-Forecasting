#!/usr/bin/env python
"""Render results tables from measured artifacts.

Usage: python scripts/render_results.py --artifacts artifacts --reports reports
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from demand.reporting import ReportingError, render


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument("--readme", type=Path, default=None)
    args = parser.parse_args()
    readme = (
        args.readme
        if args.readme is not None
        else Path(__file__).resolve().parents[1] / "README.md"
    )
    try:
        written = render(args.artifacts, args.reports, readme if readme.exists() else None)
    except ReportingError as exc:
        print(f"report failed: {exc}", file=sys.stderr)
        return 2
    for name, path in written.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
