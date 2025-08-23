#!/usr/bin/env python3
"""
Metrics CLI for artifacts produced by QVM v2.2.1 flat vectorized runner.

Functions:
- Alignment report (daily and monthly sample window)
- Rolling beta (monthly, default 6-month window)

Usage:
  python production/analytics/metrics_cli.py --artifacts <dir> --window 6

If --artifacts is omitted, the CLI uses $QVM_ARTIFACTS_DIR or the latest under artifacts/qvm_v221_flat_vectorized.
"""
from __future__ import annotations

import argparse
import sys
from typing import Optional

import pandas as pd

from production.analytics.utils import (
    resolve_artifacts_dir,
    load_returns_series,
    to_monthly_compounded,
    compute_rolling_beta,
)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="QVM Artifacts Metrics CLI")
    parser.add_argument("--artifacts", type=str, default=None, help="Artifacts directory (optional)")
    parser.add_argument("--window", type=int, default=6, help="Rolling beta window length in months")
    args = parser.parse_args(argv)

    artifacts_dir = resolve_artifacts_dir(args.artifacts)

    # Load daily series and derive monthly series
    nr_daily, bm_daily = load_returns_series(artifacts_dir)
    nr_monthly = to_monthly_compounded(nr_daily)
    bm_monthly = to_monthly_compounded(bm_daily)
    idxm = nr_monthly.index.intersection(bm_monthly.index)
    nr_monthly = nr_monthly.loc[idxm]
    bm_monthly = bm_monthly.loc[idxm]

    # Alignment report
    print(f"Artifacts: {artifacts_dir}")
    if not nr_daily.empty:
        print(f"Daily aligned days: {len(nr_daily)} | {nr_daily.index.min().date()} → {nr_daily.index.max().date()}")
    else:
        print("Daily aligned days: 0")
    if not nr_monthly.empty:
        print(f"Monthly aligned months: {len(nr_monthly)} | {nr_monthly.index.min().date()} → {nr_monthly.index.max().date()}")
    else:
        print("Monthly aligned months: 0")

    # Rolling beta on monthly
    rb = compute_rolling_beta(nr_monthly, bm_monthly, window=int(args.window))
    if rb.empty:
        print("Rolling beta unavailable (insufficient data)")
        return 0
    print(
        "Rolling beta (monthly) | mean=%.3f p10=%.3f p90=%.3f" % (
            float(rb.mean()), float(rb.quantile(0.10)), float(rb.quantile(0.90))
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())


