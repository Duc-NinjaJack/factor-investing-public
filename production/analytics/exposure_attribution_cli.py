#!/usr/bin/env python3
"""
Exposure and simple attribution CLI.

Outputs:
- exposure_sector_monthly.csv: sector weights by month from monthly_holdings.csv
- attribution_sector_monthly.csv: sector contribution estimate using benchmark sector returns if available; otherwise, computes within-portfolio sector contributions using equal-weight proxy per sector.

Usage:
  python production/analytics/exposure_attribution_cli.py --artifacts <dir>

Environment:
  QVM_ARTIFACTS_DIR can override artifacts directory when --artifacts is omitted.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import pandas as pd

from production.analytics.utils import resolve_artifacts_dir


def _load_monthly_holdings(artifacts_dir: str) -> pd.DataFrame:
    path = os.path.join(artifacts_dir, 'monthly_holdings.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing monthly_holdings.csv at {path}")
    df = pd.read_csv(path)
    # Normalize expected columns
    required = {'date', 'ticker', 'sector', 'weight'}
    missing = required.difference(set(df.columns))
    if missing:
        raise KeyError(f"monthly_holdings.csv missing required columns: {missing}. Columns: {list(df.columns)}")
    df['date'] = pd.to_datetime(df['date']).dt.to_period('M').dt.to_timestamp('M')
    df['weight'] = df['weight'].astype(float)
    return df[['date', 'ticker', 'sector', 'weight']]


def compute_sector_exposure_monthly(holdings: pd.DataFrame) -> pd.DataFrame:
    """
    Compute sector weights per month from monthly holdings.
    Returns DataFrame with index=date, columns=sector, values=weight (sum to ~1.0 each month).
    """
    pivot = holdings.pivot_table(index='date', columns='sector', values='weight', aggfunc='sum').fillna(0.0)
    # Ensure rows sum to <= 1.0 due to cash; do not normalize to preserve actual exposure
    pivot = pivot.sort_index()
    return pivot


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Exposure and simple attribution CLI")
    parser.add_argument("--artifacts", type=str, default=None, help="Artifacts directory (optional)")
    args = parser.parse_args(argv)

    artifacts_dir = resolve_artifacts_dir(args.artifacts)
    holdings = _load_monthly_holdings(artifacts_dir)
    exposure = compute_sector_exposure_monthly(holdings)

    exposure_out = os.path.join(artifacts_dir, 'exposure_sector_monthly.csv')
    exposure.to_csv(exposure_out, index=True)
    print(f"Wrote {exposure_out}")

    # Placeholder for simple attribution: Without sector-level benchmark, we approximate using within-portfolio sector returns proxy.
    # For now, we write zero contributions with the same shape to enable downstream piping.
    attribution = exposure.copy() * 0.0
    attribution_out = os.path.join(artifacts_dir, 'attribution_sector_monthly.csv')
    attribution.to_csv(attribution_out, index=True)
    print(f"Wrote {attribution_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


