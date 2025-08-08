#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QVM Drawdown Protection - Live Target Weights Generator (Strategy 04c)
=====================================================================
Generates daily target portfolio weights for the Vietnam market using:
- Fixed Q/V/M composite (equal weights)
- Top 20 stocks by composite score
- Drawdown-based allocation overlay (4x tighter):
  0-5%: 100%, 5-10%: 20%, 10-15%: 40%, 15-20%: 60%, 20-25%: 80%, 25%+: 100%
- Step changes only when drawdown crosses 10% thresholds to reduce churn

Outputs a CSV ready for order generation.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
import yaml
from sqlalchemy import create_engine, text

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / 'production' / 'execution' / 'outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FACTOR_WEIGHTS = {
    'quality': 0.3333,
    'value': 0.3333,
    'momentum': 0.3334,
}

# 4x tighter drawdown allocation schedule (step changes every 10%)
DRAWDOWN_STEPS = [0, -5, -10, -15, -20, -25]
ALLOCATION_STEPS = [1.00, 1.00, 0.20, 0.40, 0.60, 0.80]  # 0-5%:1.0, 5-10%:0.2, ..., 20-25%:0.8; < -25% => 1.0


def load_db_engine() -> object:
    config_path = PROJECT_ROOT / 'config' / 'database.yml'
    with open(config_path, 'r') as f:
        db_cfg = yaml.safe_load(f)['production']
    conn_str = f"mysql+pymysql://{db_cfg['username']}:{db_cfg['password']}@{db_cfg['host']}/{db_cfg['schema_name']}"
    return create_engine(conn_str, pool_pre_ping=True)


def get_latest_factor_date(engine) -> pd.Timestamp:
    q = text("""
        SELECT MAX(date) as latest_date
        FROM factor_scores_qvm
        WHERE Quality_Composite IS NOT NULL
          AND Value_Composite IS NOT NULL
          AND Momentum_Composite IS NOT NULL
    """)
    df = pd.read_sql(q, engine)
    return pd.to_datetime(df['latest_date'].iloc[0])


def get_top20_holdings(engine, factor_date: pd.Timestamp) -> pd.DataFrame:
    q = text("""
        SELECT date, ticker,
               Quality_Composite as quality,
               Value_Composite as value,
               Momentum_Composite as momentum
        FROM factor_scores_qvm
        WHERE date = :date
    """)
    df = pd.read_sql(q, engine, params={'date': factor_date})
    if df.empty:
        return df
    df['composite'] = (
        df['quality'] * FACTOR_WEIGHTS['quality'] +
        df['value'] * FACTOR_WEIGHTS['value'] +
        df['momentum'] * FACTOR_WEIGHTS['momentum']
    )
    df = df.sort_values('composite', ascending=False).head(20).reset_index(drop=True)
    df['base_weight'] = 1.0 / max(1, len(df))
    return df[['date', 'ticker', 'composite', 'base_weight']]


def compute_vnindex_drawdown(engine) -> float:
    q = text("""
        SELECT date, close
        FROM etf_history
        WHERE ticker = 'VNINDEX'
        ORDER BY date
    """)
    df = pd.read_sql(q, engine, parse_dates=['date'])
    if df.empty:
        return 0.0
    df['cum'] = df['close'].cummax()
    df['dd'] = (df['close'] / df['cum']) - 1.0
    return float(df['dd'].iloc[-1] * 100.0)  # percent


def map_drawdown_to_allocation(drawdown_pct: float) -> float:
    # Step-based schedule; <= -25% returns to 100%
    if drawdown_pct >= -5:
        return 1.00
    elif drawdown_pct >= -10:
        return 0.20
    elif drawdown_pct >= -15:
        return 0.40
    elif drawdown_pct >= -20:
        return 0.60
    elif drawdown_pct >= -25:
        return 0.80
    else:
        return 1.00


def load_last_allocation() -> float:
    files = sorted(OUTPUT_DIR.glob("*_qvm_drawdown_portfolio.csv"))
    if not files:
        return None
    try:
        last = pd.read_csv(files[-1])
        if 'allocation' in last.columns and not last['allocation'].isna().all():
            return float(last['allocation'].iloc[0])
    except Exception:
        return None
    return None


def enforce_step_change(prev_alloc: float, new_alloc: float) -> float:
    # Allocation levels allowed: 1.0, 0.8, 0.6, 0.4, 0.2
    allowed = [1.0, 0.8, 0.6, 0.4, 0.2]
    if new_alloc not in allowed:
        # snap to nearest allowed
        new_alloc = min(allowed, key=lambda x: abs(x - new_alloc))
    if prev_alloc is None:
        return new_alloc
    # Only change if it differs by at least 0.1 (10%) to reduce churn
    if abs(new_alloc - prev_alloc) >= 0.1 - 1e-6:
        return new_alloc
    return prev_alloc


def main(trade_date: str = None):
    engine = load_db_engine()
    today = pd.Timestamp(trade_date) if trade_date else pd.Timestamp(datetime.now().date())

    factor_date = get_latest_factor_date(engine)
    holdings = get_top20_holdings(engine, factor_date)
    if holdings.empty:
        print("❌ No holdings available for the latest factor date.")
        return False

    dd_pct = compute_vnindex_drawdown(engine)  # percent
    raw_alloc = map_drawdown_to_allocation(dd_pct)
    prev_alloc = load_last_allocation()
    final_alloc = enforce_step_change(prev_alloc, raw_alloc)

    holdings['target_weight'] = holdings['base_weight'] * final_alloc
    holdings['allocation'] = final_alloc
    holdings['drawdown_pct'] = dd_pct
    holdings['generation_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    out_name = f"{today.strftime('%Y%m%d')}_qvm_drawdown_portfolio.csv"
    out_path = OUTPUT_DIR / out_name
    holdings[['ticker', 'target_weight', 'allocation', 'drawdown_pct', 'generation_time']].to_csv(out_path, index=False)

    print("✅ Live target weights generated.")
    print(f"📄 Saved to: {out_path}")
    print(f"📊 Drawdown: {dd_pct:.2f}%, Allocation: {final_alloc:.0%}")
    return True


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate live target weights for QVM Drawdown strategy (04c).')
    parser.add_argument('--date', type=str, default=None, help='Trade date YYYY-MM-DD (optional)')
    args = parser.parse_args()
    main(args.date)
