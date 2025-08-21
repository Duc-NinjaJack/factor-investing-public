#!/usr/bin/env python3
"""
A/B Backtest for v2.2.1 Vectorized F-Score
==========================================
Compare baseline v2.2.1 (DB F-Score path) vs v2.2.1 vectorized F-Score path
over a representative period. Reports Sharpe, Max Drawdown (MDD), and Turnover.

Notes:
- Uses monthly rebalancing by default
- Equal-weighted top-N portfolio from QVM_Composite
- Simplified forward 1M return using equity_history
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy import text


def setup_logger():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger('ab_backtest_221')


def import_engine():
    prod_path = Path(__file__).parent.parent
    if str(prod_path) not in sys.path:
        sys.path.append(str(prod_path))
    from engine.qvm_engine_v2_2_1_flat import QVMEngineV221Flat
    from engine.qvm_engine_v2_2_1_flat_vectorized import install_vectorized_fscore_221
    return QVMEngineV221Flat, install_vectorized_fscore_221


def get_trading_dates(engine, start_date: str, end_date: str) -> list:
    q = text("""
        SELECT DISTINCT date FROM equity_history
        WHERE date BETWEEN :start AND :end AND volume > 0
        ORDER BY date
    """)
    df = pd.read_sql(q, engine, params={'start': start_date, 'end': end_date}, parse_dates=['date'])
    return df['date'].dt.normalize().tolist()


def get_universe(db_engine) -> list:
    q = text("SELECT DISTINCT ticker FROM master_info WHERE ticker IS NOT NULL ORDER BY ticker")
    df = pd.read_sql(q, db_engine)
    return df['ticker'].tolist()


def forward_return(db_engine, tickers: list, start_date: pd.Timestamp, days: int = 21) -> pd.Series:
    if not tickers:
        return pd.Series(dtype=float)
    end_date = start_date + pd.Timedelta(days=days)
    q = text("""
        SELECT ticker, trading_date, close_price FROM equity_history
        WHERE ticker IN :tickers AND trading_date BETWEEN :start AND :end
        ORDER BY ticker, trading_date
    """)
    df = pd.read_sql(q, db_engine, params={'tickers': tuple(tickers), 'start': start_date, 'end': end_date})
    rets = {}
    for t, g in df.groupby('ticker'):
        px = g.sort_values('trading_date')['close_price']
        if len(px) >= 2 and px.iloc[0] > 0:
            rets[t] = (px.iloc[-1] - px.iloc[0]) / px.iloc[0]
    return pd.Series(rets, dtype=float)


def compute_portfolio_metrics(path_returns: pd.Series) -> dict:
    if path_returns is None or len(path_returns) == 0:
        return {'sharpe': 0.0, 'mdd': 0.0}
    r = path_returns.fillna(0.0).values
    mean = r.mean()
    std = r.std(ddof=1) if len(r) > 1 else 0.0
    sharpe = mean / std if std > 0 else 0.0
    # Max drawdown on cumulative
    cum = np.cumprod(1 + r)
    roll_max = np.maximum.accumulate(cum)
    drawdown = (cum / roll_max) - 1.0
    mdd = float(drawdown.min()) if len(drawdown) else 0.0
    return {'sharpe': float(sharpe), 'mdd': float(mdd)}


def main():
    log = setup_logger()
    parser = argparse.ArgumentParser(description='A/B backtest v2.2.1 vectorized F-Score')
    parser.add_argument('--start', required=True, help='Start date YYYY-MM-DD')
    parser.add_argument('--end', required=True, help='End date YYYY-MM-DD')
    parser.add_argument('--top-n', type=int, default=20, help='Portfolio size (default 20)')
    args = parser.parse_args()

    QVMEngineV221Flat, install_vec = import_engine()
    config_path = Path(__file__).parent.parent.parent / 'config'

    # Baseline engine (flag OFF)
    eng_a = QVMEngineV221Flat(config_path=str(config_path), log_level='INFO')
    eng_a.use_vectorized_fscore_221 = False

    # Vectorized engine (flag ON)
    eng_b = QVMEngineV221Flat(config_path=str(config_path), log_level='INFO')
    eng_b.use_vectorized_fscore_221 = True
    install_vec(eng_b)

    db = eng_a.engine
    universe = get_universe(db)
    dates = get_trading_dates(db, args.start, args.end)

    # Monthly rebalancing: take first trading day per calendar month
    months = {}
    for d in dates:
        key = (d.year, d.month)
        if key not in months:
            months[key] = d
    rebalance_dates = list(months.values())
    log.info("Rebalance dates: %d", len(rebalance_dates))

    path_rets_a = []
    path_rets_b = []
    turnover_list = []
    prev_hold_a = set()
    prev_hold_b = set()

    for d in rebalance_dates:
        try:
            scores_a = eng_a.calculate_qvm_composite_fixed(pd.Timestamp(d), universe)
            scores_b = eng_b.calculate_qvm_composite_fixed(pd.Timestamp(d), universe)
            if not scores_a or not scores_b:
                continue

            # Rank by QVM_Composite
            df_a = pd.DataFrame.from_dict(scores_a, orient='index')
            df_b = pd.DataFrame.from_dict(scores_b, orient='index')
            top_a = set(df_a['QVM_Composite'].sort_values(ascending=False).head(args.top_n).index)
            top_b = set(df_b['QVM_Composite'].sort_values(ascending=False).head(args.top_n).index)

            # Turnover vs previous
            if prev_hold_a:
                turnover_a = 1 - len(top_a & prev_hold_a) / float(args.top_n)
            else:
                turnover_a = 1.0
            if prev_hold_b:
                turnover_b = 1 - len(top_b & prev_hold_b) / float(args.top_n)
            else:
                turnover_b = 1.0
            turnover_list.append((turnover_a + turnover_b) / 2.0)

            # Forward 1M equal-weight return
            fr_a = forward_return(db, sorted(top_a), pd.Timestamp(d))
            fr_b = forward_return(db, sorted(top_b), pd.Timestamp(d))
            if len(fr_a) > 0:
                path_rets_a.append(fr_a.mean())
            if len(fr_b) > 0:
                path_rets_b.append(fr_b.mean())

            prev_hold_a = top_a
            prev_hold_b = top_b
        except Exception:
            log.exception("Rebalance step failed for %s", d)
            continue

    # Metrics
    metr_a = compute_portfolio_metrics(pd.Series(path_rets_a))
    metr_b = compute_portfolio_metrics(pd.Series(path_rets_b))
    avg_turnover = float(pd.Series(turnover_list).mean()) if turnover_list else 0.0

    print("\nA/B Results (Baseline vs Vectorized)")
    print("===================================")
    print(f"Sharpe:     A={metr_a['sharpe']:.3f} | B={metr_b['sharpe']:.3f}")
    print(f"MaxDD:      A={metr_a['mdd']:.3%} | B={metr_b['mdd']:.3%}")
    print(f"Turnover:   ~{avg_turnover:.2%}")


if __name__ == '__main__':
    main()


