#!/usr/bin/env python3
"""
08_QVM_v221_flat_vectorized_strategy.py
=======================================

Thin orchestration runner for QVM momentum strategy using v2.2.1 flat vectorized engine.

Phases:
1) Config + validation
2) Universe + factors (engine)
3) Holdings (top-N)
4) Prices + benchmark load
5) Portfolio returns (no-risk) + (with-risk)
6) Tearsheet(s) and artifacts

Run:
  python production/tests/phase31_add_fscore/08_QVM_v221_flat_vectorized_strategy.py \
    --config /home/raymond/Documents/Projects/factor-investing-public/production/config/strategy_config_v2_0_1_simple.yml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import sys
sys.path.append('/home/raymond/Documents/Projects/factor-investing-public/production/tests/phase31_add_fscore')

from production.engine.qvm_engine_v2_2_1_flat import QVMEngineV221Flat
from production.database.connection import DatabaseManager
from production.backtester.core import (
    BacktestConfig,
    build_daily_price_matrix,
    first_trading_day_calendar,
    run_daily_pnl,
)
from production.risk.overlay import drawdown_to_cash_allocation

# 07_ utilities consolidated under scripts
from scripts.configuration_manager import (
    load_strategy_config,
    load_backtest_config,
    validate_version_compatibility,
)
from scripts.validation_manager import (
    validate_strategy_config,
)
from scripts.tearsheet_generator import (
    calculate_performance_metrics,
    generate_comprehensive_tearsheet,
    generate_comparison_tearsheet,
)


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("QVM_v221_runner")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(ch)
    return logger


def _hash_config(cfg: Dict) -> str:
    norm = json.dumps(cfg, sort_keys=True).encode("utf-8")
    return hashlib.sha256(norm).hexdigest()[:12]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _select_top_n_holdings(scores: Dict[str, Dict[str, float]], n: int) -> List[str]:
    df = pd.DataFrame.from_dict(scores, orient="index")
    if df.empty or "QVM_Composite" not in df.columns:
        return []
    ranked = df.sort_values("QVM_Composite", ascending=False)
    return ranked.index.tolist()[:n]


def main():
    logger = _setup_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False, default=None)
    parser.add_argument("--window", type=str, required=False, help="YYYY-MM-DD:YYYY-MM-DD override")
    args = parser.parse_args()

    # Phase 1: Config + validation
    strategy_cfg = load_strategy_config(args.config)
    backtest_cfg = load_backtest_config(args.config)
    validate_version_compatibility(strategy_cfg, backtest_cfg)
    validate_strategy_config(strategy_cfg, logger)

    if args.window:
        start_s, end_s = args.window.split(":")
        backtest_cfg["active_window"] = {"start": start_s, "end": end_s}

    run_id = _hash_config({"strategy": strategy_cfg, "backtest": backtest_cfg})
    artifacts_dir = Path("artifacts/qvm_v221_flat_vectorized") / run_id
    _ensure_dir(artifacts_dir)
    # Persist config snapshot for lineage
    with open(artifacts_dir / 'strategy_config.json', 'w', encoding='utf-8') as f:
        json.dump(strategy_cfg, f, indent=2, sort_keys=True)
    with open(artifacts_dir / 'backtest_config.json', 'w', encoding='utf-8') as f:
        json.dump(backtest_cfg, f, indent=2, sort_keys=True)

    # DB engine
    db = DatabaseManager()
    db_engine = db.get_engine()

    # Strategy parameters
    portfolio_size = strategy_cfg.get("strategy", {}).get("portfolio", {}).get("portfolio_size", 20)
    tc_bps = backtest_cfg.get("transaction_cost_bps", 10.0)
    start_date = pd.to_datetime(backtest_cfg.get("active_window", {}).get("start"))
    end_date = pd.to_datetime(backtest_cfg.get("active_window", {}).get("end"))

    # Phase 2-3: Universe, factors, holdings per monthly rebalance date
    engine = QVMEngineV221Flat()
    all_holdings: Dict[pd.Timestamp, List[str]] = {}
    monthly_dates: List[pd.Timestamp] = []

    # Build a trading calendar from benchmark to anchor dates
    bench_query = f"""
        SELECT trading_date AS date, close_price FROM vcsc_daily_data_complete
        WHERE ticker='VNINDEX' AND trading_date BETWEEN '{start_date.date()}' AND '{end_date.date()}'
        ORDER BY trading_date
    """
    bench_df = pd.read_sql(bench_query, db_engine)
    if bench_df.empty:
        raise RuntimeError("Benchmark data not found; cannot proceed")
    bench_df["date"] = pd.to_datetime(bench_df["date"]) 
    benchmark_prices = bench_df.set_index("date")["close_price"].sort_index()
    benchmark_returns = benchmark_prices.pct_change().dropna()

    # Monthly calendar = first trading day per month
    monthly_dates = list(benchmark_prices.index.to_period("M").to_timestamp("D"))
    monthly_dates = [benchmark_prices.index[benchmark_prices.index.searchsorted(d, side="left")] for d in monthly_dates if d >= benchmark_prices.index.min() and d <= benchmark_prices.index.max()]
    monthly_dates = sorted(pd.to_datetime(list(dict.fromkeys(monthly_dates))))

    for reb_date in monthly_dates:
        # Universe selection from DB
        from production.universe.constructors import get_liquid_universe
        universe = get_liquid_universe(reb_date, db_engine)
        if not universe:
            logger.warning(f"Empty universe on {reb_date.date()} - skipping")
            continue

        # Engine factor computation and composite
        scores = engine.calculate_qvm_composite_fixed(reb_date, universe)
        if not scores:
            logger.warning(f"No scores on {reb_date.date()} - skipping")
            continue

        holdings = _select_top_n_holdings(scores, portfolio_size)
        if not holdings:
            logger.warning(f"No holdings on {reb_date.date()} - skipping")
            continue
        all_holdings[reb_date] = holdings

    # Save holdings artifact
    if all_holdings:
        h_rows = []
        for d, lst in all_holdings.items():
            for t in lst:
                h_rows.append({"date": pd.Timestamp(d), "ticker": t})
        pd.DataFrame(h_rows).to_csv(artifacts_dir / 'monthly_holdings.csv', index=False)

    if not all_holdings:
        raise RuntimeError("No holdings generated in the window")

    # Phase 4: Prices + benchmark
    unique_tickers = sorted({t for lst in all_holdings.values() for t in lst})
    price_matrix = build_daily_price_matrix(db_engine, unique_tickers, start_date, end_date, logger)

    # Align calendar to available prices
    cal = first_trading_day_calendar(price_matrix)
    cal = [d for d in cal if d in all_holdings]
    if not cal:
        raise RuntimeError("No valid rebalance dates after alignment")

    # Phase 5: Daily PnL with and without risk overlay
    bt_cfg = BacktestConfig(transaction_cost_bps=tc_bps, portfolio_size=portfolio_size)

    def overlay_fn(bench_prices: pd.Series, current_date: pd.Timestamp) -> float:
        rules = strategy_cfg.get("risk_management", {}).get("cash_allocation", {})
        return drawdown_to_cash_allocation(bench_prices, current_date, rules)

    # No-risk path
    no_risk_returns, no_risk_equity, _ = run_daily_pnl(price_matrix, cal, all_holdings, bt_cfg, risk_overlay_fn=None, benchmark_returns=benchmark_returns, logger=logger)
    # With-risk path
    with_risk_returns, with_risk_equity, cash_df = run_daily_pnl(price_matrix, cal, all_holdings, bt_cfg, risk_overlay_fn=lambda bench, dt: overlay_fn(benchmark_prices, dt), benchmark_returns=benchmark_returns, logger=logger)

    # Phase 6: Tearsheet(s)
    artifacts = {
        "no_risk_returns.csv": no_risk_returns,
        "with_risk_returns.csv": with_risk_returns,
        "with_risk_cash.csv": cash_df["cash_allocation"],
        "benchmark_returns.csv": benchmark_returns,
    }
    for name, series in artifacts.items():
        series.to_csv(artifacts_dir / name, header=True)

    # Generate comparison tearsheet
    generate_comparison_tearsheet(with_risk_returns, no_risk_returns, benchmark_returns, strategy_cfg)

    logger.info(f"Artifacts saved under {artifacts_dir}")


if __name__ == "__main__":
    main()


