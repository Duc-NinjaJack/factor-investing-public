#!/usr/bin/env python3
"""
Backtesting Core Utilities
==========================

Purpose:
- Daily price matrix construction
- Monthly first-trading-day rebalance calendar
- Equal-weight sizing and transaction costs
- Daily PnL engine with/without risk overlay

Design:
- Config-driven; no CSV fallbacks. Fail fast on missing data.
- Deterministic outputs given same inputs.

Dependencies:
- Expects SQLAlchemy engine for DB access

"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine
from sqlalchemy import text


def build_daily_price_matrix(engine: Engine, tickers: List[str], start_date: pd.Timestamp, end_date: pd.Timestamp, logger: Optional[logging.Logger] = None) -> pd.DataFrame:

    if not tickers:
        raise ValueError("No tickers provided for price matrix build")
    if start_date > end_date:
        raise ValueError("start_date must be <= end_date")

    query = text(
        """
        SELECT trading_date AS date, ticker, close_price
        FROM vcsc_daily_data_complete
        WHERE ticker IN :tickers
          AND trading_date BETWEEN :start AND :end
        ORDER BY trading_date, ticker
        """
    )
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"tickers": tuple(tickers), "start": start_date, "end": end_date})

    if df.empty:
        raise RuntimeError("Price data query returned empty result set")

    df["date"] = pd.to_datetime(df["date"])
    pivot = df.pivot(index="date", columns="ticker", values="close_price").sort_index()
    if logger:
        logger.info(f"Built price matrix: {pivot.shape[0]} days x {pivot.shape[1]} tickers")
    return pivot


def first_trading_day_calendar(price_matrix: pd.DataFrame) -> List[pd.Timestamp]:

    if price_matrix.empty:
        return []
    dates = price_matrix.index.to_series().sort_values()
    months = dates.dt.to_period("M")
    first_idx = dates.groupby(months).head(1).index
    return list(first_idx)


def equal_weight_positions(tickers: List[str], portfolio_size: int) -> pd.Series:

    if portfolio_size <= 0:
        raise ValueError("portfolio_size must be positive")
    selected = tickers[:portfolio_size]
    if not selected:
        return pd.Series(dtype="float64")
    weight = 1.0 / len(selected)
    return pd.Series({t: weight for t in selected}, dtype="float64")


def apply_transaction_costs(prev_weights: pd.Series, next_weights: pd.Series, bps: float) -> float:

    prev = prev_weights.reindex(prev_weights.index.union(next_weights.index)).fillna(0.0)
    nxt = next_weights.reindex(prev_weights.index.union(next_weights.index)).fillna(0.0)
    turnover = (nxt - prev).abs().sum() / 2.0
    return float(turnover * (bps / 10000.0))


@dataclass
class BacktestConfig:
    transaction_cost_bps: float = 10.0
    portfolio_size: int = 20
    slippage_bps: float = 0.0


def run_daily_pnl(
    daily_prices: pd.DataFrame,
    monthly_rebalance_dates: List[pd.Timestamp],
    monthly_holdings: Dict[pd.Timestamp, List[str]],
    backtest_config: BacktestConfig,
    risk_overlay_fn=None,
    benchmark_prices: Optional[pd.Series] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:

    if daily_prices.empty:
        raise ValueError("daily_prices is empty")
    if not monthly_rebalance_dates:
        raise ValueError("No rebalance dates provided")

    prices = daily_prices.sort_index()
    daily_returns = prices.pct_change().fillna(0.0)

    dates = prices.index
    start_idx = dates.get_indexer([monthly_rebalance_dates[0]], method="nearest")[0]
    equity_curve = pd.Series(index=dates[start_idx:], dtype="float64")
    equity_curve.iloc[0] = 1.0
    strategy_returns = pd.Series(0.0, index=dates[start_idx:])
    cash_allocations = pd.Series(0.0, index=dates[start_idx:])

    current_weights = pd.Series(dtype="float64")

    for i, day in enumerate(dates[start_idx:]):
        # Rebalance if day in calendar
        if day in monthly_rebalance_dates and day in monthly_holdings:
            tickers_today = [t for t in monthly_holdings[day] if t in daily_returns.columns]
            current_weights = equal_weight_positions(tickers_today, backtest_config.portfolio_size)

            # Apply costs for changing weights
            cost_bps = backtest_config.transaction_cost_bps + backtest_config.slippage_bps
            cost = apply_transaction_costs(current_weights, current_weights, cost_bps) if i == 0 else apply_transaction_costs(prev_weights, current_weights, cost_bps)
        else:
            cost = 0.0

        # Compute gross return
        gross = float((current_weights * daily_returns.loc[day].reindex(current_weights.index).fillna(0.0)).sum()) if not current_weights.empty else 0.0
        net = gross - cost

        # Risk overlay scaling (uses benchmark prices if provided)
        cash = 0.0
        if risk_overlay_fn is not None and benchmark_prices is not None:
            try:
                cash = float(risk_overlay_fn(benchmark_prices, day))
                cash = max(0.0, min(0.99, cash))
            except Exception:
                cash = 0.0
        exposure = 1.0 - cash

        strategy_returns.loc[day] = net * exposure
        cash_allocations.loc[day] = cash
        prev_weights = current_weights.copy()

        # Update equity curve
        if i > 0:
            equity_curve.iloc[i] = equity_curve.iloc[i - 1] * (1.0 + strategy_returns.loc[day])

    cash_df = pd.DataFrame({"cash_allocation": cash_allocations})
    return strategy_returns, equity_curve, cash_df


