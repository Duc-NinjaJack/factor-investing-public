#!/usr/bin/env python3
"""
Benchmark Loader Utility
========================

Loads benchmark close price series with a simple source priority policy and
an in-process LRU-style cache. Validates the returned series schema.

Usage:
    from production.utils.benchmark_loader import load_benchmark_series

API:
    load_benchmark_series(engine, start_date, end_date, priority=None, ticker='VNINDEX', logger=None)

Priority sources (default order):
    1) etf_history (columns: date, close)
    2) vcsc_daily_data_complete (columns: trading_date, close_price)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import logging
import pandas as pd
from sqlalchemy import text

_CACHE: Dict[Tuple[str, str, str, Tuple[str, ...]], Tuple[pd.Series, pd.Series]] = {}


def _series_schema_ok(series: pd.Series) -> bool:
    try:
        if not isinstance(series, pd.Series):
            return False
        if not isinstance(series.index, pd.DatetimeIndex):
            return False
        if series.isna().all():
            return False
        if not series.index.is_monotonic_increasing:
            return False
        if series.index.has_duplicates:
            return False
        # dtype coercion check
        _ = pd.to_numeric(series, errors='coerce')
        return True
    except Exception:
        return False


def load_benchmark_series(
    engine,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    priority: Optional[List[str]] = None,
    ticker: str = 'VNINDEX',
    logger: Optional[logging.Logger] = None,
) -> Tuple[pd.Series, pd.Series]:
    """
    Load benchmark close price series within [start_date, end_date].

    Returns:
        (close_prices: pd.Series[float], daily_returns: pd.Series[float])
    """
    sources = tuple((priority or ['etf_history', 'vcsc_daily_data_complete']))
    cache_key = (ticker, str(pd.to_datetime(start_date).date()), str(pd.to_datetime(end_date).date()), sources)
    if cache_key in _CACHE:
        if logger:
            logger.debug(f"Benchmark loader cache hit for {cache_key}")
        return _CACHE[cache_key]

    if logger:
        logger.info(f"Loading benchmark '{ticker}' from sources by priority: {list(sources)}")

    # 1) etf_history
    if 'etf_history' in sources:
        try:
            q = text("""
                SELECT date AS date, close AS close_price
                FROM etf_history
                WHERE ticker = :ticker AND date BETWEEN :start AND :end
                ORDER BY date
            """)
            df = pd.read_sql(q, engine, params={'ticker': ticker, 'start': start_date, 'end': end_date})
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                close = df.set_index('date')['close_price'].astype('float64').sort_index()
                if _series_schema_ok(close):
                    rets = close.pct_change(fill_method=None).dropna()
                    _CACHE[cache_key] = (close, rets)
                    if logger:
                        logger.info("Benchmark loaded from etf_history")
                    return close, rets
        except Exception as _e:
            if logger:
                logger.debug(f"etf_history load failed: {_e}")

    # 2) vcsc_daily_data_complete
    if 'vcsc_daily_data_complete' in sources:
        q = text("""
            SELECT trading_date AS date, close_price AS close_price
            FROM vcsc_daily_data_complete
            WHERE ticker = :ticker AND trading_date BETWEEN :start AND :end
            ORDER BY trading_date
        """)
        df = pd.read_sql(q, engine, params={'ticker': ticker, 'start': start_date, 'end': end_date})
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            close = df.set_index('date')['close_price'].astype('float64').sort_index()
            if _series_schema_ok(close):
                rets = close.pct_change(fill_method=None).dropna()
                _CACHE[cache_key] = (close, rets)
                if logger:
                    logger.info("Benchmark loaded from vcsc_daily_data_complete")
                return close, rets

    # Fallthrough: return empty series
    if logger:
        logger.warning("Benchmark data not found from configured sources")
    empty = pd.Series(dtype='float64')
    _CACHE[cache_key] = (empty, empty)
    return empty, empty


