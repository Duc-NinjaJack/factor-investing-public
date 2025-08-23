#!/usr/bin/env python3
"""
Feature Engineering Utilities
=============================

Reusable, unit-testable feature functions that do not mutate inputs and
return index-aligned pandas objects. These utilities implement raw feature
calculations and normalization preparation. Sector-neutral z-scoring should
be applied by callers via production.utils.normalization to keep a single
source of truth for normalization policy.

Design goals:
- Explicit typing and docstrings
- Defensive input validation and error handling
- Avoid hidden state; pure functions where practical
- Do not perform any database I/O
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Tuple, Optional
import logging

import numpy as np
import pandas as pd


def compute_period_return(price_series: pd.Series, start_date: pd.Timestamp, end_date: pd.Timestamp) -> Optional[float]:
    """
    Compute simple return between the first available price on/after start_date
    and the last available price on/before end_date.

    Returns None if prices are insufficient.
    """
    try:
        if price_series is None or price_series.empty:
            return None
        price_series = price_series.sort_index()
        start_idx = price_series.index.searchsorted(start_date, side='left')
        if start_idx >= len(price_series):
            return None
        start_price = price_series.iloc[start_idx]
        end_idx = price_series.index.searchsorted(end_date, side='right') - 1
        if end_idx < 0:
            return None
        end_price = price_series.iloc[end_idx]
        if pd.notna(start_price) and pd.notna(end_price) and start_price > 0:
            return float(end_price / start_price - 1.0)
        return None
    except Exception:
        return None


def compute_momentum_raw(
    prices: pd.DataFrame,
    analysis_date: pd.Timestamp,
    universe: Iterable[str],
    lookbacks_months: Mapping[str, int],
    skip_months: int = 1,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, pd.Series]:
    """
    Compute raw momentum returns for multiple horizons with a skip window.

    Args:
        prices: DataFrame with columns ['date','ticker','price'] or indexed by date with tickers as columns
        analysis_date: Current analysis date
        universe: List of tickers to compute
        lookbacks_months: Mapping of label->months (e.g., {'1m':1,'3m':3,'6m':6,'12m':12})
        skip_months: Number of months to skip from analysis_date backward
        logger: Optional logger for diagnostics

    Returns:
        Dict of label -> Series[ticker -> raw_return]
    """
    try:
        # Normalize to long format with MultiIndex (date, ticker) -> price
        if {'date', 'ticker', 'price'} <= set(prices.columns):
            df = prices[['date', 'ticker', 'price']].copy()
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values(['ticker', 'date'])
        else:
            # Assume wide format (index=date, columns=tickers)
            df = prices.copy()
            df = df.stack().reset_index()
            df.columns = ['date', 'ticker', 'price']
            df['date'] = pd.to_datetime(df['date'])

        universe_set = set(str(t) for t in universe)
        df = df[df['ticker'].astype(str).isin(universe_set)]
        if df.empty:
            return {}

        raw: Dict[str, pd.Series] = {}
        for label, months in lookbacks_months.items():
            end_date = analysis_date - pd.DateOffset(months=int(skip_months))
            start_date = analysis_date - pd.DateOffset(months=int(skip_months) + int(months))
            returns: Dict[str, float] = {}
            for ticker, sub in df.groupby('ticker'):
                ts = sub.set_index('date')['price']
                r = compute_period_return(ts, start_date, end_date)
                if r is not None and np.isfinite(r):
                    returns[str(ticker)] = float(r)
            raw[label] = pd.Series(returns, name=f'momentum_{label}_raw')
            if logger:
                logger.debug("Momentum %s: computed %d/%d tickers", label, len(raw[label]), len(universe_set))
        return raw
    except Exception as e:
        if logger:
            logger.error("Failed to compute momentum raw returns: %s", e)
        return {}


def compute_low_volatility_raw(
    prices: pd.DataFrame,
    analysis_date: pd.Timestamp,
    universe: Iterable[str],
    lookback_days: int = 63,
    logger: Optional[logging.Logger] = None,
) -> pd.Series:
    """
    Compute inverse realized volatility over a rolling window as a raw score.

    Args:
        prices: DataFrame with columns ['date','ticker','price']
        analysis_date: Current analysis date
        universe: Tickers
        lookback_days: Number of trading days for volatility

    Returns:
        Series[ticker -> -std(returns)] using percentage returns; higher is better.
    """
    try:
        if not {'date', 'ticker', 'price'} <= set(prices.columns):
            raise ValueError("prices must contain columns ['date','ticker','price']")
        df = prices.copy()
        df['date'] = pd.to_datetime(df['date'])
        start_date = analysis_date - pd.DateOffset(days=int(lookback_days) + 30)
        df = df[(df['date'] >= start_date) & (df['date'] <= analysis_date)]
        if df.empty:
            return pd.Series(dtype=float, name='low_volatility_raw')
        scores: Dict[str, float] = {}
        for ticker, sub in df.groupby('ticker'):
            s = sub.sort_values('date')['price'].pct_change(fill_method=None).dropna()
            if len(s) >= lookback_days:
                vol = float(s.tail(lookback_days).std())
                scores[str(ticker)] = -vol
        return pd.Series(scores, name='low_volatility_raw')
    except Exception as e:
        if logger:
            logger.error("Failed to compute low-volatility raw: %s", e)
        return pd.Series(dtype=float, name='low_volatility_raw')


def prepare_fcf_yield_raw(
    fundamentals: pd.DataFrame,
    market_caps: pd.DataFrame,
    *,
    use_actual_capex_when_available: bool = True,
    logger: Optional[logging.Logger] = None,
) -> pd.Series:
    """
    Prepare raw FCF Yield values from provided fundamentals and market caps.

    Args:
        fundamentals: DataFrame with columns including ['ticker','NetCFO_TTM','CapEx_TTM','NetCFI_TTM']
        market_caps: DataFrame with columns ['ticker','market_cap']
        use_actual_capex_when_available: If True, compute FCF=CFO-CapEx when CapEx present; else fallback to CFI proxy

    Returns:
        Series[ticker -> fcf_yield_raw]
    """
    try:
        req_f_cols = {'ticker', 'NetCFO_TTM', 'NetCFI_TTM', 'CapEx_TTM'}
        req_m_cols = {'ticker', 'market_cap'}
        if not req_f_cols <= set(fundamentals.columns):
            raise ValueError("fundamentals missing required columns")
        if not req_m_cols <= set(market_caps.columns):
            raise ValueError("market_caps missing required columns")
        f = fundamentals[['ticker', 'NetCFO_TTM', 'NetCFI_TTM', 'CapEx_TTM']].copy()
        m = market_caps[['ticker', 'market_cap']].copy()
        df = pd.merge(f, m, on='ticker', how='inner')
        df = df.replace([np.inf, -np.inf], np.nan)
        out: Dict[str, float] = {}
        capex_imputed = 0
        capex_actual = 0
        total = 0
        for _, row in df.iterrows():
            ticker = str(row['ticker'])
            cfo = row.get('NetCFO_TTM')
            cfi = row.get('NetCFI_TTM')
            capex = row.get('CapEx_TTM')
            mcap = row.get('market_cap')
            if pd.isna(mcap) or mcap <= 0 or pd.isna(cfo):
                continue
            total += 1
            fcf = None
            if use_actual_capex_when_available and pd.notna(capex) and capex != 0:
                fcf = float(cfo - capex)
                capex_actual += 1
            elif pd.notna(cfi):
                capex_proxy = max(0.0, -float(cfi))
                fcf = float(cfo - capex_proxy)
                capex_imputed += 1
            if fcf is None:
                continue
            out[ticker] = float(fcf / mcap) if mcap > 0 else np.nan
        if logger and total > 0:
            logger.info("FCF prep: actual_capex=%d (%.1f%%), imputed=%d (%.1f%%), total=%d",
                        capex_actual, 100.0 * capex_actual / total, capex_imputed, 100.0 * capex_imputed / total, total)
        return pd.Series(out, name='fcf_yield_raw')
    except Exception as e:
        if logger:
            logger.error("Failed to prepare FCF Yield raw: %s", e)
        return pd.Series(dtype=float, name='fcf_yield_raw')


def normalize_f_score_to_unit(raw_scores: Mapping[str, Tuple[int, int]], logger: Optional[logging.Logger] = None) -> pd.Series:
    """
    Normalize Piotroski F-Score from sector-specific max scale to [0,1].

    Args:
        raw_scores: mapping ticker -> (raw_score, max_score)

    Returns:
        Series[ticker -> normalized_score]
    """
    try:
        data: Dict[str, float] = {}
        for ticker, (raw, maxv) in raw_scores.items():
            try:
                maxv_f = float(maxv)
                raw_f = float(raw)
                if not np.isfinite(maxv_f) or not np.isfinite(raw_f) or maxv_f <= 0:
                    data[str(ticker)] = 0.0
                else:
                    data[str(ticker)] = float(raw_f / maxv_f)
            except Exception:
                data[str(ticker)] = 0.0
        return pd.Series(data, name='f_score_normalized')
    except Exception as e:
        if logger:
            logger.error("Failed to normalize F-Score: %s", e)
        return pd.Series(dtype=float, name='f_score_normalized')


def make_normalization_frame(values: pd.Series, sector_map: Mapping[str, str], value_column_name: str, sector_column_name: str = 'sector') -> pd.DataFrame:
    """
    Prepare a tidy DataFrame for sector-neutral normalization from a ticker-indexed Series.

    Args:
        values: Series with index=ticker and values to normalize
        sector_map: Mapping ticker->sector
        value_column_name: Name for the metric column
        sector_column_name: Name for the sector column (default 'sector')

    Returns:
        DataFrame with columns ['ticker', value_column_name, sector_column_name]
    """
    s = values.copy()
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return pd.DataFrame(columns=['ticker', value_column_name, sector_column_name])
    tickers = s.index.astype(str)
    sectors = [sector_map.get(str(t), 'Unknown') for t in tickers]
    return pd.DataFrame({'ticker': tickers, value_column_name: s.values, sector_column_name: sectors})


