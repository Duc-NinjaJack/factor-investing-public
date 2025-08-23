#!/usr/bin/env python3
"""
Risk Overlays
=============

Implements drawdown-to-cash rules and an optional volatility targeting hook.

API focuses on pure functions for testability.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def drawdown_to_cash_allocation(
    benchmark_prices: pd.Series,
    current_date: pd.Timestamp,
    rules: Optional[Dict[str, float]] = None,
) -> float:

    if benchmark_prices is None or benchmark_prices.empty:
        return 0.0
    hist = benchmark_prices.loc[:current_date]
    if hist.empty:
        return 0.0
    peak = float(hist.max())
    cur = float(hist.iloc[-1])
    if peak <= 0:
        return 0.0
    dd = (peak - cur) / peak

    eff = rules or {
        "drawdown_5": 0.20,
        "drawdown_10": 0.40,
        "drawdown_15": 0.60,
        "drawdown_20": 0.80,
        "drawdown_25": 0.90,
        "drawdown_30": 0.95,
        "drawdown_40": 0.98,
        "drawdown_50": 0.99,
    }
    if dd < 0.05: cash = eff.get("drawdown_5", 0.20)
    elif dd < 0.10: cash = eff.get("drawdown_10", 0.40)
    elif dd < 0.15: cash = eff.get("drawdown_15", 0.60)
    elif dd < 0.20: cash = eff.get("drawdown_20", 0.80)
    elif dd < 0.25: cash = eff.get("drawdown_25", 0.90)
    elif dd < 0.30: cash = eff.get("drawdown_30", 0.95)
    elif dd < 0.40: cash = eff.get("drawdown_40", 0.98)
    elif dd < 0.50: cash = eff.get("drawdown_50", 0.99)
    else: cash = 0.99
    return float(max(0.0, min(1.0, cash)))


def volatility_targeting(
    strategy_returns: pd.Series,
    target_vol: float = 0.15,
    window: int = 60,
    min_exposure: float = 0.2,
    max_exposure: float = 1.0,
) -> Tuple[pd.Series, pd.Series]:

    if strategy_returns is None or strategy_returns.empty:
        return pd.Series(dtype="float64"), pd.Series(dtype="float64")
    realized = strategy_returns.rolling(window=window).std() * np.sqrt(252)
    exposure = (target_vol / realized).shift(1)
    exposure = exposure.clip(lower=min_exposure, upper=max_exposure).fillna(1.0)
    aligned = strategy_returns.index.intersection(exposure.index)
    managed = strategy_returns.loc[aligned] * exposure.loc[aligned]
    return managed, exposure


# New overlay: EWMA drawdown-based cash ramp
def ewma_drawdown_cash_allocation(
    benchmark_prices: pd.Series,
    current_date: pd.Timestamp,
    halflife: int = 20,
    max_cash: float = 0.9,
) -> float:
    """
    EWMA drawdown where the peak is tracked via EWMA to dampen noise.
    Returns cash allocation in [0, max_cash].
    """
    if benchmark_prices is None or benchmark_prices.empty:
        return 0.0
    hist = benchmark_prices.loc[:current_date]
    if hist.empty:
        return 0.0
    # EWMA peak approximation: use EWMA of prices and treat deviation from rolling max of ewma
    ewma = hist.ewm(halflife=halflife, adjust=False).mean()
    if ewma.empty or ewma.iloc[-1] <= 0:
        return 0.0
    running_peak = ewma.cummax()
    dd = float((running_peak.iloc[-1] - ewma.iloc[-1]) / running_peak.iloc[-1]) if running_peak.iloc[-1] > 0 else 0.0
    # Linear ramp up to max_cash at 40% EWMA DD
    cash = max(0.0, min(max_cash, dd / 0.40 * max_cash))
    return float(cash)

