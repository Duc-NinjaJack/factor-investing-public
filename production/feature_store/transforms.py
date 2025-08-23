#!/usr/bin/env python3
"""
Transforms
==========

Robust transforms used in feature engineering:
- winsorize → normalize → neutralize
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple


def winsorize_series(values: pd.Series, limits: Tuple[float, float] = (0.01, 0.01)) -> pd.Series:
    if values.empty:
        return values
    lower_q = values.quantile(limits[0])
    upper_q = values.quantile(limits[1] if limits[1] <= 1 else 1.0 - limits[1]) if limits[1] <= 1 else values.quantile(0.99)
    return values.clip(lower=lower_q, upper=upper_q)


def robust_zscore_series(values: pd.Series) -> pd.Series:
    v = values.astype(float)
    med = v.median()
    mad = (v - med).abs().median()
    scale = 1.4826 * mad if mad and mad > 0 else float(v.std() if v.std() > 0 else 1.0)
    if scale == 0:
        return pd.Series(0.0, index=values.index)
    return (v - med) / scale


def group_neutralize(df: pd.DataFrame, value_col: str, group_col: str) -> pd.Series:
    if df.empty or value_col not in df.columns or group_col not in df.columns:
        return pd.Series(0.0, index=df.index)
    def demean(g: pd.DataFrame) -> pd.Series:
        vals = g[value_col].astype(float)
        mean_val = vals.mean()
        std_val = vals.std()
        if std_val == 0:
            return pd.Series(0.0, index=g.index)
        return (vals - mean_val) / std_val
    z = df.groupby(group_col, dropna=False).apply(demean).reset_index(level=0, drop=True)
    return z.reindex(df.index).fillna(0.0)


