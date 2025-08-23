#!/usr/bin/env python3
"""
Normalization Utilities
=======================

Provides hierarchical, diagnostics-aware normalization with:
- Dynamic min-sector-size per date/universe
- Hierarchical fallback: sector → industry → market
- Thin-group shrinkage (James–Stein style) toward global mean
- Robust global fallback using median/MAD

Designed for reuse across engines. Does not mutate inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple
import logging

import pandas as pd


@dataclass
class NormalizationConfig:
    """Lightweight configuration container for normalization policy."""

    min_sector_size: Optional[int] = None  # None or integer. None implies dynamic policy
    robust: str = "median_mad"             # median_mad | standard
    fallback: Iterable[str] = ("sector", "industry", "market")

    @staticmethod
    def dynamic_min_sector_size(universe_size: int) -> int:
        # Policy: dynamic = min(10, max(3, round(0.02 * universe_size)))
        return min(10, max(3, int(round(0.02 * max(universe_size, 0)))))

    def resolve_min_sector_size(self, universe_size: int) -> int:
        return int(self.min_sector_size) if isinstance(self.min_sector_size, int) else self.dynamic_min_sector_size(universe_size)


def _robust_zscore(values: pd.Series) -> pd.Series:
    vals = pd.to_numeric(values, errors="coerce")
    med = vals.median()
    mad = (vals - med).abs().median()
    scale = 1.4826 * mad if pd.notna(mad) and mad and mad > 0 else float(vals.std() if vals.std() > 0 else 1.0)
    if scale == 0:
        return pd.Series(0.0, index=values.index)
    return (vals - med) / scale


def _standard_zscore(values: pd.Series) -> pd.Series:
    vals = pd.to_numeric(values, errors="coerce")
    mean_val = vals.mean()
    std_val = vals.std()
    if std_val == 0:
        return pd.Series(0.0, index=values.index)
    return (vals - mean_val) / std_val


def _apply_shrinkage_to_group(scores: pd.Series, group_size: int, min_size: int) -> pd.Series:
    if group_size >= min_size or min_size <= 0:
        return scores
    shrink = float(group_size) / float(min_size)
    return scores * shrink


def compute_hierarchical_zscores(
    data: pd.DataFrame,
    metric_column: str,
    sector_column: str = "sector",
    industry_column: str = "industry",
    cfg: Optional[NormalizationConfig] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[pd.Series, Dict[str, object]]:
    """
    Compute hierarchical normalized scores with diagnostics.

    Returns
    -------
    (z_scores, info) where info includes:
      - min_sector_size
      - small_sectors_fraction
      - fallback_order
      - universe_size
    """
    try:
        # Defensive inputs
        if metric_column not in data.columns:
            raise ValueError(f"metric_column '{metric_column}' not in DataFrame")

        # Determine universe size
        try:
            universe_size = int(data["ticker"].nunique()) if "ticker" in data.columns else int(len(data))
        except Exception:
            universe_size = int(len(data))

        # Resolve configuration
        cfg = cfg or NormalizationConfig()
        min_sector_size = cfg.resolve_min_sector_size(universe_size)
        robust_method = (cfg.robust or "median_mad").lower()
        fallback_order = [str(x).lower() for x in (cfg.fallback or ("sector", "industry", "market"))]

        # Persist sector size diagnostics if available
        try:
            from production.utils.diagnostics import persist_sector_sizes_row  # type: ignore
        except Exception:
            persist_sector_sizes_row = None  # type: ignore

        sector_counts = pd.Series(dtype=int)
        if sector_column in data.columns:
            try:
                sector_counts = data.groupby(sector_column)[metric_column].count()
            except Exception:
                sector_counts = pd.Series(dtype=int)

        if persist_sector_sizes_row is not None and not sector_counts.empty:
            try:
                inferred_date = None
                if "date" in data.columns:
                    inferred_date = pd.to_datetime(data["date"].iloc[0])
                persist_sector_sizes_row(
                    inferred_date or pd.to_datetime("today").normalize(),
                    sector_counts,
                    universe_size,
                    logger=logger,
                )
            except Exception:
                pass

        small_fraction = float((sector_counts < min_sector_size).mean()) if len(sector_counts) > 0 else 1.0

        # Choose z-score functions
        global_fn = _robust_zscore if robust_method == "median_mad" else _standard_zscore
        within_fn = _standard_zscore

        z_scores = pd.Series(index=data.index, dtype=float)
        remaining_idx = pd.Index(data.index)

        for level in fallback_order:
            if remaining_idx.empty:
                break

            if level == "sector" and sector_column in data.columns:
                group_key = data.loc[remaining_idx, sector_column]
            elif level == "industry" and industry_column in data.columns:
                group_key = data.loc[remaining_idx, industry_column]
            elif level in ("market", "all", "global"):
                group_key = pd.Series("market", index=remaining_idx)
            else:
                continue

            def _compute_group(g: pd.DataFrame) -> pd.Series:
                series = g[metric_column]
                if level in ("market", "all", "global"):
                    return global_fn(series)
                z = within_fn(series)
                return _apply_shrinkage_to_group(z, group_size=len(series.dropna()), min_size=min_sector_size)

            grouped = data.loc[remaining_idx].groupby(group_key, dropna=False)
            level_scores = grouped.apply(lambda grp: _compute_group(grp))

            if isinstance(level_scores, pd.DataFrame):
                if level_scores.shape[1] == 1:
                    level_scores = level_scores.iloc[:, 0]
                else:
                    squeezed = level_scores.squeeze()
                    level_scores = squeezed if isinstance(squeezed, pd.Series) else level_scores.iloc[:, 0]
            try:
                level_scores = level_scores.reset_index(level=0, drop=True)
            except Exception:
                pass
            level_scores = level_scores.reindex(data.loc[remaining_idx].index)
            level_scores = pd.Series(level_scores, dtype=float)

            assignable = level_scores.index
            z_scores.loc[assignable] = z_scores.loc[assignable].where(z_scores.loc[assignable].notna(), level_scores)
            remaining_idx = z_scores[z_scores.isna()].index

        if z_scores.isna().any():
            z_scores = z_scores.fillna(0.0)

        # Winsorize for stability
        z_scores = z_scores.clip(-3, 3)

        if logger is not None:
            logger.debug(
                "Normalization: order=%s | min_sector_size=%d | small_sectors_frac=%.2f | universe=%d",
                fallback_order,
                min_sector_size,
                small_fraction,
                universe_size,
            )

        info = {
            "min_sector_size": int(min_sector_size),
            "small_sectors_fraction": float(small_fraction),
            "fallback_order": list(fallback_order),
            "universe_size": int(universe_size),
        }
        return z_scores, info

    except Exception as e:
        if logger is not None:
            logger.error("Hierarchical normalization failed: %s", e)
        return pd.Series(0.0, index=data.index), {
            "min_sector_size": None,
            "small_sectors_fraction": None,
            "fallback_order": None,
            "universe_size": None,
        }


