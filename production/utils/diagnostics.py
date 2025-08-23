#!/usr/bin/env python3
"""
Diagnostics Utilities
=====================

Purpose:
- Persist per-date sector group sizes and quick summaries
- Track normalization fallback streaks to control log noise

Artifacts:
- artifacts/diagnostics/sector_sizes.parquet

Notes:
- Uses pandas.to_parquet with pyarrow/fastparquet if available
- Writes are append-friendly by reading existing file and concatenating
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import pandas as pd


DEFAULT_DIAG_DIR = os.path.join("artifacts", "diagnostics")
DEFAULT_SECTOR_SIZES_PATH = os.path.join(DEFAULT_DIAG_DIR, "sector_sizes.parquet")


def _ensure_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def persist_sector_sizes_row(
    analysis_date: pd.Timestamp,
    sector_counts: pd.Series,
    universe_size: int,
    output_path: str = DEFAULT_SECTOR_SIZES_PATH,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Persist per-sector sizes for a given date with summary stats.

    Schema written:
        date, sector, count, universe_size, min, median, p10, p90
    """
    try:
        _ensure_dir(output_path)
        series = sector_counts.sort_index()
        # Summary stats for the one-line log and contextual columns
        summary_min = int(series.min()) if len(series) > 0 else 0
        summary_median = float(series.median()) if len(series) > 0 else 0.0
        summary_p10 = float(series.quantile(0.10)) if len(series) > 0 else 0.0
        summary_p90 = float(series.quantile(0.90)) if len(series) > 0 else 0.0

        df_new = pd.DataFrame(
            {
                "date": [analysis_date] * len(series),
                "sector": series.index,
                "count": series.values,
                "universe_size": [universe_size] * len(series),
                "min": [summary_min] * len(series),
                "median": [summary_median] * len(series),
                "p10": [summary_p10] * len(series),
                "p90": [summary_p90] * len(series),
            }
        )

        if os.path.exists(output_path):
            try:
                df_existing = pd.read_parquet(output_path)
                df_out = pd.concat([df_existing, df_new], ignore_index=True)
            except Exception:
                # If read fails (schema drift), fall back to write fresh
                df_out = df_new
        else:
            df_out = df_new

        df_out.to_parquet(output_path, index=False)

        if logger:
            logger.info(
                "Sector sizes: min=%d | median=%.1f | p10=%.1f | p90=%.1f (sectors=%d, universe=%d)",
                summary_min,
                summary_median,
                summary_p10,
                summary_p90,
                len(series),
                universe_size,
            )
    except Exception as e:
        if logger:
            logger.debug(f"Failed to persist sector sizes diagnostics: {e}")


@dataclass
class NormalizationFallbackTracker:
    """
    Tracks consecutive rebalances where a high fraction of sectors fall below
    the configured min_sector_size, to gate WARN-level logging.
    """

    threshold_fraction: float = 0.25
    required_consecutive: int = 3
    _streak: int = 0
    _last_warned_streak: int = 0

    def update_and_should_warn(self, small_fraction: float) -> bool:
        breach = small_fraction >= self.threshold_fraction
        self._streak = self._streak + 1 if breach else 0
        # Only WARN on first attainment of the required streak to avoid spam
        if self._streak >= self.required_consecutive and self._last_warned_streak < self._streak:
            self._last_warned_streak = self._streak
            return True
        return False


