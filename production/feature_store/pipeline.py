#!/usr/bin/env python3
"""
Feature Pipeline
================

Phased pipeline: ingest → clean → features → signals
Deterministic and partitioned by date/universe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Callable
import pandas as pd

from .io import FeatureStoreIO
from .transforms import winsorize_series, robust_zscore_series, group_neutralize


@dataclass
class FeaturePipeline:
    io: FeatureStoreIO
    logger: Optional[object] = None

    def run(
        self,
        date: pd.Timestamp,
        universe: List[str],
        ingest_fn: Callable[[pd.Timestamp, List[str]], pd.DataFrame],
        sector_map: Optional[pd.Series] = None,
    ) -> Dict[str, str]:
        date_str = pd.to_datetime(date).date().isoformat()

        # Ingest
        raw_df = ingest_fn(date, universe)
        if sector_map is not None and 'sector' not in raw_df.columns and 'ticker' in raw_df.columns:
            try:
                raw_df['sector'] = raw_df['ticker'].map(sector_map)
            except Exception:
                pass
        p1 = self.io.write_partition('ingest', date_str, raw_df, universe_size=len(universe), logger=self.logger)

        # Clean
        clean_df = raw_df.copy()
        # Basic cleaning: drop duplicates, enforce dtypes where obvious
        if 'ticker' in clean_df.columns:
            clean_df = clean_df.drop_duplicates(subset=['ticker']).reset_index(drop=True)
        p2 = self.io.write_partition('clean', date_str, clean_df, universe_size=len(universe), logger=self.logger)

        # Features
        features_df = clean_df.copy()
        # Example robust transforms if numeric columns exist
        numeric_cols = [c for c in features_df.columns if c not in {'ticker', 'sector', 'date'} and pd.api.types.is_numeric_dtype(features_df[c])]
        for col in numeric_cols:
            series = features_df[col]
            series = winsorize_series(series, (0.01, 0.01))
            series = robust_zscore_series(series)
            features_df[col + '_z'] = series
        if 'sector' in features_df.columns:
            for col in [c for c in features_df.columns if c.endswith('_z')]:
                features_df[col + '_sn'] = group_neutralize(features_df[['sector', col]].assign(value=features_df[col]), 'value', 'sector')
        p3 = self.io.write_partition('features', date_str, features_df, universe_size=len(universe), logger=self.logger)

        # Signals (simple aggregation placeholder)
        signals_df = pd.DataFrame({'ticker': features_df.get('ticker', pd.Series(dtype=object))})
        for col in [c for c in features_df.columns if c.endswith('_sn')]:
            signals_df[col.replace('_sn', '_signal')] = features_df[col]
        p4 = self.io.write_partition('signals', date_str, signals_df, universe_size=len(universe), logger=self.logger)

        return {'ingest': p1, 'clean': p2, 'features': p3, 'signals': p4}


