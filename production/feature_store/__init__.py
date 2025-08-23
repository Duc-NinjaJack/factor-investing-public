"""
Feature Store Package
=====================

Phased, deterministic pipeline for ingest → clean → features → signals.
Persists intermediate Parquet partitions by date/universe with immutable metadata.
"""

from .io import FeatureStoreIO
from .transforms import winsorize_series, robust_zscore_series, group_neutralize
from .pipeline import FeaturePipeline
from .registry import SchemaRegistry, BackfillManager

__all__ = [
    'FeatureStoreIO',
    'winsorize_series',
    'robust_zscore_series',
    'group_neutralize',
    'FeaturePipeline',
    'SchemaRegistry',
    'BackfillManager',
]


