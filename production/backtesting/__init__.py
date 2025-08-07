"""
Backtesting Module

This module contains backtesting engines and strategy constructors for factor investing.
"""

from .vectorized_engine import VectorizedEngine
from .strategy_constructors import equal_weight_top_quintile, score_weighted_beta_constrained

__all__ = ['VectorizedEngine', 'equal_weight_top_quintile', 'score_weighted_beta_constrained']

