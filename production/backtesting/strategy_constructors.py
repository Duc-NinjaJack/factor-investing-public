"""
Strategy Constructors

This module contains functions for constructing different portfolio strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

def equal_weight_top_quintile(factor_scores: pd.DataFrame, 
                             returns_data: pd.DataFrame,
                             config: Dict) -> pd.DataFrame:
    """
    Construct equal-weighted portfolio from top quintile stocks.
    
    Args:
        factor_scores: DataFrame containing factor scores
        returns_data: DataFrame containing stock returns
        config: Configuration dictionary
        
    Returns:
        DataFrame containing portfolio weights
    """
    logger.info("Constructing equal-weight top quintile portfolio...")
    
    # Get selection percentile from config
    selection_percentile = config.get('portfolio', {}).get('selection_percentile', 0.8)
    
    # Sort by composite score and select top quintile
    factor_scores_sorted = factor_scores.sort_values('composite_score', ascending=False)
    
    # Limit portfolio size to reasonable number (20-30 stocks)
    max_stocks = 25
    n_stocks = min(int(len(factor_scores_sorted) * selection_percentile), max_stocks)
    selected_stocks = factor_scores_sorted.head(n_stocks)
    
    # Equal weight allocation
    portfolio_weights = selected_stocks[['ticker', 'composite_score']].copy()
    portfolio_weights['weight'] = 1.0 / len(selected_stocks)
    
    logger.info(f"Selected {len(selected_stocks)} stocks with equal weights")
    return portfolio_weights

def score_weighted_beta_constrained(factor_scores: pd.DataFrame,
                                   returns_data: pd.DataFrame,
                                   config: Dict) -> pd.DataFrame:
    """
    Construct score-weighted portfolio with beta constraints.
    
    Args:
        factor_scores: DataFrame containing factor scores
        returns_data: DataFrame containing stock returns
        config: Configuration dictionary
        
    Returns:
        DataFrame containing portfolio weights
    """
    logger.info("Constructing score-weighted beta-constrained portfolio...")
    
    # Get configuration parameters
    portfolio_config = config.get('portfolio', {})
    selection_percentile = portfolio_config.get('selection_percentile', 0.8)
    beta_target = portfolio_config.get('beta_target', 0.9)
    max_weight = portfolio_config.get('max_weight', 0.08)
    
    # Sort by composite score and select top stocks
    factor_scores_sorted = factor_scores.sort_values('composite_score', ascending=False)
    
    # Limit portfolio size to reasonable number (20-30 stocks)
    max_stocks = 25
    n_stocks = min(int(len(factor_scores_sorted) * selection_percentile), max_stocks)
    selected_stocks = factor_scores_sorted.head(n_stocks)
    
    # Score-weighted allocation (using composite score as weight)
    portfolio_weights = selected_stocks[['ticker', 'composite_score']].copy()
    
    # Normalize scores to weights
    total_score = portfolio_weights['composite_score'].sum()
    portfolio_weights['weight'] = portfolio_weights['composite_score'] / total_score
    
    # Apply maximum weight constraint
    portfolio_weights['weight'] = portfolio_weights['weight'].clip(upper=max_weight)
    
    # Renormalize weights
    total_weight = portfolio_weights['weight'].sum()
    portfolio_weights['weight'] = portfolio_weights['weight'] / total_weight
    
    logger.info(f"Selected {len(selected_stocks)} stocks with score-weighted allocation (max weight: {max_weight})")
    return portfolio_weights

