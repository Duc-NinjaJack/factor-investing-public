"""
Reporting Utilities

This module contains utility functions for calculating performance metrics and generating reports.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

def calculate_official_metrics(returns: pd.Series, 
                              benchmark_returns: pd.Series,
                              risk_free_rate: float = 0.0) -> Dict:
    """
    Calculate official performance metrics.
    
    Args:
        returns: Strategy returns
        benchmark_returns: Benchmark returns
        risk_free_rate: Risk-free rate for calculations
        
    Returns:
        Dictionary containing performance metrics
    """
    try:
        logger.info("Calculating official performance metrics...")
        
        # Align the data
        aligned_data = pd.concat([returns, benchmark_returns], axis=1).dropna()
        if len(aligned_data) == 0:
            raise ValueError("No overlapping data between returns and benchmark")
        
        strategy_returns = aligned_data.iloc[:, 0]
        benchmark_returns = aligned_data.iloc[:, 1]
        
        # Calculate basic metrics
        total_return = (1 + strategy_returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
        volatility = strategy_returns.std() * np.sqrt(252)
        
        # Calculate Sharpe ratio
        excess_returns = strategy_returns - risk_free_rate / 252
        sharpe_ratio = excess_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0
        
        # Calculate Sortino ratio
        downside_returns = strategy_returns[strategy_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = excess_returns.mean() / downside_deviation * np.sqrt(252) if downside_deviation > 0 else 0
        
        # Calculate maximum drawdown
        cumulative_returns = (1 + strategy_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calculate Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Calculate information ratio and tracking error
        active_returns = strategy_returns - benchmark_returns
        information_ratio = active_returns.mean() / active_returns.std() * np.sqrt(252) if active_returns.std() > 0 else 0
        tracking_error = active_returns.std() * np.sqrt(252)
        
        # Calculate beta and alpha
        covariance = np.cov(strategy_returns, benchmark_returns)[0, 1]
        benchmark_variance = benchmark_returns.var()
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
        
        alpha = strategy_returns.mean() - beta * benchmark_returns.mean()
        alpha_annualized = alpha * 252
        
        # Calculate R-squared
        correlation = strategy_returns.corr(benchmark_returns)
        r_squared = correlation ** 2 if not pd.isna(correlation) else 0
        
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'information_ratio': information_ratio,
            'tracking_error': tracking_error,
            'beta': beta,
            'alpha': alpha_annualized,
            'r_squared': r_squared
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to calculate official metrics: {e}")
        raise

