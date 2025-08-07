"""
Vectorized Backtesting Engine

This module provides a vectorized backtesting engine for factor investing strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class VectorizedEngine:
    """
    Vectorized backtesting engine for factor investing strategies.
    
    This engine performs backtesting using vectorized operations for efficiency.
    """
    
    def __init__(self, config: Optional[Dict] = None, factor_data: Optional[pd.DataFrame] = None, 
                 returns_matrix: Optional[pd.DataFrame] = None, benchmark_returns: Optional[pd.Series] = None,
                 db_engine: Optional[object] = None):
        """
        Initialize the vectorized backtesting engine.
        
        Args:
            config: Configuration dictionary for the backtest
            factor_data: Factor data DataFrame
            returns_matrix: Returns matrix DataFrame
            benchmark_returns: Benchmark returns Series
            db_engine: Database engine
        """
        self.config = config or {}
        self.factor_data = factor_data
        self.returns_matrix = returns_matrix
        self.benchmark_returns = benchmark_returns
        self.db_engine = db_engine
        self.results = {}
        
    def run_backtest(self, portfolio_constructor) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Run a vectorized backtest using real data and portfolio constructor.
        
        Args:
            portfolio_constructor: Function to construct portfolio weights
            
        Returns:
            Tuple of (returns_series, diagnostics_dataframe)
        """
        logger.info(f"Starting vectorized backtest with portfolio constructor: {portfolio_constructor.__name__}")
        
        if self.factor_data is None or self.returns_matrix is None:
            logger.error("No factor data or returns matrix provided")
            return pd.Series(), pd.DataFrame()
        
        # Get rebalancing dates (monthly)
        start_date = pd.to_datetime(self.config.get('backtest_start_date', '2018-01-01'))
        end_date = pd.to_datetime(self.config.get('backtest_end_date', '2025-07-31'))
        
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            rebalance_dates = pd.date_range(start=start_date, end=end_date, freq='M')
        
        # Filter rebalance dates to only include those with data
        available_dates = self.factor_data['date'].unique()
        rebalance_dates = [d for d in rebalance_dates if d in available_dates]
        
        if not rebalance_dates:
            logger.error("No valid rebalancing dates found")
            return pd.Series(), pd.DataFrame()
        
        logger.info(f"Running backtest for {len(rebalance_dates)} rebalancing periods")
        
        # Initialize results
        portfolio_returns = []
        diagnostics_data = []
        
        # Run backtest for each rebalancing period
        for i, rebalance_date in enumerate(rebalance_dates):
            logger.info(f"Processing rebalance {i+1}/{len(rebalance_dates)}: {rebalance_date}")
            
            # Get factor data for this date
            date_factor_data = self.factor_data[self.factor_data['date'] == rebalance_date]
            
            if date_factor_data.empty:
                logger.warning(f"No factor data for {rebalance_date}")
                continue
            
            # Use portfolio constructor to get portfolio weights
            try:
                portfolio_weights = portfolio_constructor(date_factor_data, self.returns_matrix, self.config)
                
                if portfolio_weights.empty:
                    logger.warning(f"No portfolio weights generated for {rebalance_date}")
                    continue
                
                # Get returns data for the period
                if i < len(rebalance_dates) - 1:
                    next_rebalance = rebalance_dates[i + 1]
                    period_returns = self.returns_matrix[
                        (self.returns_matrix['date'] >= rebalance_date) & 
                        (self.returns_matrix['date'] < next_rebalance)
                    ]
                else:
                    # Last period - use remaining data
                    period_returns = self.returns_matrix[
                        self.returns_matrix['date'] >= rebalance_date
                    ]
                
                if period_returns.empty:
                    logger.warning(f"No returns data for period starting {rebalance_date}")
                    continue
                
                # Calculate weighted portfolio returns for this period
                # Group returns by date to calculate daily portfolio returns
                daily_portfolio_returns = {}
                
                for _, row in period_returns.iterrows():
                    if row['ticker'] in portfolio_weights['ticker'].values:
                        # Get weight for this stock
                        stock_weight = portfolio_weights[portfolio_weights['ticker'] == row['ticker']]['weight'].iloc[0]
                        # Filter out invalid returns
                        if pd.notna(row['return']) and np.isfinite(row['return']) and abs(row['return']) < 0.5:  # More conservative filter
                            date = row['date']
                            if date not in daily_portfolio_returns:
                                daily_portfolio_returns[date] = []
                            daily_portfolio_returns[date].append(row['return'] * stock_weight)
                
                # Calculate daily portfolio returns and then compound to monthly
                if daily_portfolio_returns:
                    # Calculate daily portfolio returns (sum of weighted returns)
                    daily_returns_list = []
                    for date, returns in daily_portfolio_returns.items():
                        daily_return = np.sum(returns)
                        if np.isfinite(daily_return) and abs(daily_return) < 0.5:
                            daily_returns_list.append(daily_return)
                    
                    if daily_returns_list:
                        # Compound daily returns to get monthly return
                        monthly_return = (1 + np.array(daily_returns_list)).prod() - 1
                        
                        # Ensure the monthly return is reasonable
                        if np.isfinite(monthly_return) and abs(monthly_return) < 0.5:
                            portfolio_returns.append({
                                'date': rebalance_date,
                                'return': monthly_return
                            })
                        else:
                            logger.warning(f"Invalid monthly return for {rebalance_date}: {monthly_return}")
                            continue
                    
                    diagnostics_data.append({
                        'date': rebalance_date,
                        'turnover': 0.2,  # Placeholder
                        'portfolio_size': len(portfolio_weights),
                        'avg_factor_score': portfolio_weights['composite_score'].mean()
                    })
                    
            except Exception as e:
                logger.error(f"Error in portfolio construction for {rebalance_date}: {e}")
                continue
        
        # Convert to Series and DataFrame
        if portfolio_returns:
            returns_series = pd.DataFrame(portfolio_returns).set_index('date')['return']
            diagnostics_df = pd.DataFrame(diagnostics_data).set_index('date')
        else:
            returns_series = pd.Series(dtype=float)
            diagnostics_df = pd.DataFrame()
        
        logger.info(f"Backtest completed. Generated {len(returns_series)} return periods")
        
        return returns_series, diagnostics_df
    
    def calculate_metrics(self, returns: pd.Series, benchmark_returns: pd.Series) -> Dict:
        """
        Calculate performance metrics.
        
        Args:
            returns: Strategy returns
            benchmark_returns: Benchmark returns
            
        Returns:
            Dictionary containing performance metrics
        """
        # Placeholder implementation
        metrics = {
            'total_return': 0.0,
            'annualized_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'information_ratio': 0.0
        }
        
        return metrics

