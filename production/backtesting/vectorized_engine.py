"""
Vectorized Backtesting Engine

This module provides a vectorized backtesting engine for factor investing strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import os

logger = logging.getLogger(__name__)

# Parallel safety: enforce 'spawn' start method early
try:
    from production.utils.parallel import ensure_spawn_start_method, disable_db_access_in_children
    ensure_spawn_start_method(logger)
    # Set DB disable flag before any potential fan-out usage
    disable_db_access_in_children()
except Exception:
    pass

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
        # Echo transaction cost configuration explicitly in logs
        try:
            tc_bps = float(self.config.get('transaction_cost_bps', 10))
            sl_bps = float(self.config.get('slippage_bps', 0))
            logger.info("Cost model: transaction_cost_bps=%.1f | slippage_bps=%.1f", tc_bps, sl_bps)
        except Exception:
            pass

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
        
        # Get rebalancing dates (monthly) with CalendarService anchor policy
        start_date = pd.to_datetime(self.config.get('backtest_start_date', '2018-01-01'))
        end_date = pd.to_datetime(self.config.get('backtest_end_date', '2025-07-31'))

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            target_month_starts = pd.date_range(start=start_date, end=end_date, freq='MS')

        # Build calendars from provided data
        price_index = pd.DatetimeIndex(sorted(pd.to_datetime(self.returns_matrix['date'].unique()))) if self.returns_matrix is not None and 'date' in self.returns_matrix.columns else pd.DatetimeIndex([])
        holdings_index = pd.DatetimeIndex(sorted(pd.to_datetime(self.factor_data['date'].unique()))) if self.factor_data is not None and 'date' in self.factor_data.columns else pd.DatetimeIndex([])

        try:
            from production.utils.calendar_service import CalendarService
            policy = self.config.get('rebalance_anchor_policy', 'nearest:3d')
            allow_override = bool(self.config.get('allow_nearest_override', False))
            reporting_lag_days = None
            try:
                fundamentals_cfg = self.config.get('fundamentals', {}) or {}
                if isinstance(fundamentals_cfg.get('reporting_lag_days'), (int, float)):
                    reporting_lag_days = int(fundamentals_cfg['reporting_lag_days'])
            except Exception:
                reporting_lag_days = None
            cal = CalendarService.from_price_series(
                price_index,
                holdings_index,
                logger=logger,
                default_policy=policy,
                allow_nearest_override=allow_override,
                reporting_lag_days=reporting_lag_days,
            )
            anchors = []
            for d in target_month_starts:
                anchor_type, anchor_date, delta = cal.choose_anchor(d, policy)
                anchors.append({'target': d, 'anchor': anchor_date, 'anchor_type': anchor_type, 'delta_days': delta})
            anchors_df = pd.DataFrame(anchors)
            # Deduplicate anchors and ensure they exist in factor data
            rebalance_dates = [pd.to_datetime(a) for a in anchors_df['anchor'].unique()]
            available_dates = set(pd.to_datetime(self.factor_data['date'].unique()))
            rebalance_dates = [d for d in rebalance_dates if d in available_dates]
            # Persist in results for metadata and to artifacts for auditability
            self.results['calendar_anchors'] = anchors_df
            try:
                from pathlib import Path
                run_dir = Path(self.config.get('artifacts_dir', 'artifacts')) / 'calendar'
                run_dir.mkdir(parents=True, exist_ok=True)
                try:
                    anchors_df.to_parquet(run_dir / 'calendar_anchors.parquet', index=False)
                except Exception:
                    anchors_df.to_csv(run_dir / 'calendar_anchors.csv', index=False)
            except Exception:
                pass
        except Exception as e:
            logger.debug(f"CalendarService not available or failed ({e}); falling back to factor-data month ends")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                fallback = pd.date_range(start=start_date, end=end_date, freq='M')
            available_dates = set(pd.to_datetime(self.factor_data['date'].unique()))
            rebalance_dates = [d for d in fallback if d in available_dates]
        
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
        
        # Emit reproducibility artifact if requested
        try:
            emit_artifacts = bool(self.config.get('emit_run_artifact', True))
            if emit_artifacts:
                from production.utils.run_artifacts import write_run_artifact
                seeds = {
                    'numpy_random_seed': int(self.config.get('numpy_seed', 0)) if 'numpy_seed' in self.config else None,
                }
                write_run_artifact(self.config, self.results.get('calendar_anchors'), seeds)
        except Exception:
            pass
        
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

