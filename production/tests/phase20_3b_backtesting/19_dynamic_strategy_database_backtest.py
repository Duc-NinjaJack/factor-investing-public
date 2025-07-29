#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic Strategy Database Backtest
==================================
Component: Dynamic QVM Strategy with Database Data
Purpose: Apply the 15b dynamic regime-switching strategy to database data
Author: Assistant
Date Created: 2025-07-29
Status: DYNAMIC STRATEGY VALIDATION

This script applies the dynamic composite strategy from 15b to database data:
- Uses vcsc_daily_data_complete for price data
- Uses factor_scores_qvm for factor data
- Implements regime-switching factor weights (QVM vs QV-Reversal)
- Top 25 stocks, monthly rebalancing, 30 bps transaction costs
- Compares against static QVM strategy and benchmark
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, text
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
from datetime import datetime, timedelta
import logging
import pickle

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DynamicStrategyDatabaseBacktest:
    """
    Dynamic regime-switching strategy backtest using database data.
    """
    
    def __init__(self, config_path: str = "../../../config/database.yml"):
        """Initialize the dynamic backtesting."""
        self.config_path = config_path
        self.engine = self._create_database_engine()
        
        # Analysis parameters
        self.thresholds = {
            '10B_VND': 10_000_000_000,
            '3B_VND': 3_000_000_000
        }
        
        # Backtest parameters (matching 15b notebook)
        self.backtest_config = {
            'start_date': '2017-12-01',  # Match 15b start date
            'end_date': '2025-07-28',    # Match 15b end date
            'rebalance_freq': 'Q',       # Quarterly rebalancing (like 15b)
            'portfolio_size': 'quintile_5',  # Top 20% of stocks (like 15b)
            'transaction_cost': 0.003,   # 30 bps (like 15b)
            'initial_capital': 100_000_000  # 100M VND
        }
        
        # Only use 10B VND threshold (like 15b liquid universe)
        self.thresholds = {
            '10B_VND': 10_000_000_000  # 10B VND liquidity threshold
        }
        
        logger.info("Dynamic Strategy Database Backtest initialized")
    
    def _create_database_engine(self):
        """Create database engine."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            # Use production config
            db_config = config['production']
            
            connection_string = (
                f"mysql+pymysql://{db_config['username']}:{db_config['password']}"
                f"@{db_config['host']}/{db_config['schema_name']}"
            )
            return create_engine(connection_string, pool_recycle=3600)
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load all required data for backtesting."""
        logger.info("Loading data for dynamic strategy backtesting...")
        
        data = {}
        
        # Load ADTV data from pickle
        try:
            with open('unrestricted_universe_data.pkl', 'rb') as f:
                pickle_data = pickle.load(f)
            
            data['adtv_data'] = pickle_data['adtv']
            logger.info("✅ ADTV data loaded from pickle")
            
        except FileNotFoundError:
            logger.error("❌ Pickle file not found. Please run get_unrestricted_universe_data.py first.")
            raise
        
        # Load price data from vcsc_daily_data_complete (full historical data)
        price_query = """
        SELECT trading_date, ticker, close_price_adjusted
        FROM vcsc_daily_data_complete
        WHERE trading_date >= '2016-01-01'
        ORDER BY trading_date, ticker
        """
        data['price_data'] = pd.read_sql(price_query, self.engine)
        data['price_data']['trading_date'] = pd.to_datetime(data['price_data']['trading_date'])
        
        # Load factor scores from database (all three factors)
        factor_query = """
        SELECT date, ticker, Quality_Composite, Value_Composite, Momentum_Composite, QVM_Composite
        FROM factor_scores_qvm
        WHERE date >= '2016-01-01'
        ORDER BY date, ticker
        """
        data['factor_scores'] = pd.read_sql(factor_query, self.engine)
        data['factor_scores']['date'] = pd.to_datetime(data['factor_scores']['date'])
        
        # Load benchmark data (VNINDEX)
        benchmark_query = """
        SELECT date, close
        FROM etf_history
        WHERE ticker = 'VNINDEX' AND date >= '2016-01-01'
        ORDER BY date
        """
        data['benchmark'] = pd.read_sql(benchmark_query, self.engine)
        data['benchmark']['date'] = pd.to_datetime(data['benchmark']['date'])
        
        logger.info(f"✅ All data loaded successfully")
        logger.info(f"   - Price data: {len(data['price_data']):,} records")
        logger.info(f"   - Factor scores: {len(data['factor_scores']):,} records")
        logger.info(f"   - Benchmark: {len(data['benchmark']):,} records")
        logger.info(f"   - ADTV data: {data['adtv_data'].shape}")
        
        return data
    
    def create_market_regime_signals(self, benchmark_returns: pd.Series) -> pd.DataFrame:
        """
        Create market regime signals based on benchmark volatility and returns.
        This is a simplified version of the regime detection from Phase 8.
        """
        logger.info("Creating market regime signals...")
        
        # Calculate rolling volatility (60-day window)
        rolling_vol = benchmark_returns.rolling(60).std() * np.sqrt(252)
        
        # Calculate rolling returns (60-day window)
        rolling_returns = benchmark_returns.rolling(60).mean() * 252
        
        # Define regime thresholds
        vol_threshold_high = rolling_vol.quantile(0.75)
        vol_threshold_low = rolling_vol.quantile(0.25)
        return_threshold = 0.10  # 10% annual return threshold
        
        # Create regime signals
        regimes = pd.DataFrame(index=benchmark_returns.index)
        regimes['volatility'] = rolling_vol
        regimes['returns'] = rolling_returns
        
        # Regime classification
        conditions = [
            (rolling_vol > vol_threshold_high) & (rolling_returns < -return_threshold),
            (rolling_vol > vol_threshold_high) & (rolling_returns >= -return_threshold),
            (rolling_vol <= vol_threshold_high) & (rolling_vol > vol_threshold_low) & (rolling_returns >= return_threshold),
            (rolling_vol <= vol_threshold_low) & (rolling_returns >= return_threshold)
        ]
        choices = ['Stress', 'Bear', 'Bull', 'Sideways']
        
        regimes['regime'] = np.select(conditions, choices, default='Sideways')
        
        # Forward fill regimes to avoid gaps
        regimes['regime'] = regimes['regime'].fillna(method='ffill')
        
        logger.info(f"✅ Market regime signals created")
        logger.info(f"   - Regime distribution: {regimes['regime'].value_counts().to_dict()}")
        
        return regimes
    
    def prepare_data_for_backtesting(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Prepare data for backtesting."""
        logger.info("Preparing data for backtesting...")
        
        # Log raw data info
        logger.info(f"Raw price data info:")
        logger.info(f"   - Shape: {data['price_data'].shape}")
        logger.info(f"   - Date range: {data['price_data']['trading_date'].min()} to {data['price_data']['trading_date'].max()}")
        logger.info(f"   - Unique dates: {data['price_data']['trading_date'].nunique()}")
        logger.info(f"   - Unique tickers: {data['price_data']['ticker'].nunique()}")
        
        # Check for missing values
        missing_prices = data['price_data']['close_price_adjusted'].isnull().sum()
        logger.info(f"   - Missing prices: {missing_prices}")
        
        # Prepare price data
        price_pivot = data['price_data'].pivot(
            index='trading_date', columns='ticker', values='close_price_adjusted'
        )
        
        logger.info(f"Price pivot info:")
        logger.info(f"   - Shape: {price_pivot.shape}")
        logger.info(f"   - Date range: {price_pivot.index.min()} to {price_pivot.index.max()}")
        logger.info(f"   - Missing values: {price_pivot.isnull().sum().sum()}")
        
        # Forward fill missing values to preserve historical data
        price_pivot_filled = price_pivot.fillna(method='ffill')
        logger.info(f"   - Missing values after forward fill: {price_pivot_filled.isnull().sum().sum()}")
        
        # Replace zero prices with NaN to avoid division by zero in returns calculation
        price_pivot_filled = price_pivot_filled.replace(0, np.nan)
        logger.info(f"   - Zero prices replaced with NaN")
        
        # Fill remaining missing values with forward fill again
        price_pivot_filled = price_pivot_filled.fillna(method='ffill')
        logger.info(f"   - Missing values after final fill: {price_pivot_filled.isnull().sum().sum()}")
        
        # Calculate returns
        returns = price_pivot_filled.pct_change()
        # Fill the first row (which is NaN after pct_change) with 0 instead of dropping
        returns = returns.fillna(0)
        
        logger.info(f"Returns info:")
        logger.info(f"   - Shape: {returns.shape}")
        logger.info(f"   - Date range: {returns.index.min()} to {returns.index.max()}")
        logger.info(f"   - Missing values: {returns.isnull().sum().sum()}")
        logger.info(f"   - Infinite values: {np.isinf(returns.values).sum()}")
        logger.info(f"   - Returns mean: {returns.mean().mean():.6f}")
        logger.info(f"   - Returns std: {returns.std().mean():.6f}")
        
        # Prepare factor data
        factor_pivot = data['factor_scores'].pivot(
            index='date', columns='ticker', values='QVM_Composite'
        )
        
        # Prepare individual factors
        quality_pivot = data['factor_scores'].pivot(
            index='date', columns='ticker', values='Quality_Composite'
        )
        value_pivot = data['factor_scores'].pivot(
            index='date', columns='ticker', values='Value_Composite'
        )
        momentum_pivot = data['factor_scores'].pivot(
            index='date', columns='ticker', values='Momentum_Composite'
        )
        
        # Prepare benchmark returns
        benchmark_returns = data['benchmark'].set_index('date')['close'].pct_change().dropna()
        
        # Log date ranges before alignment
        logger.info(f"Date ranges before alignment:")
        logger.info(f"   - Price data: {returns.index.min()} to {returns.index.max()} ({len(returns)} dates)")
        logger.info(f"   - Factor data: {factor_pivot.index.min()} to {factor_pivot.index.max()} ({len(factor_pivot)} dates)")
        logger.info(f"   - Benchmark: {benchmark_returns.index.min()} to {benchmark_returns.index.max()} ({len(benchmark_returns)} dates)")
        
        # Create market regime signals
        market_regimes = self.create_market_regime_signals(benchmark_returns)
        
        # Align all data
        common_dates = returns.index.intersection(factor_pivot.index).intersection(benchmark_returns.index)
        returns = returns.loc[common_dates]
        factor_pivot = factor_pivot.loc[common_dates]
        quality_pivot = quality_pivot.loc[common_dates]
        value_pivot = value_pivot.loc[common_dates]
        momentum_pivot = momentum_pivot.loc[common_dates]
        benchmark_returns = benchmark_returns.loc[common_dates]
        market_regimes = market_regimes.loc[common_dates]
        
        logger.info(f"✅ Data prepared for backtesting")
        logger.info(f"   - Common dates: {len(common_dates)}")
        logger.info(f"   - Common date range: {common_dates.min()} to {common_dates.max()}")
        logger.info(f"   - Returns shape: {returns.shape}")
        logger.info(f"   - Factor scores shape: {factor_pivot.shape}")
        
        return {
            'returns': returns,
            'factor_scores': factor_pivot,
            'quality_scores': quality_pivot,
            'value_scores': value_pivot,
            'momentum_scores': momentum_pivot,
            'benchmark_returns': benchmark_returns,
            'market_regimes': market_regimes,
            'adtv_data': data['adtv_data']
        }
    
    def run_dynamic_backtest(self, threshold_name: str, threshold_value: int, 
                           prepared_data: Dict[str, pd.DataFrame]) -> Dict:
        """Run dynamic backtest with regime-switching factor weights."""
        logger.info(f"Running dynamic backtest for {threshold_name}...")
        
        returns = prepared_data['returns']
        quality_scores = prepared_data['quality_scores']
        value_scores = prepared_data['value_scores']
        momentum_scores = prepared_data['momentum_scores']
        adtv_data = prepared_data['adtv_data']
        market_regimes = prepared_data['market_regimes']
        
        # Rebalancing dates (monthly)
        rebalance_dates = pd.date_range(
            start=returns.index.min(),
            end=returns.index.max(),
            freq=self.backtest_config['rebalance_freq']
        )
        
        # Filter to dates with data
        rebalance_dates = rebalance_dates[rebalance_dates.isin(returns.index)]
        
        logger.info(f"   - Total rebalancing dates: {len(rebalance_dates)}")
        logger.info(f"   - Date range: {rebalance_dates.min()} to {rebalance_dates.max()}")
        
        portfolio_returns_dict = {}
        portfolio_holdings = []
        portfolio_values = [self.backtest_config['initial_capital']]
        
        for i, rebalance_date in enumerate(rebalance_dates[:-1]):
            next_rebalance = rebalance_dates[i + 1]
            
            # Get current market regime
            if rebalance_date in market_regimes.index:
                current_regime = market_regimes.loc[rebalance_date, 'regime']
            else:
                # Use forward fill
                current_regime = market_regimes.loc[:rebalance_date].iloc[-1]['regime']
            
            # Get factor scores as of rebalance date
            if rebalance_date in quality_scores.index:
                quality_date = quality_scores.loc[rebalance_date].dropna()
                value_date = value_scores.loc[rebalance_date].dropna()
                momentum_date = momentum_scores.loc[rebalance_date].dropna()
            else:
                # Use forward fill
                quality_date = quality_scores.loc[:rebalance_date].iloc[-1].dropna()
                value_date = value_scores.loc[:rebalance_date].iloc[-1].dropna()
                momentum_date = momentum_scores.loc[:rebalance_date].iloc[-1].dropna()
            
            # Get ADTV as of rebalance date
            if rebalance_date in adtv_data.index:
                adtv_scores = adtv_data.loc[rebalance_date].dropna()
            else:
                # Use forward fill
                adtv_scores = adtv_data.loc[:rebalance_date].iloc[-1].dropna()
            
            # Apply liquidity filter
            liquid_stocks = adtv_scores[adtv_scores >= threshold_value].index
            available_stocks = quality_date.index.intersection(value_date.index).intersection(
                momentum_date.index).intersection(liquid_stocks)
            
            logger.debug(f"   - {rebalance_date}: {len(available_stocks)} liquid stocks available")
            
            # Check minimum portfolio size requirement
            min_portfolio_size = 10 if self.backtest_config['portfolio_size'] == 'quintile_5' else self.backtest_config['portfolio_size']
            if len(available_stocks) < min_portfolio_size:
                logger.warning(f"   - {rebalance_date}: Only {len(available_stocks)} stocks available, skipping")
                continue
            
            # Normalize individual factor scores
            quality_z = (quality_date[available_stocks] - quality_date[available_stocks].mean()) / quality_date[available_stocks].std()
            value_z = (value_date[available_stocks] - value_date[available_stocks].mean()) / value_date[available_stocks].std()
            momentum_z = (momentum_date[available_stocks] - momentum_date[available_stocks].mean()) / momentum_date[available_stocks].std()
            
            # Apply dynamic weighting based on regime
            is_stress_period = current_regime in ['Bear', 'Stress']
            
            if is_stress_period:
                # Defensive QV-Reversal blend
                dynamic_composite = 0.45 * quality_z + 0.45 * value_z - 0.10 * momentum_z
            else:
                # Traditional QVM blend
                dynamic_composite = 0.40 * quality_z + 0.30 * value_z + 0.30 * momentum_z
            
            # Select top stocks (quintile 5 - top 20%)
            if self.backtest_config['portfolio_size'] == 'quintile_5':
                q5_cutoff = dynamic_composite.quantile(0.8)  # Top 20%
                top_stocks = dynamic_composite[dynamic_composite >= q5_cutoff].index
            else:
                # Fallback to fixed number
                top_stocks = dynamic_composite.nlargest(self.backtest_config['portfolio_size']).index
            
            # Equal weight portfolio
            weights = pd.Series(1.0 / len(top_stocks), index=top_stocks)
            
            # Calculate period returns
            period_returns = returns.loc[rebalance_date:next_rebalance, top_stocks]
            portfolio_return = (period_returns * weights).sum(axis=1)
            
            # Apply transaction costs (only on rebalancing)
            if i > 0:  # Not the first rebalancing
                portfolio_return.iloc[0] -= self.backtest_config['transaction_cost']
            
            # Store returns for this period
            for date, ret in portfolio_return.items():
                portfolio_returns_dict[date] = ret
            
            portfolio_holdings.append({
                'date': rebalance_date,
                'stocks': list(top_stocks),
                'weights': weights.to_dict(),
                'universe_size': len(available_stocks),
                'regime': current_regime,
                'portfolio_value': portfolio_values[-1]
            })
            
            # Update portfolio value
            period_return_series = pd.Series(portfolio_return.values, index=period_returns.index)
            cumulative_return = (1 + period_return_series).prod()
            portfolio_values.append(portfolio_values[-1] * cumulative_return)
        
        # Create portfolio returns series
        portfolio_returns_series = pd.Series(portfolio_returns_dict)
        
        logger.info(f"   - Portfolio returns: {len(portfolio_returns_series)} dates")
        logger.info(f"   - Portfolio returns range: {portfolio_returns_series.index.min()} to {portfolio_returns_series.index.max()}")
        logger.info(f"   - Portfolio returns mean: {portfolio_returns_series.mean():.6f}")
        logger.info(f"   - Portfolio returns std: {portfolio_returns_series.std():.6f}")
        
        # Align with benchmark
        aligned_data = pd.DataFrame({
            'portfolio': portfolio_returns_series,
            'benchmark': prepared_data['benchmark_returns']
        }).dropna()
        
        logger.info(f"   - Aligned data: {len(aligned_data)} dates")
        
        # Calculate metrics
        annual_return = aligned_data['portfolio'].mean() * 252
        annual_vol = aligned_data['portfolio'].std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
        
        # Calculate drawdown
        cumulative_returns = (1 + aligned_data['portfolio']).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calculate alpha and beta
        covariance = np.cov(aligned_data['portfolio'], aligned_data['benchmark'])[0, 1]
        benchmark_var = aligned_data['benchmark'].var()
        beta = covariance / benchmark_var if benchmark_var > 0 else 0
        alpha = (aligned_data['portfolio'].mean() - beta * aligned_data['benchmark'].mean()) * 252
        
        # Calculate information ratio
        excess_returns = aligned_data['portfolio'] - aligned_data['benchmark']
        tracking_error = excess_returns.std() * np.sqrt(252)
        information_ratio = (excess_returns.mean() * 252) / tracking_error if tracking_error > 0 else 0
        
        # Calculate Calmar ratio (fix division by zero)
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 and abs(max_drawdown) > 1e-10 else 0
        
        logger.info(f"   - Annual return: {annual_return:.4f}")
        logger.info(f"   - Annual vol: {annual_vol:.4f}")
        logger.info(f"   - Sharpe: {sharpe_ratio:.4f}")
        logger.info(f"   - Max drawdown: {max_drawdown:.4f}")
        logger.info(f"   - Calmar ratio: {calmar_ratio:.4f}")
        
        return {
            'threshold_name': threshold_name,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'alpha': alpha,
            'beta': beta,
            'information_ratio': information_ratio,
            'calmar_ratio': calmar_ratio,
            'portfolio_returns': portfolio_returns_series,
            'portfolio_holdings': portfolio_holdings,
            'regime_distribution': pd.DataFrame(portfolio_holdings)['regime'].value_counts().to_dict()
        }
    
    def run_static_backtest(self, threshold_name: str, threshold_value: int, 
                          prepared_data: Dict[str, pd.DataFrame]) -> Dict:
        """Run static backtest using QVM_Composite (for comparison)."""
        logger.info(f"Running static backtest for {threshold_name}...")
        
        returns = prepared_data['returns']
        factor_scores = prepared_data['factor_scores']
        adtv_data = prepared_data['adtv_data']
        
        # Rebalancing dates (monthly)
        rebalance_dates = pd.date_range(
            start=returns.index.min(),
            end=returns.index.max(),
            freq=self.backtest_config['rebalance_freq']
        )
        
        # Filter to dates with data
        rebalance_dates = rebalance_dates[rebalance_dates.isin(returns.index)]
        
        logger.info(f"   - Total rebalancing dates: {len(rebalance_dates)}")
        logger.info(f"   - Date range: {rebalance_dates.min()} to {rebalance_dates.max()}")
        
        portfolio_returns_dict = {}
        portfolio_holdings = []
        portfolio_values = [self.backtest_config['initial_capital']]
        
        for i, rebalance_date in enumerate(rebalance_dates[:-1]):
            next_rebalance = rebalance_dates[i + 1]
            
            # Get factor scores as of rebalance date
            if rebalance_date in factor_scores.index:
                factor_scores_date = factor_scores.loc[rebalance_date].dropna()
            else:
                # Use forward fill
                factor_scores_date = factor_scores.loc[:rebalance_date].iloc[-1].dropna()
            
            # Get ADTV as of rebalance date
            if rebalance_date in adtv_data.index:
                adtv_scores = adtv_data.loc[rebalance_date].dropna()
            else:
                # Use forward fill
                adtv_scores = adtv_data.loc[:rebalance_date].iloc[-1].dropna()
            
            # Apply liquidity filter
            liquid_stocks = adtv_scores[adtv_scores >= threshold_value].index
            available_stocks = factor_scores_date.index.intersection(liquid_stocks)
            
            logger.debug(f"   - {rebalance_date}: {len(available_stocks)} liquid stocks available")
            
            # Check minimum portfolio size requirement
            min_portfolio_size = 10 if self.backtest_config['portfolio_size'] == 'quintile_5' else self.backtest_config['portfolio_size']
            if len(available_stocks) < min_portfolio_size:
                logger.warning(f"   - {rebalance_date}: Only {len(available_stocks)} stocks available, skipping")
                continue
            
            # Select top stocks by QVM_Composite score (static)
            if self.backtest_config['portfolio_size'] == 'quintile_5':
                q5_cutoff = factor_scores_date[available_stocks].quantile(0.8)  # Top 20%
                top_stocks = factor_scores_date[available_stocks][factor_scores_date[available_stocks] >= q5_cutoff].index
            else:
                # Fallback to fixed number
                top_stocks = factor_scores_date[available_stocks].nlargest(
                    self.backtest_config['portfolio_size']
                ).index
            
            # Equal weight portfolio
            weights = pd.Series(1.0 / len(top_stocks), index=top_stocks)
            
            # Calculate period returns
            period_returns = returns.loc[rebalance_date:next_rebalance, top_stocks]
            portfolio_return = (period_returns * weights).sum(axis=1)
            
            # Apply transaction costs (only on rebalancing)
            if i > 0:  # Not the first rebalancing
                portfolio_return.iloc[0] -= self.backtest_config['transaction_cost']
            
            # Store returns for this period
            for date, ret in portfolio_return.items():
                portfolio_returns_dict[date] = ret
            
            portfolio_holdings.append({
                'date': rebalance_date,
                'stocks': list(top_stocks),
                'weights': weights.to_dict(),
                'universe_size': len(available_stocks),
                'regime': 'Static',
                'portfolio_value': portfolio_values[-1]
            })
            
            # Update portfolio value
            period_return_series = pd.Series(portfolio_return.values, index=period_returns.index)
            cumulative_return = (1 + period_return_series).prod()
            portfolio_values.append(portfolio_values[-1] * cumulative_return)
        
        # Create portfolio returns series
        portfolio_returns_series = pd.Series(portfolio_returns_dict)
        
        logger.info(f"   - Portfolio returns: {len(portfolio_returns_series)} dates")
        logger.info(f"   - Portfolio returns range: {portfolio_returns_series.index.min()} to {portfolio_returns_series.index.max()}")
        logger.info(f"   - Portfolio returns mean: {portfolio_returns_series.mean():.6f}")
        logger.info(f"   - Portfolio returns std: {portfolio_returns_series.std():.6f}")
        
        # Align with benchmark
        aligned_data = pd.DataFrame({
            'portfolio': portfolio_returns_series,
            'benchmark': prepared_data['benchmark_returns']
        }).dropna()
        
        logger.info(f"   - Aligned data: {len(aligned_data)} dates")
        
        # Calculate metrics
        annual_return = aligned_data['portfolio'].mean() * 252
        annual_vol = aligned_data['portfolio'].std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
        
        # Calculate drawdown
        cumulative_returns = (1 + aligned_data['portfolio']).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calculate alpha and beta
        covariance = np.cov(aligned_data['portfolio'], aligned_data['benchmark'])[0, 1]
        benchmark_var = aligned_data['benchmark'].var()
        beta = covariance / benchmark_var if benchmark_var > 0 else 0
        alpha = (aligned_data['portfolio'].mean() - beta * aligned_data['benchmark'].mean()) * 252
        
        # Calculate information ratio
        excess_returns = aligned_data['portfolio'] - aligned_data['benchmark']
        tracking_error = excess_returns.std() * np.sqrt(252)
        information_ratio = (excess_returns.mean() * 252) / tracking_error if tracking_error > 0 else 0
        
        # Calculate Calmar ratio (fix division by zero)
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 and abs(max_drawdown) > 1e-10 else 0
        
        logger.info(f"   - Annual return: {annual_return:.4f}")
        logger.info(f"   - Annual vol: {annual_vol:.4f}")
        logger.info(f"   - Sharpe: {sharpe_ratio:.4f}")
        logger.info(f"   - Max drawdown: {max_drawdown:.4f}")
        logger.info(f"   - Calmar ratio: {calmar_ratio:.4f}")
        
        return {
            'threshold_name': threshold_name,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'alpha': alpha,
            'beta': beta,
            'information_ratio': information_ratio,
            'calmar_ratio': calmar_ratio,
            'portfolio_returns': portfolio_returns_series,
            'portfolio_holdings': portfolio_holdings,
            'regime_distribution': {'Static': len(portfolio_holdings)}
        }
    
    def run_comparative_backtests(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """Run both dynamic and static backtests for comparison."""
        logger.info("Running comparative backtests...")
        
        backtest_results = {}
        
        # Run dynamic backtests
        for threshold_name, threshold_value in self.thresholds.items():
            logger.info(f"Running dynamic backtest for {threshold_name}...")
            dynamic_result = self.run_dynamic_backtest(threshold_name, threshold_value, data)
            backtest_results[f"{threshold_name}_Dynamic"] = dynamic_result
            
            logger.info(f"Running static backtest for {threshold_name}...")
            static_result = self.run_static_backtest(threshold_name, threshold_value, data)
            backtest_results[f"{threshold_name}_Static"] = static_result
        
        return backtest_results
    
    def create_performance_visualizations(self, backtest_results: Dict[str, Dict], prepared_data: Dict[str, pd.DataFrame]):
        """Create performance comparison visualizations."""
        logger.info("Creating performance visualizations...")
        
        # Prepare data for plotting
        strategies = []
        metrics = []
        
        for strategy_name, result in backtest_results.items():
            strategies.append(strategy_name)
            metrics.append({
                'Annual Return (%)': result['annual_return'] * 100,
                'Sharpe Ratio': result['sharpe_ratio'],
                'Max Drawdown (%)': result['max_drawdown'] * 100,
                'Alpha (%)': result['alpha'] * 100,
                'Beta': result['beta'],
                'Information Ratio': result['information_ratio'],
                'Calmar Ratio': result['calmar_ratio']
            })
        
        metrics_df = pd.DataFrame(metrics, index=strategies)
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Dynamic vs Static Strategy Performance Comparison', fontsize=16, fontweight='bold')
        
        # Annual Return comparison
        ax1 = axes[0, 0]
        metrics_df['Annual Return (%)'].plot(kind='bar', ax=ax1, color=['#16A085', '#34495E', '#27AE60', '#C0392B'])
        ax1.set_title('Annual Return (%)')
        ax1.set_ylabel('Return (%)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Sharpe Ratio comparison
        ax2 = axes[0, 1]
        metrics_df['Sharpe Ratio'].plot(kind='bar', ax=ax2, color=['#16A085', '#34495E', '#27AE60', '#C0392B'])
        ax2.set_title('Sharpe Ratio')
        ax2.set_ylabel('Sharpe Ratio')
        ax2.tick_params(axis='x', rotation=45)
        
        # Max Drawdown comparison
        ax3 = axes[1, 0]
        metrics_df['Max Drawdown (%)'].plot(kind='bar', ax=ax3, color=['#16A085', '#34495E', '#27AE60', '#C0392B'])
        ax3.set_title('Max Drawdown (%)')
        ax3.set_ylabel('Drawdown (%)')
        ax3.tick_params(axis='x', rotation=45)
        
        # Information Ratio comparison
        ax4 = axes[1, 1]
        metrics_df['Information Ratio'].plot(kind='bar', ax=ax4, color=['#16A085', '#34495E', '#27AE60', '#C0392B'])
        ax4.set_title('Information Ratio')
        ax4.set_ylabel('Information Ratio')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('img/dynamic_vs_static_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create cumulative returns plot
        plt.figure(figsize=(14, 8))
        
        for strategy_name, result in backtest_results.items():
            if 'portfolio_returns' in result:
                cumulative_returns = (1 + result['portfolio_returns']).cumprod()
                plt.plot(cumulative_returns.index, cumulative_returns.values, 
                        label=strategy_name, linewidth=2)
        
        # Add benchmark
        benchmark_cumulative = (1 + prepared_data['benchmark_returns']).cumprod()
        plt.plot(benchmark_cumulative.index, benchmark_cumulative.values, 
                label='VN-Index', linestyle='--', linewidth=2, color='black')
        
        plt.title('Cumulative Returns: Dynamic vs Static Strategies', fontsize=16, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig('img/dynamic_vs_static_cumulative_returns.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return metrics_df
    
    def generate_comprehensive_report(self, backtest_results: Dict[str, Dict], metrics_df: pd.DataFrame) -> str:
        """Generate comprehensive analysis report."""
        logger.info("Generating comprehensive report...")
        
        report = []
        report.append("# Dynamic Strategy Database Backtest Report")
        report.append("")
        report.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("**Purpose:** Compare dynamic regime-switching vs static QVM strategies")
        report.append("**Data Source:** Database (vcsc_daily_data_complete)")
        report.append("")
        
        report.append("## 🎯 Executive Summary")
        report.append("")
        report.append("This analysis compares the performance of:")
        report.append("- **Dynamic Strategy:** Regime-switching factor weights (QVM vs QV-Reversal)")
        report.append("- **Static Strategy:** Fixed QVM_Composite factor weights")
        report.append("- **Parameters:** Top 25 stocks, monthly rebalancing, 30 bps costs")
        report.append("")
        
        report.append("## 📊 Performance Comparison")
        report.append("")
        report.append("| Strategy | Annual Return (%) | Sharpe Ratio | Max Drawdown (%) | Alpha (%) | Beta | Information Ratio |")
        report.append("|----------|------------------|--------------|------------------|-----------|------|-------------------|")
        
        for strategy_name, result in backtest_results.items():
            report.append(f"| {strategy_name} | {result['annual_return']*100:.2f} | {result['sharpe_ratio']:.2f} | {result['max_drawdown']*100:.2f} | {result['alpha']*100:.2f} | {result['beta']:.2f} | {result['information_ratio']:.2f} |")
        
        report.append("")
        
        report.append("## 🔍 Key Findings")
        report.append("")
        
        # Find best performing strategies
        best_return = metrics_df['Annual Return (%)'].idxmax()
        best_sharpe = metrics_df['Sharpe Ratio'].idxmax()
        best_drawdown = metrics_df['Max Drawdown (%)'].idxmin()
        
        report.append(f"**Best Annual Return:** {best_return} ({metrics_df.loc[best_return, 'Annual Return (%)']:.2f}%)")
        report.append(f"**Best Sharpe Ratio:** {best_sharpe} ({metrics_df.loc[best_sharpe, 'Sharpe Ratio']:.2f})")
        report.append(f"**Best Risk Control:** {best_drawdown} ({metrics_df.loc[best_drawdown, 'Max Drawdown (%)']:.2f}%)")
        report.append("")
        
        # Regime analysis for dynamic strategies
        report.append("## 📈 Regime Analysis (Dynamic Strategies)")
        report.append("")
        
        for strategy_name, result in backtest_results.items():
            if 'Dynamic' in strategy_name and 'regime_distribution' in result:
                report.append(f"### {strategy_name}")
                report.append("")
                for regime, count in result['regime_distribution'].items():
                    percentage = (count / sum(result['regime_distribution'].values())) * 100
                    report.append(f"- **{regime}:** {count} rebalances ({percentage:.1f}%)")
                report.append("")
        
        report.append("## 🎯 Conclusions")
        report.append("")
        
        # Determine if dynamic strategy outperforms
        dynamic_10b = backtest_results.get('10B_VND_Dynamic')
        static_10b = backtest_results.get('10B_VND_Static')
        
        if dynamic_10b and static_10b:
            dynamic_better = dynamic_10b['sharpe_ratio'] > static_10b['sharpe_ratio']
            
            if dynamic_better:
                report.append("✅ **Dynamic strategy outperforms static strategy**")
                report.append(f"- Dynamic Sharpe: {dynamic_10b['sharpe_ratio']:.2f}")
                report.append(f"- Static Sharpe: {static_10b['sharpe_ratio']:.2f}")
                report.append("- Regime-switching logic provides better risk-adjusted returns")
            else:
                report.append("⚠️ **Static strategy outperforms dynamic strategy**")
                report.append(f"- Dynamic Sharpe: {dynamic_10b['sharpe_ratio']:.2f}")
                report.append(f"- Static Sharpe: {static_10b['sharpe_ratio']:.2f}")
                report.append("- Regime-switching logic may need refinement")
        
        report.append("")
        report.append("## 📋 Recommendations")
        report.append("")
        report.append("1. **Strategy Selection:** Choose based on risk tolerance and performance objectives")
        report.append("2. **Regime Detection:** Refine regime detection methodology if needed")
        report.append("3. **Parameter Tuning:** Optimize factor weights for different regimes")
        report.append("4. **Risk Management:** Implement additional risk controls for stress periods")
        report.append("")
        
        report.append("---")
        report.append(f"**Analysis completed:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return "\n".join(report)
    
    def run_complete_analysis(self):
        """Run the complete dynamic strategy analysis."""
        logger.info("🚀 Starting complete dynamic strategy analysis...")
        
        try:
            # Load data
            data = self.load_data()
            
            # Prepare data for backtesting
            prepared_data = self.prepare_data_for_backtesting(data)
            
            # Run comparative backtests
            backtest_results = self.run_comparative_backtests(prepared_data)
            
            # Create visualizations
            metrics_df = self.create_performance_visualizations(backtest_results, prepared_data)
            
            # Generate report
            report = self.generate_comprehensive_report(backtest_results, metrics_df)
            
            # Save report
            with open('dynamic_strategy_database_backtest_report.md', 'w') as f:
                f.write(report)
            
            # Save results
            with open('dynamic_strategy_database_backtest_results.pkl', 'wb') as f:
                pickle.dump({
                    'backtest_results': backtest_results,
                    'metrics_df': metrics_df,
                    'prepared_data': prepared_data
                }, f)
            
            logger.info("✅ Complete dynamic strategy analysis finished successfully!")
            logger.info("📊 Results saved to dynamic_strategy_database_backtest_report.md")
            logger.info("📈 Visualizations saved as PNG files")
            
            return backtest_results, metrics_df
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
            raise

def main():
    """Main execution function."""
    print("🔬 Dynamic Strategy Database Backtest")
    print("=" * 40)
    
    try:
        backtest = DynamicStrategyDatabaseBacktest()
        results, metrics = backtest.run_complete_analysis()
        
        print("\n✅ Dynamic strategy database backtest completed successfully!")
        print("📊 Check dynamic_strategy_database_backtest_report.md for detailed results.")
        
        print("\n📈 Key Results:")
        for strategy_name, result in results.items():
            print(f"   {strategy_name}: {result['annual_return']*100:.2f}% return, {result['sharpe_ratio']:.2f} Sharpe")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        raise

if __name__ == "__main__":
    main()