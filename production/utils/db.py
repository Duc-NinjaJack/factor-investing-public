"""
Database Utilities

This module contains utility functions for database operations.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import logging
from sqlalchemy import create_engine, text
from pathlib import Path

logger = logging.getLogger(__name__)

def create_db_connection(project_root: Path) -> Optional[object]:
    """
    Create a database connection.
    
    Args:
        project_root: Path to the project root directory
        
    Returns:
        Database engine or None if connection fails
    """
    try:
        logger.info("Creating database connection...")
        
        # Use the existing database infrastructure
        from production.database import get_engine
        
        engine = get_engine(environment='production')
        
        # Test the connection
        if engine is not None:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Database connection established successfully")
            return engine
        else:
            logger.error("Failed to get database engine")
            return None
        
    except Exception as e:
        logger.error(f"Failed to create database connection: {e}")
        return None

def load_all_data_for_backtest(config: Dict, db_engine: object) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Load all data required for backtesting from the real database.
    
    Args:
        config: Configuration dictionary
        db_engine: Database engine
        
    Returns:
        Tuple of (factor_data, daily_returns_matrix, benchmark_returns)
    """
    try:
        logger.info(f"Loading all data for period: {config.get('backtest_start_date')} to {config.get('backtest_end_date')}...")
        
        # 1. Load factor scores data
        logger.info("Loading factor scores data...")
        factor_query = """
        SELECT date, ticker, Quality_Composite as quality_score, Value_Composite as value_score, 
               Momentum_Composite as momentum_score, QVM_Composite as composite_score
        FROM factor_scores_qvm 
        WHERE date BETWEEN %s AND %s
        ORDER BY date, QVM_Composite DESC
        """
        
        factor_data = pd.read_sql(
            factor_query, 
            db_engine, 
            params=(config.get('backtest_start_date'), config.get('backtest_end_date'))
        )
        factor_data['date'] = pd.to_datetime(factor_data['date'])
        
        logger.info(f"Loaded {len(factor_data)} factor score rows for version '{config.get('signal', {}).get('db_strategy_version', 'qvm_v2.1.1_flat')}'.")
        
        # 2. Load price data for all tickers in factor data
        logger.info("Loading price data...")
        unique_tickers = factor_data['ticker'].unique()
        ticker_list = "', '".join(unique_tickers)
        
        price_query = f"""
        SELECT 
            trading_date as date,
            ticker,
            close_price
        FROM vcsc_daily_data_complete
        WHERE ticker IN ('{ticker_list}')
        AND trading_date >= '{factor_data['date'].min()}'
        AND trading_date <= '{factor_data['date'].max()}'
        ORDER BY trading_date, ticker
        """
        
        price_data = pd.read_sql(price_query, db_engine)
        price_data['date'] = pd.to_datetime(price_data['date'])
        
        # Convert to returns matrix
        price_matrix = price_data.pivot(index='date', columns='ticker', values='close_price')
        daily_returns_matrix = price_matrix.pct_change(fill_method=None).reset_index()
        daily_returns_matrix = daily_returns_matrix.melt(
            id_vars=['date'], 
            var_name='ticker', 
            value_name='return'
        ).dropna()
        
        logger.info(f"Loaded price data for {len(unique_tickers)} tickers, {len(price_data)} price records")
        
        # 3. Load benchmark data (VN-Index)
        logger.info("Loading benchmark data...")
        benchmark_query = f"""
        SELECT 
            date,
            close as close_price
        FROM etf_history
        WHERE ticker = 'VNINDEX'
        AND date >= '{factor_data['date'].min()}'
        AND date <= '{factor_data['date'].max()}'
        ORDER BY date
        """
        
        benchmark_data = pd.read_sql(benchmark_query, db_engine)
        benchmark_data['date'] = pd.to_datetime(benchmark_data['date'])
        benchmark_data['return'] = benchmark_data['close_price'].pct_change()
        
        # Create benchmark returns series
        benchmark_returns = benchmark_data.set_index('date')['return'].dropna()
        
        logger.info(f"Loaded benchmark data: {len(benchmark_data)} records")
        
        return factor_data, daily_returns_matrix, benchmark_returns
        
    except Exception as e:
        logger.error(f"Failed to load data for backtest: {e}")
        raise

