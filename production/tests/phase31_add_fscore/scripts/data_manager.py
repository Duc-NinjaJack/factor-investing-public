#!/usr/bin/env python3
"""
Data Manager for QVM Strategy
=============================

This module handles all data management operations:
- Sector mapping with caching
- Data loading with progressive fallback strategy
- Robust data operations with error handling
- Data validation and integrity checks

Author: Raymond
Created: August 17, 2025
"""

import logging
import pandas as pd
from typing import Dict, Optional, Tuple, Any
from pathlib import Path
from sqlalchemy import text


def get_sector_mapping(engine, logger: logging.Logger = None) -> pd.DataFrame:
    """Get sector mapping for all tickers with caching for performance."""
    # Check if we already have cached sector mapping
    if hasattr(get_sector_mapping, '_cached_sector_mapping') and get_sector_mapping._cached_sector_mapping is not None:
        if logger:
            logger.debug("📊 Using cached sector mapping")
        return get_sector_mapping._cached_sector_mapping
    
    try:
        if logger:
            logger.info("🔄 Loading sector mapping from database...")
        
        # Try to get real sector mapping from database first
        query = """
        SELECT DISTINCT ticker, 'Banking' as sector 
        FROM v_complete_banking_fundamentals 
        WHERE ticker IS NOT NULL 
        LIMIT 50
        """
        
        try:
            sector_data = pd.read_sql(query, engine)
            if len(sector_data) > 0:
                if logger:
                    logger.info(f"✅ Loaded real sector mapping: {len(sector_data)} records")
                # Cache the result
                get_sector_mapping._cached_sector_mapping = sector_data
                return sector_data
            else:
                if logger:
                    logger.warning("⚠️ No real sector data found, using fallback...")
        except Exception as e:
            if logger:
                logger.warning(f"⚠️ Could not load real sector data: {e}")
                logger.info("📊 Using fallback sector mapping...")
        
        # Fallback to predefined sector mapping (no synthetic data)
        fallback_sector_data = [
            {'ticker': 'VCB', 'sector': 'Banking'},
            {'ticker': 'TCB', 'sector': 'Banking'},
            {'ticker': 'BID', 'sector': 'Banking'},
            {'ticker': 'MBB', 'sector': 'Banking'},
            {'ticker': 'ACB', 'sector': 'Banking'},
            {'ticker': 'STB', 'sector': 'Banking'},
            {'ticker': 'EIB', 'sector': 'Banking'},
            {'ticker': 'HDB', 'sector': 'Banking'},
            {'ticker': 'TPB', 'sector': 'Banking'},
            {'ticker': 'SHB', 'sector': 'Banking'},
            {'ticker': 'LPB', 'sector': 'Banking'},
            {'ticker': 'MSB', 'sector': 'Banking'},
            {'ticker': 'VIB', 'sector': 'Banking'},
            {'ticker': 'OCB', 'sector': 'Banking'},
            {'ticker': 'SCB', 'sector': 'Banking'},
            {'ticker': 'VPB', 'sector': 'Banking'},
            {'ticker': 'BAB', 'sector': 'Banking'},
            {'ticker': 'NVB', 'sector': 'Banking'},
            {'ticker': 'KLB', 'sector': 'Banking'},
            {'ticker': 'SGB', 'sector': 'Banking'}
        ]
        
        if logger:
            logger.info("✅ Using fallback sector mapping")
        
        # Cache the fallback result
        get_sector_mapping._cached_sector_mapping = pd.DataFrame(fallback_sector_data)
        return get_sector_mapping._cached_sector_mapping
        
    except Exception as e:
        if logger:
            logger.error(f"Failed to get sector mapping: {e}")
        # Return empty DataFrame as fallback
        return pd.DataFrame(columns=['ticker', 'sector'])


def clear_sector_cache() -> None:
    """Clear the cached sector mapping to force reload."""
    if hasattr(get_sector_mapping, '_cached_sector_mapping'):
        delattr(get_sector_mapping, '_cached_sector_mapping')
        print("🗑️ Sector mapping cache cleared")


def get_sector_mapping_performance() -> Dict[str, Any]:
    """Get performance statistics for sector mapping."""
    if hasattr(get_sector_mapping, '_cached_sector_mapping'):
        cached_data = get_sector_mapping._cached_sector_mapping
        return {
            'cached': True,
            'records': len(cached_data),
            'memory_usage': cached_data.memory_usage(deep=True).sum()
        }
    else:
        return {
            'cached': False,
            'records': 0,
            'memory_usage': 0
        }


def load_data_with_fallback(engine, query: str, fallback_method: str = None, 
                           operation_name: str = "Data loading", 
                           logger: logging.Logger = None) -> pd.DataFrame:
    """
    Load data with progressive fallback strategy (NO synthetic data generation).
    
    Priority order:
    1. Real database data (preferred)
    2. Pre-calculated files (if available)
    3. Graceful error handling (return empty DataFrame)
    
    Args:
        engine: Database engine
        query: SQL query to execute
        fallback_method: Alternative data source method name
        operation_name: Name of the operation for logging
        logger: Logger instance for messages
    
    Returns:
        pd.DataFrame: Loaded data or empty DataFrame if all methods fail
    """
    try:
        # Method 1: Try real database data first
        if logger:
            logger.info(f"🔄 {operation_name}: Attempting real database query...")
        
        result = pd.read_sql(query, engine)
        
        if len(result) > 100:  # Sufficient real data
            if logger:
                logger.info(f"✅ {operation_name}: Loaded real data: {len(result)} records")
            return result
        else:
            if logger:
                logger.warning(f"⚠️ {operation_name}: Insufficient real data ({len(result)} records), attempting fallback...")
                
    except Exception as e:
        if logger:
            logger.warning(f"⚠️ {operation_name}: Database query failed: {e}")
            logger.info("📊 Attempting fallback method...")
    
    try:
        # Method 2: Try pre-calculated files if fallback method provided
        if fallback_method and hasattr(load_data_with_fallback, fallback_method):
            if logger:
                logger.info(f"🔄 {operation_name}: Attempting fallback method: {fallback_method}")
            
            result = getattr(load_data_with_fallback, fallback_method)()
            
            if result is not None and len(result) > 0:
                if logger:
                    logger.info(f"✅ {operation_name}: Loaded fallback data: {len(result)} records")
                return result
            else:
                if logger:
                    logger.warning(f"⚠️ {operation_name}: Fallback method returned empty result")
        else:
            if logger:
                logger.info(f"📊 {operation_name}: No fallback method specified")
                
    except Exception as e:
        if logger:
            logger.warning(f"⚠️ {operation_name}: Fallback method failed: {e}")
    
    # Method 3: Graceful failure - return empty DataFrame (NO synthetic data)
    if logger:
        logger.error(f"❌ {operation_name}: All data loading methods failed")
        logger.info(f"📊 {operation_name}: Returning empty DataFrame - no synthetic data generated")
    
    return pd.DataFrame()


def robust_data_operation(operation_name: str, operation_func, logger: logging.Logger = None, 
                         *args, **kwargs) -> Any:
    """
    Execute data operations with comprehensive error handling.
    
    Args:
        operation_name: Name of the operation for logging
        operation_func: Function to execute
        logger: Logger instance for messages
        *args, **kwargs: Arguments for the operation
    
    Returns:
        Result of operation or None if failed
    
    Raises:
        Exception: If operation fails and no fallback available
    """
    try:
        if logger:
            logger.info(f"🔄 Executing {operation_name}...")
        
        result = operation_func(*args, **kwargs)
        
        if result is not None and hasattr(result, '__len__') and len(result) > 0:
            if logger:
                logger.info(f"✅ {operation_name} completed: {len(result)} results")
            return result
        else:
            if logger:
                logger.warning(f"⚠️ {operation_name} returned empty result")
            return None
            
    except Exception as e:
        if logger:
            logger.error(f"❌ {operation_name} failed: {e}")
            
            # Log detailed error information
            import traceback
            logger.error(f"Stack trace: {traceback.format_exc()}")
        
        # Return None instead of raising (graceful degradation)
        return None


def get_most_recent_available_date(engine, logger: logging.Logger = None) -> pd.Timestamp:
    """Get the most recent available trading date for analysis."""
    try:
        # Get the most recent trading date from market data
        trading_date_query = text("""
            SELECT MAX(trading_date) as latest_trading_date
            FROM vcsc_daily_data_complete
            WHERE ticker IN ('AAA', 'AAM', 'ABT', 'ACB', 'ACC')
        """)
        
        trading_result = pd.read_sql(trading_date_query, engine)
        
        if not trading_result.empty and not pd.isna(trading_result.iloc[0]['latest_trading_date']):
            latest_trading_date = pd.Timestamp(trading_result.iloc[0]['latest_trading_date'])
            if logger:
                logger.info(f"📅 Found latest available trading date: {latest_trading_date.date()}")
            return latest_trading_date
        else:
            # No fallback - fail gracefully with warning
            if logger:
                logger.error("❌ No trading data found in database")
                logger.error("   Cannot determine most recent available date")
                logger.error("   Please ensure vcsc_daily_data_complete table has data")
            raise ValueError("No trading data available - cannot determine analysis date")
            
    except Exception as e:
        if logger:
            logger.error(f"❌ Error getting most recent available date: {e}")
            logger.error("   Cannot determine analysis date - strategy cannot proceed")
        raise ValueError(f"Failed to determine most recent available date: {e}")


def load_price_data_efficiently(engine, holdings_df: pd.DataFrame, logger: logging.Logger = None) -> pd.DataFrame:
    """Load price data efficiently for the holdings with fallback strategy (NO synthetic data)."""
    try:
        if logger:
            logger.info("Loading price data efficiently...")
        
        # Try to load real price data from database first
        try:
            unique_tickers = holdings_df['ticker'].unique()
            ticker_list = "', '".join(unique_tickers)
            
            price_query = f"""
            SELECT 
                trading_date as date,
                ticker,
                close_price
            FROM vcsc_daily_data_complete
            WHERE ticker IN ('{ticker_list}')
            AND trading_date >= '{holdings_df['date'].min()}'
            AND trading_date <= '{holdings_df['date'].max()}'
            ORDER BY trading_date, ticker
            """
            
            price_data = pd.read_sql(price_query, engine)
            price_data['date'] = pd.to_datetime(price_data['date'])
            
            if len(price_data) > 100:  # Sufficient real data
                if logger:
                    logger.info(f"✅ Loaded real price data: {len(price_data)} records")
                return price_data
            else:
                if logger:
                    logger.warning("⚠️ Limited real price data, attempting fallback...")
                
        except Exception as e:
            if logger:
                logger.warning(f"⚠️ Could not load real price data: {e}")
                logger.info("📊 Attempting fallback method...")
        
        # Try to load from pre-calculated files
        try:
            fallback_file = Path("docs/18b_complete_price_data.csv")
            if fallback_file.exists():
                if logger:
                    logger.info("📁 Using pre-calculated price data for speed...")
                
                price_df = pd.read_csv(fallback_file)
                price_df['date'] = pd.to_datetime(price_df['date'])
                
                # Filter to our holdings
                filtered_price = price_df[price_df['ticker'].isin(holdings_df['ticker'].unique())]
                if len(filtered_price) > 0:
                    if logger:
                        logger.info(f"✅ Loaded fallback price data: {len(filtered_price)} records")
                    return filtered_price
                else:
                    if logger:
                        logger.warning("⚠️ Fallback price data has no matching tickers")
            else:
                if logger:
                    logger.info("📁 Pre-calculated price file not found")
                    
        except Exception as e:
            if logger:
                logger.warning(f"⚠️ Could not load fallback price data: {e}")
        
        # Graceful failure - return empty DataFrame (NO synthetic data)
        if logger:
            logger.error("❌ All price data loading methods failed")
            logger.info("📊 Returning empty DataFrame - no synthetic data generated")
        
        return pd.DataFrame()
        
    except Exception as e:
        if logger:
            logger.error(f"Failed to load price data: {e}")
        return pd.DataFrame()


def load_benchmark_data(engine, backtest_period: Dict, logger: logging.Logger = None) -> pd.DataFrame:
    """Load benchmark data (VN-Index) with fallback strategy (NO synthetic data)."""
    try:
        if logger:
            logger.info("Loading benchmark data...")
        
        # Try to load real benchmark data from price data first (more reliable)
        try:
            benchmark_query = f"""
            SELECT 
                trading_date as date,
                close_price as close_price
            FROM vcsc_daily_data_complete
            WHERE ticker = 'VNINDEX'
            AND trading_date >= '{backtest_period['start']}'
            AND trading_date <= '{backtest_period['end']}'
            ORDER BY trading_date
            """
            
            benchmark_data = pd.read_sql(benchmark_query, engine)
            benchmark_data['date'] = pd.to_datetime(benchmark_data['date'])
            
            if len(benchmark_data) > 100:  # Sufficient real data
                if logger:
                    logger.info(f"✅ Loaded real benchmark data from price table: {len(benchmark_data)} records")
                
                # Validate benchmark data quality
                benchmark_data['return'] = benchmark_data['close_price'].pct_change()
                returns = benchmark_data['return'].dropna()
                annualized_vol = returns.std() * (252 ** 0.5)
                
                if logger:
                    logger.info(f"📊 Benchmark validation:")
                    logger.info(f"   Annualized volatility: {annualized_vol:.4f}")
                    logger.info(f"   Mean daily return: {returns.mean():.6f}")
                    logger.info(f"   Date range: {benchmark_data['date'].min()} to {benchmark_data['date'].max()}")
                    
                    if annualized_vol < 0.15:
                        logger.warning("⚠️ Low benchmark volatility - possible data quality issues")
                    elif annualized_vol > 0.40:
                        logger.warning("⚠️ High benchmark volatility - possible data quality issues")
                    else:
                        logger.info("✅ Benchmark volatility within expected range")
                
                return benchmark_data
            else:
                if logger:
                    logger.warning("⚠️ Limited real benchmark data from price table, attempting fallback...")
                
        except Exception as e:
            if logger:
                logger.warning(f"⚠️ Could not load real benchmark data from price table: {e}")
                logger.info("📊 Attempting fallback method...")
        
        # Try to load from etf_history as fallback
        try:
            fallback_query = f"""
            SELECT 
                date,
                close as close_price
            FROM etf_history
            WHERE ticker = 'VNINDEX'
            AND date >= '{backtest_period['start']}'
            AND date <= '{backtest_period['end']}'
            ORDER BY date
            """
            
            benchmark_data = pd.read_sql(fallback_query, engine)
            benchmark_data['date'] = pd.to_datetime(benchmark_data['date'])
            
            if len(benchmark_data) > 100:  # Sufficient real data
                if logger:
                    logger.info(f"✅ Loaded fallback benchmark data from etf_history: {len(benchmark_data)} records")
                
                # Validate fallback data quality
                benchmark_data['return'] = benchmark_data['close_price'].pct_change()
                returns = benchmark_data['return'].dropna()
                annualized_vol = returns.std() * (252 ** 0.5)
                
                if logger:
                    logger.info(f"📊 Fallback benchmark validation:")
                    logger.info(f"   Annualized volatility: {annualized_vol:.4f}")
                    logger.info(f"   Mean daily return: {returns.mean():.6f}")
                    logger.info(f"   Date range: {benchmark_data['date'].min()} to {benchmark_data['date'].max()}")
                    
                    if annualized_vol < 0.15:
                        logger.warning("⚠️ Low fallback benchmark volatility - possible synthetic data")
                    elif annualized_vol > 0.40:
                        logger.warning("⚠️ High fallback benchmark volatility - possible data quality issues")
                    else:
                        logger.info("✅ Fallback benchmark volatility within expected range")
                
                return benchmark_data
            else:
                if logger:
                    logger.warning("⚠️ Limited fallback benchmark data, attempting pre-calculated files...")
                
        except Exception as e:
            if logger:
                logger.warning(f"⚠️ Could not load fallback benchmark data: {e}")
                logger.info("📊 Attempting pre-calculated files...")
        
        # Try to load from pre-calculated files
        try:
            fallback_file = Path("docs/18b_complete_benchmark.csv")
            if fallback_file.exists():
                if logger:
                    logger.info("📁 Using pre-calculated benchmark data for speed...")
                
                benchmark_df = pd.read_csv(fallback_file)
                benchmark_df['date'] = pd.to_datetime(benchmark_df['date'])
                
                # Filter to our date range
                start_date = pd.to_datetime(backtest_period['start']).date()
                end_date = pd.to_datetime(backtest_period['end']).date()
                filtered_benchmark = benchmark_df[
                    (benchmark_df['date'] >= start_date) & 
                    (benchmark_df['date'] <= end_date)
                ]
                
                if len(filtered_benchmark) > 0:
                    if logger:
                        logger.info(f"✅ Loaded pre-calculated benchmark data: {len(filtered_benchmark)} records")
                        logger.warning("⚠️ Using pre-calculated data - may not reflect real market conditions")
                    return filtered_benchmark
                else:
                    if logger:
                        logger.warning("⚠️ Pre-calculated benchmark data has no matching dates")
            else:
                if logger:
                    logger.info("📁 Pre-calculated benchmark file not found")
                    
        except Exception as e:
            if logger:
                logger.warning(f"⚠️ Could not load pre-calculated benchmark data: {e}")
        
        # Graceful failure - return empty DataFrame (NO synthetic data)
        if logger:
            logger.error("❌ All benchmark data loading methods failed")
            logger.info("📊 Returning empty DataFrame - no synthetic data generated")
        
        return pd.DataFrame()
        
    except Exception as e:
        if logger:
            logger.error(f"Failed to load benchmark data: {e}")
        return pd.DataFrame()
