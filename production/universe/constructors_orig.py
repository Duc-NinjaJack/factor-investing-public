"""
Aureus Sigma Capital - Universe Construction Module
===================================================
Purpose:
    Defines and constructs investable universes based on systematic,
    point-in-time correct rules. This module is the single source of
    truth for universe definitions.
Author: Duc Nguyen, Quantitative Finance Expert
Date Created: July 28, 2025
"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
from typing import List, Dict, Optional


def get_liquid_universe(
    analysis_date: pd.Timestamp, 
    engine, 
    config: Optional[Dict] = None
) -> List[str]:
    """
    Constructs the ASC-VN-Liquid-150 universe for a given analysis date.

    Methodology:
    1. Calculates 63-day rolling ADTV for all stocks using batch processing.
    2. Filters for stocks with ADTV >= 10 Billion VND.
    3. Selects the Top 200 stocks from the filtered list.
    4. Returns the list of ticker symbols.

    Args:
        analysis_date: The date for which to construct the universe.
        engine: SQLAlchemy database engine connection.
        config: A dictionary containing parameters like adtv_threshold, etc.

    Returns:
        A list of ticker symbols comprising the liquid universe.
    """
    # Default configuration
    default_config = {
        'lookback_days': 63,
        'adtv_threshold_bn': 10.0,
        'top_n': 200,
        'min_trading_coverage': 0.8
    }
    
    if config is None:
        config = default_config
    else:
        # Merge with defaults
        config = {**default_config, **config}
    
    print(f"Constructing liquid universe for {analysis_date.date()}...")
    print(f"  Lookback: {config['lookback_days']} days")
    print(f"  ADTV threshold: {config['adtv_threshold_bn']}B VND")
    print(f"  Target size: {config['top_n']} stocks")
    
    # Calculate lookback start date
    start_date = analysis_date - timedelta(days=config['lookback_days'])
    
    # Step 1: Get all tickers with basic filtering to avoid packet size issues
    print("  Step 1: Loading ticker list...")
    ticker_query = text("""
        SELECT DISTINCT ticker
        FROM vcsc_daily_data_complete
        WHERE trading_date BETWEEN :start_date AND :end_date
            AND total_value > 0
            AND market_cap > 0
            AND close_price_adjusted > 0
        ORDER BY ticker
    """)
    
    with engine.connect() as conn:
        result = conn.execute(ticker_query, {
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': analysis_date.strftime('%Y-%m-%d')
        })
        all_tickers = [row[0] for row in result.fetchall()]
    
    print(f"    Found {len(all_tickers)} active tickers")
    
    # Step 2: Process tickers in batches to calculate ADTV
    print("  Step 2: Calculating ADTV in batches...")
    batch_size = 50  # Process 50 tickers at a time
    total_batches = (len(all_tickers) + batch_size - 1) // batch_size
    all_results = []
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, len(all_tickers))
        batch_tickers = all_tickers[start_idx:end_idx]
        
        if (batch_num + 1) % 10 == 0:
            print(f"    Processing batch {batch_num + 1}/{total_batches}...")
        
        # Query for this batch
        batch_query = text("""
            SELECT 
                v.ticker,
                COUNT(v.trading_date) as trading_days,
                AVG(v.total_value / 1e9) as adtv_bn_vnd,
                AVG(v.market_cap / 1e9) as avg_market_cap_bn
            FROM vcsc_daily_data_complete v
            WHERE v.ticker IN :tickers
                AND v.trading_date BETWEEN :start_date AND :end_date
                AND v.total_value > 0
                AND v.market_cap > 0
                AND v.close_price_adjusted > 0
            GROUP BY v.ticker
        """)
        
        with engine.connect() as conn:
            result = conn.execute(batch_query, {
                'tickers': tuple(batch_tickers),
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': analysis_date.strftime('%Y-%m-%d')
            })
            batch_results = result.fetchall()
            all_results.extend(batch_results)
    
    # Step 3: Filter and rank results
    print("  Step 3: Filtering and ranking...")
    min_trading_days = int(config['lookback_days'] * config['min_trading_coverage'])
    
    print(f"    Total batch results: {len(all_results)}")
    if len(all_results) > 0:
        print(f"    Sample result: {all_results[0]}")
    
    # Convert to DataFrame for easier processing
    df = pd.DataFrame(all_results, columns=['ticker', 'trading_days', 'adtv_bn_vnd', 'avg_market_cap_bn'])
    
    if len(df) > 0:
        print(f"    Before filters: {len(df)} stocks")
        print(f"    Trading days range: {df['trading_days'].min()}-{df['trading_days'].max()} (need >= {min_trading_days})")
        print(f"    ADTV range: {df['adtv_bn_vnd'].min():.3f}-{df['adtv_bn_vnd'].max():.3f}B VND (need >= {config['adtv_threshold_bn']})")
        
        # Apply filters step by step
        trading_days_filter = df['trading_days'] >= min_trading_days
        adtv_filter = df['adtv_bn_vnd'] >= config['adtv_threshold_bn']
        
        print(f"    Stocks passing trading days filter: {trading_days_filter.sum()}")
        print(f"    Stocks passing ADTV filter: {adtv_filter.sum()}")
        
        filtered_df = df[trading_days_filter & adtv_filter].copy()
        print(f"    After filters: {len(filtered_df)} stocks")
    else:
        filtered_df = df
    
    # Sort by ADTV and take top N
    universe_df = filtered_df.sort_values('adtv_bn_vnd', ascending=False).head(config['top_n'])
    
    tickers = universe_df['ticker'].tolist()
    
    print(f"✅ Universe constructed: {len(tickers)} stocks")
    if len(universe_df) > 0:
        print(f"  ADTV range: {universe_df['adtv_bn_vnd'].min():.1f}B - {universe_df['adtv_bn_vnd'].max():.1f}B VND")
        print(f"  Market cap range: {universe_df['avg_market_cap_bn'].min():.1f}B - {universe_df['avg_market_cap_bn'].max():.1f}B VND")
    
    return tickers


def get_liquid_universe_dataframe(
    analysis_date: pd.Timestamp, 
    engine, 
    config: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Same as get_liquid_universe but returns full DataFrame with metrics.
    
    Returns:
        DataFrame with columns: ticker, trading_days, adtv_bn_vnd, avg_market_cap_bn, sector
    """
    # Get tickers using the main function
    tickers = get_liquid_universe(analysis_date, engine, config)
    
    if not tickers:
        return pd.DataFrame()
    
    # Get sector information for the universe
    print("  Adding sector information...")
    sector_query = text("""
        SELECT ticker, sector
        FROM master_info
        WHERE ticker IN :tickers
    """)
    
    with engine.connect() as conn:
        result = conn.execute(sector_query, {'tickers': tuple(tickers)})
        sector_data = dict(result.fetchall())
    
    # Rebuild the universe data with sectors
    # This requires re-running the ADTV calculation for the final universe
    config = config or {}
    default_config = {
        'lookback_days': 63,
        'adtv_threshold_bn': 10.0,
        'top_n': 200,
        'min_trading_coverage': 0.8
    }
    config = {**default_config, **config}
    
    start_date = analysis_date - timedelta(days=config['lookback_days'])
    
    universe_query = text("""
        SELECT 
            v.ticker,
            COUNT(v.trading_date) as trading_days,
            AVG(v.total_value / 1e9) as adtv_bn_vnd,
            AVG(v.market_cap / 1e9) as avg_market_cap_bn
        FROM vcsc_daily_data_complete v
        WHERE v.ticker IN :tickers
            AND v.trading_date BETWEEN :start_date AND :end_date
            AND v.total_value > 0
            AND v.market_cap > 0
            AND v.close_price_adjusted > 0
        GROUP BY v.ticker
        ORDER BY adtv_bn_vnd DESC
    """)
    
    with engine.connect() as conn:
        result = conn.execute(universe_query, {
            'tickers': tuple(tickers),
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': analysis_date.strftime('%Y-%m-%d')
        })
        universe_data = result.fetchall()
    
    # Create DataFrame
    df = pd.DataFrame(universe_data, columns=['ticker', 'trading_days', 'adtv_bn_vnd', 'avg_market_cap_bn'])
    df['sector'] = df['ticker'].map(sector_data)
    df['universe_rank'] = range(1, len(df) + 1)
    df['universe_date'] = analysis_date.strftime('%Y-%m-%d')
    
    return df


def validate_universe_construction(
    tickers: List[str], 
    analysis_date: pd.Timestamp,
    min_size: int = 125
) -> Dict:
    """
    Validates the constructed universe meets quality standards.
    
    Args:
        tickers: List of ticker symbols in the universe
        analysis_date: Date for which universe was constructed
        min_size: Minimum required universe size
        
    Returns:
        Dictionary with validation results
    """
    validation = {
        'is_valid': True,
        'checks': {},
        'warnings': [],
        'errors': []
    }
    
    # Check minimum size
    size_check = len(tickers) >= min_size
    validation['checks']['minimum_size'] = {
        'pass': size_check,
        'actual': len(tickers),
        'required': min_size
    }
    
    if not size_check:
        validation['is_valid'] = False
        validation['errors'].append(f"Universe too small: {len(tickers)} < {min_size}")
    
    # Check for duplicates
    unique_tickers = len(set(tickers))
    duplicate_check = unique_tickers == len(tickers)
    validation['checks']['no_duplicates'] = {
        'pass': duplicate_check,
        'unique_count': unique_tickers,
        'total_count': len(tickers)
    }
    
    if not duplicate_check:
        validation['warnings'].append("Duplicate tickers found in universe")
    
    return validation


def get_quarterly_universe_dates(year: int) -> Dict[str, pd.Timestamp]:
    """
    Get standard quarterly universe refresh dates for a given year.
    
    Returns:
        Dictionary mapping quarter names to timestamp objects
    """
    return {
        'Q1': pd.Timestamp(f'{year}-03-31'),
        'Q2': pd.Timestamp(f'{year}-06-30'),
        'Q3': pd.Timestamp(f'{year}-09-30'),
        'Q4': pd.Timestamp(f'{year}-12-31')
    }


if __name__ == "__main__":
    # Simple test when run directly
    print("🧪 Universe Constructor Module - Ready for import")
    print("Available functions:")
    print("  - get_liquid_universe()")
    print("  - validate_universe_construction()")
    print("  - get_quarterly_universe_dates()")