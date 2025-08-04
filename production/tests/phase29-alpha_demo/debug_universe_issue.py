#!/usr/bin/env python3
"""
Debug script to identify universe construction issues
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
current_path = Path.cwd()
while not (current_path / 'production').is_dir():
    if current_path.parent == current_path:
        raise FileNotFoundError("Could not find the 'production' directory.")
    current_path = current_path.parent

project_root = current_path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from production.database.connection import get_database_manager
from sqlalchemy import text

def test_universe_precomputation():
    """Test the universe precomputation step by step."""
    
    print("🔍 DEBUGGING UNIVERSE CONSTRUCTION ISSUES")
    print("=" * 60)
    
    # Get database connection
    db_manager = get_database_manager()
    engine = db_manager.get_engine()
    
    # Test configuration
    config = {
        'backtest_start_date': '2016-01-01',
        'backtest_end_date': '2025-07-28',
        'universe': {
            'lookback_days': 63,
            'top_n_stocks': 200,
        }
    }
    
    print(f"📊 Testing universe precomputation for period: {config['backtest_start_date']} to {config['backtest_end_date']}")
    
    # Test 1: Check if we have any data in the table
    print("\n🔍 Test 1: Checking vcsc_daily_data table...")
    query1 = text("""
        SELECT 
            MIN(trading_date) as min_date,
            MAX(trading_date) as max_date,
            COUNT(DISTINCT ticker) as unique_tickers,
            COUNT(*) as total_rows
        FROM vcsc_daily_data
        WHERE trading_date BETWEEN :start_date AND :end_date
    """)
    
    with engine.connect() as conn:
        result1 = pd.read_sql(query1, conn, params={
            'start_date': config['backtest_start_date'],
            'end_date': config['backtest_end_date']
        })
    
    print(f"   ✅ Data available:")
    print(f"      - Date range: {result1['min_date'].iloc[0]} to {result1['max_date'].iloc[0]}")
    print(f"      - Unique tickers: {result1['unique_tickers'].iloc[0]:,}")
    print(f"      - Total rows: {result1['total_rows'].iloc[0]:,}")
    
    # Test 2: Check ADTV calculation for a specific date
    print("\n🔍 Test 2: Testing ADTV calculation for 2016-02-01...")
    test_date = '2016-02-01'
    lookback_days = config['universe']['lookback_days']
    start_date = pd.Timestamp(test_date) - pd.Timedelta(days=lookback_days)
    
    query2 = text("""
        WITH daily_adtv AS (
            SELECT 
                trading_date,
                ticker,
                total_volume * close_price as adtv_vnd
            FROM vcsc_daily_data
            WHERE trading_date BETWEEN :start_date AND :test_date
        ),
        rolling_adtv AS (
            SELECT 
                trading_date,
                ticker,
                AVG(adtv_vnd) OVER (
                    PARTITION BY ticker 
                    ORDER BY trading_date 
                    ROWS BETWEEN 62 PRECEDING AND CURRENT ROW
                ) as avg_adtv_63d
            FROM daily_adtv
        ),
        ranked_universe AS (
            SELECT 
                trading_date,
                ticker,
                ROW_NUMBER() OVER (
                    PARTITION BY trading_date 
                    ORDER BY avg_adtv_63d DESC
                ) as rank_position
            FROM rolling_adtv
            WHERE avg_adtv_63d > 0
        )
        SELECT trading_date, ticker, rank_position
        FROM ranked_universe
        WHERE rank_position <= :top_n_stocks
        AND trading_date = :test_date
        ORDER BY rank_position
        LIMIT 10
    """)
    
    with engine.connect() as conn:
        result2 = pd.read_sql(query2, conn, params={
            'start_date': start_date,
            'test_date': test_date,
            'top_n_stocks': config['universe']['top_n_stocks']
        })
    
    print(f"   ✅ ADTV calculation for {test_date}:")
    print(f"      - Found {len(result2)} stocks in top {config['universe']['top_n_stocks']}")
    if len(result2) > 0:
        print(f"      - Top 5 stocks: {result2['ticker'].head().tolist()}")
    else:
        print(f"      - ⚠️  No stocks found!")
    
    # Test 3: Check what dates have universe data
    print("\n🔍 Test 3: Checking universe data availability...")
    query3 = text("""
        WITH daily_adtv AS (
            SELECT 
                trading_date,
                ticker,
                total_volume * close_price as adtv_vnd
            FROM vcsc_daily_data
            WHERE trading_date BETWEEN :start_date AND :end_date
        ),
        rolling_adtv AS (
            SELECT 
                trading_date,
                ticker,
                AVG(adtv_vnd) OVER (
                    PARTITION BY ticker 
                    ORDER BY trading_date 
                    ROWS BETWEEN 62 PRECEDING AND CURRENT ROW
                ) as avg_adtv_63d
            FROM daily_adtv
        ),
        ranked_universe AS (
            SELECT 
                trading_date,
                ticker,
                ROW_NUMBER() OVER (
                    PARTITION BY trading_date 
                    ORDER BY avg_adtv_63d DESC
                ) as rank_position
            FROM rolling_adtv
            WHERE avg_adtv_63d > 0
        )
        SELECT 
            trading_date,
            COUNT(*) as universe_size
        FROM ranked_universe
        WHERE rank_position <= :top_n_stocks
        GROUP BY trading_date
        ORDER BY trading_date
        LIMIT 10
    """)
    
    with engine.connect() as conn:
        result3 = pd.read_sql(query3, conn, params={
            'start_date': config['backtest_start_date'],
            'end_date': config['backtest_end_date'],
            'top_n_stocks': config['universe']['top_n_stocks']
        })
    
    print(f"   ✅ Universe availability:")
    print(f"      - Found universe data for {len(result3)} dates")
    if len(result3) > 0:
        print(f"      - First 5 dates: {result3['trading_date'].head().tolist()}")
        print(f"      - Average universe size: {result3['universe_size'].mean():.0f}")
    else:
        print(f"      - ⚠️  No universe data found!")
    
    # Test 4: Check rebalance dates vs universe dates
    print("\n🔍 Test 4: Checking rebalance date alignment...")
    rebalance_dates = pd.date_range(
        start=config['backtest_start_date'],
        end=config['backtest_end_date'],
        freq='M'
    )
    
    print(f"   ✅ Rebalance dates:")
    print(f"      - Total rebalance dates: {len(rebalance_dates)}")
    print(f"      - First 5 rebalance dates: {rebalance_dates[:5].tolist()}")
    
    # Check if any rebalance dates have universe data
    if len(result3) > 0:
        universe_dates = set(result3['trading_date'].dt.date)
        rebalance_dates_set = set(rebalance_dates.date)
        
        matching_dates = universe_dates.intersection(rebalance_dates_set)
        print(f"      - Matching dates: {len(matching_dates)} out of {len(rebalance_dates)}")
        if len(matching_dates) > 0:
            print(f"      - Sample matching dates: {list(matching_dates)[:5]}")
        else:
            print(f"      - ⚠️  No rebalance dates match universe dates!")
    
    print("\n" + "=" * 60)
    print("🔍 DIAGNOSIS SUMMARY")
    print("=" * 60)
    
    if len(result2) == 0:
        print("❌ ISSUE: ADTV calculation not working")
        print("   - No stocks found in universe for test date")
        print("   - Possible causes:")
        print("     * Insufficient data for 63-day lookback")
        print("     * All ADTV values are zero or negative")
        print("     * Date range issues")
    elif len(result3) == 0:
        print("❌ ISSUE: No universe data generated")
        print("   - Universe precomputation not working")
        print("   - Possible causes:")
        print("     * SQL query errors")
        print("     * Date range issues")
        print("     * Data quality problems")
    elif len(matching_dates) == 0:
        print("❌ ISSUE: Rebalance dates don't match universe dates")
        print("   - Universe data exists but not on rebalance dates")
        print("   - Possible causes:")
        print("     * Different date formats")
        print("     * Missing trading days")
        print("     * Date alignment issues")
    else:
        print("✅ Universe construction appears to be working")
        print("   - Check other parts of the strategy")

if __name__ == "__main__":
    test_universe_precomputation() 