#!/usr/bin/env python3
"""
Check volume data structure and values
"""

import pandas as pd
import sys
import os
from pathlib import Path
from sqlalchemy import text

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

def check_volume_data():
    """Check volume data structure and values"""
    
    # Get database connection
    db_manager = get_database_manager()
    engine = db_manager.get_engine()
    
    print("🔍 CHECKING VOLUME DATA STRUCTURE")
    print("="*60)
    
    # Check sample data
    sample_query = text("""
        SELECT 
            ticker,
            trading_date,
            total_volume,
            market_cap,
            close_price_adjusted
        FROM vcsc_daily_data_complete
        WHERE trading_date = '2024-06-28'
        ORDER BY total_volume DESC
        LIMIT 10
    """)
    
    try:
        sample_df = pd.read_sql(sample_query, engine)
        print(f"   Sample data for 2024-06-28:")
        print(f"   ✅ Found {len(sample_df)} records")
        
        if not sample_df.empty:
            print(f"   Sample stocks:")
            for _, row in sample_df.head(5).iterrows():
                print(f"     - {row['ticker']}: Volume={row['total_volume']:,.0f}, MarketCap={row['market_cap']:,.0f}, Close={row['close_price_adjusted']:,.0f}")
        else:
            print(f"   ⚠️ No data found")
            
    except Exception as e:
        print(f"   ❌ Query failed: {e}")
    
    # Check volume statistics
    print(f"\n📊 VOLUME STATISTICS")
    print("-" * 60)
    
    stats_query = text("""
        SELECT 
            COUNT(*) as total_records,
            COUNT(DISTINCT ticker) as unique_tickers,
            MIN(total_volume) as min_volume,
            MAX(total_volume) as max_volume,
            AVG(total_volume) as avg_volume,
            MIN(market_cap) as min_market_cap,
            MAX(market_cap) as max_market_cap,
            AVG(market_cap) as avg_market_cap
        FROM vcsc_daily_data_complete
        WHERE trading_date = '2024-06-28'
    """)
    
    try:
        stats_df = pd.read_sql(stats_query, engine)
        print(f"   Volume statistics for 2024-06-28:")
        for _, row in stats_df.iterrows():
            print(f"     - Total records: {row['total_records']:,}")
            print(f"     - Unique tickers: {row['unique_tickers']:,}")
            print(f"     - Volume range: {row['min_volume']:,.0f} - {row['max_volume']:,.0f}")
            print(f"     - Average volume: {row['avg_volume']:,.0f}")
            print(f"     - Market cap range: {row['min_market_cap']:,.0f} - {row['max_market_cap']:,.0f}")
            print(f"     - Average market cap: {row['avg_market_cap']:,.0f}")
            
    except Exception as e:
        print(f"   ❌ Stats query failed: {e}")
    
    # Check if volume is in millions or thousands
    print(f"\n🔍 CHECKING VOLUME UNITS")
    print("-" * 60)
    
    # Test with different volume thresholds
    test_thresholds = [1000, 10000, 100000, 1000000, 10000000]
    
    for threshold in test_thresholds:
        test_query = text("""
            SELECT COUNT(*) as count
            FROM vcsc_daily_data_complete
            WHERE trading_date = '2024-06-28'
            AND total_volume >= :threshold
        """)
        
        try:
            result = pd.read_sql(test_query, engine, params={'threshold': threshold})
            count = result.iloc[0]['count']
            print(f"   Volume >= {threshold:,}: {count:,} stocks")
            
        except Exception as e:
            print(f"   ❌ Query failed for threshold {threshold}: {e}")

if __name__ == "__main__":
    check_volume_data() 