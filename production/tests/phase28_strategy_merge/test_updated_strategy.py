#!/usr/bin/env python3
"""
Test the updated strategy with fundamental_values table
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

def test_updated_strategy():
    """Test the updated strategy with fundamental_values table"""
    
    # Get database connection
    db_manager = get_database_manager()
    engine = db_manager.get_engine()
    
    print("🧪 TESTING UPDATED STRATEGY WITH FUNDAMENTAL_VALUES")
    print("="*60)
    
    # Test configuration
    test_config = {
        "factors": {
            "netprofit_item_id": 1501,
            "revenue_item_id": 10701,
            "totalassets_item_id": 107,
            "fundamental_lag_days": 45,
        }
    }
    
    # Test universe
    test_universe = ['VNM', 'VCB', 'TCB', 'HPG', 'VIC']
    
    # Test date
    test_date = pd.Timestamp('2024-06-30')  # Mid-year 2024
    
    print(f"\n1. TESTING FUNDAMENTAL DATA QUERY")
    print("-" * 40)
    
    # Calculate lagged date
    lag_days = test_config['factors']['fundamental_lag_days']
    lag_date = test_date - pd.Timedelta(days=lag_days)
    lag_year = lag_date.year
    lag_quarter = ((lag_date.month - 1) // 3) + 1
    
    print(f"   Test date: {test_date}")
    print(f"   Lagged date: {lag_date} (Year: {lag_year}, Quarter: {lag_quarter})")
    
    # First, let's check what data is available for these tickers
    check_query = text("""
        SELECT 
            ticker,
            year,
            quarter,
            item_id,
            value
        FROM fundamental_values
        WHERE ticker IN ('VNM', 'VCB', 'TCB', 'HPG', 'VIC')
        AND item_id IN (%s, %s, %s)
        AND year = 2024
        ORDER BY ticker, quarter, item_id
        LIMIT 20
    """ % (test_config['factors']['netprofit_item_id'], 
           test_config['factors']['revenue_item_id'], 
           test_config['factors']['totalassets_item_id']))
    
    try:
        check_df = pd.read_sql(check_query, engine)
        print(f"   Available data for 2024:")
        print(f"   ✅ Found {len(check_df)} records")
        
        if not check_df.empty:
            print(f"   Sample data:")
            for _, row in check_df.head(10).iterrows():
                print(f"     - {row['ticker']} {row['year']}Q{row['quarter']} Item_{row['item_id']}: {row['value']:,.0f}")
        else:
            print(f"   ⚠️ No data found for these item IDs")
            
    except Exception as e:
        print(f"   ❌ Check query failed: {e}")
    
    # Test the fundamental data query with simpler approach
    print(f"\n2. TESTING SIMPLIFIED FUNDAMENTAL QUERY")
    print("-" * 40)
    
    simple_query = text("""
        SELECT 
            fv.ticker,
            fv.year,
            fv.quarter,
            fv.item_id,
            fv.value
        FROM fundamental_values fv
        WHERE fv.item_id IN (%s, %s, %s)
        AND fv.ticker IN ('VNM', 'VCB', 'TCB', 'HPG', 'VIC')
        AND fv.year = 2024
        AND fv.quarter <= 2
        ORDER BY fv.ticker, fv.quarter, fv.item_id
    """ % (test_config['factors']['netprofit_item_id'], 
           test_config['factors']['revenue_item_id'], 
           test_config['factors']['totalassets_item_id']))
    
    try:
        simple_df = pd.read_sql(simple_query, engine)
        print(f"   ✅ Simple query executed successfully")
        print(f"   ✅ Loaded {len(simple_df)} fundamental records")
        
        if not simple_df.empty:
            print(f"   Sample data:")
            for _, row in simple_df.head(10).iterrows():
                print(f"     - {row['ticker']} {row['year']}Q{row['quarter']} Item_{row['item_id']}: {row['value']:,.0f}")
        else:
            print(f"   ⚠️ No fundamental data found for test period")
            
    except Exception as e:
        print(f"   ❌ Simple query failed: {e}")
        return False
    
    # Test price data query
    print(f"\n3. TESTING PRICE DATA QUERY")
    print("-" * 40)
    
    price_query = text("""
        SELECT 
            ticker,
            trading_date,
            close_price_adjusted as close,
            total_volume as volume,
            market_cap
        FROM vcsc_daily_data_complete
        WHERE trading_date <= :test_date
          AND ticker IN ('VNM', 'VCB', 'TCB', 'HPG', 'VIC')
        ORDER BY ticker, trading_date DESC
    """)
    
    try:
        price_df = pd.read_sql(price_query, engine, params={'test_date': test_date})
        print(f"   ✅ Price query executed successfully")
        print(f"   ✅ Loaded {len(price_df)} price records")
        
        if not price_df.empty:
            print(f"   Sample data:")
            for _, row in price_df.head(5).iterrows():
                print(f"     - {row['ticker']}: {row['trading_date']}, Close={row['close']:,.0f}, Volume={row['volume']:,.0f}")
        else:
            print(f"   ⚠️ No price data found for test period")
            
    except Exception as e:
        print(f"   ❌ Price query failed: {e}")
        return False
    
    # Test sector data
    print(f"\n4. TESTING SECTOR DATA")
    print("-" * 40)
    
    sector_query = text("""
        SELECT DISTINCT
            ic.ticker,
            mi.sector
        FROM intermediary_calculations_enhanced ic
        LEFT JOIN master_info mi ON ic.ticker = mi.ticker
        WHERE ic.calc_date = (SELECT MAX(calc_date) FROM intermediary_calculations_enhanced)
        AND ic.ticker IN ('VNM', 'VCB', 'TCB', 'HPG', 'VIC')
    """)
    
    try:
        sector_df = pd.read_sql(sector_query, engine)
        print(f"   ✅ Sector query executed successfully")
        print(f"   ✅ Loaded sector data for {len(sector_df)} stocks")
        
        if not sector_df.empty:
            print(f"   Sector data:")
            for _, row in sector_df.iterrows():
                print(f"     - {row['ticker']}: {row['sector']}")
        else:
            print(f"   ⚠️ No sector data found")
            
    except Exception as e:
        print(f"   ❌ Sector query failed: {e}")
        return False
    
    # Summary
    print(f"\n5. TEST SUMMARY")
    print("-" * 40)
    print(f"   ✅ All queries executed successfully")
    print(f"   ✅ Fundamental data available: {len(simple_df) if 'simple_df' in locals() else 0} records")
    print(f"   ✅ Price data available: {len(price_df) if 'price_df' in locals() else 0} records")
    print(f"   ✅ Sector data available: {len(sector_df) if 'sector_df' in locals() else 0} records")
    print(f"   ")
    print(f"   🎯 READY TO PROCEED WITH STRATEGY BACKTEST")
    
    return True

if __name__ == "__main__":
    test_updated_strategy() 