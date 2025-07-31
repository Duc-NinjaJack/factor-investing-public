#!/usr/bin/env python3
"""
Test universe construction to debug the issue
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

def test_universe_construction():
    """Test universe construction with different parameters"""
    
    # Get database connection
    db_manager = get_database_manager()
    engine = db_manager.get_engine()
    
    print("🔍 TESTING UNIVERSE CONSTRUCTION")
    print("="*60)
    
    # Test different parameters
    test_cases = [
        {"adtv_threshold_bn": 1.0, "min_market_cap_bn": 100.0, "lookback_days": 63},
        {"adtv_threshold_bn": 0.5, "min_market_cap_bn": 50.0, "lookback_days": 63},
        {"adtv_threshold_bn": 0.1, "min_market_cap_bn": 10.0, "lookback_days": 63},
        {"adtv_threshold_bn": 0.01, "min_market_cap_bn": 1.0, "lookback_days": 63},
    ]
    
    test_date = pd.Timestamp('2024-06-30')
    
    for i, params in enumerate(test_cases):
        print(f"\n{i+1}. Testing with ADTV >= {params['adtv_threshold_bn']}B, Market Cap >= {params['min_market_cap_bn']}B")
        print("-" * 60)
        
        adtv_threshold = params['adtv_threshold_bn'] * 1e9
        min_market_cap = params['min_market_cap_bn'] * 1e9
        lookback_days = params['lookback_days']
        
        # Test universe query
        universe_query = text("""
            SELECT 
                ticker,
                AVG(total_volume) as avg_volume,
                AVG(market_cap) as avg_market_cap
            FROM vcsc_daily_data_complete
            WHERE trading_date <= :analysis_date
              AND trading_date >= DATE_SUB(:analysis_date, INTERVAL :lookback_days DAY)
            GROUP BY ticker
            HAVING avg_volume >= :adtv_threshold AND avg_market_cap >= :min_market_cap
            ORDER BY avg_volume DESC
            LIMIT 20
        """)
        
        try:
            universe_df = pd.read_sql(universe_query, engine, 
                                     params={'analysis_date': test_date, 
                                            'lookback_days': lookback_days, 
                                            'adtv_threshold': adtv_threshold, 
                                            'min_market_cap': min_market_cap})
            
            print(f"   ✅ Found {len(universe_df)} stocks")
            
            if not universe_df.empty:
                print(f"   Sample stocks:")
                for _, row in universe_df.head(5).iterrows():
                    print(f"     - {row['ticker']}: ADTV={row['avg_volume']/1e9:.1f}B, MarketCap={row['avg_market_cap']/1e9:.1f}B")
            else:
                print(f"   ⚠️ No stocks found")
                
        except Exception as e:
            print(f"   ❌ Query failed: {e}")
    
    # Test with a very early date
    print(f"\n5. Testing with early date (2020-01-30)")
    print("-" * 60)
    
    early_date = pd.Timestamp('2020-01-30')
    early_params = {"adtv_threshold_bn": 0.01, "min_market_cap_bn": 1.0, "lookback_days": 63}
    
    adtv_threshold = early_params['adtv_threshold_bn'] * 1e9
    min_market_cap = early_params['min_market_cap_bn'] * 1e9
    lookback_days = early_params['lookback_days']
    
    try:
        universe_df = pd.read_sql(universe_query, engine, 
                                 params={'analysis_date': early_date, 
                                        'lookback_days': lookback_days, 
                                        'adtv_threshold': adtv_threshold, 
                                        'min_market_cap': min_market_cap})
        
        print(f"   ✅ Found {len(universe_df)} stocks for early date")
        
        if not universe_df.empty:
            print(f"   Sample stocks:")
            for _, row in universe_df.head(5).iterrows():
                print(f"     - {row['ticker']}: ADTV={row['avg_volume']/1e9:.1f}B, MarketCap={row['avg_market_cap']/1e9:.1f}B")
        else:
            print(f"   ⚠️ No stocks found for early date")
            
    except Exception as e:
        print(f"   ❌ Query failed: {e}")

if __name__ == "__main__":
    test_universe_construction() 