#!/usr/bin/env python3
"""
Check Item 302 Availability
==========================

This script checks if item_id 302 is available for the test stocks.
"""

import sys
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine, text

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

def check_item_302_availability():
    """Check if item_id 302 is available for test stocks."""
    print("🔍 Checking Item 302 Availability")
    print("=" * 50)
    
    # Connect to database
    db_manager = get_database_manager()
    engine = db_manager.get_engine()
    
    # Test stocks
    test_stocks = ['TCB', 'VCB', 'BID', 'CTG', 'MBB', 'ACB', 'TPB', 'STB', 'VPB', 'SHB']
    
    print(f"📊 Checking availability for {len(test_stocks)} stocks")
    
    for ticker in test_stocks:
        print(f"\n🔍 {ticker}:")
        
        # Check item 302
        query_302 = text("""
            SELECT 
                fv.item_id,
                fv.statement_type,
                COUNT(*) as count,
                MIN(fv.value) as min_value,
                MAX(fv.value) as max_value,
                AVG(fv.value) as avg_value
            FROM fundamental_values fv
            WHERE fv.ticker = :ticker
            AND fv.item_id = 302
            AND fv.statement_type = 'BS'
            GROUP BY fv.item_id, fv.statement_type
        """)
        
        result_302 = pd.read_sql(query_302, engine, params={'ticker': ticker})
        
        if not result_302.empty:
            print(f"   ✅ Item 302 (BS): {result_302['count'].iloc[0]} records, avg={result_302['avg_value'].iloc[0]:.0f}")
        else:
            print(f"   ❌ Item 302 (BS): No data")
        
        # Check item 1
        query_1 = text("""
            SELECT 
                fv.item_id,
                fv.statement_type,
                COUNT(*) as count,
                MIN(fv.value) as min_value,
                MAX(fv.value) as max_value,
                AVG(fv.value) as avg_value
            FROM fundamental_values fv
            WHERE fv.ticker = :ticker
            AND fv.item_id = 1
            AND fv.statement_type = 'PL'
            GROUP BY fv.item_id, fv.statement_type
        """)
        
        result_1 = pd.read_sql(query_1, engine, params={'ticker': ticker})
        
        if not result_1.empty:
            print(f"   ✅ Item 1 (PL): {result_1['count'].iloc[0]} records, avg={result_1['avg_value'].iloc[0]:.0f}")
        else:
            print(f"   ❌ Item 1 (PL): No data")
        
        # Check item 2
        query_2 = text("""
            SELECT 
                fv.item_id,
                fv.statement_type,
                COUNT(*) as count,
                MIN(fv.value) as min_value,
                MAX(fv.value) as max_value,
                AVG(fv.value) as avg_value
            FROM fundamental_values fv
            WHERE fv.ticker = :ticker
            AND fv.item_id = 2
            AND fv.statement_type = 'PL'
            GROUP BY fv.item_id, fv.statement_type
        """)
        
        result_2 = pd.read_sql(query_2, engine, params={'ticker': ticker})
        
        if not result_2.empty:
            print(f"   ✅ Item 2 (PL): {result_2['count'].iloc[0]} records, avg={result_2['avg_value'].iloc[0]:.0f}")
        else:
            print(f"   ❌ Item 2 (PL): No data")

if __name__ == "__main__":
    check_item_302_availability() 