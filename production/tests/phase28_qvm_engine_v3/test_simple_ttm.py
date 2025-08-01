#!/usr/bin/env python3
"""
Test Simple TTM
==============

Simple test to check TTM calculation step by step.
"""

import sys
import pandas as pd
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

def test_simple_ttm():
    """Test TTM calculation step by step."""
    print("🧪 Testing Simple TTM")
    print("=" * 40)
    
    # Connect to database
    db_manager = get_database_manager()
    engine = db_manager.get_engine()
    
    # Test date
    test_date = pd.Timestamp('2021-01-29')
    lag_days = 45
    lag_date = test_date - pd.Timedelta(days=lag_days)
    lag_year = lag_date.year
    
    # Focus on one ticker
    test_ticker = 'HPG'
    
    print(f"📅 Test Date: {test_date.date()}")
    print(f"📅 Lag Year: {lag_year}")
    print(f"📅 Test Ticker: {test_ticker}")
    
    # Step 1: Get raw data for HPG
    print(f"\n🔍 Step 1: Raw Data for {test_ticker}")
    print("-" * 40)
    
    raw_query = text(f"""
        SELECT 
            fv.ticker,
            fv.item_id,
            fv.statement_type,
            fv.year,
            fv.quarter,
            fv.value
        FROM fundamental_values fv
        WHERE fv.ticker = '{test_ticker}'
        AND fv.year <= {lag_year}
        AND (
            (fv.item_id IN (1, 2, 13) AND fv.statement_type = 'PL')
            OR (fv.item_id = 2 AND fv.statement_type = 'BS')
        )
        ORDER BY fv.item_id, fv.statement_type, fv.year DESC, fv.quarter DESC
    """)
    
    raw_data = pd.read_sql(raw_query, engine)
    
    print(f"   Raw data records: {len(raw_data)}")
    if not raw_data.empty:
        print(f"   Sample raw data:")
        for _, row in raw_data.head(10).iterrows():
            print(f"     Item {row['item_id']} ({row['statement_type']}) {row['year']}Q{row['quarter']} = {row['value']:,.0f}")
    
    # Step 2: Test TTM calculation for Net Profit only
    print(f"\n🔍 Step 2: TTM for Net Profit Only")
    print("-" * 40)
    
    netprofit_ttm_query = text(f"""
        WITH netprofit_data AS (
            SELECT 
                fv.ticker,
                fv.year,
                fv.quarter,
                fv.value / 6222444702.01 as value,
                ROW_NUMBER() OVER (ORDER BY fv.year DESC, fv.quarter DESC) as rn
            FROM fundamental_values fv
            WHERE fv.ticker = '{test_ticker}'
            AND fv.item_id = 1
            AND fv.statement_type = 'PL'
            AND fv.year <= {lag_year}
        )
        SELECT 
            ticker,
            SUM(value) as netprofit_ttm
        FROM netprofit_data
        WHERE rn <= 4
        GROUP BY ticker
    """)
    
    netprofit_result = pd.read_sql(netprofit_ttm_query, engine)
    
    print(f"   Net Profit TTM result: {len(netprofit_result)} records")
    if not netprofit_result.empty:
        for _, row in netprofit_result.iterrows():
            print(f"     {row['ticker']}: Net Profit TTM = {row['netprofit_ttm']:.2f}")
    
    # Step 3: Test TTM calculation for Total Assets only
    print(f"\n🔍 Step 3: TTM for Total Assets Only")
    print("-" * 40)
    
    assets_ttm_query = text(f"""
        WITH assets_data AS (
            SELECT 
                fv.ticker,
                fv.year,
                fv.quarter,
                fv.value / 6222444702.01 as value,
                ROW_NUMBER() OVER (ORDER BY fv.year DESC, fv.quarter DESC) as rn
            FROM fundamental_values fv
            WHERE fv.ticker = '{test_ticker}'
            AND fv.item_id = 2
            AND fv.statement_type = 'BS'
            AND fv.year <= {lag_year}
        )
        SELECT 
            ticker,
            SUM(value) as totalassets_ttm
        FROM assets_data
        WHERE rn <= 4
        GROUP BY ticker
    """)
    
    assets_result = pd.read_sql(assets_ttm_query, engine)
    
    print(f"   Total Assets TTM result: {len(assets_result)} records")
    if not assets_result.empty:
        for _, row in assets_result.iterrows():
            print(f"     {row['ticker']}: Total Assets TTM = {row['totalassets_ttm']:.2f}")
    
    # Step 4: Calculate ROAA
    print(f"\n🔍 Step 4: Calculate ROAA")
    print("-" * 40)
    
    if not netprofit_result.empty and not assets_result.empty:
        netprofit_ttm = netprofit_result.iloc[0]['netprofit_ttm']
        totalassets_ttm = assets_result.iloc[0]['totalassets_ttm']
        
        if totalassets_ttm > 0:
            roaa = netprofit_ttm / totalassets_ttm
            print(f"   ✅ ROAA = {roaa:.4f}")
            print(f"      Net Profit TTM = {netprofit_ttm:.2f}")
            print(f"      Total Assets TTM = {totalassets_ttm:.2f}")
        else:
            print(f"   ❌ Total Assets TTM is zero or negative")
    else:
        print(f"   ❌ Missing data for ROAA calculation")

if __name__ == "__main__":
    test_simple_ttm() 