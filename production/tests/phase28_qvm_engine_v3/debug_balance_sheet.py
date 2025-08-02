#!/usr/bin/env python3
"""
Debug Balance Sheet Data
=======================

Debug script to check if Balance Sheet data exists and identify the issue.
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
# VND to billions conversion factor
VND_TO_BILLIONS = 1e9  # 1 billion VND


def debug_balance_sheet():
    """Debug Balance Sheet data availability."""
    print("🔍 Debugging Balance Sheet Data")
    print("=" * 50)
    
    # Connect to database
    db_manager = get_database_manager()
    engine = db_manager.get_engine()
    
    # Test date
    test_date = pd.Timestamp('2021-01-29')
    lag_days = 45
    lag_date = test_date - pd.Timedelta(days=lag_days)
    lag_year = lag_date.year
    
    # Sample tickers
    sample_tickers = ['HPG', 'TCB', 'SHB', 'STB']
    ticker_list = "','".join(sample_tickers)
    
    print(f"📅 Test Date: {test_date.date()}")
    print(f"📅 Lag Year: {lag_year}")
    print(f"📅 Sample Tickers: {sample_tickers}")
    
    # 1. Check if Balance Sheet data exists
    print(f"\n🔍 Step 1: Checking Balance Sheet Data")
    print("-" * 40)
    
    bs_check_query = text(f"""
        SELECT 
            fv.ticker,
            fv.item_id,
            fv.statement_type,
            fv.year,
            fv.quarter,
            fv.value,
            mi.sector
        FROM fundamental_values fv
        LEFT JOIN master_info mi ON fv.ticker = mi.ticker
        WHERE fv.ticker IN ('{ticker_list}')
        AND fv.statement_type = 'BS'
        AND fv.item_id = 2
        AND fv.year <= {lag_year}
        ORDER BY fv.ticker, fv.year DESC, fv.quarter DESC
        LIMIT 10
    """)
    
    bs_data = pd.read_sql(bs_check_query, engine)
    
    print(f"   Balance Sheet data records: {len(bs_data)}")
    if not bs_data.empty:
        print(f"   Sample BS data:")
        for _, row in bs_data.head().iterrows():
            print(f"     {row['ticker']}: Item {row['item_id']} ({row['statement_type']}) {row['year']}Q{row['quarter']} = {row['value']:,.0f}")
    else:
        print("   ❌ No Balance Sheet data found!")
        
        # Check what BS data exists
        bs_all_query = text(f"""
            SELECT 
                fv.ticker,
                fv.item_id,
                fv.statement_type,
                fv.year,
                fv.quarter,
                fv.value
            FROM fundamental_values fv
            WHERE fv.ticker IN ('{ticker_list}')
            AND fv.statement_type = 'BS'
            AND fv.year <= {lag_year}
            ORDER BY fv.ticker, fv.year DESC, fv.quarter DESC
            LIMIT 10
        """)
        
        bs_all_data = pd.read_sql(bs_all_query, engine)
        print(f"   All BS data records: {len(bs_all_data)}")
        if not bs_all_data.empty:
            print(f"   Sample all BS data:")
            for _, row in bs_all_data.head().iterrows():
                print(f"     {row['ticker']}: Item {row['item_id']} ({row['statement_type']}) {row['year']}Q{row['quarter']} = {row['value']:,.0f}")
    
    # 2. Check quarterly_data CTE
    print(f"\n🔍 Step 2: Checking Quarterly Data CTE")
    print("-" * 40)
    
    quarterly_check_query = text(f"""
        SELECT 
            fv.ticker,
            fv.item_id,
            fv.statement_type,
            fv.year,
            fv.quarter,
            CASE 
                WHEN mi.sector IS NOT NULL AND LOWER(mi.sector) LIKE '%bank%' THEN fv.value / 479618082.14
                ELSE fv.value / VND_TO_BILLIONS
            END as value,
            mi.sector
        FROM fundamental_values fv
        LEFT JOIN master_info mi ON fv.ticker = mi.ticker
        WHERE (fv.item_id IN (1, 2, 13) AND fv.statement_type = 'PL')
        OR (fv.item_id = 2 AND fv.statement_type = 'BS')
        AND fv.ticker IN ('{ticker_list}')
        AND fv.year <= {lag_year}
        ORDER BY fv.ticker, fv.item_id, fv.statement_type, fv.year DESC, fv.quarter DESC
        LIMIT 20
    """)
    
    quarterly_data = pd.read_sql(quarterly_check_query, engine)
    
    print(f"   Quarterly data records: {len(quarterly_data)}")
    if not quarterly_data.empty:
        print(f"   Sample quarterly data:")
        for _, row in quarterly_data.head(10).iterrows():
            print(f"     {row['ticker']}: Item {row['item_id']} ({row['statement_type']}) {row['year']}Q{row['quarter']} = {row['value']:.2f}")
    else:
        print("   ❌ No quarterly data found!")
    
    # 3. Check if the issue is with the WHERE clause
    print(f"\n🔍 Step 3: Checking WHERE Clause")
    print("-" * 40)
    
    # Test the WHERE clause separately
    where_test_query = text(f"""
        SELECT 
            fv.ticker,
            fv.item_id,
            fv.statement_type,
            fv.year,
            fv.quarter,
            fv.value
        FROM fundamental_values fv
        WHERE fv.ticker IN ('{ticker_list}')
        AND fv.year <= {lag_year}
        AND (
            (fv.item_id IN (1, 2, 13) AND fv.statement_type = 'PL')
            OR (fv.item_id = 2 AND fv.statement_type = 'BS')
        )
        ORDER BY fv.ticker, fv.item_id, fv.statement_type, fv.year DESC, fv.quarter DESC
        LIMIT 20
    """)
    
    where_test_data = pd.read_sql(where_test_query, engine)
    
    print(f"   WHERE test records: {len(where_test_data)}")
    if not where_test_data.empty:
        print(f"   Sample WHERE test data:")
        for _, row in where_test_data.head(10).iterrows():
            print(f"     {row['ticker']}: Item {row['item_id']} ({row['statement_type']}) {row['year']}Q{row['quarter']} = {row['value']:,.0f}")
    else:
        print("   ❌ WHERE test returned no data!")

if __name__ == "__main__":
    debug_balance_sheet() 