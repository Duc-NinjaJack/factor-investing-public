#!/usr/bin/env python3
"""
Check Item IDs Script
====================

This script checks what item_ids are actually available for a test ticker
to identify the correct NetProfit item_id.
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

def check_item_ids():
    """Check what item_ids are available for TCB."""
    
    # Get database connection
    db_manager = get_database_manager()
    engine = db_manager.get_engine()
    
    test_ticker = 'TCB'
    
    print("="*80)
    print("🔍 ITEM ID INVESTIGATION")
    print("="*80)
    print(f"Test Ticker: {test_ticker}")
    print()
    
    # Check all item_ids for TCB
    print("📊 All item_ids for TCB:")
    all_items_query = text("""
        SELECT DISTINCT item_id, COUNT(*) as record_count
        FROM fundamental_values
        WHERE ticker = :ticker
        AND year BETWEEN 2020 AND 2024
        GROUP BY item_id
        ORDER BY item_id
    """)
    
    all_items = pd.read_sql(all_items_query, engine, params={'ticker': test_ticker})
    print(all_items.to_string())
    print()
    
    # Check for NetProfit-like items (look for items with negative values in CF statement)
    print("📊 NetProfit candidates (negative values in CF statement):")
    netprofit_candidates_query = text("""
        SELECT 
            item_id,
            statement_type,
            COUNT(*) as record_count,
            MIN(value) as min_value,
            MAX(value) as max_value,
            AVG(value) as avg_value
        FROM fundamental_values
        WHERE ticker = :ticker
        AND year BETWEEN 2020 AND 2024
        AND statement_type = 'CF'
        AND value < 0
        GROUP BY item_id, statement_type
        ORDER BY item_id
    """)
    
    netprofit_candidates = pd.read_sql(netprofit_candidates_query, engine, params={'ticker': test_ticker})
    print(netprofit_candidates.to_string())
    print()
    
    # Check for Revenue-like items (positive values in BS statement)
    print("📊 Revenue candidates (positive values in BS statement):")
    revenue_candidates_query = text("""
        SELECT 
            item_id,
            statement_type,
            COUNT(*) as record_count,
            MIN(value) as min_value,
            MAX(value) as max_value,
            AVG(value) as avg_value
        FROM fundamental_values
        WHERE ticker = :ticker
        AND year BETWEEN 2020 AND 2024
        AND statement_type = 'BS'
        AND value > 0
        GROUP BY item_id, statement_type
        ORDER BY item_id
    """)
    
    revenue_candidates = pd.read_sql(revenue_candidates_query, engine, params={'ticker': test_ticker})
    print(revenue_candidates.to_string())
    print()
    
    # Check for TotalAssets-like items (large positive values in BS statement)
    print("📊 TotalAssets candidates (large positive values in BS statement):")
    totalassets_candidates_query = text("""
        SELECT 
            item_id,
            statement_type,
            COUNT(*) as record_count,
            MIN(value) as min_value,
            MAX(value) as max_value,
            AVG(value) as avg_value
        FROM fundamental_values
        WHERE ticker = :ticker
        AND year BETWEEN 2020 AND 2024
        AND statement_type = 'BS'
        AND value > 1e12  -- Large values (trillions)
        GROUP BY item_id, statement_type
        ORDER BY item_id
    """)
    
    totalassets_candidates = pd.read_sql(totalassets_candidates_query, engine, params={'ticker': test_ticker})
    print(totalassets_candidates.to_string())
    print()
    
    # Test with the actual item_ids we found
    print("📊 Testing with actual item_ids found:")
    
    # Get the most likely candidates
    if not netprofit_candidates.empty:
        netprofit_id = netprofit_candidates.iloc[0]['item_id']
    else:
        netprofit_id = 1501  # Fallback
    
    if not revenue_candidates.empty:
        revenue_id = revenue_candidates.iloc[0]['item_id']
    else:
        revenue_id = 10701  # Fallback
    
    if not totalassets_candidates.empty:
        totalassets_id = totalassets_candidates.iloc[0]['item_id']
    else:
        totalassets_id = 107  # Fallback
    
    print(f"   NetProfit ID: {netprofit_id}")
    print(f"   Revenue ID: {revenue_id}")
    print(f"   TotalAssets ID: {totalassets_id}")
    
    # Test TTM calculation with these IDs
    test_ttm_query = text("""
        WITH quarterly_data AS (
            SELECT 
                ticker,
                year,
                quarter,
                item_id,
                value
            FROM fundamental_values
            WHERE ticker = :ticker
            AND item_id IN (:netprofit_id, :revenue_id, :totalassets_id)
            AND year BETWEEN 2020 AND 2024
        ),
        ranked_data AS (
            SELECT 
                ticker,
                year,
                quarter,
                item_id,
                value,
                ROW_NUMBER() OVER (PARTITION BY ticker, item_id ORDER BY year DESC, quarter DESC) as rn
            FROM quarterly_data
        ),
        ttm_calculations AS (
            SELECT 
                ticker,
                year,
                quarter,
                SUM(CASE WHEN item_id = :netprofit_id THEN value ELSE 0 END) as netprofit_ttm,
                SUM(CASE WHEN item_id = :revenue_id THEN value ELSE 0 END) as revenue_ttm,
                SUM(CASE WHEN item_id = :totalassets_id THEN value ELSE 0 END) as totalassets_ttm
            FROM ranked_data
            WHERE rn <= 4
            GROUP BY ticker, year, quarter
        )
        SELECT 
            ticker,
            year,
            quarter,
            netprofit_ttm,
            revenue_ttm,
            totalassets_ttm,
            netprofit_ttm > 0 as netprofit_positive,
            revenue_ttm > 0 as revenue_positive,
            totalassets_ttm > 0 as totalassets_positive
        FROM ttm_calculations
        ORDER BY year DESC, quarter DESC
        LIMIT 5
    """)
    
    test_params = {
        'ticker': test_ticker,
        'netprofit_id': netprofit_id,
        'revenue_id': revenue_id,
        'totalassets_id': totalassets_id
    }
    
    test_result = pd.read_sql(test_ttm_query, engine, params=test_params)
    print(f"\n   TTM test results:")
    print(test_result.to_string())
    
    print("\n" + "="*80)
    print("🔍 ITEM ID INVESTIGATION SUMMARY")
    print("="*80)
    print("This shows the actual item_ids available and tests TTM calculation.")
    print("The correct item_ids should be identified from this analysis.")

if __name__ == "__main__":
    check_item_ids() 