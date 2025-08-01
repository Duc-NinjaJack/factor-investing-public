#!/usr/bin/env python3
"""
Investigate Fundamental Data Quality Issues
==========================================

This script investigates the fundamental data quality issues to identify
the correct item_ids and data structure.
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

def investigate_fundamental_data():
    """Investigate fundamental data quality issues."""
    print("🔍 Investigating Fundamental Data Quality Issues")
    print("=" * 60)
    
    # Connect to database
    db_manager = get_database_manager()
    engine = db_manager.get_engine()
    
    # Test ticker
    test_ticker = 'TCB'
    
    print(f"📊 Investigating data for ticker: {test_ticker}")
    
    # 1. Check what item_ids are available
    print(f"\n1️⃣ Checking available item_ids for {test_ticker}:")
    print("-" * 50)
    
    item_query = text("""
        SELECT DISTINCT 
            fv.item_id,
            fv.statement_type,
            COUNT(*) as count,
            MIN(fv.value) as min_value,
            MAX(fv.value) as max_value,
            AVG(fv.value) as avg_value
        FROM fundamental_values fv
        WHERE fv.ticker = :ticker
        GROUP BY fv.item_id, fv.statement_type
        ORDER BY fv.item_id
    """)
    
    item_data = pd.read_sql(item_query, engine, params={'ticker': test_ticker})
    print(f"Available item_ids for {test_ticker}:")
    print(item_data.to_string(index=False))
    
    # 2. Check the current item_ids we're using
    print(f"\n2️⃣ Checking current item_ids (4, 2, 2):")
    print("-" * 50)
    
    current_query = text("""
        SELECT 
            fv.item_id,
            fv.statement_type,
            fv.year,
            fv.quarter,
            fv.value
        FROM fundamental_values fv
        WHERE fv.ticker = :ticker
        AND fv.item_id IN (2, 4)
        ORDER BY fv.item_id, fv.year DESC, fv.quarter DESC
        LIMIT 20
    """)
    
    current_data = pd.read_sql(current_query, engine, params={'ticker': test_ticker})
    print(f"Current item_ids data for {test_ticker}:")
    print(current_data.to_string(index=False))
    
    # 3. Check if there are other item_ids that might be correct
    print(f"\n3️⃣ Looking for potential correct item_ids:")
    print("-" * 50)
    
    # Check for different statement types
    statement_query = text("""
        SELECT DISTINCT statement_type
        FROM fundamental_values
        WHERE ticker = :ticker
    """)
    
    statements = pd.read_sql(statement_query, engine, params={'ticker': test_ticker})
    print(f"Available statement types for {test_ticker}:")
    print(statements.to_string(index=False))
    
    # 4. Check for NetProfit, Revenue, TotalAssets in different ways
    print(f"\n4️⃣ Searching for NetProfit, Revenue, TotalAssets:")
    print("-" * 50)
    
    # Look for item_ids with reasonable values
    reasonable_query = text("""
        SELECT 
            fv.item_id,
            fv.statement_type,
            COUNT(*) as count,
            MIN(fv.value) as min_value,
            MAX(fv.value) as max_value,
            AVG(fv.value) as avg_value,
            STDDEV(fv.value) as std_value
        FROM fundamental_values fv
        WHERE fv.ticker = :ticker
        AND fv.value > 0
        GROUP BY fv.item_id, fv.statement_type
        HAVING count >= 4  -- At least 4 quarters of data
        ORDER BY avg_value DESC
        LIMIT 10
    """)
    
    reasonable_data = pd.read_sql(reasonable_query, engine, params={'ticker': test_ticker})
    print(f"Item_ids with reasonable values for {test_ticker}:")
    print(reasonable_data.to_string(index=False))
    
    # 5. Test different item_id combinations
    print(f"\n5️⃣ Testing different item_id combinations:")
    print("-" * 50)
    
    # Test some common item_ids
    test_combinations = [
        (1, 1, 1),   # Common combination
        (2, 2, 2),   # Current combination
        (3, 3, 3),   # Another possibility
        (4, 4, 4),   # Another possibility
        (5, 5, 5),   # Another possibility
    ]
    
    for netprofit_id, revenue_id, totalassets_id in test_combinations:
        print(f"\nTesting combination: NetProfit={netprofit_id}, Revenue={revenue_id}, TotalAssets={totalassets_id}")
        
        test_query = text(f"""
            WITH quarterly_data AS (
                SELECT 
                    fv.ticker,
                    fv.year,
                    fv.quarter,
                    fv.value,
                    fv.item_id
                FROM fundamental_values fv
                WHERE fv.item_id IN ({netprofit_id}, {revenue_id}, {totalassets_id})
                AND fv.ticker = :ticker
                AND fv.year >= 2020
            ),
            ttm_calculations AS (
                SELECT 
                    ticker,
                    year,
                    quarter,
                    SUM(CASE WHEN item_id = {netprofit_id} THEN value ELSE 0 END) as netprofit_ttm,
                    SUM(CASE WHEN item_id = {revenue_id} THEN value ELSE 0 END) as revenue_ttm,
                    SUM(CASE WHEN item_id = {totalassets_id} THEN value ELSE 0 END) as totalassets_ttm
                FROM (
                    SELECT 
                        ticker,
                        year,
                        quarter,
                        item_id,
                        value,
                        ROW_NUMBER() OVER (PARTITION BY ticker, item_id ORDER BY year DESC, quarter DESC) as rn
                    FROM quarterly_data
                ) ranked
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
                CASE WHEN totalassets_ttm > 0 THEN netprofit_ttm / totalassets_ttm ELSE NULL END as roaa,
                CASE WHEN revenue_ttm > 0 THEN netprofit_ttm / revenue_ttm ELSE NULL END as net_margin,
                CASE WHEN totalassets_ttm > 0 THEN revenue_ttm / totalassets_ttm ELSE NULL END as asset_turnover
            FROM ttm_calculations
            WHERE netprofit_ttm > 0 AND revenue_ttm > 0 AND totalassets_ttm > 0
            ORDER BY year DESC, quarter DESC
            LIMIT 3
        """)
        
        try:
            test_result = pd.read_sql(test_query, engine, params={'ticker': test_ticker})
            if not test_result.empty:
                print(f"   ✅ Found data:")
                print(f"      ROAA range: {test_result['roaa'].min():.4f} to {test_result['roaa'].max():.4f}")
                print(f"      Net Margin range: {test_result['net_margin'].min():.4f} to {test_result['net_margin'].max():.4f}")
                print(f"      Asset Turnover range: {test_result['asset_turnover'].min():.4f} to {test_result['asset_turnover'].max():.4f}")
                
                # Check if values look reasonable
                roaa_reasonable = 0.01 <= test_result['roaa'].iloc[0] <= 0.50  # 1% to 50%
                margin_reasonable = 0.05 <= test_result['net_margin'].iloc[0] <= 0.80  # 5% to 80%
                turnover_reasonable = 0.1 <= test_result['asset_turnover'].iloc[0] <= 5.0  # 0.1 to 5.0
                
                if roaa_reasonable and margin_reasonable and turnover_reasonable:
                    print(f"      🎯 LOOKS REASONABLE!")
                else:
                    print(f"      ⚠️ Values may be incorrect")
            else:
                print(f"   ❌ No data found")
        except Exception as e:
            print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    investigate_fundamental_data() 