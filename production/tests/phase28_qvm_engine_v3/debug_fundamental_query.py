#!/usr/bin/env python3
"""
Debug Fundamental Query
======================

This script debugs the fundamental data query step by step to identify issues.
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

def debug_fundamental_query():
    """Debug the fundamental data query step by step."""
    print("🔍 Debugging Fundamental Data Query")
    print("=" * 50)
    
    # Connect to database
    db_manager = get_database_manager()
    engine = db_manager.get_engine()
    
    # Test parameters
    test_date = pd.Timestamp('2021-01-29')
    lag_days = 45
    lag_date = test_date - pd.Timedelta(days=lag_days)
    lag_year = lag_date.year
    lag_quarter = ((lag_date.month - 1) // 3) + 1
    
    print(f"📅 Test Date: {test_date.date()}")
    print(f"📅 Lag Date: {lag_date.date()} (Year: {lag_year}, Quarter: {lag_quarter})")
    
    # Step 1: Check if we have data for TCB
    print(f"\n1️⃣ Checking raw data for TCB:")
    print("-" * 30)
    
    raw_query = text("""
        SELECT 
            fv.ticker,
            fv.year,
            fv.quarter,
            fv.value,
            fv.item_id,
            fv.statement_type
        FROM fundamental_values fv
        WHERE fv.ticker = 'TCB'
        AND fv.year >= 2020
        AND (
            (fv.item_id = 1 AND fv.statement_type = 'PL')
            OR (fv.item_id = 2 AND fv.statement_type = 'PL')
            OR (fv.item_id = 2 AND fv.statement_type = 'BS')
        )
        ORDER BY fv.year DESC, fv.quarter DESC, fv.item_id, fv.statement_type
        LIMIT 20
    """)
    
    raw_data = pd.read_sql(raw_query, engine)
    print(f"Raw data count: {len(raw_data)}")
    if not raw_data.empty:
        print(raw_data.to_string(index=False))
    else:
        print("❌ No raw data found!")
        return False
    
    # Step 2: Check quarterly data
    print(f"\n2️⃣ Checking quarterly data:")
    print("-" * 30)
    
    quarterly_query = text(f"""
        SELECT 
            fv.ticker,
            fv.year,
            fv.quarter,
            fv.value,
            fv.item_id,
            fv.statement_type
        FROM fundamental_values fv
        WHERE fv.ticker = 'TCB'
        AND (
            (fv.item_id = 1 AND fv.statement_type = 'PL')
            OR (fv.item_id = 2 AND fv.statement_type = 'PL')
            OR (fv.item_id = 2 AND fv.statement_type = 'BS')
        )
        AND (fv.year < {lag_year} OR (fv.year = {lag_year} AND fv.quarter <= {lag_quarter}))
        ORDER BY fv.year DESC, fv.quarter DESC, fv.item_id, fv.statement_type
    """)
    
    quarterly_data = pd.read_sql(quarterly_query, engine)
    print(f"Quarterly data count: {len(quarterly_data)}")
    if not quarterly_data.empty:
        print(quarterly_data.head(10).to_string(index=False))
    else:
        print("❌ No quarterly data found!")
        return False
    
    # Step 3: Check TTM calculations
    print(f"\n3️⃣ Checking TTM calculations:")
    print("-" * 30)
    
    ttm_query = text(f"""
        WITH quarterly_data AS (
            SELECT 
                fv.ticker,
                fv.year,
                fv.quarter,
                fv.value,
                fv.item_id,
                fv.statement_type
            FROM fundamental_values fv
            WHERE fv.ticker = 'TCB'
            AND (
                (fv.item_id = 1 AND fv.statement_type = 'PL')
                OR (fv.item_id = 2 AND fv.statement_type = 'PL')
                OR (fv.item_id = 2 AND fv.statement_type = 'BS')
            )
            AND (fv.year < {lag_year} OR (fv.year = {lag_year} AND fv.quarter <= {lag_quarter}))
        )
        SELECT 
            ticker,
            year,
            quarter,
            SUM(CASE WHEN item_id = 1 AND statement_type = 'PL' THEN value ELSE 0 END) as netprofit_ttm,
            SUM(CASE WHEN item_id = 2 AND statement_type = 'PL' THEN value ELSE 0 END) as revenue_ttm,
            SUM(CASE WHEN item_id = 2 AND statement_type = 'BS' THEN value ELSE 0 END) as totalassets_ttm
        FROM (
            SELECT 
                ticker,
                year,
                quarter,
                item_id,
                statement_type,
                value,
                ROW_NUMBER() OVER (PARTITION BY ticker, item_id, statement_type ORDER BY year DESC, quarter DESC) as rn
            FROM quarterly_data
        ) ranked
        WHERE rn <= 4
        GROUP BY ticker, year, quarter
        ORDER BY year DESC, quarter DESC
        LIMIT 5
    """)
    
    ttm_data = pd.read_sql(ttm_query, engine)
    print(f"TTM data count: {len(ttm_data)}")
    if not ttm_data.empty:
        print(ttm_data.to_string(index=False))
    else:
        print("❌ No TTM data found!")
        return False
    
    # Step 4: Check final calculations
    print(f"\n4️⃣ Checking final calculations:")
    print("-" * 30)
    
    final_query = text(f"""
        WITH quarterly_data AS (
            SELECT 
                fv.ticker,
                fv.year,
                fv.quarter,
                fv.value,
                fv.item_id,
                fv.statement_type
            FROM fundamental_values fv
            WHERE fv.ticker = 'TCB'
            AND (
                (fv.item_id = 1 AND fv.statement_type = 'PL')
                OR (fv.item_id = 2 AND fv.statement_type = 'PL')
                OR (fv.item_id = 2 AND fv.statement_type = 'BS')
            )
            AND (fv.year < {lag_year} OR (fv.year = {lag_year} AND fv.quarter <= {lag_quarter}))
        ),
        ttm_calculations AS (
            SELECT 
                ticker,
                year,
                quarter,
                SUM(CASE WHEN item_id = 1 AND statement_type = 'PL' THEN value ELSE 0 END) as netprofit_ttm,
                SUM(CASE WHEN item_id = 2 AND statement_type = 'PL' THEN value ELSE 0 END) as revenue_ttm,
                SUM(CASE WHEN item_id = 2 AND statement_type = 'BS' THEN value ELSE 0 END) as totalassets_ttm
            FROM (
                SELECT 
                    ticker,
                    year,
                    quarter,
                    item_id,
                    statement_type,
                    value,
                    ROW_NUMBER() OVER (PARTITION BY ticker, item_id, statement_type ORDER BY year DESC, quarter DESC) as rn
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
    
    final_data = pd.read_sql(final_query, engine)
    print(f"Final data count: {len(final_data)}")
    if not final_data.empty:
        print(final_data.to_string(index=False))
        
        # Check if values look reasonable
        roaa_reasonable = all(0.001 <= roaa <= 0.50 for roaa in final_data['roaa'])
        margin_reasonable = all(0.001 <= margin <= 0.80 for margin in final_data['net_margin'])
        turnover_reasonable = all(0.01 <= turnover <= 10.0 for turnover in final_data['asset_turnover'])
        
        print(f"\n🎯 Reasonableness Check:")
        print(f"   - ROAA values reasonable: {roaa_reasonable}")
        print(f"   - Net Margin values reasonable: {margin_reasonable}")
        print(f"   - Asset Turnover values reasonable: {turnover_reasonable}")
        
        if roaa_reasonable and margin_reasonable and turnover_reasonable:
            print(f"   ✅ ALL VALUES LOOK REASONABLE!")
            return True
        else:
            print(f"   ⚠️ Some values may still be incorrect")
            return False
    else:
        print("❌ No final data found!")
        return False

if __name__ == "__main__":
    success = debug_fundamental_query()
    if success:
        print(f"\n✅ DEBUG COMPLETE: Query is working correctly!")
    else:
        print(f"\n❌ DEBUG COMPLETE: Query still has issues!") 