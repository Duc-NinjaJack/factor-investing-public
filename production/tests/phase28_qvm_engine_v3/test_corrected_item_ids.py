#!/usr/bin/env python3
"""
Test Corrected Item IDs Script
==============================

This script tests the corrected item_ids to ensure TTM calculation works properly.
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

def test_corrected_item_ids():
    """Test the corrected item_ids for TTM calculation."""
    
    # Get database connection
    db_manager = get_database_manager()
    engine = db_manager.get_engine()
    
    # Corrected item_ids
    corrected_config = {
        'netprofit_item_id': 4,     # NetProfit from CF statement
        'revenue_item_id': 2,       # Revenue from BS statement  
        'totalassets_item_id': 2,   # TotalAssets from BS statement
        'fundamental_lag_days': 45
    }
    
    test_ticker = 'TCB'
    test_date = pd.Timestamp('2021-06-30')
    
    # Calculate lag date
    lag_days = corrected_config['fundamental_lag_days']
    lag_date = test_date - pd.Timedelta(days=lag_days)
    lag_year = lag_date.year
    lag_quarter = ((lag_date.month - 1) // 3) + 1
    
    print("="*80)
    print("🔧 TESTING CORRECTED ITEM IDs")
    print("="*80)
    print(f"Test Ticker: {test_ticker}")
    print(f"Test Date: {test_date}")
    print(f"Corrected Item IDs:")
    print(f"   NetProfit: {corrected_config['netprofit_item_id']}")
    print(f"   Revenue: {corrected_config['revenue_item_id']}")
    print(f"   TotalAssets: {corrected_config['totalassets_item_id']}")
    print(f"Lag Date: {lag_date} (Year: {lag_year}, Quarter: {lag_quarter})")
    print()
    
    # Test the exact same query as the strategy with corrected item_ids
    strategy_query = text("""
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
            AND (year < :lag_year OR (year = :lag_year AND quarter <= :lag_quarter))
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
            ttm.ticker,
            mi.sector,
            ttm.year,
            ttm.quarter,
            ttm.netprofit_ttm,
            ttm.revenue_ttm,
            ttm.totalassets_ttm,
            CASE 
                WHEN ttm.totalassets_ttm > 0 THEN ttm.netprofit_ttm / ttm.totalassets_ttm 
                ELSE NULL 
            END as roaa,
            CASE 
                WHEN ttm.revenue_ttm > 0 THEN (ttm.netprofit_ttm / ttm.revenue_ttm)
                ELSE NULL 
            END as net_margin,
            CASE 
                WHEN ttm.totalassets_ttm > 0 THEN ttm.revenue_ttm / ttm.totalassets_ttm 
                ELSE NULL 
            END as asset_turnover
        FROM ttm_calculations ttm
        LEFT JOIN master_info mi ON ttm.ticker = mi.ticker
        WHERE ttm.netprofit_ttm > 0 AND ttm.revenue_ttm > 0 AND ttm.totalassets_ttm > 0
        ORDER BY ttm.year DESC, ttm.quarter DESC
    """)
    
    strategy_params = {
        'ticker': test_ticker,
        'netprofit_id': corrected_config['netprofit_item_id'],
        'revenue_id': corrected_config['revenue_item_id'],
        'totalassets_id': corrected_config['totalassets_item_id'],
        'lag_year': lag_year,
        'lag_quarter': lag_quarter
    }
    
    result = pd.read_sql(strategy_query, engine, params=strategy_params)
    
    print("📊 Strategy Query Results:")
    print(f"   Records found: {len(result)}")
    if not result.empty:
        print(f"   ✅ SUCCESS! TTM calculation works with corrected item_ids")
        print(f"   Sample data:")
        print(result.to_string())
        
        # Show the ratios
        print(f"\n📈 Calculated Ratios:")
        for _, row in result.iterrows():
            print(f"   {row['ticker']} {row['year']}Q{row['quarter']} ({row['sector']}):")
            print(f"     ROAA: {row['roaa']:.6f}")
            print(f"     Net Margin: {row['net_margin']:.6f}")
            print(f"     Asset Turnover: {row['asset_turnover']:.6f}")
    else:
        print(f"   ❌ Still no data - need to investigate further")
        
        # Check what the TTM values look like before filtering
        check_query = text("""
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
                AND (year < :lag_year OR (year = :lag_year AND quarter <= :lag_quarter))
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
        """)
        
        check_result = pd.read_sql(check_query, engine, params=strategy_params)
        print(f"   TTM values before filtering:")
        print(check_result.to_string())
    
    print("\n" + "="*80)
    print("🔧 CORRECTED ITEM IDs TEST SUMMARY")
    print("="*80)
    if not result.empty:
        print("✅ SUCCESS: The corrected item_ids work properly!")
        print("   The strategy should now be able to calculate TTM values correctly.")
    else:
        print("❌ ISSUE: Still need to investigate the filtering conditions.")
        print("   The TTM calculation works but the final filtering removes all data.")

if __name__ == "__main__":
    test_corrected_item_ids() 