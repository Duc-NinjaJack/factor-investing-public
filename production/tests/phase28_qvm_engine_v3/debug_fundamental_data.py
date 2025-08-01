#!/usr/bin/env python3
"""
Debug fundamental data availability and query issues
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

def debug_fundamental_data():
    """Debug fundamental data availability and query issues"""
    
    # Get database connection
    db_manager = get_database_manager()
    engine = db_manager.get_engine()
    
    print("🔍 DEBUGGING FUNDAMENTAL DATA AVAILABILITY")
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
    
    # Test dates
    test_dates = [
        pd.Timestamp('2020-01-30'),
        pd.Timestamp('2021-06-30'),
        pd.Timestamp('2022-06-30'),
        pd.Timestamp('2023-06-30'),
        pd.Timestamp('2024-06-30')
    ]
    
    print(f"\n1. CHECKING DATA AVAILABILITY BY YEAR")
    print("-" * 60)
    
    for year in range(2020, 2025):
        year_query = text("""
            SELECT 
                COUNT(*) as total_records,
                COUNT(DISTINCT ticker) as unique_tickers,
                COUNT(DISTINCT year) as years,
                COUNT(DISTINCT quarter) as quarters
            FROM fundamental_values
            WHERE year = %s
            AND item_id IN (%s, %s, %s)
        """ % (year, 
               test_config['factors']['netprofit_item_id'],
               test_config['factors']['revenue_item_id'],
               test_config['factors']['totalassets_item_id']))
        
        try:
            result = pd.read_sql(year_query, engine)
            print(f"   {year}: {result.iloc[0]['total_records']:,} records, {result.iloc[0]['unique_tickers']:,} tickers, {result.iloc[0]['quarters']} quarters")
        except Exception as e:
            print(f"   {year}: ❌ Query failed - {e}")
    
    print(f"\n2. CHECKING SPECIFIC TICKERS")
    print("-" * 60)
    
    for ticker in test_universe:
        ticker_query = text("""
            SELECT 
                ticker,
                year,
                quarter,
                item_id,
                value
            FROM fundamental_values
            WHERE ticker = %s
            AND item_id IN (%s, %s, %s)
            AND year >= 2020
            ORDER BY year DESC, quarter DESC, item_id
            LIMIT 10
        """ % (ticker,
               test_config['factors']['netprofit_item_id'],
               test_config['factors']['revenue_item_id'],
               test_config['factors']['totalassets_item_id']))
        
        try:
            result = pd.read_sql(ticker_query, engine)
            print(f"   {ticker}: {len(result)} records")
            if not result.empty:
                for _, row in result.head(3).iterrows():
                    print(f"     - {row['year']}Q{row['quarter']} Item_{row['item_id']}: {row['value']:,.0f}")
        except Exception as e:
            print(f"   {ticker}: ❌ Query failed - {e}")
    
    print(f"\n3. TESTING SIMPLIFIED FUNDAMENTAL QUERY")
    print("-" * 60)
    
    for test_date in test_dates:
        print(f"\n   Testing date: {test_date}")
        
        # Calculate lagged date
        lag_days = test_config['factors']['fundamental_lag_days']
        lag_date = test_date - pd.Timedelta(days=lag_days)
        lag_year = lag_date.year
        lag_quarter = ((lag_date.month - 1) // 3) + 1
        
        print(f"     Lagged date: {lag_date} (Year: {lag_year}, Quarter: {lag_quarter})")
        
        # Test simple query first
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
            AND fv.year = %s
            AND fv.quarter <= %s
            ORDER BY fv.ticker, fv.quarter, fv.item_id
        """ % (test_config['factors']['netprofit_item_id'],
               test_config['factors']['revenue_item_id'],
               test_config['factors']['totalassets_item_id'],
               lag_year, lag_quarter))
        
        try:
            result = pd.read_sql(simple_query, engine)
            print(f"     ✅ Found {len(result)} records")
            
            if not result.empty:
                print(f"     Sample data:")
                for _, row in result.head(3).iterrows():
                    print(f"       - {row['ticker']} {row['year']}Q{row['quarter']} Item_{row['item_id']}: {row['value']:,.0f}")
            else:
                print(f"     ⚠️ No data found for this period")
                
        except Exception as e:
            print(f"     ❌ Query failed: {e}")
    
    print(f"\n4. TESTING TTM CALCULATION")
    print("-" * 60)
    
    # Test TTM calculation for a specific ticker and date
    test_ticker = 'VNM'
    test_year = 2024
    test_quarter = 2
    
    ttm_query = text("""
        WITH quarterly_data AS (
            SELECT 
                fv.ticker,
                fv.year,
                fv.quarter,
                fv.value,
                fv.item_id
            FROM fundamental_values fv
            WHERE fv.item_id IN (%s, %s, %s)
            AND fv.ticker = '%s'
            AND (fv.year < %s OR (fv.year = %s AND fv.quarter <= %s))
        ),
        ttm_calculations AS (
            SELECT 
                ticker,
                year,
                quarter,
                SUM(CASE WHEN item_id = %s THEN value ELSE 0 END) as netprofit_ttm,
                SUM(CASE WHEN item_id = %s THEN value ELSE 0 END) as revenue_ttm,
                SUM(CASE WHEN item_id = %s THEN value ELSE 0 END) as totalassets_ttm
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
            WHERE rn <= 4  -- Last 4 quarters for TTM
            GROUP BY ticker, year, quarter
        )
        SELECT 
            ttm.ticker,
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
            END as net_margin
        FROM ttm_calculations ttm
        WHERE ttm.netprofit_ttm > 0 AND ttm.revenue_ttm > 0 AND ttm.totalassets_ttm > 0
        AND (ttm.year < %s OR (ttm.year = %s AND ttm.quarter <= %s))
        ORDER BY ttm.year DESC, ttm.quarter DESC
    """ % (test_config['factors']['netprofit_item_id'],
           test_config['factors']['revenue_item_id'],
           test_config['factors']['totalassets_item_id'],
           test_ticker, test_year, test_year, test_quarter,
           test_config['factors']['netprofit_item_id'],
           test_config['factors']['revenue_item_id'],
           test_config['factors']['totalassets_item_id'],
           test_year, test_year, test_quarter))
    
    try:
        result = pd.read_sql(ttm_query, engine)
        print(f"   TTM calculation for {test_ticker} {test_year}Q{test_quarter}:")
        print(f"   ✅ Found {len(result)} TTM records")
        
        if not result.empty:
            for _, row in result.head(3).iterrows():
                print(f"     - {row['year']}Q{row['quarter']}: ROAA={row['roaa']:.4f}, NetMargin={row['net_margin']:.4f}")
                print(f"       NetProfit_TTM={row['netprofit_ttm']:,.0f}, Revenue_TTM={row['revenue_ttm']:,.0f}")
        else:
            print(f"   ⚠️ No TTM data found")
            
    except Exception as e:
        print(f"   ❌ TTM query failed: {e}")

if __name__ == "__main__":
    debug_fundamental_data() 