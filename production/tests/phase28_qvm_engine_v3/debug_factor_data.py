#!/usr/bin/env python3
"""
Debug Factor Data Issue
=======================

Debug script to investigate why factor data is empty for all rebalance dates.
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
from production.database.mappings.financial_mapping_manager import FinancialMappingManager

def debug_factor_data():
    """Debug the factor data issue."""
    print("🔍 Debugging Factor Data Issue")
    print("=" * 50)
    
    # Connect to database
    db_manager = get_database_manager()
    engine = db_manager.get_engine()
    mapping_manager = FinancialMappingManager()
    
    # Test date
    test_date = pd.Timestamp('2021-01-29')  # One of the failing dates
    lag_days = 45
    lag_date = test_date - pd.Timedelta(days=lag_days)
    lag_year = lag_date.year
    lag_quarter = ((lag_date.month - 1) // 3) + 1
    
    print(f"📅 Test Date: {test_date.date()}")
    print(f"📅 Lag Date: {lag_date.date()}")
    print(f"📅 Lag Year/Quarter: {lag_year}Q{lag_quarter}")
    
    # 1. Check universe construction
    print(f"\n🔍 Step 1: Checking Universe Construction")
    print("-" * 40)
    
    universe_query = text("""
        SELECT 
            ticker,
            AVG(total_volume) as avg_volume,
            AVG(market_cap) as avg_market_cap
        FROM vcsc_daily_data_complete
        WHERE trading_date <= :analysis_date
          AND trading_date >= DATE_SUB(:analysis_date, INTERVAL 63 DAY)
        GROUP BY ticker
        HAVING avg_volume >= 1000000 AND avg_market_cap >= 100000000000
        ORDER BY avg_volume DESC
        LIMIT 10
    """)
    
    universe_df = pd.read_sql(universe_query, engine, 
                             params={'analysis_date': test_date})
    
    print(f"   Universe size: {len(universe_df)} stocks")
    if not universe_df.empty:
        print(f"   Top 5 stocks by volume:")
        for _, row in universe_df.head().iterrows():
            print(f"     {row['ticker']}: {row['avg_volume']:,.0f} shares, {row['avg_market_cap']:,.0f} VND")
    else:
        print("   ❌ No stocks in universe!")
        return
    
    # 2. Check fundamental data availability
    print(f"\n🔍 Step 2: Checking Fundamental Data Availability")
    print("-" * 50)
    
    # Get sample tickers
    sample_tickers = universe_df['ticker'].head(5).tolist()
    ticker_list = "','".join(sample_tickers)
    
    # Check what fundamental data exists
    fundamental_check_query = text(f"""
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
        AND fv.item_id IN (1, 2, 13)
        AND fv.statement_type = 'PL'
        AND fv.year <= {lag_year}
        ORDER BY fv.ticker, fv.year DESC, fv.quarter DESC
    """)
    
    fundamental_check = pd.read_sql(fundamental_check_query, engine)
    
    print(f"   Fundamental data records: {len(fundamental_check)}")
    if not fundamental_check.empty:
        print(f"   Sample data:")
        for _, row in fundamental_check.head(10).iterrows():
            print(f"     {row['ticker']}: Item {row['item_id']} ({row['statement_type']}) {row['year']}Q{row['quarter']} = {row['value']:,.0f}")
    else:
        print("   ❌ No fundamental data found!")
        return
    
    # 3. Test the actual factor calculation query
    print(f"\n🔍 Step 3: Testing Factor Calculation Query")
    print("-" * 50)
    
    # Get dynamic mappings
    netprofit_id, netprofit_stmt = mapping_manager.get_net_profit_mapping('Corporate')
    revenue_corp_id, revenue_corp_stmt = mapping_manager.get_revenue_mapping('Corporate')
    revenue_bank_id, revenue_bank_stmt = mapping_manager.get_revenue_mapping('Banks')
    totalassets_id, totalassets_stmt = mapping_manager.get_total_assets_mapping('Corporate')
    
    print(f"   Mappings:")
    print(f"     Net Profit: {netprofit_id} ({netprofit_stmt})")
    print(f"     Revenue Corp: {revenue_corp_id} ({revenue_corp_stmt})")
    print(f"     Revenue Bank: {revenue_bank_id} ({revenue_bank_stmt})")
    print(f"     Total Assets: {totalassets_id} ({totalassets_stmt})")
    
    # Test the simplified query
    test_query = text(f"""
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
        AND fv.item_id IN (1, 2, 13)
        AND fv.statement_type = 'PL'
        AND fv.year <= {lag_year}
        ORDER BY fv.ticker, fv.year DESC, fv.quarter DESC
    """)
    
    test_result = pd.read_sql(test_query, engine)
    
    print(f"   Test query result: {len(test_result)} records")
    if not test_result.empty:
        print(f"   Sample results:")
        for _, row in test_result.head(10).iterrows():
            print(f"     {row['ticker']}: Item {row['item_id']} {row['year']}Q{row['quarter']} = {row['value']:,.0f}")
    else:
        print("   ❌ Test query returned no data!")
        return
    
    # 4. Check if the issue is with the TTM calculation
    print(f"\n🔍 Step 4: Testing TTM Calculation")
    print("-" * 40)
    
    # Check if we have enough quarters for TTM
    quarters_check_query = text(f"""
        SELECT 
            ticker,
            COUNT(*) as quarters_count,
            GROUP_CONCAT(CONCAT(year, 'Q', quarter) ORDER BY year DESC, quarter DESC) as quarters
        FROM fundamental_values fv
        WHERE fv.ticker IN ('{ticker_list}')
        AND fv.item_id = 1
        AND fv.statement_type = 'PL'
        AND fv.year <= {lag_year}
        GROUP BY ticker
        HAVING quarters_count >= 4
    """)
    
    quarters_check = pd.read_sql(quarters_check_query, engine)
    
    print(f"   Tickers with >=4 quarters: {len(quarters_check)}")
    if not quarters_check.empty:
        print(f"   Sample TTM data:")
        for _, row in quarters_check.head().iterrows():
            print(f"     {row['ticker']}: {row['quarters_count']} quarters ({row['quarters']})")
    else:
        print("   ❌ No tickers have enough quarters for TTM!")
        return
    
    # 5. Test the complete factor query
    print(f"\n🔍 Step 5: Testing Complete Factor Query")
    print("-" * 45)
    
    # Simplified version of the factor query
    factor_query = text(f"""
        WITH quarterly_data AS (
            SELECT 
                fv.ticker,
                fv.year,
                fv.quarter,
                fv.statement_type,
                CASE 
                    WHEN mi.sector IS NOT NULL AND LOWER(mi.sector) LIKE '%bank%' THEN fv.value / 479618082.14
                    ELSE fv.value / 6222444702.01
                END as value,
                fv.item_id,
                mi.sector
            FROM fundamental_values fv
            LEFT JOIN master_info mi ON fv.ticker = mi.ticker
            WHERE fv.item_id IN (1, 2, 13)
            AND fv.statement_type = 'PL'
            AND fv.ticker IN ('{ticker_list}')
            AND fv.year <= {lag_year}
        ),
        ttm_calculations AS (
            SELECT 
                ticker,
                year,
                quarter,
                sector,
                SUM(CASE WHEN item_id = 1 THEN value ELSE 0 END) as netprofit_ttm,
                SUM(CASE WHEN item_id = 2 THEN value ELSE 0 END) as revenue_nonbank_ttm,
                SUM(CASE WHEN item_id = 13 THEN value ELSE 0 END) as revenue_bank_ttm,
                SUM(CASE WHEN item_id = 2 THEN value ELSE 0 END) as totalassets_ttm
            FROM (
                SELECT 
                    ticker,
                    year,
                    quarter,
                    item_id,
                    value,
                    sector,
                    ROW_NUMBER() OVER (PARTITION BY ticker, item_id ORDER BY year DESC, quarter DESC) as rn
                FROM quarterly_data
            ) ranked
            WHERE rn <= 4
            GROUP BY ticker, year, quarter, sector
        )
        SELECT 
            ttm.ticker,
            ttm.sector,
            ttm.netprofit_ttm,
            ttm.revenue_nonbank_ttm,
            ttm.revenue_bank_ttm,
            ttm.totalassets_ttm,
            CASE 
                WHEN ttm.totalassets_ttm > 0 THEN ttm.netprofit_ttm / ttm.totalassets_ttm 
                ELSE NULL 
            END as roaa
        FROM ttm_calculations ttm
        WHERE ttm.netprofit_ttm > 0 AND ttm.totalassets_ttm > 0
        ORDER BY ttm.year DESC, ttm.quarter DESC
        LIMIT 10
    """)
    
    factor_result = pd.read_sql(factor_query, engine)
    
    print(f"   Factor query result: {len(factor_result)} records")
    if not factor_result.empty:
        print(f"   Sample factor data:")
        for _, row in factor_result.head().iterrows():
            print(f"     {row['ticker']}: ROAA = {row['roaa']:.4f}, Net Profit = {row['netprofit_ttm']:.2f}, Total Assets = {row['totalassets_ttm']:.2f}")
    else:
        print("   ❌ Factor query returned no data!")
        
        # Check what's in quarterly_data
        quarterly_check = text(f"""
            SELECT COUNT(*) as count
            FROM fundamental_values fv
            LEFT JOIN master_info mi ON fv.ticker = mi.ticker
            WHERE fv.item_id IN (1, 2, 13)
            AND fv.statement_type = 'PL'
            AND fv.ticker IN ('{ticker_list}')
            AND fv.year <= {lag_year}
        """)
        
        quarterly_count = pd.read_sql(quarterly_check, engine).iloc[0]['count']
        print(f"   Quarterly data count: {quarterly_count}")

if __name__ == "__main__":
    debug_factor_data() 