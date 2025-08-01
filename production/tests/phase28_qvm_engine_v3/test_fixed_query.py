#!/usr/bin/env python3
"""
Test Fixed Query
===============

Test script to verify the fixed fundamental data query works correctly.
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

def test_fixed_query():
    """Test the fixed fundamental data query."""
    print("🧪 Testing Fixed Query")
    print("=" * 40)
    
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
    
    # Test the fixed query
    fixed_query = text(f"""
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
            WHERE (fv.item_id IN (1, 2, 13) AND fv.statement_type = 'PL')
            OR (fv.item_id = 2 AND fv.statement_type = 'BS')
            AND fv.ticker IN ('{ticker_list}')
            AND fv.year <= {lag_year}
        ),
        ttm_calculations AS (
            SELECT 
                ticker,
                year,
                quarter,
                sector,
                SUM(CASE WHEN item_id = 1 AND statement_type = 'PL' THEN value ELSE 0 END) as netprofit_ttm,
                SUM(CASE WHEN item_id = 2 AND statement_type = 'PL' THEN value ELSE 0 END) as revenue_nonbank_ttm,
                SUM(CASE WHEN item_id = 13 AND statement_type = 'PL' THEN value ELSE 0 END) as revenue_bank_ttm,
                SUM(CASE WHEN item_id = 2 AND statement_type = 'BS' THEN value ELSE 0 END) as totalassets_ttm
            FROM (
                SELECT 
                    ticker,
                    year,
                    quarter,
                    item_id,
                    value,
                    sector,
                    statement_type,
                    ROW_NUMBER() OVER (PARTITION BY ticker, item_id, statement_type ORDER BY year DESC, quarter DESC) as rn
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
            END as roaa,
            CASE 
                WHEN (CASE 
                    WHEN ttm.sector IS NOT NULL AND LOWER(ttm.sector) LIKE '%bank%' THEN ttm.revenue_bank_ttm
                    ELSE ttm.revenue_nonbank_ttm
                END) > 0 THEN ttm.netprofit_ttm / (CASE 
                    WHEN ttm.sector IS NOT NULL AND LOWER(ttm.sector) LIKE '%bank%' THEN ttm.revenue_bank_ttm
                    ELSE ttm.revenue_nonbank_ttm
                END)
                ELSE NULL 
            END as net_margin
        FROM ttm_calculations ttm
        WHERE ttm.netprofit_ttm > 0 AND ttm.totalassets_ttm > 0
        ORDER BY ttm.year DESC, ttm.quarter DESC
        LIMIT 10
    """)
    
    result = pd.read_sql(fixed_query, engine)
    
    print(f"✅ Fixed query result: {len(result)} records")
    if not result.empty:
        print(f"   Sample results:")
        for _, row in result.head().iterrows():
            print(f"     {row['ticker']}: ROAA = {row['roaa']:.4f}, Net Margin = {row['net_margin']:.4f}")
            print(f"       Net Profit = {row['netprofit_ttm']:.2f}, Total Assets = {row['totalassets_ttm']:.2f}")
    else:
        print("   ❌ Fixed query returned no data!")

if __name__ == "__main__":
    test_fixed_query() 