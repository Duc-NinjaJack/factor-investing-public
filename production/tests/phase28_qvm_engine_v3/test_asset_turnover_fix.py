#!/usr/bin/env python3
"""
Test Asset Turnover Fix
=======================

This script tests if the Asset Turnover fix (using Item 302 for Total Assets) works correctly.
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

def test_asset_turnover_fix():
    """Test if the Asset Turnover fix works correctly."""
    print("🔍 Testing Asset Turnover Fix")
    print("=" * 50)
    
    # Connect to database
    db_manager = get_database_manager()
    engine = db_manager.get_engine()
    
    # Test tickers
    test_tickers = ['VCB', 'VNM']
    
    for test_ticker in test_tickers:
        print(f"\n📊 Testing {test_ticker}:")
        print("-" * 30)
        
        # Test with a recent date
        test_date = pd.Timestamp('2025-01-29')
        lag_days = 45
        lag_date = test_date - pd.Timedelta(days=lag_days)
        lag_year = lag_date.year
        lag_quarter = ((lag_date.month - 1) // 3) + 1
        
        # Test with Item 302 as Total Assets (FIXED)
        test_query = text(f"""
            WITH quarterly_data AS (
                SELECT 
                    fv.ticker,
                    fv.year,
                    fv.quarter,
                    fv.value,
                    fv.item_id,
                    fv.statement_type,
                    mi.sector
                FROM fundamental_values fv
                LEFT JOIN master_info mi ON fv.ticker = mi.ticker
                WHERE fv.ticker = :ticker
                AND (
                    (fv.item_id = 1 AND fv.statement_type = 'PL')
                    OR (fv.item_id IN (2, 101) AND fv.statement_type = 'PL')
                    OR (fv.item_id = 302 AND fv.statement_type = 'BS')
                )
                AND (fv.year < {lag_year} OR (fv.year = {lag_year} AND fv.quarter <= {lag_quarter}))
            ),
            ttm_calculations AS (
                SELECT 
                    ticker,
                    year,
                    quarter,
                    sector,
                    SUM(CASE WHEN item_id = 1 THEN value ELSE 0 END) as netprofit_ttm,
                    SUM(CASE WHEN item_id = 2 THEN value ELSE 0 END) as revenue_net_sales_ttm,
                    SUM(CASE WHEN item_id = 101 THEN value ELSE 0 END) as revenue_total_income_ttm,
                    SUM(CASE WHEN item_id = 302 THEN value ELSE 0 END) as totalassets_ttm
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
            ),
            latest_data AS (
                SELECT 
                    ttm.ticker,
                    ttm.sector,
                    ttm.year,
                    ttm.quarter,
                    ttm.netprofit_ttm,
                    CASE 
                        WHEN ttm.sector IS NOT NULL AND LOWER(ttm.sector) LIKE '%bank%' THEN ttm.revenue_total_income_ttm
                        ELSE ttm.revenue_net_sales_ttm
                    END as revenue_ttm,
                    ttm.totalassets_ttm,
                    CASE 
                        WHEN ttm.totalassets_ttm > 0 THEN ttm.netprofit_ttm / ttm.totalassets_ttm 
                        ELSE NULL 
                    END as roaa,
                    CASE 
                        WHEN (CASE 
                            WHEN ttm.sector IS NOT NULL AND LOWER(ttm.sector) LIKE '%bank%' THEN ttm.revenue_total_income_ttm
                            ELSE ttm.revenue_net_sales_ttm
                        END) > 0 THEN ttm.netprofit_ttm / (CASE 
                            WHEN ttm.sector IS NOT NULL AND LOWER(ttm.sector) LIKE '%bank%' THEN ttm.revenue_total_income_ttm
                            ELSE ttm.revenue_net_sales_ttm
                        END)
                        ELSE NULL 
                    END as net_margin,
                    CASE 
                        WHEN ttm.totalassets_ttm > 0 THEN (CASE 
                            WHEN ttm.sector IS NOT NULL AND LOWER(ttm.sector) LIKE '%bank%' THEN ttm.revenue_total_income_ttm
                            ELSE ttm.revenue_net_sales_ttm
                        END) / ttm.totalassets_ttm 
                        ELSE NULL 
                    END as asset_turnover,
                    ROW_NUMBER() OVER (PARTITION BY ttm.ticker ORDER BY ttm.year DESC, ttm.quarter DESC) as rn
                FROM ttm_calculations ttm
                WHERE ttm.netprofit_ttm > 0 AND ttm.totalassets_ttm > 0
                AND (
                    (ttm.sector IS NOT NULL AND LOWER(ttm.sector) LIKE '%bank%' AND ttm.revenue_total_income_ttm > 0)
                    OR (ttm.sector IS NULL OR LOWER(ttm.sector) NOT LIKE '%bank%') AND ttm.revenue_net_sales_ttm > 0
                )
                AND (ttm.year < {lag_year} OR (ttm.year = {lag_year} AND ttm.quarter <= {lag_quarter}))
            )
            SELECT 
                ticker,
                sector,
                roaa,
                net_margin,
                asset_turnover
            FROM latest_data
            WHERE rn = 1
        """)
        
        try:
            test_result = pd.read_sql(test_query, engine, params={'ticker': test_ticker})
            
            if not test_result.empty:
                row = test_result.iloc[0]
                roaa = row['roaa']
                net_margin = row['net_margin']
                asset_turnover = row['asset_turnover']
                
                print(f"   Sector: {row['sector']}")
                print(f"   ROAA: {roaa:.4f} ({roaa*100:.2f}%)")
                print(f"   Net Margin: {net_margin:.4f} ({net_margin*100:.2f}%)")
                print(f"   Asset Turnover: {asset_turnover:.4f} ({asset_turnover*100:.2f}%)")
                
                # Check if Asset Turnover is now reasonable (not 100%)
                if asset_turnover != 1.0:
                    print(f"   ✅ Asset Turnover FIXED! No longer 100%")
                else:
                    print(f"   ❌ Asset Turnover still 100% - issue persists")
                
                # Check if ratios look reasonable
                roaa_reasonable = 0.001 <= roaa <= 0.50
                margin_reasonable = 0.001 <= net_margin <= 0.80
                turnover_reasonable = 0.01 <= asset_turnover <= 10.0
                
                print(f"   ROAA reasonable: {'✅' if roaa_reasonable else '❌'}")
                print(f"   Net Margin reasonable: {'✅' if margin_reasonable else '❌'}")
                print(f"   Asset Turnover reasonable: {'✅' if turnover_reasonable else '❌'}")
                
                if roaa_reasonable and margin_reasonable and turnover_reasonable:
                    print(f"   🎉 ALL RATIOS LOOK REASONABLE!")
                else:
                    print(f"   ⚠️  Some ratios still need adjustment")
            else:
                print(f"   ❌ No data found")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    test_asset_turnover_fix() 