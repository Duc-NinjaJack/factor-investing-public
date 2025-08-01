#!/usr/bin/env python3
"""
Test VNM Fix
============

This script tests the corrected mappings for VNM to verify that
Net Margin and Asset Turnover are now reasonable.
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
from production.database.mappings.financial_mapping_manager import FinancialMappingManager

def test_vnm_fix():
    """Test the corrected mappings for VNM."""
    print("🔍 Testing VNM Fix with Corrected Mappings")
    print("=" * 50)
    
    # Connect to database
    db_manager = get_database_manager()
    engine = db_manager.get_engine()
    
    # Initialize mapping manager
    mapping_manager = FinancialMappingManager()
    
    # Test date
    test_date = pd.Timestamp('2025-01-29')
    lag_days = 45
    lag_date = test_date - pd.Timedelta(days=lag_days)
    lag_year = lag_date.year
    lag_quarter = ((lag_date.month - 1) // 3) + 1
    
    print(f"\n📊 Testing VNM with corrected mappings:")
    print(f"   Test Date: {test_date.date()}")
    print(f"   Lag Date: {lag_date.date()}")
    print(f"   Lag Year/Quarter: {lag_year}Q{lag_quarter}")
    
    # Get corrected mappings
    netprofit_id, netprofit_stmt = mapping_manager.get_net_profit_mapping('Food & Beverage')
    revenue_id, revenue_stmt = mapping_manager.get_revenue_mapping('Food & Beverage')
    totalassets_id, totalassets_stmt = mapping_manager.get_total_assets_mapping('Food & Beverage')
    
    print(f"\n🔧 Corrected Mappings:")
    print(f"   NetProfit: Item {netprofit_id} ({netprofit_stmt})")
    print(f"   Revenue: Item {revenue_id} ({revenue_stmt})")
    print(f"   TotalAssets: Item {totalassets_id} ({totalassets_stmt})")
    
    # Test the corrected query
    test_query = text(f"""
        WITH quarterly_data AS (
            SELECT 
                fv.ticker,
                fv.year,
                fv.quarter,
                                        fv.value / 6222444702.01 as value,  -- Convert to billions using Net Profit factor
                fv.item_id,
                fv.statement_type
            FROM fundamental_values fv
            WHERE fv.ticker = 'VNM'
            AND (
                (fv.item_id = {netprofit_id} AND fv.statement_type = '{netprofit_stmt}')
                OR (fv.item_id = {revenue_id} AND fv.statement_type = '{revenue_stmt}')
                OR (fv.item_id = {totalassets_id} AND fv.statement_type = '{totalassets_stmt}')
            )
            AND (fv.year < {lag_year} OR (fv.year = {lag_year} AND fv.quarter <= {lag_quarter}))
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
        LIMIT 1
    """)
    
    result = pd.read_sql(test_query, engine)
    
    if not result.empty:
        row = result.iloc[0]
        print(f"\n✅ Test Results:")
        print(f"   NetProfit (TTM): {row['netprofit_ttm']:.0f} bn VND")
        print(f"   Revenue (TTM): {row['revenue_ttm']:.0f} bn VND")
        print(f"   TotalAssets (TTM): {row['totalassets_ttm']:.0f} bn VND")
        print(f"   ROAA: {row['roaa']:.4f} ({row['roaa']*100:.2f}%)")
        print(f"   Net Margin: {row['net_margin']:.4f} ({row['net_margin']*100:.2f}%)")
        print(f"   Asset Turnover: {row['asset_turnover']:.4f} ({row['asset_turnover']*100:.2f}%)")
        
        # Compare with manual calculations from financial statements
        print(f"\n📋 Comparison with Manual Calculations (Q2 2025):")
        print(f"   Manual Net Margin: 14.88%")
        print(f"   Manual Asset Turnover: 30.26%")
        print(f"   Database Net Margin: {row['net_margin']*100:.2f}%")
        print(f"   Database Asset Turnover: {row['asset_turnover']*100:.2f}%")
        
        # Check if ratios are reasonable
        print(f"\n🔍 Reasonableness Check:")
        
        # Net Margin check
        if 0.05 <= row['net_margin'] <= 0.30:
            print(f"   ✅ Net Margin {row['net_margin']*100:.2f}% is reasonable (5-30% expected)")
        else:
            print(f"   ❌ Net Margin {row['net_margin']*100:.2f}% is outside expected range (5-30%)")
        
        # Asset Turnover check
        if 0.50 <= row['asset_turnover'] <= 2.0:
            print(f"   ✅ Asset Turnover {row['asset_turnover']*100:.2f}% is reasonable (50-200% expected)")
        else:
            print(f"   ❌ Asset Turnover {row['asset_turnover']*100:.2f}% is outside expected range (50-200%)")
        
        # ROAA check
        if 0.05 <= row['roaa'] <= 0.25:
            print(f"   ✅ ROAA {row['roaa']*100:.2f}% is reasonable (5-25% expected)")
        else:
            print(f"   ❌ ROAA {row['roaa']*100:.2f}% is outside expected range (5-25%)")
            
    else:
        print("❌ No data found with corrected mappings")
    
    # Test other companies to ensure they still work
    print(f"\n🧪 Testing Other Companies:")
    test_companies = [
        ('VCB', 'Banks'),
        ('HPG', 'Materials'),
        ('TCB', 'Banks')
    ]
    
    for ticker, sector in test_companies:
        try:
            netprofit_id, netprofit_stmt = mapping_manager.get_net_profit_mapping(sector)
            revenue_id, revenue_stmt = mapping_manager.get_revenue_mapping(sector)
            totalassets_id, totalassets_stmt = mapping_manager.get_total_assets_mapping(sector)
            
            # Simple test query
            simple_query = text(f"""
                SELECT 
                    fv.ticker,
                    fv.item_id,
                    fv.statement_type,
                    fv.value
                FROM fundamental_values fv
                WHERE fv.ticker = :ticker
                AND (
                    (fv.item_id = {netprofit_id} AND fv.statement_type = '{netprofit_stmt}')
                    OR (fv.item_id = {revenue_id} AND fv.statement_type = '{revenue_stmt}')
                    OR (fv.item_id = {totalassets_id} AND fv.statement_type = '{totalassets_stmt}')
                )
                AND (fv.year < {lag_year} OR (fv.year = {lag_year} AND fv.quarter <= {lag_quarter}))
                ORDER BY fv.year DESC, fv.quarter DESC, fv.item_id
                LIMIT 3
            """)
            
            result = pd.read_sql(simple_query, engine, params={'ticker': ticker})
            
            if not result.empty:
                print(f"   ✅ {ticker} ({sector}): Data found with corrected mappings")
            else:
                print(f"   ⚠️ {ticker} ({sector}): No data found")
                
        except Exception as e:
            print(f"   ❌ {ticker} ({sector}): Error - {e}")

if __name__ == "__main__":
    test_vnm_fix() 