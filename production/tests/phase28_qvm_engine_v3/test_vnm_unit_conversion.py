#!/usr/bin/env python3
"""
Test VNM Unit Conversion
========================

This script tests VNM with unit conversion to fix the scaling issue
between database and financial statement values.
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

def test_vnm_unit_conversion():
    """Test VNM with unit conversion."""
    print("🔍 Testing VNM Unit Conversion")
    print("=" * 50)
    
    # Connect to database
    db_manager = get_database_manager()
    engine = db_manager.get_engine()
    
    # Known values from VNM financial statements (Q2 2025)
    known_net_sales = 16724.60  # bn VND
    known_net_profit = 2488.58  # bn VND
    known_total_assets = 55282.66  # bn VND
    
    print(f"📋 Known Values from VNM Financial Statements (Q2 2025):")
    print(f"   Net Sales: {known_net_sales:.2f} bn VND")
    print(f"   Net Profit: {known_net_profit:.2f} bn VND")
    print(f"   Total Assets: {known_total_assets:.2f} bn VND")
    
    # Test date
    test_date = pd.Timestamp('2025-01-29')
    lag_days = 45
    lag_date = test_date - pd.Timedelta(days=lag_days)
    lag_year = lag_date.year
    lag_quarter = ((lag_date.month - 1) // 3) + 1
    
    print(f"\n🔍 Testing with unit conversion (database values / 1000):")
    print(f"   Test Date: {test_date.date()}")
    print(f"   Lag Date: {lag_date.date()}")
    print(f"   Lag Year/Quarter: {lag_year}Q{lag_quarter}")
    
    # Test different item combinations with unit conversion
    test_combinations = [
        (1, 2, 2),  # NetProfit, Revenue, TotalAssets
        (1, 3, 2),  # NetProfit, Alternative Revenue, TotalAssets
        (1, 4, 2),  # NetProfit, Another Revenue, TotalAssets
        (1, 15, 2), # NetProfit, Another Revenue, TotalAssets
    ]
    
    print(f"\n🧪 Testing different combinations with unit conversion:")
    print("-" * 80)
    
    for netprofit_id, revenue_id, totalassets_id in test_combinations:
        try:
            # Test query with unit conversion
            test_query = text(f"""
                WITH quarterly_data AS (
                    SELECT 
                        fv.ticker,
                        fv.year,
                        fv.quarter,
                        fv.value / 6222444702.01 as value_bn,  -- Convert using Net Profit factor
                        fv.item_id,
                        fv.statement_type
                    FROM fundamental_values fv
                    WHERE fv.ticker = 'VNM'
                    AND (
                        (fv.item_id = {netprofit_id} AND fv.statement_type = 'PL')
                        OR (fv.item_id = {revenue_id} AND fv.statement_type = 'PL')
                        OR (fv.item_id = {totalassets_id} AND fv.statement_type = 'BS')
                    )
                    AND (fv.year < {lag_year} OR (fv.year = {lag_year} AND fv.quarter <= {lag_quarter}))
                ),
                ttm_calculations AS (
                    SELECT 
                        ticker,
                        year,
                        quarter,
                        SUM(CASE WHEN item_id = {netprofit_id} THEN value_bn ELSE 0 END) as netprofit_ttm,
                        SUM(CASE WHEN item_id = {revenue_id} THEN value_bn ELSE 0 END) as revenue_ttm,
                        SUM(CASE WHEN item_id = {totalassets_id} THEN value_bn ELSE 0 END) as totalassets_ttm
                    FROM (
                        SELECT 
                            ticker,
                            year,
                            quarter,
                            item_id,
                            value_bn,
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
                print(f"\n   Combination: NetProfit={netprofit_id}, Revenue={revenue_id}, TotalAssets={totalassets_id}")
                print(f"   NetProfit (TTM): {row['netprofit_ttm']:.0f} bn VND")
                print(f"   Revenue (TTM): {row['revenue_ttm']:.0f} bn VND")
                print(f"   TotalAssets (TTM): {row['totalassets_ttm']:.0f} bn VND")
                print(f"   ROAA: {row['roaa']:.4f} ({row['roaa']*100:.2f}%)")
                print(f"   Net Margin: {row['net_margin']:.4f} ({row['net_margin']*100:.2f}%)")
                print(f"   Asset Turnover: {row['asset_turnover']:.4f} ({row['asset_turnover']*100:.2f}%)")
                
                # Compare with known values
                netprofit_diff = abs(row['netprofit_ttm'] - known_net_profit)
                revenue_diff = abs(row['revenue_ttm'] - known_net_sales)
                totalassets_diff = abs(row['totalassets_ttm'] - known_total_assets)
                
                print(f"   Comparison with known values:")
                print(f"     NetProfit diff: {netprofit_diff:.0f} bn VND")
                print(f"     Revenue diff: {revenue_diff:.0f} bn VND")
                print(f"     TotalAssets diff: {totalassets_diff:.0f} bn VND")
                
                # Check reasonableness
                score = 0
                if 0.05 <= row['net_margin'] <= 0.30:
                    score += 1
                if 0.50 <= row['asset_turnover'] <= 2.0:
                    score += 1
                if 0.05 <= row['roaa'] <= 0.25:
                    score += 1
                if revenue_diff < 1000:  # Within 1000 bn VND
                    score += 1
                if netprofit_diff < 500:  # Within 500 bn VND
                    score += 1
                
                print(f"   Reasonableness Score: {score}/5")
                
            else:
                print(f"\n   Combination: NetProfit={netprofit_id}, Revenue={revenue_id}, TotalAssets={totalassets_id}")
                print(f"   ❌ No data found")
                
        except Exception as e:
            print(f"\n   Combination: NetProfit={netprofit_id}, Revenue={revenue_id}, TotalAssets={totalassets_id}")
            print(f"   ❌ Error: {e}")
    
    # Test the best combination with the mapping manager
    print(f"\n🎯 Testing with FinancialMappingManager:")
    print("-" * 50)
    
    try:
        from production.database.mappings.financial_mapping_manager import FinancialMappingManager
        
        mapping_manager = FinancialMappingManager()
        
        netprofit_id, netprofit_stmt = mapping_manager.get_net_profit_mapping('Food & Beverage')
        revenue_id, revenue_stmt = mapping_manager.get_revenue_mapping('Food & Beverage')
        totalassets_id, totalassets_stmt = mapping_manager.get_total_assets_mapping('Food & Beverage')
        
        print(f"   Mappings: NetProfit={netprofit_id}, Revenue={revenue_id}, TotalAssets={totalassets_id}")
        
        # Test with unit conversion
        mapping_query = text(f"""
            WITH quarterly_data AS (
                SELECT 
                    fv.ticker,
                    fv.year,
                    fv.quarter,
                    fv.value / 6222444702.01 as value_bn,  -- Convert using Net Profit factor
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
                    SUM(CASE WHEN item_id = {netprofit_id} THEN value_bn ELSE 0 END) as netprofit_ttm,
                    SUM(CASE WHEN item_id = {revenue_id} THEN value_bn ELSE 0 END) as revenue_ttm,
                    SUM(CASE WHEN item_id = {totalassets_id} THEN value_bn ELSE 0 END) as totalassets_ttm
                FROM (
                    SELECT 
                        ticker,
                        year,
                        quarter,
                        item_id,
                        value_bn,
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
        
        mapping_result = pd.read_sql(mapping_query, engine)
        
        if not mapping_result.empty:
            row = mapping_result.iloc[0]
            print(f"   ✅ Mapping Manager Results:")
            print(f"     NetProfit (TTM): {row['netprofit_ttm']:.0f} bn VND")
            print(f"     Revenue (TTM): {row['revenue_ttm']:.0f} bn VND")
            print(f"     TotalAssets (TTM): {row['totalassets_ttm']:.0f} bn VND")
            print(f"     ROAA: {row['roaa']:.4f} ({row['roaa']*100:.2f}%)")
            print(f"     Net Margin: {row['net_margin']:.4f} ({row['net_margin']*100:.2f}%)")
            print(f"     Asset Turnover: {row['asset_turnover']:.4f} ({row['asset_turnover']*100:.2f}%)")
        else:
            print(f"   ❌ No data found with mapping manager")
            
    except Exception as e:
        print(f"   ❌ Error with mapping manager: {e}")

if __name__ == "__main__":
    test_vnm_unit_conversion() 