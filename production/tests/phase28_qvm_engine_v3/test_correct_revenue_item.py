#!/usr/bin/env python3
"""
Test Correct Revenue Item ID
===========================

This script tests Item 9 (PL) as the correct Revenue item_id based on the data analysis.
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

def test_correct_revenue_item():
    """Test Item 9 (PL) as the correct Revenue item_id."""
    print("🔍 Testing Correct Revenue Item ID")
    print("=" * 60)
    print("   NetProfit: Item 1 (PL)")
    print("   Revenue: Item 9 (PL) - Based on data analysis")
    print("   TotalAssets: Item 2 (BS)")
    print("=" * 60)
    
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
    
    # Test the corrected combination
    test_combinations = [
        (1, 9, 2, "Corrected: NetProfit(1) + Revenue(9) + TotalAssets(2)"),
        (1, 2, 2, "Current: NetProfit(1) + Revenue(2) + TotalAssets(2)"),
        (1, 11, 2, "Alternative: NetProfit(1) + Revenue(11) + TotalAssets(2)"),
        (1, 13, 2, "Alternative: NetProfit(1) + Revenue(13) + TotalAssets(2)"),
    ]
    
    results = []
    
    for netprofit_id, revenue_id, totalassets_id, description in test_combinations:
        print(f"\n🔍 Testing: {description}")
        print(f"   NetProfit: {netprofit_id}, Revenue: {revenue_id}, TotalAssets: {totalassets_id}")
        
        try:
            test_query = text(f"""
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
                        SUM(CASE WHEN item_id = {netprofit_id} AND statement_type = 'PL' THEN value ELSE 0 END) as netprofit_ttm,
                        SUM(CASE WHEN item_id = {revenue_id} AND statement_type = 'PL' THEN value ELSE 0 END) as revenue_ttm,
                        SUM(CASE WHEN item_id = {totalassets_id} AND statement_type = 'BS' THEN value ELSE 0 END) as totalassets_ttm
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
                LIMIT 1
            """)
            
            test_result = pd.read_sql(test_query, engine)
            
            if not test_result.empty:
                row = test_result.iloc[0]
                roaa = row['roaa']
                net_margin = row['net_margin']
                asset_turnover = row['asset_turnover']
                
                # Check if values look reasonable
                roaa_reasonable = 0.001 <= roaa <= 0.50  # 0.1% to 50%
                margin_reasonable = 0.001 <= net_margin <= 0.80  # 0.1% to 80%
                turnover_reasonable = 0.01 <= asset_turnover <= 10.0  # 0.01 to 10.0
                
                score = sum([roaa_reasonable, margin_reasonable, turnover_reasonable])
                
                print(f"   ✅ Found data:")
                print(f"      NetProfit: {row['netprofit_ttm']:.0f}")
                print(f"      Revenue: {row['revenue_ttm']:.0f}")
                print(f"      TotalAssets: {row['totalassets_ttm']:.0f}")
                print(f"      ROAA: {roaa:.4f} ({roaa*100:.2f}%) - {'✅' if roaa_reasonable else '❌'}")
                print(f"      Net Margin: {net_margin:.4f} ({net_margin*100:.2f}%) - {'✅' if margin_reasonable else '❌'}")
                print(f"      Asset Turnover: {asset_turnover:.4f} ({asset_turnover*100:.2f}%) - {'✅' if turnover_reasonable else '❌'}")
                print(f"      Score: {score}/3")
                
                results.append({
                    'netprofit_id': netprofit_id,
                    'revenue_id': revenue_id,
                    'totalassets_id': totalassets_id,
                    'description': description,
                    'roaa': roaa,
                    'net_margin': net_margin,
                    'asset_turnover': asset_turnover,
                    'roaa_reasonable': roaa_reasonable,
                    'margin_reasonable': margin_reasonable,
                    'turnover_reasonable': turnover_reasonable,
                    'score': score
                })
            else:
                print(f"   ❌ No data found")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Sort results by score
    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('score', ascending=False)
        
        print(f"\n🏆 Best Combinations:")
        print("=" * 50)
        for i, row in results_df.head(5).iterrows():
            print(f"\n#{results_df.index.get_loc(i)+1} - Score: {row['score']}/3")
            print(f"   {row['description']}")
            print(f"   NetProfit: {row['netprofit_id']}, Revenue: {row['revenue_id']}, TotalAssets: {row['totalassets_id']}")
            print(f"   ROAA: {row['roaa']:.4f} ({row['roaa']*100:.2f}%)")
            print(f"   Net Margin: {row['net_margin']:.4f} ({row['net_margin']*100:.2f}%)")
            print(f"   Asset Turnover: {row['asset_turnover']:.4f} ({row['asset_turnover']*100:.2f}%)")
        
        # Find best combination with all 3 reasonable
        perfect_combinations = results_df[results_df['score'] == 3]
        if not perfect_combinations.empty:
            best = perfect_combinations.iloc[0]
            print(f"\n🎯 PERFECT COMBINATION FOUND:")
            print("=" * 40)
            print(f"   NetProfit: {best['netprofit_id']}")
            print(f"   Revenue: {best['revenue_id']}")
            print(f"   TotalAssets: {best['totalassets_id']}")
            print(f"   Description: {best['description']}")
            return best['netprofit_id'], best['revenue_id'], best['totalassets_id']
        else:
            print(f"\n⚠️ No perfect combination found. Using best available.")
            best = results_df.iloc[0]
            return best['netprofit_id'], best['revenue_id'], best['totalassets_id']
    
    return None, None, None

if __name__ == "__main__":
    netprofit_id, revenue_id, totalassets_id = test_correct_revenue_item()
    if netprofit_id is not None:
        print(f"\n✅ RECOMMENDED ITEM IDs:")
        print(f"   NetProfit: {netprofit_id}")
        print(f"   Revenue: {revenue_id}")
        print(f"   TotalAssets: {totalassets_id}")
    else:
        print(f"\n❌ No suitable combination found!") 