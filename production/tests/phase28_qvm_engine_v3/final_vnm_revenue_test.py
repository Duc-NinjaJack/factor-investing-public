#!/usr/bin/env python3
"""
Final VNM Revenue Test
======================

This script finds the best revenue item for VNM to fix the Net Margin issue.
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

def final_vnm_revenue_test():
    """Find the best revenue item for VNM."""
    print("🔍 Final VNM Revenue Test")
    print("=" * 50)
    
    # Connect to database
    db_manager = get_database_manager()
    engine = db_manager.get_engine()
    
    # Test different revenue items for VNM
    revenue_items = [2, 3, 4, 5, 9, 11, 13, 15, 19, 21]
    
    print(f"Testing different revenue items for VNM:")
    print("-" * 50)
    
    # Test with a recent date
    test_date = pd.Timestamp('2025-01-29')
    lag_days = 45
    lag_date = test_date - pd.Timedelta(days=lag_days)
    lag_year = lag_date.year
    lag_quarter = ((lag_date.month - 1) // 3) + 1
    
    results = []
    
    for revenue_item in revenue_items:
        print(f"\n🔍 Testing Revenue Item {revenue_item} (PL):")
        
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
                WHERE fv.ticker = 'VNM'
                AND (
                    (fv.item_id = 1 AND fv.statement_type = 'PL')
                    OR (fv.item_id = {revenue_item} AND fv.statement_type = 'PL')
                    OR (fv.item_id = 302 AND fv.statement_type = 'BS')
                )
                AND (fv.year < {lag_year} OR (fv.year = {lag_year} AND fv.quarter <= {lag_quarter}))
            ),
            ttm_calculations AS (
                SELECT 
                    ticker,
                    year,
                    quarter,
                    SUM(CASE WHEN item_id = 1 THEN value ELSE 0 END) as netprofit_ttm,
                    SUM(CASE WHEN item_id = {revenue_item} THEN value ELSE 0 END) as revenue_ttm,
                    SUM(CASE WHEN item_id = 302 THEN value ELSE 0 END) as totalassets_ttm
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
        
        try:
            test_result = pd.read_sql(test_query, engine)
            
            if not test_result.empty:
                row = test_result.iloc[0]
                roaa = row['roaa']
                net_margin = row['net_margin']
                asset_turnover = row['asset_turnover']
                
                print(f"   NetProfit: {row['netprofit_ttm']:.0f} bn VND")
                print(f"   Revenue: {row['revenue_ttm']:.0f} bn VND")
                print(f"   TotalAssets: {row['totalassets_ttm']:.0f} bn VND")
                print(f"   ROAA: {roaa:.4f} ({roaa*100:.2f}%)")
                print(f"   Net Margin: {net_margin:.4f} ({net_margin*100:.2f}%)")
                print(f"   Asset Turnover: {asset_turnover:.4f} ({asset_turnover*100:.2f}%)")
                
                # Check if ratios look reasonable
                roaa_reasonable = 0.001 <= roaa <= 0.50
                margin_reasonable = 0.001 <= net_margin <= 0.80
                turnover_reasonable = 0.01 <= asset_turnover <= 10.0
                
                # Check if close to VNM's expected 14.88% Net Margin
                margin_close_to_vnm = abs(net_margin - 0.1488) < 0.05
                
                score = sum([roaa_reasonable, margin_reasonable, turnover_reasonable])
                if margin_close_to_vnm:
                    score += 3
                
                print(f"   ROAA reasonable: {'✅' if roaa_reasonable else '❌'}")
                print(f"   Net Margin reasonable: {'✅' if margin_reasonable else '❌'}")
                print(f"   Asset Turnover reasonable: {'✅' if turnover_reasonable else '❌'}")
                print(f"   Close to VNM (14.88%): {'✅' if margin_close_to_vnm else '❌'}")
                print(f"   Score: {score}/6")
                
                results.append({
                    'revenue_item': revenue_item,
                    'roaa': roaa,
                    'net_margin': net_margin,
                    'asset_turnover': asset_turnover,
                    'roaa_reasonable': roaa_reasonable,
                    'margin_reasonable': margin_reasonable,
                    'turnover_reasonable': turnover_reasonable,
                    'margin_close_to_vnm': margin_close_to_vnm,
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
        
        print(f"\n🏆 Best Revenue Items for VNM:")
        print("=" * 50)
        for i, row in results_df.head(5).iterrows():
            print(f"\n#{i+1} - Score: {row['score']}/6")
            print(f"   Revenue Item: {row['revenue_item']}")
            print(f"   ROAA: {row['roaa']:.4f} ({row['roaa']*100:.2f}%)")
            print(f"   Net Margin: {row['net_margin']:.4f} ({row['net_margin']*100:.2f}%)")
            print(f"   Asset Turnover: {row['asset_turnover']:.4f} ({row['asset_turnover']*100:.2f}%)")
            print(f"   Close to VNM: {'✅' if row['margin_close_to_vnm'] else '❌'}")

if __name__ == "__main__":
    final_vnm_revenue_test() 