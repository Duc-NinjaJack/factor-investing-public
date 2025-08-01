#!/usr/bin/env python3
"""
Find Correct VCB Mapping Based on Income Statement
==================================================

This script finds the correct database item_ids based on the VCB Income Statement
data and VCSC bank mappings.
"""

import sys
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine, text
import json

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

def load_vcsc_mappings():
    """Load VCSC financial statement mappings."""
    mappings_path = project_root / 'production' / 'database' / 'mappings'
    
    # Load bank mappings
    with open(mappings_path / 'bank_code_name_mapping.json', 'r') as f:
        bank_mappings = json.load(f)
    
    return bank_mappings

def find_correct_vcb_mapping():
    """Find correct database item_ids based on VCB Income Statement."""
    print("🔍 Finding Correct VCB Mapping Based on Income Statement")
    print("=" * 70)
    
    # Load mappings
    bank_mappings = load_vcsc_mappings()
    
    print("📋 VCSC Bank Income Statement Mappings:")
    print("-" * 50)
    for code, name in bank_mappings['Income Statement'].items():
        print(f"   {code}: {name}")
    
    # Connect to database
    db_manager = get_database_manager()
    engine = db_manager.get_engine()
    
    # Test ticker - let's try VCB first, then TCB
    test_tickers = ['VCB', 'TCB']
    
    for test_ticker in test_tickers:
        print(f"\n📊 Analyzing data for ticker: {test_ticker}")
        print("=" * 50)
        
        # Get all available item_ids for this ticker
        item_query = text("""
            SELECT DISTINCT 
                fv.item_id,
                fv.statement_type,
                COUNT(*) as count,
                MIN(fv.value) as min_value,
                MAX(fv.value) as max_value,
                AVG(fv.value) as avg_value
            FROM fundamental_values fv
            WHERE fv.ticker = :ticker
            AND fv.statement_type = 'PL'
            GROUP BY fv.item_id, fv.statement_type
            ORDER BY avg_value DESC
        """)
        
        item_data = pd.read_sql(item_query, engine, params={'ticker': test_ticker})
        print(f"Available PL item_ids for {test_ticker} (sorted by average value):")
        print(item_data.to_string(index=False))
        
        # Based on VCB Income Statement, we expect:
        # - Revenue (Net sales): ~16,724 bn VND (largest PL item)
        # - Net Profit: ~2,488 bn VND (significant PL item)
        # - Net Margin: ~14.88%
        
        print(f"\n🎯 Expected Values from VCB Income Statement (Q2 2025):")
        print(f"   - Net sales (Revenue): 16,724.60 bn VND")
        print(f"   - Net profit after tax: 2,488.58 bn VND")
        print(f"   - Net Margin: 14.88%")
        
        # Test different combinations based on the data
        if not item_data.empty:
            # Get the largest PL items as candidates
            largest_items = item_data.head(10)
            
            print(f"\n🔍 Testing combinations for {test_ticker}:")
            print("-" * 50)
            
            # Test combinations of the largest items
            test_combinations = []
            
            # Based on VCSC mappings and data analysis
            test_combinations.extend([
                (1, 2, 2, f"Current: NetProfit(1) + Revenue(2) + TotalAssets(2)"),
                (1, 9, 2, f"Previous Best: NetProfit(1) + Revenue(9) + TotalAssets(2)"),
                (21, 2, 2, f"VCSC Mapping: NetProfit(21) + Revenue(2) + TotalAssets(2)"),
                (21, 9, 2, f"VCSC + Previous: NetProfit(21) + Revenue(9) + TotalAssets(2)"),
            ])
            
            # Add combinations based on largest items
            for i, row1 in largest_items.iterrows():
                for j, row2 in largest_items.iterrows():
                    if i != j:
                        test_combinations.append((
                            int(row1['item_id']), 
                            int(row2['item_id']), 
                            2,  # TotalAssets always item 2 (BS)
                            f"Data-driven: NetProfit({int(row1['item_id'])}) + Revenue({int(row2['item_id'])}) + TotalAssets(2)"
                        ))
            
            # Limit to top combinations to avoid too many tests
            test_combinations = test_combinations[:20]
            
            results = []
            
            for netprofit_id, revenue_id, totalassets_id, description in test_combinations:
                print(f"\n🔍 Testing: {description}")
                
                try:
                    # Test with a recent date
                    test_date = pd.Timestamp('2025-01-29')
                    lag_days = 45
                    lag_date = test_date - pd.Timedelta(days=lag_days)
                    lag_year = lag_date.year
                    lag_quarter = ((lag_date.month - 1) // 3) + 1
                    
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
                            WHERE fv.ticker = :ticker
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
                    
                    test_result = pd.read_sql(test_query, engine, params={'ticker': test_ticker})
                    
                    if not test_result.empty:
                        row = test_result.iloc[0]
                        roaa = row['roaa']
                        net_margin = row['net_margin']
                        asset_turnover = row['asset_turnover']
                        
                        # Check if values look reasonable
                        roaa_reasonable = 0.001 <= roaa <= 0.50  # 0.1% to 50%
                        margin_reasonable = 0.001 <= net_margin <= 0.80  # 0.1% to 80%
                        turnover_reasonable = 0.01 <= asset_turnover <= 10.0  # 0.01 to 10.0
                        
                        # Special check for Net Margin close to VCB's 14.88%
                        margin_close_to_vcb = abs(net_margin - 0.1488) < 0.05  # Within 5% of VCB's 14.88%
                        
                        score = sum([roaa_reasonable, margin_reasonable, turnover_reasonable])
                        if margin_close_to_vcb:
                            score += 2  # Bonus for close match to VCB
                        
                        print(f"   ✅ Found data:")
                        print(f"      NetProfit: {row['netprofit_ttm']:.0f} bn VND")
                        print(f"      Revenue: {row['revenue_ttm']:.0f} bn VND")
                        print(f"      TotalAssets: {row['totalassets_ttm']:.0f} bn VND")
                        print(f"      ROAA: {roaa:.4f} ({roaa*100:.2f}%) - {'✅' if roaa_reasonable else '❌'}")
                        print(f"      Net Margin: {net_margin:.4f} ({net_margin*100:.2f}%) - {'✅' if margin_reasonable else '❌'}")
                        print(f"      Asset Turnover: {asset_turnover:.4f} ({asset_turnover*100:.2f}%) - {'✅' if turnover_reasonable else '❌'}")
                        print(f"      Close to VCB (14.88%): {'✅' if margin_close_to_vcb else '❌'}")
                        print(f"      Score: {score}/5")
                        
                        results.append({
                            'ticker': test_ticker,
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
                            'margin_close_to_vcb': margin_close_to_vcb,
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
                
                print(f"\n🏆 Best Combinations for {test_ticker}:")
                print("=" * 60)
                for i, row in results_df.head(5).iterrows():
                    print(f"\n#{results_df.index.get_loc(i)+1} - Score: {row['score']}/5")
                    print(f"   {row['description']}")
                    print(f"   NetProfit: {row['netprofit_id']}, Revenue: {row['revenue_id']}, TotalAssets: {row['totalassets_id']}")
                    print(f"   ROAA: {row['roaa']:.4f} ({row['roaa']*100:.2f}%)")
                    print(f"   Net Margin: {row['net_margin']:.4f} ({row['net_margin']*100:.2f}%)")
                    print(f"   Asset Turnover: {row['asset_turnover']:.4f} ({row['asset_turnover']*100:.2f}%)")
                    print(f"   Close to VCB: {'✅' if row['margin_close_to_vcb'] else '❌'}")

if __name__ == "__main__":
    find_correct_vcb_mapping() 