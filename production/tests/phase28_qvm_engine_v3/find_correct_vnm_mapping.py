#!/usr/bin/env python3
"""
Find Correct VNM (Non-Bank) Mapping Based on Financial Statements
==================================================================

This script finds the correct database item_ids for VNM (non-bank) based on the
VNM Income Statement and Balance Sheet data.
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
    
    # Load corporate mappings (for non-bank companies like VNM)
    with open(mappings_path / 'corp_code_name_mapping.json', 'r') as f:
        corp_mappings = json.load(f)
    
    return corp_mappings

def find_correct_vnm_mapping():
    """Find correct database item_ids for VNM (non-bank) based on financial statements."""
    print("🔍 Finding Correct VNM (Non-Bank) Mapping Based on Financial Statements")
    print("=" * 80)
    
    # Load mappings
    corp_mappings = load_vcsc_mappings()
    
    print("📋 VCSC Corporate Income Statement Mappings:")
    print("-" * 60)
    for code, name in corp_mappings['Income Statement'].items():
        print(f"   {code}: {name}")
    
    # Connect to database
    db_manager = get_database_manager()
    engine = db_manager.get_engine()
    
    # Test ticker - VNM (non-bank)
    test_ticker = 'VNM'
    
    print(f"\n📊 Analyzing data for ticker: {test_ticker} (Non-Bank)")
    print("=" * 60)
    
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
    
    # Based on VNM Income Statement, we expect:
    # - Net sales (Revenue): ~16,724 bn VND (largest PL item)
    # - Net profit after tax: ~2,488 bn VND (significant PL item)
    # - Net Margin: ~14.88%
    
    print(f"\n🎯 Expected Values from VNM Income Statement (Q2 2025):")
    print(f"   - Net sales (Revenue): 16,724.60 bn VND")
    print(f"   - Net profit after tax: 2,488.58 bn VND")
    print(f"   - Net Margin: 14.88%")
    print(f"   - Total Assets: 55,282.66 bn VND")
    
    # Test different combinations based on the data
    if not item_data.empty:
        # Get the largest PL items as candidates
        largest_items = item_data.head(15)
        
        print(f"\n🔍 Testing combinations for {test_ticker}:")
        print("-" * 60)
        
        # Test combinations of the largest items
        test_combinations = []
        
        # Based on VCSC mappings and data analysis for non-bank companies
        test_combinations.extend([
            (1, 2, 2, f"Current: NetProfit(1) + Revenue(2) + TotalAssets(2)"),
            (1, 9, 2, f"Previous Best: NetProfit(1) + Revenue(9) + TotalAssets(2)"),
            (1, 101, 2, f"Bank Mapping: NetProfit(1) + Revenue(101) + TotalAssets(2)"),
        ])
        
        # Add combinations based on largest items for non-bank
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
        test_combinations = test_combinations[:25]
        
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
                    
                    # Check if values look reasonable for non-bank companies
                    roaa_reasonable = 0.001 <= roaa <= 0.30  # 0.1% to 30% (non-bank)
                    margin_reasonable = 0.001 <= net_margin <= 0.50  # 0.1% to 50% (non-bank)
                    turnover_reasonable = 0.1 <= asset_turnover <= 5.0  # 0.1 to 5.0 (non-bank)
                    
                    # Special check for Net Margin close to VNM's 14.88%
                    margin_close_to_vnm = abs(net_margin - 0.1488) < 0.05  # Within 5% of VNM's 14.88%
                    
                    # Check if revenue is close to VNM's expected ~16,724 bn VND
                    revenue_close_to_vnm = abs(row['revenue_ttm'] - 16724600000000) < 5000000000000  # Within 5 trillion VND
                    
                    # Check if net profit is close to VNM's expected ~2,488 bn VND
                    profit_close_to_vnm = abs(row['netprofit_ttm'] - 2488580000000) < 1000000000000  # Within 1 trillion VND
                    
                    score = sum([roaa_reasonable, margin_reasonable, turnover_reasonable])
                    if margin_close_to_vnm:
                        score += 3  # Bonus for close match to VNM's 14.88%
                    if revenue_close_to_vnm:
                        score += 2  # Bonus for close revenue match
                    if profit_close_to_vnm:
                        score += 2  # Bonus for close profit match
                    
                    print(f"   ✅ Found data:")
                    print(f"      NetProfit: {row['netprofit_ttm']:.0f} bn VND")
                    print(f"      Revenue: {row['revenue_ttm']:.0f} bn VND")
                    print(f"      TotalAssets: {row['totalassets_ttm']:.0f} bn VND")
                    print(f"      ROAA: {roaa:.4f} ({roaa*100:.2f}%) - {'✅' if roaa_reasonable else '❌'}")
                    print(f"      Net Margin: {net_margin:.4f} ({net_margin*100:.2f}%) - {'✅' if margin_reasonable else '❌'}")
                    print(f"      Asset Turnover: {asset_turnover:.4f} ({asset_turnover*100:.2f}%) - {'✅' if turnover_reasonable else '❌'}")
                    print(f"      Close to VNM (14.88%): {'✅' if margin_close_to_vnm else '❌'}")
                    print(f"      Revenue close to 16,724 bn: {'✅' if revenue_close_to_vnm else '❌'}")
                    print(f"      Profit close to 2,488 bn: {'✅' if profit_close_to_vnm else '❌'}")
                    print(f"      Score: {score}/8")
                    
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
                        'margin_close_to_vnm': margin_close_to_vnm,
                        'revenue_close_to_vnm': revenue_close_to_vnm,
                        'profit_close_to_vnm': profit_close_to_vnm,
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
            
            print(f"\n🏆 Best Combinations for {test_ticker} (Non-Bank):")
            print("=" * 70)
            for i, row in results_df.head(10).iterrows():
                print(f"\n#{results_df.index.get_loc(i)+1} - Score: {row['score']}/8")
                print(f"   {row['description']}")
                print(f"   NetProfit: {row['netprofit_id']}, Revenue: {row['revenue_id']}, TotalAssets: {row['totalassets_id']}")
                print(f"   ROAA: {row['roaa']:.4f} ({row['roaa']*100:.2f}%)")
                print(f"   Net Margin: {row['net_margin']:.4f} ({row['net_margin']*100:.2f}%)")
                print(f"   Asset Turnover: {row['asset_turnover']:.4f} ({row['asset_turnover']*100:.2f}%)")
                print(f"   Close to VNM: {'✅' if row['margin_close_to_vnm'] else '❌'}")
                print(f"   Revenue Match: {'✅' if row['revenue_close_to_vnm'] else '❌'}")
                print(f"   Profit Match: {'✅' if row['profit_close_to_vnm'] else '❌'}")

if __name__ == "__main__":
    find_correct_vnm_mapping() 