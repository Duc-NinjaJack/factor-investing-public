#!/usr/bin/env python3
"""
Find Correct Total Assets Mapping
=================================

This script finds the correct database item_id for Total Assets by analyzing
the Balance Sheet structure and comparing with VCSC mappings.
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
    
    # Load both mappings
    with open(mappings_path / 'corp_code_name_mapping.json', 'r') as f:
        corp_mappings = json.load(f)
    
    with open(mappings_path / 'bank_code_name_mapping.json', 'r') as f:
        bank_mappings = json.load(f)
    
    return corp_mappings, bank_mappings

def find_correct_total_assets():
    """Find correct database item_id for Total Assets."""
    print("🔍 Finding Correct Total Assets Mapping")
    print("=" * 60)
    
    # Load mappings
    corp_mappings, bank_mappings = load_vcsc_mappings()
    
    print("📋 VCSC Corporate Balance Sheet Mappings:")
    print("-" * 50)
    for code, name in corp_mappings['Balance Sheet'].items():
        if 'total' in name.lower() or 'assets' in name.lower():
            print(f"   {code}: {name}")
    
    print("\n📋 VCSC Bank Balance Sheet Mappings:")
    print("-" * 50)
    for code, name in bank_mappings['Balance Sheet'].items():
        if 'total' in name.lower() or 'assets' in name.lower():
            print(f"   {code}: {name}")
    
    # Connect to database
    db_manager = get_database_manager()
    engine = db_manager.get_engine()
    
    # Test tickers - both bank and non-bank
    test_tickers = ['VCB', 'VNM']
    
    for test_ticker in test_tickers:
        print(f"\n📊 Analyzing Balance Sheet data for ticker: {test_ticker}")
        print("=" * 60)
        
        # Get all available BS item_ids for this ticker
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
            AND fv.statement_type = 'BS'
            GROUP BY fv.item_id, fv.statement_type
            ORDER BY avg_value DESC
        """)
        
        item_data = pd.read_sql(item_query, engine, params={'ticker': test_ticker})
        print(f"Available BS item_ids for {test_ticker} (sorted by average value):")
        print(item_data.to_string(index=False))
        
        # Based on VNM Balance Sheet, we expect:
        # - Total Assets: ~55,282 bn VND (from Balance Sheet)
        
        print(f"\n🎯 Expected Values from {test_ticker} Balance Sheet:")
        if test_ticker == 'VNM':
            print(f"   - Total Assets: 55,282.66 bn VND")
        else:
            print(f"   - Total Assets: Should be largest BS item")
        
        # Test different item_ids as potential Total Assets
        if not item_data.empty:
            print(f"\n🔍 Testing potential Total Assets items for {test_ticker}:")
            print("-" * 60)
            
            # Test the largest BS items
            largest_items = item_data.head(10)
            
            for i, row in largest_items.iterrows():
                item_id = int(row['item_id'])
                avg_value = row['avg_value']
                
                print(f"\n🔍 Testing Item {item_id} (BS):")
                print(f"   Average Value: {avg_value:.0f} bn VND")
                
                # Check if this looks like Total Assets
                if test_ticker == 'VNM':
                    # For VNM, we expect ~55,282 bn VND
                    expected_total_assets = 55282660000000  # 55,282.66 bn VND
                    close_to_expected = abs(avg_value - expected_total_assets) < 10000000000000  # Within 10 trillion
                    
                    if close_to_expected:
                        print(f"   ✅ CLOSE MATCH to VNM Total Assets (55,282 bn VND)")
                    else:
                        print(f"   ❌ Not close to expected Total Assets")
                
                # Check if value is reasonable for Total Assets
                reasonable_size = 1000000000000 <= avg_value <= 1000000000000000  # 1 trillion to 1 quadrillion VND
                print(f"   Reasonable size for Total Assets: {'✅' if reasonable_size else '❌'}")
                
                # Check if it's the largest item (Total Assets should be the largest)
                is_largest = i == 0
                print(f"   Largest BS item: {'✅' if is_largest else '❌'}")
                
                # Calculate a score
                score = 0
                if reasonable_size:
                    score += 1
                if is_largest:
                    score += 2
                if test_ticker == 'VNM' and close_to_expected:
                    score += 3
                
                print(f"   Score: {score}/6")
                
                # If this looks promising, test it with our factor calculation
                if score >= 3:
                    print(f"   🎯 PROMISING CANDIDATE for Total Assets!")
                    
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
                                (fv.item_id = 1 AND fv.statement_type = 'PL')
                                OR (fv.item_id = 2 AND fv.statement_type = 'PL')
                                OR (fv.item_id = {item_id} AND fv.statement_type = 'BS')
                            )
                            AND (fv.year < {lag_year} OR (fv.year = {lag_year} AND fv.quarter <= {lag_quarter}))
                        ),
                        ttm_calculations AS (
                            SELECT 
                                ticker,
                                year,
                                quarter,
                                SUM(CASE WHEN item_id = 1 THEN value ELSE 0 END) as netprofit_ttm,
                                SUM(CASE WHEN item_id = 2 THEN value ELSE 0 END) as revenue_ttm,
                                SUM(CASE WHEN item_id = {item_id} THEN value ELSE 0 END) as totalassets_ttm
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
                        test_result = pd.read_sql(test_query, engine, params={'ticker': test_ticker})
                        
                        if not test_result.empty:
                            row = test_result.iloc[0]
                            roaa = row['roaa']
                            net_margin = row['net_margin']
                            asset_turnover = row['asset_turnover']
                            
                            print(f"   📊 Test Results:")
                            print(f"      NetProfit: {row['netprofit_ttm']:.0f} bn VND")
                            print(f"      Revenue: {row['revenue_ttm']:.0f} bn VND")
                            print(f"      TotalAssets: {row['totalassets_ttm']:.0f} bn VND")
                            print(f"      ROAA: {roaa:.4f} ({roaa*100:.2f}%)")
                            print(f"      Net Margin: {net_margin:.4f} ({net_margin*100:.2f}%)")
                            print(f"      Asset Turnover: {asset_turnover:.4f} ({asset_turnover*100:.2f}%)")
                            
                            # Check if ratios look reasonable
                            roaa_reasonable = 0.001 <= roaa <= 0.50
                            margin_reasonable = 0.001 <= net_margin <= 0.80
                            turnover_reasonable = 0.01 <= asset_turnover <= 10.0
                            
                            print(f"      ROAA reasonable: {'✅' if roaa_reasonable else '❌'}")
                            print(f"      Net Margin reasonable: {'✅' if margin_reasonable else '❌'}")
                            print(f"      Asset Turnover reasonable: {'✅' if turnover_reasonable else '❌'}")
                            
                            if roaa_reasonable and margin_reasonable and turnover_reasonable:
                                print(f"   🎉 EXCELLENT CANDIDATE! All ratios look reasonable!")
                        else:
                            print(f"   ❌ No data found with this item_id")
                            
                    except Exception as e:
                        print(f"   ❌ Error testing: {e}")

if __name__ == "__main__":
    find_correct_total_assets() 