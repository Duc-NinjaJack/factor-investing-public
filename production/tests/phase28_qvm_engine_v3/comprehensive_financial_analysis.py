#!/usr/bin/env python3
"""
Comprehensive Financial Analysis
================================

This script analyzes financial data to identify correct mappings for
Net Margin and Asset Turnover calculations.
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

def comprehensive_financial_analysis():
    """Comprehensive analysis of financial data to identify correct mappings."""
    print("🔍 Comprehensive Financial Analysis")
    print("=" * 60)
    
    # Connect to database
    db_manager = get_database_manager()
    engine = db_manager.get_engine()
    
    # Test companies
    test_companies = [
        ('VNM', 'Food & Beverage'),
        ('VCB', 'Banks'),
        ('HPG', 'Materials'),
        ('TCB', 'Banks')
    ]
    
    for ticker, sector in test_companies:
        print(f"\n📊 Analyzing {ticker} ({sector}):")
        print("=" * 50)
        
        # Test date
        test_date = pd.Timestamp('2025-01-29')
        lag_days = 45
        lag_date = test_date - pd.Timedelta(days=lag_days)
        lag_year = lag_date.year
        lag_quarter = ((lag_date.month - 1) // 3) + 1
        
        # 1. Show all available PL items for this ticker
        print(f"\n1️⃣ Available Income Statement (PL) Items for {ticker}:")
        print("-" * 50)
        pl_query = text("""
            SELECT DISTINCT 
                fv.item_id,
                COUNT(*) as count,
                MIN(fv.value) as min_value,
                MAX(fv.value) as max_value,
                AVG(fv.value) as avg_value
            FROM fundamental_values fv
            WHERE fv.ticker = :ticker
            AND fv.statement_type = 'PL'
            AND (fv.year < :lag_year OR (fv.year = :lag_year AND fv.quarter <= :lag_quarter))
            GROUP BY fv.item_id
            ORDER BY avg_value DESC
        """)
        
        pl_data = pd.read_sql(pl_query, engine, params={
            'ticker': ticker,
            'lag_year': lag_year,
            'lag_quarter': lag_quarter
        })
        
        if not pl_data.empty:
            print(pl_data.to_string(index=False))
        else:
            print("No PL data found")
        
        # 2. Show all available BS items for this ticker
        print(f"\n2️⃣ Available Balance Sheet (BS) Items for {ticker}:")
        print("-" * 50)
        bs_query = text("""
            SELECT DISTINCT 
                fv.item_id,
                COUNT(*) as count,
                MIN(fv.value) as min_value,
                MAX(fv.value) as max_value,
                AVG(fv.value) as avg_value
            FROM fundamental_values fv
            WHERE fv.ticker = :ticker
            AND fv.statement_type = 'BS'
            AND (fv.year < :lag_year OR (fv.year = :lag_year AND fv.quarter <= :lag_quarter))
            GROUP BY fv.item_id
            ORDER BY avg_value DESC
        """)
        
        bs_data = pd.read_sql(bs_query, engine, params={
            'ticker': ticker,
            'lag_year': lag_year,
            'lag_quarter': lag_quarter
        })
        
        if not bs_data.empty:
            print(bs_data.to_string(index=False))
        else:
            print("No BS data found")
        
        # 3. Test different Revenue and TotalAssets combinations
        print(f"\n3️⃣ Testing Different Revenue/TotalAssets Combinations for {ticker}:")
        print("-" * 60)
        
        # Get top PL items (potential revenue)
        top_pl_items = pl_data.head(5)['item_id'].tolist() if not pl_data.empty else []
        # Get top BS items (potential total assets)
        top_bs_items = bs_data.head(5)['item_id'].tolist() if not bs_data.empty else []
        
        # Test combinations
        test_combinations = []
        for pl_item in top_pl_items[:3]:  # Top 3 PL items
            for bs_item in top_bs_items[:3]:  # Top 3 BS items
                test_combinations.append((pl_item, bs_item))
        
        results = []
        for revenue_id, totalassets_id in test_combinations:
            try:
                # Test with NetProfit = 1 (PL)
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
                            OR (fv.item_id = {revenue_id} AND fv.statement_type = 'PL')
                            OR (fv.item_id = {totalassets_id} AND fv.statement_type = 'BS')
                        )
                        AND (fv.year < :lag_year OR (fv.year = :lag_year AND fv.quarter <= :lag_quarter))
                    ),
                    ttm_calculations AS (
                        SELECT 
                            ticker,
                            year,
                            quarter,
                            SUM(CASE WHEN item_id = 1 THEN value ELSE 0 END) as netprofit_ttm,
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
                
                result = pd.read_sql(test_query, engine, params={
                    'ticker': ticker,
                    'lag_year': lag_year,
                    'lag_quarter': lag_quarter
                })
                
                if not result.empty:
                    row = result.iloc[0]
                    results.append({
                        'revenue_id': revenue_id,
                        'totalassets_id': totalassets_id,
                        'netprofit_ttm': row['netprofit_ttm'],
                        'revenue_ttm': row['revenue_ttm'],
                        'totalassets_ttm': row['totalassets_ttm'],
                        'roaa': row['roaa'],
                        'net_margin': row['net_margin'],
                        'asset_turnover': row['asset_turnover']
                    })
                    
            except Exception as e:
                print(f"   Error testing Revenue={revenue_id}, TotalAssets={totalassets_id}: {e}")
        
        # Display results
        if results:
            results_df = pd.DataFrame(results)
            
            # Add reasonableness scores
            def score_ratios(row):
                score = 0
                
                # ROAA should be 0.001 to 0.50
                if 0.001 <= row['roaa'] <= 0.50:
                    score += 1
                
                # Net Margin should be 0.001 to 0.80
                if 0.001 <= row['net_margin'] <= 0.80:
                    score += 1
                
                # Asset Turnover should be 0.01 to 10.0
                if 0.01 <= row['asset_turnover'] <= 10.0:
                    score += 1
                
                # Sector-specific checks
                if 'bank' in sector.lower():
                    # Banks typically have lower asset turnover
                    if 0.01 <= row['asset_turnover'] <= 0.20:
                        score += 1
                    # Banks typically have higher net margins
                    if 0.10 <= row['net_margin'] <= 0.80:
                        score += 1
                else:
                    # Non-banks typically have higher asset turnover
                    if 0.50 <= row['asset_turnover'] <= 2.0:
                        score += 1
                    # Non-banks typically have lower net margins
                    if 0.05 <= row['net_margin'] <= 0.30:
                        score += 1
                
                return score
            
            results_df['score'] = results_df.apply(score_ratios, axis=1)
            results_df = results_df.sort_values('score', ascending=False)
            
            print(f"\n   Top Combinations (sorted by reasonableness score):")
            for i, row in results_df.head(5).iterrows():
                print(f"\n   #{i+1} - Score: {row['score']}/5")
                print(f"      Revenue Item: {row['revenue_id']}, TotalAssets Item: {row['totalassets_id']}")
                print(f"      NetProfit: {row['netprofit_ttm']:.0f} bn VND")
                print(f"      Revenue: {row['revenue_ttm']:.0f} bn VND")
                print(f"      TotalAssets: {row['totalassets_ttm']:.0f} bn VND")
                print(f"      ROAA: {row['roaa']:.4f} ({row['roaa']*100:.2f}%)")
                print(f"      Net Margin: {row['net_margin']:.4f} ({row['net_margin']*100:.2f}%)")
                print(f"      Asset Turnover: {row['asset_turnover']:.4f} ({row['asset_turnover']*100:.2f}%)")
        
        # 4. Show quarterly vs TTM comparison
        print(f"\n4️⃣ Quarterly vs TTM Comparison for {ticker}:")
        print("-" * 50)
        
        # Get latest quarterly data
        quarterly_query = text("""
            SELECT 
                fv.ticker,
                fv.year,
                fv.quarter,
                fv.item_id,
                fv.statement_type,
                fv.value
            FROM fundamental_values fv
            WHERE fv.ticker = :ticker
            AND fv.item_id IN (1, 2, 101, 302)
            AND (fv.year < :lag_year OR (fv.year = :lag_year AND fv.quarter <= :lag_quarter))
            ORDER BY fv.year DESC, fv.quarter DESC, fv.item_id
            LIMIT 20
        """)
        
        quarterly_data = pd.read_sql(quarterly_query, engine, params={
            'ticker': ticker,
            'lag_year': lag_year,
            'lag_quarter': lag_quarter
        })
        
        if not quarterly_data.empty:
            print("   Latest Quarterly Data:")
            for _, row in quarterly_data.iterrows():
                print(f"      {row['year']}Q{row['quarter']} - Item {row['item_id']} ({row['statement_type']}): {row['value']:.0f} bn VND")
        else:
            print("   No quarterly data found")

def show_expected_ratios():
    """Show expected ratios for different sectors."""
    print("\n📋 Expected Financial Ratios by Sector:")
    print("=" * 50)
    
    expected_ratios = {
        'Banks': {
            'ROAA': (0.005, 0.050),  # 0.5% to 5%
            'Net Margin': (0.10, 0.80),  # 10% to 80%
            'Asset Turnover': (0.01, 0.20)  # 1% to 20%
        },
        'Food & Beverage': {
            'ROAA': (0.05, 0.25),  # 5% to 25%
            'Net Margin': (0.05, 0.30),  # 5% to 30%
            'Asset Turnover': (0.80, 2.0)  # 80% to 200%
        },
        'Materials': {
            'ROAA': (0.05, 0.20),  # 5% to 20%
            'Net Margin': (0.05, 0.25),  # 5% to 25%
            'Asset Turnover': (0.50, 1.5)  # 50% to 150%
        }
    }
    
    for sector, ratios in expected_ratios.items():
        print(f"\n🏢 {sector}:")
        for metric, (min_val, max_val) in ratios.items():
            print(f"   {metric}: {min_val*100:.1f}% to {max_val*100:.1f}%")

if __name__ == "__main__":
    comprehensive_financial_analysis()
    show_expected_ratios() 