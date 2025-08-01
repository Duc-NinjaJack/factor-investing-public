#!/usr/bin/env python3
"""
Test Corrected Fundamental Data Query
====================================

This script tests the corrected fundamental data query to ensure it returns
reasonable values for ROAA, Net Margin, and Asset Turnover.
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

def test_corrected_fundamental_data():
    """Test the corrected fundamental data query."""
    print("🔍 Testing Corrected Fundamental Data Query")
    print("=" * 60)
    print("   NetProfit: Item 1 (PL)")
    print("   Revenue: Item 2 (PL)")
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
    
    # Sample universe
    test_universe = ['TCB', 'VCB', 'BID', 'CTG', 'MBB', 'ACB', 'TPB', 'STB', 'VPB', 'SHB']
    ticker_list = "','".join(test_universe)
    
    print(f"📅 Test Date: {test_date.date()}")
    print(f"📅 Lag Date: {lag_date.date()} (Year: {lag_year}, Quarter: {lag_quarter})")
    print(f"📊 Test Universe: {len(test_universe)} stocks")
    
    # Corrected query
    fundamental_query = text(f"""
        WITH quarterly_data AS (
            SELECT 
                fv.ticker,
                fv.year,
                fv.quarter,
                fv.value,
                fv.item_id,
                fv.statement_type
            FROM fundamental_values fv
            WHERE (fv.item_id = 1 AND fv.statement_type = 'PL')
               OR (fv.item_id = 2 AND fv.statement_type = 'PL')
               OR (fv.item_id = 2 AND fv.statement_type = 'BS')
            AND fv.ticker IN ('{ticker_list}')
            AND (fv.year < :lag_year OR (fv.year = :lag_year AND fv.quarter <= :lag_quarter))
        ),
        ttm_calculations AS (
            SELECT 
                ticker,
                year,
                quarter,
                SUM(CASE WHEN item_id = 1 AND statement_type = 'PL' THEN value ELSE 0 END) as netprofit_ttm,
                SUM(CASE WHEN item_id = 2 AND statement_type = 'PL' THEN value ELSE 0 END) as revenue_ttm,
                SUM(CASE WHEN item_id = 2 AND statement_type = 'BS' THEN value ELSE 0 END) as totalassets_ttm
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
            WHERE rn <= 4  -- Last 4 quarters for TTM
            GROUP BY ticker, year, quarter
        ),
        latest_data AS (
            SELECT 
                ttm.ticker,
                mi.sector,
                ttm.year,
                ttm.quarter,
                ttm.netprofit_ttm,
                ttm.revenue_ttm,
                ttm.totalassets_ttm,
                CASE 
                    WHEN ttm.totalassets_ttm > 0 THEN ttm.netprofit_ttm / ttm.totalassets_ttm 
                    ELSE NULL 
                END as roaa,
                CASE 
                    WHEN ttm.revenue_ttm > 0 THEN ttm.netprofit_ttm / ttm.revenue_ttm
                    ELSE NULL 
                END as net_margin,
                CASE 
                    WHEN ttm.totalassets_ttm > 0 THEN ttm.revenue_ttm / ttm.totalassets_ttm 
                    ELSE NULL 
                END as asset_turnover,
                ROW_NUMBER() OVER (PARTITION BY ttm.ticker ORDER BY ttm.year DESC, ttm.quarter DESC) as rn
            FROM ttm_calculations ttm
            LEFT JOIN master_info mi ON ttm.ticker = mi.ticker
            WHERE ttm.netprofit_ttm > 0 AND ttm.revenue_ttm > 0 AND ttm.totalassets_ttm > 0
            AND (ttm.year < :lag_year OR (ttm.year = :lag_year AND ttm.quarter <= :lag_quarter))
        )
        SELECT 
            ticker,
            sector,
            netprofit_ttm,
            revenue_ttm,
            totalassets_ttm,
            roaa,
            net_margin,
            asset_turnover
        FROM latest_data
        WHERE rn = 1  -- Get most recent data for each ticker
    """)
    
    # Execute query
    params_dict = {
        'lag_year': lag_year,
        'lag_quarter': lag_quarter
    }
    
    try:
        fundamental_df = pd.read_sql(fundamental_query, engine, params=params_dict)
        
        print(f"\n✅ Query executed successfully!")
        print(f"📊 Results:")
        print(f"   - Total rows returned: {len(fundamental_df)}")
        print(f"   - Unique tickers: {fundamental_df['ticker'].nunique()}")
        print(f"   - Tickers with data: {list(fundamental_df['ticker'].unique())}")
        
        if not fundamental_df.empty:
            print(f"\n📈 Sample Data:")
            print(fundamental_df[['ticker', 'sector', 'roaa', 'net_margin', 'asset_turnover']].head().to_string(index=False))
            
            print(f"\n📊 Data Quality Analysis:")
            print(f"   - ROAA range: {fundamental_df['roaa'].min():.4f} to {fundamental_df['roaa'].max():.4f}")
            print(f"   - Net Margin range: {fundamental_df['net_margin'].min():.4f} to {fundamental_df['net_margin'].max():.4f}")
            print(f"   - Asset Turnover range: {fundamental_df['asset_turnover'].min():.4f} to {fundamental_df['asset_turnover'].max():.4f}")
            
            # Check if values look reasonable
            roaa_reasonable = all(0.001 <= roaa <= 0.50 for roaa in fundamental_df['roaa'])  # 0.1% to 50%
            margin_reasonable = all(0.001 <= margin <= 0.80 for margin in fundamental_df['net_margin'])  # 0.1% to 80%
            turnover_reasonable = all(0.01 <= turnover <= 10.0 for turnover in fundamental_df['asset_turnover'])  # 0.01 to 10.0
            
            print(f"\n🎯 Reasonableness Check:")
            print(f"   - ROAA values reasonable: {roaa_reasonable}")
            print(f"   - Net Margin values reasonable: {margin_reasonable}")
            print(f"   - Asset Turnover values reasonable: {turnover_reasonable}")
            
            if roaa_reasonable and margin_reasonable and turnover_reasonable:
                print(f"   ✅ ALL VALUES LOOK REASONABLE!")
            else:
                print(f"   ⚠️ Some values may still be incorrect")
            
            # Show raw values for verification
            print(f"\n🔍 Raw Values (first 3 stocks):")
            for i, row in fundamental_df.head(3).iterrows():
                print(f"   {row['ticker']}: NetProfit={row['netprofit_ttm']:.0f}, Revenue={row['revenue_ttm']:.0f}, TotalAssets={row['totalassets_ttm']:.0f}")
        
        return len(fundamental_df) > 1 and roaa_reasonable and margin_reasonable and turnover_reasonable
        
    except Exception as e:
        print(f"❌ Query failed: {e}")
        return False

if __name__ == "__main__":
    success = test_corrected_fundamental_data()
    if success:
        print(f"\n✅ CORRECTION VERIFIED: Fundamental data now has reasonable values!")
    else:
        print(f"\n❌ CORRECTION FAILED: Fundamental data still has issues!") 