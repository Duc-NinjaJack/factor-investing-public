#!/usr/bin/env python3
"""
Simple Strategy Test
===================

Simplified version of the strategy to test basic factor calculation.
"""

import sys
import pandas as pd
from pathlib import Path
from sqlalchemy import text

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

def simple_factor_calculation(universe, analysis_date, engine):
    """Simplified factor calculation that works."""
    try:
        # Get fundamental data with proper lagging (45 days)
        lag_days = 45
        lag_date = analysis_date - pd.Timedelta(days=lag_days)
        lag_year = lag_date.year
        
        # Build ticker list for IN clause with proper quoting
        ticker_list = "','".join(universe)
        
        # Simplified query that works
        fundamental_query = text(f"""
            WITH netprofit_ttm AS (
                SELECT 
                    fv.ticker,
                    SUM(fv.value / 6222444702.01) as netprofit_ttm
                FROM fundamental_values fv
                WHERE fv.ticker IN ('{ticker_list}')
                AND fv.item_id = 1
                AND fv.statement_type = 'PL'
                AND fv.year <= {lag_year}
                AND fv.year >= {lag_year - 1}  -- Last 4 quarters
                GROUP BY fv.ticker
            ),
            totalassets_ttm AS (
                SELECT 
                    fv.ticker,
                    SUM(fv.value / 6222444702.01) as totalassets_ttm
                FROM fundamental_values fv
                WHERE fv.ticker IN ('{ticker_list}')
                AND fv.item_id = 2
                AND fv.statement_type = 'BS'
                AND fv.year <= {lag_year}
                AND fv.year >= {lag_year - 1}  -- Last 4 quarters
                GROUP BY fv.ticker
            ),
            revenue_ttm AS (
                SELECT 
                    fv.ticker,
                    SUM(fv.value / 6222444702.01) as revenue_ttm
                FROM fundamental_values fv
                WHERE fv.ticker IN ('{ticker_list}')
                AND fv.item_id = 2
                AND fv.statement_type = 'PL'
                AND fv.year <= {lag_year}
                AND fv.year >= {lag_year - 1}  -- Last 4 quarters
                GROUP BY fv.ticker
            )
            SELECT 
                np.ticker,
                np.netprofit_ttm,
                ta.totalassets_ttm,
                rv.revenue_ttm,
                CASE 
                    WHEN ta.totalassets_ttm > 0 THEN np.netprofit_ttm / ta.totalassets_ttm 
                    ELSE NULL 
                END as roaa,
                CASE 
                    WHEN rv.revenue_ttm > 0 THEN np.netprofit_ttm / rv.revenue_ttm
                    ELSE NULL 
                END as net_margin,
                CASE 
                    WHEN ta.totalassets_ttm > 0 THEN rv.revenue_ttm / ta.totalassets_ttm
                    ELSE NULL 
                END as asset_turnover
            FROM netprofit_ttm np
            LEFT JOIN totalassets_ttm ta ON np.ticker = ta.ticker
            LEFT JOIN revenue_ttm rv ON np.ticker = rv.ticker
            WHERE np.netprofit_ttm > 0 
            AND ta.totalassets_ttm > 0
            AND rv.revenue_ttm > 0
        """)
        
        fundamental_df = pd.read_sql(fundamental_query, engine)
        
        print(f"   ✅ Simple factor calculation: {len(fundamental_df)} records")
        if not fundamental_df.empty:
            print(f"   Sample results:")
            for _, row in fundamental_df.head(3).iterrows():
                print(f"     {row['ticker']}: ROAA = {row['roaa']:.4f}, Net Margin = {row['net_margin']:.4f}")
        
        return fundamental_df
        
    except Exception as e:
        print(f"   ❌ Error in simple factor calculation: {e}")
        return pd.DataFrame()

def test_simple_strategy():
    """Test the simplified strategy."""
    print("🧪 Testing Simple Strategy")
    print("=" * 40)
    
    # Connect to database
    db_manager = get_database_manager()
    engine = db_manager.get_engine()
    
    # Test date
    test_date = pd.Timestamp('2021-01-29')
    
    # Get universe
    universe_query = text("""
        SELECT 
            ticker,
            AVG(total_volume) as avg_volume,
            AVG(market_cap) as avg_market_cap
        FROM vcsc_daily_data_complete
        WHERE trading_date <= :analysis_date
          AND trading_date >= DATE_SUB(:analysis_date, INTERVAL 63 DAY)
        GROUP BY ticker
        HAVING avg_volume >= 1000000 AND avg_market_cap >= 100000000000
        ORDER BY avg_volume DESC
        LIMIT 10
    """)
    
    universe_df = pd.read_sql(universe_query, engine, 
                             params={'analysis_date': test_date})
    
    print(f"   Universe size: {len(universe_df)} stocks")
    if universe_df.empty:
        print("   ❌ No stocks in universe!")
        return
    
    universe = universe_df['ticker'].tolist()
    print(f"   Sample tickers: {universe[:5]}")
    
    # Calculate factors
    factors_df = simple_factor_calculation(universe, test_date, engine)
    
    if not factors_df.empty:
        print(f"\n✅ Strategy test successful!")
        print(f"   - Found {len(factors_df)} stocks with factor data")
        print(f"   - Average ROAA: {factors_df['roaa'].mean():.4f}")
        print(f"   - Average Net Margin: {factors_df['net_margin'].mean():.4f}")
    else:
        print(f"\n❌ Strategy test failed - no factor data")

if __name__ == "__main__":
    test_simple_strategy() 