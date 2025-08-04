import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from production.database.connection import get_engine
import importlib.util

# Import the module
spec = importlib.util.spec_from_file_location("module", "production/tests/phase29-alpha_demo/08_integrated_strategy_with_validated_factors_fixed.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def debug_portfolio_construction():
    """Debug why portfolios aren't being constructed early in the backtest."""
    print("🔍 Debugging Portfolio Construction Issues...")
    
    # Initialize database connection
    engine = get_engine()
    
    # Test dates - early, middle, and late in the backtest period
    test_dates = [
        pd.Timestamp('2016-02-01'),  # Early
        pd.Timestamp('2019-06-01'),  # Middle  
        pd.Timestamp('2024-08-01')   # Late (where it works)
    ]
    
    for test_date in test_dates:
        print(f"\n📅 Testing date: {test_date}")
        
        # Get universe for this date
        universe_query = f"""
        WITH daily_adtv AS (
            SELECT 
                trading_date,
                ticker,
                total_volume * close_price as adtv_vnd
            FROM vcsc_daily_data
            WHERE trading_date BETWEEN '{test_date - pd.Timedelta(days=100)}' AND '{test_date}'
        ),
        rolling_adtv AS (
            SELECT 
                trading_date,
                ticker,
                AVG(adtv_vnd) OVER (
                    PARTITION BY ticker 
                    ORDER BY trading_date 
                    ROWS BETWEEN 62 PRECEDING AND CURRENT ROW
                ) as avg_adtv_63d
            FROM daily_adtv
            WHERE adtv_vnd > 0
        ),
        ranked_universe AS (
            SELECT 
                trading_date,
                ticker,
                ROW_NUMBER() OVER (
                    PARTITION BY trading_date 
                    ORDER BY avg_adtv_63d DESC
                ) as rank_position
            FROM rolling_adtv
            WHERE avg_adtv_63d > 0
        )
        SELECT ticker
        FROM ranked_universe
        WHERE trading_date = '{test_date}'
        AND rank_position <= 200
        ORDER BY rank_position
        """
        
        universe_data = pd.read_sql(universe_query, engine)
        universe = universe_data['ticker'].tolist()
        
        print(f"   📊 Universe size: {len(universe)} stocks")
        
        if len(universe) == 0:
            print("   ❌ No universe found!")
            continue
        
        # Test factor calculations
        calculator = module.ValidatedFactorsCalculator(engine)
        
        # Test F-Score
        try:
            fscore_result = calculator.calculate_piotroski_fscore(universe[:10], test_date)
            print(f"   ✅ F-Score: {len(fscore_result)} results")
        except Exception as e:
            print(f"   ❌ F-Score error: {e}")
        
        # Test FCF Yield
        try:
            fcf_result = calculator.calculate_fcf_yield(universe[:10], test_date)
            print(f"   ✅ FCF Yield: {len(fcf_result)} results")
        except Exception as e:
            print(f"   ❌ FCF Yield error: {e}")
        
        # Test P/E data availability
        pe_query = f"""
        SELECT COUNT(*) as count
        FROM financial_metrics 
        WHERE Date = '{test_date.date()}'
        AND PE IS NOT NULL AND PE > 0
        """
        pe_count = pd.read_sql(pe_query, engine).iloc[0]['count']
        print(f"   📊 P/E data available: {pe_count} stocks")
        
        # Test momentum data availability
        momentum_query = f"""
        SELECT COUNT(DISTINCT ticker) as count
        FROM vcsc_daily_data
        WHERE trading_date BETWEEN '{test_date - pd.Timedelta(days=300)}' AND '{test_date}'
        AND ticker IN ({','.join([f"'{t}'" for t in universe[:10]])})
        """
        momentum_count = pd.read_sql(momentum_query, engine).iloc[0]['count']
        print(f"   📊 Momentum data available: {momentum_count} stocks")

if __name__ == "__main__":
    debug_portfolio_construction() 