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

def debug_factor_calculations():
    """Debug why factor calculations are failing."""
    print("🔍 Debugging Factor Calculation Issues...")
    
    # Initialize database connection
    engine = get_engine()
    
    # Test date
    test_date = pd.Timestamp('2016-02-01')
    test_tickers = ['VNM', 'VIC', 'HPG', 'TCB', 'VCB']
    
    print(f"\n📅 Testing factor calculations for {test_date}")
    print(f"📊 Test tickers: {test_tickers}")
    
    # Initialize calculator
    calculator = module.ValidatedFactorsCalculator(engine)
    
    # Test 1: F-Score calculation
    print(f"\n🧪 Test 1: Piotroski F-Score")
    try:
        fscore_result = calculator.calculate_piotroski_fscore(test_tickers, test_date)
        print(f"   ✅ F-Score calculation successful")
        print(f"   📊 Results: {len(fscore_result)} stocks")
        if not fscore_result.empty:
            print(f"   📈 Sample F-Scores:")
            for _, row in fscore_result.head().iterrows():
                print(f"      {row['ticker']}: {row['fscore']}")
        else:
            print(f"   ❌ No F-Score results returned")
    except Exception as e:
        print(f"   ❌ F-Score calculation failed: {e}")
    
    # Test 2: FCF Yield calculation
    print(f"\n🧪 Test 2: FCF Yield")
    try:
        fcf_result = calculator.calculate_fcf_yield(test_tickers, test_date)
        print(f"   ✅ FCF Yield calculation successful")
        print(f"   📊 Results: {len(fcf_result)} stocks")
        if not fcf_result.empty:
            print(f"   📈 Sample FCF Yields:")
            for _, row in fcf_result.head().iterrows():
                print(f"      {row['ticker']}: {row['fcf_yield']:.4f}")
        else:
            print(f"   ❌ No FCF Yield results returned")
    except Exception as e:
        print(f"   ❌ FCF Yield calculation failed: {e}")
    
    # Test 3: Low-Volatility factor
    print(f"\n🧪 Test 3: Low-Volatility Factor")
    try:
        # Get price data for the test tickers
        price_query = f"""
        SELECT 
            trading_date,
            ticker,
            close_price
        FROM vcsc_daily_data
        WHERE ticker IN ({','.join([f"'{t}'" for t in test_tickers])})
        AND trading_date BETWEEN '{test_date - pd.Timedelta(days=300)}' AND '{test_date}'
        ORDER BY trading_date
        """
        price_data = pd.read_sql(price_query, engine)
        
        if not price_data.empty:
            low_vol_result = calculator.calculate_low_volatility_factor(price_data, 252)
            print(f"   ✅ Low-Volatility calculation successful")
            print(f"   📊 Results: {len(low_vol_result)} observations")
            if not low_vol_result.empty:
                latest_low_vol = low_vol_result.groupby('ticker').tail(1)
                print(f"   📈 Latest Low-Vol Scores:")
                for _, row in latest_low_vol.iterrows():
                    print(f"      {row['ticker']}: {row['low_vol_score']:.4f}")
            else:
                print(f"   ❌ No Low-Vol results returned")
        else:
            print(f"   ❌ No price data available")
    except Exception as e:
        print(f"   ❌ Low-Volatility calculation failed: {e}")
    
    # Test 4: Check what's in the precomputed data
    print(f"\n🧪 Test 4: Precomputed Data Structure")
    try:
        # Get a sample of precomputed data
        config = {
            'backtest_start_date': '2016-01-01',
            'backtest_end_date': '2016-12-31',
            'universe': {'target_size': 200},
            'factors': {
                'fundamental_lag_days': 30,
                'volatility_lookback': 252,
                'momentum_horizons': [21, 63, 126, 252],
                'skip_months': 1
            }
        }
        
        precomputed_data = module.precompute_all_data(config, engine)
        
        print(f"   📊 Precomputed data keys: {list(precomputed_data.keys())}")
        
        for key, data in precomputed_data.items():
            print(f"   📈 {key}: {len(data)} rows")
            if not data.empty:
                print(f"      Columns: {list(data.columns)}")
                print(f"      Sample dates: {data.iloc[:3]['trading_date' if 'trading_date' in data.columns else 'date'].tolist()}")
    except Exception as e:
        print(f"   ❌ Precomputed data check failed: {e}")

if __name__ == "__main__":
    debug_factor_calculations() 