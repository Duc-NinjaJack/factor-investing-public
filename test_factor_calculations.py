import sys
sys.path.append('.')

import pandas as pd
from production.database.connection import get_engine
import importlib.util
spec = importlib.util.spec_from_file_location("module", "production/tests/phase29-alpha_demo/08_integrated_strategy_with_validated_factors_fixed.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
ValidatedFactorsCalculator = module.ValidatedFactorsCalculator

def test_factor_calculations():
    """Test factor calculations with correct database tables."""
    print("🧪 Testing factor calculations...")
    
    # Initialize database connection
    engine = get_engine()
    
    # Initialize calculator
    calculator = ValidatedFactorsCalculator(engine)
    
    # Test date
    test_date = pd.Timestamp('2016-02-01')
    test_tickers = ['VNM', 'VIC', 'HPG']
    
    print(f"\n📊 Testing Piotroski F-Score calculation for {test_tickers}...")
    try:
        fscore_result = calculator.calculate_piotroski_fscore(test_tickers, test_date)
        print(f"✅ F-Score calculation successful: {len(fscore_result)} results")
        if not fscore_result.empty:
            print(f"   Sample results:")
            print(f"   {fscore_result.head()}")
    except Exception as e:
        print(f"❌ F-Score calculation failed: {e}")
    
    print(f"\n📊 Testing FCF Yield calculation for {test_tickers}...")
    try:
        fcf_result = calculator.calculate_fcf_yield(test_tickers, test_date)
        print(f"✅ FCF Yield calculation successful: {len(fcf_result)} results")
        if not fcf_result.empty:
            print(f"   Sample results:")
            print(f"   {fcf_result.head()}")
    except Exception as e:
        print(f"❌ FCF Yield calculation failed: {e}")
    
    print("\n✅ Factor calculation test completed!")

if __name__ == "__main__":
    test_factor_calculations() 