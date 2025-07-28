#!/usr/bin/env python3
"""
Quick test script to validate component fix
"""
import sys
import os
import pandas as pd
from sqlalchemy import create_engine

# Change to project root directory
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
os.chdir(project_root)

# Add production engine to path
sys.path.insert(0, os.path.join(project_root, 'production', 'engine'))

from qvm_engine_v2_enhanced import QVMEngineV2Enhanced

def test_component_structure():
    """Test that engine returns proper component structure"""
    try:
        # Initialize engine
        print("🔧 Testing Enhanced Engine v2 component structure...")
        engine = QVMEngineV2Enhanced()
        
        # Test with small universe
        test_date = pd.Timestamp('2024-07-01')
        test_universe = ['FPT', 'VIC', 'VHM', 'TCB', 'VCB']  # 5 tickers only
        
        print(f"📊 Testing {len(test_universe)} tickers on {test_date.date()}")
        
        # Calculate scores
        results = engine.calculate_qvm_composite(test_date, test_universe)
        
        if not results:
            print("❌ No results returned")
            return False
            
        print(f"✅ Results returned for {len(results)} tickers")
        
        # Check structure
        sample_ticker = list(results.keys())[0]
        sample_result = results[sample_ticker]
        
        print(f"🔍 Sample result structure for {sample_ticker}:")
        print(f"   Type: {type(sample_result)}")
        print(f"   Keys: {list(sample_result.keys()) if isinstance(sample_result, dict) else 'Not a dict'}")
        
        # Validate expected structure
        if isinstance(sample_result, dict):
            required_keys = ['Quality_Composite', 'Value_Composite', 'Momentum_Composite', 'QVM_Composite']
            has_all_keys = all(key in sample_result for key in required_keys)
            
            if has_all_keys:
                print("✅ Component structure validation PASSED")
                for ticker, components in list(results.items())[:3]:  # Show first 3
                    print(f"   {ticker}: Q={components['Quality_Composite']:.4f}, V={components['Value_Composite']:.4f}, M={components['Momentum_Composite']:.4f}, QVM={components['QVM_Composite']:.4f}")
                return True
            else:
                print(f"❌ Missing required keys. Expected: {required_keys}")
                print(f"   Found: {list(sample_result.keys())}")
                return False
        else:
            print("❌ Result is not a dictionary - old format detected")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_component_structure()
    sys.exit(0 if success else 1)