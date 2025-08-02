# ============================================================================
# Phase 28: Baseline Integration Test
# File: test_baseline_integration.py
#
# Objective:
#   To test the baseline integration module and verify that it can be
#   imported and used correctly within the Phase 28 framework.
# ============================================================================

import sys
from pathlib import Path
import pandas as pd
import numpy as np

def test_baseline_import():
    """Test that the baseline comparison module can be imported."""
    try:
        from baseline_comparison import (
            BaselinePortfolioEngine, 
            BaselineComparisonFramework, 
            DEFAULT_BASELINE_CONFIG
        )
        print("✅ Successfully imported baseline comparison module")
        return True
    except ImportError as e:
        print(f"❌ Failed to import baseline comparison module: {e}")
        return False

def test_baseline_config():
    """Test that the default baseline configuration is valid."""
    try:
        from baseline_comparison import DEFAULT_BASELINE_CONFIG
        
        required_keys = [
            'strategy_name', 'backtest_start_date', 'backtest_end_date',
            'rebalance_frequency', 'transaction_cost_bps', 'universe',
            'signal', 'portfolio'
        ]
        
        for key in required_keys:
            if key not in DEFAULT_BASELINE_CONFIG:
                print(f"❌ Missing required key in baseline config: {key}")
                return False
        
        print("✅ Baseline configuration is valid")
        return True
    except Exception as e:
        print(f"❌ Failed to validate baseline configuration: {e}")
        return False

def test_baseline_engine_creation():
    """Test that the baseline engine can be instantiated."""
    try:
        from baseline_comparison import BaselinePortfolioEngine, DEFAULT_BASELINE_CONFIG
        
        # Create mock data for testing
        mock_factor_data = pd.DataFrame({
            'date': pd.date_range('2020-01-01', '2020-12-31', freq='D'),
            'ticker': ['AAA'] * 366,
            'Value_Composite': np.random.randn(366)
        })
        
        mock_returns_matrix = pd.DataFrame(
            np.random.randn(366, 10) * 0.01,
            index=pd.date_range('2020-01-01', '2020-12-31', freq='D'),
            columns=[f'STOCK_{i}' for i in range(10)]
        )
        
        mock_benchmark_returns = pd.Series(
            np.random.randn(366) * 0.01,
            index=pd.date_range('2020-01-01', '2020-12-31', freq='D'),
            name='VN-Index'
        )
        
        # Create mock database engine (None for testing)
        mock_db_engine = None
        
        # Test engine creation
        engine = BaselinePortfolioEngine(
            config=DEFAULT_BASELINE_CONFIG,
            factor_data=mock_factor_data,
            returns_matrix=mock_returns_matrix,
            benchmark_returns=mock_benchmark_returns,
            db_engine=mock_db_engine
        )
        
        print("✅ Baseline engine can be instantiated")
        return True
    except Exception as e:
        print(f"❌ Failed to create baseline engine: {e}")
        return False

def test_comparison_framework_creation():
    """Test that the comparison framework can be instantiated."""
    try:
        from baseline_comparison import BaselineComparisonFramework, DEFAULT_BASELINE_CONFIG
        
        # Create mock configurations
        baseline_config = DEFAULT_BASELINE_CONFIG.copy()
        enhanced_config = {
            'strategy_name': 'Test_Enhanced',
            'backtest_start_date': '2020-01-01',
            'backtest_end_date': '2020-12-31'
        }
        
        # Create mock database engine (None for testing)
        mock_db_engine = None
        
        # Test framework creation
        framework = BaselineComparisonFramework(
            baseline_config=baseline_config,
            enhanced_config=enhanced_config,
            db_engine=mock_db_engine
        )
        
        print("✅ Comparison framework can be instantiated")
        return True
    except Exception as e:
        print(f"❌ Failed to create comparison framework: {e}")
        return False

def test_baseline_methods():
    """Test that baseline engine methods exist and are callable."""
    try:
        from baseline_comparison import BaselinePortfolioEngine, DEFAULT_BASELINE_CONFIG
        
        # Create mock data
        mock_factor_data = pd.DataFrame({
            'date': pd.date_range('2020-01-01', '2020-12-31', freq='D'),
            'ticker': ['AAA'] * 366,
            'Value_Composite': np.random.randn(366)
        })
        
        mock_returns_matrix = pd.DataFrame(
            np.random.randn(366, 10) * 0.01,
            index=pd.date_range('2020-01-01', '2020-12-31', freq='D'),
            columns=[f'STOCK_{i}' for i in range(10)]
        )
        
        mock_benchmark_returns = pd.Series(
            np.random.randn(366) * 0.01,
            index=pd.date_range('2020-01-01', '2020-12-31', freq='D'),
            name='VN-Index'
        )
        
        mock_db_engine = None
        
        engine = BaselinePortfolioEngine(
            config=DEFAULT_BASELINE_CONFIG,
            factor_data=mock_factor_data,
            returns_matrix=mock_returns_matrix,
            benchmark_returns=mock_benchmark_returns,
            db_engine=mock_db_engine
        )
        
        # Test that required methods exist
        required_methods = [
            'run_backtest',
            '_generate_rebalance_dates',
            '_run_backtesting_loop',
            '_calculate_target_portfolio',
            '_calculate_net_returns'
        ]
        
        for method_name in required_methods:
            if not hasattr(engine, method_name):
                print(f"❌ Missing required method: {method_name}")
                return False
            if not callable(getattr(engine, method_name)):
                print(f"❌ Method is not callable: {method_name}")
                return False
        
        print("✅ All required baseline methods exist and are callable")
        return True
    except Exception as e:
        print(f"❌ Failed to test baseline methods: {e}")
        return False

def run_all_tests():
    """Run all baseline integration tests."""
    print("="*80)
    print("🧪 PHASE 28 BASELINE INTEGRATION TESTS")
    print("="*80)
    
    tests = [
        ("Baseline Import", test_baseline_import),
        ("Baseline Config", test_baseline_config),
        ("Engine Creation", test_baseline_engine_creation),
        ("Framework Creation", test_comparison_framework_creation),
        ("Baseline Methods", test_baseline_methods)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing: {test_name}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print(f"\n" + "="*80)
    print(f"📊 TEST RESULTS: {passed}/{total} tests passed")
    print("="*80)
    
    if passed == total:
        print("🎉 All baseline integration tests passed!")
        print("✅ The baseline integration is working correctly.")
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 