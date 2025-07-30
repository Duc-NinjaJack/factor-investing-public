#!/usr/bin/env python3
"""
Test script for Phase 22 Weighted Composite Backtesting

This script provides a simple way to test the weighted composite backtesting
implementation without running the full analysis.
"""

import sys
import logging
from pathlib import Path
import importlib.util
import pandas as pd

# Add paths for imports
sys.path.append('../../../production/database')
sys.path.append('../../../production/scripts')

try:
    from connection import get_database_manager
    from real_data_backtesting import RealDataBacktesting
    
    # Dynamic import for module with number prefix
    spec = importlib.util.spec_from_file_location(
        "weighted_composite_backtest",
        "22_weighted_composite_real_data_backtest.py"
    )
    weighted_composite_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(weighted_composite_module)
    WeightedCompositeBacktesting = weighted_composite_module.WeightedCompositeBacktesting
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all dependencies are installed and paths are correct.")
    sys.exit(1)


def test_database_connection():
    """Test database connection."""
    print("🔍 Testing database connection...")
    try:
        db_manager = get_database_manager()
        engine = db_manager.get_engine()
        
        # Test simple query using pandas
        result = pd.read_sql("SELECT 1 as test", engine)
        if not result.empty and result.iloc[0]['test'] == 1:
            print("✅ Database connection successful")
            return True
        else:
            print("❌ Database connection test failed")
            return False
    except Exception as e:
        print(f"❌ Database connection error: {e}")
        return False


def test_factor_data_availability():
    """Test if factor data is available."""
    print("🔍 Testing factor data availability...")
    try:
        db_manager = get_database_manager()
        engine = db_manager.get_engine()
        
        # Check if factor_scores_qvm table exists and has data
        query = """
        SELECT COUNT(*) as count, 
               MIN(date) as min_date, 
               MAX(date) as max_date,
               COUNT(DISTINCT ticker) as unique_tickers
        FROM factor_scores_qvm
        WHERE date >= '2018-01-01'
        """
        
        result = pd.read_sql(query, engine)
        
        if not result.empty and result.iloc[0]['count'] > 0:
            row = result.iloc[0]
            print(f"✅ Factor data available:")
            print(f"   - Records: {row['count']:,}")
            print(f"   - Date range: {row['min_date']} to {row['max_date']}")
            print(f"   - Unique tickers: {row['unique_tickers']}")
            return True
        else:
            print("❌ No factor data found")
            return False
    except Exception as e:
        print(f"❌ Factor data test error: {e}")
        return False


def test_price_data_availability():
    """Test if price data is available."""
    print("🔍 Testing price data availability...")
    try:
        db_manager = get_database_manager()
        engine = db_manager.get_engine()
        
        # Check if vcsc_daily_data_complete table exists and has data
        query = """
        SELECT COUNT(*) as count, 
               MIN(trading_date) as min_date, 
               MAX(trading_date) as max_date,
               COUNT(DISTINCT ticker) as unique_tickers
        FROM vcsc_daily_data_complete
        WHERE trading_date >= '2018-01-01'
        """
        
        result = pd.read_sql(query, engine)
        
        if not result.empty and result.iloc[0]['count'] > 0:
            row = result.iloc[0]
            print(f"✅ Price data available:")
            print(f"   - Records: {row['count']:,}")
            print(f"   - Date range: {row['min_date']} to {row['max_date']}")
            print(f"   - Unique tickers: {row['unique_tickers']}")
            return True
        else:
            print("❌ No price data found")
            return False
    except Exception as e:
        print(f"❌ Price data test error: {e}")
        return False


def test_benchmark_data_availability():
    """Test if benchmark data is available."""
    print("🔍 Testing benchmark data availability...")
    try:
        db_manager = get_database_manager()
        engine = db_manager.get_engine()
        
        # Check if etf_history table exists and has VNINDEX data
        query = """
        SELECT COUNT(*) as count, 
               MIN(date) as min_date, 
               MAX(date) as max_date
        FROM etf_history
        WHERE ticker = 'VNINDEX' AND date >= '2018-01-01'
        """
        
        result = pd.read_sql(query, engine)
        
        if not result.empty and result.iloc[0]['count'] > 0:
            row = result.iloc[0]
            print(f"✅ Benchmark data available:")
            print(f"   - Records: {row['count']:,}")
            print(f"   - Date range: {row['min_date']} to {row['max_date']}")
            return True
        else:
            print("❌ No benchmark data found")
            return False
    except Exception as e:
        print(f"❌ Benchmark data test error: {e}")
        return False


def test_adtv_data_availability():
    """Test if ADTV data is available."""
    print("🔍 Testing ADTV data availability...")
    try:
        # Check if pickle file exists
        pickle_paths = [
            'unrestricted_universe_data.pkl',
            '../../../data/unrestricted_universe_data.pkl',
            '../../data/unrestricted_universe_data.pkl'
        ]
        
        for path in pickle_paths:
            if Path(path).exists():
                print(f"✅ ADTV data found at: {path}")
                return True
        
        print("❌ ADTV pickle file not found")
        print("   Please ensure unrestricted_universe_data.pkl is available")
        return False
    except Exception as e:
        print(f"❌ ADTV data test error: {e}")
        return False


def test_weighted_composite_initialization():
    """Test weighted composite backtesting initialization."""
    print("🔍 Testing weighted composite initialization...")
    try:
        # Initialize with minimal configuration
        backtesting = WeightedCompositeBacktesting()
        
        # Check configuration
        assert backtesting.weighting_scheme['Value'] == 0.6
        assert backtesting.weighting_scheme['Quality'] == 0.2
        assert backtesting.weighting_scheme['Reversal'] == 0.2
        
        print("✅ Weighted composite initialization successful")
        print(f"   - Weighting scheme: {backtesting.weighting_scheme}")
        print(f"   - Portfolio size: {backtesting.strategy_config['portfolio_size']}")
        return True
    except Exception as e:
        print(f"❌ Weighted composite initialization error: {e}")
        return False


def test_factor_data_loading():
    """Test factor data loading functionality."""
    print("🔍 Testing factor data loading...")
    try:
        backtesting = WeightedCompositeBacktesting()
        
        # Test factor data loading
        data = backtesting.load_factor_data()
        
        if 'factor_data' in data and 'adtv_data' in data:
            factor_data = data['factor_data']
            print(f"✅ Factor data loaded successfully:")
            print(f"   - Factor records: {len(factor_data):,}")
            print(f"   - Date range: {factor_data['date'].min()} to {factor_data['date'].max()}")
            print(f"   - Unique tickers: {factor_data['ticker'].nunique()}")
            print(f"   - ADTV data shape: {data['adtv_data'].shape}")
            return True
        else:
            print("❌ Factor data loading failed")
            return False
    except Exception as e:
        print(f"❌ Factor data loading error: {e}")
        return False


def run_comprehensive_test():
    """Run all tests."""
    print("=" * 60)
    print("🧪 PHASE 22 WEIGHTED COMPOSITE BACKTESTING - TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Database Connection", test_database_connection),
        ("Factor Data Availability", test_factor_data_availability),
        ("Price Data Availability", test_price_data_availability),
        ("Benchmark Data Availability", test_benchmark_data_availability),
        ("ADTV Data Availability", test_adtv_data_availability),
        ("Weighted Composite Initialization", test_weighted_composite_initialization),
        ("Factor Data Loading", test_factor_data_loading),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The implementation is ready to use.")
        print("\nNext steps:")
        print("1. Run the full backtest: python 22_weighted_composite_real_data_backtest.py")
        print("2. Review the generated plots and reports")
        print("3. Analyze the performance results")
    else:
        print("⚠️ Some tests failed. Please address the issues before running the full backtest.")
        print("\nCommon fixes:")
        print("1. Check database configuration in config/database.yml")
        print("2. Ensure all required data tables exist")
        print("3. Verify ADTV pickle file is available")
        print("4. Check Python dependencies are installed")
    
    return passed == total


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    success = run_comprehensive_test()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)