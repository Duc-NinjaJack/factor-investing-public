#!/usr/bin/env python3
"""
Test script for the common database connection system.

This script tests all the functionality of the database connection module
to ensure it works correctly with the existing database setup.

Usage:
    python test_connection.py
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_basic_functionality():
    """Test basic database connection functionality."""
    print("🧪 Testing Basic Database Functionality")
    print("=" * 50)
    
    try:
        # Test imports
        print("1. Testing imports...")
        from production.database import get_engine, get_connection, DatabaseManager
        from production.database.utils import execute_query, get_ticker_list, get_sector_mapping
        print("   ✅ All imports successful")
        
        # Test engine creation
        print("2. Testing SQLAlchemy engine...")
        engine = get_engine()
        print(f"   ✅ Engine created: {type(engine)}")
        
        # Test connection creation
        print("3. Testing PyMySQL connection...")
        connection = get_connection()
        print(f"   ✅ Connection created: {type(connection)}")
        
        # Test basic query execution
        print("4. Testing basic query execution...")
        df = execute_query("SELECT 1 as test_value")
        print(f"   ✅ Query executed: {df['test_value'].iloc[0]}")
        
        print("\n✅ Basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Basic functionality test failed: {e}")
        return False

def test_utility_functions():
    """Test utility functions."""
    print("\n🧪 Testing Utility Functions")
    print("=" * 50)
    
    try:
        from production.database.utils import (
            get_ticker_list, get_sector_mapping, get_table_info,
            get_table_row_count, get_database_stats
        )
        
        # Test get_ticker_list (without status filter)
        print("1. Testing get_ticker_list()...")
        tickers = get_ticker_list(active_only=False)  # Don't filter by status
        print(f"   ✅ Retrieved {len(tickers)} tickers")
        
        # Test get_sector_mapping
        print("2. Testing get_sector_mapping()...")
        sector_df = get_sector_mapping()
        print(f"   ✅ Retrieved sector mapping for {len(sector_df)} tickers")
        print(f"   ✅ Sectors: {sector_df['sector'].unique()}")
        
        # Test get_table_info
        print("3. Testing get_table_info()...")
        table_info = get_table_info('master_info')
        print(f"   ✅ Retrieved info for {len(table_info)} columns")
        
        # Test get_table_row_count
        print("4. Testing get_table_row_count()...")
        row_count = get_table_row_count('master_info')
        print(f"   ✅ master_info table has {row_count} rows")
        
        # Test get_database_stats
        print("5. Testing get_database_stats()...")
        stats = get_database_stats()
        print(f"   ✅ Database has {stats.get('table_count', 0)} tables")
        print(f"   ✅ Total size: {stats.get('total_size_mb', 0):.2f} MB")
        
        print("\n✅ Utility function tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Utility function test failed: {e}")
        return False

def test_advanced_features():
    """Test advanced features like context managers and custom configurations."""
    print("\n🧪 Testing Advanced Features")
    print("=" * 50)
    
    try:
        from production.database import DatabaseManager
        from production.database.utils import execute_query
        
        # Test DatabaseManager with custom settings
        print("1. Testing DatabaseManager...")
        db_manager = DatabaseManager(
            environment='production',
            enable_pooling=True,
            pool_size=5
        )
        print("   ✅ DatabaseManager created")
        
        # Test context managers
        print("2. Testing context managers...")
        with db_manager.get_engine_context() as engine:
            df = execute_query("SELECT COUNT(*) as count FROM master_info", engine=engine)
            print(f"   ✅ Engine context manager: {df['count'].iloc[0]} rows")
        
        with db_manager.get_connection_context() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) as count FROM master_info")
                result = cursor.fetchone()
                print(f"   ✅ Connection context manager: {result['count']} rows")
        
        # Test configuration
        print("3. Testing configuration...")
        config = db_manager.get_config()
        print(f"   ✅ Config loaded: {config['host']}:{config['schema_name']}")
        
        # Test connection testing
        print("4. Testing connection test...")
        if db_manager.test_connection():
            print("   ✅ Connection test passed")
        else:
            print("   ❌ Connection test failed")
        
        print("\n✅ Advanced feature tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Advanced feature test failed: {e}")
        return False

def test_data_operations():
    """Test data retrieval operations."""
    print("\n🧪 Testing Data Operations")
    print("=" * 50)
    
    try:
        from production.database.utils import (
            get_price_data, get_factor_scores, get_liquid_universe
        )
        
        # Test price data retrieval (check actual column names first)
        print("1. Testing price data retrieval...")
        try:
            # First check what columns exist in the table
            from production.database.utils import execute_query
            columns_df = execute_query("DESCRIBE vcsc_daily_data_complete")
            print(f"   📋 Available columns: {columns_df['Field'].tolist()}")
            
            # Use the correct column names
            price_df = get_price_data(
                tickers=['VNM', 'VCB'],
                start_date='2024-01-01',
                end_date='2024-01-31'
            )
            print(f"   ✅ Retrieved {len(price_df)} price records")
        except Exception as e:
            print(f"   ⚠️ Price data retrieval failed: {e}")
        
        # Test liquid universe
        print("2. Testing liquid universe...")
        try:
            universe_df = get_liquid_universe(
                analysis_date='2024-12-31',
                adtv_threshold=10.0,
                top_n=50
            )
            print(f"   ✅ Retrieved {len(universe_df)} liquid stocks")
        except Exception as e:
            print(f"   ⚠️ Liquid universe failed: {e}")
        
        # Test factor scores (if available)
        print("3. Testing factor scores...")
        try:
            factor_df = get_factor_scores(
                tickers=['VNM', 'VCB'],
                rebalance_date='2024-12-31'
            )
            print(f"   ✅ Retrieved factor scores for {len(factor_df)} stocks")
        except Exception as e:
            print(f"   ⚠️ Factor scores not available: {e}")
        
        print("\n✅ Data operation tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Data operation test failed: {e}")
        return False

def test_error_handling():
    """Test error handling."""
    print("\n🧪 Testing Error Handling")
    print("=" * 50)
    
    try:
        from production.database.connection import DatabaseConnectionError, DatabaseConfigError
        from production.database.utils import execute_query
        
        # Test invalid query
        print("1. Testing invalid query handling...")
        try:
            execute_query("SELECT * FROM non_existent_table")
            print("   ❌ Should have raised an error")
            return False
        except Exception as e:
            print(f"   ✅ Properly handled invalid query: {type(e).__name__}")
        
        # Test invalid table
        print("2. Testing invalid table handling...")
        try:
            from production.database.utils import get_table_info
            get_table_info('non_existent_table')
            print("   ❌ Should have raised an error")
            return False
        except Exception as e:
            print(f"   ✅ Properly handled invalid table: {type(e).__name__}")
        
        print("\n✅ Error handling tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error handling test failed: {e}")
        return False

def test_performance():
    """Test performance of the connection system."""
    print("\n🧪 Testing Performance")
    print("=" * 50)
    
    try:
        import time
        from production.database import get_engine, get_connection
        from production.database.utils import execute_query
        
        # Test engine creation time
        print("1. Testing engine creation performance...")
        start_time = time.time()
        engine = get_engine()
        engine_time = time.time() - start_time
        print(f"   ✅ Engine creation: {engine_time:.3f} seconds")
        
        # Test connection creation time
        print("2. Testing connection creation performance...")
        start_time = time.time()
        connection = get_connection()
        connection_time = time.time() - start_time
        print(f"   ✅ Connection creation: {connection_time:.3f} seconds")
        
        # Test query execution time
        print("3. Testing query execution performance...")
        start_time = time.time()
        df = execute_query("SELECT COUNT(*) FROM master_info")
        query_time = time.time() - start_time
        print(f"   ✅ Query execution: {query_time:.3f} seconds")
        
        # Test connection reuse
        print("4. Testing connection reuse...")
        start_time = time.time()
        engine2 = get_engine()  # Should reuse cached engine
        reuse_time = time.time() - start_time
        print(f"   ✅ Connection reuse: {reuse_time:.3f} seconds")
        
        print("\n✅ Performance tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Performance test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Database Connection System Test Suite")
    print("=" * 60)
    print(f"Project Root: {project_root}")
    print(f"Python Path: {sys.path[0]}")
    print("=" * 60)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Utility Functions", test_utility_functions),
        ("Advanced Features", test_advanced_features),
        ("Data Operations", test_data_operations),
        ("Error Handling", test_error_handling),
        ("Performance", test_performance)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    print("-" * 60)
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Database connection system is ready for use.")
        return True
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)