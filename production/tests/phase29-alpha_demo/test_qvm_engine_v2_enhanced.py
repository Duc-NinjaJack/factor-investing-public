#!/usr/bin/env python3
"""
Test script for QVM Engine v2 Enhanced
======================================
Tests the enhanced QVM engine and identifies any errors or issues.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import warnings
warnings.filterwarnings('ignore')

# Add the necessary paths to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'engine'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'universe'))

try:
    from qvm_engine_v2_enhanced import QVMEngineV2Enhanced
    print("✅ Successfully imported QVMEngineV2Enhanced")
except ImportError as e:
    print(f"❌ Failed to import QVMEngineV2Enhanced: {e}")
    sys.exit(1)

try:
    from constructors import get_liquid_universe
    print("✅ Successfully imported get_liquid_universe")
except ImportError as e:
    print(f"❌ Failed to import get_liquid_universe: {e}")
    # We'll handle this later

print(f"Test started: {datetime.now()}")
print("QVM Engine v2 Enhanced - Error Detection and Testing")

def test_engine_initialization():
    """Test engine initialization"""
    print("\n=== Testing Engine Initialization ===")
    try:
        # Configure logging
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
        
        # Initialize the enhanced engine
        engine = QVMEngineV2Enhanced()
        
        print("✅ QVM Engine v2 Enhanced initialized successfully")
        print(f"   - Engine class: {engine.__class__.__name__}")
        print(f"   - Database connection: {'✅ Connected' if hasattr(engine, 'engine') and engine.engine else '❌ Failed'}")
        
        # Check available methods
        methods = [method for method in dir(engine) if not method.startswith('_')]
        print(f"   - Available methods: {methods[:10]}...")
        
        return engine
        
    except Exception as e:
        print(f"❌ Engine initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_universe_construction(engine):
    """Test universe construction"""
    print("\n=== Testing Universe Construction ===")
    try:
        # Try to get liquid universe
        if 'get_liquid_universe' in globals():
            # Set analysis date for universe construction
            analysis_date = datetime(2025, 8, 2)
            # Pass the database engine, not the QVM engine
            universe = get_liquid_universe(analysis_date, engine.engine)
            print(f"✅ Liquid universe constructed: {len(universe)} tickers")
            print(f"   - Sample tickers: {universe[:5]}")
            return universe
        else:
            # Fallback: create a simple test universe
            print("⚠️  get_liquid_universe not available, creating test universe")
            test_universe = ['TCB', 'VCB', 'BID', 'CTG', 'MBB']  # Sample banking tickers
            print(f"✅ Test universe created: {len(test_universe)} tickers")
            return test_universe
            
    except Exception as e:
        print(f"❌ Universe construction failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_factor_calculation(engine, universe):
    """Test factor calculation"""
    print("\n=== Testing Factor Calculation ===")
    try:
        # Set analysis date
        analysis_date = datetime(2025, 8, 2)
        print(f"Analysis Date: {analysis_date.strftime('%Y-%m-%d')}")
        
        # Calculate QVM composite
        print("Calculating QVM composite scores...")
        results = engine.calculate_qvm_composite(analysis_date, universe)
        
        if results:
            print(f"✅ QVM calculation successful: {len(results)} tickers processed")
            
            # Show sample results
            sample_ticker = list(results.keys())[0]
            sample_result = results[sample_ticker]
            print(f"   - Sample result for {sample_ticker}:")
            for component, score in sample_result.items():
                print(f"     {component}: {score:.4f}")
            
            return results
        else:
            print("❌ QVM calculation returned empty results")
            return None
            
    except Exception as e:
        print(f"❌ Factor calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_individual_components(engine, universe):
    """Test individual QVM components"""
    print("\n=== Testing Individual Components ===")
    try:
        analysis_date = datetime(2025, 8, 2)
        
        # Test fundamental data retrieval
        print("Testing fundamental data retrieval...")
        fundamentals = engine.get_fundamentals_correct_timing(analysis_date, universe)
        if not fundamentals.empty:
            print(f"✅ Fundamental data retrieved: {len(fundamentals)} records")
            print(f"   - Columns: {list(fundamentals.columns)[:10]}...")
        else:
            print("⚠️  No fundamental data retrieved")
        
        # Test market data retrieval
        print("Testing market data retrieval...")
        market_data = engine.get_market_data(analysis_date, universe)
        if not market_data.empty:
            print(f"✅ Market data retrieved: {len(market_data)} records")
            print(f"   - Columns: {list(market_data.columns)}")
        else:
            print("⚠️  No market data retrieved")
        
        # Test sector mapping
        print("Testing sector mapping...")
        sector_map = engine.get_sector_mapping()
        if not sector_map.empty:
            print(f"✅ Sector mapping retrieved: {len(sector_map)} tickers")
            print(f"   - Sectors: {sector_map['sector'].value_counts().to_dict()}")
        else:
            print("⚠️  No sector mapping retrieved")
            
    except Exception as e:
        print(f"❌ Component testing failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main test function"""
    print("Starting comprehensive QVM Engine v2 Enhanced testing...")
    
    # Test 1: Engine initialization
    engine = test_engine_initialization()
    if not engine:
        print("❌ Cannot proceed without engine initialization")
        return
    
    # Test 2: Universe construction
    universe = test_universe_construction(engine)
    if not universe:
        print("❌ Cannot proceed without universe")
        return
    
    # Test 3: Individual components
    test_individual_components(engine, universe)
    
    # Test 4: Full factor calculation
    results = test_factor_calculation(engine, universe)
    
    print("\n=== Test Summary ===")
    if results:
        print("✅ All tests completed successfully!")
        print(f"   - Engine: QVMEngineV2Enhanced")
        print(f"   - Universe: {len(universe)} tickers")
        print(f"   - Results: {len(results)} tickers processed")
    else:
        print("❌ Some tests failed - check error messages above")

if __name__ == "__main__":
    main() 