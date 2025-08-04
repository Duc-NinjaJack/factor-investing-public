#!/usr/bin/env python3
"""
Debug script to check fundamental data columns.
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import sys
import os

# Add project root to path
project_root = "/Users/raymond/Documents/Projects/factor-investing-public"
sys.path.append(project_root)

from production.database.connection import DatabaseManager

def create_db_connection():
    """Create database connection."""
    try:
        db_manager = DatabaseManager()
        engine = db_manager.get_engine()
        print("✅ Database connection established successfully.")
        return engine
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return None

def check_fundamental_data_columns(engine):
    """Check what columns are available in the precomputed fundamental data."""
    print("\n🔍 Checking precomputed fundamental data columns...")
    
    # Import the precompute function
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "strategy_module", 
        "production/tests/phase29-alpha_demo/08_integrated_strategy_with_validated_factors_fixed.py"
    )
    strategy_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(strategy_module)
    
    # Configuration
    config = {
        'backtest_start_date': '2016-01-01',
        'backtest_end_date': '2025-07-28',
        'universe': {'target_size': 200},
        'factors': {
            'fundamental_lag_days': 30,
            'volatility_lookback': 252,
            'momentum_horizons': [21, 63, 126, 252],
            'skip_months': 1
        }
    }
    
    try:
        # Run the precompute function
        fundamental_data = strategy_module.precompute_fundamental_factors(config, engine)
        
        print(f"   ✅ Fundamental data loaded: {len(fundamental_data):,} records")
        print(f"   - Columns: {list(fundamental_data.columns)}")
        
        # Check for key columns
        key_columns = ['pe', 'roaa', 'net_margin', 'pb', 'eps', 'market_cap']
        for col in key_columns:
            if col in fundamental_data.columns:
                non_null_count = fundamental_data[col].notna().sum()
                print(f"   - {col}: {non_null_count:,} non-null values")
                
                # Show sample values
                sample_values = fundamental_data[col].dropna().head(5)
                print(f"     Sample values: {sample_values.values}")
            else:
                print(f"   - {col}: MISSING COLUMN")
        
        # Show sample data
        print(f"\n   - Sample fundamental data:")
        print(fundamental_data.head(10).to_string())
        
        return fundamental_data
        
    except Exception as e:
        print(f"   ❌ Error loading fundamental data: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_factor_integration(engine, fundamental_data, analysis_date):
    """Test how the fundamental data is integrated into factors."""
    print(f"\n🧪 Testing factor integration for {analysis_date}...")
    
    # Import the strategy module
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "strategy_module", 
        "production/tests/phase29-alpha_demo/08_integrated_strategy_with_validated_factors_fixed.py"
    )
    strategy_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(strategy_module)
    
    # Create mock data
    universe = ['VNM', 'VCB', 'TCB', 'HPG', 'FPT']
    
    # Create mock price data
    price_data = pd.DataFrame({
        'ticker': universe * 100,  # 100 days of data
        'trading_date': pd.date_range('2024-01-01', periods=100).repeat(len(universe)),
        'close_price': np.random.randn(500) * 100 + 1000
    })
    
    # Create mock returns matrix
    returns_matrix = pd.DataFrame(
        np.random.randn(100, len(universe)) * 0.02,
        index=pd.date_range('2024-01-01', periods=100),
        columns=universe
    )
    
    # Create mock benchmark returns
    benchmark_returns = pd.Series(
        np.random.randn(100) * 0.015,
        index=pd.date_range('2024-01-01', periods=100)
    )
    
    # Create mock precomputed data with actual fundamental data
    precomputed_data = {
        'universe': pd.DataFrame({
            'trading_date': [analysis_date] * len(universe),
            'ticker': universe,
            'rank': range(1, len(universe) + 1)
        }),
        'fundamentals': fundamental_data[fundamental_data['ticker'].isin(universe)].copy(),
        'momentum': pd.DataFrame({
            'trading_date': [analysis_date] * len(universe),
            'ticker': universe,
            'momentum_21d': np.random.randn(len(universe)) * 0.05,
            'momentum_63d': np.random.randn(len(universe)) * 0.10,
            'momentum_126d': np.random.randn(len(universe)) * 0.15,
            'momentum_252d': np.random.randn(len(universe)) * 0.20
        })
    }
    
    # Configuration
    config = {
        'backtest_start_date': '2016-01-01',
        'backtest_end_date': '2025-07-28',
        'universe': {'target_size': 200},
        'factors': {
            'fundamental_lag_days': 30,
            'volatility_lookback': 252,
            'momentum_horizons': [21, 63, 126, 252],
            'skip_months': 1,
            'value_factors': {
                'pe_weight': 0.5,
                'fcf_yield_weight': 0.5
            },
            'quality_factors': {
                'roaa_weight': 0.5,
                'fscore_weight': 0.5
            },
            'momentum_weight': 0.34
        }
    }
    
    try:
        # Create the engine instance
        engine_instance = strategy_module.QVMEngineV3jValidatedFactors(
            config, price_data, fundamental_data, returns_matrix, 
            benchmark_returns, engine, precomputed_data
        )
        
        # Test factor calculation
        factors_df = engine_instance._get_validated_factors_from_precomputed(universe, analysis_date)
        
        print(f"   ✅ Factor calculation completed")
        print(f"   - Factors shape: {factors_df.shape}")
        print(f"   - Columns: {list(factors_df.columns)}")
        
        # Check for key factors
        key_factors = ['pe', 'roaa', 'fscore', 'fcf_yield', 'momentum_score', 'composite_score']
        for factor in key_factors:
            if factor in factors_df.columns:
                non_null_count = factors_df[factor].notna().sum()
                print(f"   - {factor}: {non_null_count:,} non-null values")
                
                # Show sample values
                if non_null_count > 0:
                    sample_values = factors_df[factor].dropna().head(3)
                    print(f"     Sample values: {sample_values.values}")
            else:
                print(f"   - {factor}: MISSING FACTOR")
        
        # Show sample factors
        print(f"   - Sample factors:")
        print(factors_df.to_string())
        
        return factors_df
        
    except Exception as e:
        print(f"   ❌ Factor calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function."""
    print("🔍 FUNDAMENTAL DATA COLUMNS DEBUG")
    print("=" * 50)
    
    # Create database connection
    engine = create_db_connection()
    if not engine:
        return
    
    # Check fundamental data columns
    fundamental_data = check_fundamental_data_columns(engine)
    
    if fundamental_data is not None:
        # Test factor integration
        test_date = pd.Timestamp('2020-06-01')
        factors_df = test_factor_integration(engine, fundamental_data, test_date)

if __name__ == "__main__":
    main() 