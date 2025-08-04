#!/usr/bin/env python3
"""
Diagnostic script to check fundamental data loading and factor calculation issues.
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import sys
import os
from datetime import datetime

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

def check_precomputed_fundamental_data(engine, analysis_date):
    """Check if precomputed fundamental data exists and is accessible."""
    print(f"\n🔍 Checking precomputed fundamental data for {analysis_date}...")
    
    # Check fundamental_values table
    fundamental_query = text("""
        SELECT COUNT(*) as count
        FROM fundamental_values 
        WHERE year = :year AND quarter = :quarter
        AND item_id IN (1, 2)
    """)
    
    year = analysis_date.year
    quarter = (analysis_date.month - 1) // 3 + 1
    
    result = pd.read_sql(fundamental_query, engine, 
                        params={'year': year, 'quarter': quarter})
    print(f"   - Fundamental values records: {result['count'].iloc[0]:,}")
    
    # Check financial_metrics table
    metrics_query = text("""
        SELECT COUNT(*) as count
        FROM financial_metrics 
        WHERE Date BETWEEN :start_date AND :end_date
        AND PE IS NOT NULL AND PE > 0
    """)
    
    start_date = f"{year}-{quarter*3-2:02d}-01"
    end_date = f"{year}-{quarter*3:02d}-28"
    
    result = pd.read_sql(metrics_query, engine, 
                        params={'start_date': start_date, 'end_date': end_date})
    print(f"   - Financial metrics records: {result['count'].iloc[0]:,}")
    
    # Check specific data for a few tickers
    sample_query = text("""
        SELECT 
            fv.ticker,
            fv.year,
            fv.quarter,
            fv.item_id,
            fv.statement_type,
            fv.value / 1e9 as value_bn
        FROM fundamental_values fv
        WHERE fv.year = :year AND fv.quarter = :quarter
        AND fv.item_id IN (1, 2)
        AND fv.ticker IN ('VNM', 'VCB', 'TCB')
        ORDER BY fv.ticker, fv.item_id
    """)
    
    sample_data = pd.read_sql(sample_query, engine, 
                             params={'year': year, 'quarter': quarter})
    print(f"   - Sample fundamental data:")
    print(sample_data.to_string())
    
    # Check financial metrics for same tickers
    metrics_sample_query = text("""
        SELECT 
            ticker,
            Date,
            PE,
            PB,
            EPS,
            MarketCapitalization / 1e9 as market_cap_bn
        FROM financial_metrics 
        WHERE Date BETWEEN :start_date AND :end_date
        AND ticker IN ('VNM', 'VCB', 'TCB')
        ORDER BY ticker, Date
    """)
    
    metrics_sample = pd.read_sql(metrics_sample_query, engine, 
                                params={'start_date': start_date, 'end_date': end_date})
    print(f"   - Sample financial metrics:")
    print(metrics_sample.to_string())

def test_precompute_fundamental_factors(engine, config):
    """Test the precompute_fundamental_factors function."""
    print(f"\n🧪 Testing precompute_fundamental_factors function...")
    
    # Import the function
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "strategy_module", 
        "production/tests/phase29-alpha_demo/08_integrated_strategy_with_validated_factors_fixed.py"
    )
    strategy_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(strategy_module)
    
    # Test the function
    try:
        fundamental_data = strategy_module.precompute_fundamental_factors(config, engine)
        print(f"   ✅ Function executed successfully")
        print(f"   - Data shape: {fundamental_data.shape}")
        print(f"   - Columns: {list(fundamental_data.columns)}")
        
        # Check for key columns
        key_columns = ['pe', 'roaa', 'net_margin']
        for col in key_columns:
            if col in fundamental_data.columns:
                non_null_count = fundamental_data[col].notna().sum()
                print(f"   - {col}: {non_null_count:,} non-null values")
            else:
                print(f"   - {col}: MISSING COLUMN")
        
        # Show sample data
        print(f"   - Sample data:")
        print(fundamental_data.head(10).to_string())
        
        return fundamental_data
        
    except Exception as e:
        print(f"   ❌ Function failed: {e}")
        return None

def test_factor_integration(engine, config, analysis_date):
    """Test how factors are integrated in the main calculation."""
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
    
    # Create mock fundamental data
    fundamental_data = pd.DataFrame({
        'ticker': universe,
        'date': [analysis_date] * len(universe),
        'pe': [10.5, 8.2, 12.1, 15.3, 20.1],
        'roaa': [0.08, 0.12, 0.06, 0.09, 0.15],
        'net_margin': [0.15, 0.25, 0.10, 0.18, 0.30]
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
    
    # Create mock precomputed data
    precomputed_data = {
        'universe': pd.DataFrame({
            'trading_date': [analysis_date] * len(universe),
            'ticker': universe,
            'rank': range(1, len(universe) + 1)
        }),
        'fundamental': fundamental_data,
        'momentum': pd.DataFrame({
            'trading_date': [analysis_date] * len(universe),
            'ticker': universe,
            'momentum_21d': np.random.randn(len(universe)) * 0.05,
            'momentum_63d': np.random.randn(len(universe)) * 0.10,
            'momentum_126d': np.random.randn(len(universe)) * 0.15,
            'momentum_252d': np.random.randn(len(universe)) * 0.20
        })
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
    """Main diagnostic function."""
    print("🔍 FUNDAMENTAL DATA DIAGNOSTIC")
    print("=" * 50)
    
    # Create database connection
    engine = create_db_connection()
    if not engine:
        return
    
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
    
    # Test dates
    test_dates = [
        pd.Timestamp('2016-02-01'),
        pd.Timestamp('2020-06-01'),
        pd.Timestamp('2024-07-01')
    ]
    
    for test_date in test_dates:
        print(f"\n{'='*60}")
        print(f"TESTING DATE: {test_date}")
        print(f"{'='*60}")
        
        # Check precomputed data
        check_precomputed_fundamental_data(engine, test_date)
        
        # Test precompute function
        fundamental_data = test_precompute_fundamental_factors(engine, config)
        
        # Test factor integration
        factors_df = test_factor_integration(engine, config, test_date)
        
        print(f"\n{'='*60}")
        print(f"SUMMARY FOR {test_date}")
        print(f"{'='*60}")
        
        if fundamental_data is not None:
            print(f"✅ Fundamental data available: {len(fundamental_data):,} records")
        else:
            print(f"❌ Fundamental data FAILED")
            
        if factors_df is not None:
            print(f"✅ Factor calculation successful: {len(factors_df):,} stocks")
            if 'composite_score' in factors_df.columns:
                avg_score = factors_df['composite_score'].mean()
                print(f"   - Average composite score: {avg_score:.4f}")
        else:
            print(f"❌ Factor calculation FAILED")

if __name__ == "__main__":
    main() 