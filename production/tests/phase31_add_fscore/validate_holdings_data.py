#!/usr/bin/env python3
"""
Validate Holdings Data - Compare Generated vs Saved Data
========================================================

This script:
1. Generates a sample of holdings data using the QVM engine
2. Compares it against the saved qvm_v2.0_enhanced table
3. Validates data quality, consistency, and factor score distributions
4. Identifies any discrepancies or data quality issues

Evidence-based validation with real numbers and statistics.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
from sqlalchemy import text

# Add the project root to the path
sys.path.append('/home/raymond/Documents/Projects/factor-investing-public')

# Import required modules
from production.database.connection import get_engine
# from production.tests.phase31_add_fscore.07_QVM_flat_config import QVMFlatConfigEngine

def load_saved_holdings_data():
    """Load holdings data from the saved qvm_v2.0_enhanced table."""
    
    print("📊 LOADING SAVED HOLDINGS DATA")
    print("=" * 50)
    
    engine = get_engine()
    
    # Load sample data from saved table
    saved_query = """
    SELECT date, ticker, Quality_Composite, Value_Composite, Momentum_Composite, QVM_Composite
    FROM factor_scores_qvm 
    WHERE strategy_version = 'qvm_v2.0_enhanced'
    AND date >= '2021-01-01' AND date <= '2021-12-31'
    ORDER BY date, ticker
    LIMIT 1000
    """
    
    try:
        saved_df = pd.read_sql(saved_query, engine)
        print(f"✅ Loaded saved holdings data: {len(saved_df)} records")
        print(f"📅 Date range: {saved_df['date'].min()} to {saved_df['date'].max()}")
        print(f"📊 Unique tickers: {saved_df['ticker'].nunique()}")
        print(f"📊 Unique dates: {saved_df['date'].nunique()}")
        
        return saved_df
        
    except Exception as e:
        print(f"❌ Failed to load saved holdings data: {e}")
        return pd.DataFrame()

def generate_sample_holdings_data():
    """Generate a sample of holdings data using the QVM engine."""
    
    print("\n📊 GENERATING SAMPLE HOLDINGS DATA")
    print("=" * 50)
    
    # Create test configuration
    test_config = {
        'strategy': {
            'name': 'QVM Test',
            'version': 'v2.0.1',
            'portfolio': {
                'universe_size': 100,
                'portfolio_size': 20,
                'starting_capital': 10_000_000_000
            }
        },
        'factor_weights': {
            'quality': 0.25,
            'value': 0.25,
            'momentum': 0.25,
            'defensive': 0.25
        },
        'risk_management': {
            'enabled': True,
            'cash_allocation': {
                'drawdown_5': 0.20,
                'drawdown_10': 0.40,
                'drawdown_15': 0.60,
                'drawdown_20': 0.80,
                'drawdown_25': 0.90
            },
            'default_cash': 0.05
        }
    }
    
    test_backtest_config = {
        'active_window': 'FULL_2016_2025',
        'backtest_windows': {
            'LIQUID_2018_2025': {
                'start': '2021-01-01',
                'end': '2021-12-31',
                'description': 'Test period'
            }
        }
    }
    
    try:
        # Initialize engine
        import importlib.util
        spec = importlib.util.spec_from_file_location("qvm_config", "07_QVM_flat_config.py")
        qvm_config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(qvm_config)
        QVMFlatConfigEngine = qvm_config.QVMFlatConfigEngine
        
        engine_instance = QVMFlatConfigEngine(
            strategy_config=test_config,
            backtest_config=test_backtest_config,
            log_level='WARNING'
        )
        
        print("✅ Engine initialized successfully")
        
        # Generate holdings
        print("📊 Generating holdings data...")
        generated_df = engine_instance.generate_holdings_with_flat_methodology()
        
        if generated_df is not None and len(generated_df) > 0:
            print(f"✅ Generated holdings data: {len(generated_df)} records")
            print(f"📅 Date range: {generated_df['date'].min()} to {generated_df['date'].max()}")
            print(f"📊 Unique tickers: {generated_df['ticker'].nunique()}")
            print(f"📊 Unique dates: {generated_df['date'].nunique()}")
            
            return generated_df
        else:
            print("❌ Generated holdings data is empty")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"❌ Failed to generate holdings data: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def compare_holdings_data(saved_df, generated_df):
    """Compare saved vs generated holdings data."""
    
    print("\n📊 COMPARING HOLDINGS DATA")
    print("=" * 50)
    
    if len(saved_df) == 0 or len(generated_df) == 0:
        print("❌ Cannot compare - one or both datasets are empty")
        return
    
    # Basic statistics comparison
    print("📈 BASIC STATISTICS COMPARISON")
    print("-" * 30)
    
    # Compare factor score distributions
    factors = ['Quality_Composite', 'Value_Composite', 'Momentum_Composite', 'QVM_Composite']
    
    for factor in factors:
        if factor in saved_df.columns and factor in generated_df.columns:
            print(f"\n📊 {factor}:")
            
            saved_stats = saved_df[factor].describe()
            generated_stats = generated_df[factor].describe()
            
            print(f"   Saved Data:")
            print(f"     Mean: {saved_stats['mean']:.4f}")
            print(f"     Std: {saved_stats['std']:.4f}")
            print(f"     Min: {saved_stats['min']:.4f}")
            print(f"     Max: {saved_stats['max']:.4f}")
            print(f"     Null count: {saved_df[factor].isnull().sum()}")
            
            print(f"   Generated Data:")
            print(f"     Mean: {generated_stats['mean']:.4f}")
            print(f"     Std: {generated_stats['std']:.4f}")
            print(f"     Min: {generated_stats['min']:.4f}")
            print(f"     Max: {generated_stats['max']:.4f}")
            print(f"     Null count: {generated_df[factor].isnull().sum()}")
            
            # Calculate differences
            mean_diff = abs(saved_stats['mean'] - generated_stats['mean'])
            std_diff = abs(saved_stats['std'] - generated_stats['std'])
            
            print(f"   Differences:")
            print(f"     Mean diff: {mean_diff:.4f}")
            print(f"     Std diff: {std_diff:.4f}")
            
            if mean_diff > 0.1 or std_diff > 0.1:
                print(f"   ⚠️ WARNING: Large differences detected")
            else:
                print(f"   ✅ Differences within acceptable range")
    
    # Compare ticker overlap
    print(f"\n📊 TICKER OVERLAP ANALYSIS")
    print("-" * 30)
    
    saved_tickers = set(saved_df['ticker'].unique())
    generated_tickers = set(generated_df['ticker'].unique())
    
    overlap = saved_tickers.intersection(generated_tickers)
    saved_only = saved_tickers - generated_tickers
    generated_only = generated_tickers - saved_tickers
    
    print(f"   Saved tickers: {len(saved_tickers)}")
    print(f"   Generated tickers: {len(generated_tickers)}")
    print(f"   Overlap: {len(overlap)} ({len(overlap)/len(saved_tickers)*100:.1f}%)")
    print(f"   Saved only: {len(saved_only)}")
    print(f"   Generated only: {len(generated_only)}")
    
    if len(overlap) > 0:
        print(f"   Sample overlapping tickers: {list(overlap)[:5]}")
    
    # Compare date overlap
    print(f"\n📊 DATE OVERLAP ANALYSIS")
    print("-" * 30)
    
    saved_dates = set(saved_df['date'].unique())
    generated_dates = set(generated_df['date'].unique())
    
    overlap_dates = saved_dates.intersection(generated_dates)
    saved_only_dates = saved_dates - generated_dates
    generated_only_dates = generated_dates - saved_dates
    
    print(f"   Saved dates: {len(saved_dates)}")
    print(f"   Generated dates: {len(generated_dates)}")
    print(f"   Overlap: {len(overlap_dates)} ({len(overlap_dates)/len(saved_dates)*100:.1f}%)")
    print(f"   Saved only: {len(saved_only_dates)}")
    print(f"   Generated only: {len(generated_only_dates)}")
    
    if len(overlap_dates) > 0:
        print(f"   Sample overlapping dates: {sorted(list(overlap_dates))[:5]}")

def detailed_factor_analysis(saved_df, generated_df):
    """Perform detailed factor analysis and comparison."""
    
    print("\n📊 DETAILED FACTOR ANALYSIS")
    print("=" * 50)
    
    if len(saved_df) == 0 or len(generated_df) == 0:
        print("❌ Cannot analyze - one or both datasets are empty")
        return
    
    # Find common tickers and dates for detailed comparison
    common_tickers = set(saved_df['ticker'].unique()).intersection(set(generated_df['ticker'].unique()))
    common_dates = set(saved_df['date'].unique()).intersection(set(generated_df['date'].unique()))
    
    if len(common_tickers) == 0 or len(common_dates) == 0:
        print("❌ No common tickers or dates for detailed comparison")
        return
    
    print(f"📊 Detailed comparison using:")
    print(f"   Common tickers: {len(common_tickers)}")
    print(f"   Common dates: {len(common_dates)}")
    
    # Filter to common data
    saved_common = saved_df[
        (saved_df['ticker'].isin(common_tickers)) & 
        (saved_df['date'].isin(common_dates))
    ].copy()
    
    generated_common = generated_df[
        (generated_df['ticker'].isin(common_tickers)) & 
        (generated_df['date'].isin(common_dates))
    ].copy()
    
    # Merge for direct comparison
    comparison_df = saved_common.merge(
        generated_common,
        on=['date', 'ticker'],
        suffixes=('_saved', '_generated')
    )
    
    if len(comparison_df) == 0:
        print("❌ No matching records for direct comparison")
        return
    
    print(f"✅ Direct comparison records: {len(comparison_df)}")
    
    # Compare factor scores directly
    factors = ['Quality_Composite', 'Value_Composite', 'Momentum_Composite', 'QVM_Composite']
    
    for factor in factors:
        saved_col = f"{factor}_saved"
        generated_col = f"{factor}_generated"
        
        if saved_col in comparison_df.columns and generated_col in comparison_df.columns:
            print(f"\n📊 {factor} Direct Comparison:")
            
            # Calculate differences
            comparison_df[f'{factor}_diff'] = comparison_df[generated_col] - comparison_df[saved_col]
            comparison_df[f'{factor}_abs_diff'] = abs(comparison_df[f'{factor}_diff'])
            
            diff_stats = comparison_df[f'{factor}_diff'].describe()
            abs_diff_stats = comparison_df[f'{factor}_abs_diff'].describe()
            
            print(f"   Mean difference: {diff_stats['mean']:.6f}")
            print(f"   Std difference: {diff_stats['std']:.6f}")
            print(f"   Mean absolute difference: {abs_diff_stats['mean']:.6f}")
            print(f"   Max absolute difference: {abs_diff_stats['max']:.6f}")
            
            # Check for large differences
            large_diffs = comparison_df[comparison_df[f'{factor}_abs_diff'] > 0.1]
            print(f"   Records with >0.1 difference: {len(large_diffs)} ({len(large_diffs)/len(comparison_df)*100:.1f}%)")
            
            if len(large_diffs) > 0:
                print(f"   ⚠️ WARNING: Large differences detected in {len(large_diffs)} records")
                
                # Show examples of large differences
                sample_large = large_diffs.head(3)
                for _, row in sample_large.iterrows():
                    print(f"     {row['ticker']} ({row['date']}): {row[saved_col]:.4f} vs {row[generated_col]:.4f} (diff: {row[f'{factor}_diff']:.4f})")
            else:
                print(f"   ✅ All differences within acceptable range")
    
    # Correlation analysis
    print(f"\n📊 CORRELATION ANALYSIS")
    print("-" * 30)
    
    for factor in factors:
        saved_col = f"{factor}_saved"
        generated_col = f"{factor}_generated"
        
        if saved_col in comparison_df.columns and generated_col in comparison_df.columns:
            correlation = comparison_df[saved_col].corr(comparison_df[generated_col])
            print(f"   {factor}: {correlation:.4f}")
            
            if correlation < 0.8:
                print(f"   ⚠️ WARNING: Low correlation detected")
            else:
                print(f"   ✅ High correlation - data consistency good")

def generate_validation_report(saved_df, generated_df):
    """Generate a comprehensive validation report."""
    
    print("\n📊 VALIDATION REPORT")
    print("=" * 50)
    
    # Data quality metrics
    print("📈 DATA QUALITY METRICS")
    print("-" * 30)
    
    if len(saved_df) > 0:
        print(f"   Saved Data Quality:")
        print(f"     Total records: {len(saved_df):,}")
        print(f"     Unique tickers: {saved_df['ticker'].nunique()}")
        print(f"     Unique dates: {saved_df['date'].nunique()}")
        print(f"     Date range: {saved_df['date'].min()} to {saved_df['date'].max()}")
        print(f"     Records per ticker: {len(saved_df)/saved_df['ticker'].nunique():.1f}")
        print(f"     Records per date: {len(saved_df)/saved_df['date'].nunique():.1f}")
    
    if len(generated_df) > 0:
        print(f"   Generated Data Quality:")
        print(f"     Total records: {len(generated_df):,}")
        print(f"     Unique tickers: {generated_df['ticker'].nunique()}")
        print(f"     Unique dates: {generated_df['date'].nunique()}")
        print(f"     Date range: {generated_df['date'].min()} to {generated_df['date'].max()}")
        print(f"     Records per ticker: {len(generated_df)/generated_df['ticker'].nunique():.1f}")
        print(f"     Records per date: {len(generated_df)/generated_df['date'].nunique():.1f}")
    
    # Factor score quality
    print(f"\n📊 FACTOR SCORE QUALITY")
    print("-" * 30)
    
    factors = ['Quality_Composite', 'Value_Composite', 'Momentum_Composite', 'QVM_Composite']
    
    for factor in factors:
        if factor in saved_df.columns:
            saved_factor = saved_df[factor].dropna()
            print(f"   {factor} (Saved):")
            print(f"     Mean: {saved_factor.mean():.4f}")
            print(f"     Std: {saved_factor.std():.4f}")
            print(f"     Range: [{saved_factor.min():.4f}, {saved_factor.max():.4f}]")
            print(f"     Null %: {saved_df[factor].isnull().sum()/len(saved_df)*100:.1f}%")
        
        if factor in generated_df.columns:
            generated_factor = generated_df[factor].dropna()
            print(f"   {factor} (Generated):")
            print(f"     Mean: {generated_factor.mean():.4f}")
            print(f"     Std: {generated_factor.std():.4f}")
            print(f"     Range: [{generated_factor.min():.4f}, {generated_factor.max():.4f}]")
            print(f"     Null %: {generated_df[factor].isnull().sum()/len(generated_df)*100:.1f}%")
    
    # Summary and recommendations
    print(f"\n📋 SUMMARY AND RECOMMENDATIONS")
    print("-" * 30)
    
    if len(saved_df) > 0 and len(generated_df) > 0:
        print("✅ Both datasets loaded successfully")
        
        # Check for major issues
        saved_tickers = set(saved_df['ticker'].unique())
        generated_tickers = set(generated_df['ticker'].unique())
        overlap = len(saved_tickers.intersection(generated_tickers))
        
        if overlap > 0:
            print(f"✅ Found {overlap} overlapping tickers for comparison")
        else:
            print("❌ No overlapping tickers - cannot compare factor scores")
        
        # Check data consistency
        if len(saved_df) > len(generated_df) * 10:
            print("⚠️ Saved data has significantly more records than generated data")
        elif len(generated_df) > len(saved_df) * 10:
            print("⚠️ Generated data has significantly more records than saved data")
        else:
            print("✅ Data volume ratios are reasonable")
        
    else:
        print("❌ One or both datasets are empty")
        if len(saved_df) == 0:
            print("   - Saved data is empty")
        if len(generated_df) == 0:
            print("   - Generated data is empty")

def main():
    """Main validation function."""
    print("🚀 HOLDINGS DATA VALIDATION")
    print("=" * 60)
    print("Comparing generated vs saved holdings data")
    print("Focus: Data quality, consistency, and factor score validation")
    
    # Load saved data
    saved_df = load_saved_holdings_data()
    
    # Generate sample data
    generated_df = generate_sample_holdings_data()
    
    # Compare data
    compare_holdings_data(saved_df, generated_df)
    
    # Detailed analysis
    detailed_factor_analysis(saved_df, generated_df)
    
    # Generate report
    generate_validation_report(saved_df, generated_df)
    
    print("\n✅ VALIDATION COMPLETE")
    print("=" * 60)
    print("Check the output above for data quality assessment and recommendations.")

if __name__ == "__main__":
    main()
