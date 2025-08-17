#!/usr/bin/env python3
"""
Comprehensive test to trace the entire data flow and identify corruption.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def test_comprehensive_data_flow():
    """Test the entire data flow to identify corruption point."""
    
    print("🧪 COMPREHENSIVE DATA FLOW TEST")
    print("=" * 50)
    
    # 1. Load the actual data from the main script output
    print("\n📊 STEP 1: Loading actual data from main script")
    print("-" * 50)
    
    daily_returns_path = Path("output/daily_returns.csv")
    if not daily_returns_path.exists():
        print(f"❌ File not found: {daily_returns_path}")
        return
    
    df = pd.read_csv(daily_returns_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Create strategy returns exactly as in main script
    strategy_returns = df.set_index('date')['portfolio_return']
    strategy_returns.index = pd.to_datetime(strategy_returns.index)
    
    print(f"✅ Strategy returns loaded:")
    print(f"   Shape: {strategy_returns.shape}")
    print(f"   Mean: {strategy_returns.mean():.6f}")
    print(f"   Std: {strategy_returns.std():.6f}")
    print(f"   Hash: {hash(strategy_returns.to_string())}")
    
    # 2. Create benchmark data exactly as in main script
    print("\n📊 STEP 2: Creating benchmark data")
    print("-" * 50)
    
    # Create sample benchmark data (since we don't have the actual benchmark)
    sample_dates = strategy_returns.index
    benchmark_returns = pd.Series(
        np.random.normal(0.0005, 0.012, len(sample_dates)),
        index=sample_dates
    )
    
    print(f"✅ Benchmark returns created:")
    print(f"   Shape: {benchmark_returns.shape}")
    print(f"   Mean: {benchmark_returns.mean():.6f}")
    print(f"   Std: {benchmark_returns.std():.6f}")
    print(f"   Hash: {hash(benchmark_returns.to_string())}")
    
    # 3. Test period filtering exactly as in main script
    print("\n📊 STEP 3: Testing period filtering (main script logic)")
    print("-" * 50)
    
    # CRITICAL: Use the exact same logic as the main script
    strategy_returns_copy = strategy_returns.copy()
    benchmark_returns_copy = benchmark_returns.copy()
    
    print(f"✅ Deep copies created:")
    print(f"   Strategy copy hash: {hash(strategy_returns_copy.to_string())}")
    print(f"   Benchmark copy hash: {hash(benchmark_returns_copy.to_string())}")
    
    # First period (2016-2020) - EXACT same logic as main script
    first_period_mask = (strategy_returns_copy.index >= '2016-01-01') & (strategy_returns_copy.index <= '2020-12-31')
    first_period_strategy_returns = strategy_returns_copy[first_period_mask]
    first_period_benchmark_returns = benchmark_returns_copy.reindex(first_period_strategy_returns.index).fillna(0)
    
    print(f"✅ First period data created:")
    print(f"   Strategy shape: {first_period_strategy_returns.shape}")
    print(f"   Strategy mean: {first_period_strategy_returns.mean():.6f}")
    print(f"   Strategy hash: {hash(first_period_strategy_returns.to_string())}")
    print(f"   Benchmark shape: {first_period_benchmark_returns.shape}")
    print(f"   Benchmark mean: {first_period_benchmark_returns.mean():.6f}")
    print(f"   Benchmark hash: {hash(first_period_benchmark_returns.to_string())}")
    
    # 4. Test data integrity after period filtering
    print("\n📊 STEP 4: Testing data integrity after period filtering")
    print("-" * 50)
    
    print(f"Original strategy unchanged? {hash(strategy_returns.to_string()) == hash(strategy_returns_copy.to_string())}")
    print(f"Original benchmark unchanged? {hash(benchmark_returns.to_string()) == hash(benchmark_returns_copy.to_string())}")
    print(f"First period strategy subset? {first_period_strategy_returns.equals(strategy_returns_copy.loc[first_period_strategy_returns.index])}")
    
    # 5. Test if the issue is in the data itself
    print("\n📊 STEP 5: Testing data integrity patterns")
    print("-" * 50)
    
    print(f"Strategy returns unique values: {strategy_returns.nunique()}")
    print(f"First period unique values: {first_period_strategy_returns.nunique()}")
    print(f"Strategy returns > 0: {(strategy_returns > 0).sum()}")
    print(f"Strategy returns < 0: {(strategy_returns < 0).sum()}")
    print(f"Strategy returns = 0: {(strategy_returns == 0).sum()}")
    
    # 6. Check for suspicious patterns
    print("\n📊 STEP 6: Checking for suspicious patterns")
    print("-" * 50)
    
    # Check if all returns are the same
    if strategy_returns.nunique() == 1:
        print("🚨 ALERT: All strategy returns are identical!")
        print(f"   Single value: {strategy_returns.iloc[0]}")
    else:
        print("✅ Strategy returns have multiple unique values")
        
        # Check if there are suspicious clusters
        unique_values = strategy_returns.unique()
        print(f"   Number of unique values: {len(unique_values)}")
        print(f"   Value range: {unique_values.min():.6f} to {unique_values.max():.6f}")
        
        # Check if most values are the same
        value_counts = strategy_returns.value_counts()
        most_common = value_counts.iloc[0]
        total_values = len(strategy_returns)
        print(f"   Most common value: {most_common} ({(most_common/total_values)*100:.1f}% of data)")
    
    # 7. Summary and diagnosis
    print("\n📊 STEP 7: Summary and Diagnosis")
    print("-" * 50)
    
    print("✅ Comprehensive data flow test completed.")
    print("\n🎯 Diagnosis:")
    
    if strategy_returns.nunique() == 1:
        print("🚨 ALERT: Data corruption detected in the source data!")
        print("   → Issue is in the daily returns calculation")
    elif hash(strategy_returns.to_string()) != hash(strategy_returns_copy.to_string()):
        print("🚨 ALERT: Data corruption during deep copy!")
        print("   → Issue is in the copy operation")
    elif not first_period_strategy_returns.equals(strategy_returns_copy.loc[first_period_strategy_returns.index]):
        print("🚨 ALERT: Data corruption during period filtering!")
        print("   → Issue is in the boolean mask operation")
    else:
        print("✅ Data integrity maintained throughout the process")
        print("   → Issue must be elsewhere in the main script execution")
    
    return strategy_returns, benchmark_returns, first_period_strategy_returns

if __name__ == "__main__":
    test_comprehensive_data_flow()
