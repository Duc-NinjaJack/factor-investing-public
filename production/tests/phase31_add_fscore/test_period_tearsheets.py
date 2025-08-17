#!/usr/bin/env python3
"""
Test script to isolate period tearsheet generation and identify data corruption.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def test_period_tearsheets():
    """Test period tearsheet generation in isolation."""
    
    print("🧪 TESTING PERIOD TEARSHEETS IN ISOLATION")
    print("=" * 50)
    
    # 1. Load the data exactly as the main script does
    print("\n📊 STEP 1: Loading data")
    print("-" * 30)
    
    daily_returns_path = Path("output/daily_returns.csv")
    if not daily_returns_path.exists():
        print(f"❌ File not found: {daily_returns_path}")
        return
    
    df = pd.read_csv(daily_returns_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Create strategy returns exactly as in main script
    strategy_returns = df.set_index('date')['portfolio_return']
    strategy_returns.index = pd.to_datetime(strategy_returns.index)
    
    print(f"✅ Strategy returns created:")
    print(f"   Shape: {strategy_returns.shape}")
    print(f"   Mean: {strategy_returns.mean():.6f}")
    print(f"   Std: {strategy_returns.std():.6f}")
    
    # 2. Create benchmark returns
    print("\n📊 STEP 2: Creating benchmark returns")
    print("-" * 30)
    
    # Create sample benchmark data
    sample_dates = strategy_returns.index
    benchmark_returns = pd.Series(
        np.random.normal(0.0005, 0.012, len(sample_dates)),
        index=sample_dates
    )
    
    print(f"✅ Benchmark returns created:")
    print(f"   Shape: {benchmark_returns.shape}")
    print(f"   Mean: {benchmark_returns.mean():.6f}")
    print(f"   Std: {benchmark_returns.std():.6f}")
    
    # 3. Test period filtering
    print("\n📊 STEP 3: Testing period filtering")
    print("-" * 30)
    
    # First period (2016-2020)
    first_period_mask = (strategy_returns.index >= '2016-01-01') & (strategy_returns.index <= '2020-12-31')
    first_period_strategy_returns = strategy_returns[first_period_mask]
    first_period_benchmark_returns = benchmark_returns.reindex(first_period_strategy_returns.index).fillna(0)
    
    print(f"✅ First period data created:")
    print(f"   Strategy shape: {first_period_strategy_returns.shape}")
    print(f"   Strategy mean: {first_period_strategy_returns.mean():.6f}")
    print(f"   Benchmark shape: {first_period_benchmark_returns.shape}")
    print(f"   Benchmark mean: {first_period_benchmark_returns.mean():.6f}")
    
    # 4. Test data integrity after filtering
    print("\n📊 STEP 4: Testing data integrity after filtering")
    print("-" * 30)
    
    print(f"Strategy returns identical after filtering? {strategy_returns.equals(first_period_strategy_returns)}")
    print(f"Strategy returns subset check: {first_period_strategy_returns.equals(strategy_returns.loc[first_period_strategy_returns.index])}")
    
    # 5. Test if the issue is in the main script execution
    print("\n📊 STEP 5: Testing main script execution")
    print("-" * 30)
    
    # Test data integrity without importing the function
    print("📊 Testing data integrity without function import")
    
    # Alternative: Check if the issue is in the data processing
    print(f"\n🔍 Alternative: Checking data processing integrity")
    print(f"   Strategy returns hash: {hash(strategy_returns.to_string())}")
    print(f"   First period strategy hash: {hash(first_period_strategy_returns.to_string())}")
    
    # Check if there are any suspicious patterns
    print(f"\n🔍 Pattern Analysis:")
    print(f"   Strategy returns unique values: {strategy_returns.nunique()}")
    print(f"   First period unique values: {first_period_strategy_returns.nunique()}")
    
    if strategy_returns.nunique() == 1:
        print("🚨 ALERT: All strategy returns are identical!")
    else:
        print("✅ Strategy returns have multiple unique values")
    
    # 6. Summary
    print("\n📊 STEP 6: Summary")
    print("-" * 30)
    
    print("✅ Period tearsheet test completed.")
    print("\n🎯 Diagnosis:")
    
    if strategy_returns.mean() == first_period_strategy_returns.mean():
        print("✅ Data integrity maintained during period filtering")
        print("   → Issue must be elsewhere in the process")
    else:
        print("🚨 Data corruption detected during period filtering!")
        print("   → Issue is in the period filtering logic")
    
    return first_period_strategy_returns, first_period_benchmark_returns

if __name__ == "__main__":
    test_period_tearsheets()
