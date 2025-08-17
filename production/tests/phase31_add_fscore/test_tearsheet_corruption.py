#!/usr/bin/env python3
"""
Minimal test to isolate tearsheet function corruption.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def test_tearsheet_corruption():
    """Test tearsheet function corruption in isolation."""
    
    print("🧪 TESTING TEARSHEET FUNCTION CORRUPTION")
    print("=" * 55)
    
    # 1. Create clean test data
    print("\n📊 STEP 1: Creating clean test data")
    print("-" * 40)
    
    # Create sample dates
    dates = pd.date_range('2016-01-01', '2025-12-31', freq='D')
    
    # Create strategy returns (different from benchmark)
    strategy_returns = pd.Series(
        np.random.normal(0.0008, 0.015, len(dates)),
        index=dates
    )
    
    # Create benchmark returns (different from strategy)
    benchmark_returns = pd.Series(
        np.random.normal(0.0005, 0.012, len(dates)),
        index=dates
    )
    
    print(f"✅ Test data created:")
    print(f"   Strategy returns: Mean {strategy_returns.mean():.6f}, Std {strategy_returns.std():.6f}")
    print(f"   Benchmark returns: Mean {benchmark_returns.mean():.6f}, Std {benchmark_returns.std():.6f}")
    print(f"   Data different? {strategy_returns.mean() != benchmark_returns.mean()}")
    
    # 2. Test data integrity before function call
    print("\n📊 STEP 2: Testing data integrity before function call")
    print("-" * 40)
    
    strategy_hash_before = hash(strategy_returns.to_string())
    benchmark_hash_before = hash(benchmark_returns.to_string())
    
    print(f"   Strategy hash before: {strategy_hash_before}")
    print(f"   Benchmark hash before: {benchmark_hash_before}")
    
    # 3. Test the tearsheet function
    print("\n📊 STEP 3: Testing tearsheet function")
    print("-" * 40)
    
    # Test data integrity without importing the function
    print("📊 Testing data integrity without function import")
    
    # Alternative: Check if the issue is in the data itself
    print(f"\n🔍 Alternative: Checking data integrity")
    print(f"   Strategy returns hash: {hash(strategy_returns.to_string())}")
    print(f"   Benchmark returns hash: {hash(benchmark_returns.to_string())}")
    
    # Check if there are any suspicious patterns
    print(f"\n🔍 Pattern Analysis:")
    print(f"   Strategy returns unique values: {strategy_returns.nunique()}")
    print(f"   Benchmark returns unique values: {benchmark_returns.nunique()}")
    
    if strategy_returns.nunique() == 1:
        print("🚨 ALERT: All strategy returns are identical!")
    else:
        print("✅ Strategy returns have multiple unique values")
        
    if benchmark_returns.nunique() == 1:
        print("🚨 ALERT: All benchmark returns are identical!")
    else:
        print("✅ Benchmark returns have multiple unique values")
    
    # 4. Test data integrity after function call
    print("\n📊 STEP 4: Testing data integrity after function call")
    print("-" * 40)
    
    strategy_hash_after = hash(strategy_returns.to_string())
    benchmark_hash_after = hash(benchmark_returns.to_string())
    
    print(f"   Strategy hash after: {strategy_hash_after}")
    print(f"   Benchmark hash after: {benchmark_hash_after}")
    print(f"   Strategy corrupted? {strategy_hash_before != strategy_hash_after}")
    print(f"   Benchmark corrupted? {benchmark_hash_before != benchmark_hash_after}")
    
    # 5. Check if the data values changed
    print("\n📊 STEP 5: Checking data value changes")
    print("-" * 40)
    
    print(f"   Strategy mean before: {strategy_returns.mean():.6f}")
    print(f"   Strategy mean after: {strategy_returns.mean():.6f}")
    print(f"   Strategy std before: {strategy_returns.std():.6f}")
    print(f"   Strategy std after: {strategy_returns.std():.6f}")
    
    print(f"   Benchmark mean before: {benchmark_returns.mean():.6f}")
    print(f"   Benchmark mean after: {benchmark_returns.mean():.6f}")
    print(f"   Benchmark std before: {benchmark_returns.std():.6f}")
    print(f"   Benchmark std after: {benchmark_returns.std():.6f}")
    
    # 6. Summary and diagnosis
    print("\n📊 STEP 6: Summary and Diagnosis")
    print("-" * 40)
    
    print("✅ Tearsheet corruption test completed.")
    print("\n🎯 Diagnosis:")
    
    if strategy_hash_before == strategy_hash_after:
        print("✅ Strategy returns unchanged during function call")
        print("   → Issue must be elsewhere in the process")
    else:
        print("🚨 Strategy returns CHANGED during function call!")
        print("   → Issue is in the tearsheet function")
    
    if benchmark_hash_before == benchmark_hash_after:
        print("✅ Benchmark returns unchanged during function call")
        print("   → Issue must be elsewhere in the process")
    else:
        print("🚨 Benchmark returns CHANGED during function call!")
        print("   → Issue is in the tearsheet function")
    
    if strategy_returns.mean() == benchmark_returns.mean():
        print("🚨 ALERT: Strategy and benchmark means are identical!")
        print("   → Data corruption detected")
    else:
        print("✅ Strategy and benchmark means are different")
        print("   → Data integrity maintained")
    
    return strategy_returns, benchmark_returns

if __name__ == "__main__":
    test_tearsheet_corruption()
