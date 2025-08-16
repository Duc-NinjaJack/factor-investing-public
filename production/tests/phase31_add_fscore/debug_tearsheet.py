#!/usr/bin/env python3
"""
Debug script to trace tearsheet generation and identify where strategy returns are corrupted.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def debug_tearsheet_generation():
    """Debug the tearsheet generation process to identify corruption point."""
    
    print("🔍 DEBUGGING TEARSHEET GENERATION - IDENTIFYING CORRUPTION")
    print("=" * 65)
    
    # 1. Load the data exactly as the main script does
    print("\n📊 STEP 1: Loading data exactly as main script")
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
    
    print(f"✅ Strategy returns created:")
    print(f"   Shape: {strategy_returns.shape}")
    print(f"   Mean: {strategy_returns.mean():.6f}")
    print(f"   Std: {strategy_returns.std():.6f}")
    print(f"   Sample values: {strategy_returns.head(3).tolist()}")
    
    # 2. Simulate benchmark data creation
    print("\n📊 STEP 2: Simulating benchmark data creation")
    print("-" * 50)
    
    # Create sample benchmark data (since we don't have the actual benchmark data)
    # This will help us see if the issue is in the benchmark creation
    sample_dates = strategy_returns.index
    sample_benchmark_returns = pd.Series(
        np.random.normal(0.0005, 0.012, len(sample_dates)),
        index=sample_dates
    )
    
    print(f"✅ Sample benchmark returns created:")
    print(f"   Shape: {sample_benchmark_returns.shape}")
    print(f"   Mean: {sample_benchmark_returns.mean():.6f}")
    print(f"   Std: {sample_benchmark_returns.std():.6f}")
    print(f"   Sample values: {sample_benchmark_returns.head(3).tolist()}")
    
    # 3. Simulate date alignment
    print("\n📊 STEP 3: Simulating date alignment")
    print("-" * 50)
    
    # This is where the corruption might be happening
    common_dates = strategy_returns.index.intersection(sample_benchmark_returns.index)
    
    print(f"✅ Date alignment:")
    print(f"   Original strategy dates: {len(strategy_returns.index)}")
    print(f"   Original benchmark dates: {len(sample_benchmark_returns.index)}")
    print(f"   Common dates: {len(common_dates)}")
    
    # Create aligned versions (this is where corruption might occur)
    aligned_strategy_returns = strategy_returns.loc[common_dates]
    aligned_benchmark_returns = sample_benchmark_returns.loc[common_dates]
    
    print(f"\n✅ After alignment:")
    print(f"   Aligned strategy shape: {aligned_strategy_returns.shape}")
    print(f"   Aligned strategy mean: {aligned_strategy_returns.mean():.6f}")
    print(f"   Aligned strategy std: {aligned_strategy_returns.std():.6f}")
    print(f"   Aligned benchmark shape: {aligned_benchmark_returns.shape}")
    print(f"   Aligned benchmark mean: {aligned_benchmark_returns.mean():.6f}")
    print(f"   Aligned benchmark std: {aligned_benchmark_returns.std():.6f}")
    
    # 4. Check if alignment corrupted the data
    print("\n📊 STEP 4: Checking for data corruption during alignment")
    print("-" * 50)
    
    # Compare original vs aligned
    print(f"Strategy returns identical after alignment? {strategy_returns.equals(aligned_strategy_returns)}")
    print(f"Strategy returns subset check: {aligned_strategy_returns.equals(strategy_returns.loc[common_dates])}")
    
    # Check if the issue is in the .loc operation
    print(f"\n🔍 Detailed comparison:")
    print(f"   Original first 3 values: {strategy_returns.head(3).tolist()}")
    print(f"   Aligned first 3 values: {aligned_strategy_returns.head(3).tolist()}")
    
    # 5. Simulate the performance metrics calculation
    print("\n📊 STEP 5: Simulating performance metrics calculation")
    print("-" * 50)
    
    # This is where the final corruption might be happening
    print("📊 Manual calculation for comparison:")
    print(f"   Strategy returns mean: {aligned_strategy_returns.mean():.6f}")
    print(f"   Strategy returns std: {aligned_strategy_returns.std():.6f}")
    print(f"   Benchmark returns mean: {aligned_benchmark_returns.mean():.6f}")
    print(f"   Benchmark returns std: {aligned_benchmark_returns.std():.6f}")
    
    # Calculate excess returns
    excess_returns = aligned_strategy_returns - aligned_benchmark_returns
    print(f"   Excess returns mean: {excess_returns.mean():.6f}")
    print(f"   Excess returns std: {excess_returns.std():.6f}")
    
    # 6. Check for any suspicious data modifications
    print("\n📊 STEP 6: Checking for suspicious data modifications")
    print("-" * 50)
    
    # Check if the data was modified during processing
    print(f"Strategy returns modified during processing? {not strategy_returns.equals(aligned_strategy_returns)}")
    
    # Check if there are any NaN or infinite values introduced
    print(f"Aligned strategy has NaN: {aligned_strategy_returns.isna().sum()}")
    print(f"Aligned strategy has infinite: {np.isinf(aligned_strategy_returns).sum()}")
    
    # 7. Summary and diagnosis
    print("\n📊 STEP 7: Summary and Diagnosis")
    print("-" * 50)
    
    print("✅ Tearsheet generation debug completed.")
    print("\n🎯 Diagnosis:")
    
    if strategy_returns.equals(aligned_strategy_returns):
        print("✅ Strategy returns unchanged during alignment")
        print("   → Issue must be elsewhere in the process")
    else:
        print("🚨 Strategy returns CHANGED during alignment!")
        print("   → Issue is in the date alignment logic")
    
    if aligned_strategy_returns.mean() == aligned_benchmark_returns.mean():
        print("🚨 ALERT: Strategy and benchmark means are identical!")
        print("   → Data corruption detected")
    else:
        print("✅ Strategy and benchmark means are different")
        print("   → Data integrity maintained")
    
    return aligned_strategy_returns, aligned_benchmark_returns

if __name__ == "__main__":
    debug_tearsheet_generation()
