#!/usr/bin/env python3
"""
Minimal test script to isolate the data corruption issue.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def minimal_qvm_test():
    """Run minimal QVM test to isolate data corruption."""
    
    print("🧪 MINIMAL QVM TEST - ISOLATING DATA CORRUPTION")
    print("=" * 55)
    
    # 1. Load the daily returns data
    print("\n📊 STEP 1: Loading daily returns data")
    print("-" * 40)
    
    daily_returns_path = Path("output/daily_returns.csv")
    if not daily_returns_path.exists():
        print(f"❌ File not found: {daily_returns_path}")
        return
    
    df = pd.read_csv(daily_returns_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Create strategy returns
    strategy_returns = df.set_index('date')['portfolio_return']
    strategy_returns.index = pd.to_datetime(strategy_returns.index)
    
    print(f"✅ Strategy returns loaded:")
    print(f"   Shape: {strategy_returns.shape}")
    print(f"   Mean: {strategy_returns.mean():.6f}")
    print(f"   Std: {strategy_returns.std():.6f}")
    
    # 2. Create sample benchmark data
    print("\n📊 STEP 2: Creating sample benchmark data")
    print("-" * 40)
    
    sample_dates = strategy_returns.index
    benchmark_returns = pd.Series(
        np.random.normal(0.0005, 0.012, len(sample_dates)),
        index=sample_dates
    )
    
    print(f"✅ Benchmark returns created:")
    print(f"   Shape: {benchmark_returns.shape}")
    print(f"   Mean: {benchmark_returns.mean():.6f}")
    print(f"   Std: {benchmark_returns.std():.6f}")
    
    # 3. Verify data integrity
    print("\n📊 STEP 3: Verifying data integrity")
    print("-" * 40)
    
    print(f"Strategy and benchmark are identical? {strategy_returns.equals(benchmark_returns)}")
    print(f"Strategy mean == benchmark mean? {strategy_returns.mean() == benchmark_returns.mean()}")
    
    # 4. Summary
    print("\n📊 STEP 4: Summary")
    print("-" * 40)
    
    if strategy_returns.mean() == benchmark_returns.mean():
        print("🚨 ALERT: Data corruption detected!")
    else:
        print("✅ Data integrity maintained")
    
    return strategy_returns, benchmark_returns

if __name__ == "__main__":
    minimal_qvm_test()



