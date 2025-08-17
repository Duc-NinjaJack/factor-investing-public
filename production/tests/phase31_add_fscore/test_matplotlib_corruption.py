#!/usr/bin/env python3
"""
Test to check if matplotlib operations are corrupting the data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def test_matplotlib_corruption():
    """Test if matplotlib operations corrupt the data."""
    
    print("🧪 TESTING MATPLOTLIB CORRUPTION")
    print("=" * 50)
    
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
    
    # 2. Test data integrity before matplotlib operations
    print("\n📊 STEP 2: Testing data integrity before matplotlib")
    print("-" * 40)
    
    strategy_hash_before = hash(strategy_returns.to_string())
    benchmark_hash_before = hash(benchmark_returns.to_string())
    
    print(f"   Strategy hash before: {strategy_hash_before}")
    print(f"   Benchmark hash before: {benchmark_hash_before}")
    
    # 3. Simulate matplotlib operations from tearsheet functions
    print("\n📊 STEP 3: Simulating matplotlib operations")
    print("-" * 40)
    
    try:
        # Create a figure (like in tearsheet functions)
        fig = plt.figure(figsize=(18, 30))
        gs = fig.add_gridspec(6, 2, height_ratios=[1.2, 0.8, 0.8, 0.8, 0.8, 1.2], hspace=0.7, wspace=0.2)
        
        # Simulate the exact operations from the tearsheet function
        first_trade_date = strategy_returns.loc[strategy_returns.ne(0)].index.min()
        aligned_strategy_returns = strategy_returns.loc[first_trade_date:]
        aligned_benchmark_returns = benchmark_returns.loc[first_trade_date:]
        
        print(f"✅ Data alignment completed:")
        print(f"   First trade date: {first_trade_date}")
        print(f"   Aligned strategy shape: {aligned_strategy_returns.shape}")
        print(f"   Aligned benchmark shape: {aligned_benchmark_returns.shape}")
        
        # Create plots (like in tearsheet functions)
        ax1 = fig.add_subplot(gs[0, :])
        
        # Plot the main equity curves (EXACT same code as tearsheet)
        (1 + aligned_strategy_returns).cumprod().plot(ax=ax1, label='QVM Engine v3 (F-Score)', color='#16A085', lw=2.5)
        (1 + aligned_benchmark_returns).cumprod().plot(ax=ax1, label='VN-Index (Aligned)', color='#34495E', linestyle='--', lw=2)
        
        ax1.set_title('Cumulative Performance (Log Scale)', fontweight='bold')
        ax1.set_ylabel('Growth of 1 VND')
        ax1.set_yscale('log')
        ax1.legend(loc='upper left')
        ax1.grid(True, which='both', linestyle='--', alpha=0.5)
        
        print("✅ Equity curve plot created")
        
        # Create more plots (like in tearsheet functions)
        ax2 = fig.add_subplot(gs[1, :])
        ax2.text(0.5, 0.5, 'Cash Allocation Chart', ha='center', va='center', transform=ax2.transAxes, fontsize=14)
        ax2.set_title('Cash Allocation Over Time', fontweight='bold')
        
        ax3 = fig.add_subplot(gs[2, :])
        drawdown = ((1 + aligned_strategy_returns).cumprod() / (1 + aligned_strategy_returns).cumprod().cummax() - 1) * 100
        drawdown.plot(ax=ax3, color='#C0392B')
        ax3.fill_between(drawdown.index, drawdown, 0, color='#C0392B', alpha=0.1)
        ax3.set_title('Drawdown Analysis', fontweight='bold')
        ax3.set_ylabel('Drawdown (%)')
        ax3.grid(True, linestyle='--', alpha=0.5)
        
        print("✅ Additional plots created")
        
        # Close the figure to free memory
        plt.close(fig)
        print("✅ Figure closed")
        
    except Exception as e:
        print(f"❌ Error during matplotlib operations: {e}")
        return
    
    # 4. Test data integrity after matplotlib operations
    print("\n📊 STEP 4: Testing data integrity after matplotlib")
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
    
    # 6. Check for global variable pollution
    print("\n📊 STEP 6: Checking for global variable pollution")
    print("-" * 40)
    
    # Check if matplotlib created any global variables that might interfere
    global_vars = [var for var in globals() if 'strategy' in var.lower() or 'benchmark' in var.lower()]
    print(f"   Global variables with 'strategy' or 'benchmark': {global_vars}")
    
    # Check if there are any hidden variables
    import sys
    module_vars = [var for var in dir(sys.modules[__name__]) if 'strategy' in var.lower() or 'benchmark' in var.lower()]
    print(f"   Module variables with 'strategy' or 'benchmark': {module_vars}")
    
    # 7. Summary and diagnosis
    print("\n📊 STEP 7: Summary and Diagnosis")
    print("-" * 40)
    
    print("✅ Matplotlib corruption test completed.")
    print("\n🎯 Diagnosis:")
    
    if strategy_hash_before == strategy_hash_after:
        print("✅ Strategy returns unchanged during matplotlib operations")
        print("   → Issue must be elsewhere in the process")
    else:
        print("🚨 Strategy returns CHANGED during matplotlib operations!")
        print("   → Issue is in matplotlib plotting")
    
    if benchmark_hash_before == benchmark_hash_after:
        print("✅ Benchmark returns unchanged during matplotlib operations")
        print("   → Issue must be elsewhere in the process")
    else:
        print("🚨 Benchmark returns CHANGED during matplotlib operations!")
        print("   → Issue is in matplotlib plotting")
    
    if strategy_returns.mean() == benchmark_returns.mean():
        print("🚨 ALERT: Strategy and benchmark means are identical!")
        print("   → Data corruption detected")
    else:
        print("✅ Strategy and benchmark means are different")
        print("   → Data integrity maintained")
    
    return strategy_returns, benchmark_returns

if __name__ == "__main__":
    test_matplotlib_corruption()
