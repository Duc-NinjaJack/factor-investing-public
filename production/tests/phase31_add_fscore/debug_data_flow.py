#!/usr/bin/env python3
"""
Debug script to trace data flow and identify where strategy returns are corrupted.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def debug_data_flow():
    """Debug the data flow to identify where corruption occurs."""
    
    print("🔍 DEBUGGING DATA FLOW - IDENTIFYING CORRUPTION POINT")
    print("=" * 60)
    
    # 1. Check the daily returns CSV file
    print("\n📊 STEP 1: Checking daily returns CSV file")
    print("-" * 40)
    
    daily_returns_path = Path("output/daily_returns.csv")
    if daily_returns_path.exists():
        df = pd.read_csv(daily_returns_path)
        print(f"✅ File exists: {daily_returns_path}")
        print(f"📊 Shape: {df.shape}")
        print(f"📋 Columns: {df.columns.tolist()}")
        
        # Check portfolio returns
        portfolio_returns = df['portfolio_return']
        print(f"\n💰 Portfolio Returns Analysis:")
        print(f"   Count: {len(portfolio_returns)}")
        print(f"   Mean: {portfolio_returns.mean():.6f}")
        print(f"   Std: {portfolio_returns.std():.6f}")
        print(f"   Min: {portfolio_returns.min():.6f}")
        print(f"   Max: {portfolio_returns.max():.6f}")
        print(f"   Sample values: {portfolio_returns.head(5).tolist()}")
        
        # Check allocations
        allocations = df['allocation']
        print(f"\n📈 Allocation Analysis:")
        print(f"   Unique values: {allocations.unique()}")
        print(f"   Mean allocation: {allocations.mean():.2%}")
        
        # Check cash allocations
        cash_allocations = df['cash_allocation']
        print(f"\n💵 Cash Allocation Analysis:")
        print(f"   Unique values: {cash_allocations.unique()}")
        print(f"   Mean cash: {cash_allocations.mean():.2%}")
        
    else:
        print(f"❌ File not found: {daily_returns_path}")
        return
    
    # 2. Simulate the data processing steps
    print("\n📊 STEP 2: Simulating data processing steps")
    print("-" * 40)
    
    # Convert to datetime and set index
    df['date'] = pd.to_datetime(df['date'])
    strategy_returns = df.set_index('date')['portfolio_return']
    
    print(f"✅ Strategy returns created:")
    print(f"   Shape: {strategy_returns.shape}")
    print(f"   Index type: {type(strategy_returns.index)}")
    print(f"   First few dates: {strategy_returns.index[:5].tolist()}")
    print(f"   Mean: {strategy_returns.mean():.6f}")
    print(f"   Std: {strategy_returns.std():.6f}")
    
    # 3. Check if there are any data type issues
    print("\n📊 STEP 3: Checking data types and values")
    print("-" * 40)
    
    print(f"Strategy returns dtype: {strategy_returns.dtype}")
    print(f"Strategy returns has NaN: {strategy_returns.isna().sum()}")
    print(f"Strategy returns has infinite: {np.isinf(strategy_returns).sum()}")
    
    # Check for duplicate indices
    print(f"Strategy returns has duplicate indices: {strategy_returns.index.duplicated().sum()}")
    
    # 4. Check if the issue is in the CSV reading
    print("\n📊 STEP 4: Checking CSV reading integrity")
    print("-" * 40)
    
    # Read the CSV again to see if there are any issues
    df2 = pd.read_csv(daily_returns_path)
    df2['date'] = pd.to_datetime(df2['date'])
    strategy_returns2 = df2.set_index('date')['portfolio_return']
    
    print(f"Second read - Strategy returns:")
    print(f"   Shape: {strategy_returns2.shape}")
    print(f"   Mean: {strategy_returns2.mean():.6f}")
    print(f"   Std: {strategy_returns2.std():.6f}")
    
    # Compare the two reads
    print(f"Are the two reads identical? {strategy_returns.equals(strategy_returns2)}")
    
    # 5. Check if there's a memory issue
    print("\n📊 STEP 5: Checking for memory/data corruption")
    print("-" * 40)
    
    # Create a copy and modify it
    strategy_returns_copy = strategy_returns.copy()
    strategy_returns_copy.iloc[0] = 999.0  # Modify first value
    
    print(f"Original first value: {strategy_returns.iloc[0]}")
    print(f"Modified first value: {strategy_returns_copy.iloc[0]}")
    print(f"Original unchanged? {strategy_returns.iloc[0] != 999.0}")
    
    # 6. Check if the issue is in the original calculation
    print("\n📊 STEP 6: Checking if issue is in original calculation")
    print("-" * 40)
    
    # Check if all returns are the same (indicating calculation issue)
    unique_returns = strategy_returns.nunique()
    print(f"Number of unique return values: {unique_returns}")
    
    if unique_returns == 1:
        print("🚨 ALERT: All returns are identical! Issue is in calculation.")
        print(f"   Single return value: {strategy_returns.iloc[0]}")
    else:
        print(f"✅ Returns have {unique_returns} unique values - calculation looks OK")
    
    # Check for suspicious patterns
    print(f"\n🔍 Pattern Analysis:")
    print(f"   Returns > 0: {(strategy_returns > 0).sum()}")
    print(f"   Returns < 0: {(strategy_returns < 0).sum()}")
    print(f"   Returns = 0: {(strategy_returns == 0).sum()}")
    
    # 7. Summary and next steps
    print("\n📊 STEP 7: Summary and Next Steps")
    print("-" * 40)
    
    print("✅ Data integrity check completed.")
    print("\n🎯 Next Steps:")
    print("1. If all returns are identical → Issue is in daily returns calculation")
    print("2. If returns are different → Issue is in tearsheet generation")
    print("3. If data types are wrong → Issue is in data conversion")
    
    return strategy_returns

if __name__ == "__main__":
    debug_data_flow()



