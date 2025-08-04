#!/usr/bin/env python3
"""
Diagnostic script to understand regime detection and strategy underperformance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import sys
import os

# Add the parent directory to the path to import the strategy
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the strategy module
import importlib.util
spec = importlib.util.spec_from_file_location(
    "strategy_module", 
    "08_integrated_strategy_with_validated_factors_fixed.py"
)
strategy_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(strategy_module)

def analyze_market_data_characteristics():
    """Analyze the actual market data characteristics to understand regime detection"""
    print("=== MARKET DATA CHARACTERISTICS ANALYSIS ===")
    
    # Load market data
    engine = strategy_module.create_db_connection()
    
    # Get VNINDEX data
    query = """
    SELECT date, close, volume
    FROM etf_history 
    WHERE ticker = 'VNINDEX' 
    AND date >= '2016-01-01' 
    AND date <= '2025-12-31'
    ORDER BY date
    """
    
    market_data = pd.read_sql(query, engine, params=())
    market_data['date'] = pd.to_datetime(market_data['date'])
    market_data.set_index('date', inplace=True)
    
    # Calculate returns and volatility
    market_data['returns'] = market_data['close'].pct_change()
    market_data['volatility_21d'] = market_data['returns'].rolling(21).std()
    market_data['volatility_63d'] = market_data['returns'].rolling(63).std()
    market_data['returns_21d'] = market_data['returns'].rolling(21).mean()
    market_data['returns_63d'] = market_data['returns'].rolling(63).mean()
    
    print(f"Market data shape: {market_data.shape}")
    print(f"Date range: {market_data.index.min()} to {market_data.index.max()}")
    
    # Analyze volatility distribution
    print("\n=== VOLATILITY ANALYSIS ===")
    print(f"21-day volatility - Mean: {market_data['volatility_21d'].mean():.6f}")
    print(f"21-day volatility - Median: {market_data['volatility_21d'].median():.6f}")
    print(f"21-day volatility - 25th percentile: {market_data['volatility_21d'].quantile(0.25):.6f}")
    print(f"21-day volatility - 75th percentile: {market_data['volatility_21d'].quantile(0.75):.6f}")
    print(f"21-day volatility - 95th percentile: {market_data['volatility_21d'].quantile(0.95):.6f}")
    
    print(f"\n63-day volatility - Mean: {market_data['volatility_63d'].mean():.6f}")
    print(f"63-day volatility - Median: {market_data['volatility_63d'].median():.6f}")
    print(f"63-day volatility - 25th percentile: {market_data['volatility_63d'].quantile(0.25):.6f}")
    print(f"63-day volatility - 75th percentile: {market_data['volatility_63d'].quantile(0.75):.6f}")
    print(f"63-day volatility - 95th percentile: {market_data['volatility_63d'].quantile(0.95):.6f}")
    
    # Analyze returns distribution
    print("\n=== RETURNS ANALYSIS ===")
    print(f"21-day returns - Mean: {market_data['returns_21d'].mean():.6f}")
    print(f"21-day returns - Median: {market_data['returns_21d'].median():.6f}")
    print(f"21-day returns - 25th percentile: {market_data['returns_21d'].quantile(0.25):.6f}")
    print(f"21-day returns - 75th percentile: {market_data['returns_21d'].quantile(0.75):.6f}")
    print(f"21-day returns - 95th percentile: {market_data['returns_21d'].quantile(0.95):.6f}")
    
    print(f"\n63-day returns - Mean: {market_data['returns_63d'].mean():.6f}")
    print(f"63-day returns - Median: {market_data['returns_63d'].median():.6f}")
    print(f"63-day returns - 25th percentile: {market_data['returns_63d'].quantile(0.25):.6f}")
    print(f"63-day returns - 75th percentile: {market_data['returns_63d'].quantile(0.75):.6f}")
    print(f"63-day returns - 95th percentile: {market_data['returns_63d'].quantile(0.95):.6f}")
    
    # Test current regime thresholds
    current_thresholds = strategy_module.QVM_CONFIG['regime']
    print(f"\n=== CURRENT REGIME THRESHOLDS ===")
    print(f"Lookback period: {current_thresholds['lookback_period']}")
    print(f"Volatility threshold: {current_thresholds['volatility_threshold']}")
    print(f"Return threshold: {current_thresholds['return_threshold']}")
    print(f"Low return threshold: {current_thresholds['low_return_threshold']}")
    
    # Test regime detection with current thresholds
    print(f"\n=== REGIME DETECTION TEST WITH CURRENT THRESHOLDS ===")
    
    # Create a sample of dates to test
    test_dates = market_data.index[100:200]  # Skip first 100 days to have enough history
    
    regime_counts = {'Bull': 0, 'Bear': 0, 'Sideways': 0, 'Volatile': 0}
    
    for date in test_dates:
        # Get historical data for regime detection
        start_date = date - timedelta(days=current_thresholds['lookback_period'])
        historical_data = market_data.loc[start_date:date]
        
        if len(historical_data) < current_thresholds['lookback_period']:
            continue
            
        # Calculate regime metrics
        volatility = historical_data['returns'].std()
        returns = historical_data['returns'].mean()
        
        # Apply regime detection logic
        if volatility > current_thresholds['volatility_threshold']:
            regime = 'Volatile'
        elif returns > current_thresholds['return_threshold']:
            regime = 'Bull'
        elif returns < -current_thresholds['return_threshold']:
            regime = 'Bear'
        elif returns < current_thresholds['low_return_threshold']:
            regime = 'Sideways'
        else:
            regime = 'Sideways'
            
        regime_counts[regime] += 1
        
        if date in test_dates[:10]:  # Show first 10 examples
            print(f"Date: {date.date()}, Volatility: {volatility:.6f}, Returns: {returns:.6f}, Regime: {regime}")
    
    print(f"\nTotal dates processed: {sum(regime_counts.values())}")
    print(f"Dates with insufficient history: {len(test_dates) - sum(regime_counts.values())}")
    
    print(f"\nRegime distribution in test sample:")
    total_count = sum(regime_counts.values())
    if total_count > 0:
        for regime, count in regime_counts.items():
            percentage = (count / total_count) * 100
            print(f"{regime}: {count} times ({percentage:.1f}%)")
    else:
        print("No regimes detected in test sample!")
        print("This suggests the thresholds are too restrictive.")
    
    return market_data

def test_different_regime_thresholds():
    """Test different regime threshold combinations"""
    print("\n=== TESTING DIFFERENT REGIME THRESHOLDS ===")
    
    # Load market data
    engine = strategy_module.create_db_connection()
    query = """
    SELECT date, close
    FROM etf_history 
    WHERE ticker = 'VNINDEX' 
    AND date >= '2016-01-01' 
    AND date <= '2025-12-31'
    ORDER BY date
    """
    
    market_data = pd.read_sql(query, engine, params=())
    market_data['date'] = pd.to_datetime(market_data['date'])
    market_data.set_index('date', inplace=True)
    market_data['returns'] = market_data['close'].pct_change()
    
    # Test different threshold combinations
    threshold_combinations = [
        {'lookback': 21, 'vol': 0.0080, 'ret': 0.0005, 'low_ret': 0.0010},  # Current
        {'lookback': 42, 'vol': 0.0120, 'ret': 0.0010, 'low_ret': 0.0005},  # Previous
        {'lookback': 63, 'vol': 0.0160, 'ret': 0.0015, 'low_ret': 0.0001},  # Earlier
        {'lookback': 30, 'vol': 0.0100, 'ret': 0.0008, 'low_ret': 0.0003},  # Balanced
        {'lookback': 60, 'vol': 0.0200, 'ret': 0.0020, 'low_ret': 0.0005},  # Conservative
    ]
    
    test_dates = market_data.index[100:300]  # Larger sample
    
    for i, thresholds in enumerate(threshold_combinations):
        print(f"\n--- Threshold Set {i+1}: {thresholds} ---")
        
        regime_counts = {'Bull': 0, 'Bear': 0, 'Sideways': 0, 'Volatile': 0}
        
        for date in test_dates:
            start_date = date - timedelta(days=thresholds['lookback'])
            historical_data = market_data.loc[start_date:date]
            
            if len(historical_data) < thresholds['lookback']:
                continue
                
            volatility = historical_data['returns'].std()
            returns = historical_data['returns'].mean()
            
            # Apply regime detection logic
            if volatility > thresholds['vol']:
                regime = 'Volatile'
            elif returns > thresholds['ret']:
                regime = 'Bull'
            elif returns < -thresholds['ret']:
                regime = 'Bear'
            elif returns < thresholds['low_ret']:
                regime = 'Sideways'
            else:
                regime = 'Sideways'
                
            regime_counts[regime] += 1
        
        total = sum(regime_counts.values())
        print(f"Regime distribution:")
        for regime, count in regime_counts.items():
            percentage = (count / total) * 100 if total > 0 else 0
            print(f"  {regime}: {count} times ({percentage:.1f}%)")

def compare_with_working_strategy():
    """Compare current strategy with the working 07_integrated_strategy_enhanced"""
    print("\n=== COMPARING WITH WORKING STRATEGY ===")
    
    # Load the working strategy
    spec = importlib.util.spec_from_file_location(
        "working_strategy", 
        "07_integrated_strategy_enhanced.py"
    )
    working_strategy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(working_strategy)
    
    print("Current strategy config:")
    print(f"  Strategy name: {strategy_module.QVM_CONFIG['strategy_name']}")
    print(f"  Factor weights: {strategy_module.QVM_CONFIG['factors']}")
    print(f"  Adaptive rebalancing: {'Yes' if 'adaptive_rebalancing' in strategy_module.QVM_CONFIG else 'No'}")
    
    print("\nWorking strategy config:")
    print(f"  Strategy name: {working_strategy.QVM_CONFIG['strategy_name']}")
    print(f"  Factor weights: {working_strategy.QVM_CONFIG['factors']}")
    print(f"  Adaptive rebalancing: {'Yes' if 'adaptive_rebalancing' in working_strategy.QVM_CONFIG else 'No'}")
    
    # Check if working strategy has regime detection
    if hasattr(working_strategy, 'QVM_CONFIG') and 'regime' in working_strategy.QVM_CONFIG:
        print(f"  Regime detection: {working_strategy.QVM_CONFIG['regime']}")
    else:
        print("  Regime detection: No")

def analyze_factor_calculations():
    """Analyze if factor calculations are working correctly"""
    print("\n=== FACTOR CALCULATION ANALYSIS ===")
    
    # Test factor calculations on a sample date
    engine = strategy_module.create_db_connection()
    
    # Get a sample date
    query = """
    SELECT DISTINCT trading_date 
    FROM vcsc_daily_data 
    WHERE trading_date >= '2020-01-01' 
    ORDER BY trading_date 
    LIMIT 1
    """
    
    sample_date = pd.read_sql(query, engine, params=()).iloc[0]['trading_date']
    print(f"Testing factor calculations for date: {sample_date}")
    
    # Test individual factor functions
    try:
        # Test F-Score calculation
        fscore_data = strategy_module.calculate_piotroski_fscore(sample_date)
        print(f"F-Score calculation - Shape: {fscore_data.shape if hasattr(fscore_data, 'shape') else 'No data'}")
        if hasattr(fscore_data, 'shape') and fscore_data.shape[0] > 0:
            print(f"F-Score range: {fscore_data['fscore'].min()} to {fscore_data['fscore'].max()}")
        
        # Test FCF Yield calculation
        fcf_data = strategy_module.calculate_fcf_yield(sample_date)
        print(f"FCF Yield calculation - Shape: {fcf_data.shape if hasattr(fcf_data, 'shape') else 'No data'}")
        if hasattr(fcf_data, 'shape') and fcf_data.shape[0] > 0:
            print(f"FCF Yield range: {fcf_data['fcf_yield'].min()} to {fcf_data['fcf_yield'].max()}")
        
        # Test Low-Volatility calculation
        lowvol_data = strategy_module.calculate_low_volatility_factor(sample_date)
        print(f"Low-Volatility calculation - Shape: {lowvol_data.shape if hasattr(lowvol_data, 'shape') else 'No data'}")
        if hasattr(lowvol_data, 'shape') and lowvol_data.shape[0] > 0:
            print(f"Low-Volatility range: {lowvol_data['low_volatility_score'].min()} to {lowvol_data['low_volatility_score'].max()}")
            
    except Exception as e:
        print(f"Error testing factor calculations: {e}")

if __name__ == "__main__":
    print("=== REGIME DETECTION AND STRATEGY PERFORMANCE DIAGNOSTIC ===")
    
    # Run all analyses
    market_data = analyze_market_data_characteristics()
    test_different_regime_thresholds()
    compare_with_working_strategy()
    analyze_factor_calculations()
    
    print("\n=== DIAGNOSTIC COMPLETE ===") 