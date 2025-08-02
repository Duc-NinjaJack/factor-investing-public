#!/usr/bin/env python3
"""
Test script for QVM Engine v3e percentile-based regime detection
"""

import pandas as pd
import numpy as np
import yaml
from datetime import datetime, timedelta

# Import the regime detector
import importlib.util
spec = importlib.util.spec_from_file_location('qvm_v3e', '06_qvm_engine_v3e_fixed.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

RegimeDetector = module.RegimeDetector

def create_test_data():
    """Create test price data with different market conditions"""
    np.random.seed(42)
    
    # Create date range
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    
    # Create different market regimes
    data = []
    
    # Normal regime (first 200 days)
    normal_returns = np.random.normal(0.0005, 0.015, 200)
    normal_prices = 1000 * np.exp(np.cumsum(normal_returns))
    
    # Momentum regime (next 100 days)
    momentum_returns = np.random.normal(0.002, 0.025, 100)
    momentum_prices = normal_prices[-1] * np.exp(np.cumsum(momentum_returns))
    
    # Stress regime (next 100 days)
    stress_returns = np.random.normal(-0.001, 0.030, 100)
    stress_prices = momentum_prices[-1] * np.exp(np.cumsum(stress_returns))
    
    # Back to normal (remaining days)
    remaining_days = len(dates) - 400
    final_returns = np.random.normal(0.0003, 0.018, remaining_days)
    final_prices = stress_prices[-1] * np.exp(np.cumsum(final_returns))
    
    # Combine all prices
    all_prices = np.concatenate([normal_prices, momentum_prices, stress_prices, final_prices])
    
    # Create DataFrame
    price_data = pd.DataFrame({
        'close': all_prices
    }, index=dates[:len(all_prices)])
    
    return price_data

def test_regime_detection():
    """Test the regime detection with different market conditions"""
    print("Testing QVM Engine v3e - Percentile-based Regime Detection")
    print("=" * 60)
    
    # Load configuration from config file
    try:
        with open('config/config_v3e_percentile_regime.yml', 'r') as f:
            config = yaml.safe_load(f)
        regime_config = config.get('regime', {})
        print("Configuration loaded from config/config_v3e_percentile_regime.yml")
    except FileNotFoundError:
        print("Config file not found, using default configuration")
        regime_config = {
            'lookback_period': 90,
            'volatility_percentile_high': 75.0,
            'return_percentile_high': 75.0,
            'return_percentile_low': 25.0
        }
    
    # Create test data
    price_data = create_test_data()
    print(f"Created test data: {len(price_data)} days")
    
    # Initialize regime detector
    regime_detector = RegimeDetector(
        lookback_period=regime_config.get('lookback_period', 90),
        volatility_percentile_high=regime_config.get('volatility_percentile_high', 75.0),
        return_percentile_high=regime_config.get('return_percentile_high', 75.0),
        return_percentile_low=regime_config.get('return_percentile_low', 25.0)
    )
    
    # Test regime detection over time
    regimes = []
    allocations = []
    
    for i in range(100, len(price_data), 10):  # Test every 10 days
        current_data = price_data.iloc[:i+1]
        regime = regime_detector.detect_regime(current_data)
        allocation = regime_detector.get_regime_allocation(regime)
        
        regimes.append(regime)
        allocations.append(allocation)
        
        if i % 200 == 0:
            print(f"Day {i}: Regime = {regime}, Allocation = {allocation:.2f}")
    
    # Analyze results
    regime_counts = pd.Series(regimes).value_counts()
    print(f"\nRegime Distribution:")
    print("=" * 30)
    for regime, count in regime_counts.items():
        percentage = (count / len(regimes)) * 100
        print(f"{regime}: {count} ({percentage:.1f}%)")
    
    # Check if multiple regimes are detected
    unique_regimes = len(regime_counts)
    print(f"\nUnique regimes detected: {unique_regimes}")
    
    if unique_regimes > 1:
        print("✅ SUCCESS: Multiple regimes detected - percentile-based approach is working!")
    else:
        print("❌ ISSUE: Only one regime detected - needs investigation")
    
    # Show allocation distribution
    allocation_series = pd.Series(allocations)
    print(f"\nAllocation Statistics:")
    print("=" * 30)
    print(f"Mean allocation: {allocation_series.mean():.3f}")
    print(f"Std allocation: {allocation_series.std():.3f}")
    print(f"Min allocation: {allocation_series.min():.3f}")
    print(f"Max allocation: {allocation_series.max():.3f}")
    
    return regime_counts, allocation_series

if __name__ == "__main__":
    regime_counts, allocations = test_regime_detection() 