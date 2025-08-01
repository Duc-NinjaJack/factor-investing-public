#!/usr/bin/env python3
"""
Test Regime Detection Script
============================

This script tests the regime detection logic with the adjusted thresholds
to ensure it properly identifies different market regimes.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sqlalchemy import create_engine, text

# Add project root to path
current_path = Path.cwd()
while not (current_path / 'production').is_dir():
    if current_path.parent == current_path:
        raise FileNotFoundError("Could not find the 'production' directory.")
    current_path = current_path.parent

project_root = current_path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from production.database.connection import get_database_manager

class RegimeDetector:
    """
    Simple regime detection based on volatility and return thresholds.
    Based on insights from phase26_regime_analysis.
    """
    def __init__(self, lookback_period: int = 60):
        self.lookback_period = lookback_period
    
    def detect_regime(self, price_data: pd.DataFrame) -> str:
        """Detect market regime based on volatility and return."""
        if len(price_data) < self.lookback_period:
            return 'Sideways'
        
        recent_data = price_data.tail(self.lookback_period)
        returns = recent_data['close'].pct_change().dropna()
        
        volatility = returns.std()
        avg_return = returns.mean()
        
        # Adjusted thresholds for Vietnamese market (more sensitive)
        if volatility > 0.015:  # High volatility (reduced from 0.02)
            if avg_return > 0.005:  # High return (reduced from 0.01)
                return 'Bull'
            else:
                return 'Bear'
        else:  # Low volatility
            if abs(avg_return) < 0.002:  # Low return (reduced from 0.005)
                return 'Sideways'
            else:
                return 'Stress'
    
    def get_regime_allocation(self, regime: str) -> float:
        """Get target allocation based on regime."""
        regime_allocations = {
            'Bull': 1.0,      # Fully invested
            'Bear': 0.8,      # 80% invested
            'Sideways': 0.6,  # 60% invested
            'Stress': 0.4     # 40% invested
        }
        return regime_allocations.get(regime, 0.6)

def test_regime_detection():
    """Test regime detection with real market data."""
    print("🔍 Testing Regime Detection with Adjusted Thresholds")
    print("=" * 60)
    
    # Connect to database
    db_manager = get_database_manager()
    engine = db_manager.get_engine()
    
    # Get VN-Index data for testing
    query = text("""
        SELECT date, close
        FROM etf_history
        WHERE ticker = 'VNINDEX' 
        AND date >= '2020-01-01'
        ORDER BY date
    """)
    
    benchmark_data = pd.read_sql(query, engine, parse_dates=['date'])
    benchmark_data = benchmark_data.set_index('date')
    
    # Calculate returns
    benchmark_data['return'] = benchmark_data['close'].pct_change()
    
    # Initialize regime detector
    detector = RegimeDetector(lookback_period=60)
    
    # Test regime detection at different points
    test_dates = [
        '2020-03-30',  # COVID crash
        '2020-07-30',  # Recovery
        '2021-01-29',  # Bull market
        '2022-03-30',  # Ukraine war
        '2023-01-30',  # Sideways
        '2024-01-30',  # Recent
        '2025-01-30'   # Latest
    ]
    
    print("\n📊 Regime Detection Results:")
    print("-" * 60)
    
    regime_counts = {'Bull': 0, 'Bear': 0, 'Sideways': 0, 'Stress': 0}
    
    for test_date in test_dates:
        try:
            # Get data up to test date
            data_up_to_date = benchmark_data.loc[:test_date]
            
            if len(data_up_to_date) >= 60:
                # Create price series for regime detection
                price_series = (1 + data_up_to_date['return']).cumprod()
                price_data = pd.DataFrame({'close': price_series})
                
                # Detect regime
                regime = detector.detect_regime(price_data)
                allocation = detector.get_regime_allocation(regime)
                
                # Calculate actual metrics for verification
                recent_returns = data_up_to_date['return'].tail(60)
                volatility = recent_returns.std()
                avg_return = recent_returns.mean()
                
                print(f"📅 {test_date}: {regime} (Allocation: {allocation:.1%})")
                print(f"   Volatility: {volatility:.4f}, Avg Return: {avg_return:.4f}")
                
                regime_counts[regime] += 1
            else:
                print(f"📅 {test_date}: Insufficient data")
                
        except Exception as e:
            print(f"📅 {test_date}: Error - {e}")
    
    print("\n📈 Regime Distribution:")
    print("-" * 30)
    total_tests = sum(regime_counts.values())
    for regime, count in regime_counts.items():
        percentage = (count / total_tests * 100) if total_tests > 0 else 0
        print(f"   {regime}: {count} times ({percentage:.1f}%)")
    
    print(f"\n✅ Regime detection test completed!")
    print(f"   - Total test periods: {total_tests}")
    print(f"   - Regimes detected: {sum(1 for count in regime_counts.values() if count > 0)}")

if __name__ == "__main__":
    test_regime_detection() 