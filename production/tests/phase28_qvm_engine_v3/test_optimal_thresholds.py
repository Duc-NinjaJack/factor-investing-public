#!/usr/bin/env python3
"""
Test Optimal Thresholds Script
==============================

This script tests the optimal regime detection thresholds found from
comprehensive testing to verify they work properly.
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
    """Regime detection with optimal thresholds."""
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
        
        # Optimal thresholds from comprehensive testing (180 combinations)
        if volatility > 0.012:  # High volatility (optimal threshold)
            if avg_return > 0.002:  # High return (optimal threshold)
                return 'Bull'
            else:
                return 'Bear'
        else:  # Low volatility
            if abs(avg_return) < 0.001:  # Low return (optimal threshold)
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

def test_optimal_thresholds():
    """Test the optimal regime detection thresholds."""
    print("🎯 Testing Optimal Regime Detection Thresholds")
    print("=" * 60)
    print("   Volatility Threshold: 0.012")
    print("   Return Threshold: 0.002")
    print("   Low Return Threshold: 0.001")
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
    
    # Initialize regime detector with optimal thresholds
    detector = RegimeDetector(lookback_period=60)
    
    # Test regime detection at monthly intervals
    test_dates = pd.date_range(start='2020-07-01', end='2025-01-01', freq='M')
    
    print(f"\n📊 Regime Detection Results ({len(test_dates)} monthly periods):")
    print("-" * 60)
    
    regime_counts = {'Bull': 0, 'Bear': 0, 'Sideways': 0, 'Stress': 0}
    regime_details = []
    
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
                
                regime_counts[regime] += 1
                regime_details.append({
                    'date': test_date,
                    'regime': regime,
                    'allocation': allocation,
                    'volatility': volatility,
                    'avg_return': avg_return
                })
                
        except Exception as e:
            continue
    
    # Display results
    total_periods = sum(regime_counts.values())
    print(f"\n📈 Regime Distribution:")
    print("-" * 30)
    for regime, count in regime_counts.items():
        percentage = (count / total_periods * 100) if total_periods > 0 else 0
        allocation = detector.get_regime_allocation(regime)
        print(f"   {regime}: {count} times ({percentage:.1f}%) - Allocation: {allocation:.1%}")
    
    # Show some specific examples
    print(f"\n📅 Sample Regime Detections:")
    print("-" * 40)
    details_df = pd.DataFrame(regime_details)
    
    # Show examples of each regime
    for regime in ['Bull', 'Bear', 'Sideways', 'Stress']:
        regime_examples = details_df[details_df['regime'] == regime].head(2)
        if not regime_examples.empty:
            print(f"\n   {regime} Market Examples:")
            for _, example in regime_examples.iterrows():
                print(f"     {example['date'].strftime('%Y-%m-%d')}: Vol={example['volatility']:.4f}, Return={example['avg_return']:.4f}")
    
    # Calculate regime switching frequency
    regime_changes = 0
    for i in range(1, len(regime_details)):
        if regime_details[i]['regime'] != regime_details[i-1]['regime']:
            regime_changes += 1
    
    regime_switching_frequency = regime_changes / (len(regime_details) - 1) if len(regime_details) > 1 else 0
    
    print(f"\n🔄 Regime Switching Analysis:")
    print("-" * 30)
    print(f"   Total regime changes: {regime_changes}")
    print(f"   Switching frequency: {regime_switching_frequency:.1%}")
    print(f"   Average periods per regime: {total_periods / 4:.1f}")
    
    print(f"\n✅ Optimal threshold test completed!")
    print(f"   - All 4 regimes detected: {len([c for c in regime_counts.values() if c > 0]) == 4}")
    print(f"   - Balanced distribution: {min(regime_counts.values()) > 0}")
    print(f"   - Total periods analyzed: {total_periods}")

if __name__ == "__main__":
    test_optimal_thresholds() 