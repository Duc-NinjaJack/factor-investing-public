#!/usr/bin/env python3
"""
Test Regime Detection Thresholds Script
=======================================

This script tests different threshold levels for regime detection
to find the optimal settings that will properly identify different market regimes.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sqlalchemy import create_engine, text
from itertools import product

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
    """Regime detection with configurable thresholds."""
    def __init__(self, lookback_period: int = 60, 
                 volatility_threshold: float = 0.015,
                 return_threshold: float = 0.005,
                 low_return_threshold: float = 0.002):
        self.lookback_period = lookback_period
        self.volatility_threshold = volatility_threshold
        self.return_threshold = return_threshold
        self.low_return_threshold = low_return_threshold
    
    def detect_regime(self, price_data: pd.DataFrame) -> str:
        """Detect market regime based on volatility and return."""
        if len(price_data) < self.lookback_period:
            return 'Sideways'
        
        recent_data = price_data.tail(self.lookback_period)
        returns = recent_data['close'].pct_change().dropna()
        
        volatility = returns.std()
        avg_return = returns.mean()
        
        if volatility > self.volatility_threshold:  # High volatility
            if avg_return > self.return_threshold:  # High return
                return 'Bull'
            else:
                return 'Bear'
        else:  # Low volatility
            if abs(avg_return) < self.low_return_threshold:  # Low return
                return 'Sideways'
            else:
                return 'Stress'

def test_regime_thresholds():
    """Test different threshold combinations for regime detection."""
    print("🔍 Testing Regime Detection Thresholds")
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
    
    # Test dates (monthly rebalance dates)
    test_dates = pd.date_range(start='2020-07-01', end='2025-01-01', freq='M')
    
    # Threshold combinations to test
    volatility_thresholds = [0.010, 0.012, 0.015, 0.018, 0.020, 0.025]
    return_thresholds = [0.002, 0.003, 0.005, 0.008, 0.010, 0.015]
    low_return_thresholds = [0.001, 0.002, 0.003, 0.005, 0.008]
    
    print(f"\n📊 Testing {len(volatility_thresholds)} × {len(return_thresholds)} × {len(low_return_thresholds)} = {len(volatility_thresholds) * len(return_thresholds) * len(low_return_thresholds)} combinations")
    print(f"📅 Test period: {len(test_dates)} monthly dates from {test_dates[0].date()} to {test_dates[-1].date()}")
    
    # Store results
    results = []
    
    # Test each combination
    for vol_thresh, ret_thresh, low_ret_thresh in product(volatility_thresholds, return_thresholds, low_return_thresholds):
        detector = RegimeDetector(
            lookback_period=60,
            volatility_threshold=vol_thresh,
            return_threshold=ret_thresh,
            low_return_threshold=low_ret_thresh
        )
        
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
                    regime_counts[regime] += 1
                    
            except Exception as e:
                continue
        
        # Calculate metrics
        total_periods = sum(regime_counts.values())
        if total_periods == 0:
            continue
            
        # Calculate diversity score (how well distributed the regimes are)
        regime_percentages = {k: v/total_periods for k, v in regime_counts.items()}
        diversity_score = 1 - max(regime_percentages.values())  # Lower max = higher diversity
        
        # Calculate balance score (how many regimes are detected)
        regimes_detected = sum(1 for count in regime_counts.values() if count > 0)
        balance_score = regimes_detected / 4  # 4 possible regimes
        
        # Overall score
        overall_score = (diversity_score + balance_score) / 2
        
        results.append({
            'volatility_threshold': vol_thresh,
            'return_threshold': ret_thresh,
            'low_return_threshold': low_ret_thresh,
            'bull_count': regime_counts['Bull'],
            'bear_count': regime_counts['Bear'],
            'sideways_count': regime_counts['Sideways'],
            'stress_count': regime_counts['Stress'],
            'total_periods': total_periods,
            'diversity_score': diversity_score,
            'balance_score': balance_score,
            'overall_score': overall_score,
            'regimes_detected': regimes_detected
        })
    
    # Sort by overall score
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('overall_score', ascending=False)
    
    print(f"\n🏆 Top 10 Threshold Combinations:")
    print("=" * 80)
    
    for i, row in results_df.head(10).iterrows():
        print(f"\n#{results_df.index.get_loc(i)+1} - Score: {row['overall_score']:.3f}")
        print(f"   Volatility Threshold: {row['volatility_threshold']:.3f}")
        print(f"   Return Threshold: {row['return_threshold']:.3f}")
        print(f"   Low Return Threshold: {row['low_return_threshold']:.3f}")
        print(f"   Regimes Detected: {row['regimes_detected']}/4")
        print(f"   Distribution: Bull={row['bull_count']}, Bear={row['bear_count']}, Sideways={row['sideways_count']}, Stress={row['stress_count']}")
        print(f"   Percentages: Bull={row['bull_count']/row['total_periods']:.1%}, Bear={row['bear_count']/row['total_periods']:.1%}, Sideways={row['sideways_count']/row['total_periods']:.1%}, Stress={row['stress_count']/row['total_periods']:.1%}")
    
    # Find best combination with all 4 regimes
    best_4_regimes = results_df[results_df['regimes_detected'] == 4]
    if not best_4_regimes.empty:
        best_4 = best_4_regimes.iloc[0]
        print(f"\n🎯 BEST 4-REGIME COMBINATION:")
        print("=" * 50)
        print(f"   Volatility Threshold: {best_4['volatility_threshold']:.3f}")
        print(f"   Return Threshold: {best_4['return_threshold']:.3f}")
        print(f"   Low Return Threshold: {best_4['low_return_threshold']:.3f}")
        print(f"   Overall Score: {best_4['overall_score']:.3f}")
        print(f"   Distribution: Bull={best_4['bull_count']}, Bear={best_4['bear_count']}, Sideways={best_4['sideways_count']}, Stress={best_4['stress_count']}")
    
    # Summary statistics
    print(f"\n📈 Summary Statistics:")
    print("=" * 30)
    print(f"   Total combinations tested: {len(results_df)}")
    print(f"   Combinations with all 4 regimes: {len(results_df[results_df['regimes_detected'] == 4])}")
    print(f"   Average overall score: {results_df['overall_score'].mean():.3f}")
    print(f"   Best overall score: {results_df['overall_score'].max():.3f}")
    
    return results_df

if __name__ == "__main__":
    results = test_regime_thresholds() 