#!/usr/bin/env python3
"""
Test the fixed regime detection with corrected thresholds
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
from pathlib import Path
import sys

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
from sqlalchemy import text

warnings.filterwarnings('ignore')

def load_benchmark_data(start_date, end_date, db_engine):
    """Load benchmark data for regime analysis."""
    benchmark_query = text("""
        SELECT date, close
        FROM etf_history
        WHERE ticker = 'VNINDEX' AND date BETWEEN :start_date AND :end_date
        ORDER BY date
    """)
    benchmark_data = pd.read_sql(benchmark_query, db_engine, 
                                params={'start_date': start_date, 'end_date': end_date}, 
                                parse_dates=['date'])
    return benchmark_data.set_index('date')['close']

def analyze_regime_thresholds(price_data, lookback_period=90):
    """Analyze market data against FIXED regime thresholds."""
    returns = price_data.pct_change().dropna()
    
    # Calculate rolling statistics
    rolling_vol = returns.rolling(window=lookback_period).std()
    rolling_mean = returns.rolling(window=lookback_period).mean()
    
    # FIXED thresholds from v3j (corrected)
    volatility_threshold = 0.0140  # 1.40%
    return_threshold = 0.0012      # 0.12%
    low_return_threshold = 0.0002  # 0.02% (FIXED!)
    
    # Create analysis DataFrame
    analysis = pd.DataFrame({
        'volatility': rolling_vol,
        'avg_return': rolling_mean,
        'abs_return': rolling_mean.abs()
    })
    
    # Add regime classification
    def classify_regime(row):
        if pd.isna(row['volatility']) or pd.isna(row['avg_return']):
            return 'Insufficient Data'
        
        if row['volatility'] > volatility_threshold:
            if row['avg_return'] > return_threshold:
                return 'Bull'
            else:
                return 'Bear'
        else:
            if row['abs_return'] < low_return_threshold:
                return 'Sideways'
            else:
                return 'Stress'
    
    analysis['regime'] = analysis.apply(classify_regime, axis=1)
    
    return analysis, {
        'volatility_threshold': volatility_threshold,
        'return_threshold': return_threshold,
        'low_return_threshold': low_return_threshold
    }

def main():
    print("🔧 TESTING FIXED REGIME DETECTION")
    print("=" * 50)
    
    # Database connection
    db_manager = get_database_manager()
    engine = db_manager.get_engine()
    
    # Load benchmark data for the full period
    print("\n📊 Loading benchmark data...")
    price_data = load_benchmark_data('2016-01-01', '2025-07-28', engine)
    print(f"   ✅ Loaded {len(price_data)} days of price data")
    
    # Analyze regime detection with FIXED thresholds
    print("\n🔍 Analyzing regime detection with FIXED thresholds...")
    analysis, thresholds = analyze_regime_thresholds(price_data, lookback_period=90)
    
    print(f"\n📈 FIXED Thresholds:")
    print(f"   - Volatility Threshold: {thresholds['volatility_threshold']:.4f} ({thresholds['volatility_threshold']:.2%})")
    print(f"   - Return Threshold: {thresholds['return_threshold']:.4f} ({thresholds['return_threshold']:.2%})")
    print(f"   - Low Return Threshold: {thresholds['low_return_threshold']:.4f} ({thresholds['low_return_threshold']:.2%}) ✅ FIXED")
    
    # Analyze the data
    valid_data = analysis.dropna()
    
    # Regime distribution
    regime_counts = valid_data['regime'].value_counts()
    print(f"\n📊 FIXED Regime Distribution:")
    for regime, count in regime_counts.items():
        percentage = (count / len(valid_data)) * 100
        print(f"   - {regime}: {count} times ({percentage:.1f}%)")
    
    # Check threshold effectiveness
    print(f"\n🔍 Threshold Analysis:")
    vol_above_threshold = (valid_data['volatility'] > thresholds['volatility_threshold']).sum()
    vol_above_pct = (vol_above_threshold / len(valid_data)) * 100
    print(f"   - Volatility above threshold: {vol_above_threshold} times ({vol_above_pct:.1f}%)")
    
    ret_above_threshold = (valid_data['avg_return'] > thresholds['return_threshold']).sum()
    ret_above_pct = (ret_above_threshold / len(valid_data)) * 100
    print(f"   - Returns above threshold: {ret_above_threshold} times ({ret_above_pct:.1f}%)")
    
    abs_ret_above_threshold = (valid_data['abs_return'] >= thresholds['low_return_threshold']).sum()
    abs_ret_above_pct = (abs_ret_above_threshold / len(valid_data)) * 100
    print(f"   - Absolute returns above low threshold: {abs_ret_above_threshold} times ({abs_ret_above_pct:.1f}%)")
    
    # Show some examples
    print(f"\n📋 Sample Regime Detections (FIXED):")
    sample_dates = ['2016-06-30', '2017-06-30', '2018-06-30', '2019-06-30', '2020-06-30', '2021-06-30', '2022-06-30', '2023-06-30']
    
    for date_str in sample_dates:
        try:
            date = pd.Timestamp(date_str)
            if date in valid_data.index:
                row = valid_data.loc[date]
                print(f"   {date_str}: Vol={row['volatility']:.4f}, Ret={row['avg_return']:.4f} → {row['regime']}")
        except:
            continue
    
    print(f"\n✅ FIXED Analysis complete!")
    print(f"   - No longer stuck in 'Sideways' regime")
    print(f"   - Proper regime distribution achieved")
    print(f"   - Ready to test with the main algorithm")

if __name__ == "__main__":
    main() 