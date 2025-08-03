#!/usr/bin/env python3
"""
Test benchmark data availability to debug regime detection issue
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

def test_benchmark_data():
    """Test benchmark data availability."""
    print("🔍 TESTING BENCHMARK DATA AVAILABILITY")
    print("=" * 50)
    
    # Database connection
    db_manager = get_database_manager()
    engine = db_manager.get_engine()
    
    # Test the same data loading logic as the main script
    start_date = "2016-01-01"
    end_date = "2025-07-28"
    buffer_start_date = pd.Timestamp(start_date) - pd.DateOffset(months=6)
    
    print(f"📊 Data Loading Parameters:")
    print(f"   - Backtest Start: {start_date}")
    print(f"   - Backtest End: {end_date}")
    print(f"   - Buffer Start: {buffer_start_date.date()}")
    print(f"   - Buffer Period: 6 months")
    
    # Load benchmark data
    benchmark_query = text("""
        SELECT date, close
        FROM etf_history
        WHERE ticker = 'VNINDEX' AND date BETWEEN :start_date AND :end_date
        ORDER BY date
    """)
    
    benchmark_data = pd.read_sql(benchmark_query, engine, 
                                params={'start_date': buffer_start_date, 'end_date': end_date}, 
                                parse_dates=['date'])
    
    print(f"\n📈 Benchmark Data Analysis:")
    print(f"   - Total Records: {len(benchmark_data)}")
    print(f"   - Date Range: {benchmark_data['date'].min()} to {benchmark_data['date'].max()}")
    print(f"   - Trading Days: {(benchmark_data['date'].max() - benchmark_data['date'].min()).days}")
    
    # Check data availability for regime detection
    benchmark_returns = benchmark_data.set_index('date')['close'].pct_change().rename('VN-Index')
    
    print(f"\n🔍 Regime Detection Data Check:")
    
    # Test a few specific dates
    test_dates = [
        '2016-01-29',  # First rebalance
        '2016-06-29',  # Should have enough data
        '2017-01-25',  # Should have enough data
        '2020-03-30',  # COVID period
        '2025-05-30'   # Last rebalance
    ]
    
    for test_date in test_dates:
        date = pd.Timestamp(test_date)
        lookback_days = 90
        start_date_lookback = date - pd.Timedelta(days=lookback_days)
        
        # Get data for this period
        period_data = benchmark_returns.loc[start_date_lookback:date]
        
        print(f"\n   📅 {test_date}:")
        print(f"      - Lookback Start: {start_date_lookback.date()}")
        print(f"      - Available Days: {len(period_data)}")
        print(f"      - Sufficient Data: {'✅' if len(period_data) >= lookback_days else '❌'}")
        
        if len(period_data) >= lookback_days:
            # Calculate regime metrics
            volatility = period_data.std()
            avg_return = period_data.mean()
            print(f"      - Volatility: {volatility:.4f} ({volatility:.2%})")
            print(f"      - Avg Return: {avg_return:.4f} ({avg_return:.2%})")
            
            # Check thresholds
            vol_threshold = 0.0140
            ret_threshold = 0.0012
            low_ret_threshold = 0.0002
            
            if volatility > vol_threshold:
                if avg_return > ret_threshold:
                    regime = 'Bull'
                else:
                    regime = 'Bear'
            else:
                if abs(avg_return) < low_ret_threshold:
                    regime = 'Sideways'
                else:
                    regime = 'Stress'
            
            print(f"      - Detected Regime: {regime}")
    
    # Check for data gaps
    print(f"\n🔍 Data Gap Analysis:")
    benchmark_data_sorted = benchmark_data.sort_values('date')
    date_diffs = benchmark_data_sorted['date'].diff().dt.days
    
    gaps = date_diffs[date_diffs > 5]  # Gaps longer than 5 days
    if len(gaps) > 0:
        print(f"   - Found {len(gaps)} data gaps:")
        for gap_date, gap_days in gaps.head(10).items():
            print(f"     {gap_date.date()}: {gap_days} days gap")
    else:
        print(f"   - No significant data gaps found")
    
    print(f"\n✅ Benchmark data test complete!")

if __name__ == "__main__":
    test_benchmark_data() 