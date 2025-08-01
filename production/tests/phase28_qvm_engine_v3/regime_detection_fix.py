# ============================================================================
# Regime Detection Fix - Addressing Data Insufficiency Issue
# ============================================================================
# Purpose: Fix the regime detection logic that's causing all "Sideways" regimes
#          due to insufficient data points

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
from pathlib import Path
import sys

# Database connectivity
from sqlalchemy import create_engine, text

# Suppress warnings
warnings.filterwarnings('ignore')

# Add project root to path
try:
    current_path = Path.cwd()
    while not (current_path / 'production').is_dir():
        if current_path.parent == current_path:
            raise FileNotFoundError("Could not find the 'production' directory.")
        current_path = current_path.parent
    
    project_root = current_path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    from production.database.connection import get_database_manager
    print(f"✅ Successfully imported production modules.")
except Exception as e:
    print(f"❌ ERROR: {e}")
    raise

# ============================================================================
# PHASE 1: IDENTIFY THE PROBLEM
# ============================================================================

def analyze_data_availability():
    """Analyze why we're getting insufficient data points."""
    print("\n" + "="*80)
    print("🔍 DATA AVAILABILITY ANALYSIS")
    print("="*80)
    
    try:
        db_manager = get_database_manager()
        engine = db_manager.get_engine()
        
        # Load benchmark data
        benchmark_query = text("""
            SELECT date, close
            FROM etf_history
            WHERE ticker = 'VNINDEX' 
            AND date BETWEEN '2019-07-01' AND '2025-07-31'
            ORDER BY date
        """)
        
        benchmark_data = pd.read_sql(benchmark_query, engine, parse_dates=['date'])
        benchmark_data = benchmark_data.set_index('date')
        benchmark_data['return'] = benchmark_data['close'].pct_change()
        
        print(f"✅ Loaded {len(benchmark_data)} benchmark observations")
        print(f"   - Date range: {benchmark_data.index.min()} to {benchmark_data.index.max()}")
        
        # Analyze trading day frequency
        trading_days_per_month = []
        for year in range(2020, 2026):
            for month in range(1, 13):
                start_date = pd.Timestamp(f'{year}-{month:02d}-01')
                # Get the last day of the month
                if month == 12:
                    end_date = pd.Timestamp(f'{year+1}-01-01') - pd.Timedelta(days=1)
                else:
                    end_date = pd.Timestamp(f'{year}-{month+1:02d}-01') - pd.Timedelta(days=1)
                
                month_data = benchmark_data.loc[start_date:end_date]
                if len(month_data) > 0:
                    trading_days_per_month.append(len(month_data))
        
        print(f"\n📊 Trading Days Analysis:")
        print(f"   - Average trading days per month: {np.mean(trading_days_per_month):.1f}")
        print(f"   - Min trading days per month: {np.min(trading_days_per_month)}")
        print(f"   - Max trading days per month: {np.max(trading_days_per_month)}")
        
        # Check 60-day windows
        print(f"\n📅 60-Day Window Analysis:")
        test_dates = pd.date_range('2020-01-01', '2025-07-31', freq='M')
        
        window_sizes = []
        for date in test_dates[:10]:  # Test first 10 dates
            start_date = date - pd.Timedelta(days=60)
            window_data = benchmark_data.loc[start_date:date]
            window_sizes.append(len(window_data))
            print(f"   - {date.date()}: {len(window_data)} trading days in 60-day window")
        
        print(f"   - Average window size: {np.mean(window_sizes):.1f}")
        print(f"   - Problem: Need 60 days but only getting ~{np.mean(window_sizes):.0f} days")
        
        return benchmark_data
        
    except Exception as e:
        print(f"❌ Error in data availability analysis: {e}")
        raise

# ============================================================================
# PHASE 2: FIXED REGIME DETECTOR
# ============================================================================

class FixedRegimeDetector:
    """Fixed regime detector that addresses data insufficiency issues."""
    
    def __init__(self, lookback_days: int = 60, min_data_points: int = 30):
        self.lookback_days = lookback_days
        self.min_data_points = min_data_points
        self.original_thresholds = {
            'volatility_threshold': 0.012,
            'return_threshold': 0.002,
            'low_return_threshold': 0.001
        }
    
    def detect_regime_fixed(self, benchmark_data: pd.Series, analysis_date: pd.Timestamp) -> dict:
        """Fixed regime detection that handles data insufficiency."""
        
        # Method 1: Use calendar days but get more data
        start_date = analysis_date - pd.Timedelta(days=self.lookback_days)
        period_data = benchmark_data.loc[start_date:analysis_date]
        
        # Method 2: If insufficient data, use a longer lookback period
        if len(period_data) < self.min_data_points:
            # Try extending the lookback period
            extended_days = int(self.lookback_days * 1.5)  # 90 days instead of 60
            start_date = analysis_date - pd.Timedelta(days=extended_days)
            period_data = benchmark_data.loc[start_date:analysis_date]
            
            if len(period_data) < self.min_data_points:
                # If still insufficient, use all available data
                start_date = benchmark_data.index[0]
                period_data = benchmark_data.loc[start_date:analysis_date]
        
        # Create price series
        price_series = (1 + period_data).cumprod()
        
        # Calculate metrics
        returns = period_data.dropna()
        if len(returns) < 10:  # Need at least 10 returns
            return {
                'date': str(analysis_date.date()),
                'regime': 'Sideways',
                'reason': 'Insufficient return data',
                'volatility': None,
                'avg_return': None,
                'data_points': len(returns),
                'method': 'insufficient_data'
            }
        
        volatility = returns.std()
        avg_return = returns.mean()
        
        # Apply regime logic
        if volatility > self.original_thresholds['volatility_threshold']:
            if avg_return > self.original_thresholds['return_threshold']:
                regime = 'Bull'
                reason = f"High volatility ({volatility:.4f} > {self.original_thresholds['volatility_threshold']}) and high return ({avg_return:.4f} > {self.original_thresholds['return_threshold']})"
            else:
                regime = 'Bear'
                reason = f"High volatility ({volatility:.4f} > {self.original_thresholds['volatility_threshold']}) but low return ({avg_return:.4f} <= {self.original_thresholds['return_threshold']})"
        else:
            if abs(avg_return) < self.original_thresholds['low_return_threshold']:
                regime = 'Sideways'
                reason = f"Low volatility ({volatility:.4f} <= {self.original_thresholds['volatility_threshold']}) and low return (|{avg_return:.4f}| < {self.original_thresholds['low_return_threshold']})"
            else:
                regime = 'Stress'
                reason = f"Low volatility ({volatility:.4f} <= {self.original_thresholds['volatility_threshold']}) but significant return (|{avg_return:.4f}| >= {self.original_thresholds['low_return_threshold']})"
        
        return {
            'date': str(analysis_date.date()),
            'regime': regime,
            'reason': reason,
            'volatility': volatility,
            'avg_return': avg_return,
            'data_points': len(returns),
            'method': 'standard' if len(period_data) >= self.lookback_days else 'extended'
        }
    
    def test_adjusted_thresholds(self, benchmark_data: pd.Series, analysis_date: pd.Timestamp) -> dict:
        """Test regime detection with adjusted thresholds."""
        
        # Get data using fixed method
        start_date = analysis_date - pd.Timedelta(days=self.lookback_days)
        period_data = benchmark_data.loc[start_date:analysis_date]
        
        if len(period_data) < self.min_data_points:
            extended_days = int(self.lookback_days * 1.5)
            start_date = analysis_date - pd.Timedelta(days=extended_days)
            period_data = benchmark_data.loc[start_date:analysis_date]
        
        returns = period_data.dropna()
        if len(returns) < 10:
            return {'date': str(analysis_date.date()), 'error': 'Insufficient data'}
        
        volatility = returns.std()
        avg_return = returns.mean()
        
        # Test different threshold combinations
        threshold_tests = []
        
        # More reasonable thresholds based on actual VN-Index characteristics
        vol_thresholds = [0.008, 0.010, 0.012, 0.015]
        return_thresholds = [0.0005, 0.001, 0.002, 0.003]
        low_return_thresholds = [0.0003, 0.0005, 0.001]
        
        for vol_thresh in vol_thresholds:
            for ret_thresh in return_thresholds:
                for low_ret_thresh in low_return_thresholds:
                    if volatility > vol_thresh:
                        if avg_return > ret_thresh:
                            regime = 'Bull'
                        else:
                            regime = 'Bear'
                    else:
                        if abs(avg_return) < low_ret_thresh:
                            regime = 'Sideways'
                        else:
                            regime = 'Stress'
                    
                    threshold_tests.append({
                        'vol_threshold': vol_thresh,
                        'return_threshold': ret_thresh,
                        'low_return_threshold': low_ret_thresh,
                        'regime': regime
                    })
        
        return {
            'date': str(analysis_date.date()),
            'actual_volatility': volatility,
            'actual_avg_return': avg_return,
            'threshold_tests': threshold_tests
        }

# ============================================================================
# PHASE 3: COMPREHENSIVE TESTING
# ============================================================================

def test_fixed_regime_detection():
    """Test the fixed regime detection logic."""
    print("\n" + "="*80)
    print("🔧 TESTING FIXED REGIME DETECTION")
    print("="*80)
    
    # Load data
    benchmark_data = analyze_data_availability()
    
    # Generate rebalance dates
    rebalance_dates = pd.date_range('2020-01-01', '2025-07-31', freq='M')
    
    # Filter to actual trading dates
    actual_rebalance_dates = []
    for date in rebalance_dates:
        trading_dates = benchmark_data.index[benchmark_data.index <= date]
        if len(trading_dates) > 0:
            actual_rebalance_dates.append(trading_dates[-1])
    
    print(f"\n📅 Testing {len(actual_rebalance_dates)} rebalance dates")
    
    # Initialize fixed regime detector
    fixed_detector = FixedRegimeDetector(lookback_days=60, min_data_points=30)
    
    # Test each rebalance date
    fixed_results = []
    threshold_analysis = []
    
    for i, rebal_date in enumerate(actual_rebalance_dates[:20]):  # Test first 20
        print(f"   - Testing rebalance {i+1}: {rebal_date.date()}...", end="")
        
        # Test fixed regime detection
        regime_result = fixed_detector.detect_regime_fixed(benchmark_data['return'], rebal_date)
        fixed_results.append(regime_result)
        
        # Test threshold combinations
        threshold_result = fixed_detector.test_adjusted_thresholds(benchmark_data['return'], rebal_date)
        threshold_analysis.append(threshold_result)
        
        vol_str = f"{regime_result['volatility']:.4f}" if regime_result['volatility'] is not None else "N/A"
        ret_str = f"{regime_result['avg_return']:.4f}" if regime_result['avg_return'] is not None else "N/A"
        print(f" ✅ {regime_result['regime']} (Vol: {vol_str}, Ret: {ret_str}, Method: {regime_result['method']})")
    
    # Analyze results
    print(f"\n📊 FIXED REGIME DETECTION RESULTS")
    print("="*50)
    
    fixed_df = pd.DataFrame(fixed_results)
    if not fixed_df.empty:
        regime_counts = fixed_df['regime'].value_counts()
        print("\n🔍 Regime Distribution:")
        for regime, count in regime_counts.items():
            percentage = (count / len(fixed_df)) * 100
            print(f"   - {regime}: {count} times ({percentage:.1f}%)")
        
        print(f"\n📈 Data Quality:")
        method_counts = fixed_df['method'].value_counts()
        for method, count in method_counts.items():
            percentage = (count / len(fixed_df)) * 100
            print(f"   - {method}: {count} times ({percentage:.1f}%)")
        
        print(f"\n📈 Volatility Statistics:")
        valid_vol = fixed_df['volatility'].dropna()
        if len(valid_vol) > 0:
            print(f"   - Mean: {valid_vol.mean():.4f}")
            print(f"   - Std: {valid_vol.std():.4f}")
            print(f"   - Min: {valid_vol.min():.4f}")
            print(f"   - Max: {valid_vol.max():.4f}")
        
        print(f"\n📈 Return Statistics:")
        valid_ret = fixed_df['avg_return'].dropna()
        if len(valid_ret) > 0:
            print(f"   - Mean: {valid_ret.mean():.4f}")
            print(f"   - Std: {valid_ret.std():.4f}")
            print(f"   - Min: {valid_ret.min():.4f}")
            print(f"   - Max: {valid_ret.max():.4f}")
        
        # Show detailed breakdown
        print(f"\n🔍 DETAILED BREAKDOWN:")
        for _, row in fixed_df.iterrows():
            print(f"   {row['date']}: {row['regime']} - {row['reason']}")
    
    return fixed_df, threshold_analysis

# ============================================================================
# PHASE 4: RECOMMENDED FIXES
# ============================================================================

def recommend_fixes():
    """Recommend specific fixes for the regime detection issue."""
    print("\n" + "="*80)
    print("🎯 RECOMMENDED FIXES FOR REGIME DETECTION")
    print("="*80)
    
    print("\n🔧 ISSUE IDENTIFIED:")
    print("   The regime detection is failing because:")
    print("   1. 60-day lookback period requires 60 trading days")
    print("   2. VN-Index only has ~22-23 trading days per month")
    print("   3. 60 calendar days = ~42-45 trading days (insufficient)")
    print("   4. All regimes default to 'Sideways' due to insufficient data")
    
    print("\n💡 RECOMMENDED SOLUTIONS:")
    print("\n   1. ADJUST LOOKBACK PERIOD:")
    print("      - Change from 60 calendar days to 90 calendar days")
    print("      - This provides ~63-69 trading days (sufficient)")
    print("      - Or use 3 months (90 days) instead of 2 months (60 days)")
    
    print("\n   2. ADJUST THRESHOLDS:")
    print("      - Current volatility threshold: 0.012 (1.2% daily)")
    print("      - Suggested: 0.010 (1.0% daily) - more reasonable for VN-Index")
    print("      - Current return threshold: 0.002 (0.2% daily)")
    print("      - Suggested: 0.001 (0.1% daily) - more reasonable")
    
    print("\n   3. IMPLEMENT ADAPTIVE LOOKBACK:")
    print("      - Start with 60 days, extend if insufficient data")
    print("      - Use minimum 30 trading days for calculations")
    print("      - Fall back to longer periods if needed")
    
    print("\n   4. UPDATE QVM ENGINE CONFIGURATION:")
    print("      - Modify QVM_CONFIG['regime']['lookback_period'] from 60 to 90")
    print("      - Adjust volatility_threshold from 0.012 to 0.010")
    print("      - Adjust return_threshold from 0.002 to 0.001")
    
    print("\n✅ EXPECTED OUTCOME:")
    print("   - Diverse regime detection (Bull, Bear, Sideways, Stress)")
    print("   - Better portfolio allocation based on market conditions")
    print("   - Improved strategy performance through regime-aware positioning")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("🚀 Starting Regime Detection Fix Analysis...")
    
    try:
        # Phase 1: Analyze data availability
        benchmark_data = analyze_data_availability()
        
        # Phase 2: Test fixed regime detection
        fixed_results, threshold_analysis = test_fixed_regime_detection()
        
        # Phase 3: Recommend fixes
        recommend_fixes()
        
        print("\n✅ Analysis complete!")
        print("\n🎯 NEXT STEPS:")
        print("   1. Update QVM_CONFIG with recommended settings")
        print("   2. Test the fixed regime detection in the main notebook")
        print("   3. Verify that different regimes are now detected")
        print("   4. Monitor performance improvement")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        raise 