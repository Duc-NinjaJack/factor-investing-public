# ============================================================================
# Regime Detection Diagnostic Script
# ============================================================================
# Purpose: Investigate why regime detection is always returning "Sideways"
#          from 2020-2025 in the QVM Engine v3

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
# PHASE 1: DATA VALIDATION
# ============================================================================

def load_benchmark_data():
    """Load VN-Index data for the same period as the notebook."""
    try:
        db_manager = get_database_manager()
        engine = db_manager.get_engine()
        
        # Load benchmark data (VN-Index) for 2020-2025
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
        print(f"   - Total return: {(benchmark_data['close'].iloc[-1] / benchmark_data['close'].iloc[0] - 1):.2%}")
        
        return benchmark_data
        
    except Exception as e:
        print(f"❌ Error loading benchmark data: {e}")
        raise

# ============================================================================
# PHASE 2: REGIME DETECTION ANALYSIS
# ============================================================================

class RegimeDetectorDiagnostic:
    """Enhanced regime detector with diagnostic capabilities."""
    
    def __init__(self, lookback_period: int = 60):
        self.lookback_period = lookback_period
        self.original_thresholds = {
            'volatility_threshold': 0.012,
            'return_threshold': 0.002,
            'low_return_threshold': 0.001
        }
    
    def detect_regime_with_diagnostics(self, price_data: pd.DataFrame, date: str = "Unknown") -> dict:
        """Detect regime with full diagnostic information."""
        if len(price_data) < self.lookback_period:
            return {
                'date': date,
                'regime': 'Sideways',
                'reason': 'Insufficient data',
                'volatility': None,
                'avg_return': None,
                'data_points': len(price_data)
            }
        
        recent_data = price_data.tail(self.lookback_period)
        returns = recent_data['close'].pct_change().dropna()
        
        volatility = returns.std()
        avg_return = returns.mean()
        
        # Original logic
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
            'date': date,
            'regime': regime,
            'reason': reason,
            'volatility': volatility,
            'avg_return': avg_return,
            'data_points': len(returns),
            'volatility_threshold': self.original_thresholds['volatility_threshold'],
            'return_threshold': self.original_thresholds['return_threshold'],
            'low_return_threshold': self.original_thresholds['low_return_threshold']
        }
    
    def test_threshold_combinations(self, price_data: pd.DataFrame, date: str = "Unknown") -> dict:
        """Test different threshold combinations to find what would work."""
        if len(price_data) < self.lookback_period:
            return {'date': date, 'error': 'Insufficient data'}
        
        recent_data = price_data.tail(self.lookback_period)
        returns = recent_data['close'].pct_change().dropna()
        
        volatility = returns.std()
        avg_return = returns.mean()
        
        # Test different threshold combinations
        threshold_tests = []
        
        # Test 1: More lenient thresholds
        vol_thresholds = [0.008, 0.010, 0.012, 0.015, 0.020]
        return_thresholds = [0.001, 0.002, 0.003, 0.005]
        low_return_thresholds = [0.0005, 0.001, 0.002]
        
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
            'date': date,
            'actual_volatility': volatility,
            'actual_avg_return': avg_return,
            'threshold_tests': threshold_tests
        }

# ============================================================================
# PHASE 3: COMPREHENSIVE ANALYSIS
# ============================================================================

def analyze_regime_detection():
    """Comprehensive analysis of regime detection issues."""
    print("\n" + "="*80)
    print("🔍 REGIME DETECTION DIAGNOSTIC ANALYSIS")
    print("="*80)
    
    # Load data
    benchmark_data = load_benchmark_data()
    
    # Generate rebalance dates (monthly)
    rebalance_dates = pd.date_range(
        start='2020-01-01',
        end='2025-07-31',
        freq='M'
    )
    
    # Filter to actual trading dates
    actual_rebalance_dates = []
    for date in rebalance_dates:
        # Find the closest trading date before or on this date
        trading_dates = benchmark_data.index[benchmark_data.index <= date]
        if len(trading_dates) > 0:
            actual_rebalance_dates.append(trading_dates[-1])
    
    print(f"\n📅 Generated {len(actual_rebalance_dates)} rebalance dates")
    
    # Initialize regime detector
    regime_detector = RegimeDetectorDiagnostic(lookback_period=60)
    
    # Analyze each rebalance date
    regime_results = []
    threshold_analysis = []
    
    for i, rebal_date in enumerate(actual_rebalance_dates[:20]):  # Analyze first 20 for efficiency
        print(f"   - Analyzing rebalance {i+1}: {rebal_date.date()}...", end="")
        
        # Get data for regime detection
        start_date = rebal_date - pd.Timedelta(days=60)
        period_data = benchmark_data.loc[start_date:rebal_date]
        
        print(f"     Data points: {len(period_data)}", end="")
        
        if len(period_data) < 30:  # Need at least 30 days
            print(" ⚠️ Insufficient data")
            continue
        
        # Create price series (same as in notebook)
        price_series = (1 + period_data['return']).cumprod()
        price_data = pd.DataFrame({'close': price_series})
        
        print(f", Price series length: {len(price_data)}", end="")
        
        # Detect regime with diagnostics
        regime_result = regime_detector.detect_regime_with_diagnostics(price_data, str(rebal_date.date()))
        regime_results.append(regime_result)
        
        # Test threshold combinations
        threshold_result = regime_detector.test_threshold_combinations(price_data, str(rebal_date.date()))
        threshold_analysis.append(threshold_result)
        
        vol_str = f"{regime_result['volatility']:.4f}" if regime_result['volatility'] is not None else "N/A"
        ret_str = f"{regime_result['avg_return']:.4f}" if regime_result['avg_return'] is not None else "N/A"
        print(f" ✅ {regime_result['regime']} (Vol: {vol_str}, Ret: {ret_str})")
    
    # Analyze results
    print(f"\n📊 REGIME DETECTION RESULTS SUMMARY")
    print("="*50)
    
    regime_df = pd.DataFrame(regime_results)
    if not regime_df.empty:
        regime_counts = regime_df['regime'].value_counts()
        print("\n🔍 Regime Distribution:")
        for regime, count in regime_counts.items():
            percentage = (count / len(regime_df)) * 100
            print(f"   - {regime}: {count} times ({percentage:.1f}%)")
        
        print(f"\n📈 Volatility Statistics:")
        print(f"   - Mean: {regime_df['volatility'].mean():.4f}")
        print(f"   - Std: {regime_df['volatility'].std():.4f}")
        print(f"   - Min: {regime_df['volatility'].min():.4f}")
        print(f"   - Max: {regime_df['volatility'].max():.4f}")
        if 'volatility_threshold' in regime_df.columns:
            print(f"   - Threshold: {regime_df['volatility_threshold'].iloc[0]:.4f}")
        
        print(f"\n📈 Return Statistics:")
        print(f"   - Mean: {regime_df['avg_return'].mean():.4f}")
        print(f"   - Std: {regime_df['avg_return'].std():.4f}")
        print(f"   - Min: {regime_df['avg_return'].min():.4f}")
        print(f"   - Max: {regime_df['avg_return'].max():.4f}")
        if 'return_threshold' in regime_df.columns:
            print(f"   - Threshold: {regime_df['return_threshold'].iloc[0]:.4f}")
        
        # Show detailed breakdown
        print(f"\n🔍 DETAILED BREAKDOWN:")
        for _, row in regime_df.iterrows():
            print(f"   {row['date']}: {row['regime']} - {row['reason']}")
    
    # Analyze threshold combinations
    print(f"\n🔧 THRESHOLD COMBINATION ANALYSIS")
    print("="*50)
    
    if threshold_analysis:
        # Count regimes for different threshold combinations
        all_tests = []
        for analysis in threshold_analysis:
            if 'threshold_tests' in analysis:
                for test in analysis['threshold_tests']:
                    all_tests.append(test)
        
        if all_tests:
            tests_df = pd.DataFrame(all_tests)
            
            # Find combinations that would give more diverse regimes
            regime_counts_by_threshold = tests_df.groupby(['vol_threshold', 'return_threshold', 'low_return_threshold'])['regime'].value_counts().unstack(fill_value=0)
            
            print("\n📊 Regime distribution for different threshold combinations:")
            print(regime_counts_by_threshold.head(10))
    
    return regime_df, threshold_analysis

# ============================================================================
# PHASE 4: VISUALIZATION
# ============================================================================

def create_diagnostic_plots(benchmark_data, regime_results):
    """Create diagnostic plots to visualize the issue."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Regime Detection Diagnostic Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: VN-Index price and volatility
    ax1 = axes[0, 0]
    benchmark_data['close'].plot(ax=ax1, color='blue', alpha=0.7)
    ax1.set_title('VN-Index Price (2020-2025)')
    ax1.set_ylabel('Price')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Rolling volatility
    ax2 = axes[0, 1]
    rolling_vol = benchmark_data['return'].rolling(60).std()
    rolling_vol.plot(ax=ax2, color='red', alpha=0.7)
    ax2.axhline(y=0.012, color='black', linestyle='--', label='Volatility Threshold')
    ax2.set_title('60-Day Rolling Volatility')
    ax2.set_ylabel('Volatility')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Rolling returns
    ax3 = axes[1, 0]
    rolling_ret = benchmark_data['return'].rolling(60).mean()
    rolling_ret.plot(ax=ax3, color='green', alpha=0.7)
    ax3.axhline(y=0.002, color='black', linestyle='--', label='Return Threshold')
    ax3.axhline(y=-0.002, color='black', linestyle='--')
    ax3.set_title('60-Day Rolling Average Return')
    ax3.set_ylabel('Average Return')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Regime distribution
    ax4 = axes[1, 1]
    if len(regime_results) > 0:
        regime_df = pd.DataFrame(regime_results)
        regime_counts = regime_df['regime'].value_counts()
        regime_counts.plot(kind='bar', ax=ax4, color=['red', 'blue', 'green', 'orange'])
        ax4.set_title('Regime Distribution')
        ax4.set_ylabel('Count')
        ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("🚀 Starting Regime Detection Diagnostic...")
    
    try:
        # Load benchmark data
        benchmark_data = load_benchmark_data()
        
        # Run comprehensive analysis
        regime_results, threshold_analysis = analyze_regime_detection()
        
        # Create diagnostic plots
        create_diagnostic_plots(benchmark_data, regime_results)
        
        print("\n✅ Diagnostic analysis complete!")
        print("\n🎯 NEXT STEPS:")
        print("   1. Review the regime distribution above")
        print("   2. Check if thresholds need adjustment")
        print("   3. Verify data quality and calculations")
        print("   4. Implement fixes based on findings")
        
    except Exception as e:
        print(f"❌ Error during diagnostic analysis: {e}")
        raise 