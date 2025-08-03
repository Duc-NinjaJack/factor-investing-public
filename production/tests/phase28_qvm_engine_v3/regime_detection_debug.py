#!/usr/bin/env python3
"""
Regime Detection Debug Script

This script tests the regime detection logic to identify why it's stuck on "Sideways".
It includes:
1. Testing with synthetic data
2. Analyzing real VN-Index data distributions
3. Proposing better thresholds
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
import sys
from pathlib import Path

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

# Set up plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class RegimeDetector:
    """
    Simple regime detection based on volatility and return thresholds.
    """
    def __init__(self, lookback_period: int = 90, volatility_threshold: float = 0.2659, 
                 return_threshold: float = 0.2588, low_return_threshold: float = 0.2131):
        self.lookback_period = lookback_period
        self.volatility_threshold = volatility_threshold
        self.return_threshold = return_threshold
        self.low_return_threshold = low_return_threshold
        print(f"✅ RegimeDetector initialized with thresholds:")
        print(f"   - Volatility: {self.volatility_threshold:.2%}")
        print(f"   - Return: {self.return_threshold:.2%}")
        print(f"   - Low Return: {self.low_return_threshold:.2%}")
    
    def detect_regime(self, price_data: pd.DataFrame) -> str:
        """Detect market regime based on volatility and return."""
        if len(price_data) < self.lookback_period:
            return 'Sideways'
        
        recent_data = price_data.tail(self.lookback_period)
        returns = recent_data['close'].pct_change().dropna()
        
        volatility = returns.std()
        avg_return = returns.mean()
        
        # Debug output
        print(f"   🔍 Regime Debug: Vol={volatility:.2%}, AvgRet={avg_return:.2%}")
        
        if volatility > self.volatility_threshold:
            if avg_return > self.return_threshold:
                print(f"   📈 Detected: Bull (Vol={volatility:.2%} > {self.volatility_threshold:.2%}, Ret={avg_return:.2%} > {self.return_threshold:.2%})")
                return 'Bull'
            else:
                print(f"   📉 Detected: Bear (Vol={volatility:.2%} > {self.volatility_threshold:.2%}, Ret={avg_return:.2%} <= {self.return_threshold:.2%})")
                return 'Bear'
        else:
            if abs(avg_return) < self.low_return_threshold:
                print(f"   ↔️  Detected: Sideways (Vol={volatility:.2%} <= {self.volatility_threshold:.2%}, |Ret|={abs(avg_return):.2%} < {self.low_return_threshold:.2%})")
                return 'Sideways'
            else:
                print(f"   ⚠️  Detected: Stress (Vol={volatility:.2%} <= {self.volatility_threshold:.2%}, |Ret|={abs(avg_return):.2%} >= {self.low_return_threshold:.2%})")
                return 'Stress'

def generate_synthetic_data(regime_type: str, days: int = 90) -> pd.DataFrame:
    """Generate synthetic price data for testing regime detection."""
    np.random.seed(42)
    
    if regime_type == 'Bull':
        # High volatility, high positive returns
        returns = np.random.normal(0.002, 0.03, days)  # 0.2% daily return, 3% daily vol
    elif regime_type == 'Bear':
        # High volatility, negative returns
        returns = np.random.normal(-0.001, 0.025, days)  # -0.1% daily return, 2.5% daily vol
    elif regime_type == 'Sideways':
        # Low volatility, low returns
        returns = np.random.normal(0.0001, 0.01, days)  # 0.01% daily return, 1% daily vol
    elif regime_type == 'Stress':
        # Low volatility, high absolute returns (could be positive or negative)
        returns = np.random.normal(0.001, 0.015, days)  # 0.1% daily return, 1.5% daily vol
    else:
        raise ValueError(f"Unknown regime type: {regime_type}")
    
    # Generate price series
    prices = [100]  # Start at 100
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    dates = pd.date_range(start='2020-01-01', periods=len(prices), freq='D')
    return pd.DataFrame({'close': prices}, index=dates)

def test_synthetic_data():
    """Test regime detection with synthetic data."""
    print("="*80)
    print("🧪 PHASE 1: TESTING WITH SYNTHETIC DATA")
    print("="*80)
    
    # Test with current thresholds
    detector = RegimeDetector(
        lookback_period=90,
        volatility_threshold=0.2659,
        return_threshold=0.2588,
        low_return_threshold=0.2131
    )
    
    regimes = ['Bull', 'Bear', 'Sideways', 'Stress']
    
    for regime in regimes:
        print(f"\n📊 Testing {regime} regime:")
        data = generate_synthetic_data(regime)
        detected = detector.detect_regime(data)
        print(f"   Expected: {regime}, Detected: {detected}")
        
        # Calculate actual metrics
        returns = data['close'].pct_change().dropna()
        vol = returns.std()
        avg_ret = returns.mean()
        print(f"   Actual - Vol: {vol:.2%}, AvgRet: {avg_ret:.2%}")

def analyze_real_data():
    """Analyze real VN-Index data to understand actual distributions."""
    print("\n" + "="*80)
    print("📈 PHASE 2: ANALYZING REAL VN-INDEX DATA")
    print("="*80)
    
    # Connect to database
    try:
        db_manager = get_database_manager()
        engine = db_manager.get_engine()
        print("✅ Database connection established.")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return
    
    # Load VN-Index data
    query = text("""
        SELECT date, close
        FROM etf_history
        WHERE ticker = 'VNINDEX' 
        AND date >= '2016-01-01'
        ORDER BY date
    """)
    
    try:
        data = pd.read_sql(query, engine, parse_dates=['date'])
        data = data.set_index('date')
        print(f"✅ Loaded {len(data)} days of VN-Index data")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return
    
    # Calculate rolling metrics
    lookback_days = 90
    data['returns'] = data['close'].pct_change()
    data['rolling_vol'] = data['returns'].rolling(lookback_days).std()
    data['rolling_avg_return'] = data['returns'].rolling(lookback_days).mean()
    
    # Remove NaN values
    data_clean = data.dropna()
    
    print(f"\n📊 VN-Index Data Analysis (90-day rolling windows):")
    print(f"   - Total periods: {len(data_clean)}")
    print(f"   - Volatility range: {data_clean['rolling_vol'].min():.2%} to {data_clean['rolling_vol'].max():.2%}")
    print(f"   - Return range: {data_clean['rolling_avg_return'].min():.2%} to {data_clean['rolling_avg_return'].max():.2%}")
    
    # Calculate percentiles
    vol_percentiles = data_clean['rolling_vol'].quantile([0.25, 0.5, 0.75, 0.9, 0.95])
    ret_percentiles = data_clean['rolling_avg_return'].quantile([0.25, 0.5, 0.75, 0.9, 0.95])
    abs_ret_percentiles = data_clean['rolling_avg_return'].abs().quantile([0.25, 0.5, 0.75, 0.9, 0.95])
    
    print(f"\n📈 Volatility Percentiles:")
    for p, val in vol_percentiles.items():
        print(f"   - {p*100:.0f}th: {val:.2%}")
    
    print(f"\n📈 Return Percentiles:")
    for p, val in ret_percentiles.items():
        print(f"   - {p*100:.0f}th: {val:.2%}")
    
    print(f"\n📈 Absolute Return Percentiles:")
    for p, val in abs_ret_percentiles.items():
        print(f"   - {p*100:.0f}th: {val:.2%}")
    
    # Test current thresholds
    print(f"\n🔍 Testing Current Thresholds:")
    print(f"   - Current vol threshold: {0.2659:.2%}")
    print(f"   - Current return threshold: {0.2588:.2%}")
    print(f"   - Current low return threshold: {0.2131:.2%}")
    
    # Count how many periods would be classified as each regime
    bull_count = ((data_clean['rolling_vol'] > 0.2659) & (data_clean['rolling_avg_return'] > 0.2588)).sum()
    bear_count = ((data_clean['rolling_vol'] > 0.2659) & (data_clean['rolling_avg_return'] <= 0.2588)).sum()
    sideways_count = ((data_clean['rolling_vol'] <= 0.2659) & (data_clean['rolling_avg_return'].abs() < 0.2131)).sum()
    stress_count = ((data_clean['rolling_vol'] <= 0.2659) & (data_clean['rolling_avg_return'].abs() >= 0.2131)).sum()
    
    total = len(data_clean)
    print(f"\n📊 Regime Distribution with Current Thresholds:")
    print(f"   - Bull: {bull_count} ({bull_count/total:.1%})")
    print(f"   - Bear: {bear_count} ({bear_count/total:.1%})")
    print(f"   - Sideways: {sideways_count} ({sideways_count/total:.1%})")
    print(f"   - Stress: {stress_count} ({stress_count/total:.1%})")
    
    # Propose better thresholds
    print(f"\n💡 Proposed Better Thresholds:")
    proposed_vol_threshold = vol_percentiles[0.75]  # 75th percentile
    proposed_ret_threshold = ret_percentiles[0.75]  # 75th percentile
    proposed_low_ret_threshold = abs_ret_percentiles[0.25]  # 25th percentile
    
    print(f"   - Volatility threshold: {proposed_vol_threshold:.2%} (75th percentile)")
    print(f"   - Return threshold: {proposed_ret_threshold:.2%} (75th percentile)")
    print(f"   - Low return threshold: {proposed_low_ret_threshold:.2%} (25th percentile)")
    
    # Test proposed thresholds
    bull_count_new = ((data_clean['rolling_vol'] > proposed_vol_threshold) & (data_clean['rolling_avg_return'] > proposed_ret_threshold)).sum()
    bear_count_new = ((data_clean['rolling_vol'] > proposed_vol_threshold) & (data_clean['rolling_avg_return'] <= proposed_ret_threshold)).sum()
    sideways_count_new = ((data_clean['rolling_vol'] <= proposed_vol_threshold) & (data_clean['rolling_avg_return'].abs() < proposed_low_ret_threshold)).sum()
    stress_count_new = ((data_clean['rolling_vol'] <= proposed_vol_threshold) & (data_clean['rolling_avg_return'].abs() >= proposed_low_ret_threshold)).sum()
    
    print(f"\n📊 Regime Distribution with Proposed Thresholds:")
    print(f"   - Bull: {bull_count_new} ({bull_count_new/total:.1%})")
    print(f"   - Bear: {bear_count_new} ({bear_count_new/total:.1%})")
    print(f"   - Sideways: {sideways_count_new} ({sideways_count_new/total:.1%})")
    print(f"   - Stress: {stress_count_new} ({stress_count_new/total:.1%})")
    
    return {
        'current_thresholds': {
            'volatility': 0.2659,
            'return': 0.2588,
            'low_return': 0.2131
        },
        'proposed_thresholds': {
            'volatility': proposed_vol_threshold,
            'return': proposed_ret_threshold,
            'low_return': proposed_low_ret_threshold
        },
        'data': data_clean
    }

def test_proposed_thresholds(analysis_results):
    """Test the proposed thresholds on real data."""
    print("\n" + "="*80)
    print("🧪 PHASE 3: TESTING PROPOSED THRESHOLDS")
    print("="*80)
    
    if not analysis_results:
        print("❌ No analysis results available")
        return
    
    data = analysis_results['data']
    proposed = analysis_results['proposed_thresholds']
    
    # Create detector with proposed thresholds
    detector = RegimeDetector(
        lookback_period=90,
        volatility_threshold=proposed['volatility'],
        return_threshold=proposed['return'],
        low_return_threshold=proposed['low_return']
    )
    
    # Test on a few sample periods
    sample_dates = data.index[::len(data)//10]  # Every 10th date
    
    print(f"\n📊 Testing on {len(sample_dates)} sample periods:")
    regime_counts = {'Bull': 0, 'Bear': 0, 'Sideways': 0, 'Stress': 0}
    
    for date in sample_dates:
        # Get data up to this date
        data_subset = data.loc[:date]
        if len(data_subset) >= 90:
            # Create price series for regime detection
            price_data = pd.DataFrame({'close': data_subset['close']})
            regime = detector.detect_regime(price_data)
            regime_counts[regime] += 1
            print(f"   {date.date()}: {regime}")
    
    print(f"\n📊 Sample Regime Distribution:")
    for regime, count in regime_counts.items():
        print(f"   - {regime}: {count}")

def main():
    """Main execution function."""
    print("🚀 REGIME DETECTION DEBUG SCRIPT")
    print("="*80)
    
    # Phase 1: Test with synthetic data
    test_synthetic_data()
    
    # Phase 2: Analyze real data
    analysis_results = analyze_real_data()
    
    # Phase 3: Test proposed thresholds
    test_proposed_thresholds(analysis_results)
    
    print("\n" + "="*80)
    print("✅ DEBUG COMPLETE")
    print("="*80)
    
    if analysis_results:
        print(f"\n📋 SUMMARY:")
        print(f"   - Current thresholds are too high for 90-day periods")
        print(f"   - Proposed volatility threshold: {analysis_results['proposed_thresholds']['volatility']:.2%}")
        print(f"   - Proposed return threshold: {analysis_results['proposed_thresholds']['return']:.2%}")
        print(f"   - Proposed low return threshold: {analysis_results['proposed_thresholds']['low_return']:.2%}")
        print(f"\n💡 Next steps:")
        print(f"   1. Update the notebook configuration with proposed thresholds")
        print(f"   2. Test the full backtest with new thresholds")
        print(f"   3. Verify regime variety in the results")

if __name__ == "__main__":
    main() 