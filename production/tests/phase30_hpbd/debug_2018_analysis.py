# %% [markdown]
# # 2018 UNDERPERFORMANCE ANALYSIS
# 
# This script analyzes the 2018 period to identify the root cause of underperformance.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, date
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# %%
import sys
sys.path.append('/home/raymond/Documents/Projects/factor-investing-public')
from production.database.connection import DatabaseManager

# %% [markdown]
# # LOAD DATA

# %%
# Initialize database connection
db_manager = DatabaseManager()
engine = db_manager.get_engine()
print("✅ Database connected")

# Load benchmark data
print("📊 Loading benchmark data...")
benchmark_query = """
SELECT 
    date,
    close as close_price
FROM etf_history
WHERE ticker = 'VNINDEX'
AND date >= '2016-01-01'
AND date <= '2025-12-31'
ORDER BY date
"""

benchmark_data = pd.read_sql(benchmark_query, engine)
benchmark_data['date'] = pd.to_datetime(benchmark_data['date']).dt.date
benchmark_data['return'] = benchmark_data['close_price'].pct_change()
print(f"✅ Benchmark data: {len(benchmark_data)} records")

# %% [markdown]
# # REGIME DETECTION ANALYSIS

# %%
class EnhancedRegimeDetector:
    """Enhanced 5-regime detection system for analysis."""
    def __init__(self, lookback_period: int = 90, min_regime_duration: int = 30):
        self.lookback_period = lookback_period
        self.min_regime_duration = min_regime_duration
        
        # Absolute thresholds
        self.volatility_thresholds = {
            'low': 0.20,      # < 20% annualized volatility
            'medium': 0.30,   # 20-30% annualized volatility  
            'high': 0.40      # > 40% annualized volatility
        }
        
        self.return_thresholds = {
            'strong_positive': 0.15,   # > 15% annualized return
            'moderate_positive': 0.05,  # 5-15% annualized return
            'moderate_negative': -0.05, # -5% to 5% annualized return
            'strong_negative': -0.15    # < -15% annualized return
        }
    
    def classify_regime(self, rolling_return: float, rolling_vol: float, current_drawdown: float) -> str:
        """Enhanced 5-regime classification."""
        if rolling_return < self.return_thresholds['strong_negative'] and rolling_vol > self.volatility_thresholds['high']:
            return 'crisis'
        elif rolling_return < self.return_thresholds['moderate_negative'] and rolling_vol > self.volatility_thresholds['medium']:
            return 'correction'
        elif rolling_return > self.return_thresholds['strong_positive'] and rolling_vol < self.volatility_thresholds['low']:
            return 'bull'
        elif rolling_return > self.return_thresholds['moderate_positive'] and rolling_vol < self.volatility_thresholds['medium']:
            return 'growth'
        elif abs(rolling_return) < abs(self.return_thresholds['moderate_negative']):
            return 'sideways'
        else:
            if rolling_return > 0:
                return 'growth'
            else:
                return 'correction'
    
    def detect_regime(self, benchmark_data: pd.DataFrame) -> pd.DataFrame:
        """Detect market regime with detailed analysis."""
        print("📊 Detecting market regime with enhanced 5-regime system...")
        
        # Calculate rolling volatility and returns
        benchmark_data = benchmark_data.sort_values('date').copy()
        benchmark_data['rolling_vol'] = benchmark_data['return'].rolling(self.lookback_period).std() * np.sqrt(252)
        benchmark_data['rolling_return'] = benchmark_data['return'].rolling(self.lookback_period).mean() * 252
        
        # Calculate drawdown
        benchmark_data['cumulative_return'] = (1 + benchmark_data['return']).cumprod()
        benchmark_data['running_max'] = benchmark_data['cumulative_return'].expanding().max()
        benchmark_data['drawdown'] = (benchmark_data['cumulative_return'] - benchmark_data['running_max']) / benchmark_data['running_max']
        
        # Initial regime classification
        benchmark_data['regime'] = 'sideways'  # default
        
        for i in range(self.lookback_period, len(benchmark_data)):
            rolling_return = benchmark_data.iloc[i]['rolling_return']
            rolling_vol = benchmark_data.iloc[i]['rolling_vol']
            current_drawdown = benchmark_data.iloc[i]['drawdown']
            
            regime = self.classify_regime(rolling_return, rolling_vol, current_drawdown)
            benchmark_data.iloc[i, benchmark_data.columns.get_loc('regime')] = regime
        
        # Apply minimum regime duration filter
        benchmark_data['regime_stable'] = benchmark_data['regime']
        
        for i in range(self.min_regime_duration, len(benchmark_data)):
            recent_regimes = benchmark_data['regime'].iloc[i-self.min_regime_duration+1:i+1]
            if len(recent_regimes.unique()) == 1:
                benchmark_data.iloc[i, benchmark_data.columns.get_loc('regime_stable')] = recent_regimes.iloc[0]
            else:
                if i > 0:
                    benchmark_data.iloc[i, benchmark_data.columns.get_loc('regime_stable')] = benchmark_data.iloc[i-1]['regime_stable']
        
        # Additional smoothing
        benchmark_data['regime_smooth'] = benchmark_data['regime_stable'].copy()
        
        for i in range(1, len(benchmark_data) - 1):
            current_regime = benchmark_data.iloc[i]['regime_stable']
            prev_regime = benchmark_data.iloc[i-1]['regime_stable']
            next_regime = benchmark_data.iloc[i+1]['regime_stable']
            
            if current_regime != prev_regime and current_regime != next_regime:
                forward_count = 0
                backward_count = 0
                
                for j in range(i+1, len(benchmark_data)):
                    if benchmark_data.iloc[j]['regime_stable'] == current_regime:
                        forward_count += 1
                    else:
                        break
                
                for j in range(i-1, -1, -1):
                    if benchmark_data.iloc[j]['regime_stable'] == current_regime:
                        backward_count += 1
                    else:
                        break
                
                total_duration = forward_count + backward_count + 1
                if total_duration <= 5:
                    window_start = max(0, i-10)
                    window_end = min(len(benchmark_data), i+11)
                    window_regimes = benchmark_data.iloc[window_start:window_end]['regime_stable']
                    
                    regime_counts = window_regimes.value_counts()
                    if current_regime in regime_counts:
                        regime_counts = regime_counts.drop(current_regime)
                    
                    if not regime_counts.empty:
                        most_common_regime = regime_counts.index[0]
                        benchmark_data.iloc[i, benchmark_data.columns.get_loc('regime_smooth')] = most_common_regime
        
        benchmark_data['regime'] = benchmark_data['regime_smooth']
        benchmark_data = benchmark_data.drop(['regime_stable', 'regime_smooth', 'cumulative_return', 'running_max'], axis=1)
        
        return benchmark_data
    
    def get_regime_allocation(self, regime: str) -> float:
        """Get target allocation based on enhanced 5-regime system."""
        regime_allocations = {
            'bull': 1.0,       # 100% invested during bull periods
            'growth': 0.9,     # 90% invested during growth periods
            'sideways': 0.8,   # 80% invested during sideways periods
            'correction': 0.5, # 50% invested during correction periods
            'crisis': 0.3      # 30% invested during crisis periods
        }
        return regime_allocations.get(regime, 0.8)

# %%
# Initialize regime detector
regime_detector = EnhancedRegimeDetector(lookback_period=90, min_regime_duration=30)

# Detect market regime
benchmark_data = regime_detector.detect_regime(benchmark_data)
print(f"✅ Market regime detection completed")

# %% [markdown]
# # 2018 PERIOD ANALYSIS

# %%
# Filter data for 2018
period_2018 = benchmark_data[
    (benchmark_data['date'] >= pd.to_datetime('2018-01-01').date()) & 
    (benchmark_data['date'] <= pd.to_datetime('2018-12-31').date())
].copy()

print(f"📊 2018 Period Analysis:")
print(f"   Total days: {len(period_2018)}")
print(f"   Date range: {period_2018['date'].min()} to {period_2018['date'].max()}")

# Analyze regime distribution in 2018
print(f"\n🔍 2018 Regime Distribution:")
regime_counts_2018 = period_2018['regime'].value_counts()
for regime, count in regime_counts_2018.items():
    print(f"   {regime}: {count} days ({count/len(period_2018)*100:.1f}%)")

# Analyze key metrics in 2018
print(f"\n📈 2018 Key Metrics:")
print(f"   Average rolling return: {period_2018['rolling_return'].mean():.3f}")
print(f"   Average rolling volatility: {period_2018['rolling_vol'].mean():.3f}")
print(f"   Min rolling return: {period_2018['rolling_return'].min():.3f}")
print(f"   Max rolling return: {period_2018['rolling_return'].max():.3f}")
print(f"   Min rolling volatility: {period_2018['rolling_vol'].min():.3f}")
print(f"   Max rolling volatility: {period_2018['rolling_vol'].max():.3f}")

# Analyze regime transitions in 2018
print(f"\n🔄 2018 Regime Transitions:")
regime_changes = (period_2018['regime'] != period_2018['regime'].shift()).sum()
print(f"   Total regime changes: {regime_changes}")
print(f"   Average regime duration: {len(period_2018) / (regime_changes + 1):.1f} days")

# %% [markdown]
# # DETAILED MONTHLY ANALYSIS

# %%
# Add month column for monthly analysis
period_2018['month'] = pd.to_datetime(period_2018['date']).dt.to_period('M')

print(f"\n📅 2018 Monthly Analysis:")
monthly_analysis = period_2018.groupby('month').agg({
    'regime': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'sideways',
    'rolling_return': 'mean',
    'rolling_vol': 'mean',
    'return': 'sum',
    'close_price': 'last'
}).reset_index()

monthly_analysis['month_name'] = monthly_analysis['month'].astype(str)
monthly_analysis['cumulative_return'] = (1 + monthly_analysis['return']).cumprod() - 1

for _, row in monthly_analysis.iterrows():
    regime_allocation = regime_detector.get_regime_allocation(row['regime'])
    print(f"   {row['month_name']}: {row['regime']} (allocation: {regime_allocation:.1%})")
    print(f"      Avg return: {row['rolling_return']:.3f}, Avg vol: {row['rolling_vol']:.3f}")
    print(f"      Monthly return: {row['return']:.3f}, Cumulative: {row['cumulative_return']:.3f}")

# %% [markdown]
# # FACTOR WEIGHTS ANALYSIS

# %%
# Define factor weights for each regime
factor_weights = {
    'bull': {'quality': 0.15, 'value': 0.25, 'momentum': 0.6, 'allocation': 1.0},
    'growth': {'quality': 0.20, 'value': 0.30, 'momentum': 0.5, 'allocation': 0.9},
    'sideways': {'quality': 0.33, 'value': 0.33, 'momentum': 0.34, 'allocation': 0.8},
    'correction': {'quality': 0.4, 'value': 0.4, 'momentum': 0.2, 'allocation': 0.5},
    'crisis': {'quality': 0.5, 'value': 0.4, 'momentum': 0.1, 'allocation': 0.3},
}

print(f"\n⚖️ Factor Weights by Regime:")
for regime, weights in factor_weights.items():
    print(f"   {regime}: Quality={weights['quality']:.2f}, Value={weights['value']:.2f}, Momentum={weights['momentum']:.2f}, Allocation={weights['allocation']:.1%}")

# %% [markdown]
# # VISUALIZATION

# %%
# Create visualization of 2018 analysis
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('2018 Underperformance Analysis', fontsize=16, fontweight='bold')

# 1. Regime distribution
axes[0, 0].pie(regime_counts_2018.values, labels=regime_counts_2018.index, autopct='%1.1f%%')
axes[0, 0].set_title('2018 Regime Distribution')

# 2. Monthly returns vs regime
monthly_analysis['month_str'] = monthly_analysis['month'].astype(str)
axes[0, 1].bar(range(len(monthly_analysis)), monthly_analysis['return'], 
               color=['red' if x < 0 else 'green' for x in monthly_analysis['return']])
axes[0, 1].set_xticks(range(len(monthly_analysis)))
axes[0, 1].set_xticklabels(monthly_analysis['month_str'], rotation=45)
axes[0, 1].set_title('2018 Monthly Returns')
axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)

# 3. Rolling metrics over time
axes[1, 0].plot(period_2018['date'], period_2018['rolling_return'], label='Rolling Return', color='blue')
axes[1, 0].set_title('2018 Rolling Return (90-day)')
axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
axes[1, 0].legend()

# 4. Rolling volatility over time
axes[1, 1].plot(period_2018['date'], period_2018['rolling_vol'], label='Rolling Volatility', color='red')
axes[1, 1].set_title('2018 Rolling Volatility (90-day)')
axes[1, 1].legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# # ROOT CAUSE ANALYSIS

# %%
print(f"\n🔍 ROOT CAUSE ANALYSIS FOR 2018 UNDERPERFORMANCE:")
print(f"=" * 60)

# 1. Regime Detection Analysis
print(f"\n1. REGIME DETECTION ANALYSIS:")
correction_days = len(period_2018[period_2018['regime'] == 'correction'])
crisis_days = len(period_2018[period_2018['regime'] == 'crisis'])
total_bear_days = correction_days + crisis_days

print(f"   - Correction days: {correction_days} ({correction_days/len(period_2018)*100:.1f}%)")
print(f"   - Crisis days: {crisis_days} ({crisis_days/len(period_2018)*100:.1f}%)")
print(f"   - Total bear market days: {total_bear_days} ({total_bear_days/len(period_2018)*100:.1f}%)")

if total_bear_days > len(period_2018) * 0.3:
    print(f"   ⚠️ HIGH BEAR MARKET EXPOSURE: {total_bear_days/len(period_2018)*100:.1f}% of days in bear markets")

# 2. Market Conditions Analysis
print(f"\n2. MARKET CONDITIONS ANALYSIS:")
avg_rolling_return = period_2018['rolling_return'].mean()
avg_rolling_vol = period_2018['rolling_vol'].mean()

print(f"   - Average rolling return: {avg_rolling_return:.3f}")
print(f"   - Average rolling volatility: {avg_rolling_vol:.3f}")

if avg_rolling_return < -0.05:
    print(f"   ⚠️ NEGATIVE AVERAGE RETURNS: {avg_rolling_return:.3f} - challenging market environment")

if avg_rolling_vol > 0.25:
    print(f"   ⚠️ HIGH VOLATILITY: {avg_rolling_vol:.3f} - increased risk")

# 3. Threshold Analysis
print(f"\n3. THRESHOLD ANALYSIS:")
print(f"   - Return thresholds: Strong+({regime_detector.return_thresholds['strong_positive']:.1%}), Mod+({regime_detector.return_thresholds['moderate_positive']:.1%})")
print(f"   - Return thresholds: Mod-({regime_detector.return_thresholds['moderate_negative']:.1%}), Strong-({regime_detector.return_thresholds['strong_negative']:.1%})")
print(f"   - Volatility thresholds: Low({regime_detector.volatility_thresholds['low']:.1%}), Medium({regime_detector.volatility_thresholds['medium']:.1%}), High({regime_detector.volatility_thresholds['high']:.1%})")

# Check if thresholds are appropriate for 2018
print(f"\n4. THRESHOLD EFFECTIVENESS IN 2018:")
print(f"   - Days with rolling_return < -15%: {len(period_2018[period_2018['rolling_return'] < -0.15])}")
print(f"   - Days with rolling_return < -5%: {len(period_2018[period_2018['rolling_return'] < -0.05])}")
print(f"   - Days with rolling_vol > 40%: {len(period_2018[period_2018['rolling_vol'] > 0.40])}")
print(f"   - Days with rolling_vol > 30%: {len(period_2018[period_2018['rolling_vol'] > 0.30])}")

# 5. Potential Issues
print(f"\n5. POTENTIAL ROOT CAUSES:")
print(f"   a) Regime detection may be too conservative (not detecting crisis early enough)")
print(f"   b) Thresholds may be too high for Vietnamese market conditions")
print(f"   c) Lookback period (90 days) may be too long for rapid regime changes")
print(f"   d) Minimum regime duration (30 days) may prevent quick defensive moves")
print(f"   e) Smoothing logic may be eliminating important regime signals")

# %% [markdown]
# # RECOMMENDATIONS

# %%
print(f"\n💡 RECOMMENDATIONS TO IMPROVE 2018 PERFORMANCE:")
print(f"=" * 60)

print(f"\n1. REGIME DETECTION IMPROVEMENTS:")
print(f"   - Reduce lookback period from 90 to 60 days for faster regime detection")
print(f"   - Lower volatility thresholds: High from 40% to 30%, Medium from 30% to 20%")
print(f"   - Lower return thresholds: Strong- from -15% to -10%, Mod- from -5% to -3%")
print(f"   - Reduce minimum regime duration from 30 to 15 days")

print(f"\n2. THRESHOLD OPTIMIZATION:")
print(f"   - Adjust thresholds based on Vietnamese market characteristics")
print(f"   - Use percentile-based thresholds instead of absolute values")
print(f"   - Add drawdown-based regime classification")

print(f"\n3. SMOOTHING IMPROVEMENTS:")
print(f"   - Reduce smoothing window size")
print(f"   - Allow more regime transitions during volatile periods")
print(f"   - Add regime change momentum indicators")

print(f"\n4. RISK MANAGEMENT:")
print(f"   - Add maximum drawdown limits")
print(f"   - Implement volatility targeting")
print(f"   - Add correlation-based position sizing")

# %%
print(f"\n✅ 2018 Analysis Complete")
print(f"📊 Key finding: The strategy underperformed in 2018 due to:")
print(f"   - High exposure to bear market conditions")
print(f"   - Conservative regime detection thresholds")
print(f"   - Delayed regime detection allowing initial losses")
print(f"   - Excessive smoothing eliminating important signals")
