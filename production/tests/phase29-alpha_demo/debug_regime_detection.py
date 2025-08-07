#!/usr/bin/env python3

# %% [markdown]
# # Debug Regime Detection
# 
# This script analyzes the regime detection logic to understand why it's showing 65% bull regime.

# %%
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add Project Root to Python Path
current_path = Path.cwd()
while not (current_path / 'production').is_dir():
    if current_path.parent == current_path:
        raise FileNotFoundError("Could not find the 'production' directory.")
    current_path = current_path.parent

project_root = current_path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from production.database.connection import get_database_manager

print("✅ Debug script initialized")

# %%
def create_db_connection():
    """Create database connection."""
    try:
        db_manager = get_database_manager()
        engine = db_manager.get_engine()
        print("✅ Database connection established")
        return engine
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        raise

def load_benchmark_data(db_engine):
    """Load benchmark data for regime analysis."""
    print("📊 Loading benchmark data...")
    
    query = """
    SELECT 
        date,
        close as close_price
    FROM etf_history
    WHERE ticker = 'VNINDEX' 
    AND date >= '2016-01-01' AND date <= '2025-12-31'
    ORDER BY date
    """
    
    try:
        data = pd.read_sql(query, db_engine)
        data['date'] = pd.to_datetime(data['date'])
        data = data.sort_values('date')
        data['return'] = data['close_price'].pct_change()
        data['cumulative_return'] = (1 + data['return']).cumprod()
        
        print(f"   ✅ Loaded {len(data)} benchmark records")
        print(f"   📈 Date range: {data['date'].min()} to {data['date'].max()}")
        print(f"   📊 Total return: {(data['cumulative_return'].iloc[-1] - 1)*100:.1f}%")
        print(f"   📊 Average daily return: {data['return'].mean()*100:.3f}%")
        print(f"   📊 Daily volatility: {data['return'].std()*100:.3f}%")
        
        return data
        
    except Exception as e:
        print(f"   ❌ Failed to load benchmark data: {e}")
        return pd.DataFrame()

def analyze_regime_detection(benchmark_data):
    """Analyze the regime detection logic step by step."""
    print("🔍 Analyzing regime detection logic...")
    
    # Current regime detection parameters
    lookback_days = 30
    vol_threshold_pct = 0.75  # 75th percentile for high volatility (conservative)
    return_threshold_pct = 0.25  # 25th percentile for low returns (conservative)
    bull_return_threshold_pct = 0.75  # 75th percentile for high returns (bull regime)
    min_regime_duration = 5
    
    # Calculate rolling metrics
    benchmark_data = benchmark_data.sort_values('date')
    benchmark_data['rolling_vol'] = benchmark_data['return'].rolling(lookback_days).std() * np.sqrt(252)
    benchmark_data['rolling_return'] = benchmark_data['return'].rolling(lookback_days).mean() * 252
    
    # Remove NaN values for analysis
    analysis_data = benchmark_data.dropna().copy()
    
    print(f"   📊 Analysis period: {len(analysis_data)} days")
    print(f"   📊 Rolling volatility stats:")
    print(f"      Mean: {analysis_data['rolling_vol'].mean():.3f}")
    print(f"      Std: {analysis_data['rolling_vol'].std():.3f}")
    print(f"      Min: {analysis_data['rolling_vol'].min():.3f}")
    print(f"      Max: {analysis_data['rolling_vol'].max():.3f}")
    print(f"      {vol_threshold_pct*100}th percentile: {analysis_data['rolling_vol'].quantile(vol_threshold_pct):.3f}")
    
    print(f"   📊 Rolling return stats:")
    print(f"      Mean: {analysis_data['rolling_return'].mean():.3f}")
    print(f"      Std: {analysis_data['rolling_return'].std():.3f}")
    print(f"      Min: {analysis_data['rolling_return'].min():.3f}")
    print(f"      Max: {analysis_data['rolling_return'].max():.3f}")
    print(f"      {return_threshold_pct*100}th percentile: {analysis_data['rolling_return'].quantile(return_threshold_pct):.3f}")
    
    # Define regime thresholds
    vol_threshold = analysis_data['rolling_vol'].quantile(vol_threshold_pct)
    return_threshold = analysis_data['rolling_return'].quantile(return_threshold_pct)
    bull_return_threshold = analysis_data['rolling_return'].quantile(bull_return_threshold_pct)
    
    print(f"   🎯 Regime thresholds:")
    print(f"      Volatility threshold: {vol_threshold:.3f}")
    print(f"      Return threshold (stress): {return_threshold:.3f}")
    print(f"      Bull return threshold: {bull_return_threshold:.3f}")
    
    # Initial regime classification
    analysis_data['regime'] = 'normal'
    analysis_data.loc[
        (analysis_data['rolling_vol'] > vol_threshold) & 
        (analysis_data['rolling_return'] < return_threshold), 'regime'
    ] = 'stress'
    analysis_data.loc[
        (analysis_data['rolling_vol'] < vol_threshold) & 
        (analysis_data['rolling_return'] > bull_return_threshold), 'regime'
    ] = 'bull'
    
    # Analyze initial regime distribution
    print(f"   📊 Initial regime distribution:")
    initial_counts = analysis_data['regime'].value_counts()
    for regime, count in initial_counts.items():
        print(f"      {regime}: {count} days ({count/len(analysis_data)*100:.1f}%)")
    
    # Analyze regime conditions
    print(f"   📊 Regime condition analysis:")
    stress_conditions = analysis_data[
        (analysis_data['rolling_vol'] > vol_threshold) & 
        (analysis_data['rolling_return'] < return_threshold)
    ]
    bull_conditions = analysis_data[
        (analysis_data['rolling_vol'] < vol_threshold) & 
        (analysis_data['rolling_return'] > bull_return_threshold)
    ]
    normal_conditions = analysis_data[
        ~((analysis_data['rolling_vol'] > vol_threshold) & (analysis_data['rolling_return'] < return_threshold)) &
        ~((analysis_data['rolling_vol'] < vol_threshold) & (analysis_data['rolling_return'] > bull_return_threshold))
    ]
    
    print(f"      Stress conditions met: {len(stress_conditions)} days")
    print(f"      Bull conditions met: {len(bull_conditions)} days")
    print(f"      Normal conditions (neither): {len(normal_conditions)} days")
    
    # Check specific periods
    print(f"   📊 Specific period analysis:")
    
    # 2020 COVID crash
    covid_period = analysis_data[
        (analysis_data['date'] >= '2020-02-01') & 
        (analysis_data['date'] <= '2020-04-30')
    ]
    if len(covid_period) > 0:
        print(f"      COVID period (Feb-Apr 2020):")
        print(f"        Regime distribution: {covid_period['regime'].value_counts().to_dict()}")
        print(f"        Avg volatility: {covid_period['rolling_vol'].mean():.3f}")
        print(f"        Avg return: {covid_period['rolling_return'].mean():.3f}")
    
    # 2022 crash
    crash_2022 = analysis_data[
        (analysis_data['date'] >= '2022-01-01') & 
        (analysis_data['date'] <= '2022-12-31')
    ]
    if len(crash_2022) > 0:
        print(f"      2022 period:")
        print(f"        Regime distribution: {crash_2022['regime'].value_counts().to_dict()}")
        print(f"        Avg volatility: {crash_2022['rolling_vol'].mean():.3f}")
        print(f"        Avg return: {crash_2022['rolling_return'].mean():.3f}")
    
    return analysis_data

def plot_regime_analysis(benchmark_data, analysis_data):
    """Plot regime analysis for visual inspection."""
    print("📊 Creating regime analysis plots...")
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    
    # 1. Price and cumulative return
    axes[0].plot(benchmark_data['date'], benchmark_data['cumulative_return'], 
                label='VNINDEX Cumulative Return', linewidth=2, color='blue')
    axes[0].set_title('VNINDEX Cumulative Return', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Cumulative Return')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # 2. Rolling volatility with threshold
    vol_threshold = analysis_data['rolling_vol'].quantile(0.75)
    axes[1].plot(analysis_data['date'], analysis_data['rolling_vol'], 
                label='30-Day Rolling Volatility', linewidth=2, color='red')
    axes[1].axhline(y=vol_threshold, color='red', linestyle='--', alpha=0.7, 
                   label=f'75th percentile threshold: {vol_threshold:.3f}')
    axes[1].set_title('30-Day Rolling Volatility', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Annualized Volatility')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # 3. Rolling return with threshold
    return_threshold = analysis_data['rolling_return'].quantile(0.25)
    axes[2].plot(analysis_data['date'], analysis_data['rolling_return'], 
                label='30-Day Rolling Return', linewidth=2, color='green')
    axes[2].axhline(y=return_threshold, color='green', linestyle='--', alpha=0.7,
                   label=f'25th percentile threshold: {return_threshold:.3f}')
    axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[2].set_title('30-Day Rolling Return', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Annualized Return')
    axes[2].set_xlabel('Date')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig('insights/regime_analysis_debug.png', dpi=300, bbox_inches='tight')
    print("   ✅ Plot saved to insights/regime_analysis_debug.png")
    
    return fig

# %%
def main():
    """Main analysis function."""
    print("🔍 Starting Regime Detection Debug Analysis")
    print("="*60)
    
    try:
        # Create database connection
        db_engine = create_db_connection()
        
        # Load benchmark data
        benchmark_data = load_benchmark_data(db_engine)
        
        if benchmark_data.empty:
            print("❌ Failed to load data")
            return
        
        # Analyze regime detection
        analysis_data = analyze_regime_detection(benchmark_data)
        
        # Create plots
        fig = plot_regime_analysis(benchmark_data, analysis_data)
        
        # Save analysis results
        results_dir = Path("insights")
        results_dir.mkdir(exist_ok=True)
        
        analysis_data.to_csv(results_dir / "regime_analysis_debug.csv", index=False)
        
        print(f"\n✅ Analysis completed and saved to {results_dir}/")
        print(f"📊 Key findings:")
        print(f"   - Total analysis days: {len(analysis_data)}")
        print(f"   - Regime distribution: {analysis_data['regime'].value_counts().to_dict()}")
        print(f"   - Volatility threshold: {analysis_data['rolling_vol'].quantile(0.75):.3f}")
        print(f"   - Return threshold: {analysis_data['rolling_return'].quantile(0.25):.3f}")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 