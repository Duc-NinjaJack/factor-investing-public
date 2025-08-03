# Regime Threshold Analysis for Vietnamese Market
# Investigating what 26.59% volatility threshold means and market characteristics

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
import sys
from pathlib import Path
current_path = Path.cwd()
while not (current_path / 'production').is_dir():
    current_path = current_path.parent
project_root = current_path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from production.database.connection import get_database_manager

def create_db_connection():
    """Establishes a SQLAlchemy database engine connection."""
    try:
        db_manager = get_database_manager()
        engine = db_manager.get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print(f"✅ Database connection established successfully.")
        return engine
    except Exception as e:
        print(f"❌ FAILED to connect to the database: {e}")
        return None

def load_vnindex_data(engine, start_date, end_date):
    """Load VN-Index data for analysis."""
    query = text("""
        SELECT date, close
        FROM etf_history
        WHERE ticker = 'VNINDEX' 
        AND date BETWEEN :start_date AND :end_date
        ORDER BY date
    """)
    
    data = pd.read_sql(query, engine, 
                       params={'start_date': start_date, 'end_date': end_date},
                       parse_dates=['date'])
    data.set_index('date', inplace=True)
    return data

def calculate_volatility_statistics(data, window=90):
    """Calculate volatility statistics for different periods."""
    print(f"\n📊 Calculating volatility statistics with {window}-day rolling window...")
    
    # Calculate daily returns
    data['returns'] = data['close'].pct_change()
    
    # Calculate rolling volatility (annualized)
    data['volatility_90d'] = data['returns'].rolling(window=window).std() * np.sqrt(252)
    
    # Calculate rolling average returns (annualized)
    data['avg_return_90d'] = data['returns'].rolling(window=window).mean() * 252
    
    # Remove NaN values
    data_clean = data.dropna()
    
    print(f"   - Total observations: {len(data_clean):,}")
    print(f"   - Date range: {data_clean.index.min().date()} to {data_clean.index.max().date()}")
    
    return data_clean

def analyze_volatility_quintiles(data):
    """Analyze volatility quintiles and their characteristics."""
    print(f"\n🎯 VOLATILITY QUINTILE ANALYSIS")
    print("="*50)
    
    # Calculate quintiles
    volatility_quintiles = pd.qcut(data['volatility_90d'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    data['vol_quintile'] = volatility_quintiles
    
    # Analyze each quintile
    quintile_stats = []
    for quintile in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
        quintile_data = data[data['vol_quintile'] == quintile]
        
        stats = {
            'Quintile': quintile,
            'Count': len(quintile_data),
            'Volatility_Mean': quintile_data['volatility_90d'].mean(),
            'Volatility_Std': quintile_data['volatility_90d'].std(),
            'Volatility_Min': quintile_data['volatility_90d'].min(),
            'Volatility_Max': quintile_data['volatility_90d'].max(),
            'Return_Mean': quintile_data['avg_return_90d'].mean(),
            'Return_Std': quintile_data['avg_return_90d'].std(),
            'Period_Start': quintile_data.index.min(),
            'Period_End': quintile_data.index.max()
        }
        quintile_stats.append(stats)
    
    quintile_df = pd.DataFrame(quintile_stats)
    
    print("📈 Volatility Quintile Statistics:")
    print(quintile_df.round(4).to_string(index=False))
    
    return quintile_df, data

def analyze_percentile_thresholds(data):
    """Analyze what different percentile thresholds mean."""
    print(f"\n📊 PERCENTILE THRESHOLD ANALYSIS")
    print("="*50)
    
    percentiles = [25, 50, 75, 90, 95, 99]
    threshold_stats = []
    
    for p in percentiles:
        vol_threshold = np.percentile(data['volatility_90d'], p)
        return_threshold = np.percentile(data['avg_return_90d'], p)
        low_return_threshold = np.percentile(np.abs(data['avg_return_90d']), p)
        
        # Count observations above/below thresholds
        high_vol_count = len(data[data['volatility_90d'] > vol_threshold])
        high_return_count = len(data[data['avg_return_90d'] > return_threshold])
        high_abs_return_count = len(data[np.abs(data['avg_return_90d']) > low_return_threshold])
        
        stats = {
            'Percentile': p,
            'Volatility_Threshold': vol_threshold,
            'Return_Threshold': return_threshold,
            'LowReturn_Threshold': low_return_threshold,
            'High_Vol_Count': high_vol_count,
            'High_Vol_Pct': high_vol_count / len(data) * 100,
            'High_Return_Count': high_return_count,
            'High_Return_Pct': high_return_count / len(data) * 100,
            'High_AbsReturn_Count': high_abs_return_count,
            'High_AbsReturn_Pct': high_abs_return_count / len(data) * 100
        }
        threshold_stats.append(stats)
    
    threshold_df = pd.DataFrame(threshold_stats)
    
    print("🎯 Percentile Threshold Analysis:")
    print(threshold_df.round(4).to_string(index=False))
    
    return threshold_df

def identify_turbulent_periods(data, vol_threshold_pct=75):
    """Identify turbulent and quiet periods in the market."""
    print(f"\n🌪️ TURBULENT vs QUIET PERIODS ANALYSIS")
    print("="*50)
    
    vol_threshold = np.percentile(data['volatility_90d'], vol_threshold_pct)
    return_threshold = np.percentile(data['avg_return_90d'], vol_threshold_pct)
    
    print(f"   - Volatility threshold ({vol_threshold_pct}th percentile): {vol_threshold:.2%}")
    print(f"   - Return threshold ({vol_threshold_pct}th percentile): {return_threshold:.2%}")
    
    # Classify periods
    data['period_type'] = 'Normal'
    data.loc[data['volatility_90d'] > vol_threshold, 'period_type'] = 'Turbulent'
    data.loc[data['volatility_90d'] < np.percentile(data['volatility_90d'], 25), 'period_type'] = 'Quiet'
    
    # Analyze each period type
    period_stats = []
    for period_type in ['Quiet', 'Normal', 'Turbulent']:
        period_data = data[data['period_type'] == period_type]
        
        if len(period_data) > 0:
            stats = {
                'Period_Type': period_type,
                'Count': len(period_data),
                'Percentage': len(period_data) / len(data) * 100,
                'Volatility_Mean': period_data['volatility_90d'].mean(),
                'Volatility_Std': period_data['volatility_90d'].std(),
                'Return_Mean': period_data['avg_return_90d'].mean(),
                'Return_Std': period_data['avg_return_90d'].std(),
                'Date_Range': f"{period_data.index.min().date()} to {period_data.index.max().date()}"
            }
            period_stats.append(stats)
    
    period_df = pd.DataFrame(period_stats)
    
    print("📊 Period Type Analysis:")
    print(period_df.round(4).to_string(index=False))
    
    return period_df, data

def analyze_specific_periods(data):
    """Analyze specific known turbulent and quiet periods."""
    print(f"\n📅 SPECIFIC PERIOD ANALYSIS")
    print("="*50)
    
    # Define known periods
    periods = {
        'COVID_Crash_2020': ('2020-02-01', '2020-04-30'),
        'COVID_Recovery_2020': ('2020-05-01', '2020-12-31'),
        'Bull_Market_2021': ('2021-01-01', '2021-12-31'),
        'Bear_Market_2022': ('2022-01-01', '2022-12-31'),
        'Recovery_2023': ('2023-01-01', '2023-12-31'),
        'Stable_2024': ('2024-01-01', '2024-12-31')
    }
    
    period_analysis = []
    for period_name, (start_date, end_date) in periods.items():
        try:
            period_data = data[(data.index >= start_date) & (data.index <= end_date)]
            
            if len(period_data) > 0:
                stats = {
                    'Period': period_name,
                    'Start_Date': start_date,
                    'End_Date': end_date,
                    'Days': len(period_data),
                    'Volatility_Mean': period_data['volatility_90d'].mean(),
                    'Volatility_Std': period_data['volatility_90d'].std(),
                    'Return_Mean': period_data['avg_return_90d'].mean(),
                    'Return_Std': period_data['avg_return_90d'].std(),
                    'Max_Volatility': period_data['volatility_90d'].max(),
                    'Min_Volatility': period_data['volatility_90d'].min()
                }
                period_analysis.append(stats)
        except Exception as e:
            print(f"   ⚠️ Error analyzing {period_name}: {e}")
    
    period_df = pd.DataFrame(period_analysis)
    
    print("📈 Specific Period Analysis:")
    print(period_df.round(4).to_string(index=False))
    
    return period_df

def create_visualizations(data, quintile_df, threshold_df, period_df):
    """Create comprehensive visualizations."""
    print(f"\n📊 Creating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Vietnamese Market Regime Analysis', fontsize=16, fontweight='bold')
    
    # 1. Volatility over time with quintiles
    ax1 = axes[0, 0]
    colors = ['#2E8B57', '#3CB371', '#FFD700', '#FF8C00', '#DC143C']
    for i, quintile in enumerate(['Q1', 'Q2', 'Q3', 'Q4', 'Q5']):
        quintile_data = data[data['vol_quintile'] == quintile]
        ax1.scatter(quintile_data.index, quintile_data['volatility_90d'], 
                   c=colors[i], alpha=0.6, s=20, label=f'{quintile} (Vol: {quintile_data["volatility_90d"].mean():.1%})')
    ax1.set_title('Volatility Over Time by Quintile')
    ax1.set_ylabel('90-Day Rolling Volatility (Annualized)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Volatility distribution
    ax2 = axes[0, 1]
    data['volatility_90d'].hist(bins=50, alpha=0.7, ax=ax2, color='skyblue', edgecolor='black')
    ax2.axvline(data['volatility_90d'].mean(), color='red', linestyle='--', label=f'Mean: {data["volatility_90d"].mean():.1%}')
    ax2.axvline(np.percentile(data['volatility_90d'], 75), color='orange', linestyle='--', 
                label=f'75th: {np.percentile(data["volatility_90d"], 75):.1%}')
    ax2.axvline(0.2659, color='purple', linestyle='--', label=f'Fixed: 26.59%')
    ax2.set_title('Volatility Distribution')
    ax2.set_xlabel('90-Day Rolling Volatility')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Volatility vs Returns scatter
    ax3 = axes[1, 0]
    scatter = ax3.scatter(data['volatility_90d'], data['avg_return_90d'], 
                         c=data['volatility_90d'], cmap='viridis', alpha=0.6, s=20)
    ax3.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax3.axvline(0.2659, color='red', linestyle='--', label='Fixed Vol Threshold')
    ax3.set_title('Volatility vs Returns')
    ax3.set_xlabel('90-Day Rolling Volatility')
    ax3.set_ylabel('90-Day Rolling Returns (Annualized)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax3, label='Volatility')
    
    # 4. Period type distribution
    ax4 = axes[1, 1]
    period_counts = data['period_type'].value_counts()
    colors_period = ['#90EE90', '#FFD700', '#FF6B6B']
    ax4.pie(period_counts.values, labels=period_counts.index, autopct='%1.1f%%', 
            colors=colors_period, startangle=90)
    ax4.set_title('Market Period Distribution')
    
    plt.tight_layout()
    plt.savefig('regime_analysis_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("   ✅ Visualizations saved as 'regime_analysis_visualization.png'")

def main():
    """Main analysis function."""
    print("🔍 VIETNAMESE MARKET REGIME THRESHOLD ANALYSIS")
    print("="*60)
    
    # Connect to database
    engine = create_db_connection()
    if engine is None:
        return
    
    # Load data for different periods
    periods = {
        '2016-2020': ('2016-01-01', '2020-12-31'),
        '2020-2025': ('2020-01-01', '2025-07-28'),
        'Full_Period': ('2016-01-01', '2025-07-28')
    }
    
    all_results = {}
    
    for period_name, (start_date, end_date) in periods.items():
        print(f"\n{'='*20} {period_name} PERIOD {'='*20}")
        
        # Load data
        data = load_vnindex_data(engine, start_date, end_date)
        if data.empty:
            print(f"   ⚠️ No data available for {period_name}")
            continue
        
        # Calculate statistics
        data_clean = calculate_volatility_statistics(data, window=90)
        
        # Analyze quintiles
        quintile_df, data_quintiles = analyze_volatility_quintiles(data_clean)
        
        # Analyze percentiles
        threshold_df = analyze_percentile_thresholds(data_clean)
        
        # Analyze periods
        period_df, data_periods = identify_turbulent_periods(data_clean)
        
        # Analyze specific periods
        specific_periods_df = analyze_specific_periods(data_clean)
        
        # Store results
        all_results[period_name] = {
            'data': data_clean,
            'quintiles': quintile_df,
            'thresholds': threshold_df,
            'periods': period_df,
            'specific_periods': specific_periods_df
        }
    
    # Create visualizations for the full period
    if 'Full_Period' in all_results:
        create_visualizations(
            all_results['Full_Period']['data'],
            all_results['Full_Period']['quintiles'],
            all_results['Full_Period']['thresholds'],
            all_results['Full_Period']['periods']
        )
    
    # Summary analysis
    print(f"\n{'='*60}")
    print("📋 SUMMARY ANALYSIS")
    print("="*60)
    
    print("\n🎯 What does 26.59% volatility threshold mean?")
    print("-" * 50)
    
    for period_name, results in all_results.items():
        vol_75th = np.percentile(results['data']['volatility_90d'], 75)
        vol_90th = np.percentile(results['data']['volatility_90d'], 90)
        vol_95th = np.percentile(results['data']['volatility_90d'], 95)
        
        print(f"\n{period_name}:")
        print(f"   - 75th percentile: {vol_75th:.2%}")
        print(f"   - 90th percentile: {vol_90th:.2%}")
        print(f"   - 95th percentile: {vol_95th:.2%}")
        print(f"   - Fixed threshold (26.59%): {'ABOVE' if 0.2659 > vol_95th else 'BELOW'} 95th percentile")
    
    print(f"\n💡 RECOMMENDATIONS:")
    print("-" * 50)
    print("1. The 26.59% threshold was calibrated for 2016-2020 (high volatility period)")
    print("2. For 2020-2025, this threshold is too high - market is much less volatile")
    print("3. Consider period-specific thresholds or adaptive percentile-based approach")
    print("4. 75th percentile thresholds should be recalculated for each period")

if __name__ == "__main__":
    main() 