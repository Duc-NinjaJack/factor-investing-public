#!/usr/bin/env python3

# %% [markdown]
# # Factor and Universe Investigation
# 
# This script investigates factor calculations and universe selection with proper ADTV threshold.

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

print("✅ Factor and universe investigation script initialized")

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

def load_sample_data(db_engine, sample_date='2022-06-30'):
    """Load sample data for factor investigation."""
    print(f"📊 Loading sample data for {sample_date}...")
    
    # Load price data with ADTV calculation
    price_query = f"""
    SELECT 
        trading_date as date,
        ticker,
        close_price_adjusted as close_price,
        total_volume as volume,
        market_cap,
        (close_price_adjusted * total_volume) as daily_value
    FROM vcsc_daily_data_complete
    WHERE trading_date BETWEEN DATE_SUB('{sample_date}', INTERVAL 90 DAY) AND '{sample_date}'
    ORDER BY trading_date, ticker
    """
    
    # Load factor scores
    factor_query = f"""
    SELECT 
        date,
        ticker,
        Quality_Composite,
        Value_Composite,
        Momentum_Composite,
        QVM_Composite
    FROM factor_scores_qvm
    WHERE date = '{sample_date}'
    ORDER BY ticker
    """
    
    # Load fundamental data for validation
    fundamental_query = f"""
    SELECT 
        ticker,
        roaa,
        roe,
        pe_ratio,
        pb_ratio,
        debt_to_equity,
        current_ratio,
        gross_margin,
        net_margin
    FROM fundamental_data
    WHERE date = '{sample_date}'
    ORDER BY ticker
    """
    
    try:
        price_data = pd.read_sql(price_query, db_engine)
        price_data['date'] = pd.to_datetime(price_data['date'])
        
        factor_data = pd.read_sql(factor_query, db_engine)
        factor_data['date'] = pd.to_datetime(factor_data['date'])
        
        # Try to load fundamental data (might not exist)
        try:
            fundamental_data = pd.read_sql(fundamental_query, db_engine)
        except:
            print("   ⚠️ Fundamental data not available, skipping validation")
            fundamental_data = pd.DataFrame()
        
        print(f"   ✅ Loaded {len(price_data):,} price records")
        print(f"   ✅ Loaded {len(factor_data):,} factor records")
        if not fundamental_data.empty:
            print(f"   ✅ Loaded {len(fundamental_data):,} fundamental records")
        
        return price_data, factor_data, fundamental_data
        
    except Exception as e:
        print(f"   ❌ Failed to load data: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def calculate_adtv_universe(price_data, sample_date, min_adtv_bn=10.0):
    """Calculate ADTV universe with proper threshold."""
    print(f"📊 Calculating ADTV universe for {sample_date}...")
    
    # Calculate 63-day rolling ADTV
    lookback_days = 63
    price_data = price_data.sort_values(['ticker', 'date'])
    price_data['adtv'] = price_data.groupby('ticker')['daily_value'].rolling(
        window=lookback_days, min_periods=lookback_days//2
    ).mean().reset_index(0, drop=True)
    
    # Get ADTV for the sample date
    sample_adtv = price_data[price_data['date'] == sample_date].copy()
    sample_adtv = sample_adtv.dropna(subset=['adtv'])
    
    # Convert to billion VND
    sample_adtv['adtv_bn'] = sample_adtv['adtv'] / 1e9
    
    # Apply 10 billion VND threshold
    liquid_universe = sample_adtv[sample_adtv['adtv_bn'] >= min_adtv_bn].copy()
    liquid_universe = liquid_universe.sort_values('adtv_bn', ascending=False)
    
    print(f"   📊 ADTV Statistics:")
    print(f"      Total stocks: {len(sample_adtv)}")
    print(f"      Liquid stocks (>= {min_adtv_bn}B VND): {len(liquid_universe)}")
    print(f"      Average ADTV: {sample_adtv['adtv_bn'].mean():.1f}B VND")
    print(f"      Median ADTV: {sample_adtv['adtv_bn'].median():.1f}B VND")
    print(f"      Max ADTV: {sample_adtv['adtv_bn'].max():.1f}B VND")
    
    print(f"   📊 Top 10 most liquid stocks:")
    for i, (_, row) in enumerate(liquid_universe.head(10).iterrows()):
        print(f"      {i+1:2d}. {row['ticker']}: {row['adtv_bn']:.1f}B VND")
    
    return liquid_universe

def investigate_factor_calculations(factor_data, liquid_universe, fundamental_data):
    """Investigate factor calculations with common sense checks."""
    print("🔍 Investigating factor calculations...")
    
    # Merge factor data with liquid universe
    factor_analysis = factor_data.merge(
        liquid_universe[['ticker', 'adtv_bn']], 
        on='ticker', 
        how='inner'
    )
    
    print(f"   📊 Factor analysis for {len(factor_analysis)} liquid stocks:")
    
    # Basic statistics
    for factor in ['Quality_Composite', 'Value_Composite', 'Momentum_Composite', 'QVM_Composite']:
        stats = factor_analysis[factor].describe()
        print(f"   📈 {factor}:")
        print(f"      Mean: {stats['mean']:.3f}")
        print(f"      Std: {stats['std']:.3f}")
        print(f"      Min: {stats['min']:.3f}")
        print(f"      Max: {stats['max']:.3f}")
        print(f"      Range: {stats['max'] - stats['min']:.3f}")
    
    # Check for extreme values
    print(f"   🔍 Extreme value analysis:")
    for factor in ['Quality_Composite', 'Value_Composite', 'Momentum_Composite']:
        extreme_high = factor_analysis.nlargest(5, factor)[['ticker', factor, 'adtv_bn']]
        extreme_low = factor_analysis.nsmallest(5, factor)[['ticker', factor, 'adtv_bn']]
        
        print(f"   📊 {factor} - Top 5:")
        for _, row in extreme_high.iterrows():
            print(f"      {row['ticker']}: {row[factor]:.3f} (ADTV: {row['adtv_bn']:.1f}B)")
        
        print(f"   📊 {factor} - Bottom 5:")
        for _, row in extreme_low.iterrows():
            print(f"      {row['ticker']}: {row[factor]:.3f} (ADTV: {row['adtv_bn']:.1f}B)")
    
    # Check factor correlations
    print(f"   📊 Factor correlations:")
    factor_corr = factor_analysis[['Quality_Composite', 'Value_Composite', 'Momentum_Composite', 'QVM_Composite']].corr()
    print(factor_corr.round(3))
    
    # Check if fundamental data is available for validation
    if not fundamental_data.empty:
        print(f"   🔍 Fundamental data validation:")
        validation_data = factor_analysis.merge(fundamental_data, on='ticker', how='inner')
        
        if len(validation_data) > 0:
            print(f"   📊 Quality factor vs ROAA/ROE:")
            quality_corr = validation_data[['Quality_Composite', 'roaa', 'roe']].corr()
            print(quality_corr['Quality_Composite'].round(3))
            
            print(f"   📊 Value factor vs P/E:")
            value_corr = validation_data[['Value_Composite', 'pe_ratio']].corr()
            print(value_corr['Value_Composite'].round(3))
    
    return factor_analysis

def analyze_stock_selection_sample(factor_analysis, sample_date):
    """Analyze stock selection for a sample period."""
    print(f"📊 Analyzing stock selection for {sample_date}...")
    
    # Simulate stock selection process
    # Sort by QVM_Composite and select top 20
    top_stocks = factor_analysis.nlargest(20, 'QVM_Composite').copy()
    
    print(f"   📊 Selected stocks (Top 20 by QVM_Composite):")
    for i, (_, row) in enumerate(top_stocks.iterrows()):
        print(f"   {i+1:2d}. {row['ticker']}: QVM={row['QVM_Composite']:.3f}, Q={row['Quality_Composite']:.3f}, V={row['Value_Composite']:.3f}, M={row['Momentum_Composite']:.3f}, ADTV={row['adtv_bn']:.1f}B")
    
    # Analyze factor distribution in selected stocks
    print(f"   📊 Factor distribution in selected stocks:")
    for factor in ['Quality_Composite', 'Value_Composite', 'Momentum_Composite']:
        stats = top_stocks[factor].describe()
        print(f"   📈 {factor}: Mean={stats['mean']:.3f}, Std={stats['std']:.3f}, Min={stats['min']:.3f}, Max={stats['max']:.3f}")
    
    # Check for potential issues
    print(f"   🔍 Potential issues:")
    
    # Check if all factors are positive (might indicate normalization issues)
    all_positive_quality = (top_stocks['Quality_Composite'] > 0).all()
    all_positive_value = (top_stocks['Value_Composite'] > 0).all()
    all_positive_momentum = (top_stocks['Momentum_Composite'] > 0).all()
    
    print(f"   📊 All Quality scores positive: {all_positive_quality}")
    print(f"   📊 All Value scores positive: {all_positive_value}")
    print(f"   📊 All Momentum scores positive: {all_positive_momentum}")
    
    # Check factor ranges
    for factor in ['Quality_Composite', 'Value_Composite', 'Momentum_Composite']:
        factor_range = top_stocks[factor].max() - top_stocks[factor].min()
        print(f"   📊 {factor} range: {factor_range:.3f}")
    
    return top_stocks

def check_adtv_threshold_impact(price_data, sample_date):
    """Check the impact of different ADTV thresholds."""
    print(f"📊 Checking ADTV threshold impact...")
    
    # Calculate ADTV for different thresholds
    thresholds = [1, 5, 10, 20, 50]  # Billion VND
    
    results = []
    for threshold in thresholds:
        liquid_universe = calculate_adtv_universe(price_data, sample_date, threshold)
        results.append({
            'threshold_bn': threshold,
            'liquid_stocks': len(liquid_universe),
            'avg_adtv': liquid_universe['adtv_bn'].mean() if len(liquid_universe) > 0 else 0
        })
    
    results_df = pd.DataFrame(results)
    
    print(f"   📊 ADTV threshold impact:")
    for _, row in results_df.iterrows():
        print(f"   >= {row['threshold_bn']:2.0f}B VND: {int(row['liquid_stocks']):3d} stocks (avg ADTV: {row['avg_adtv']:.1f}B)")
    
    return results_df

def plot_factor_analysis(factor_analysis, top_stocks):
    """Create plots for factor analysis."""
    print("📊 Creating factor analysis plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Factor distributions
    axes[0, 0].hist(factor_analysis['Quality_Composite'], bins=20, alpha=0.7, label='All Liquid Stocks')
    axes[0, 0].hist(top_stocks['Quality_Composite'], bins=10, alpha=0.7, label='Selected Stocks')
    axes[0, 0].set_title('Quality Factor Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Quality Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Factor correlations
    factor_corr = factor_analysis[['Quality_Composite', 'Value_Composite', 'Momentum_Composite']].corr()
    im = axes[0, 1].imshow(factor_corr, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0, 1].set_title('Factor Correlations', fontsize=14, fontweight='bold')
    axes[0, 1].set_xticks(range(len(factor_corr.columns)))
    axes[0, 1].set_yticks(range(len(factor_corr.columns)))
    axes[0, 1].set_xticklabels(factor_corr.columns, rotation=45)
    axes[0, 1].set_yticklabels(factor_corr.columns)
    
    # Add correlation values
    for i in range(len(factor_corr.columns)):
        for j in range(len(factor_corr.columns)):
            text = axes[0, 1].text(j, i, f'{factor_corr.iloc[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontsize=10)
    
    # 3. ADTV vs Factor scores
    axes[1, 0].scatter(factor_analysis['adtv_bn'], factor_analysis['QVM_Composite'], alpha=0.6)
    axes[1, 0].scatter(top_stocks['adtv_bn'], top_stocks['QVM_Composite'], color='red', alpha=0.8, s=50)
    axes[1, 0].set_title('ADTV vs QVM Composite Score', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('ADTV (Billion VND)')
    axes[1, 0].set_ylabel('QVM Composite Score')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Factor score ranges
    factor_ranges = []
    factor_names = []
    for factor in ['Quality_Composite', 'Value_Composite', 'Momentum_Composite']:
        factor_ranges.append(factor_analysis[factor].max() - factor_analysis[factor].min())
        factor_names.append(factor.replace('_Composite', ''))
    
    axes[1, 1].bar(factor_names, factor_ranges, color=['blue', 'green', 'red'])
    axes[1, 1].set_title('Factor Score Ranges', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Score Range')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig('insights/factor_analysis_investigation.png', dpi=300, bbox_inches='tight')
    print("   ✅ Plot saved to insights/factor_analysis_investigation.png")
    
    return fig

# %%
def main():
    """Main investigation function."""
    print("🔍 Starting Factor and Universe Investigation")
    print("="*60)
    
    try:
        # Create database connection
        db_engine = create_db_connection()
        
        # Use a sample date from 2022
        sample_date = '2022-06-30'
        
        # Load sample data
        price_data, factor_data, fundamental_data = load_sample_data(db_engine, sample_date)
        
        if price_data.empty or factor_data.empty:
            print("❌ Failed to load data")
            return
        
        # Calculate ADTV universe with 10B VND threshold
        liquid_universe = calculate_adtv_universe(price_data, sample_date, min_adtv_bn=10.0)
        
        # Investigate factor calculations
        factor_analysis = investigate_factor_calculations(factor_data, liquid_universe, fundamental_data)
        
        # Analyze stock selection
        top_stocks = analyze_stock_selection_sample(factor_analysis, sample_date)
        
        # Check ADTV threshold impact
        threshold_impact = check_adtv_threshold_impact(price_data, sample_date)
        
        # Create plots
        fig = plot_factor_analysis(factor_analysis, top_stocks)
        
        # Save results
        results_dir = Path("insights")
        results_dir.mkdir(exist_ok=True)
        
        factor_analysis.to_csv(results_dir / "factor_analysis_sample.csv", index=False)
        top_stocks.to_csv(results_dir / "selected_stocks_sample.csv", index=False)
        threshold_impact.to_csv(results_dir / "adtv_threshold_impact.csv", index=False)
        
        print(f"\n✅ Investigation completed and saved to {results_dir}/")
        print(f"📊 Key findings:")
        print(f"   - Liquid universe (>=10B VND): {len(liquid_universe)} stocks")
        print(f"   - Factor analysis: {len(factor_analysis)} stocks")
        print(f"   - Selected stocks: {len(top_stocks)} stocks")
        print(f"   - ADTV threshold impact analyzed for {len(threshold_impact)} thresholds")
        
    except Exception as e:
        print(f"❌ Investigation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
