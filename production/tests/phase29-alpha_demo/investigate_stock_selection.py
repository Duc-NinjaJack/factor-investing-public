#!/usr/bin/env python3

# %% [markdown]
# # Stock Selection Investigation
# 
# This script investigates the stock selection logic and analyzes which stocks were selected during 2022-2025 period.

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

print("✅ Stock selection investigation script initialized")

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

def load_data_for_period(db_engine, start_date, end_date):
    """Load all data for the specified period."""
    print(f"📊 Loading data for {start_date} to {end_date}...")
    
    # Load price data
    price_query = f"""
    SELECT 
        trading_date as date,
        ticker,
        close_price_adjusted as close_price,
        total_volume as volume,
        market_cap,
        (close_price_adjusted * total_volume) as daily_value
    FROM vcsc_daily_data_complete
    WHERE trading_date >= '{start_date}' AND trading_date <= '{end_date}'
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
    WHERE date >= '{start_date}' AND date <= '{end_date}'
    ORDER BY date, ticker
    """
    
    # Load benchmark data
    benchmark_query = f"""
    SELECT 
        date,
        close as close_price
    FROM etf_history
    WHERE ticker = 'VNINDEX' 
    AND date >= '{start_date}' AND date <= '{end_date}'
    ORDER BY date
    """
    
    try:
        price_data = pd.read_sql(price_query, db_engine)
        price_data['date'] = pd.to_datetime(price_data['date'])
        price_data['return'] = price_data.groupby('ticker')['close_price'].pct_change()
        
        factor_data = pd.read_sql(factor_query, db_engine)
        factor_data['date'] = pd.to_datetime(factor_data['date'])
        
        benchmark_data = pd.read_sql(benchmark_query, db_engine)
        benchmark_data['date'] = pd.to_datetime(benchmark_data['date'])
        benchmark_data['return'] = benchmark_data['close_price'].pct_change()
        benchmark_data['cumulative_return'] = (1 + benchmark_data['return']).cumprod()
        
        print(f"   ✅ Loaded {len(price_data):,} price records")
        print(f"   ✅ Loaded {len(factor_data):,} factor records")
        print(f"   ✅ Loaded {len(benchmark_data)} benchmark records")
        
        return price_data, factor_data, benchmark_data
        
    except Exception as e:
        print(f"   ❌ Failed to load data: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def detect_regime_for_period(benchmark_data):
    """Detect market regime for the period."""
    print("📊 Detecting market regime...")
    
    # Regime detection parameters
    lookback_days = 30
    vol_threshold_pct = 0.75
    return_threshold_pct = 0.25
    bull_return_threshold_pct = 0.75
    min_regime_duration = 5
    
    # Calculate rolling metrics
    benchmark_data = benchmark_data.sort_values('date')
    benchmark_data['rolling_vol'] = benchmark_data['return'].rolling(lookback_days).std() * np.sqrt(252)
    benchmark_data['rolling_return'] = benchmark_data['return'].rolling(lookback_days).mean() * 252
    
    # Define regime thresholds
    vol_threshold = benchmark_data['rolling_vol'].quantile(vol_threshold_pct)
    return_threshold = benchmark_data['rolling_return'].quantile(return_threshold_pct)
    bull_return_threshold = benchmark_data['rolling_return'].quantile(bull_return_threshold_pct)
    
    # Initial regime classification
    benchmark_data['regime'] = 'normal'
    benchmark_data.loc[
        (benchmark_data['rolling_vol'] > vol_threshold) & 
        (benchmark_data['rolling_return'] < return_threshold), 'regime'
    ] = 'stress'
    benchmark_data.loc[
        (benchmark_data['rolling_vol'] < vol_threshold) & 
        (benchmark_data['rolling_return'] > bull_return_threshold), 'regime'
    ] = 'bull'
    
    # Apply minimum regime duration filter
    benchmark_data['regime_stable'] = benchmark_data['regime']
    
    for i in range(min_regime_duration, len(benchmark_data)):
        recent_regimes = benchmark_data['regime'].iloc[i-min_regime_duration+1:i+1]
        if len(recent_regimes.unique()) == 1:
            benchmark_data.iloc[i, benchmark_data.columns.get_loc('regime_stable')] = recent_regimes.iloc[0]
        else:
            if i > 0:
                benchmark_data.iloc[i, benchmark_data.columns.get_loc('regime_stable')] = benchmark_data.iloc[i-1]['regime_stable']
    
    benchmark_data['regime'] = benchmark_data['regime_stable']
    benchmark_data = benchmark_data.drop('regime_stable', axis=1)
    
    print(f"   ✅ Regime detection completed")
    regime_counts = benchmark_data['regime'].value_counts()
    for regime, count in regime_counts.items():
        print(f"      {regime}: {count} days ({count/len(benchmark_data)*100:.1f}%)")
    
    return benchmark_data

def analyze_stock_selection_for_period(price_data, factor_data, benchmark_data, start_date, end_date):
    """Analyze stock selection for the specified period."""
    print(f"🔍 Analyzing stock selection for {start_date} to {end_date}...")
    
    # Generate monthly rebalancing dates
    rebalance_dates = pd.date_range(start=start_date, end=end_date, freq='M')
    
    all_selected_stocks = []
    portfolio_performance = []
    
    for i, rebalance_date in enumerate(rebalance_dates[:-1]):
        next_rebalance_date = rebalance_dates[i + 1]
        
        print(f"   📅 Analyzing {rebalance_date.strftime('%Y-%m-%d')} to {next_rebalance_date.strftime('%Y-%m-%d')}")
        
        # Get universe rankings for this date
        date_price_data = price_data[price_data['date'] == rebalance_date].copy()
        if len(date_price_data) == 0:
            continue
            
        # Calculate ADTV rankings (63-day lookback)
        lookback_days = 63
        date_price_data['adtv'] = date_price_data['daily_value']
        date_price_data['adtv_rank'] = date_price_data['adtv'].rank(ascending=False)
        date_price_data['in_universe'] = date_price_data['adtv_rank'] <= 200
        
        # Get factor scores for universe stocks
        universe_tickers = date_price_data[date_price_data['in_universe']]['ticker'].tolist()
        date_factor_data = factor_data[
            (factor_data['date'] == rebalance_date) & 
            (factor_data['ticker'].isin(universe_tickers))
        ].copy()
        
        if len(date_factor_data) == 0:
            continue
        
        # Get regime for this date
        regime_data = benchmark_data[benchmark_data['date'] == rebalance_date]
        current_regime = regime_data['regime'].iloc[0] if len(regime_data) > 0 else 'normal'
        
        # Calculate regime-adjusted scores
        regime_weights = {
            'normal': {'quality': 0.33, 'value': 0.33, 'momentum': 0.34},
            'stress': {'quality': 0.4, 'value': 0.3, 'momentum': 0.3, 'allocation': 0.6},
            'bull': {'quality': 0.15, 'value': 0.35, 'momentum': 0.5, 'allocation': 1.0},
        }
        
        weights = regime_weights.get(current_regime, regime_weights['normal'])
        date_factor_data['composite_score'] = (
            date_factor_data['Quality_Composite'] * weights['quality'] +
            date_factor_data['Value_Composite'] * weights['value'] +
            date_factor_data['Momentum_Composite'] * weights['momentum']
        )
        
        # Select top 20 stocks
        portfolio = date_factor_data.nlargest(20, 'composite_score').copy()
        allocation_multiplier = weights.get('allocation', 1.0)
        portfolio['weight'] = (1.0 / len(portfolio)) * allocation_multiplier
        
        # Calculate 1-month performance
        portfolio_tickers = portfolio['ticker'].tolist()
        performance_data = price_data[
            (price_data['date'] >= rebalance_date) & 
            (price_data['date'] <= next_rebalance_date) & 
            (price_data['ticker'].isin(portfolio_tickers))
        ].copy()
        
        if len(performance_data) > 0:
            # Calculate individual stock returns
            stock_returns = []
            for ticker in portfolio_tickers:
                ticker_data = performance_data[performance_data['ticker'] == ticker].copy()
                if len(ticker_data) > 1:
                    start_price = ticker_data.iloc[0]['close_price']
                    end_price = ticker_data.iloc[-1]['close_price']
                    stock_return = (end_price / start_price) - 1
                    stock_returns.append({
                        'ticker': ticker,
                        'return': stock_return,
                        'weight': portfolio[portfolio['ticker'] == ticker]['weight'].iloc[0],
                        'composite_score': portfolio[portfolio['ticker'] == ticker]['composite_score'].iloc[0],
                        'quality_score': portfolio[portfolio['ticker'] == ticker]['Quality_Composite'].iloc[0],
                        'value_score': portfolio[portfolio['ticker'] == ticker]['Value_Composite'].iloc[0],
                        'momentum_score': portfolio[portfolio['ticker'] == ticker]['Momentum_Composite'].iloc[0]
                    })
            
            if stock_returns:
                returns_df = pd.DataFrame(stock_returns)
                portfolio_return = (returns_df['weight'] * returns_df['return']).sum()
                
                # Store portfolio info
                portfolio_performance.append({
                    'rebalance_date': rebalance_date,
                    'regime': current_regime,
                    'portfolio_return': portfolio_return,
                    'allocation_multiplier': allocation_multiplier,
                    'avg_composite_score': portfolio['composite_score'].mean(),
                    'avg_quality': portfolio['Quality_Composite'].mean(),
                    'avg_value': portfolio['Value_Composite'].mean(),
                    'avg_momentum': portfolio['Momentum_Composite'].mean()
                })
                
                # Store individual stock info
                for _, row in returns_df.iterrows():
                    all_selected_stocks.append({
                        'rebalance_date': rebalance_date,
                        'regime': current_regime,
                        'ticker': row['ticker'],
                        'return': row['return'],
                        'weight': row['weight'],
                        'composite_score': row['composite_score'],
                        'quality_score': row['quality_score'],
                        'value_score': row['value_score'],
                        'momentum_score': row['momentum_score']
                    })
    
    return pd.DataFrame(all_selected_stocks), pd.DataFrame(portfolio_performance)

def analyze_stock_performance(stock_data, benchmark_data):
    """Analyze stock performance patterns."""
    print("📊 Analyzing stock performance patterns...")
    
    # Calculate benchmark returns for comparison
    benchmark_data = benchmark_data.sort_values('date')
    benchmark_data['monthly_return'] = benchmark_data['close_price'].pct_change(periods=21)  # Approximate monthly
    
    # Merge stock data with benchmark data
    stock_data['rebalance_date'] = pd.to_datetime(stock_data['rebalance_date'])
    stock_data = stock_data.merge(
        benchmark_data[['date', 'monthly_return']], 
        left_on='rebalance_date', 
        right_on='date', 
        how='left'
    )
    stock_data['excess_return'] = stock_data['return'] - stock_data['monthly_return']
    
    # Analyze by regime
    print(f"   📊 Performance by regime:")
    regime_stats = stock_data.groupby('regime').agg({
        'return': ['count', 'mean', 'std'],
        'excess_return': ['mean', 'std'],
        'composite_score': 'mean',
        'quality_score': 'mean',
        'value_score': 'mean',
        'momentum_score': 'mean'
    }).round(4)
    
    print(regime_stats)
    
    # Analyze top underperformers
    print(f"   📊 Top 10 worst performing stocks:")
    worst_stocks = stock_data.nsmallest(10, 'return')[['ticker', 'rebalance_date', 'regime', 'return', 'excess_return', 'composite_score']]
    print(worst_stocks)
    
    # Analyze top performers
    print(f"   📊 Top 10 best performing stocks:")
    best_stocks = stock_data.nlargest(10, 'return')[['ticker', 'rebalance_date', 'regime', 'return', 'excess_return', 'composite_score']]
    print(best_stocks)
    
    # Analyze factor score patterns
    print(f"   📊 Factor score analysis:")
    factor_corr = stock_data[['return', 'composite_score', 'quality_score', 'value_score', 'momentum_score']].corr()
    print(factor_corr['return'])
    
    return stock_data

def plot_stock_analysis(stock_data, benchmark_data):
    """Create plots for stock analysis."""
    print("📊 Creating stock analysis plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Stock returns distribution
    axes[0, 0].hist(stock_data['return'], bins=50, alpha=0.7, color='blue')
    axes[0, 0].axvline(x=stock_data['return'].mean(), color='red', linestyle='--', label=f'Mean: {stock_data["return"].mean():.3f}')
    axes[0, 0].set_title('Distribution of Stock Returns', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Monthly Return')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Returns by regime
    regime_returns = stock_data.groupby('regime')['return'].mean()
    axes[0, 1].bar(regime_returns.index, regime_returns.values, color=['red', 'blue', 'green'])
    axes[0, 1].set_title('Average Returns by Regime', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Average Monthly Return')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Factor score vs returns
    axes[1, 0].scatter(stock_data['composite_score'], stock_data['return'], alpha=0.6)
    axes[1, 0].set_title('Composite Score vs Returns', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Composite Score')
    axes[1, 0].set_ylabel('Monthly Return')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Excess returns over time
    stock_data['rebalance_date'] = pd.to_datetime(stock_data['rebalance_date'])
    monthly_excess = stock_data.groupby('rebalance_date')['excess_return'].mean()
    axes[1, 1].plot(monthly_excess.index, monthly_excess.values, linewidth=2)
    axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[1, 1].set_title('Monthly Excess Returns vs Benchmark', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Date')
    axes[1, 1].set_ylabel('Excess Return')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig('insights/stock_selection_analysis.png', dpi=300, bbox_inches='tight')
    print("   ✅ Plot saved to insights/stock_selection_analysis.png")
    
    return fig

# %%
def main():
    """Main analysis function."""
    print("🔍 Starting Stock Selection Investigation")
    print("="*60)
    
    try:
        # Create database connection
        db_engine = create_db_connection()
        
        # Analyze 2022-2025 period
        start_date = '2022-01-01'
        end_date = '2025-12-31'
        
        # Load data
        price_data, factor_data, benchmark_data = load_data_for_period(db_engine, start_date, end_date)
        
        if price_data.empty or factor_data.empty or benchmark_data.empty:
            print("❌ Failed to load data")
            return
        
        # Detect regime
        benchmark_data = detect_regime_for_period(benchmark_data)
        
        # Analyze stock selection
        stock_data, portfolio_data = analyze_stock_selection_for_period(
            price_data, factor_data, benchmark_data, start_date, end_date
        )
        
        if stock_data.empty:
            print("❌ No stock selection data generated")
            return
        
        # Analyze performance
        stock_data = analyze_stock_performance(stock_data, benchmark_data)
        
        # Create plots
        fig = plot_stock_analysis(stock_data, benchmark_data)
        
        # Save results
        results_dir = Path("insights")
        results_dir.mkdir(exist_ok=True)
        
        stock_data.to_csv(results_dir / "stock_selection_analysis.csv", index=False)
        portfolio_data.to_csv(results_dir / "portfolio_performance_analysis.csv", index=False)
        
        print(f"\n✅ Analysis completed and saved to {results_dir}/")
        print(f"📊 Key findings:")
        print(f"   - Total stock selections: {len(stock_data)}")
        print(f"   - Average stock return: {stock_data['return'].mean():.3f}")
        print(f"   - Average excess return: {stock_data['excess_return'].mean():.3f}")
        print(f"   - Regime distribution: {stock_data['regime'].value_counts().to_dict()}")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

