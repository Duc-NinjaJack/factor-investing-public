#!/usr/bin/env python3

# %% [markdown]
# # Analysis of 2022 Performance Drop
# 
# This script analyzes the massive performance drop in 2022 to understand:
# 1. Which stocks were held during the critical period
# 2. Regime changes and factor performance
# 3. Portfolio turnover and transaction costs
# 4. Individual stock performance vs market

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

print("✅ Analysis script initialized")

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

def load_2022_data(db_engine):
    """Load data specifically for 2022 analysis."""
    print("📊 Loading 2022 data for analysis...")
    
    # Load price data for 2021-2023 to get context
    price_query = """
    SELECT 
        trading_date as date,
        ticker,
        close_price_adjusted as close_price,
        total_volume as volume,
        market_cap,
        (close_price_adjusted * total_volume) as daily_value
    FROM vcsc_daily_data_complete
    WHERE trading_date >= '2021-01-01' AND trading_date <= '2023-12-31'
    ORDER BY trading_date, ticker
    """
    
    # Load factor scores
    factor_query = """
    SELECT 
        date,
        ticker,
        Quality_Composite,
        Value_Composite,
        Momentum_Composite,
        QVM_Composite
    FROM factor_scores_qvm
    WHERE date >= '2021-01-01' AND date <= '2023-12-31'
    ORDER BY date, ticker
    """
    
    # Load benchmark data
    benchmark_query = """
    SELECT 
        date,
        close as close_price
    FROM etf_history
    WHERE ticker = 'VNINDEX' 
    AND date >= '2021-01-01' AND date <= '2023-12-31'
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

def detect_regime_2022(benchmark_data):
    """Detect market regime specifically for 2022 analysis."""
    print("📊 Detecting market regime for 2022...")
    
    # Calculate rolling metrics
    benchmark_data = benchmark_data.sort_values('date')
    benchmark_data['rolling_vol'] = benchmark_data['return'].rolling(63).std() * np.sqrt(252)
    benchmark_data['rolling_return'] = benchmark_data['return'].rolling(63).mean() * 252
    
    # Define regime thresholds
    vol_threshold = benchmark_data['rolling_vol'].quantile(0.7)
    return_threshold = benchmark_data['rolling_return'].quantile(0.3)
    
    # Classify regimes
    benchmark_data['regime'] = 'normal'
    benchmark_data.loc[
        (benchmark_data['rolling_vol'] > vol_threshold) & 
        (benchmark_data['rolling_return'] < return_threshold), 'regime'
    ] = 'stress'
    benchmark_data.loc[
        (benchmark_data['rolling_vol'] < vol_threshold) & 
        (benchmark_data['rolling_return'] > return_threshold), 'regime'
    ] = 'bull'
    
    # Focus on 2022
    benchmark_2022 = benchmark_data[benchmark_data['date'].dt.year == 2022].copy()
    
    print(f"   📊 2022 Regime distribution:")
    regime_counts = benchmark_2022['regime'].value_counts()
    for regime, count in regime_counts.items():
        print(f"      {regime}: {count} days ({count/len(benchmark_2022)*100:.1f}%)")
    
    return benchmark_data, benchmark_2022

def analyze_portfolio_2022(price_data, factor_data, benchmark_data):
    """Analyze portfolio composition and performance in 2022."""
    print("📊 Analyzing 2022 portfolio performance...")
    
    # Define rebalancing dates (monthly)
    rebalance_dates = pd.date_range(start='2022-01-01', end='2022-12-31', freq='M')
    
    # Sample 5 rebalancing periods
    sample_dates = ['2022-01-31', '2022-03-31', '2022-06-30', '2022-09-30', '2022-12-31']
    
    portfolio_analysis = []
    
    for i, rebalance_date in enumerate(sample_dates):
        rebalance_date = pd.to_datetime(rebalance_date)
        next_date = rebalance_dates[rebalance_dates > rebalance_date][0] if len(rebalance_dates[rebalance_dates > rebalance_date]) > 0 else pd.to_datetime('2022-12-31')
        
        print(f"\n📅 Analyzing rebalancing: {rebalance_date.strftime('%Y-%m-%d')}")
        
        # Get universe rankings for this date
        date_price_data = price_data[price_data['date'] == rebalance_date].copy()
        if len(date_price_data) == 0:
            print(f"   ❌ No price data for {rebalance_date}")
            continue
            
        # Calculate ADTV rankings
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
            print(f"   ❌ No factor data for {rebalance_date}")
            continue
        
        # Get regime for this date
        regime_data = benchmark_data[benchmark_data['date'] == rebalance_date]
        current_regime = regime_data['regime'].iloc[0] if len(regime_data) > 0 else 'normal'
        
        # Calculate regime-adjusted scores
        regime_weights = {
            'normal': {'quality': 0.4, 'value': 0.3, 'momentum': 0.3},
            'stress': {'quality': 0.6, 'value': 0.3, 'momentum': 0.1},
            'bull': {'quality': 0.2, 'value': 0.2, 'momentum': 0.6}
        }
        
        weights = regime_weights.get(current_regime, regime_weights['normal'])
        date_factor_data['composite_score'] = (
            date_factor_data['Quality_Composite'] * weights['quality'] +
            date_factor_data['Value_Composite'] * weights['value'] +
            date_factor_data['Momentum_Composite'] * weights['momentum']
        )
        
        # Select top 20 stocks
        portfolio = date_factor_data.nlargest(20, 'composite_score').copy()
        portfolio['weight'] = 1.0 / len(portfolio)  # Equal weight
        
        print(f"   📊 Regime: {current_regime}")
        print(f"   📊 Portfolio size: {len(portfolio)}")
        print(f"   📊 Top 5 stocks:")
        for idx, row in portfolio.head().iterrows():
            print(f"      {row['ticker']}: Score={row['composite_score']:.3f}, Q={row['Quality_Composite']:.3f}, V={row['Value_Composite']:.3f}, M={row['Momentum_Composite']:.3f}")
        
        # Calculate 1-month performance
        portfolio_tickers = portfolio['ticker'].tolist()
        performance_data = price_data[
            (price_data['date'] >= rebalance_date) & 
            (price_data['date'] <= next_date) & 
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
                        'weight': 1.0 / len(portfolio)
                    })
            
            if stock_returns:
                returns_df = pd.DataFrame(stock_returns)
                portfolio_return = (returns_df['weight'] * returns_df['return']).sum()
                
                # Transaction costs (assume 30 bps)
                transaction_cost = 0.003  # 30 bps
                net_return = portfolio_return - transaction_cost
                
                print(f"   📈 Portfolio return: {portfolio_return:.2%}")
                print(f"   💰 Transaction cost: {transaction_cost:.2%}")
                print(f"   📊 Net return: {net_return:.2%}")
                
                # Store analysis
                portfolio_analysis.append({
                    'rebalance_date': rebalance_date,
                    'regime': current_regime,
                    'portfolio_size': len(portfolio),
                    'portfolio_return': portfolio_return,
                    'transaction_cost': transaction_cost,
                    'net_return': net_return,
                    'top_stocks': portfolio['ticker'].head().tolist(),
                    'avg_quality': portfolio['Quality_Composite'].mean(),
                    'avg_value': portfolio['Value_Composite'].mean(),
                    'avg_momentum': portfolio['Momentum_Composite'].mean(),
                    'avg_composite': portfolio['composite_score'].mean()
                })
    
    return pd.DataFrame(portfolio_analysis)

def analyze_market_performance_2022(price_data, benchmark_data):
    """Analyze overall market performance in 2022."""
    print("📊 Analyzing 2022 market performance...")
    
    # Get 2022 data
    price_2022 = price_data[price_data['date'].dt.year == 2022].copy()
    benchmark_2022 = benchmark_data[benchmark_data['date'].dt.year == 2022].copy()
    
    # Calculate monthly returns
    monthly_returns = []
    for month in range(1, 13):
        month_data = price_2022[price_2022['date'].dt.month == month]
        if len(month_data) > 0:
            # Calculate average return for the month
            avg_return = month_data['return'].mean()
            monthly_returns.append({
                'month': month,
                'avg_return': avg_return,
                'volatility': month_data['return'].std(),
                'trading_days': len(month_data['date'].unique())
            })
    
    monthly_df = pd.DataFrame(monthly_returns)
    
    print("   📊 Monthly market performance in 2022:")
    for _, row in monthly_df.iterrows():
        print(f"      Month {row['month']}: Return={row['avg_return']:.2%}, Vol={row['volatility']:.2%}")
    
    return monthly_df

# %%
def main():
    """Main analysis function."""
    print("🔍 Starting 2022 Performance Drop Analysis")
    print("="*60)
    
    try:
        # Create database connection
        db_engine = create_db_connection()
        
        # Load 2022 data
        price_data, factor_data, benchmark_data = load_2022_data(db_engine)
        
        if price_data.empty or factor_data.empty or benchmark_data.empty:
            print("❌ Failed to load data")
            return
        
        # Detect regime changes
        benchmark_data, benchmark_2022 = detect_regime_2022(benchmark_data)
        
        # Analyze portfolio performance
        portfolio_analysis = analyze_portfolio_2022(price_data, factor_data, benchmark_data)
        
        # Analyze market performance
        market_analysis = analyze_market_performance_2022(price_data, benchmark_data)
        
        # Summary
        print("\n" + "="*60)
        print("📊 2022 PERFORMANCE DROP ANALYSIS SUMMARY")
        print("="*60)
        
        if not portfolio_analysis.empty:
            print(f"\n📈 Portfolio Performance Summary:")
            print(f"   Average monthly return: {portfolio_analysis['net_return'].mean():.2%}")
            print(f"   Worst month: {portfolio_analysis['net_return'].min():.2%}")
            print(f"   Best month: {portfolio_analysis['net_return'].max():.2%}")
            print(f"   Total return (5 months): {(1 + portfolio_analysis['net_return']).prod() - 1:.2%}")
            
            print(f"\n🎯 Regime Analysis:")
            regime_counts = portfolio_analysis['regime'].value_counts()
            for regime, count in regime_counts.items():
                print(f"   {regime}: {count} rebalancings")
            
            print(f"\n📊 Factor Performance:")
            print(f"   Average Quality Score: {portfolio_analysis['avg_quality'].mean():.3f}")
            print(f"   Average Value Score: {portfolio_analysis['avg_value'].mean():.3f}")
            print(f"   Average Momentum Score: {portfolio_analysis['avg_momentum'].mean():.3f}")
            print(f"   Average Composite Score: {portfolio_analysis['avg_composite'].mean():.3f}")
        
        print(f"\n📈 Market Performance Summary:")
        print(f"   Average monthly return: {market_analysis['avg_return'].mean():.2%}")
        print(f"   Worst month: {market_analysis['avg_return'].min():.2%}")
        print(f"   Best month: {market_analysis['avg_return'].max():.2%}")
        print(f"   Average volatility: {market_analysis['volatility'].mean():.2%}")
        
        # Save analysis results
        results_dir = Path("insights")
        results_dir.mkdir(exist_ok=True)
        
        if not portfolio_analysis.empty:
            portfolio_analysis.to_csv(results_dir / "2022_portfolio_analysis.csv", index=False)
        market_analysis.to_csv(results_dir / "2022_market_analysis.csv", index=False)
        
        print(f"\n✅ Analysis completed and saved to {results_dir}/")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

