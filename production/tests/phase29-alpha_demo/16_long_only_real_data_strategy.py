#!/usr/bin/env python3
"""
QVM Engine v3j Long-Only Real Data Strategy
==========================================

This strategy uses REAL data with PROPER long-only returns.
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

# LONG-ONLY REAL DATA STRATEGY CONFIGURATION
QVM_CONFIG = {
    "strategy_name": "QVM_Engine_v3j_Long_Only_Real_Data",
    "backtest_start_date": "2020-01-01",
    "backtest_end_date": "2023-12-31",
    "rebalance_frequency": "M",
    "transaction_cost_bps": 30,
    "universe": {
        "lookback_days": 63,
        "top_n_stocks": 40,
        "max_position_size": 0.05,
        "target_portfolio_size": 20,
    },
    "strategy_type": "long_only",
}

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

def load_real_price_data(db_engine, start_date, end_date):
    """Load REAL price data with returns."""
    print(f"📊 Loading REAL price data from {start_date} to {end_date}...")
    
    query = f"""
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
    
    try:
        data = pd.read_sql(query, db_engine)
        data['date'] = pd.to_datetime(data['date'])
        
        # Calculate daily returns
        data = data.sort_values(['ticker', 'date'])
        data['return'] = data.groupby('ticker')['close_price'].pct_change()
        
        print(f"   ✅ Loaded {len(data):,} price records")
        print(f"   📈 Date range: {data['date'].min()} to {data['date'].max()}")
        print(f"   🏢 Unique tickers: {data['ticker'].nunique()}")
        print(f"   📊 Average daily return: {data['return'].mean():.4f}")
        
        return data
        
    except Exception as e:
        print(f"   ❌ Failed to load price data: {e}")
        return pd.DataFrame()

def load_real_factor_scores(db_engine, start_date, end_date):
    """Load REAL factor scores."""
    print(f"📊 Loading REAL factor scores from {start_date} to {end_date}...")
    
    query = f"""
    SELECT 
        date,
        ticker,
        QVM_Composite
    FROM factor_scores_qvm
    WHERE date >= '{start_date}' AND date <= '{end_date}'
    ORDER BY date, ticker
    """
    
    try:
        data = pd.read_sql(query, db_engine)
        data['date'] = pd.to_datetime(data['date'])
        print(f"   ✅ Loaded {len(data):,} factor score records")
        return data
        
    except Exception as e:
        print(f"   ❌ Failed to load factor scores: {e}")
        return pd.DataFrame()

def load_real_benchmark_data(db_engine, start_date, end_date):
    """Load REAL benchmark data."""
    print(f"📊 Loading REAL benchmark data from {start_date} to {end_date}...")
    
    query = f"""
    SELECT 
        date,
        close as close_price
    FROM etf_history
    WHERE ticker = 'VNINDEX' 
    AND date >= '{start_date}' AND date <= '{end_date}'
    ORDER BY date
    """
    
    try:
        data = pd.read_sql(query, db_engine)
        data['date'] = pd.to_datetime(data['date'])
        data = data.sort_values('date')
        data['return'] = data['close_price'].pct_change()
        data['cumulative_return'] = (1 + data['return']).cumprod()
        
        print(f"   ✅ Loaded {len(data)} benchmark records")
        return data
        
    except Exception as e:
        print(f"   ❌ Failed to load benchmark data: {e}")
        return pd.DataFrame()

def calculate_universe_rankings(price_data, config):
    """Calculate universe rankings."""
    print("📊 Calculating universe rankings...")
    
    lookback_days = config['universe']['lookback_days']
    
    # Calculate rolling ADTV
    price_data = price_data.sort_values(['ticker', 'date'])
    price_data['adtv'] = price_data.groupby('ticker')['daily_value'].rolling(
        window=lookback_days, min_periods=lookback_days//2
    ).mean().reset_index(0, drop=True)
    
    # Calculate rankings for each date
    rankings = []
    unique_dates = price_data['date'].unique()
    
    for i, date in enumerate(unique_dates):
        if i % 100 == 0:
            print(f"   📅 Processing date {i+1}/{len(unique_dates)}: {date.strftime('%Y-%m-%d')}")
        
        date_data = price_data[price_data['date'] == date].copy()
        date_data = date_data.dropna(subset=['adtv'])
        
        if len(date_data) > 0:
            date_data['adtv_rank'] = date_data['adtv'].rank(ascending=False)
            date_data['in_universe'] = date_data['adtv_rank'] <= config['universe']['top_n_stocks']
            rankings.append(date_data[['ticker', 'date', 'adtv', 'adtv_rank', 'in_universe']])
    
    if rankings:
        rankings_df = pd.concat(rankings, ignore_index=True)
        print(f"   ✅ Calculated rankings for {len(rankings_df):,} records")
        return rankings_df
    else:
        return pd.DataFrame()

def calculate_portfolio_returns(portfolio, price_data, start_date, end_date):
    """Calculate ACTUAL portfolio returns using real price data."""
    if len(portfolio) == 0:
        return pd.Series(dtype=float)
    
    portfolio_tickers = portfolio['ticker'].tolist()
    
    # Get price data for portfolio stocks in the period
    period_data = price_data[
        (price_data['date'] >= start_date) & 
        (price_data['date'] <= end_date) & 
        (price_data['ticker'].isin(portfolio_tickers))
    ].copy()
    
    if len(period_data) == 0:
        return pd.Series(dtype=float)
    
    # Calculate weighted returns for each day
    daily_returns = []
    
    for date in period_data['date'].unique():
        date_data = period_data[period_data['date'] == date].copy()
        
        # Merge with portfolio weights
        date_data = date_data.merge(
            portfolio[['ticker', 'weight']], 
            on='ticker', 
            how='inner'
        )
        
        if len(date_data) > 0:
            # Calculate weighted return for this date
            weighted_return = (date_data['weight'] * date_data['return']).sum()
            daily_returns.append({
                'date': date,
                'portfolio_return': weighted_return
            })
    
    if daily_returns:
        returns_df = pd.DataFrame(daily_returns)
        returns_df = returns_df.set_index('date')
        return returns_df['portfolio_return']
    else:
        return pd.Series(dtype=float)

def run_long_only_backtest(config, db_engine):
    """Run long-only backtest using REAL data."""
    print("🚀 Starting LONG-ONLY REAL data backtest...")
    
    # Load REAL data
    price_data = load_real_price_data(db_engine, config['backtest_start_date'], config['backtest_end_date'])
    factor_scores = load_real_factor_scores(db_engine, config['backtest_start_date'], config['backtest_end_date'])
    benchmark_data = load_real_benchmark_data(db_engine, config['backtest_start_date'], config['backtest_end_date'])
    
    if price_data.empty or factor_scores.empty or benchmark_data.empty:
        print("❌ Failed to load data")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # Calculate universe rankings
    universe_rankings = calculate_universe_rankings(price_data, config)
    
    # Merge factor scores with universe rankings
    combined_data = factor_scores.merge(
        universe_rankings[['ticker', 'date', 'in_universe']], 
        on=['ticker', 'date'], 
        how='inner'
    )
    combined_data = combined_data[combined_data['in_universe'] == True].copy()
    combined_data['composite_score'] = combined_data['QVM_Composite']
    
    # Generate rebalancing dates
    start_date = pd.to_datetime(config['backtest_start_date'])
    end_date = pd.to_datetime(config['backtest_end_date'])
    rebalance_dates = pd.date_range(start=start_date, end=end_date, freq=config['rebalance_frequency'])
    
    # Run long-only backtest
    backtest_results = []
    all_daily_returns = []
    
    for i, rebalance_date in enumerate(rebalance_dates[:-1]):
        next_rebalance_date = rebalance_dates[i + 1]
        
        if i % 6 == 0:
            print(f"📊 Processing {rebalance_date.strftime('%Y-%m-%d')} to {next_rebalance_date.strftime('%Y-%m-%d')}")
        
        # Get data for this date
        date_data = combined_data[combined_data['date'] == rebalance_date].copy()
        
        if len(date_data) > 0:
            # Sort by composite score (descending) - LONG-ONLY
            date_data = date_data.sort_values('composite_score', ascending=False)
            
            # Select top stocks for long-only portfolio
            target_size = config['universe']['target_portfolio_size']
            max_position = config['universe']['max_position_size']
            
            # Equal weight portfolio
            weights = np.ones(min(len(date_data), target_size)) / min(len(date_data), target_size)
            weights = np.minimum(weights, max_position)
            weights = weights / weights.sum()
            
            portfolio = date_data.head(len(weights)).copy()
            portfolio['weight'] = weights
            
            # Calculate ACTUAL returns using real price data
            period_returns = calculate_portfolio_returns(portfolio, price_data, rebalance_date, next_rebalance_date)
            
            if len(period_returns) > 0:
                # Apply transaction costs
                transaction_cost = 0.0015  # 15 bps
                net_returns = period_returns - transaction_cost / len(period_returns)
                
                # Store daily returns for equity curve
                for date, ret in net_returns.items():
                    all_daily_returns.append({
                        'date': date,
                        'return': ret,
                        'rebalance_date': rebalance_date
                    })
                
                # Calculate period metrics
                period_return = (1 + net_returns).prod() - 1
                
                backtest_results.append({
                    'date': rebalance_date,
                    'next_date': next_rebalance_date,
                    'portfolio_return': period_return,
                    'portfolio_size': len(portfolio),
                    'avg_score': portfolio['composite_score'].mean()
                })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(backtest_results)
    if not results_df.empty:
        results_df['date'] = pd.to_datetime(results_df['date'])
        results_df['cumulative_return'] = (1 + results_df['portfolio_return']).cumprod()
    
    # Create daily returns DataFrame for equity curve
    daily_returns_df = pd.DataFrame(all_daily_returns)
    if not daily_returns_df.empty:
        daily_returns_df['date'] = pd.to_datetime(daily_returns_df['date'])
        daily_returns_df = daily_returns_df.sort_values('date')
        daily_returns_df['cumulative_return'] = (1 + daily_returns_df['return']).cumprod()
    
    print(f"✅ LONG-ONLY REAL data backtest completed: {len(results_df)} periods")
    return results_df, benchmark_data, daily_returns_df

def calculate_performance_metrics(backtest_results, daily_returns, benchmark_data):
    """Calculate performance metrics."""
    if backtest_results.empty or daily_returns.empty:
        return {}, {}
    
    # Strategy metrics from daily returns
    strategy_returns = daily_returns['return']
    strategy_cumulative = daily_returns['cumulative_return']
    
    # Benchmark metrics
    benchmark_returns = benchmark_data['return'].reindex(daily_returns['date']).fillna(0)
    benchmark_cumulative = benchmark_data['cumulative_return'].reindex(daily_returns['date']).fillna(1)
    
    # Basic metrics
    total_return = strategy_cumulative.iloc[-1] - 1
    annualized_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
    volatility = strategy_returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
    
    # Benchmark comparison
    benchmark_total_return = benchmark_cumulative.iloc[-1] - 1
    benchmark_annualized = (1 + benchmark_total_return) ** (252 / len(benchmark_returns)) - 1
    benchmark_volatility = benchmark_returns.std() * np.sqrt(252)
    benchmark_sharpe = benchmark_annualized / benchmark_volatility if benchmark_volatility > 0 else 0
    
    # Excess returns
    excess_returns = strategy_returns - benchmark_returns
    excess_return = (1 + excess_returns).prod() - 1
    information_ratio = excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
    
    # Drawdown analysis
    peak = strategy_cumulative.expanding().max()
    drawdown = (strategy_cumulative - peak) / peak
    max_drawdown = drawdown.min()
    
    # Win rate
    win_rate = (strategy_returns > 0).mean()
    
    metrics = {
        'Strategy': {
            'Total Return': f"{total_return:.2%}",
            'Annualized Return': f"{annualized_return:.2%}",
            'Volatility': f"{volatility:.2%}",
            'Sharpe Ratio': f"{sharpe_ratio:.3f}",
            'Max Drawdown': f"{max_drawdown:.2%}",
            'Win Rate': f"{win_rate:.2%}",
            'Information Ratio': f"{information_ratio:.3f}",
            'Excess Return': f"{excess_return:.2%}"
        },
        'Benchmark': {
            'Total Return': f"{benchmark_total_return:.2%}",
            'Annualized Return': f"{benchmark_annualized:.2%}",
            'Volatility': f"{benchmark_volatility:.2%}",
            'Sharpe Ratio': f"{benchmark_sharpe:.3f}"
        }
    }
    
    return metrics, {
        'strategy_returns': strategy_returns,
        'benchmark_returns': benchmark_returns,
        'strategy_cumulative': strategy_cumulative,
        'benchmark_cumulative': benchmark_cumulative,
        'drawdown': drawdown,
        'excess_returns': excess_returns,
        'daily_dates': daily_returns['date']
    }

def create_equity_curve_tearsheet(backtest_results, daily_returns, benchmark_data, metrics, performance_data, config):
    """Create tearsheet with equity curve."""
    if backtest_results.empty or daily_returns.empty:
        return None
    
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Equity Curve
    axes[0, 0].plot(performance_data['daily_dates'], performance_data['strategy_cumulative'], 
                   label='QVM Long-Only Strategy', linewidth=3, color='#2E86AB')
    axes[0, 0].plot(performance_data['daily_dates'], performance_data['benchmark_cumulative'], 
                   label='VNINDEX Benchmark', linewidth=2, color='#A23B72', alpha=0.8)
    axes[0, 0].set_title('Equity Curve - Cumulative Returns', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Cumulative Return')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Drawdown Analysis
    axes[0, 1].fill_between(performance_data['daily_dates'], performance_data['drawdown'], 0, 
                           color='#F18F01', alpha=0.6)
    axes[0, 1].set_title('Drawdown Analysis', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Drawdown')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Rolling Sharpe Ratio
    rolling_sharpe = performance_data['strategy_returns'].rolling(63).mean() / \
                     performance_data['strategy_returns'].rolling(63).std() * np.sqrt(252)
    axes[1, 0].plot(performance_data['daily_dates'], rolling_sharpe, color='#C73E1D', linewidth=2)
    axes[1, 0].axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
    axes[1, 0].set_title('63-Day Rolling Sharpe Ratio', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Sharpe Ratio')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Performance Metrics Table
    axes[1, 1].axis('tight')
    axes[1, 1].axis('off')
    
    strategy_metrics = list(metrics['Strategy'].items())
    benchmark_metrics = list(metrics['Benchmark'].items())
    
    table_data = []
    for i in range(max(len(strategy_metrics), len(benchmark_metrics))):
        row = []
        if i < len(strategy_metrics):
            row.extend([strategy_metrics[i][0], strategy_metrics[i][1]])
        else:
            row.extend(['', ''])
        if i < len(benchmark_metrics):
            row.extend([benchmark_metrics[i][0], benchmark_metrics[i][1]])
        else:
            row.extend(['', ''])
        table_data.append(row)
    
    table = axes[1, 1].table(cellText=table_data,
                            colLabels=['Strategy Metric', 'Value', 'Benchmark Metric', 'Value'],
                            cellLoc='center', loc='center', colWidths=[0.25, 0.15, 0.25, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    axes[1, 1].set_title('Performance Metrics Summary', fontsize=14, fontweight='bold', pad=20)
    
    plt.suptitle(f'{config["strategy_name"]} - Long-Only Real Data Performance', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Print metrics
    print("\n" + "="*80)
    print("LONG-ONLY REAL DATA PERFORMANCE METRICS SUMMARY")
    print("="*80)
    
    print("\nSTRATEGY METRICS (Long-Only Real Data):")
    for metric, value in metrics['Strategy'].items():
        print(f"  {metric}: {value}")
    
    print("\nBENCHMARK METRICS (Long-Only Real Data):")
    for metric, value in metrics['Benchmark'].items():
        print(f"  {metric}: {value}")
    
    print("\n" + "="*80)
    
    return fig

def save_results(backtest_results, daily_returns, metrics, performance_data, config):
    """Save results."""
    print("💾 Saving results...")
    
    results_dir = Path("insights")
    results_dir.mkdir(exist_ok=True)
    
    if not backtest_results.empty:
        backtest_results.to_csv(results_dir / "long_only_backtest_results.csv", index=False)
    
    if not daily_returns.empty:
        daily_returns.to_csv(results_dir / "long_only_daily_returns.csv", index=False)
    
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(results_dir / "long_only_performance_metrics.csv")
    
    print(f"✅ Results saved to {results_dir}/")

def main():
    """Main execution function."""
    print("🚀 QVM Engine v3j Long-Only Real Data Strategy")
    print("="*80)
    
    try:
        db_engine = create_db_connection()
        
        backtest_results, benchmark_data, daily_returns = run_long_only_backtest(QVM_CONFIG, db_engine)
        
        if backtest_results.empty or daily_returns.empty:
            print("❌ No backtest results generated")
            return
        
        metrics, performance_data = calculate_performance_metrics(backtest_results, daily_returns, benchmark_data)
        
        fig = create_equity_curve_tearsheet(backtest_results, daily_returns, benchmark_data, metrics, performance_data, QVM_CONFIG)
        
        save_results(backtest_results, daily_returns, metrics, performance_data, QVM_CONFIG)
        
        print("\n🎯 LONG-ONLY REAL DATA STRATEGY SUMMARY:")
        print(f"   - Strategy: {QVM_CONFIG['strategy_name']}")
        print(f"   - Period: {QVM_CONFIG['backtest_start_date']} to {QVM_CONFIG['backtest_end_date']}")
        print(f"   - Strategy Type: {QVM_CONFIG['strategy_type']}")
        print(f"   - Total Return: {metrics['Strategy']['Total Return']}")
        print(f"   - Annualized Return: {metrics['Strategy']['Annualized Return']}")
        print(f"   - Volatility: {metrics['Strategy']['Volatility']}")
        print(f"   - Sharpe Ratio: {metrics['Strategy']['Sharpe Ratio']}")
        print(f"   - Max Drawdown: {metrics['Strategy']['Max Drawdown']}")
        print(f"   - Excess Return vs VNINDEX: {metrics['Strategy']['Excess Return']}")
        
        print("\n✅ Long-only real data strategy completed successfully!")
        
        if fig:
            plt.show()
        
    except Exception as e:
        print(f"❌ Long-only real data strategy failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
