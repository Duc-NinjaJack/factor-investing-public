#!/usr/bin/env python3
"""
QVM Engine v3j Comprehensive Multi-Factor Strategy - Complete Backtest & Tearsheet
================================================================================

This script runs a comprehensive backtest from 2016-2025 with detailed tearsheet analysis
similar to the demonstration format.

Strategy: 6-factor comprehensive multi-factor strategy using VNSC data for maximum coverage
- Quality Factors (1/3): ROAA, F-Score
- Value Factors (1/3): P/E Ratio, FCF Yield  
- Momentum Factors (1/3): Multi-horizon momentum, Low volatility

Data Sources: VNSC daily data (728 tickers) + Raw fundamental data (7.3M+ records)
Period: 2016-01-01 to 2025-07-28
Benchmark: VNINDEX
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Database connectivity
from sqlalchemy import create_engine, text

# Add Project Root to Python Path
current_path = Path.cwd()
while not (current_path / 'production').is_dir():
    if current_path.parent == current_path:
        raise FileNotFoundError("Could not find the 'production' directory.")
    current_path = current_path.parent

project_root = current_path

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import production modules
from production.database.connection import get_database_manager

# Import our custom components
sys.path.append(str(project_root / 'production' / 'tests' / 'phase29-alpha_demo' / 'components'))
from fundamental_factor_calculator import FundamentalFactorCalculator
from momentum_volatility_calculator import MomentumVolatilityCalculator

print(f"✅ Successfully imported production modules.")
print(f"   - Project Root set to: {project_root}")

# COMPREHENSIVE MULTI-FACTOR CONFIGURATION
QVM_CONFIG = {
    "strategy_name": "QVM_Engine_v3j_Comprehensive_Multi_Factor_Backtest",
    "backtest_start_date": "2020-01-01",
    "backtest_end_date": "2023-12-31",
    "rebalance_frequency": "M",
    "transaction_cost_bps": 30,
    "universe": {
        "lookback_days": 63,
        "top_n_stocks": 40,
        "max_position_size": 0.035,
        "max_sector_exposure": 0.25,
        "target_portfolio_size": 35,
    },
    "factors": {
        "roaa_weight": 0.167,
        "f_score_weight": 0.167,
        "pe_weight": 0.167,
        "fcf_yield_weight": 0.167,
        "momentum_weight": 0.167,
        "low_vol_weight": 0.167,
        "momentum_horizons": [21, 63, 126, 252],
        "skip_months": 1,
        "fundamental_lag_days": 45,
    }
}

print("\n⚙️  QVM Engine v3j Configuration Loaded:")
print(f"   - Strategy: {QVM_CONFIG['strategy_name']}")
print(f"   - Period: {QVM_CONFIG['backtest_start_date']} to {QVM_CONFIG['backtest_end_date']}")
print(f"   - Universe: Top {QVM_CONFIG['universe']['top_n_stocks']} stocks by ADTV")

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

def load_universe_data(config, db_engine):
    """Load universe data with ADTV calculations."""
    print("📊 Loading universe data...")
    
    lookback_days = config['universe']['lookback_days']
    start_date = pd.to_datetime(config['backtest_start_date']) - timedelta(days=lookback_days)
    
    query = f"""
    SELECT 
        ticker,
        trading_date as date,
        close_price,
        total_volume as volume,
        (close_price * total_volume) as daily_value
    FROM vcsc_daily_data 
    WHERE trading_date >= '{start_date.strftime('%Y-%m-%d')}'
    AND trading_date <= '{config['backtest_end_date']}'
    ORDER BY ticker, trading_date
    """
    
    universe_data = pd.read_sql(query, db_engine)
    universe_data['date'] = pd.to_datetime(universe_data['date'])
    
    print(f"   ✅ Loaded {len(universe_data):,} universe records")
    print(f"   📈 Date range: {universe_data['date'].min()} to {universe_data['date'].max()}")
    print(f"   🏢 Unique tickers: {universe_data['ticker'].nunique()}")
    
    return universe_data

def calculate_universe_rankings(universe_data, config):
    """Calculate universe rankings based on ADTV."""
    print("📊 Calculating universe rankings...")
    
    lookback_days = config['universe']['lookback_days']
    
    # Calculate rolling ADTV
    universe_data = universe_data.sort_values(['ticker', 'date'])
    universe_data['adtv'] = universe_data.groupby('ticker')['daily_value'].rolling(
        window=lookback_days, min_periods=lookback_days//2
    ).mean().reset_index(0, drop=True)
    
    # Calculate rankings
    rankings = []
    for date in universe_data['date'].unique():
        date_data = universe_data[universe_data['date'] == date].copy()
        date_data = date_data.dropna(subset=['adtv'])
        
        if len(date_data) > 0:
            date_data['adtv_rank'] = date_data['adtv'].rank(ascending=False)
            date_data['in_universe'] = date_data['adtv_rank'] <= config['universe']['top_n_stocks']
            rankings.append(date_data[['ticker', 'date', 'adtv', 'adtv_rank', 'in_universe']])
    
    rankings_df = pd.concat(rankings, ignore_index=True)
    
    print(f"   ✅ Calculated rankings for {len(rankings_df):,} records")
    print(f"   📈 Universe coverage: {rankings_df['in_universe'].sum():,} selections")
    
    return rankings_df

def load_benchmark_data(config, db_engine):
    """Load VNINDEX benchmark data."""
    print("📊 Loading benchmark data...")
    
    query = f"""
    SELECT 
        trading_date as date,
        close_price
    FROM vcsc_daily_data 
    WHERE ticker = 'VNINDEX'
    AND trading_date >= '{config['backtest_start_date']}'
    AND trading_date <= '{config['backtest_end_date']}'
    ORDER BY trading_date
    """
    
    benchmark_data = pd.read_sql(query, db_engine)
    benchmark_data['date'] = pd.to_datetime(benchmark_data['date'])
    benchmark_data = benchmark_data.set_index('date')
    
    # Calculate returns
    benchmark_data['return'] = benchmark_data['close_price'].pct_change()
    benchmark_data['cumulative_return'] = (1 + benchmark_data['return']).cumprod()
    
    print(f"   ✅ Loaded {len(benchmark_data)} benchmark records")
    print(f"   📈 Benchmark period: {benchmark_data.index.min()} to {benchmark_data.index.max()}")
    
    return benchmark_data

def calculate_fundamental_factors(config, db_engine):
    """Calculate fundamental factors using raw data."""
    print("📊 Calculating fundamental factors...")
    
    calculator = FundamentalFactorCalculator(db_engine)
    factors = calculator.calculate_all_factors(config['backtest_start_date'], config['backtest_end_date'])
    
    print(f"   ✅ Calculated fundamental factors for {len(factors)} records")
    return factors

def calculate_momentum_volatility_factors(config, db_engine):
    """Calculate momentum and volatility factors using VNSC data."""
    print("📊 Calculating momentum and volatility factors...")
    
    calculator = MomentumVolatilityCalculator(db_engine)
    factors = calculator.calculate_all_factors(config['backtest_start_date'], config['backtest_end_date'], config['factors']['momentum_horizons'])
    
    print(f"   ✅ Calculated momentum/volatility factors for {len(factors)} records")
    return factors

def combine_all_factors(fundamental_factors, momentum_vol_factors, universe_rankings):
    """Combine all factors with universe filtering."""
    print("📊 Combining all factors...")
    
    # Merge fundamental factors with universe
    combined = fundamental_factors.merge(
        universe_rankings[['ticker', 'date', 'in_universe']], 
        on=['ticker', 'date'], 
        how='inner'
    )
    
    # Merge momentum/volatility factors
    combined = combined.merge(momentum_vol_factors, on=['ticker', 'date'], how='inner')
    
    # Filter to universe stocks only
    combined = combined[combined['in_universe'] == True].copy()
    
    print(f"   ✅ Combined factors for {len(combined)} records")
    print(f"   🏢 Unique tickers: {combined['ticker'].nunique()}")
    
    return combined

def normalize_factor(factor_series):
    """Normalize factor to 0-1 range using winsorization and z-score."""
    q1 = factor_series.quantile(0.01)
    q99 = factor_series.quantile(0.99)
    winsorized = factor_series.clip(lower=q1, upper=q99)
    
    z_score = (winsorized - winsorized.mean()) / winsorized.std()
    normalized = 0.5 * (1 + np.tanh(z_score / 2))
    
    return normalized.fillna(0.5)

def calculate_composite_scores(combined_factors, config):
    """Calculate composite factor scores."""
    print("📊 Calculating composite scores...")
    
    # Normalize individual factors
    combined_factors['roaa_score'] = normalize_factor(combined_factors['roaa'])
    combined_factors['pe_score'] = normalize_factor(-combined_factors['pe_ratio'])
    combined_factors['fcf_yield_score'] = normalize_factor(combined_factors['fcf_yield'])
    combined_factors['f_score_score'] = normalize_factor(combined_factors['f_score'])
    combined_factors['momentum_score'] = normalize_factor(combined_factors['composite_momentum'])
    combined_factors['low_vol_score_final'] = normalize_factor(combined_factors['low_vol_score'])
    
    # Calculate composite scores by category
    quality_score = (combined_factors['roaa_score'] * 0.5 + combined_factors['f_score_score'] * 0.5)
    value_score = (combined_factors['pe_score'] * 0.5 + combined_factors['fcf_yield_score'] * 0.5)
    momentum_score = (combined_factors['momentum_score'] * 0.5 + combined_factors['low_vol_score_final'] * 0.5)
    
    # Final composite score: 1/3 Quality + 1/3 Value + 1/3 Momentum
    combined_factors['composite_score'] = (quality_score * (1/3) + value_score * (1/3) + momentum_score * (1/3))
    
    print(f"   ✅ Calculated composite scores for {len(combined_factors)} records")
    return combined_factors

def construct_portfolio(factor_data, date, config):
    """Construct portfolio for a given date."""
    date_data = factor_data[factor_data['date'] == date].copy()
    
    if len(date_data) == 0:
        return pd.DataFrame()
    
    # Sort by composite score (descending)
    date_data = date_data.sort_values('composite_score', ascending=False)
    
    # Select top stocks
    target_size = config['universe']['target_portfolio_size']
    max_position = config['universe']['max_position_size']
    
    # Equal weight portfolio with position size limits
    weights = np.ones(min(len(date_data), target_size)) / min(len(date_data), target_size)
    weights = np.minimum(weights, max_position)
    weights = weights / weights.sum()  # Renormalize
    
    portfolio = date_data.head(len(weights)).copy()
    portfolio['weight'] = weights
    
    return portfolio[['ticker', 'weight', 'composite_score']]

def calculate_portfolio_returns(portfolio, next_date, db_engine):
    """Calculate portfolio returns for the period."""
    if len(portfolio) == 0:
        return 0.0
    
    # Get price data for portfolio stocks
    tickers = portfolio['ticker'].tolist()
    tickers_str = "', '".join(tickers)
    
    query = f"""
    SELECT ticker, trading_date as date, close_price
    FROM vcsc_daily_data 
    WHERE ticker IN ('{tickers_str}')
    AND trading_date IN ('{portfolio['date'].iloc[0]}', '{next_date}')
    ORDER BY ticker, trading_date
    """
    
    price_data = pd.read_sql(query, db_engine)
    price_data['date'] = pd.to_datetime(price_data['date'])
    
    # Calculate returns
    returns = []
    for ticker in tickers:
        ticker_data = price_data[price_data['ticker'] == ticker]
        if len(ticker_data) == 2:
            start_price = ticker_data.iloc[0]['close_price']
            end_price = ticker_data.iloc[1]['close_price']
            ticker_return = (end_price - start_price) / start_price
            returns.append(ticker_return)
        else:
            returns.append(0.0)
    
    # Calculate weighted portfolio return
    portfolio_return = np.sum(np.array(returns) * portfolio['weight'].values)
    
    return portfolio_return

def run_backtest(config, db_engine, combined_factors):
    """Run complete backtest with portfolio construction and rebalancing."""
    print("🚀 Starting comprehensive backtest...")
    
    # Generate rebalancing dates
    start_date = pd.to_datetime(config['backtest_start_date'])
    end_date = pd.to_datetime(config['backtest_end_date'])
    
    rebalance_dates = pd.date_range(start=start_date, end=end_date, freq=config['rebalance_frequency'])
    
    print(f"📅 Generated {len(rebalance_dates)} rebalancing dates")
    
    # Initialize backtest results
    backtest_results = []
    current_portfolio = pd.DataFrame()
    
    # Run backtest
    for i, rebalance_date in enumerate(rebalance_dates[:-1]):
        next_rebalance_date = rebalance_dates[i + 1]
        
        if i % 12 == 0:  # Print progress every year
            print(f"📊 Processing {rebalance_date.strftime('%Y-%m-%d')} to {next_rebalance_date.strftime('%Y-%m-%d')}")
        
        # Construct new portfolio
        new_portfolio = construct_portfolio(combined_factors, rebalance_date, config)
        
        if len(new_portfolio) > 0:
            # Calculate returns for the period
            portfolio_return = calculate_portfolio_returns(new_portfolio, next_rebalance_date, db_engine)
            
            # Apply transaction costs if portfolio changed
            transaction_cost = 0.0
            if len(current_portfolio) > 0:
                turnover = 0.5  # Assume 50% turnover
                transaction_cost = turnover * config['transaction_cost_bps'] / 10000
            
            net_return = portfolio_return - transaction_cost
            
            # Store results
            backtest_results.append({
                'date': rebalance_date,
                'next_date': next_rebalance_date,
                'portfolio_return': portfolio_return,
                'transaction_cost': transaction_cost,
                'net_return': net_return,
                'portfolio_size': len(new_portfolio),
                'avg_score': new_portfolio['composite_score'].mean()
            })
            
            current_portfolio = new_portfolio.copy()
    
    # Convert to DataFrame
    results_df = pd.DataFrame(backtest_results)
    results_df['date'] = pd.to_datetime(results_df['date'])
    results_df['cumulative_return'] = (1 + results_df['net_return']).cumprod()
    
    print(f"✅ Backtest completed: {len(results_df)} periods")
    
    return results_df

def calculate_performance_metrics(backtest_results, benchmark_data):
    """Calculate comprehensive performance metrics."""
    print("📊 Calculating performance metrics...")
    
    # Strategy metrics
    strategy_returns = backtest_results['net_return']
    strategy_cumulative = backtest_results['cumulative_return']
    
    # Benchmark metrics
    benchmark_returns = benchmark_data['return'].reindex(backtest_results['date']).fillna(0)
    benchmark_cumulative = benchmark_data['cumulative_return'].reindex(backtest_results['date']).fillna(1)
    
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
    
    # Additional metrics
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
    sortino_ratio = annualized_return / (strategy_returns[strategy_returns < 0].std() * np.sqrt(252)) if strategy_returns[strategy_returns < 0].std() > 0 else 0
    
    metrics = {
        'Strategy': {
            'Total Return': f"{total_return:.2%}",
            'Annualized Return': f"{annualized_return:.2%}",
            'Volatility': f"{volatility:.2%}",
            'Sharpe Ratio': f"{sharpe_ratio:.3f}",
            'Sortino Ratio': f"{sortino_ratio:.3f}",
            'Calmar Ratio': f"{calmar_ratio:.3f}",
            'Max Drawdown': f"{max_drawdown:.2%}",
            'Win Rate': f"{win_rate:.2%}",
            'Information Ratio': f"{information_ratio:.3f}",
            'Excess Return': f"{excess_return:.2%}"
        },
        'Benchmark': {
            'Total Return': f"{benchmark_total_return:.2%}",
            'Annualized Return': f"{benchmark_annualized:.2%}",
            'Volatility': f"{benchmark_volatility:.2%}",
            'Sharpe Ratio': f"{benchmark_sharpe:.3f}",
            'Max Drawdown': f"{(benchmark_cumulative / benchmark_cumulative.expanding().max() - 1).min():.2%}"
        }
    }
    
    print("✅ Performance metrics calculated")
    
    return metrics, {
        'strategy_returns': strategy_returns,
        'benchmark_returns': benchmark_returns,
        'strategy_cumulative': strategy_cumulative,
        'benchmark_cumulative': benchmark_cumulative,
        'drawdown': drawdown,
        'excess_returns': excess_returns
    }

def create_comprehensive_tearsheet(backtest_results, benchmark_data, metrics, performance_data, config):
    """Create comprehensive tearsheet with detailed analysis."""
    print("📊 Creating comprehensive tearsheet...")
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(20, 24))
    
    # Create a comprehensive layout
    gs = fig.add_gridspec(6, 3, hspace=0.3, wspace=0.3)
    
    # 1. Cumulative Returns (top row, full width)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(backtest_results['date'], performance_data['strategy_cumulative'], 
             label='QVM Engine v3j Strategy', linewidth=3, color='#2E86AB')
    ax1.plot(backtest_results['date'], performance_data['benchmark_cumulative'], 
             label='VNINDEX Benchmark', linewidth=2, color='#A23B72', alpha=0.8)
    ax1.set_title('Cumulative Returns Comparison', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Cumulative Return', fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 2. Drawdown Analysis
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.fill_between(backtest_results['date'], performance_data['drawdown'], 0, 
                     color='#F18F01', alpha=0.6)
    ax2.set_title('Drawdown Analysis', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Drawdown', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # 3. Rolling Sharpe Ratio (12-month)
    ax3 = fig.add_subplot(gs[1, 1])
    rolling_sharpe = performance_data['strategy_returns'].rolling(12).mean() / \
                     performance_data['strategy_returns'].rolling(12).std() * np.sqrt(252)
    ax3.plot(backtest_results['date'], rolling_sharpe, color='#C73E1D', linewidth=2)
    ax3.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
    ax3.set_title('12-Month Rolling Sharpe Ratio', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Sharpe Ratio', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # 4. Portfolio Size Over Time
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.plot(backtest_results['date'], backtest_results['portfolio_size'], 
             color='#6B5B95', linewidth=2)
    ax4.set_title('Portfolio Size', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Number of Stocks', fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    # 5. Monthly Returns Heatmap
    ax5 = fig.add_subplot(gs[2, :])
    monthly_returns = backtest_results.set_index('date')['net_return'].resample('M').sum()
    monthly_returns_pivot = monthly_returns.groupby([monthly_returns.index.year, monthly_returns.index.month]).first().unstack()
    sns.heatmap(monthly_returns_pivot, annot=True, fmt='.2%', cmap='RdYlGn', center=0, ax=ax5)
    ax5.set_title('Monthly Returns Heatmap', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Month', fontsize=12)
    ax5.set_ylabel('Year', fontsize=12)
    
    # 6. Rolling Volatility
    ax6 = fig.add_subplot(gs[3, 0])
    rolling_vol = performance_data['strategy_returns'].rolling(12).std() * np.sqrt(252)
    ax6.plot(backtest_results['date'], rolling_vol, color='#8B4513', linewidth=2)
    ax6.set_title('12-Month Rolling Volatility', fontsize=14, fontweight='bold')
    ax6.set_ylabel('Volatility', fontsize=12)
    ax6.grid(True, alpha=0.3)
    
    # 7. Excess Returns
    ax7 = fig.add_subplot(gs[3, 1])
    ax7.plot(backtest_results['date'], performance_data['excess_returns'].cumsum(), 
             color='#228B22', linewidth=2)
    ax7.set_title('Cumulative Excess Returns', fontsize=14, fontweight='bold')
    ax7.set_ylabel('Excess Return', fontsize=12)
    ax7.grid(True, alpha=0.3)
    
    # 8. Factor Score Distribution
    ax8 = fig.add_subplot(gs[3, 2])
    ax8.hist(backtest_results['avg_score'], bins=30, alpha=0.7, color='#4682B4')
    ax8.set_title('Portfolio Average Score Distribution', fontsize=14, fontweight='bold')
    ax8.set_xlabel('Average Composite Score', fontsize=12)
    ax8.set_ylabel('Frequency', fontsize=12)
    ax8.grid(True, alpha=0.3)
    
    # 9. Performance Metrics Table
    ax9 = fig.add_subplot(gs[4, :])
    ax9.axis('tight')
    ax9.axis('off')
    
    # Create metrics table
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
    
    table = ax9.table(cellText=table_data,
                     colLabels=['Strategy Metric', 'Value', 'Benchmark Metric', 'Value'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.25, 0.15, 0.25, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    ax9.set_title('Performance Metrics Summary', fontsize=16, fontweight='bold', pad=20)
    
    # 10. Strategy Configuration Summary
    ax10 = fig.add_subplot(gs[5, :])
    ax10.axis('tight')
    ax10.axis('off')
    
    config_summary = [
        ['Strategy Name', config['strategy_name']],
        ['Backtest Period', f"{config['backtest_start_date']} to {config['backtest_end_date']}"],
        ['Rebalancing Frequency', config['rebalance_frequency']],
        ['Transaction Costs', f"{config['transaction_cost_bps']} bps"],
        ['Universe Size', f"Top {config['universe']['top_n_stocks']} stocks"],
        ['Target Portfolio Size', config['universe']['target_portfolio_size']],
        ['Max Position Size', f"{config['universe']['max_position_size']:.1%}"],
        ['Quality Weight', '33.3% (ROAA + F-Score)'],
        ['Value Weight', '33.3% (P/E + FCF Yield)'],
        ['Momentum Weight', '33.3% (Multi-horizon + Low Vol)']
    ]
    
    table2 = ax10.table(cellText=config_summary,
                       colLabels=['Parameter', 'Value'],
                       cellLoc='left',
                       loc='center',
                       colWidths=[0.4, 0.6])
    table2.auto_set_font_size(False)
    table2.set_fontsize(10)
    table2.scale(1, 1.5)
    ax10.set_title('Strategy Configuration', fontsize=16, fontweight='bold', pad=20)
    
    plt.suptitle(f'{config["strategy_name"]} - Comprehensive Performance Analysis', 
                fontsize=20, fontweight='bold', y=0.98)
    
    # Print metrics
    print("\n" + "="*80)
    print("PERFORMANCE METRICS SUMMARY")
    print("="*80)
    
    print("\nSTRATEGY METRICS:")
    for metric, value in metrics['Strategy'].items():
        print(f"  {metric}: {value}")
    
    print("\nBENCHMARK METRICS:")
    for metric, value in metrics['Benchmark'].items():
        print(f"  {metric}: {value}")
    
    print("\n" + "="*80)
    
    return fig

def save_results(backtest_results, metrics, performance_data, config, combined_factors):
    """Save all results to files."""
    print("💾 Saving results...")
    
    # Create results directory
    results_dir = Path("insights")
    results_dir.mkdir(exist_ok=True)
    
    # Save backtest results
    backtest_results.to_csv(results_dir / "backtest_results.csv", index=False)
    
    # Save metrics
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(results_dir / "performance_metrics.csv")
    
    # Save performance data
    performance_df = pd.DataFrame(performance_data)
    performance_df.to_csv(results_dir / "performance_data.csv", index=False)
    
    # Save factor data
    combined_factors.to_csv(results_dir / "factor_data.csv", index=False)
    
    # Save configuration
    config_df = pd.DataFrame([config])
    config_df.to_csv(results_dir / "strategy_config.csv", index=False)
    
    print(f"✅ Results saved to {results_dir}/")

def main():
    """Main execution function."""
    print("🚀 QVM Engine v3j Comprehensive Multi-Factor Strategy - Complete Backtest & Tearsheet")
    print("="*80)
    
    try:
        # Create database connection
        db_engine = create_db_connection()
        
        # Load data
        print("\n📊 Loading data...")
        universe_data = load_universe_data(QVM_CONFIG, db_engine)
        universe_rankings = calculate_universe_rankings(universe_data, QVM_CONFIG)
        benchmark_data = load_benchmark_data(QVM_CONFIG, db_engine)
        
        # Calculate factors
        print("\n📊 Calculating factors...")
        fundamental_factors = calculate_fundamental_factors(QVM_CONFIG, db_engine)
        momentum_vol_factors = calculate_momentum_volatility_factors(QVM_CONFIG, db_engine)
        
        # Combine factors
        print("\n📊 Combining factors...")
        combined_factors = combine_all_factors(fundamental_factors, momentum_vol_factors, universe_rankings)
        combined_factors = calculate_composite_scores(combined_factors, QVM_CONFIG)
        
        # Run backtest
        print("\n🚀 Running backtest...")
        backtest_results = run_backtest(QVM_CONFIG, db_engine, combined_factors)
        
        # Calculate performance metrics
        print("\n📊 Calculating performance metrics...")
        metrics, performance_data = calculate_performance_metrics(backtest_results, benchmark_data)
        
        # Create tearsheet
        print("\n📊 Creating tearsheet...")
        fig = create_comprehensive_tearsheet(backtest_results, benchmark_data, metrics, performance_data, QVM_CONFIG)
        
        # Save results
        print("\n💾 Saving results...")
        save_results(backtest_results, metrics, performance_data, QVM_CONFIG, combined_factors)
        
        # Display summary
        print("\n🎯 STRATEGY SUMMARY:")
        print(f"   - Period: {QVM_CONFIG['backtest_start_date']} to {QVM_CONFIG['backtest_end_date']}")
        print(f"   - Total Return: {metrics['Strategy']['Total Return']}")
        print(f"   - Sharpe Ratio: {metrics['Strategy']['Sharpe Ratio']}")
        print(f"   - Max Drawdown: {metrics['Strategy']['Max Drawdown']}")
        print(f"   - Excess Return vs VNINDEX: {metrics['Strategy']['Excess Return']}")
        
        print("\n✅ QVM Engine v3j backtest completed successfully!")
        
        # Show plot
        plt.show()
        
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
