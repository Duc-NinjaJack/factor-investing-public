#!/usr/bin/env python3

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add Project Root to Python Path
current_path = Path.cwd()
while not (current_path / 'production').is_dir():
    current_path = current_path.parent
project_root = current_path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from production.database.connection import get_database_manager
from sqlalchemy import text

print("✅ QVM Engine v3j Complete Backtest (v18b)")

# Configuration
CONFIG = {
    'strategy_name': 'QVM_Engine_v3j_Complete_v18b',
    'top_n_stocks': 20,
    'backtest_start_date': '2016-01-01',
    'backtest_end_date': '2025-12-31',
    'transaction_cost_bps': 10,  # 10 basis points
    'initial_capital': 1000000   # 1M VND
}

def calculate_portfolio_returns(holdings_df, price_data, benchmark_data):
    """Calculate comprehensive portfolio returns with benchmark comparison."""
    print("📈 Calculating comprehensive portfolio returns...")
    
    portfolio_values = []
    all_daily_returns = []
    current_capital = CONFIG['initial_capital']
    
    # Get all unique dates for proper sequencing
    all_dates = sorted(holdings_df['date'].unique())
    
    for i, date in enumerate(all_dates):
        date_holdings = holdings_df[holdings_df['date'] == date]
        
        if date_holdings.empty:
            continue
            
        # Get price data for this date
        current_prices = price_data[price_data['date'] == date]
        
        if current_prices.empty:
            continue
            
        # Calculate portfolio value
        portfolio_value = 0
        
        for _, holding in date_holdings.iterrows():
            ticker = holding['ticker']
            ticker_price = current_prices[current_prices['ticker'] == ticker]
            
            if not ticker_price.empty:
                price = ticker_price['close_price'].iloc[0]
                # Equal weight allocation
                position_size = current_capital / len(date_holdings)
                shares = position_size / price
                portfolio_value += shares * price
        
        if portfolio_value > 0:
            portfolio_values.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'capital': current_capital
            })
            
            # Calculate daily returns for the period until next rebalancing
            if i < len(all_dates) - 1:
                next_date = all_dates[i + 1]
                
                # Get price data for the period
                period_prices = price_data[
                    (price_data['date'] >= date) & 
                    (price_data['date'] <= next_date) &
                    (price_data['ticker'].isin(date_holdings['ticker'].unique()))
                ]
                
                if not period_prices.empty:
                    # Calculate daily returns for each stock
                    period_prices = period_prices.sort_values(['ticker', 'date'])
                    period_prices['stock_return'] = period_prices.groupby('ticker')['close_price'].pct_change()
                    
                    # Calculate portfolio daily returns
                    for daily_date in period_prices['date'].unique():
                        daily_data = period_prices[period_prices['date'] == daily_date]
                        if not daily_data.empty:
                            # Equal weight portfolio return
                            portfolio_return = daily_data['stock_return'].mean()
                            
                            # Apply transaction costs on rebalancing day
                            if daily_date == date:
                                transaction_cost = CONFIG['transaction_cost_bps'] / 10000
                                portfolio_return -= transaction_cost
                            
                            all_daily_returns.append({
                                'date': daily_date,
                                'portfolio_return': portfolio_return,
                                'rebalance_date': date
                            })
            
            # Update capital for next period
            current_capital = portfolio_value
    
    portfolio_df = pd.DataFrame(portfolio_values)
    daily_returns_df = pd.DataFrame(all_daily_returns)
    
    print(f"   ✅ Portfolio values: {len(portfolio_df)} records")
    print(f"   ✅ Daily returns: {len(daily_returns_df)} records")
    
    return portfolio_df, daily_returns_df

def calculate_performance_metrics(portfolio_values, daily_returns, benchmark_data):
    """Calculate comprehensive performance metrics."""
    print("📊 Calculating comprehensive performance metrics...")
    
    if portfolio_values.empty or daily_returns.empty:
        print("   ⚠️ No data available for performance calculation")
        return {}
    
    # Calculate portfolio metrics
    portfolio_values = portfolio_values.sort_values('date')
    daily_returns = daily_returns.sort_values('date')
    
    print(f"   📊 Portfolio values: {len(portfolio_values)} records")
    print(f"   📊 Daily returns: {len(daily_returns)} records")
    print(f"   📊 Benchmark data: {len(benchmark_data)} records")
    
    # Merge with benchmark data
    daily_returns = daily_returns.merge(benchmark_data, on='date', how='left')
    daily_returns['benchmark_return'] = daily_returns['benchmark_close'].pct_change()
    
    # Remove NaN values
    daily_returns = daily_returns.dropna(subset=['portfolio_return', 'benchmark_return'])
    
    if daily_returns.empty:
        print("   ⚠️ No valid data after merging")
        return {}
    
    print(f"   📊 Valid daily returns: {len(daily_returns)} records")
    
    # Calculate cumulative returns
    daily_returns['cumulative_portfolio_return'] = (1 + daily_returns['portfolio_return']).cumprod()
    daily_returns['cumulative_benchmark_return'] = (1 + daily_returns['benchmark_return']).cumprod()
    
    # Basic metrics
    total_return = (1 + daily_returns['portfolio_return']).prod() - 1
    benchmark_total_return = (1 + daily_returns['benchmark_return']).prod() - 1
    
    # Annualized return
    days = (pd.to_datetime(daily_returns['date'].iloc[-1]) - pd.to_datetime(daily_returns['date'].iloc[0])).days
    annualized_return = ((1 + total_return) ** (365 / days)) - 1
    benchmark_annualized_return = ((1 + benchmark_total_return) ** (365 / days)) - 1
    
    # Volatility
    volatility = daily_returns['portfolio_return'].std() * np.sqrt(252)
    benchmark_volatility = daily_returns['benchmark_return'].std() * np.sqrt(252)
    
    # Sharpe ratio
    risk_free_rate = 0.05  # 5% annual
    sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
    benchmark_sharpe_ratio = (benchmark_annualized_return - risk_free_rate) / benchmark_volatility if benchmark_volatility > 0 else 0
    
    # Maximum drawdown
    daily_returns['running_max'] = daily_returns['cumulative_portfolio_return'].expanding().max()
    daily_returns['drawdown'] = (daily_returns['cumulative_portfolio_return'] - daily_returns['running_max']) / daily_returns['running_max']
    max_drawdown = daily_returns['drawdown'].min()
    
    # Win rate
    win_rate = (daily_returns['portfolio_return'] > 0).mean()
    
    # Information ratio (vs benchmark)
    excess_returns = daily_returns['portfolio_return'] - daily_returns['benchmark_return']
    information_ratio = excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
    
    # Beta and Alpha
    covariance = np.cov(daily_returns['portfolio_return'], daily_returns['benchmark_return'])[0, 1]
    benchmark_variance = daily_returns['benchmark_return'].var()
    beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0
    
    alpha = annualized_return - (risk_free_rate + beta * (benchmark_annualized_return - risk_free_rate))
    
    # Calmar ratio
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    print(f"   ✅ Performance metrics calculated successfully")
    
    return {
        'total_return': total_return,
        'benchmark_total_return': benchmark_total_return,
        'annualized_return': annualized_return,
        'benchmark_annualized_return': benchmark_annualized_return,
        'volatility': volatility,
        'benchmark_volatility': benchmark_volatility,
        'sharpe_ratio': sharpe_ratio,
        'benchmark_sharpe_ratio': benchmark_sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'information_ratio': information_ratio,
        'beta': beta,
        'alpha': alpha,
        'calmar_ratio': calmar_ratio,
        'days': days
    }

def main():
    print("🚀 Starting Complete Backtest with Performance Analysis")
    print("=" * 80)
    
    try:
        # Database connection
        db_manager = get_database_manager()
        engine = db_manager.get_engine()
        print("✅ Database connected")
        
        # Get available dates
        dates_query = f"""
        SELECT DISTINCT date
        FROM factor_scores_qvm
        WHERE date >= '{CONFIG['backtest_start_date']}'
        AND date <= '{CONFIG['backtest_end_date']}'
        AND strategy_version = 'qvm_v2.0_enhanced'
        ORDER BY date
        """
        
        dates_df = pd.read_sql(dates_query, engine)
        print(f"📅 Available dates: {len(dates_df)}")
        
        # Monthly rebalancing
        rebalance_dates = []
        current_month = None
        
        for date in dates_df['date']:
            date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
            month = date_str[:7]
            
            if month != current_month:
                rebalance_dates.append(date_str)
                current_month = month
        
        print(f"📊 Rebalancing dates: {len(rebalance_dates)}")
        
        # Process all dates (full backtest)
        portfolio_holdings = []
        
        print(f"🔄 Processing {len(rebalance_dates)} rebalancing periods...")
        
        for i, date in enumerate(rebalance_dates):
            if i % 20 == 0:
                print(f"   Progress: {i}/{len(rebalance_dates)} periods")
            
            # Load factor scores
            factor_query = f"""
            SELECT 
                ticker,
                Quality_Composite,
                Value_Composite,
                Momentum_Composite,
                QVM_Composite
            FROM factor_scores_qvm
            WHERE date = '{date}'
            AND strategy_version = 'qvm_v2.0_enhanced'
            ORDER BY QVM_Composite DESC
            LIMIT {CONFIG['top_n_stocks']}
            """
            
            factors_df = pd.read_sql(factor_query, engine)
            
            if not factors_df.empty:
                # Apply ranking normalization
                for col in ['Quality_Composite', 'Value_Composite', 'Momentum_Composite']:
                    factors_df[f'{col}_Rank'] = factors_df[col].rank(ascending=True, method='min')
                    factors_df[f'{col}_Normalized'] = (factors_df[f'{col}_Rank'] - 1) / (len(factors_df) - 1)
                
                # Create fixed QVM composite
                factors_df['QVM_Composite_Fixed'] = (
                    factors_df['Quality_Composite_Normalized'] * 0.3 +
                    factors_df['Value_Composite_Normalized'] * 0.4 +
                    factors_df['Momentum_Composite_Normalized'] * 0.3
                )
                
                # Record holdings
                for _, stock in factors_df.iterrows():
                    portfolio_holdings.append({
                        'date': date,
                        'ticker': stock['ticker'],
                        'quality_score': stock['Quality_Composite_Normalized'],
                        'value_score': stock['Value_Composite_Normalized'],
                        'momentum_score': stock['Momentum_Composite_Normalized'],
                        'qvm_score': stock['QVM_Composite_Fixed']
                    })
        
        # Convert to DataFrame
        holdings_df = pd.DataFrame(portfolio_holdings)
        print(f"✅ Portfolio holdings: {len(holdings_df)} records")
        
        # Load price data for performance calculation
        print("📊 Loading price data for performance calculation...")
        unique_tickers = holdings_df['ticker'].unique()
        ticker_list = "', '".join(unique_tickers)
        
        price_query = f"""
        SELECT 
            trading_date as date,
            ticker,
            close_price
        FROM vcsc_daily_data_complete
        WHERE ticker IN ('{ticker_list}')
        AND trading_date >= '{holdings_df['date'].min()}'
        AND trading_date <= '{holdings_df['date'].max()}'
        ORDER BY trading_date, ticker
        """
        
        price_data = pd.read_sql(price_query, engine)
        print(f"✅ Price data: {len(price_data)} records")
        
        # Load benchmark data (VNINDEX) from etf_history table
        benchmark_query = f"""
        SELECT 
            date,
            close as benchmark_close
        FROM etf_history
        WHERE ticker = 'VNINDEX'
        AND date >= '{holdings_df['date'].min()}'
        AND date <= '{holdings_df['date'].max()}'
        ORDER BY date
        """
        
        benchmark_data = pd.read_sql(benchmark_query, engine)
        print(f"✅ Benchmark data: {len(benchmark_data)} records")
        
        # Calculate returns
        portfolio_values, daily_returns = calculate_portfolio_returns(holdings_df, price_data, benchmark_data)
        
        # Calculate performance metrics
        performance_metrics = calculate_performance_metrics(portfolio_values, daily_returns, benchmark_data)
        
        # Generate comprehensive tearsheet
        generate_complete_tearsheet(holdings_df, portfolio_values, daily_returns, performance_metrics, CONFIG)
        
        # Save results
        results_dir = Path("insights")
        results_dir.mkdir(exist_ok=True)
        
        holdings_df.to_csv(results_dir / "18b_complete_holdings.csv", index=False)
        portfolio_values.to_csv(results_dir / "18b_complete_portfolio_values.csv", index=False)
        daily_returns.to_csv(results_dir / "18b_complete_daily_returns.csv", index=False)
        
        # Save performance metrics
        with open(results_dir / "18b_complete_performance_metrics.txt", 'w') as f:
            for metric, value in performance_metrics.items():
                f.write(f"{metric}: {value}\n")
        
        print(f"📁 Results saved to insights/")
        print(f"   - 18b_complete_holdings.csv: {len(holdings_df)} holdings")
        print(f"   - 18b_complete_portfolio_values.csv: {len(portfolio_values)} portfolio values")
        print(f"   - 18b_complete_daily_returns.csv: {len(daily_returns)} daily returns")
        print(f"   - 18b_complete_performance_metrics.txt: Performance metrics")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def generate_complete_tearsheet(holdings_df, portfolio_values, daily_returns, performance_metrics, config):
    """Generate comprehensive tearsheet with performance analysis."""
    print("\n📊 COMPREHENSIVE PERFORMANCE TEARSHEET")
    print("=" * 80)
    
    # Strategy Overview
    print("\n🎯 STRATEGY OVERVIEW")
    print("-" * 40)
    print(f"Strategy Name: {config['strategy_name']}")
    print(f"Backtest Period: {config['backtest_start_date']} to {config['backtest_end_date']}")
    print(f"Portfolio Size: {config['top_n_stocks']} stocks")
    print(f"Initial Capital: {config['initial_capital']:,} VND")
    print(f"Transaction Cost: {config['transaction_cost_bps']} bps")
    
    # Portfolio Statistics
    print("\n📊 PORTFOLIO STATISTICS")
    print("-" * 40)
    print(f"Total Holdings: {len(holdings_df)}")
    print(f"Unique Dates: {holdings_df['date'].nunique()}")
    print(f"Unique Tickers: {holdings_df['ticker'].nunique()}")
    print(f"Average Holdings per Date: {len(holdings_df) / holdings_df['date'].nunique():.1f}")
    
    # Performance Metrics
    print("\n📈 PERFORMANCE METRICS")
    print("-" * 40)
    if performance_metrics:
        print(f"Total Return: {performance_metrics['total_return']:.2%}")
        print(f"Benchmark Return: {performance_metrics['benchmark_total_return']:.2%}")
        print(f"Excess Return: {performance_metrics['total_return'] - performance_metrics['benchmark_total_return']:.2%}")
        print(f"Annualized Return: {performance_metrics['annualized_return']:.2%}")
        print(f"Benchmark Annualized: {performance_metrics['benchmark_annualized_return']:.2%}")
        print(f"Volatility: {performance_metrics['volatility']:.2%}")
        print(f"Benchmark Volatility: {performance_metrics['benchmark_volatility']:.2%}")
        print(f"Sharpe Ratio: {performance_metrics['sharpe_ratio']:.3f}")
        print(f"Benchmark Sharpe: {performance_metrics['benchmark_sharpe_ratio']:.3f}")
        print(f"Maximum Drawdown: {performance_metrics['max_drawdown']:.2%}")
        print(f"Win Rate: {performance_metrics['win_rate']:.2%}")
        print(f"Information Ratio: {performance_metrics['information_ratio']:.3f}")
        print(f"Beta: {performance_metrics['beta']:.3f}")
        print(f"Alpha: {performance_metrics['alpha']:.2%}")
        print(f"Calmar Ratio: {performance_metrics['calmar_ratio']:.3f}")
        print(f"Backtest Days: {performance_metrics['days']}")
    else:
        print("Performance metrics not available")
    
    # Factor Score Analysis
    print("\n📈 FACTOR SCORE ANALYSIS")
    print("-" * 40)
    print(f"Quality Score Range: {holdings_df['quality_score'].min():.3f} to {holdings_df['quality_score'].max():.3f}")
    print(f"Value Score Range: {holdings_df['value_score'].min():.3f} to {holdings_df['value_score'].max():.3f}")
    print(f"Momentum Score Range: {holdings_df['momentum_score'].min():.3f} to {holdings_df['momentum_score'].max():.3f}")
    print(f"QVM Score Range: {holdings_df['qvm_score'].min():.3f} to {holdings_df['qvm_score'].max():.3f}")
    
    # Top Holdings
    print("\n🏆 TOP HOLDINGS (Most Frequently Selected)")
    print("-" * 40)
    top_stocks = holdings_df['ticker'].value_counts().head(15)
    for ticker, count in top_stocks.items():
        percentage = count / holdings_df['date'].nunique() * 100
        print(f"{ticker}: {count} periods ({percentage:.1f}%)")
    
    # Portfolio Value Analysis
    if not portfolio_values.empty:
        print("\n💰 PORTFOLIO VALUE ANALYSIS")
        print("-" * 40)
        print(f"Final Portfolio Value: {portfolio_values['portfolio_value'].iloc[-1]:,.0f} VND")
        print(f"Portfolio Value Range: {portfolio_values['portfolio_value'].min():,.0f} to {portfolio_values['portfolio_value'].max():,.0f} VND")
        print(f"Average Portfolio Value: {portfolio_values['portfolio_value'].mean():,.0f} VND")
    
    # Risk Analysis
    print("\n⚠️ RISK ANALYSIS")
    print("-" * 40)
    if performance_metrics:
        if performance_metrics['sharpe_ratio'] > 1.0:
            print(f"✅ Sharpe Ratio: {performance_metrics['sharpe_ratio']:.3f} (Excellent > 1.0)")
        elif performance_metrics['sharpe_ratio'] > 0.5:
            print(f"✅ Sharpe Ratio: {performance_metrics['sharpe_ratio']:.3f} (Good > 0.5)")
        else:
            print(f"⚠️ Sharpe Ratio: {performance_metrics['sharpe_ratio']:.3f} (Needs improvement)")
        
        if performance_metrics['max_drawdown'] > -0.35:
            print(f"✅ Max Drawdown: {performance_metrics['max_drawdown']:.2%} (Good < -35%)")
        else:
            print(f"⚠️ Max Drawdown: {performance_metrics['max_drawdown']:.2%} (High > -35%)")
        
        if performance_metrics['win_rate'] > 0.55:
            print(f"✅ Win Rate: {performance_metrics['win_rate']:.2%} (Good > 55%)")
        else:
            print(f"⚠️ Win Rate: {performance_metrics['win_rate']:.2%} (Needs improvement)")
        
        if performance_metrics['information_ratio'] > 0.5:
            print(f"✅ Information Ratio: {performance_metrics['information_ratio']:.3f} (Good > 0.5)")
        else:
            print(f"⚠️ Information Ratio: {performance_metrics['information_ratio']:.3f} (Needs improvement)")
    
    print("\n✅ Complete tearsheet analysis completed")

if __name__ == "__main__":
    main()
