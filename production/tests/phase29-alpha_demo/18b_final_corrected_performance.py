#!/usr/bin/env python3

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add Project Root to Python Path
current_path = Path.cwd()
while not (current_path / 'production').is_dir():
    current_path = current_path.parent
project_root = current_path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from production.database.connection import get_database_manager
from sqlalchemy import text

print("✅ QVM Engine v3j Final Corrected Performance (v18b)")

# Configuration
CONFIG = {
    'strategy_name': 'QVM_Engine_v3j_Final_Corrected_v18b',
    'top_n_stocks': 20,
    'backtest_start_date': '2016-01-01',
    'backtest_end_date': '2025-12-31',
    'transaction_cost_bps': 10,
    'initial_capital': 10000000000  # 10 billion VND
}

def calculate_corrected_returns(holdings_df, price_data, benchmark_data):
    """Calculate corrected portfolio returns with proper trading day filtering."""
    print("📈 Calculating corrected portfolio returns...")
    
    # Convert dates to datetime
    holdings_df['date'] = pd.to_datetime(holdings_df['date'])
    price_data['date'] = pd.to_datetime(price_data['date'])
    benchmark_data['date'] = pd.to_datetime(benchmark_data['date'])
    
    # Create price matrix with forward filling
    print("   📊 Creating price matrix with forward filling...")
    price_matrix = price_data.pivot(index='date', columns='ticker', values='close_price')
    
    # Forward fill prices (carry last known price forward)
    price_matrix = price_matrix.fillna(method='ffill')
    
    # Backward fill any remaining NaN values at the beginning
    price_matrix = price_matrix.fillna(method='bfill')
    
    print(f"   ✅ Price matrix created: {price_matrix.shape}")
    
    # Get unique rebalancing dates
    unique_dates = sorted(holdings_df['date'].unique())
    
    portfolio_values = []
    daily_returns = []
    current_capital = CONFIG['initial_capital']
    
    for i, date in enumerate(unique_dates):
        # Get holdings for this date
        date_holdings = holdings_df[holdings_df['date'] == date]
        
        if date_holdings.empty:
            continue
        
        # Get prices for this date from the forward-filled matrix
        if date in price_matrix.index:
            date_prices = price_matrix.loc[date]
        else:
            # Find the closest available date
            available_dates = price_matrix.index[price_matrix.index <= date]
            if not available_dates.empty:
                closest_date = available_dates[-1]
                date_prices = price_matrix.loc[closest_date]
            else:
                continue
        
        # Calculate portfolio value
        portfolio_value = 0
        valid_holdings = 0
        
        for _, holding in date_holdings.iterrows():
            ticker = holding['ticker']
            if ticker in date_prices.index:
                price = date_prices[ticker]
                if pd.notna(price) and price > 0:
                    position_size = current_capital / len(date_holdings)
                    shares = position_size / price
                    portfolio_value += shares * price
                    valid_holdings += 1
        
        if portfolio_value > 0 and valid_holdings > 0:
            portfolio_values.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'capital': current_capital,
                'valid_holdings': valid_holdings,
                'total_holdings': len(date_holdings)
            })
            
            # Calculate daily returns for the period until next rebalancing
            if i < len(unique_dates) - 1:
                next_date = unique_dates[i + 1]
                
                # Get price data for the period (only trading days)
                period_dates = price_matrix.index[
                    (price_matrix.index >= date) & 
                    (price_matrix.index <= next_date)
                ]
                
                if len(period_dates) > 1:
                    # Calculate daily returns for each stock
                    period_prices = price_matrix.loc[period_dates]
                    
                    # Calculate daily returns (pct_change)
                    period_returns = period_prices.pct_change()
                    
                    # Calculate portfolio daily returns
                    for daily_date in period_returns.index[1:]:  # Skip first date (no return)
                        daily_returns_data = period_returns.loc[daily_date]
                        
                        # Get only the stocks in our portfolio
                        portfolio_tickers = date_holdings['ticker'].unique()
                        portfolio_daily_returns = daily_returns_data[daily_returns_data.index.isin(portfolio_tickers)]
                        
                        if not portfolio_daily_returns.empty:
                            # Filter out extreme returns (likely data errors)
                            portfolio_daily_returns = portfolio_daily_returns[
                                (portfolio_daily_returns >= -0.5) & (portfolio_daily_returns <= 0.5)
                            ]
                            
                            if len(portfolio_daily_returns) > 0:
                                # Equal weight portfolio return
                                portfolio_return = portfolio_daily_returns.mean()
                                
                                # Apply transaction costs on rebalancing day
                                if daily_date == date:
                                    transaction_cost = CONFIG['transaction_cost_bps'] / 10000
                                    portfolio_return -= transaction_cost
                                
                                # Only include valid returns (not NaN or extreme)
                                if pd.notna(portfolio_return) and abs(portfolio_return) < 0.5:
                                    daily_returns.append({
                                        'date': daily_date,
                                        'portfolio_return': portfolio_return,
                                        'rebalance_date': date
                                    })
            
            # Update capital for next period
            current_capital = portfolio_value
    
    portfolio_df = pd.DataFrame(portfolio_values)
    daily_returns_df = pd.DataFrame(daily_returns)
    
    print(f"   ✅ Portfolio values: {len(portfolio_df)} records")
    print(f"   ✅ Daily returns: {len(daily_returns_df)} records")
    
    return portfolio_df, daily_returns_df

def calculate_performance_metrics(portfolio_values, daily_returns, benchmark_data):
    """Calculate performance metrics with proper data handling."""
    print("📊 Calculating performance metrics...")
    
    if portfolio_values.empty or daily_returns.empty:
        print("   ⚠️ No data available for performance calculation")
        return {}
    
    # Process daily returns
    daily_returns = daily_returns.sort_values('date')
    daily_returns = daily_returns.dropna(subset=['portfolio_return'])
    
    # Filter out extreme returns
    daily_returns = daily_returns[
        (daily_returns['portfolio_return'] >= -0.5) & 
        (daily_returns['portfolio_return'] <= 0.5)
    ]
    
    if daily_returns.empty:
        print("   ⚠️ No valid daily returns")
        return {}
    
    # Merge with benchmark data
    daily_returns = daily_returns.merge(benchmark_data, on='date', how='left')
    daily_returns['benchmark_return'] = daily_returns['benchmark_close'].pct_change()
    daily_returns = daily_returns.dropna(subset=['portfolio_return', 'benchmark_return'])
    
    if daily_returns.empty:
        print("   ⚠️ No valid data after benchmark merge")
        return {}
    
    print(f"   📊 Valid daily returns: {len(daily_returns)} records")
    
    # Calculate metrics with proper validation
    total_return = (1 + daily_returns['portfolio_return']).prod() - 1
    benchmark_total_return = (1 + daily_returns['benchmark_return']).prod() - 1
    
    # Annualized return
    days = (pd.to_datetime(daily_returns['date'].iloc[-1]) - pd.to_datetime(daily_returns['date'].iloc[0])).days
    if days > 0:
        annualized_return = ((1 + total_return) ** (365 / days)) - 1
        benchmark_annualized_return = ((1 + benchmark_total_return) ** (365 / days)) - 1
    else:
        annualized_return = 0
        benchmark_annualized_return = 0
    
    # Volatility
    volatility = daily_returns['portfolio_return'].std() * np.sqrt(252)
    benchmark_volatility = daily_returns['benchmark_return'].std() * np.sqrt(252)
    
    # Sharpe ratio
    risk_free_rate = 0.05
    sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
    benchmark_sharpe_ratio = (benchmark_annualized_return - risk_free_rate) / benchmark_volatility if benchmark_volatility > 0 else 0
    
    # Maximum drawdown
    daily_returns['cumulative_return'] = (1 + daily_returns['portfolio_return']).cumprod()
    daily_returns['running_max'] = daily_returns['cumulative_return'].expanding().max()
    daily_returns['drawdown'] = (daily_returns['cumulative_return'] - daily_returns['running_max']) / daily_returns['running_max']
    max_drawdown = daily_returns['drawdown'].min()
    
    # Win rate
    win_rate = (daily_returns['portfolio_return'] > 0).mean()
    
    # Information ratio
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
    print("🚀 Starting Final Corrected Performance Analysis")
    print("=" * 80)
    
    try:
        # Database connection
        db_manager = get_database_manager()
        engine = db_manager.get_engine()
        print("✅ Database connected")
        
        # Load holdings data
        holdings_file = Path("insights/18b_complete_holdings.csv")
        if holdings_file.exists():
            holdings_df = pd.read_csv(holdings_file)
            print(f"✅ Loaded holdings: {len(holdings_df)} records")
        else:
            print("❌ Holdings file not found")
            return
        
        # Load price data
        print("📊 Loading price data...")
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
        
        # Load benchmark data
        print("📊 Loading benchmark data...")
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
        
        # Calculate returns with corrections
        portfolio_values, daily_returns = calculate_corrected_returns(holdings_df, price_data, benchmark_data)
        
        # Calculate performance metrics
        performance_metrics = calculate_performance_metrics(portfolio_values, daily_returns, benchmark_data)
        
        # Generate tearsheet
        generate_corrected_tearsheet(holdings_df, portfolio_values, daily_returns, performance_metrics, CONFIG)
        
        # Save results
        results_dir = Path("insights")
        results_dir.mkdir(exist_ok=True)
        
        portfolio_values.to_csv(results_dir / "18b_corrected_portfolio_values.csv", index=False)
        daily_returns.to_csv(results_dir / "18b_corrected_daily_returns.csv", index=False)
        
        # Save performance metrics
        with open(results_dir / "18b_corrected_performance_metrics.txt", 'w') as f:
            for metric, value in performance_metrics.items():
                f.write(f"{metric}: {value}\n")
        
        print(f"📁 Results saved to insights/")
        print(f"   - 18b_corrected_portfolio_values.csv: {len(portfolio_values)} portfolio values")
        print(f"   - 18b_corrected_daily_returns.csv: {len(daily_returns)} daily returns")
        print(f"   - 18b_corrected_performance_metrics.txt: Performance metrics")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def generate_corrected_tearsheet(holdings_df, portfolio_values, daily_returns, performance_metrics, config):
    """Generate comprehensive tearsheet."""
    print("\n📊 COMPREHENSIVE PERFORMANCE TEARSHEET (CORRECTED)")
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
    print("\n📈 PERFORMANCE METRICS (CORRECTED)")
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
        print(f"Average Valid Holdings: {portfolio_values['valid_holdings'].mean():.1f}/{portfolio_values['total_holdings'].mean():.1f}")
    
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
    
    print("\n✅ Corrected tearsheet analysis completed")

if __name__ == "__main__":
    main()
