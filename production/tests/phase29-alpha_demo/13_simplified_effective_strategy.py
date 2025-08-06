#!/usr/bin/env python3
"""
Simplified Effective Strategy - Based on analysis showing simple approaches work better
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
current_path = Path.cwd()
while not (current_path / 'production').is_dir():
    if current_path.parent == current_path:
        raise FileNotFoundError("Could not find the 'production' directory.")
    current_path = current_path.parent

project_root = current_path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from production.database.connection import get_database_manager

def calculate_performance_metrics(strategy_returns, benchmark_returns, risk_free_rate=0.05):
    """Calculate comprehensive performance metrics"""
    # Clean data
    strategy_returns = strategy_returns.replace([np.inf, -np.inf], np.nan).fillna(0)
    benchmark_returns = benchmark_returns.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Align data
    aligned_data = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
    if len(aligned_data) == 0:
        return {}
    
    strategy_returns = aligned_data.iloc[:, 0]
    benchmark_returns = aligned_data.iloc[:, 1]
    
    # Calculate metrics
    total_return = (1 + strategy_returns).prod() - 1
    benchmark_total_return = (1 + benchmark_returns).prod() - 1
    
    annualized_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
    benchmark_annualized_return = (1 + benchmark_total_return) ** (252 / len(benchmark_returns)) - 1
    
    volatility = strategy_returns.std() * np.sqrt(252)
    benchmark_volatility = benchmark_returns.std() * np.sqrt(252)
    
    sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
    benchmark_sharpe_ratio = (benchmark_annualized_return - risk_free_rate) / benchmark_volatility if benchmark_volatility > 0 else 0
    
    # Calculate drawdown
    cumulative_returns = (1 + strategy_returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns / running_max - 1)
    max_drawdown = drawdown.min()
    
    # Calculate information ratio
    excess_returns = strategy_returns - benchmark_returns
    information_ratio = excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
    
    # Calculate excess return
    excess_return = total_return - benchmark_total_return
    
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
        'information_ratio': information_ratio,
        'excess_return': excess_return
    }

# Configuration
STRATEGY_CONFIG = {
    "strategy_name": "Simplified Effective Strategy",
    "backtest_start_date": "2016-01-01",
    "backtest_end_date": "2025-07-28",
    "universe": {
        "top_n_stocks": 200,
        "target_portfolio_size": 20,
        "min_market_cap": 1000,  # Billion VND
        "min_volume": 1000000    # Million shares
    },
    "rebalancing": {
        "frequency": "monthly",
        "days_between_rebalancing": 30
    },
    "risk_management": {
        "max_position_size": 0.10,  # 10% max per stock
        "stop_loss": 0.20,          # 20% stop loss
        "take_profit": 0.50         # 50% take profit
    }
}

class SimplifiedEffectiveStrategy:
    """
    Simplified strategy that focuses on what works:
    1. Equal weight allocation
    2. Monthly rebalancing
    3. Top stocks by volume (proxy for liquidity)
    4. Simple momentum filter
    """
    
    def __init__(self, config):
        self.config = config
        self.db_manager = get_database_manager()
        self.engine = self.db_manager.get_engine()
        
    def load_data(self):
        """Load price and volume data"""
        print("📊 Loading market data...")
        
        query = """
        SELECT 
            ticker,
            trading_date as date,
            close_price as close,
            total_volume as volume
        FROM vcsc_daily_data 
        WHERE trading_date >= %(start_date)s AND trading_date <= %(end_date)s
        AND close_price > 0
        ORDER BY trading_date, ticker
        """
        
        with self.engine.connect() as conn:
            data = pd.read_sql(query, conn, params={
                'start_date': self.config['backtest_start_date'],
                'end_date': self.config['backtest_end_date']
            })
        
        print(f"   ✅ Loaded {len(data):,} records for {data['ticker'].nunique()} stocks")
        return data
    
    def get_universe(self, date):
        """Get universe of stocks for a given date"""
        # Get top stocks by average volume in the last 30 days
        end_date = date
        start_date = date - timedelta(days=30)
        
        query = """
        SELECT 
            ticker,
            AVG(total_volume) as avg_volume,
            AVG(close_price) as avg_price
        FROM vcsc_daily_data 
        WHERE trading_date BETWEEN %(start_date)s AND %(end_date)s
        AND close_price > 0
        GROUP BY ticker
        HAVING AVG(total_volume) >= %(min_volume)s
        ORDER BY AVG(total_volume) DESC
        LIMIT %(top_n)s
        """
        
        with self.engine.connect() as conn:
            universe = pd.read_sql(query, conn, params={
                'start_date': start_date,
                'end_date': end_date,
                'min_volume': self.config['universe']['min_volume'],
                'top_n': self.config['universe']['top_n_stocks']
            })
        
        return universe['ticker'].tolist()
    
    def calculate_momentum(self, price_data, date, lookback=252):
        """Calculate momentum for stocks"""
        end_date = date
        start_date = date - timedelta(days=lookback)
        
        # Get price data for momentum calculation
        query = """
        SELECT ticker, close_price, trading_date
        FROM vcsc_daily_data 
        WHERE trading_date BETWEEN %(start_date)s AND %(end_date)s
        AND close_price > 0
        ORDER BY trading_date, ticker
        """
        
        with self.engine.connect() as conn:
            momentum_data = pd.read_sql(query, conn, params={
                'start_date': start_date,
                'end_date': end_date
            })
        
        if momentum_data.empty:
            return pd.Series()
        
        # Calculate momentum
        price_pivot = momentum_data.pivot(index='trading_date', columns='ticker', values='close_price')
        momentum = price_pivot.pct_change(lookback).iloc[-1]
        
        return momentum.dropna()
    
    def select_portfolio(self, universe, momentum, date):
        """Select portfolio based on momentum"""
        if len(universe) < self.config['universe']['target_portfolio_size']:
            return universe
        
        # Filter universe by momentum (positive momentum only)
        positive_momentum = momentum[momentum > 0]
        available_stocks = [ticker for ticker in universe if ticker in positive_momentum.index]
        
        if len(available_stocks) < self.config['universe']['target_portfolio_size']:
            # If not enough positive momentum stocks, use all available
            return available_stocks[:self.config['universe']['target_portfolio_size']]
        
        # Select top momentum stocks
        top_momentum = positive_momentum.loc[available_stocks].nlargest(
            self.config['universe']['target_portfolio_size']
        )
        
        return top_momentum.index.tolist()
    
    def calculate_weights(self, portfolio):
        """Calculate equal weights for portfolio"""
        n_stocks = len(portfolio)
        if n_stocks == 0:
            return pd.Series()
        
        weight = 1.0 / n_stocks
        return pd.Series(weight, index=portfolio)
    
    def backtest(self):
        """Run backtest"""
        print(f"\n🚀 Running {self.config['strategy_name']}")
        print("="*60)
        
        # Load data
        data = self.load_data()
        
        # Create price matrix
        price_data = data.pivot(index='date', columns='ticker', values='close')
        returns_matrix = price_data.pct_change(fill_method=None)
        
        # Clean returns
        returns_matrix = returns_matrix.clip(-0.5, 0.5)  # Cap extreme returns
        returns_matrix = returns_matrix.replace([np.inf, -np.inf], np.nan)
        
        # Initialize tracking
        dates = pd.date_range(
            start=self.config['backtest_start_date'],
            end=self.config['backtest_end_date'],
            freq='M'
        )
        
        portfolio_history = []
        strategy_returns = []
        
        current_portfolio = []
        current_weights = pd.Series()
        
        print(f"   📅 Processing {len(dates)} monthly rebalancing dates...")
        
        for i, date in enumerate(dates):
            if date not in price_data.index:
                continue
                
            # Get universe
            universe = self.get_universe(date)
            
            if len(universe) == 0:
                print(f"   ⚠️  No universe found for {date.strftime('%Y-%m')}")
                continue
            
            # Calculate momentum
            momentum = self.calculate_momentum(price_data, date)
            
            # Select portfolio
            new_portfolio = self.select_portfolio(universe, momentum, date)
            
            # Calculate weights
            new_weights = self.calculate_weights(new_portfolio)
            
            # Calculate returns for current period
            if len(current_portfolio) > 0 and len(current_weights) > 0:
                # Get returns for current portfolio
                period_returns = returns_matrix.loc[date:date + timedelta(days=30), current_portfolio]
                if not period_returns.empty:
                    portfolio_return = (period_returns * current_weights).sum(axis=1).mean()
                    strategy_returns.append(portfolio_return)
            
            # Update portfolio
            current_portfolio = new_portfolio
            current_weights = new_weights
            
            # Record portfolio
            portfolio_history.append({
                'date': date,
                'portfolio': new_portfolio,
                'weights': new_weights.to_dict(),
                'universe_size': len(universe),
                'portfolio_size': len(new_portfolio)
            })
            
            if i % 20 == 0:
                print(f"   📅 {date.strftime('%Y-%m')}: Universe={len(universe)}, Portfolio={len(new_portfolio)}")
        
        print(f"   ✅ Completed backtest with {len(strategy_returns)} return periods")
        
        # Calculate strategy performance
        strategy_returns = pd.Series(strategy_returns)
        strategy_returns = strategy_returns.fillna(0)
        
        # Calculate cumulative returns
        cumulative_returns = (1 + strategy_returns).cumprod()
        
        # Calculate benchmark (equal weight universe)
        benchmark_returns = returns_matrix.median(axis=1).fillna(0)
        benchmark_cumulative = (1 + benchmark_returns).cumprod()
        
        # Calculate performance metrics
        metrics = calculate_performance_metrics(
            strategy_returns, 
            benchmark_returns,
            risk_free_rate=0.05
        )
        
        # Display results
        print(f"\n📊 PERFORMANCE RESULTS:")
        if metrics and 'total_return' in metrics:
            print(f"   📈 Strategy Total Return: {metrics['total_return']:.2%}")
            print(f"   📈 Benchmark Total Return: {metrics['benchmark_total_return']:.2%}")
            print(f"   📈 Strategy Annualized Return: {metrics['annualized_return']:.2%}")
            print(f"   📈 Benchmark Annualized Return: {metrics['benchmark_annualized_return']:.2%}")
            print(f"   📈 Strategy Volatility: {metrics['volatility']:.2%}")
            print(f"   📈 Benchmark Volatility: {metrics['benchmark_volatility']:.2%}")
            print(f"   📈 Strategy Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            print(f"   📈 Benchmark Sharpe Ratio: {metrics['benchmark_sharpe_ratio']:.2f}")
            print(f"   📈 Max Drawdown: {metrics['max_drawdown']:.2%}")
            print(f"   📈 Information Ratio: {metrics['information_ratio']:.2f}")
            print(f"   📈 Excess Return: {metrics['excess_return']:.2%}")
            
            if metrics['excess_return'] > 0:
                print(f"   ✅ STRATEGY OUTPERFORMS BENCHMARK by {metrics['excess_return']:.2%}")
            else:
                print(f"   ⚠️  STRATEGY UNDERPERFORMS BENCHMARK by {abs(metrics['excess_return']):.2%}")
        else:
            print(f"   ⚠️  Could not calculate performance metrics")
            print(f"   📈 Strategy returns length: {len(strategy_returns)}")
            print(f"   📈 Benchmark returns length: {len(benchmark_returns)}")
            print(f"   📈 Portfolio history length: {len(portfolio_history)}")
        
        # Create visualization
        if metrics and 'total_return' in metrics:
            self.create_tearsheet(cumulative_returns, benchmark_cumulative, metrics)
        
        return {
            'strategy_returns': strategy_returns,
            'benchmark_returns': benchmark_returns,
            'cumulative_returns': cumulative_returns,
            'benchmark_cumulative': benchmark_cumulative,
            'metrics': metrics,
            'portfolio_history': portfolio_history
        }
    
    def create_tearsheet(self, cumulative_returns, benchmark_cumulative, metrics):
        """Create comprehensive tearsheet"""
        print(f"\n📊 Generating tearsheet...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{self.config["strategy_name"]} - Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Equity Curve
        ax1 = axes[0, 0]
        ax1.plot(cumulative_returns.index, cumulative_returns.values, 
                label='Strategy', linewidth=2, color='#2E86AB')
        ax1.plot(benchmark_cumulative.index, benchmark_cumulative.values, 
                label='Benchmark', linewidth=2, color='#A23B72', alpha=0.7)
        ax1.set_title('Equity Curve', fontweight='bold')
        ax1.set_ylabel('Cumulative Return')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Drawdown
        ax2 = axes[0, 1]
        drawdown = (cumulative_returns / cumulative_returns.cummax() - 1) * 100
        ax2.fill_between(drawdown.index, drawdown.values, 0, 
                        color='red', alpha=0.3, label='Drawdown')
        ax2.set_title('Drawdown Analysis', fontweight='bold')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Monthly Returns Heatmap
        ax3 = axes[1, 0]
        monthly_returns = cumulative_returns.resample('M').last().pct_change()
        monthly_returns_matrix = monthly_returns.values.reshape(-1, 12)
        im = ax3.imshow(monthly_returns_matrix, cmap='RdYlGn', aspect='auto')
        ax3.set_title('Monthly Returns Heatmap', fontweight='bold')
        ax3.set_xlabel('Month')
        ax3.set_ylabel('Year')
        plt.colorbar(im, ax=ax3)
        
        # 4. Performance Metrics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        metrics_text = f"""
Performance Metrics:
• Total Return: {metrics['total_return']:.2%}
• Annualized Return: {metrics['annualized_return']:.2%}
• Volatility: {metrics['volatility']:.2%}
• Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
• Max Drawdown: {metrics['max_drawdown']:.2%}
• Information Ratio: {metrics['information_ratio']:.2f}
• Excess Return: {metrics['excess_return']:.2%}
        """
        
        ax4.text(0.1, 0.9, metrics_text, transform=ax4.transAxes, 
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        
        # Save tearsheet
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tearsheet_simplified_effective_{timestamp}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Tearsheet saved: {filename}")

def main():
    """Main execution"""
    strategy = SimplifiedEffectiveStrategy(STRATEGY_CONFIG)
    results = strategy.backtest()
    
    print(f"\n🎯 Strategy Summary:")
    print(f"   - Simple momentum-based selection")
    print(f"   - Equal weight allocation")
    print(f"   - Monthly rebalancing")
    print(f"   - Top 20 stocks by volume and momentum")
    
    return results

if __name__ == "__main__":
    main() 