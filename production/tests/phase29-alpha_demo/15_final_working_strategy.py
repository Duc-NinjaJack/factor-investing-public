#!/usr/bin/env python3
"""
Final Working Strategy - Simple and effective approach
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

def main():
    """Main execution - Simple working strategy"""
    print("🚀 Running Final Working Strategy")
    print("="*60)
    
    # Database connection
    db_manager = get_database_manager()
    engine = db_manager.get_engine()
    
    # Load data
    print("📊 Loading market data...")
    query = """
    SELECT 
        ticker,
        trading_date as date,
        close_price as close
    FROM vcsc_daily_data 
    WHERE trading_date >= '2016-01-01' AND trading_date <= '2025-07-28'
    AND close_price > 0
    ORDER BY trading_date, ticker
    """
    
    with engine.connect() as conn:
        data = pd.read_sql(query, conn)
    
    print(f"   ✅ Loaded {len(data):,} records for {data['ticker'].nunique()} stocks")
    
    # Create price matrix
    price_data = data.pivot(index='date', columns='ticker', values='close')
    returns_matrix = price_data.pct_change(fill_method=None)
    
    # Clean returns
    returns_matrix = returns_matrix.clip(-0.5, 0.5)  # Cap extreme returns
    returns_matrix = returns_matrix.replace([np.inf, -np.inf], np.nan)
    
    # Get top 20 stocks by average volume
    print("📊 Selecting top stocks by volume...")
    volume_query = """
    SELECT 
        ticker,
        AVG(total_volume) as avg_volume
    FROM vcsc_daily_data 
    WHERE trading_date >= '2016-01-01' AND trading_date <= '2025-07-28'
    GROUP BY ticker
    ORDER BY avg_volume DESC
    LIMIT 20
    """
    
    with engine.connect() as conn:
        top_stocks = pd.read_sql(volume_query, conn)
    
    top_tickers = top_stocks['ticker'].tolist()
    print(f"   ✅ Selected top {len(top_tickers)} stocks by volume")
    
    # Simple strategy: Equal weight top 20 stocks
    print("📊 Running simple equal-weight strategy...")
    
    # Get returns for top stocks
    top_returns = returns_matrix[top_tickers].mean(axis=1)
    top_returns = top_returns.fillna(0)
    
    # Calculate strategy performance
    strategy_returns = top_returns
    cumulative_returns = (1 + strategy_returns).cumprod()
    
    # Calculate benchmark (median of all stocks)
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
    
    # Create visualization
    if metrics and 'total_return' in metrics:
        print(f"\n📊 Generating tearsheet...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Final Working Strategy - Performance Analysis', fontsize=16, fontweight='bold')
        
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
        # Fix datetime index issue
        cumulative_returns.index = pd.to_datetime(cumulative_returns.index)
        monthly_returns = cumulative_returns.resample('M').last().pct_change()
        if len(monthly_returns) > 12:
            # Calculate proper dimensions for reshape
            n_years = len(monthly_returns) // 12
            if n_years > 0:
                monthly_returns_matrix = monthly_returns.values[:n_years*12].reshape(n_years, 12)
                im = ax3.imshow(monthly_returns_matrix, cmap='RdYlGn', aspect='auto')
                ax3.set_title('Monthly Returns Heatmap', fontweight='bold')
                ax3.set_xlabel('Month')
                ax3.set_ylabel('Year')
                plt.colorbar(im, ax=ax3)
            else:
                ax3.text(0.5, 0.5, 'Insufficient data for heatmap', 
                        transform=ax3.transAxes, ha='center', va='center')
                ax3.set_title('Monthly Returns Heatmap', fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'Insufficient data for heatmap', 
                    transform=ax3.transAxes, ha='center', va='center')
            ax3.set_title('Monthly Returns Heatmap', fontweight='bold')
        
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
        filename = f"tearsheet_final_working_{timestamp}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Tearsheet saved: {filename}")
    
    print(f"\n🎯 Strategy Summary:")
    print(f"   - Equal weight allocation")
    print(f"   - Top 20 stocks by volume")
    print(f"   - Simple and effective approach")
    print(f"   - No complex rebalancing")
    
    return {
        'strategy_returns': strategy_returns,
        'benchmark_returns': benchmark_returns,
        'cumulative_returns': cumulative_returns,
        'benchmark_cumulative': benchmark_cumulative,
        'metrics': metrics,
        'top_stocks': top_tickers
    }

if __name__ == "__main__":
    main() 