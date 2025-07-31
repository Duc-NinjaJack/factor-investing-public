#!/usr/bin/env python3
"""
================================================================================
Phase 24: Value-Only Factor Backtesting & Tearsheet Analysis
================================================================================
Purpose:
    Comprehensive analysis of pure value factor strategy compared against benchmark
    using the ValueOnlyBacktesting subclass for pure value factor analysis.

Methodology:
    1. Load and validate value-only backtest results and benchmark data
    2. Generate comprehensive performance analysis vs benchmark
    3. Create institutional-grade tearsheet with benchmark comparison
    4. Provide strategic insights and recommendations for value factor strategy

Author: Quantitative Strategy Team
Date: January 2025
Status: PRODUCTION READY
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import pickle
import logging
from pathlib import Path
import sys

# Add scripts directory to path
scripts_dir = Path(__file__).parent.parent.parent / 'scripts'
sys.path.append(str(scripts_dir))

# Import our value-only backtesting engine
from value_only_backtesting import ValueOnlyBacktesting

# Setup
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    "backtest_start": "2017-12-01",
    "backtest_end": "2025-07-28",
    "rebalance_freq": "M",
    "transaction_cost_bps": 20,
    "portfolio_size": 25
}

def calculate_performance_metrics(returns, benchmark, risk_free_rate=0.0):
    """Calculate comprehensive performance metrics vs benchmark."""
    common_index = returns.index.intersection(benchmark.index)
    returns, benchmark = returns.loc[common_index], benchmark.loc[common_index]
    
    n_years = len(returns) / 252
    
    # Basic metrics
    total_return = (1 + returns).prod() - 1
    annual_return = (1 + total_return) ** (1 / n_years) - 1
    annual_volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
    
    # Drawdown analysis
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Benchmark comparison
    benchmark_total_return = (1 + benchmark).prod() - 1
    benchmark_annual_return = (1 + benchmark_total_return) ** (1 / n_years) - 1
    benchmark_volatility = benchmark.std() * np.sqrt(252)
    
    # Alpha and Beta calculation
    excess_returns = returns - benchmark
    beta = np.cov(returns, benchmark)[0, 1] / np.var(benchmark)
    alpha = annual_return - (risk_free_rate + beta * (benchmark_annual_return - risk_free_rate))
    
    # Information ratio
    tracking_error = excess_returns.std() * np.sqrt(252)
    information_ratio = excess_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0
    
    # Win rate and other metrics
    win_rate = (returns > 0).mean()
    benchmark_win_rate = (benchmark > 0).mean()
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'benchmark_total_return': benchmark_total_return,
        'benchmark_annual_return': benchmark_annual_return,
        'benchmark_volatility': benchmark_volatility,
        'alpha': alpha,
        'beta': beta,
        'information_ratio': information_ratio,
        'tracking_error': tracking_error,
        'win_rate': win_rate,
        'benchmark_win_rate': benchmark_win_rate,
        'excess_return': annual_return - benchmark_annual_return
    }

def calculate_risk_metrics(returns):
    """Calculate comprehensive risk metrics."""
    n_years = len(returns) / 252
    
    # Basic statistics
    mean_return = returns.mean() * 252
    std_return = returns.std() * np.sqrt(252)
    skewness = returns.skew()
    kurtosis = returns.kurtosis()
    
    # VaR and CVaR
    var_95 = np.percentile(returns, 5)
    cvar_95 = returns[returns <= var_95].mean()
    
    # Calmar ratio
    total_return = (1 + returns).prod() - 1
    annual_return = (1 + total_return) ** (1 / n_years) - 1
    
    # Calculate max drawdown
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    return {
        'mean_return': mean_return,
        'std_return': std_return,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'var_95': var_95,
        'cvar_95': cvar_95,
        'calmar_ratio': calmar_ratio
    }

def create_visualizations(value_strategy_returns, benchmark_returns, save_path='phase24_value_only_plots.png'):
    """Create comprehensive visualizations for value-only strategy."""
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Phase 24: Value-Only Factor Strategy Analysis', fontsize=16, fontweight='bold')
    
    # 1. Cumulative Returns Comparison
    cumulative_value = (1 + value_strategy_returns).cumprod()
    cumulative_benchmark = (1 + benchmark_returns).cumprod()
    
    axes[0, 0].plot(cumulative_value.index, cumulative_value.values, label='Value Strategy', linewidth=2, color='blue')
    axes[0, 0].plot(cumulative_benchmark.index, cumulative_benchmark.values, label='VNINDEX Benchmark', linewidth=2, color='red', alpha=0.7)
    axes[0, 0].set_title('Cumulative Returns Comparison')
    axes[0, 0].set_ylabel('Cumulative Return')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Rolling Sharpe Ratio
    rolling_sharpe = value_strategy_returns.rolling(window=252).mean() / value_strategy_returns.rolling(window=252).std() * np.sqrt(252)
    axes[0, 1].plot(rolling_sharpe.index, rolling_sharpe.values, color='green', linewidth=2)
    axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[0, 1].set_title('Rolling 1-Year Sharpe Ratio')
    axes[0, 1].set_ylabel('Sharpe Ratio')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Drawdown Analysis
    running_max = cumulative_value.expanding().max()
    drawdown = (cumulative_value - running_max) / running_max
    axes[1, 0].fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3)
    axes[1, 0].plot(drawdown.index, drawdown.values, color='red', linewidth=1)
    axes[1, 0].set_title('Drawdown Analysis')
    axes[1, 0].set_ylabel('Drawdown')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Rolling Beta
    rolling_beta = value_strategy_returns.rolling(window=252).cov(benchmark_returns) / benchmark_returns.rolling(window=252).var()
    axes[1, 1].plot(rolling_beta.index, rolling_beta.values, color='purple', linewidth=2)
    axes[1, 1].axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Beta = 1')
    axes[1, 1].set_title('Rolling 1-Year Beta vs VNINDEX')
    axes[1, 1].set_ylabel('Beta')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Visualizations saved to {save_path}")

def main():
    """Main execution function."""
    print("🚀 PHASE 24: VALUE-ONLY FACTOR BACKTESTING & TEARSHEET ANALYSIS")
    print(f"📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🎯 Strategy: Pure Value Factor (Value_Composite only)")
    print("=" * 80)
    
    try:
        # Run value-only backtest
        print("📂 Running value-only factor backtest...")
        
        # Initialize value-only backtesting engine
        value_backtesting = ValueOnlyBacktesting()
        
        # Customize configuration for value-only analysis
        value_backtesting.backtest_config.update({
            'start_date': CONFIG['backtest_start'],
            'end_date': CONFIG['backtest_end'],
            'rebalance_freq': CONFIG['rebalance_freq'],
            'portfolio_size': CONFIG['portfolio_size'],
            'transaction_cost': CONFIG['transaction_cost_bps'] / 10000  # Convert bps to decimal
        })
        
        # Run complete analysis
        results = value_backtesting.run_complete_analysis(
            save_plots=True,
            save_report=True
        )
        
        backtest_results = results['backtest_results']
        prepared_data = results['prepared_data']
        
        print("✅ Value-only backtest completed successfully")
        
        # Extract strategy returns and benchmark
        print("📊 Extracting backtest results...")
        
        # Get the first threshold result (assuming we have at least one)
        threshold_name = list(backtest_results.keys())[0]
        value_strategy_returns = backtest_results[threshold_name]['returns']
        benchmark_returns = prepared_data['benchmark_returns']
        
        print(f"Value strategy returns: {len(value_strategy_returns)} observations")
        print(f"Benchmark returns: {len(benchmark_returns)} observations")
        
        # Align data to common date range
        common_dates = value_strategy_returns.index.intersection(benchmark_returns.index)
        value_strategy_returns = value_strategy_returns.loc[common_dates]
        benchmark_returns = benchmark_returns.loc[common_dates]
        
        print(f"Aligned data: {len(common_dates)} observations from {common_dates.min()} to {common_dates.max()}")
        
        # Calculate performance metrics
        print("📊 Calculating performance metrics...")
        value_metrics = calculate_performance_metrics(value_strategy_returns, benchmark_returns)
        
        # Create performance summary table
        print("📋 Creating performance summary table...")
        performance_summary = pd.DataFrame({
            'Metric': [
                'Total Return',
                'Annual Return',
                'Annual Volatility',
                'Sharpe Ratio',
                'Maximum Drawdown',
                'Alpha',
                'Beta',
                'Information Ratio',
                'Tracking Error',
                'Win Rate',
                'Excess Return vs Benchmark'
            ],
            'Value Strategy': [
                f"{value_metrics['total_return']:.2%}",
                f"{value_metrics['annual_return']:.2%}",
                f"{value_metrics['annual_volatility']:.2%}",
                f"{value_metrics['sharpe_ratio']:.2f}",
                f"{value_metrics['max_drawdown']:.2%}",
                f"{value_metrics['alpha']:.2%}",
                f"{value_metrics['beta']:.2f}",
                f"{value_metrics['information_ratio']:.2f}",
                f"{value_metrics['tracking_error']:.2%}",
                f"{value_metrics['win_rate']:.2%}",
                f"{value_metrics['excess_return']:.2%}"
            ],
            'Benchmark (VNINDEX)': [
                f"{value_metrics['benchmark_total_return']:.2%}",
                f"{value_metrics['benchmark_annual_return']:.2%}",
                f"{value_metrics['benchmark_volatility']:.2%}",
                "N/A",
                "N/A",
                "N/A",
                "1.00",
                "N/A",
                "N/A",
                f"{value_metrics['benchmark_win_rate']:.2%}",
                "N/A"
            ]
        })
        
        print("\n📊 PERFORMANCE SUMMARY - VALUE-ONLY FACTOR STRATEGY")
        print("=" * 80)
        print(performance_summary.to_string(index=False))
        print("=" * 80)
        
        # Create visualizations
        print("📈 Creating comprehensive visualizations...")
        create_visualizations(value_strategy_returns, benchmark_returns)
        
        # Risk analysis
        print("📊 Creating risk analysis...")
        value_risk = calculate_risk_metrics(value_strategy_returns)
        benchmark_risk = calculate_risk_metrics(benchmark_returns)
        
        # Create risk comparison table
        risk_comparison = pd.DataFrame({
            'Risk Metric': [
                'Annualized Mean Return',
                'Annualized Volatility',
                'Skewness',
                'Kurtosis',
                'VaR (95%)',
                'CVaR (95%)',
                'Calmar Ratio'
            ],
            'Value Strategy': [
                f"{value_risk['mean_return']:.2%}",
                f"{value_risk['std_return']:.2%}",
                f"{value_risk['skewness']:.3f}",
                f"{value_risk['kurtosis']:.3f}",
                f"{value_risk['var_95']:.2%}",
                f"{value_risk['cvar_95']:.2%}",
                f"{value_risk['calmar_ratio']:.3f}"
            ],
            'Benchmark': [
                f"{benchmark_risk['mean_return']:.2%}",
                f"{benchmark_risk['std_return']:.2%}",
                f"{benchmark_risk['skewness']:.3f}",
                f"{benchmark_risk['kurtosis']:.3f}",
                f"{benchmark_risk['var_95']:.2%}",
                f"{benchmark_risk['cvar_95']:.2%}",
                f"{benchmark_risk['calmar_ratio']:.3f}"
            ]
        })
        
        print("\n📊 RISK ANALYSIS COMPARISON")
        print("=" * 80)
        print(risk_comparison.to_string(index=False))
        print("=" * 80)
        
        # Save results
        print("💾 Saving results...")
        phase24_results = {
            'backtest_results': backtest_results,
            'prepared_data': prepared_data,
            'value_strategy_returns': value_strategy_returns,
            'benchmark_returns': benchmark_returns,
            'performance_metrics': value_metrics,
            'risk_metrics': value_risk,
            'config': CONFIG,
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open('phase24_value_only_results.pkl', 'wb') as f:
            pickle.dump(phase24_results, f)
        
        print("✅ Results saved to phase24_value_only_results.pkl")
        
        # Generate summary report
        print("\n" + "="*80)
        print("🎯 PHASE 24: VALUE-ONLY FACTOR STRATEGY SUMMARY")
        print("="*80)
        print(f"📅 Analysis Period: {CONFIG['backtest_start']} to {CONFIG['backtest_end']}")
        print(f"🎯 Strategy: Pure Value Factor (Value_Composite only)")
        print(f"📊 Portfolio Size: {CONFIG['portfolio_size']} stocks")
        print(f"🔄 Rebalancing: {CONFIG['rebalance_freq']}")
        print(f"💰 Transaction Cost: {CONFIG['transaction_cost_bps']} bps")
        print(f"\n📈 Key Performance Metrics:")
        print(f"   • Annual Return: {value_metrics['annual_return']:.2%}")
        print(f"   • Sharpe Ratio: {value_metrics['sharpe_ratio']:.2f}")
        print(f"   • Max Drawdown: {value_metrics['max_drawdown']:.2%}")
        print(f"   • Alpha: {value_metrics['alpha']:.2%}")
        print(f"   • Beta: {value_metrics['beta']:.2f}")
        print(f"   • Information Ratio: {value_metrics['information_ratio']:.2f}")
        print(f"\n🏆 vs Benchmark (VNINDEX):")
        print(f"   • Excess Return: {value_metrics['excess_return']:.2%}")
        print(f"   • Benchmark Return: {value_metrics['benchmark_annual_return']:.2%}")
        print("="*80)
        print("✅ Phase 24 Value-Only Factor Analysis Completed Successfully!")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        raise

if __name__ == "__main__":
    main() 