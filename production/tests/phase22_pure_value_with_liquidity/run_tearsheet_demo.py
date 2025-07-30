#!/usr/bin/env python3
"""
Phase 22: Weighted Composite Backtesting - Demo Tearsheet

This script demonstrates the comprehensive tearsheet analysis for the Phase 22
weighted composite strategy, comparing against the VNINDEX benchmark.
Uses sample data to show the full analysis structure.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import logging

# Setup
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_sample_data():
    """Generate realistic sample data for demonstration."""
    print("📊 Generating sample data for Phase 22 weighted composite strategy...")
    
    # Generate date range
    start_date = datetime(2018, 1, 1)
    end_date = datetime(2024, 12, 31)
    dates = pd.date_range(start_date, end_date, freq='D')
    
    # Filter to trading days (weekdays)
    trading_dates = dates[dates.weekday < 5]
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate benchmark returns (VNINDEX) - realistic Vietnamese market
    benchmark_daily_return = 0.0006  # ~15% annual
    benchmark_vol = 0.018  # ~28% annual volatility
    benchmark_returns = pd.Series(
        np.random.normal(benchmark_daily_return, benchmark_vol, len(trading_dates)),
        index=trading_dates
    )
    
    # Generate strategy returns with alpha
    # 10B VND strategy: moderate alpha, lower volatility
    strategy_10b_alpha = 0.0002  # ~5% annual alpha
    strategy_10b_vol = 0.020  # ~32% annual volatility
    strategy_10b_returns = pd.Series(
        np.random.normal(benchmark_daily_return + strategy_10b_alpha, strategy_10b_vol, len(trading_dates)),
        index=trading_dates
    )
    
    # 3B VND strategy: higher alpha, higher volatility
    strategy_3b_alpha = 0.0004  # ~10% annual alpha
    strategy_3b_vol = 0.022  # ~35% annual volatility
    strategy_3b_returns = pd.Series(
        np.random.normal(benchmark_daily_return + strategy_3b_alpha, strategy_3b_vol, len(trading_dates)),
        index=trading_dates
    )
    
    # Add some realistic market patterns
    # Market crashes (2020 COVID, 2022 inflation)
    crash_periods = [
        (datetime(2020, 3, 1), datetime(2020, 4, 30)),  # COVID crash
        (datetime(2022, 6, 1), datetime(2022, 8, 31)),  # Inflation concerns
    ]
    
    for start, end in crash_periods:
        crash_mask = (trading_dates >= start) & (trading_dates <= end)
        crash_returns = np.random.normal(-0.002, 0.025, crash_mask.sum())
        benchmark_returns[crash_mask] = crash_returns
        strategy_10b_returns[crash_mask] = crash_returns * 1.1  # Slightly worse
        strategy_3b_returns[crash_mask] = crash_returns * 1.2  # More volatile
    
    # Add recovery periods
    recovery_periods = [
        (datetime(2020, 5, 1), datetime(2020, 8, 31)),  # Post-COVID recovery
        (datetime(2022, 9, 1), datetime(2022, 12, 31)),  # Late 2022 recovery
    ]
    
    for start, end in recovery_periods:
        recovery_mask = (trading_dates >= start) & (trading_dates <= end)
        recovery_returns = np.random.normal(0.0015, 0.015, recovery_mask.sum())
        benchmark_returns[recovery_mask] = recovery_returns
        strategy_10b_returns[recovery_mask] = recovery_returns * 1.05  # Slightly better
        strategy_3b_returns[recovery_mask] = recovery_returns * 1.1  # Better recovery
    
    print(f"✅ Sample data generated: {len(trading_dates)} trading days")
    print(f"   - Date range: {trading_dates.min()} to {trading_dates.max()}")
    
    return {
        'strategy_10b': strategy_10b_returns,
        'strategy_3b': strategy_3b_returns,
        'benchmark': benchmark_returns,
        'trading_dates': trading_dates
    }

def calculate_performance_metrics(returns, benchmark, risk_free_rate=0.05):
    """Calculate comprehensive performance metrics vs benchmark."""
    common_index = returns.index.intersection(benchmark.index)
    returns, benchmark = returns.loc[common_index], benchmark.loc[common_index]
    
    n_years = len(returns) / 252
    
    # Basic metrics
    total_return = (1 + returns).prod() - 1
    annual_return = (1 + total_return) ** (1 / n_years) - 1
    annual_vol = returns.std() * np.sqrt(252)
    sharpe_ratio = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0
    
    # Benchmark metrics
    benchmark_total_return = (1 + benchmark).prod() - 1
    benchmark_annual_return = (1 + benchmark_total_return) ** (1 / n_years) - 1
    benchmark_vol = benchmark.std() * np.sqrt(252)
    
    # Risk metrics
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Alpha and beta calculation
    covariance = np.cov(returns, benchmark)[0, 1]
    benchmark_var = benchmark.var()
    beta = covariance / benchmark_var if benchmark_var > 0 else 0
    alpha = annual_return - (beta * benchmark_annual_return)
    
    # Additional metrics
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    information_ratio = alpha / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
    tracking_error = (returns - benchmark).std() * np.sqrt(252)
    
    # Win rate and other statistics
    positive_days = (returns > 0).sum()
    total_days = len(returns)
    win_rate = positive_days / total_days if total_days > 0 else 0
    
    # VaR and CVaR
    var_95 = np.percentile(returns, 5)
    cvar_95 = returns[returns <= var_95].mean()
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_volatility': annual_vol,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'alpha': alpha,
        'beta': beta,
        'calmar_ratio': calmar_ratio,
        'information_ratio': information_ratio,
        'tracking_error': tracking_error,
        'win_rate': win_rate,
        'var_95': var_95,
        'cvar_95': cvar_95,
        'benchmark_return': benchmark_annual_return,
        'benchmark_volatility': benchmark_vol,
        'excess_return': annual_return - benchmark_annual_return
    }

def create_comprehensive_tearsheet(data):
    """Create comprehensive tearsheet with 12-panel analysis."""
    print("📊 Creating comprehensive tearsheet...")
    
    strategy_10b = data['strategy_10b']
    strategy_3b = data['strategy_3b']
    benchmark_returns = data['benchmark']
    
    # Calculate metrics
    metrics_10b = calculate_performance_metrics(strategy_10b, benchmark_returns)
    metrics_3b = calculate_performance_metrics(strategy_3b, benchmark_returns)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    fig = plt.figure(figsize=(20, 24))
    
    # Configuration
    CONFIG = {
        "weighting_scheme": {
            'Value': 0.6,
            'Quality': 0.2,
            'Reversal': 0.2
        }
    }
    
    # 1. Cumulative Returns Comparison
    ax1 = plt.subplot(4, 3, 1)
    cumulative_strategy_10b = (1 + strategy_10b).cumprod()
    cumulative_strategy_3b = (1 + strategy_3b).cumprod()
    cumulative_benchmark = (1 + benchmark_returns).cumprod()
    
    ax1.plot(cumulative_strategy_10b.index, cumulative_strategy_10b.values, 
             label='10B VND Strategy', linewidth=2, color='#2E86AB')
    ax1.plot(cumulative_strategy_3b.index, cumulative_strategy_3b.values, 
             label='3B VND Strategy', linewidth=2, color='#A23B72')
    ax1.plot(cumulative_benchmark.index, cumulative_benchmark.values, 
             label='VNINDEX Benchmark', linewidth=2, color='#F18F01', linestyle='--')
    
    ax1.set_title('Cumulative Returns Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Cumulative Return')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 2. Drawdown Analysis
    ax2 = plt.subplot(4, 3, 2)
    running_max_10b = cumulative_strategy_10b.expanding().max()
    running_max_3b = cumulative_strategy_3b.expanding().max()
    running_max_benchmark = cumulative_benchmark.expanding().max()
    
    drawdown_10b = (cumulative_strategy_10b - running_max_10b) / running_max_10b
    drawdown_3b = (cumulative_strategy_3b - running_max_3b) / running_max_3b
    drawdown_benchmark = (cumulative_benchmark - running_max_benchmark) / running_max_benchmark
    
    ax2.fill_between(drawdown_10b.index, drawdown_10b.values, 0, alpha=0.3, label='10B VND Strategy', color='#2E86AB')
    ax2.fill_between(drawdown_3b.index, drawdown_3b.values, 0, alpha=0.3, label='3B VND Strategy', color='#A23B72')
    ax2.fill_between(drawdown_benchmark.index, drawdown_benchmark.values, 0, alpha=0.3, label='VNINDEX Benchmark', color='#F18F01')
    
    ax2.set_title('Drawdown Analysis', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Drawdown')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Rolling Sharpe Ratio
    ax3 = plt.subplot(4, 3, 3)
    rolling_sharpe_10b = strategy_10b.rolling(window=252).mean() / strategy_10b.rolling(window=252).std() * np.sqrt(252)
    rolling_sharpe_3b = strategy_3b.rolling(window=252).mean() / strategy_3b.rolling(window=252).std() * np.sqrt(252)
    rolling_sharpe_benchmark = benchmark_returns.rolling(window=252).mean() / benchmark_returns.rolling(window=252).std() * np.sqrt(252)
    
    ax3.plot(rolling_sharpe_10b.index, rolling_sharpe_10b.values, label='10B VND Strategy', linewidth=2, color='#2E86AB')
    ax3.plot(rolling_sharpe_3b.index, rolling_sharpe_3b.values, label='3B VND Strategy', linewidth=2, color='#A23B72')
    ax3.plot(rolling_sharpe_benchmark.index, rolling_sharpe_benchmark.values, label='VNINDEX Benchmark', linewidth=2, color='#F18F01', linestyle='--')
    
    ax3.set_title('Rolling Sharpe Ratio (1-Year)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Sharpe Ratio')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Performance Metrics Comparison
    ax4 = plt.subplot(4, 3, 4)
    metrics_to_plot = ['annual_return', 'sharpe_ratio', 'max_drawdown', 'alpha']
    x = np.arange(len(metrics_to_plot))
    width = 0.25
    
    values_10b = [metrics_10b[metric] for metric in metrics_to_plot]
    values_3b = [metrics_3b[metric] for metric in metrics_to_plot]
    
    ax4.bar(x - width, values_10b, width, label='10B VND Strategy', alpha=0.8, color='#2E86AB')
    ax4.bar(x + width, values_3b, width, label='3B VND Strategy', alpha=0.8, color='#A23B72')
    
    ax4.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels([m.replace('_', ' ').title() for m in metrics_to_plot])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Monthly Returns Heatmap (10B VND)
    ax5 = plt.subplot(4, 3, 5)
    monthly_returns_10b = strategy_10b.resample('M').apply(lambda x: (1 + x).prod() - 1)
    monthly_returns_pivot_10b = monthly_returns_10b.groupby([monthly_returns_10b.index.year, monthly_returns_10b.index.month]).first().unstack()
    
    sns.heatmap(monthly_returns_pivot_10b, annot=True, fmt='.2%', cmap='RdYlGn', center=0, ax=ax5)
    ax5.set_title('Monthly Returns Heatmap (10B VND)', fontsize=14, fontweight='bold')
    
    # 6. Monthly Returns Heatmap (3B VND)
    ax6 = plt.subplot(4, 3, 6)
    monthly_returns_3b = strategy_3b.resample('M').apply(lambda x: (1 + x).prod() - 1)
    monthly_returns_pivot_3b = monthly_returns_3b.groupby([monthly_returns_3b.index.year, monthly_returns_3b.index.month]).first().unstack()
    
    sns.heatmap(monthly_returns_pivot_3b, annot=True, fmt='.2%', cmap='RdYlGn', center=0, ax=ax6)
    ax6.set_title('Monthly Returns Heatmap (3B VND)', fontsize=14, fontweight='bold')
    
    # 7. Risk-Return Scatter
    ax7 = plt.subplot(4, 3, 7)
    ax7.scatter(metrics_10b['annual_volatility'], metrics_10b['annual_return'], 
               label='10B VND Strategy', s=200, alpha=0.7, color='#2E86AB')
    ax7.scatter(metrics_3b['annual_volatility'], metrics_3b['annual_return'], 
               label='3B VND Strategy', s=200, alpha=0.7, color='#A23B72')
    ax7.scatter(metrics_10b['benchmark_volatility'], metrics_10b['benchmark_return'], 
               label='VNINDEX Benchmark', s=200, alpha=0.7, color='#F18F01')
    
    ax7.set_title('Risk-Return Profile', fontsize=14, fontweight='bold')
    ax7.set_xlabel('Annual Volatility')
    ax7.set_ylabel('Annual Return')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Rolling Beta
    ax8 = plt.subplot(4, 3, 8)
    rolling_beta_10b = (strategy_10b.rolling(window=252).cov(benchmark_returns) / benchmark_returns.rolling(window=252).var()).fillna(0)
    rolling_beta_3b = (strategy_3b.rolling(window=252).cov(benchmark_returns) / benchmark_returns.rolling(window=252).var()).fillna(0)
    
    ax8.plot(rolling_beta_10b.index, rolling_beta_10b.values, label='10B VND Strategy', linewidth=2, color='#2E86AB')
    ax8.plot(rolling_beta_3b.index, rolling_beta_3b.values, label='3B VND Strategy', linewidth=2, color='#A23B72')
    ax8.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Beta = 1')
    
    ax8.set_title('Rolling Beta (1-Year)', fontsize=14, fontweight='bold')
    ax8.set_ylabel('Beta')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. Rolling Alpha
    ax9 = plt.subplot(4, 3, 9)
    rolling_alpha_10b = (strategy_10b.rolling(window=252).mean() - rolling_beta_10b * benchmark_returns.rolling(window=252).mean()) * 252
    rolling_alpha_3b = (strategy_3b.rolling(window=252).mean() - rolling_beta_3b * benchmark_returns.rolling(window=252).mean()) * 252
    
    ax9.plot(rolling_alpha_10b.index, rolling_alpha_10b.values, label='10B VND Strategy', linewidth=2, color='#2E86AB')
    ax9.plot(rolling_alpha_3b.index, rolling_alpha_3b.values, label='3B VND Strategy', linewidth=2, color='#A23B72')
    ax9.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Alpha = 0')
    
    ax9.set_title('Rolling Alpha (1-Year)', fontsize=14, fontweight='bold')
    ax9.set_ylabel('Alpha')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    # 10. Return Distribution
    ax10 = plt.subplot(4, 3, 10)
    ax10.hist(strategy_10b, bins=50, alpha=0.6, label='10B VND Strategy', color='#2E86AB', density=True)
    ax10.hist(strategy_3b, bins=50, alpha=0.6, label='3B VND Strategy', color='#A23B72', density=True)
    ax10.hist(benchmark_returns, bins=50, alpha=0.6, label='VNINDEX Benchmark', color='#F18F01', density=True)
    
    ax10.set_title('Return Distribution', fontsize=14, fontweight='bold')
    ax10.set_xlabel('Daily Returns')
    ax10.set_ylabel('Density')
    ax10.legend()
    ax10.grid(True, alpha=0.3)
    
    # 11. Rolling Information Ratio
    ax11 = plt.subplot(4, 3, 11)
    rolling_excess_10b = strategy_10b - benchmark_returns
    rolling_excess_3b = strategy_3b - benchmark_returns
    
    rolling_ir_10b = rolling_excess_10b.rolling(window=252).mean() / rolling_excess_10b.rolling(window=252).std() * np.sqrt(252)
    rolling_ir_3b = rolling_excess_3b.rolling(window=252).mean() / rolling_excess_3b.rolling(window=252).std() * np.sqrt(252)
    
    ax11.plot(rolling_ir_10b.index, rolling_ir_10b.values, label='10B VND Strategy', linewidth=2, color='#2E86AB')
    ax11.plot(rolling_ir_3b.index, rolling_ir_3b.values, label='3B VND Strategy', linewidth=2, color='#A23B72')
    ax11.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='IR = 0')
    
    ax11.set_title('Rolling Information Ratio (1-Year)', fontsize=14, fontweight='bold')
    ax11.set_ylabel('Information Ratio')
    ax11.legend()
    ax11.grid(True, alpha=0.3)
    
    # 12. Strategy Weights Visualization
    ax12 = plt.subplot(4, 3, 12)
    weights = list(CONFIG['weighting_scheme'].values())
    labels = list(CONFIG['weighting_scheme'].keys())
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    ax12.pie(weights, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    ax12.set_title('Weighted Composite: Factor Weights', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the tearsheet
    tearsheet_path = f"phase22_weighted_composite_tearsheet_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(tearsheet_path, dpi=300, bbox_inches='tight')
    print(f"✅ Tearsheet saved to: {tearsheet_path}")
    
    plt.show()
    
    return {
        'metrics_10b': metrics_10b,
        'metrics_3b': metrics_3b,
        'tearsheet_path': tearsheet_path
    }

def generate_performance_summary(metrics_10b, metrics_3b):
    """Generate comprehensive performance summary."""
    print("\n" + "=" * 80)
    print("📊 PHASE 22 WEIGHTED COMPOSITE BACKTESTING - PERFORMANCE SUMMARY")
    print("=" * 80)
    
    # Create performance summary table
    summary_data = []
    for threshold, metrics in [('10B VND', metrics_10b), ('3B VND', metrics_3b)]:
        summary_data.append({
            'Strategy': threshold,
            'Annual Return': f"{metrics['annual_return']:.2%}",
            'Sharpe Ratio': f"{metrics['sharpe_ratio']:.2f}",
            'Max Drawdown': f"{metrics['max_drawdown']:.2%}",
            'Alpha': f"{metrics['alpha']:.2%}",
            'Beta': f"{metrics['beta']:.2f}",
            'Information Ratio': f"{metrics['information_ratio']:.2f}",
            'Calmar Ratio': f"{metrics['calmar_ratio']:.2f}",
            'Win Rate': f"{metrics['win_rate']:.1%}",
            'Excess Return': f"{metrics['excess_return']:.2%}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    print("\nPerformance Summary:")
    print(summary_df.to_string(index=False))
    
    # Benchmark comparison
    print(f"\n📈 BENCHMARK COMPARISON (VNINDEX):")
    print(f"   Annual Return: {metrics_10b['benchmark_return']:.2%}")
    print(f"   Annual Volatility: {metrics_10b['benchmark_volatility']:.2%}")
    print(f"   Sharpe Ratio: {(metrics_10b['benchmark_return'] - 0.05) / metrics_10b['benchmark_volatility']:.2f}")
    
    # Strategy insights
    print(f"\n🎯 STRATEGY INSIGHTS:")
    print(f"   Weighting Scheme: 60% Value + 20% Quality + 20% Reversal")
    print(f"   Portfolio Size: 25 stocks")
    print(f"   Rebalancing: Monthly")
    print(f"   Transaction Costs: 20 bps")
    
    # Performance analysis
    best_strategy = '3B VND' if metrics_3b['sharpe_ratio'] > metrics_10b['sharpe_ratio'] else '10B VND'
    best_metrics = metrics_3b if best_strategy == '3B VND' else metrics_10b
    
    print(f"\n🏆 BEST PERFORMING STRATEGY: {best_strategy}")
    print(f"   Sharpe Ratio: {best_metrics['sharpe_ratio']:.2f}")
    print(f"   Annual Return: {best_metrics['annual_return']:.2%}")
    print(f"   Alpha: {best_metrics['alpha']:.2%}")
    print(f"   Max Drawdown: {best_metrics['max_drawdown']:.2%}")
    
    # Risk assessment
    print(f"\n⚠️ RISK ASSESSMENT:")
    if best_metrics['sharpe_ratio'] > 1.0:
        print("   ✅ Excellent risk-adjusted performance")
    elif best_metrics['sharpe_ratio'] > 0.5:
        print("   ✅ Good risk-adjusted performance")
    elif best_metrics['sharpe_ratio'] > 0.0:
        print("   ⚠️ Moderate risk-adjusted performance")
    else:
        print("   ❌ Poor risk-adjusted performance")
    
    if best_metrics['alpha'] > 0.05:
        print("   ✅ Strong alpha generation")
    elif best_metrics['alpha'] > 0.02:
        print("   ✅ Moderate alpha generation")
    else:
        print("   ⚠️ Limited alpha generation")
    
    if best_metrics['max_drawdown'] > -0.3:
        print("   ✅ Acceptable drawdown levels")
    else:
        print("   ⚠️ High drawdown risk")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"   1. {'Consider 3B VND threshold for higher returns' if best_strategy == '3B VND' else 'Consider 10B VND threshold for better risk-adjusted returns'}")
    print(f"   2. Monitor factor correlations and adjust weights if needed")
    print(f"   3. Implement risk management overlays for drawdown control")
    print(f"   4. Consider dynamic weighting based on market regimes")
    print(f"   5. Validate results with out-of-sample testing")
    
    print("\n" + "=" * 80)
    print("✅ PHASE 22 WEIGHTED COMPOSITE TEARSHEET ANALYSIS COMPLETE")
    print("=" * 80)
    
    return summary_df

def main():
    """Main function to run the tearsheet analysis."""
    print("🚀 PHASE 22: WEIGHTED COMPOSITE BACKTESTING - DEMO TEARSHEET")
    print("=" * 80)
    print(f"📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    try:
        # Generate sample data
        data = generate_sample_data()
        
        # Create comprehensive tearsheet
        results = create_comprehensive_tearsheet(data)
        
        # Generate performance summary
        summary_df = generate_performance_summary(results['metrics_10b'], results['metrics_3b'])
        
        print("\n🎉 Demo tearsheet analysis completed successfully!")
        print(f"📊 Tearsheet saved to: {results['tearsheet_path']}")
        
        return {
            'summary_df': summary_df,
            'metrics_10b': results['metrics_10b'],
            'metrics_3b': results['metrics_3b'],
            'tearsheet_path': results['tearsheet_path']
        }
        
    except Exception as e:
        print(f"❌ Demo tearsheet analysis failed: {e}")
        return None

if __name__ == "__main__":
    results = main()
    if results:
        print("✅ Demo tearsheet analysis completed successfully!")
    else:
        print("❌ Demo tearsheet analysis failed!") 