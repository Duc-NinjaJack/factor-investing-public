#!/usr/bin/env python3
"""
Quick script to calculate benchmark performance for Phase 20 comparison
"""

import pickle
import pandas as pd
import numpy as np

def calculate_performance_metrics(returns, benchmark=None, risk_free_rate=0.0):
    """Calculate comprehensive performance metrics."""
    if benchmark is None:
        benchmark = pd.Series(0, index=returns.index)
    
    common_index = returns.index.intersection(benchmark.index)
    returns, benchmark = returns.loc[common_index], benchmark.loc[common_index]
    
    n_years = len(returns) / 252
    annual_return = (1 + returns).prod() ** (1 / n_years) - 1 if n_years > 0 else 0
    annual_vol = returns.std() * np.sqrt(252)
    sharpe_ratio = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0
    
    cumulative = (1 + returns).cumprod()
    drawdown = (cumulative / cumulative.cummax() - 1)
    max_drawdown = drawdown.min()
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 and abs(max_drawdown) > 1e-10 else 0
    
    return {
        'Annual Return': annual_return * 100,
        'Annual Volatility': annual_vol * 100,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown * 100,
        'Calmar Ratio': calmar_ratio,
        'Total Return': ((1 + returns).prod() - 1) * 100
    }

# Load Phase 20 data
print("📂 Loading Phase 20 data...")
with open('data/dynamic_strategy_database_backtest_results.pkl', 'rb') as f:
    results = pickle.load(f)

backtest_results = results['backtest_results']
prepared_data = results['prepared_data']

# Extract data
dynamic_10b = backtest_results['10B_VND_Dynamic']['portfolio_returns']
static_10b = backtest_results['10B_VND_Static']['portfolio_returns']
benchmark_returns = prepared_data['benchmark_returns']

# Align data
common_dates = dynamic_10b.index.intersection(static_10b.index).intersection(benchmark_returns.index)
dynamic_10b = dynamic_10b.loc[common_dates]
static_10b = static_10b.loc[common_dates]
benchmark_returns = benchmark_returns.loc[common_dates]

print(f"📊 Data aligned: {len(common_dates)} observations from {common_dates.min()} to {common_dates.max()}")

# Calculate metrics
dynamic_metrics = calculate_performance_metrics(dynamic_10b)
static_metrics = calculate_performance_metrics(static_10b)
benchmark_metrics = calculate_performance_metrics(benchmark_returns)

print("\n" + "="*80)
print("🏆 PHASE 20 PERFORMANCE COMPARISON")
print("="*80)
print(f"{'Metric':<20} {'Dynamic':<12} {'Static':<12} {'Benchmark':<12}")
print("-" * 80)
for key in ['Annual Return', 'Sharpe Ratio', 'Max Drawdown', 'Calmar Ratio']:
    print(f"{key:<20} {dynamic_metrics[key]:<12.2f} {static_metrics[key]:<12.2f} {benchmark_metrics[key]:<12.2f}")

print("\n" + "="*80)
print("📊 PHASE 16b vs PHASE 20 COMPARISON")
print("="*80)
print("Phase 16b (2016-2025):")
print("  - Standalone Value: 13.93% return, 0.50 Sharpe, -66.90% max drawdown")
print("  - Weighted QVR: 13.29% return, 0.48 Sharpe, -66.60% max drawdown")
print("  - VN-Index: 10.73% return, 0.59 Sharpe, -45.26% max drawdown")

print("\nPhase 20 (2017-2025):")
print(f"  - Dynamic Strategy: {dynamic_metrics['Annual Return']:.2f}% return, {dynamic_metrics['Sharpe Ratio']:.2f} Sharpe, {dynamic_metrics['Max Drawdown']:.2f}% max drawdown")
print(f"  - Static Strategy: {static_metrics['Annual Return']:.2f}% return, {static_metrics['Sharpe Ratio']:.2f} Sharpe, {static_metrics['Max Drawdown']:.2f}% max drawdown")
print(f"  - VN-Index: {benchmark_metrics['Annual Return']:.2f}% return, {benchmark_metrics['Sharpe Ratio']:.2f} Sharpe, {benchmark_metrics['Max Drawdown']:.2f}% max drawdown")

print("\n" + "="*80)
print("🔍 KEY DIFFERENCES")
print("="*80)
print(f"Return Gap (Phase 16b Value vs Phase 20 Dynamic): {13.93 - dynamic_metrics['Annual Return']:+.2f}%")
print(f"Sharpe Gap (Phase 16b Value vs Phase 20 Dynamic): {0.50 - dynamic_metrics['Sharpe Ratio']:+.2f}")
print(f"Benchmark Return Gap: {10.73 - benchmark_metrics['Annual Return']:+.2f}%")
print(f"Benchmark Sharpe Gap: {0.59 - benchmark_metrics['Sharpe Ratio']:+.2f}")

print("\n✅ Benchmark performance calculation complete!")