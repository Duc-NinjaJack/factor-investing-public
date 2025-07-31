#!/usr/bin/env python3
"""
Quick Market Cap Quartile Momentum Analysis

Simplified analysis of momentum factor performance across market cap quartiles
for key time periods (2016-2020, 2021-2025).

Author: Factor Investing Team
Date: 2025-07-30
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add production path
sys.path.append('../../../production')

from engine.qvm_engine_v2_enhanced import QVMEngineV2Enhanced

def get_universe_with_market_cap(engine, analysis_date, limit=200):
    """Get universe with market cap data."""
    try:
        query = f"""
        SELECT 
            eh.ticker,
            eh.market_cap,
            mi.sector
        FROM (
            SELECT 
                ticker,
                market_cap,
                ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) as rn
            FROM equity_history_with_market_cap
            WHERE date <= '{analysis_date.date()}'
              AND market_cap IS NOT NULL
              AND market_cap >= 5000000000  -- 5B VND minimum
        ) eh
        LEFT JOIN master_info mi ON eh.ticker = mi.ticker
        WHERE eh.rn = 1
        ORDER BY eh.market_cap DESC
        LIMIT {limit}
        """
        
        universe_data = pd.read_sql(query, engine.engine)
        
        if universe_data.empty:
            print(f"❌ No universe data found for {analysis_date}")
            return pd.DataFrame()
        
        print(f"✅ Universe: {len(universe_data)} stocks")
        print(f"📊 Market Cap Range: {universe_data['market_cap'].min():,.0f} - {universe_data['market_cap'].max():,.0f} VND")
        
        return universe_data
        
    except Exception as e:
        print(f"❌ Error getting universe: {e}")
        return pd.DataFrame()

def create_market_cap_quartiles(universe_data):
    """Create market cap quartiles from universe data."""
    if universe_data.empty:
        return {}
    
    # Create quartiles
    universe_data['quartile'] = pd.qcut(
        universe_data['market_cap'], 
        q=4, 
        labels=['Q1 (Smallest)', 'Q2', 'Q3', 'Q4 (Largest)']
    )
    
    quartiles = {}
    for quartile in universe_data['quartile'].unique():
        quartile_stocks = universe_data[universe_data['quartile'] == quartile]
        quartiles[quartile] = {
            'tickers': quartile_stocks['ticker'].tolist(),
            'market_cap_range': (
                quartile_stocks['market_cap'].min(),
                quartile_stocks['market_cap'].max()
            ),
            'count': len(quartile_stocks),
            'avg_market_cap': quartile_stocks['market_cap'].mean()
        }
        
        print(f"📊 {quartile}: {len(quartile_stocks)} stocks, "
              f"Market Cap: {quartile_stocks['market_cap'].min():,.0f} - {quartile_stocks['market_cap'].max():,.0f} VND")
    
    return quartiles

def calculate_momentum_ic_by_quartile(engine, analysis_date, quartile_data, forward_months=1):
    """Calculate momentum IC for a specific market cap quartile."""
    try:
        tickers = quartile_data['tickers']
        if len(tickers) < 10:  # Minimum 10 stocks for quartile
            return None
        
        # Get fundamental data for sector mapping
        fundamental_data = engine.get_fundamentals_correct_timing(analysis_date, tickers)
        if fundamental_data.empty:
            return None
        
        # Calculate momentum factors
        momentum_scores = engine._calculate_enhanced_momentum_composite(
            fundamental_data, analysis_date, tickers
        )
        if not momentum_scores:
            return None
        
        # Calculate forward returns
        end_date = analysis_date + pd.DateOffset(months=forward_months)
        ticker_str = "', '".join(tickers)
        
        query = f"""
        SELECT ticker, date, close as adj_close
        FROM equity_history
        WHERE ticker IN ('{ticker_str}')
          AND date BETWEEN '{analysis_date.date()}' AND '{end_date.date()}'
        ORDER BY ticker, date
        """
        
        with engine.engine.connect() as conn:
            price_data = pd.read_sql(query, conn, parse_dates=['date'])
        
        if price_data.empty:
            return None
        
        forward_returns = {}
        for ticker in tickers:
            ticker_data = price_data[price_data['ticker'] == ticker].sort_values('date')
            if len(ticker_data) >= 2:
                start_price = ticker_data.iloc[0]['adj_close']
                end_price = ticker_data.iloc[-1]['adj_close']
                if start_price > 0:
                    forward_returns[ticker] = (end_price / start_price) - 1
        
        common_tickers = set(momentum_scores.keys()) & set(forward_returns.keys())
        if len(common_tickers) < 5:  # Minimum 5 stocks for quartile
            return None
        
        factor_series = pd.Series([momentum_scores[t] for t in common_tickers], index=list(common_tickers))
        return_series = pd.Series([forward_returns[t] for t in common_tickers], index=list(common_tickers))
        
        # Calculate Spearman correlation manually
        ic = _calculate_spearman_correlation(factor_series, return_series)
        
        return {
            'date': analysis_date,
            'ic': ic,
            'n_stocks': len(common_tickers),
            'momentum_mean': factor_series.mean(),
            'momentum_std': factor_series.std(),
            'return_mean': return_series.mean(),
            'return_std': return_series.std(),
            'quartile': list(quartile_data.keys())[0] if quartile_data else 'Unknown'
        }
        
    except Exception as e:
        print(f"❌ Error calculating IC for quartile: {e}")
        return None

def _calculate_spearman_correlation(x, y):
    """Calculate Spearman correlation manually to avoid scipy dependency."""
    try:
        # Get ranks
        x_ranks = x.rank()
        y_ranks = y.rank()

        # Calculate correlation using Pearson formula on ranks
        n = len(x)
        if n < 2:
            return 0.0

        x_mean = x_ranks.mean()
        y_mean = y_ranks.mean()

        numerator = ((x_ranks - x_mean) * (y_ranks - y_mean)).sum()
        x_var = ((x_ranks - x_mean) ** 2).sum()
        y_var = ((y_ranks - y_mean) ** 2).sum()

        if x_var == 0 or y_var == 0:
            return 0.0

        correlation = numerator / (x_var * y_var) ** 0.5
        return correlation
    except:
        return 0.0

def test_key_dates(engine, regime_name, test_dates, forward_months=6):
    """Test momentum factor across market cap quartiles for key dates."""
    print(f"\n{'='*60}")
    print(f"🧪 TESTING QUARTILE REGIME: {regime_name}")
    print(f"📈 Forward Horizon: {forward_months} month(s)")
    print(f"{'='*60}")
    
    quartile_results = {
        'Q1 (Smallest)': [],
        'Q2': [],
        'Q3': [],
        'Q4 (Largest)': []
    }
    
    for i, test_date in enumerate(test_dates):
        print(f"\n📅 Processing {test_date.date()} ({i+1}/{len(test_dates)})")
        
        # Get universe with market cap data
        universe_data = get_universe_with_market_cap(engine, test_date, limit=200)
        if universe_data.empty:
            print(f"⚠️ No universe data for {test_date}")
            continue
        
        # Create quartiles
        quartiles = create_market_cap_quartiles(universe_data)
        if not quartiles:
            print(f"⚠️ Failed to create quartiles for {test_date}")
            continue
        
        # Calculate IC for each quartile
        for quartile_name, quartile_data in quartiles.items():
            ic_result = calculate_momentum_ic_by_quartile(
                engine, test_date, quartile_data, forward_months
            )
            if ic_result:
                quartile_results[quartile_name].append(ic_result)
                print(f"✅ {quartile_name}: IC={ic_result['ic']:.4f} (n={ic_result['n_stocks']})")
            else:
                print(f"❌ {quartile_name}: Failed to calculate IC")
    
    return quartile_results

def analyze_quartile_results(quartile_results, regime_name):
    """Analyze results for each market cap quartile."""
    print(f"\n📊 QUARTILE ANALYSIS FOR {regime_name}")
    print(f"{'='*60}")
    
    analysis_summary = {}
    
    for quartile_name, results in quartile_results.items():
        if not results:
            print(f"\n❌ No results for {quartile_name}")
            continue
        
        ic_values = [r['ic'] for r in results if not np.isnan(r['ic'])]
        if not ic_values:
            print(f"\n❌ No valid IC values for {quartile_name}")
            continue
        
        ic_series = pd.Series(ic_values)
        
        print(f"\n📊 {quartile_name}:")
        print(f"  Number of observations: {len(ic_series)}")
        print(f"  Mean IC: {ic_series.mean():.4f}")
        print(f"  Std IC: {ic_series.std():.4f}")
        print(f"  Min IC: {ic_series.min():.4f}")
        print(f"  Max IC: {ic_series.max():.4f}")
        
        # Calculate t-statistic
        t_stat = ic_series.mean() / (ic_series.std() / np.sqrt(len(ic_series)))
        print(f"  T-statistic: {t_stat:.4f}")
        
        # Calculate hit rate
        hit_rate = (ic_series > 0).mean()
        print(f"  Hit Rate: {hit_rate:.1%}")
        
        # Quality gates
        mean_ic = ic_series.mean()
        quality_gates = {
            'Mean IC > 0.02': mean_ic > 0.02,
            'T-stat > 2.0': t_stat > 2.0,
            'Hit Rate > 55%': hit_rate > 0.55
        }
        
        gates_passed = sum(quality_gates.values())
        print(f"  Quality Gates Passed: {gates_passed}/3")
        
        analysis_summary[quartile_name] = {
            'mean_ic': mean_ic,
            't_stat': t_stat,
            'hit_rate': hit_rate,
            'n_obs': len(ic_series),
            'gates_passed': gates_passed
        }
    
    return analysis_summary

def create_quartile_visualizations(regime_results, regime_name):
    """Create visualizations for quartile analysis."""
    print(f"\n📊 Creating visualizations for {regime_name}...")
    
    # Prepare data for plotting
    quartiles = []
    mean_ics = []
    hit_rates = []
    
    for quartile_name, summary in regime_results.items():
        quartiles.append(quartile_name)
        mean_ics.append(summary['mean_ic'])
        hit_rates.append(summary['hit_rate'])
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Mean IC by quartile
    colors = ['red', 'orange', 'blue', 'green']
    bars1 = ax1.bar(quartiles, mean_ics, color=colors, alpha=0.7)
    ax1.set_xlabel('Market Cap Quartile')
    ax1.set_ylabel('Mean Information Coefficient (IC)')
    ax1.set_title(f'Mean IC by Market Cap Quartile - {regime_name}')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Add value labels on bars
    for bar, ic in zip(bars1, mean_ics):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{ic:.3f}', ha='center', va='bottom' if height > 0 else 'top')
    
    # Plot 2: Hit Rate by quartile
    bars2 = ax2.bar(quartiles, hit_rates, color=colors, alpha=0.7)
    ax2.set_xlabel('Market Cap Quartile')
    ax2.set_ylabel('Hit Rate')
    ax2.set_title(f'Hit Rate by Market Cap Quartile - {regime_name}')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.55, color='red', linestyle='--', alpha=0.7, label='Quality Gate (55%)')
    ax2.legend()
    
    # Add value labels on bars
    for bar, rate in zip(bars2, hit_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1%}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'img/market_cap_quartile_analysis_{regime_name.replace(" ", "_").replace("-", "_")}.png', 
                dpi=300, bbox_inches='tight')
    print(f"✅ Visualization saved to: img/market_cap_quartile_analysis_{regime_name.replace(' ', '_').replace('-', '_')}.png")

def main():
    """Main execution function for quick market cap quartile analysis."""
    print("🚀 QUICK MARKET CAP QUARTILE MOMENTUM ANALYSIS")
    print("=" * 60)
    print("Testing momentum factor across market cap quartiles")
    print("Analyzing performance differences by company size")
    print("=" * 60)
    
    # Initialize engine
    try:
        engine = QVMEngineV2Enhanced()
        print("✅ Engine initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize engine: {e}")
        return
    
    # Define key test dates for each regime
    regime_2016_2020_dates = [
        pd.to_datetime('2017-06-30'),
        pd.to_datetime('2018-06-30'),
        pd.to_datetime('2019-06-30'),
        pd.to_datetime('2020-06-30')
    ]
    
    regime_2021_2025_dates = [
        pd.to_datetime('2022-06-30'),
        pd.to_datetime('2023-06-30'),
        pd.to_datetime('2024-06-30'),
        pd.to_datetime('2025-06-30')
    ]
    
    all_results = {}
    
    # Test 2016-2020 regime
    print(f"\n{'='*60}")
    print(f"🧪 TESTING REGIME: 2016-2020")
    print(f"{'='*60}")
    
    quartile_results_2016_2020 = test_key_dates(
        engine, "2016-2020", regime_2016_2020_dates, forward_months=6
    )
    
    analysis_summary_2016_2020 = analyze_quartile_results(quartile_results_2016_2020, "2016-2020 (6M)")
    if analysis_summary_2016_2020:
        all_results['2016-2020_6M'] = analysis_summary_2016_2020
        create_quartile_visualizations(analysis_summary_2016_2020, "2016-2020 (6M)")
    
    # Test 2021-2025 regime
    print(f"\n{'='*60}")
    print(f"🧪 TESTING REGIME: 2021-2025")
    print(f"{'='*60}")
    
    quartile_results_2021_2025 = test_key_dates(
        engine, "2021-2025", regime_2021_2025_dates, forward_months=6
    )
    
    analysis_summary_2021_2025 = analyze_quartile_results(quartile_results_2021_2025, "2021-2025 (6M)")
    if analysis_summary_2021_2025:
        all_results['2021-2025_6M'] = analysis_summary_2021_2025
        create_quartile_visualizations(analysis_summary_2021_2025, "2021-2025 (6M)")
    
    # Summary report
    print(f"\n{'='*60}")
    print(f"📊 MARKET CAP QUARTILE ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    for key, regime_results in all_results.items():
        print(f"\n{key}:")
        for quartile_name, summary in regime_results.items():
            print(f"  {quartile_name}:")
            print(f"    Mean IC: {summary['mean_ic']:.4f}")
            print(f"    T-stat: {summary['t_stat']:.4f}")
            print(f"    Hit Rate: {summary['hit_rate']:.1%}")
            print(f"    Gates Passed: {summary['gates_passed']}/3")
    
    # Cross-quartile analysis
    print(f"\n{'='*60}")
    print(f"🔍 CROSS-QUARTILE ANALYSIS")
    print(f"{'='*60}")
    
    for key, regime_results in all_results.items():
        if len(regime_results) == 4:  # All quartiles available
            print(f"\n{key}:")
            
            # Find best and worst performing quartiles
            quartile_performance = [(name, summary['mean_ic']) for name, summary in regime_results.items()]
            quartile_performance.sort(key=lambda x: x[1], reverse=True)
            
            best_quartile = quartile_performance[0]
            worst_quartile = quartile_performance[-1]
            
            print(f"  Best Quartile: {best_quartile[0]} (IC: {best_quartile[1]:.4f})")
            print(f"  Worst Quartile: {worst_quartile[0]} (IC: {worst_quartile[1]:.4f})")
            print(f"  Performance Spread: {best_quartile[1] - worst_quartile[1]:.4f}")
    
    print(f"\n✅ Quick market cap quartile analysis completed!")

if __name__ == "__main__":
    main() 