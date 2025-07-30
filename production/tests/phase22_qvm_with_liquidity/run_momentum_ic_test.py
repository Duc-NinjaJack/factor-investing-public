#!/usr/bin/env python3
"""
Simple Momentum Factor IC Test
==============================

This script provides a simple way to test the Information Coefficient (IC)
of the momentum factor for the two specified periods: 2016-2020 and 2021-2025.

Usage:
    python run_momentum_ic_test.py

Author: Factor Investing Team
Date: 2025-07-30
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Add paths for imports
sys.path.append('../../../production/database')
sys.path.append('../../../production/engine')

try:
    from database.connection import get_engine
    from engine.qvm_engine_v2_enhanced import QVMEngineV2Enhanced
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all dependencies are installed and paths are correct.")
    sys.exit(1)

def setup_logging():
    """Setup basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger('MomentumICTest')

def get_test_universe(engine, analysis_date):
    """Get a test universe of stocks."""
    try:
        query = f"""
        SELECT DISTINCT ticker
        FROM equity_history
        WHERE date <= '{analysis_date.date()}'
          AND close > 5000
        GROUP BY ticker
        HAVING COUNT(*) >= 252
        ORDER BY ticker
        LIMIT 50  -- Limit for testing
        """
        
        with engine.engine.connect() as conn:
            result = pd.read_sql(query, conn)
        
        return result['ticker'].tolist()
    except Exception as e:
        print(f"❌ Failed to get universe: {e}")
        return []

def calculate_momentum_ic(engine, analysis_date, universe, forward_months=1):
    """
    Calculate momentum factor IC for a specific date.
    
    Args:
        engine: QVM engine instance
        analysis_date: Date for analysis
        universe: List of ticker symbols
        forward_months: Forward return horizon in months
    
    Returns:
        IC value and metadata
    """
    try:
        print(f"📊 Calculating momentum IC for {analysis_date.date()}")
        
        # Get fundamental data for sector mapping
        fundamental_data = engine.get_fundamentals_correct_timing(analysis_date, universe)
        
        if fundamental_data.empty:
            print(f"⚠️ No fundamental data for {analysis_date.date()}")
            return None
        
        # Calculate momentum factors
        momentum_scores = engine._calculate_enhanced_momentum_composite(
            fundamental_data, analysis_date, universe
        )
        
        if not momentum_scores:
            print(f"⚠️ No momentum scores calculated for {analysis_date.date()}")
            return None
        
        print(f"✅ Calculated momentum factors for {len(momentum_scores)} stocks")
        
        # Calculate forward returns
        end_date = analysis_date + pd.DateOffset(months=forward_months)
        
        ticker_str = "', '".join(universe)
        query = f"""
        SELECT 
            ticker,
            date,
            close as adj_close
        FROM equity_history
        WHERE ticker IN ('{ticker_str}')
          AND date BETWEEN '{analysis_date.date()}' AND '{end_date.date()}'
        ORDER BY ticker, date
        """
        
        with engine.engine.connect() as conn:
            price_data = pd.read_sql(query, conn, parse_dates=['date'])
        
        if price_data.empty:
            print(f"⚠️ No price data for forward returns")
            return None
        
        # Calculate forward returns
        forward_returns = {}
        for ticker in universe:
            ticker_data = price_data[price_data['ticker'] == ticker].sort_values('date')
            
            if len(ticker_data) >= 2:
                start_price = ticker_data.iloc[0]['adj_close']
                end_price = ticker_data.iloc[-1]['adj_close']
                
                if start_price > 0:
                    forward_returns[ticker] = (end_price / start_price) - 1
        
        if not forward_returns:
            print(f"⚠️ No forward returns calculated")
            return None
        
        print(f"✅ Calculated forward returns for {len(forward_returns)} stocks")
        
        # Calculate IC (Spearman correlation)
        common_tickers = set(momentum_scores.keys()) & set(forward_returns.keys())
        
        if len(common_tickers) < 10:
            print(f"⚠️ Insufficient data for IC: {len(common_tickers)} stocks")
            return None
        
        factor_series = pd.Series([momentum_scores[t] for t in common_tickers], 
                                index=list(common_tickers))
        return_series = pd.Series([forward_returns[t] for t in common_tickers], 
                                index=list(common_tickers))
        
        ic = factor_series.corr(return_series, method='spearman')
        
        return {
            'date': analysis_date,
            'ic': ic,
            'n_stocks': len(common_tickers),
            'momentum_mean': factor_series.mean(),
            'momentum_std': factor_series.std(),
            'return_mean': return_series.mean(),
            'return_std': return_series.std()
        }
        
    except Exception as e:
        print(f"❌ Error calculating IC: {e}")
        return None

def test_period(engine, period_name, start_date, end_date, forward_months=1):
    """
    Test IC for a specific period.
    
    Args:
        engine: QVM engine instance
        period_name: Name of the period
        start_date: Start date string
        end_date: End date string
        forward_months: Forward return horizon
    
    Returns:
        List of IC results
    """
    print(f"\n{'='*60}")
    print(f"🧪 TESTING PERIOD: {period_name}")
    print(f"📅 Period: {start_date} to {end_date}")
    print(f"📈 Forward Horizon: {forward_months} month(s)")
    print(f"{'='*60}")
    
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    # Generate test dates (quarterly for efficiency)
    test_dates = pd.date_range(
        start=start_dt + pd.DateOffset(months=12),  # Need 12 months for momentum
        end=end_dt - pd.DateOffset(months=forward_months),  # Need forward months for returns
        freq='Q'  # Quarterly for testing
    )
    
    print(f"📊 Generated {len(test_dates)} test dates")
    
    results = []
    
    for i, test_date in enumerate(test_dates):
        print(f"\n📅 Processing {test_date.date()} ({i+1}/{len(test_dates)})")
        
        # Get universe
        universe = get_test_universe(engine, test_date)
        if len(universe) < 20:
            print(f"⚠️ Universe too small: {len(universe)} stocks")
            continue
        
        # Calculate IC
        ic_result = calculate_momentum_ic(engine, test_date, universe, forward_months)
        
        if ic_result:
            results.append(ic_result)
            print(f"✅ IC: {ic_result['ic']:.4f} (n={ic_result['n_stocks']})")
        else:
            print(f"❌ Failed to calculate IC")
    
    return results

def analyze_results(results, period_name):
    """Analyze IC results for a period."""
    if not results:
        print(f"❌ No results for {period_name}")
        return
    
    ic_values = [r['ic'] for r in results if not np.isnan(r['ic'])]
    
    if not ic_values:
        print(f"❌ No valid IC values for {period_name}")
        return
    
    ic_series = pd.Series(ic_values)
    
    print(f"\n📊 IC ANALYSIS FOR {period_name}")
    print(f"{'='*50}")
    print(f"Number of observations: {len(ic_series)}")
    print(f"Mean IC: {ic_series.mean():.4f}")
    print(f"Std IC: {ic_series.std():.4f}")
    print(f"Min IC: {ic_series.min():.4f}")
    print(f"Max IC: {ic_series.max():.4f}")
    
    # T-statistic
    t_stat = ic_series.mean() / (ic_series.std() / np.sqrt(len(ic_series)))
    print(f"T-statistic: {t_stat:.3f}")
    
    # Hit rate
    hit_rate = (ic_series > 0).mean()
    print(f"Hit rate: {hit_rate:.1%}")
    
    # Quality assessment
    ic_quality = "✅ GOOD" if ic_series.mean() > 0.02 and t_stat > 2.0 else "❌ POOR"
    hit_quality = "✅ GOOD" if hit_rate > 0.55 else "❌ POOR"
    
    print(f"\n🎯 QUALITY ASSESSMENT:")
    print(f"IC Quality: {ic_quality}")
    print(f"Hit Rate Quality: {hit_quality}")
    
    return {
        'period': period_name,
        'n_observations': len(ic_series),
        'mean_ic': ic_series.mean(),
        'std_ic': ic_series.std(),
        't_stat': t_stat,
        'hit_rate': hit_rate,
        'min_ic': ic_series.min(),
        'max_ic': ic_series.max()
    }

def main():
    """Main execution function."""
    print("🚀 MOMENTUM FACTOR IC TEST")
    print("=" * 60)
    print("Testing Information Coefficient (IC) for momentum factor")
    print("Periods: 2016-2020 and 2021-2025")
    print("=" * 60)
    
    try:
        # Initialize engine
        print("🔧 Initializing QVM Engine...")
        engine = QVMEngineV2Enhanced()
        print("✅ QVM Engine initialized")
        
        # Test periods
        periods = {
            '2016-2020': {
                'start_date': '2016-01-01',
                'end_date': '2020-12-31',
                'description': 'Pre-COVID period'
            },
            '2021-2025': {
                'start_date': '2021-01-01',
                'end_date': '2025-12-31', 
                'description': 'Post-COVID period'
            }
        }
        
        all_results = {}
        
        # Test each period
        for period_name, period_config in periods.items():
            print(f"\n{'='*60}")
            print(f"🧪 TESTING: {period_name}")
            print(f"📝 {period_config['description']}")
            print(f"{'='*60}")
            
            # Test different forward horizons
            for forward_months in [1, 3, 6, 12]:
                print(f"\n📈 Testing {forward_months}M forward returns...")
                
                results = test_period(
                    engine, 
                    period_name, 
                    period_config['start_date'], 
                    period_config['end_date'],
                    forward_months
                )
                
                if results:
                    analysis = analyze_results(results, f"{period_name} ({forward_months}M)")
                    if analysis:
                        key = f"{period_name}_{forward_months}M"
                        all_results[key] = analysis
        
        # Summary comparison
        print(f"\n{'='*80}")
        print("📊 SUMMARY COMPARISON")
        print(f"{'='*80}")
        
        if all_results:
            print(f"{'Period':<20} {'Horizon':<8} {'Mean IC':<10} {'T-Stat':<8} {'Hit Rate':<10}")
            print("-" * 60)
            
            for key, result in all_results.items():
                period, horizon = key.split('_')
                print(f"{period:<20} {horizon:<8} {result['mean_ic']:<10.4f} "
                      f"{result['t_stat']:<8.3f} {result['hit_rate']:<10.1%}")
        
        print(f"\n✅ Momentum Factor IC Test completed successfully!")
        
        return all_results
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main() 