#!/usr/bin/env python3
"""
Check Real VNINDEX Benchmark Data
=================================
Purpose: Load and analyze real VNINDEX data from database to compare with fake demo data
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

# Add paths for imports
sys.path.append('../../../production/database')

# Setup
warnings.filterwarnings('ignore')

def check_real_benchmark_data():
    """Check real VNINDEX benchmark data from database."""
    print("🔍 CHECKING REAL VNINDEX BENCHMARK DATA")
    print("=" * 60)
    
    try:
        # Import database connection
        from connection import get_database_manager
        
        # Get database connection
        db_manager = get_database_manager()
        engine = db_manager.get_engine()
        
        print("📊 Loading real VNINDEX data from database...")
        
        # Load VNINDEX data from etf_history table
        benchmark_query = """
        SELECT date, close
        FROM etf_history
        WHERE ticker = 'VNINDEX' 
        AND date >= '2018-01-01'
        ORDER BY date
        """
        
        benchmark_data = pd.read_sql(benchmark_query, engine)
        benchmark_data['date'] = pd.to_datetime(benchmark_data['date'])
        benchmark_data = benchmark_data.set_index('date')
        
        # Calculate returns
        benchmark_returns = benchmark_data['close'].pct_change().dropna()
        
        print(f"✅ Real VNINDEX data loaded successfully!")
        print(f"   - Records: {len(benchmark_returns):,}")
        print(f"   - Date range: {benchmark_returns.index.min()} to {benchmark_returns.index.max()}")
        
        # Calculate performance metrics
        n_years = len(benchmark_returns) / 252
        
        # Basic metrics
        total_return = (1 + benchmark_returns).prod() - 1
        annual_return = (1 + total_return) ** (1 / n_years) - 1
        annual_vol = benchmark_returns.std() * np.sqrt(252)
        sharpe_ratio = (annual_return - 0.05) / annual_vol if annual_vol > 0 else 0
        
        # Risk metrics
        cumulative_returns = (1 + benchmark_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Additional metrics
        win_rate = (benchmark_returns > 0).sum() / len(benchmark_returns)
        var_95 = np.percentile(benchmark_returns, 5)
        
        print(f"\n📈 REAL VNINDEX PERFORMANCE (2018-2024):")
        print(f"   Annual Return: {annual_return:.2%}")
        print(f"   Annual Volatility: {annual_vol:.2%}")
        print(f"   Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"   Max Drawdown: {max_drawdown:.2%}")
        print(f"   Win Rate: {win_rate:.1%}")
        print(f"   VaR (95%): {var_95:.2%}")
        print(f"   Total Return: {total_return:.2%}")
        
        # Compare with fake demo data
        print(f"\n🚨 COMPARISON WITH FAKE DEMO DATA:")
        print(f"   {'Metric':<20} {'Real VNINDEX':<15} {'Fake Demo':<15} {'Difference':<15}")
        print(f"   {'-'*20} {'-'*15} {'-'*15} {'-'*15}")
        print(f"   {'Annual Return':<20} {annual_return:>14.2%} {0.2801:>14.2%} {(annual_return-0.2801):>14.2%}")
        print(f"   {'Sharpe Ratio':<20} {sharpe_ratio:>14.2f} {0.81:>14.2f} {(sharpe_ratio-0.81):>14.2f}")
        print(f"   {'Max Drawdown':<20} {max_drawdown:>14.2%} {-0.352:>14.2%} {(max_drawdown+0.352):>14.2%}")
        print(f"   {'Annual Volatility':<20} {annual_vol:>14.2%} {0.2855:>14.2%} {(annual_vol-0.2855):>14.2%}")
        
        # Monthly returns analysis
        monthly_returns = benchmark_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        print(f"\n📅 MONTHLY RETURNS ANALYSIS:")
        print(f"   Average Monthly Return: {monthly_returns.mean():.2%}")
        print(f"   Monthly Return Std Dev: {monthly_returns.std():.2%}")
        print(f"   Best Month: {monthly_returns.max():.2%}")
        print(f"   Worst Month: {monthly_returns.min():.2%}")
        
        # Yearly returns
        yearly_returns = benchmark_returns.resample('Y').apply(lambda x: (1 + x).prod() - 1)
        
        print(f"\n📊 YEARLY RETURNS:")
        for year, ret in yearly_returns.items():
            print(f"   {year.year}: {ret:.2%}")
        
        # Check for specific periods
        print(f"\n🔍 SPECIFIC PERIODS ANALYSIS:")
        
        # COVID period (March-April 2020)
        covid_mask = (benchmark_returns.index >= '2020-03-01') & (benchmark_returns.index <= '2020-04-30')
        if covid_mask.sum() > 0:
            covid_returns = benchmark_returns[covid_mask]
            covid_total = (1 + covid_returns).prod() - 1
            print(f"   COVID Crash (Mar-Apr 2020): {covid_total:.2%}")
        
        # 2022 inflation period (June-August 2022)
        inflation_mask = (benchmark_returns.index >= '2022-06-01') & (benchmark_returns.index <= '2022-08-31')
        if inflation_mask.sum() > 0:
            inflation_returns = benchmark_returns[inflation_mask]
            inflation_total = (1 + inflation_returns).prod() - 1
            print(f"   Inflation Period (Jun-Aug 2022): {inflation_total:.2%}")
        
        # Recovery periods
        recovery_2020_mask = (benchmark_returns.index >= '2020-05-01') & (benchmark_returns.index <= '2020-08-31')
        if recovery_2020_mask.sum() > 0:
            recovery_2020_returns = benchmark_returns[recovery_2020_mask]
            recovery_2020_total = (1 + recovery_2020_returns).prod() - 1
            print(f"   Post-COVID Recovery (May-Aug 2020): {recovery_2020_total:.2%}")
        
        recovery_2022_mask = (benchmark_returns.index >= '2022-09-01') & (benchmark_returns.index <= '2022-12-31')
        if recovery_2022_mask.sum() > 0:
            recovery_2022_returns = benchmark_returns[recovery_2022_mask]
            recovery_2022_total = (1 + recovery_2022_returns).prod() - 1
            print(f"   Late 2022 Recovery (Sep-Dec 2022): {recovery_2022_total:.2%}")
        
        print(f"\n✅ Real benchmark analysis complete!")
        
        return {
            'benchmark_returns': benchmark_returns,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'annual_vol': annual_vol
        }
        
    except Exception as e:
        print(f"❌ Error loading real benchmark data: {e}")
        return None

if __name__ == "__main__":
    results = check_real_benchmark_data()
    if results:
        print("\n🎉 Real benchmark data analysis completed successfully!")
    else:
        print("\n❌ Real benchmark data analysis failed!") 