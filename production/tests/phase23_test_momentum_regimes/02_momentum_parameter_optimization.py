#!/usr/bin/env python3
"""
Momentum Parameter Optimization Analysis

This script performs comprehensive parameter optimization for the momentum factor
IC analysis, testing various combinations of lookback periods, weights, and
forward horizons to identify optimal settings.

Author: Factor Investing Team
Date: 2025-07-30
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add paths for imports
sys.path.append('../../../production/database')
sys.path.append('../../../production/engine')

try:
    from database.connection import get_engine
    from engine.qvm_engine_v2_enhanced import QVMEngineV2Enhanced
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

class MomentumParameterOptimizer:
    """
    Comprehensive parameter optimizer for momentum factor IC analysis.
    """
    
    def __init__(self):
        """Initialize the parameter optimizer."""
        self.results = {}
        self.optimal_params = {}
        self.engine = None
        
    def initialize_engine(self):
        """Initialize the QVM engine."""
        try:
            self.engine = QVMEngineV2Enhanced()
            print("✅ QVM Engine initialized")
        except Exception as e:
            print(f"❌ Failed to initialize engine: {e}")
            raise
    
    def get_test_universe(self, analysis_date, limit=50):
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
            LIMIT {limit}
            """
            with self.engine.engine.connect() as conn:
                result = pd.read_sql(query, conn)
            return result['ticker'].tolist()
        except Exception as e:
            print(f"❌ Failed to get universe: {e}")
            return []
    
    def calculate_momentum_ic_custom(self, analysis_date, universe, 
                                   lookback_weights, forward_months=1):
        """
        Calculate momentum IC with custom parameters.
        
        Args:
            analysis_date: Date for analysis
            universe: List of ticker symbols
            lookback_weights: Dict of lookback periods and weights
            forward_months: Forward return horizon
        """
        try:
            # Get fundamental data for sector mapping
            fundamental_data = self.engine.get_fundamentals_correct_timing(analysis_date, universe)
            if fundamental_data.empty:
                return None
            
            # Calculate momentum with custom weights
            momentum_scores = self._calculate_custom_momentum(
                fundamental_data, analysis_date, universe, lookback_weights
            )
            
            if not momentum_scores:
                return None
            
            # Calculate forward returns
            end_date = analysis_date + pd.DateOffset(months=forward_months)
            ticker_str = "', '".join(universe)
            query = f"""
            SELECT ticker, date, close as adj_close
            FROM equity_history
            WHERE ticker IN ('{ticker_str}')
              AND date BETWEEN '{analysis_date.date()}' AND '{end_date.date()}'
            ORDER BY ticker, date
            """
            
            with self.engine.engine.connect() as conn:
                price_data = pd.read_sql(query, conn, parse_dates=['date'])
            
            if price_data.empty:
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
            
            # Calculate IC
            common_tickers = set(momentum_scores.keys()) & set(forward_returns.keys())
            if len(common_tickers) < 10:
                return None
            
            factor_series = pd.Series([momentum_scores[t] for t in common_tickers], 
                                    index=list(common_tickers))
            return_series = pd.Series([forward_returns[t] for t in common_tickers], 
                                    index=list(common_tickers))
            
            ic = factor_series.corr(return_series, method='spearman')
            
            return {
                'date': analysis_date,
                'ic': ic,
                'n_stocks': len(common_tickers)
            }
            
        except Exception as e:
            print(f"❌ Error calculating IC: {e}")
            return None
    
    def _calculate_custom_momentum(self, fundamental_data, analysis_date, universe, lookback_weights):
        """Calculate momentum with custom lookback weights."""
        try:
            momentum_scores = {}
            
            # Get price data
            ticker_str = "', '".join(universe)
            start_date = analysis_date - pd.DateOffset(months=14)
            
            price_query = f"""
            SELECT date, ticker, close as adj_close
            FROM equity_history
            WHERE ticker IN ('{ticker_str}')
              AND date BETWEEN '{start_date.date()}' AND '{analysis_date.date()}'
            ORDER BY ticker, date
            """
            
            price_data = pd.read_sql(price_query, self.engine.engine, parse_dates=['date'])
            
            if price_data.empty:
                return {}
            
            # Calculate returns for each lookback period
            period_returns = {}
            for lookback_months, weight in lookback_weights.items():
                end_date = analysis_date - pd.DateOffset(months=1)  # Skip 1 month
                start_date_period = analysis_date - pd.DateOffset(months=lookback_months + 1)
                returns = self._calculate_returns_fixed(price_data, start_date_period, end_date)
                
                if not returns.empty:
                    period_returns[lookback_months] = returns
            
            # Combine momentum periods with custom weights
            for ticker in universe:
                momentum_components = []
                
                for lookback_months, weight in lookback_weights.items():
                    if lookback_months in period_returns and ticker in period_returns[lookback_months]:
                        momentum_components.append(weight * period_returns[lookback_months][ticker])
                
                if momentum_components:
                    momentum_scores[ticker] = sum(momentum_components)
                else:
                    momentum_scores[ticker] = 0.0
            
            # Normalize
            if momentum_scores:
                momentum_series = pd.Series(momentum_scores)
                momentum_mean = momentum_series.mean()
                momentum_std = momentum_series.std()
                
                if momentum_std > 0:
                    for ticker in momentum_scores:
                        momentum_scores[ticker] = (momentum_scores[ticker] - momentum_mean) / momentum_std
                        momentum_scores[ticker] = np.clip(momentum_scores[ticker], -3, 3)
            
            return momentum_scores
            
        except Exception as e:
            print(f"❌ Error calculating custom momentum: {e}")
            return {}
    
    def _calculate_returns_fixed(self, price_data, start_date, end_date):
        """Calculate returns between two dates."""
        try:
            returns = {}
            
            for ticker in price_data['ticker'].unique():
                ticker_data = price_data[price_data['ticker'] == ticker].sort_values('date')
                
                # Find closest dates
                start_mask = ticker_data['date'] >= start_date
                end_mask = ticker_data['date'] <= end_date
                
                if start_mask.any() and end_mask.any():
                    start_price = ticker_data[start_mask].iloc[0]['adj_close']
                    end_price = ticker_data[end_mask].iloc[-1]['adj_close']
                    
                    if start_price > 0:
                        returns[ticker] = (end_price / start_price) - 1
            
            return pd.Series(returns)
            
        except Exception as e:
            print(f"❌ Error calculating returns: {e}")
            return pd.Series()
    
    def test_parameter_combination(self, lookback_weights, forward_months, test_dates):
        """Test a specific parameter combination."""
        results = []
        
        for test_date in test_dates:
            universe = self.get_test_universe(test_date)
            if len(universe) < 20:
                continue
            
            ic_result = self.calculate_momentum_ic_custom(
                test_date, universe, lookback_weights, forward_months
            )
            
            if ic_result:
                results.append(ic_result)
        
        if results:
            ic_values = [r['ic'] for r in results if not np.isnan(r['ic'])]
            if ic_values:
                ic_series = pd.Series(ic_values)
                return {
                    'mean_ic': ic_series.mean(),
                    'std_ic': ic_series.std(),
                    't_stat': ic_series.mean() / (ic_series.std() / np.sqrt(len(ic_series))) if ic_series.std() > 0 else 0,
                    'hit_rate': (ic_series > 0).mean(),
                    'n_observations': len(ic_series)
                }
        
        return None
    
    def optimize_parameters(self, start_date='2017-01-01', end_date='2024-12-31'):
        """Perform comprehensive parameter optimization."""
        print("🚀 Starting Momentum Parameter Optimization")
        print("=" * 60)
        
        # Generate test dates
        test_dates = pd.date_range(
            start=pd.to_datetime(start_date) + pd.DateOffset(months=12),
            end=pd.to_datetime(end_date) - pd.DateOffset(months=12),
            freq='Q'
        )
        
        print(f"📊 Generated {len(test_dates)} test dates")
        
        # Parameter combinations to test
        lookback_combinations = [
            {1: 0.25, 3: 0.25, 6: 0.25, 12: 0.25},  # Equal weights
            {1: 0.15, 3: 0.25, 6: 0.30, 12: 0.30},  # Current weights
            {1: 0.10, 3: 0.20, 6: 0.35, 12: 0.35},  # Long-term focus
            {1: 0.40, 3: 0.30, 6: 0.20, 12: 0.10},  # Short-term focus
            {1: 0.20, 3: 0.30, 6: 0.30, 12: 0.20},  # Medium-term focus
        ]
        
        forward_horizons = [1, 3, 6, 12]
        
        # Test all combinations
        for lookback_weights in lookback_combinations:
            for forward_months in forward_horizons:
                print(f"\n📈 Testing: {lookback_weights} with {forward_months}M forward returns")
                
                result = self.test_parameter_combination(
                    lookback_weights, forward_months, test_dates
                )
                
                if result:
                    key = f"{str(lookback_weights)}_{forward_months}M"
                    self.results[key] = result
                    print(f"✅ Mean IC: {result['mean_ic']:.4f}, T-stat: {result['t_stat']:.3f}")
        
        # Find optimal parameters
        self._find_optimal_parameters()
        
        return self.results
    
    def _find_optimal_parameters(self):
        """Find optimal parameter combinations."""
        if not self.results:
            return
        
        # Find best by mean IC
        best_ic = max(self.results.items(), key=lambda x: x[1]['mean_ic'])
        
        # Find best by t-statistic
        best_tstat = max(self.results.items(), key=lambda x: x[1]['t_stat'])
        
        # Find best by hit rate
        best_hitrate = max(self.results.items(), key=lambda x: x[1]['hit_rate'])
        
        self.optimal_params = {
            'best_ic': best_ic,
            'best_tstat': best_tstat,
            'best_hitrate': best_hitrate
        }
        
        print(f"\n🎯 OPTIMAL PARAMETERS:")
        print(f"Best IC: {best_ic[0]} (IC: {best_ic[1]['mean_ic']:.4f})")
        print(f"Best T-stat: {best_tstat[0]} (T-stat: {best_tstat[1]['t_stat']:.3f})")
        print(f"Best Hit Rate: {best_hitrate[0]} (Hit Rate: {best_hitrate[1]['hit_rate']:.1%})")
    
    def generate_optimization_report(self):
        """Generate comprehensive optimization report."""
        if not self.results:
            return "No results available"
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("MOMENTUM PARAMETER OPTIMIZATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Results table
        report_lines.append("PARAMETER COMBINATION RESULTS")
        report_lines.append("-" * 60)
        report_lines.append(f"{'Parameters':<40} {'Mean IC':<10} {'T-Stat':<8} {'Hit Rate':<10}")
        report_lines.append("-" * 60)
        
        for key, result in self.results.items():
            report_lines.append(
                f"{key:<40} {result['mean_ic']:<10.4f} {result['t_stat']:<8.3f} {result['hit_rate']:<10.1%}"
            )
        
        report_lines.append("")
        
        # Optimal parameters
        if self.optimal_params:
            report_lines.append("OPTIMAL PARAMETERS")
            report_lines.append("-" * 30)
            for metric, (params, stats) in self.optimal_params.items():
                report_lines.append(f"{metric.replace('_', ' ').title()}: {params}")
                report_lines.append(f"  IC: {stats['mean_ic']:.4f}, T-stat: {stats['t_stat']:.3f}, Hit Rate: {stats['hit_rate']:.1%}")
                report_lines.append("")
        
        return "\n".join(report_lines)

def main():
    """Main execution function."""
    print("🚀 MOMENTUM PARAMETER OPTIMIZATION")
    print("=" * 60)
    
    try:
        optimizer = MomentumParameterOptimizer()
        optimizer.initialize_engine()
        
        # Run optimization
        results = optimizer.optimize_parameters()
        
        # Generate report
        report = optimizer.generate_optimization_report()
        print("\n" + report)
        
        # Save report
        with open('data/momentum_parameter_optimization_report.txt', 'w') as f:
            f.write(report)
        
        print("\n✅ Parameter optimization completed successfully!")
        print("📄 Report saved to: data/momentum_parameter_optimization_report.txt")
        
        return results
        
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main() 