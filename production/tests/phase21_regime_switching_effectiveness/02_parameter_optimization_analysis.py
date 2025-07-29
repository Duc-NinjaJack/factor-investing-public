#!/usr/bin/env python3
"""
Parameter Optimization Analysis for Regime Switching

This script performs comprehensive parameter optimization for the regime switching
methodology, testing various combinations of thresholds and windows to identify
optimal settings.

Author: AI Assistant
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class RegimeParameterOptimizer:
    """
    Comprehensive parameter optimizer for regime switching methodology.
    """
    
    def __init__(self):
        """Initialize the parameter optimizer."""
        self.results = {}
        self.optimal_params = {}
        
    def identify_market_regimes(self, benchmark_returns: pd.Series, 
                              bear_threshold: float = -0.20,
                              vol_window: int = 60,
                              trend_window: int = 200) -> pd.DataFrame:
        """
        Identifies market regimes using multiple criteria.
        """
        # Calculate cumulative returns and drawdowns
        cumulative = (1 + benchmark_returns).cumprod()
        drawdown = (cumulative / cumulative.cummax() - 1)
        
        # 1. Bear Market Regime
        is_bear = drawdown < bear_threshold
        
        # 2. High-Stress Regime (rolling volatility)
        rolling_vol = benchmark_returns.rolling(vol_window).std() * np.sqrt(252)
        vol_75th = rolling_vol.quantile(0.75)
        is_stress = rolling_vol > vol_75th
        
        # 3. Bull/Sideways (trend-based)
        trend_ma = cumulative.rolling(trend_window).mean()
        is_above_trend = cumulative > trend_ma
        
        # Combine into regime classification
        regimes = pd.DataFrame(index=benchmark_returns.index)
        regimes['is_bear'] = is_bear
        regimes['is_stress'] = is_stress
        regimes['is_bull'] = is_above_trend & ~is_bear & ~is_stress
        regimes['is_sideways'] = ~is_above_trend & ~is_bear & ~is_stress
        
        # Create primary regime classification
        regimes['regime'] = 'Undefined'
        regimes.loc[regimes['is_bear'], 'regime'] = 'Bear'
        regimes.loc[regimes['is_stress'] & ~regimes['is_bear'], 'regime'] = 'Stress'
        regimes.loc[regimes['is_bull'], 'regime'] = 'Bull'
        regimes.loc[regimes['is_sideways'], 'regime'] = 'Sideways'
        
        return regimes
    
    def calculate_regime_accuracy(self, regimes: pd.DataFrame, 
                                known_periods: Dict[str, Tuple[str, str]]) -> float:
        """
        Calculate regime identification accuracy against known market periods.
        """
        accuracies = []
        
        for period_name, (start_date, end_date) in known_periods.items():
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            # Get regime classifications for this period
            period_mask = (regimes.index >= start_dt) & (regimes.index <= end_dt)
            period_regimes = regimes[period_mask]
            
            if len(period_regimes) > 0:
                # Calculate regime distribution
                regime_dist = period_regimes['regime'].value_counts(normalize=True)
                
                # Expected regimes for different periods
                expected_regimes = {
                    '2008_crisis': ['Bear', 'Stress'],
                    '2020_covid': ['Bear', 'Stress'],
                    '2018_trade_war': ['Bear', 'Sideways'],
                    '2022_inflation': ['Stress', 'Sideways']
                }
                
                expected = expected_regimes.get(period_name, [])
                
                # Calculate accuracy (percentage of days in expected regimes)
                if expected:
                    accuracy = sum(regime_dist.get(regime, 0) for regime in expected)
                else:
                    accuracy = 0
                
                accuracies.append(accuracy)
        
        return np.mean(accuracies) if accuracies else 0
    
    def optimize_parameters(self, benchmark_returns: pd.Series,
                          factor_returns: Dict[str, pd.Series]) -> Dict:
        """
        Perform comprehensive parameter optimization.
        """
        print("🔍 Starting parameter optimization analysis...")
        
        # Define parameter ranges to test
        bear_thresholds = [-0.15, -0.20, -0.25, -0.30]
        vol_windows = [30, 60, 90, 120]
        trend_windows = [100, 200, 300, 400]
        
        # Known market periods for validation
        known_periods = {
            '2008_crisis': ('2008-09-01', '2009-03-31'),
            '2020_covid': ('2020-02-20', '2020-04-30'),
            '2018_trade_war': ('2018-03-01', '2018-12-31'),
            '2022_inflation': ('2022-01-01', '2022-12-31')
        }
        
        # Test all parameter combinations
        results = []
        total_combinations = len(bear_thresholds) * len(vol_windows) * len(trend_windows)
        
        print(f"📊 Testing {total_combinations} parameter combinations...")
        
        for i, (bear_thresh, vol_win, trend_win) in enumerate(
            product(bear_thresholds, vol_windows, trend_windows)):
            
            # Generate regimes with current parameters
            regimes = self.identify_market_regimes(
                benchmark_returns, bear_thresh, vol_win, trend_win)
            
            # Calculate accuracy
            accuracy = self.calculate_regime_accuracy(regimes, known_periods)
            
            # Calculate regime distribution
            regime_dist = regimes['regime'].value_counts(normalize=True)
            
            # Calculate strategy performance
            strategy_performance = self.calculate_strategy_performance(
                regimes, factor_returns, benchmark_returns)
            
            result = {
                'bear_threshold': bear_thresh,
                'vol_window': vol_win,
                'trend_window': trend_win,
                'accuracy': accuracy,
                'annual_return': strategy_performance['annual_return'],
                'sharpe_ratio': strategy_performance['sharpe_ratio'],
                'max_drawdown': strategy_performance['max_drawdown'],
                'regime_distribution': regime_dist.to_dict()
            }
            
            results.append(result)
            
            if (i + 1) % 10 == 0:
                print(f"   Progress: {i + 1}/{total_combinations} combinations tested")
        
        # Convert to DataFrame for analysis
        results_df = pd.DataFrame(results)
        
        # Find optimal parameters by different criteria
        optimal_by_accuracy = results_df.loc[results_df['accuracy'].idxmax()]
        optimal_by_sharpe = results_df.loc[results_df['sharpe_ratio'].idxmax()]
        optimal_by_return = results_df.loc[results_df['annual_return'].idxmax()]
        
        # Calculate composite score (weighted average)
        results_df['composite_score'] = (
            results_df['accuracy'] * 0.4 +
            results_df['sharpe_ratio'] * 0.3 +
            (results_df['annual_return'] * 100) * 0.3  # Scale return to similar range
        )
        
        optimal_composite = results_df.loc[results_df['composite_score'].idxmax()]
        
        self.results = {
            'all_results': results_df,
            'optimal_by_accuracy': optimal_by_accuracy,
            'optimal_by_sharpe': optimal_by_sharpe,
            'optimal_by_return': optimal_by_return,
            'optimal_composite': optimal_composite
        }
        
        return self.results
    
    def calculate_strategy_performance(self, regimes: pd.DataFrame,
                                     factor_returns: Dict[str, pd.Series],
                                     benchmark_returns: pd.Series) -> Dict:
        """
        Calculate dynamic strategy performance with given regimes.
        """
        # Define dynamic weights based on regimes
        dynamic_weights = {
            'Bear': {'Quality': 0.50, 'Value': 0.30, 'Momentum': 0.20},
            'Stress': {'Quality': 0.40, 'Value': 0.35, 'Momentum': 0.25},
            'Bull': {'Quality': 0.25, 'Value': 0.30, 'Momentum': 0.45},
            'Sideways': {'Quality': 0.35, 'Value': 0.35, 'Momentum': 0.30}
        }
        
        # Calculate regime-weighted returns
        weighted_returns = pd.Series(0.0, index=benchmark_returns.index)
        
        for date in weighted_returns.index:
            if date in regimes.index:
                regime = regimes.loc[date, 'regime']
                weights = dynamic_weights.get(regime, {})
                
                for factor_name, factor_returns_series in factor_returns.items():
                    weight = weights.get(factor_name, 0)
                    weighted_returns.loc[date] += weight * factor_returns_series.loc[date]
        
        # Calculate performance metrics
        total_return = (1 + weighted_returns).prod() - 1
        n_years = len(weighted_returns) / 252
        annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        
        volatility = weighted_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Drawdown calculation
        cumulative = (1 + weighted_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative / running_max - 1)
        max_drawdown = drawdown.min()
        
        return {
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
    
    def generate_optimization_report(self) -> str:
        """
        Generate comprehensive optimization report.
        """
        if not self.results:
            return "No optimization results available."
        
        report = []
        report.append("="*80)
        report.append("REGIME SWITCHING PARAMETER OPTIMIZATION REPORT")
        report.append("="*80)
        
        # Summary statistics
        results_df = self.results['all_results']
        report.append(f"\n📊 OPTIMIZATION SUMMARY:")
        report.append(f"   Total combinations tested: {len(results_df)}")
        report.append(f"   Accuracy range: {results_df['accuracy'].min():.1%} - {results_df['accuracy'].max():.1%}")
        report.append(f"   Sharpe ratio range: {results_df['sharpe_ratio'].min():.2f} - {results_df['sharpe_ratio'].max():.2f}")
        report.append(f"   Annual return range: {results_df['annual_return'].min():.2%} - {results_df['annual_return'].max():.2%}")
        
        # Optimal parameters by different criteria
        report.append(f"\n🎯 OPTIMAL PARAMETERS BY CRITERIA:")
        
        # By accuracy
        opt_acc = self.results['optimal_by_accuracy']
        report.append(f"\n   📈 BY ACCURACY (Best: {opt_acc['accuracy']:.1%}):")
        report.append(f"      Bear threshold: {opt_acc['bear_threshold']}")
        report.append(f"      Volatility window: {opt_acc['vol_window']} days")
        report.append(f"      Trend window: {opt_acc['trend_window']} days")
        report.append(f"      Annual return: {opt_acc['annual_return']:.2%}")
        report.append(f"      Sharpe ratio: {opt_acc['sharpe_ratio']:.2f}")
        
        # By Sharpe ratio
        opt_sharpe = self.results['optimal_by_sharpe']
        report.append(f"\n   📊 BY SHARPE RATIO (Best: {opt_sharpe['sharpe_ratio']:.2f}):")
        report.append(f"      Bear threshold: {opt_sharpe['bear_threshold']}")
        report.append(f"      Volatility window: {opt_sharpe['vol_window']} days")
        report.append(f"      Trend window: {opt_sharpe['trend_window']} days")
        report.append(f"      Accuracy: {opt_sharpe['accuracy']:.1%}")
        report.append(f"      Annual return: {opt_sharpe['annual_return']:.2%}")
        
        # By composite score
        opt_comp = self.results['optimal_composite']
        report.append(f"\n   🏆 BY COMPOSITE SCORE (Best: {opt_comp['composite_score']:.3f}):")
        report.append(f"      Bear threshold: {opt_comp['bear_threshold']}")
        report.append(f"      Volatility window: {opt_comp['vol_window']} days")
        report.append(f"      Trend window: {opt_comp['trend_window']} days")
        report.append(f"      Accuracy: {opt_comp['accuracy']:.1%}")
        report.append(f"      Sharpe ratio: {opt_comp['sharpe_ratio']:.2f}")
        report.append(f"      Annual return: {opt_comp['annual_return']:.2%}")
        
        # Recommendations
        report.append(f"\n💡 RECOMMENDATIONS:")
        report.append(f"   1. Use composite score optimal parameters for balanced performance")
        report.append(f"   2. Consider accuracy-optimal parameters if regime identification is priority")
        report.append(f"   3. Use Sharpe-optimal parameters if risk-adjusted returns are priority")
        
        return "\n".join(report)

def main():
    """
    Main execution function for parameter optimization.
    """
    print("🚀 Starting Regime Switching Parameter Optimization")
    print("="*80)
    
    # Initialize optimizer
    optimizer = RegimeParameterOptimizer()
    
    # Create synthetic test data (same as validation tests)
    print("\n📊 Creating synthetic test data...")
    
    np.random.seed(42)
    dates = pd.date_range('2008-01-01', '2025-01-01', freq='D')
    dates = dates[dates.weekday < 5]  # Business days only
    
    # Create realistic market returns with regime changes
    n_days = len(dates)
    returns = np.random.normal(0.0005, 0.015, n_days)  # Base returns
    
    # Add regime-specific patterns
    # 2008 crisis period
    crisis_start = dates.get_loc('2008-09-01')
    crisis_end = dates.get_loc('2009-03-31')
    returns[crisis_start:crisis_end] += np.random.normal(-0.002, 0.025, crisis_end-crisis_start)
    
    # 2020 COVID period
    covid_start = dates.get_loc('2020-02-20')
    covid_end = dates.get_loc('2020-04-30')
    returns[covid_start:covid_end] += np.random.normal(-0.003, 0.030, covid_end-covid_start)
    
    benchmark_returns = pd.Series(returns, index=dates)
    
    # Generate synthetic factor returns
    factor_returns = {
        'Quality': benchmark_returns + np.random.normal(0.0001, 0.008, n_days),
        'Value': benchmark_returns + np.random.normal(0.0002, 0.010, n_days),
        'Momentum': benchmark_returns + np.random.normal(0.0003, 0.012, n_days)
    }
    
    print(f"✅ Test data created: {len(benchmark_returns):,} trading days")
    
    # Run parameter optimization
    print("\n🧪 Running parameter optimization...")
    results = optimizer.optimize_parameters(benchmark_returns, factor_returns)
    
    # Generate and display report
    print("\n" + "="*80)
    print("📋 GENERATING OPTIMIZATION REPORT")
    print("="*80)
    
    report = optimizer.generate_optimization_report()
    print(report)
    
    # Save results
    print(f"\n💾 Saving results...")
    
    # Save optimization report
    with open('docs/parameter_optimization_report.md', 'w') as f:
        f.write(report)
    
    # Save detailed results
    results['all_results'].to_csv('data/parameter_optimization_results.csv', index=False)
    
    print("✅ Parameter optimization completed successfully!")
    print("📁 Results saved to:")
    print("   - docs/parameter_optimization_report.md")
    print("   - data/parameter_optimization_results.csv")

if __name__ == "__main__":
    main()