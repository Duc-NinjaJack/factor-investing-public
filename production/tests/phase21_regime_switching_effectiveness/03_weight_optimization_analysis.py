#!/usr/bin/env python3
"""
Weight Optimization Analysis for Regime Switching

This script optimizes factor weights for different market regimes to maximize
performance while maintaining risk management objectives.

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

class RegimeWeightOptimizer:
    """
    Optimizes factor weights for different market regimes.
    """
    
    def __init__(self):
        """Initialize the weight optimizer."""
        self.results = {}
        self.optimal_weights = {}
        
    def identify_market_regimes(self, benchmark_returns: pd.Series) -> pd.DataFrame:
        """
        Identifies market regimes using optimized parameters.
        """
        # Use optimized parameters from previous analysis
        bear_threshold = -0.25  # Optimized from parameter analysis
        vol_window = 90
        trend_window = 300
        
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
    
    def optimize_regime_weights(self, regimes: pd.DataFrame,
                              factor_returns: Dict[str, pd.Series],
                              benchmark_returns: pd.Series) -> Dict:
        """
        Optimize weights for each regime separately.
        """
        print("🔍 Starting weight optimization analysis...")
        
        # Define weight ranges to test (in 10% increments)
        weight_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        # Optimize weights for each regime
        regime_optimizations = {}
        
        for regime in ['Bear', 'Stress', 'Bull', 'Sideways']:
            print(f"\n📊 Optimizing weights for {regime} regime...")
            
            # Get regime-specific data
            regime_mask = regimes['regime'] == regime
            if regime_mask.sum() < 50:  # Need sufficient data
                print(f"   ⚠️ Insufficient data for {regime} regime ({regime_mask.sum()} days)")
                continue
            
            regime_returns = {name: returns[regime_mask] for name, returns in factor_returns.items()}
            regime_benchmark = benchmark_returns[regime_mask]
            
            # Test different weight combinations
            best_performance = {'sharpe_ratio': -np.inf}
            best_weights = {}
            
            # Test all valid weight combinations (sum to 1.0)
            valid_combinations = 0
            
            for q_weight in weight_values:
                for v_weight in weight_values:
                    for m_weight in weight_values:
                        # Check if weights sum to 1.0
                        if abs(q_weight + v_weight + m_weight - 1.0) < 0.001:
                            valid_combinations += 1
                            
                            # Calculate weighted returns
                            weighted_returns = (
                                q_weight * regime_returns['Quality'] +
                                v_weight * regime_returns['Value'] +
                                m_weight * regime_returns['Momentum']
                            )
                            
                            # Calculate performance metrics
                            performance = self.calculate_performance_metrics(
                                weighted_returns, regime_benchmark)
                            
                            # Update best if better Sharpe ratio
                            if performance['sharpe_ratio'] > best_performance['sharpe_ratio']:
                                best_performance = performance
                                best_weights = {
                                    'Quality': q_weight,
                                    'Value': v_weight,
                                    'Momentum': m_weight
                                }
            
            print(f"   Tested {valid_combinations} weight combinations")
            print(f"   Best Sharpe: {best_performance['sharpe_ratio']:.2f}")
            print(f"   Best weights: {best_weights}")
            
            regime_optimizations[regime] = {
                'optimal_weights': best_weights,
                'performance': best_performance,
                'days_analyzed': regime_mask.sum()
            }
        
        self.results = regime_optimizations
        return regime_optimizations
    
    def calculate_performance_metrics(self, returns: pd.Series, 
                                    benchmark: pd.Series) -> Dict:
        """
        Calculate comprehensive performance metrics.
        """
        # Basic metrics
        total_return = (1 + returns).prod() - 1
        n_years = len(returns) / 252
        annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Drawdown calculation
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative / running_max - 1)
        max_drawdown = drawdown.min()
        
        # Information ratio
        excess_returns = returns - benchmark
        tracking_error = excess_returns.std() * np.sqrt(252)
        information_ratio = excess_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0
        
        return {
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'information_ratio': information_ratio
        }
    
    def test_optimized_strategy(self, regimes: pd.DataFrame,
                              factor_returns: Dict[str, pd.Series],
                              benchmark_returns: pd.Series,
                              optimized_weights: Dict) -> Dict:
        """
        Test the optimized strategy against baseline.
        """
        print("\n🧪 Testing optimized strategy...")
        
        # Baseline weights (equal weight)
        baseline_weights = {
            'Bear': {'Quality': 0.33, 'Value': 0.33, 'Momentum': 0.34},
            'Stress': {'Quality': 0.33, 'Value': 0.33, 'Momentum': 0.34},
            'Bull': {'Quality': 0.33, 'Value': 0.33, 'Momentum': 0.34},
            'Sideways': {'Quality': 0.33, 'Value': 0.33, 'Momentum': 0.34}
        }
        
        # Calculate returns for both strategies
        baseline_returns = self.calculate_regime_weighted_returns(
            factor_returns, regimes, baseline_weights)
        optimized_returns = self.calculate_regime_weighted_returns(
            factor_returns, regimes, optimized_weights)
        
        # Calculate performance metrics
        baseline_performance = self.calculate_performance_metrics(
            baseline_returns, benchmark_returns)
        optimized_performance = self.calculate_performance_metrics(
            optimized_returns, benchmark_returns)
        
        # Calculate improvements
        improvements = {
            'return_improvement': optimized_performance['annual_return'] - baseline_performance['annual_return'],
            'sharpe_improvement': optimized_performance['sharpe_ratio'] - baseline_performance['sharpe_ratio'],
            'max_dd_improvement': baseline_performance['max_drawdown'] - optimized_performance['max_drawdown'],
            'info_ratio_improvement': optimized_performance['information_ratio'] - baseline_performance['information_ratio']
        }
        
        return {
            'baseline_performance': baseline_performance,
            'optimized_performance': optimized_performance,
            'improvements': improvements
        }
    
    def calculate_regime_weighted_returns(self, factor_returns: Dict[str, pd.Series],
                                        regimes: pd.DataFrame,
                                        regime_weights: Dict[str, Dict[str, float]]) -> pd.Series:
        """
        Calculate regime-weighted returns.
        """
        weighted_returns = pd.Series(0.0, index=list(factor_returns.values())[0].index)
        
        for date in weighted_returns.index:
            if date in regimes.index:
                regime = regimes.loc[date, 'regime']
                weights = regime_weights.get(regime, {})
                
                for factor_name, factor_returns_series in factor_returns.items():
                    weight = weights.get(factor_name, 0)
                    weighted_returns.loc[date] += weight * factor_returns_series.loc[date]
        
        return weighted_returns
    
    def generate_weight_optimization_report(self) -> str:
        """
        Generate comprehensive weight optimization report.
        """
        if not self.results:
            return "No weight optimization results available."
        
        report = []
        report.append("="*80)
        report.append("REGIME SWITCHING WEIGHT OPTIMIZATION REPORT")
        report.append("="*80)
        
        # Summary of optimizations
        report.append(f"\n📊 WEIGHT OPTIMIZATION SUMMARY:")
        
        for regime, result in self.results.items():
            weights = result['optimal_weights']
            performance = result['performance']
            days = result['days_analyzed']
            
            report.append(f"\n   🎯 {regime.upper()} REGIME ({days} days):")
            report.append(f"      Optimal weights: Quality={weights['Quality']:.1%}, "
                         f"Value={weights['Value']:.1%}, Momentum={weights['Momentum']:.1%}")
            report.append(f"      Annual return: {performance['annual_return']:.2%}")
            report.append(f"      Sharpe ratio: {performance['sharpe_ratio']:.2f}")
            report.append(f"      Max drawdown: {performance['max_drawdown']:.2%}")
        
        # Strategy comparison
        if hasattr(self, 'strategy_comparison'):
            comp = self.strategy_comparison
            report.append(f"\n📈 STRATEGY COMPARISON:")
            report.append(f"   Baseline vs Optimized:")
            report.append(f"      Return improvement: {comp['improvements']['return_improvement']:+.2%}")
            report.append(f"      Sharpe improvement: {comp['improvements']['sharpe_improvement']:+.2f}")
            report.append(f"      Max DD improvement: {comp['improvements']['max_dd_improvement']:+.2%}")
            report.append(f"      Info ratio improvement: {comp['improvements']['info_ratio_improvement']:+.2f}")
        
        # Recommendations
        report.append(f"\n💡 RECOMMENDATIONS:")
        report.append(f"   1. Implement regime-specific optimal weights")
        report.append(f"   2. Monitor regime transitions for smooth weight changes")
        report.append(f"   3. Consider transaction costs in implementation")
        report.append(f"   4. Validate with out-of-sample testing")
        
        return "\n".join(report)

def main():
    """
    Main execution function for weight optimization.
    """
    print("🚀 Starting Regime Switching Weight Optimization")
    print("="*80)
    
    # Initialize optimizer
    optimizer = RegimeWeightOptimizer()
    
    # Create synthetic test data
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
    
    # Identify regimes
    print("\n🔍 Identifying market regimes...")
    regimes = optimizer.identify_market_regimes(benchmark_returns)
    
    # Optimize weights for each regime
    print("\n🧪 Running weight optimization...")
    weight_results = optimizer.optimize_regime_weights(regimes, factor_returns, benchmark_returns)
    
    # Extract optimized weights
    optimized_weights = {regime: result['optimal_weights'] 
                        for regime, result in weight_results.items()}
    
    # Test optimized strategy
    print("\n🧪 Testing optimized strategy...")
    strategy_comparison = optimizer.test_optimized_strategy(
        regimes, factor_returns, benchmark_returns, optimized_weights)
    
    optimizer.strategy_comparison = strategy_comparison
    
    # Generate and display report
    print("\n" + "="*80)
    print("📋 GENERATING WEIGHT OPTIMIZATION REPORT")
    print("="*80)
    
    report = optimizer.generate_weight_optimization_report()
    print(report)
    
    # Save results
    print(f"\n💾 Saving results...")
    
    # Save weight optimization report
    with open('docs/weight_optimization_report.md', 'w') as f:
        f.write(report)
    
    # Save optimized weights
    weights_df = pd.DataFrame(optimized_weights).T
    weights_df.to_csv('data/optimized_regime_weights.csv')
    
    print("✅ Weight optimization completed successfully!")
    print("📁 Results saved to:")
    print("   - docs/weight_optimization_report.md")
    print("   - data/optimized_regime_weights.csv")

if __name__ == "__main__":
    main()