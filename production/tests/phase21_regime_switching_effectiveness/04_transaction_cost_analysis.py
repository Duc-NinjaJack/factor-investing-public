#!/usr/bin/env python3
"""
Transaction Cost Analysis for Regime Switching

This script analyzes the impact of transaction costs on regime switching
strategies and provides recommendations for cost-effective implementation.

Author: AI Assistant
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class TransactionCostAnalyzer:
    """
    Analyzes transaction costs for regime switching strategies.
    """
    
    def __init__(self):
        """Initialize the transaction cost analyzer."""
        self.results = {}
        
    def identify_market_regimes(self, benchmark_returns: pd.Series) -> pd.DataFrame:
        """
        Identifies market regimes using optimized parameters.
        """
        # Use optimized parameters
        bear_threshold = -0.25
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
    
    def calculate_portfolio_weights(self, factor_returns: Dict[str, pd.Series],
                                  regimes: pd.DataFrame,
                                  regime_weights: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Calculate portfolio weights over time based on regime switching.
        """
        weights_df = pd.DataFrame(index=factor_returns['Quality'].index)
        
        for factor_name in factor_returns.keys():
            weights_df[factor_name] = 0.0
        
        for date in weights_df.index:
            if date in regimes.index:
                regime = regimes.loc[date, 'regime']
                weights = regime_weights.get(regime, {})
                
                for factor_name in factor_returns.keys():
                    weights_df.loc[date, factor_name] = weights.get(factor_name, 0.0)
        
        return weights_df
    
    def calculate_turnover(self, weights_df: pd.DataFrame) -> pd.Series:
        """
        Calculate portfolio turnover over time.
        """
        turnover = pd.Series(0.0, index=weights_df.index)
        
        for i in range(1, len(weights_df)):
            current_weights = weights_df.iloc[i]
            previous_weights = weights_df.iloc[i-1]
            
            # Calculate absolute change in weights
            weight_changes = abs(current_weights - previous_weights)
            turnover.iloc[i] = weight_changes.sum()
        
        return turnover
    
    def analyze_transaction_costs(self, weights_df: pd.DataFrame,
                                factor_returns: Dict[str, pd.Series],
                                benchmark_returns: pd.Series,
                                cost_scenarios: Dict[str, float]) -> Dict:
        """
        Analyze transaction costs under different scenarios.
        """
        print("🔍 Analyzing transaction costs...")
        
        # Calculate turnover
        turnover = self.calculate_turnover(weights_df)
        
        # Calculate strategy returns without costs
        strategy_returns = pd.Series(0.0, index=weights_df.index)
        for factor_name, factor_returns_series in factor_returns.items():
            strategy_returns += weights_df[factor_name] * factor_returns_series
        
        # Analyze each cost scenario
        cost_analysis = {}
        
        for scenario_name, cost_rate in cost_scenarios.items():
            print(f"   Analyzing {scenario_name} scenario ({cost_rate:.1%} cost rate)...")
            
            # Calculate transaction costs
            transaction_costs = turnover * cost_rate
            
            # Calculate net returns
            net_returns = strategy_returns - transaction_costs
            
            # Calculate performance metrics
            performance = self.calculate_performance_metrics(net_returns, benchmark_returns)
            
            # Calculate cost statistics
            total_cost = transaction_costs.sum()
            avg_cost = transaction_costs.mean()
            max_cost = transaction_costs.max()
            
            # Calculate turnover statistics
            avg_turnover = turnover.mean()
            max_turnover = turnover.max()
            total_turnover = turnover.sum()
            
            cost_analysis[scenario_name] = {
                'performance': performance,
                'total_cost': total_cost,
                'avg_cost': avg_cost,
                'max_cost': max_cost,
                'avg_turnover': avg_turnover,
                'max_turnover': max_turnover,
                'total_turnover': total_turnover,
                'cost_rate': cost_rate
            }
        
        return cost_analysis
    
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
    
    def optimize_rebalancing_frequency(self, weights_df: pd.DataFrame,
                                     factor_returns: Dict[str, pd.Series],
                                     benchmark_returns: pd.Series,
                                     cost_rate: float = 0.001) -> Dict:
        """
        Optimize rebalancing frequency to minimize transaction costs.
        """
        print("🔍 Optimizing rebalancing frequency...")
        
        # Test different rebalancing frequencies
        frequencies = [1, 5, 10, 20, 30, 60, 90]  # days
        frequency_results = {}
        
        for freq in frequencies:
            print(f"   Testing {freq}-day rebalancing...")
            
            # Create rebalanced weights (only change every 'freq' days)
            rebalanced_weights = weights_df.copy()
            
            for i in range(freq, len(weights_df), freq):
                # Keep weights constant for 'freq' days
                start_idx = i - freq
                end_idx = min(i, len(weights_df))
                
                # Use weights from the start of the period
                period_weights = weights_df.iloc[start_idx]
                
                for j in range(start_idx + 1, end_idx):
                    rebalanced_weights.iloc[j] = period_weights
            
            # Calculate turnover for rebalanced strategy
            rebalanced_turnover = self.calculate_turnover(rebalanced_weights)
            
            # Calculate strategy returns
            strategy_returns = pd.Series(0.0, index=rebalanced_weights.index)
            for factor_name, factor_returns_series in factor_returns.items():
                strategy_returns += rebalanced_weights[factor_name] * factor_returns_series
            
            # Calculate transaction costs
            transaction_costs = rebalanced_turnover * cost_rate
            net_returns = strategy_returns - transaction_costs
            
            # Calculate performance
            performance = self.calculate_performance_metrics(net_returns, benchmark_returns)
            
            # Calculate cost statistics
            total_cost = transaction_costs.sum()
            avg_turnover = rebalanced_turnover.mean()
            
            frequency_results[freq] = {
                'performance': performance,
                'total_cost': total_cost,
                'avg_turnover': avg_turnover,
                'frequency': freq
            }
        
        return frequency_results
    
    def generate_transaction_cost_report(self) -> str:
        """
        Generate comprehensive transaction cost analysis report.
        """
        if not self.results:
            return "No transaction cost analysis results available."
        
        report = []
        report.append("="*80)
        report.append("TRANSACTION COST ANALYSIS REPORT")
        report.append("="*80)
        
        # Cost scenario analysis
        if 'cost_analysis' in self.results:
            report.append(f"\n📊 TRANSACTION COST SCENARIO ANALYSIS:")
            
            for scenario, results in self.results['cost_analysis'].items():
                perf = results['performance']
                report.append(f"\n   💰 {scenario.upper()} SCENARIO:")
                report.append(f"      Cost rate: {results['cost_rate']:.1%}")
                report.append(f"      Annual return: {perf['annual_return']:.2%}")
                report.append(f"      Sharpe ratio: {perf['sharpe_ratio']:.2f}")
                report.append(f"      Total cost: {results['total_cost']:.2%}")
                report.append(f"      Average turnover: {results['avg_turnover']:.2%}")
        
        # Rebalancing frequency optimization
        if 'frequency_optimization' in self.results:
            report.append(f"\n📈 REBALANCING FREQUENCY OPTIMIZATION:")
            
            # Find optimal frequency by Sharpe ratio
            freq_results = self.results['frequency_optimization']
            best_sharpe = max(freq_results.items(), key=lambda x: x[1]['performance']['sharpe_ratio'])
            
            report.append(f"\n   🏆 OPTIMAL FREQUENCY: {best_sharpe[0]} days")
            report.append(f"      Sharpe ratio: {best_sharpe[1]['performance']['sharpe_ratio']:.2f}")
            report.append(f"      Annual return: {best_sharpe[1]['performance']['annual_return']:.2%}")
            report.append(f"      Total cost: {best_sharpe[1]['total_cost']:.2%}")
            report.append(f"      Average turnover: {best_sharpe[1]['avg_turnover']:.2%}")
        
        # Recommendations
        report.append(f"\n💡 RECOMMENDATIONS:")
        report.append(f"   1. Implement minimum rebalancing intervals (e.g., 20-30 days)")
        report.append(f"   2. Use regime confidence thresholds to reduce unnecessary switches")
        report.append(f"   3. Consider transaction costs in regime switching decisions")
        report.append(f"   4. Monitor turnover and adjust frequency based on market conditions")
        
        return "\n".join(report)

def main():
    """
    Main execution function for transaction cost analysis.
    """
    print("🚀 Starting Transaction Cost Analysis")
    print("="*80)
    
    # Initialize analyzer
    analyzer = TransactionCostAnalyzer()
    
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
    regimes = analyzer.identify_market_regimes(benchmark_returns)
    
    # Define regime weights (optimized from previous analysis)
    regime_weights = {
        'Bear': {'Quality': 0.60, 'Value': 0.30, 'Momentum': 0.10},
        'Stress': {'Quality': 0.30, 'Value': 0.40, 'Momentum': 0.30},
        'Bull': {'Quality': 0.20, 'Value': 0.25, 'Momentum': 0.55},
        'Sideways': {'Quality': 0.40, 'Value': 0.35, 'Momentum': 0.25}
    }
    
    # Calculate portfolio weights over time
    print("\n📊 Calculating portfolio weights...")
    weights_df = analyzer.calculate_portfolio_weights(factor_returns, regimes, regime_weights)
    
    # Analyze transaction costs under different scenarios
    print("\n🧪 Analyzing transaction costs...")
    cost_scenarios = {
        'Low Cost': 0.0005,    # 5 bps
        'Medium Cost': 0.001,  # 10 bps
        'High Cost': 0.002,    # 20 bps
        'Very High Cost': 0.005 # 50 bps
    }
    
    cost_analysis = analyzer.analyze_transaction_costs(
        weights_df, factor_returns, benchmark_returns, cost_scenarios)
    
    # Optimize rebalancing frequency
    print("\n🔍 Optimizing rebalancing frequency...")
    frequency_optimization = analyzer.optimize_rebalancing_frequency(
        weights_df, factor_returns, benchmark_returns, cost_rate=0.001)
    
    # Store results
    analyzer.results = {
        'cost_analysis': cost_analysis,
        'frequency_optimization': frequency_optimization
    }
    
    # Generate and display report
    print("\n" + "="*80)
    print("📋 GENERATING TRANSACTION COST REPORT")
    print("="*80)
    
    report = analyzer.generate_transaction_cost_report()
    print(report)
    
    # Save results
    print(f"\n💾 Saving results...")
    
    # Save transaction cost report
    with open('docs/transaction_cost_analysis_report.md', 'w') as f:
        f.write(report)
    
    # Save detailed results
    cost_df = pd.DataFrame([
        {
            'scenario': scenario,
            'cost_rate': results['cost_rate'],
            'annual_return': results['performance']['annual_return'],
            'sharpe_ratio': results['performance']['sharpe_ratio'],
            'total_cost': results['total_cost'],
            'avg_turnover': results['avg_turnover']
        }
        for scenario, results in cost_analysis.items()
    ])
    cost_df.to_csv('data/transaction_cost_analysis.csv', index=False)
    
    freq_df = pd.DataFrame([
        {
            'frequency_days': freq,
            'annual_return': results['performance']['annual_return'],
            'sharpe_ratio': results['performance']['sharpe_ratio'],
            'total_cost': results['total_cost'],
            'avg_turnover': results['avg_turnover']
        }
        for freq, results in frequency_optimization.items()
    ])
    freq_df.to_csv('data/rebalancing_frequency_optimization.csv', index=False)
    
    print("✅ Transaction cost analysis completed successfully!")
    print("📁 Results saved to:")
    print("   - docs/transaction_cost_analysis_report.md")
    print("   - data/transaction_cost_analysis.csv")
    print("   - data/rebalancing_frequency_optimization.csv")

if __name__ == "__main__":
    main()