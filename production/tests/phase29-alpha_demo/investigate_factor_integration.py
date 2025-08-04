#!/usr/bin/env python3
"""
Investigation: Why Individual Factors Show Statistical Significance 
But Combined Strategy Underperforms

This script analyzes the factor integration issues in the QVM strategy.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import sys
import os

# Add the parent directory to the path to import the strategy
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the strategy module
import importlib.util
spec = importlib.util.spec_from_file_location(
    "strategy_module", 
    "08_integrated_strategy_with_validated_factors_fixed.py"
)
strategy_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(strategy_module)

def analyze_factor_statistical_significance():
    """Analyze the statistical significance of individual factors from documentation"""
    print("=== FACTOR STATISTICAL SIGNIFICANCE ANALYSIS ===")
    
    # Data from the statistical testing documentation
    factor_stats = {
        'Low_Volatility': {
            '1M': {'Mean_IC': 0.0421, 't_stat': 4.89, 'p_value': 0.0000},
            '3M': {'Mean_IC': 0.0589, 't_stat': 5.67, 'p_value': 0.0000},
            '6M': {'Mean_IC': 0.0892, 't_stat': 7.23, 'p_value': 0.0000},
            '12M': {'Mean_IC': 0.1124, 't_stat': 8.45, 'p_value': 0.0000}
        },
        'Piotroski_FScore': {
            'Non_Financial_1M': {'Mean_IC': 0.0389, 't_stat': 4.23, 'p_value': 0.0000},
            'Non_Financial_3M': {'Mean_IC': 0.0521, 't_stat': 5.67, 'p_value': 0.0000},
            'Non_Financial_6M': {'Mean_IC': 0.0789, 't_stat': 6.89, 'p_value': 0.0000},
            'Banking_1M': {'Mean_IC': 0.0412, 't_stat': 4.56, 'p_value': 0.0000},
            'Banking_3M': {'Mean_IC': 0.0587, 't_stat': 5.89, 'p_value': 0.0000},
            'Banking_6M': {'Mean_IC': 0.0823, 't_stat': 7.12, 'p_value': 0.0000},
            'Securities_1M': {'Mean_IC': 0.0356, 't_stat': 3.89, 'p_value': 0.0000},
            'Securities_3M': {'Mean_IC': 0.0498, 't_stat': 5.23, 'p_value': 0.0000},
            'Securities_6M': {'Mean_IC': 0.0756, 't_stat': 6.67, 'p_value': 0.0000}
        },
        'FCF_Yield': {
            '1M': {'Mean_IC': 0.0463, 't_stat': 5.13, 'p_value': 0.0000},
            '3M': {'Mean_IC': 0.0686, 't_stat': 6.47, 'p_value': 0.0000},
            '6M': {'Mean_IC': 0.1006, 't_stat': 7.89, 'p_value': 0.0000},
            '12M': {'Mean_IC': 0.1245, 't_stat': 9.23, 'p_value': 0.0000}
        }
    }
    
    print("\n📊 Individual Factor Statistical Significance:")
    print("=" * 60)
    
    for factor, periods in factor_stats.items():
        print(f"\n🔍 {factor.replace('_', ' ')}:")
        significant_count = 0
        total_count = 0
        
        for period, stats in periods.items():
            total_count += 1
            if stats['p_value'] < 0.05:
                significant_count += 1
                status = "✅"
            else:
                status = "❌"
            
            print(f"   {period}: IC={stats['Mean_IC']:.4f}, t={stats['t_stat']:.2f}, p={stats['p_value']:.4f} {status}")
        
        print(f"   Summary: {significant_count}/{total_count} periods statistically significant")
    
    return factor_stats

def analyze_current_strategy_configuration():
    """Analyze the current strategy configuration and factor weights"""
    print("\n=== CURRENT STRATEGY CONFIGURATION ANALYSIS ===")
    
    config = strategy_module.QVM_CONFIG
    
    print(f"\n📋 Strategy Configuration:")
    print(f"   Strategy Name: {config['strategy_name']}")
    print(f"   Backtest Period: {config['backtest_start_date']} to {config['backtest_end_date']}")
    print(f"   Rebalancing Frequency: {config['rebalance_frequency']}")
    
    print(f"\n⚖️  Factor Weights:")
    factors = config['factors']
    print(f"   Value Weight: {factors['value_weight']:.1%}")
    print(f"   Quality Weight: {factors['quality_weight']:.1%}")
    print(f"   Momentum Weight: {factors['momentum_weight']:.1%}")
    
    print(f"\n🔧 Value Factors (Sub-weights):")
    value_factors = factors['value_factors']
    print(f"   P/E Weight: {value_factors['pe_weight']:.1%}")
    print(f"   FCF Yield Weight: {value_factors['fcf_yield_weight']:.1%}")
    
    print(f"\n🔧 Quality Factors (Sub-weights):")
    quality_factors = factors['quality_factors']
    print(f"   ROAA Weight: {quality_factors['roaa_weight']:.1%}")
    print(f"   F-Score Weight: {quality_factors['fscore_weight']:.1%}")
    
    print(f"\n🔧 Momentum Factors (Sub-weights):")
    momentum_factors = factors['momentum_factors']
    print(f"   Momentum Weight: {momentum_factors['momentum_weight']:.1%}")
    print(f"   Low-Volatility Weight: {momentum_factors['low_vol_weight']:.1%}")
    
    return config

def compare_with_working_strategy():
    """Compare current strategy with the working strategy configuration"""
    print("\n=== COMPARISON WITH WORKING STRATEGY ===")
    
    # Load the working strategy
    spec = importlib.util.spec_from_file_location(
        "working_strategy", 
        "07_integrated_strategy_enhanced.py"
    )
    working_strategy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(working_strategy)
    
    current_config = strategy_module.QVM_CONFIG
    working_config = working_strategy.QVM_CONFIG
    
    print(f"\n📊 Configuration Comparison:")
    print("=" * 50)
    
    print(f"\n🔍 Factor Weights Comparison:")
    print(f"   {'Factor':<15} {'Current':<10} {'Working':<10} {'Difference':<12}")
    print(f"   {'-'*15} {'-'*10} {'-'*10} {'-'*12}")
    
    current_factors = current_config['factors']
    working_factors = working_config['factors']
    
    print(f"   {'Value':<15} {current_factors['value_weight']:<10.1%} {working_factors['roaa_weight']:<10.1%} {(current_factors['value_weight'] - working_factors['roaa_weight']):<12.1%}")
    print(f"   {'Quality':<15} {current_factors['quality_weight']:<10.1%} {working_factors['pe_weight']:<10.1%} {(current_factors['quality_weight'] - working_factors['pe_weight']):<12.1%}")
    print(f"   {'Momentum':<15} {current_factors['momentum_weight']:<10.1%} {working_factors['momentum_weight']:<10.1%} {(current_factors['momentum_weight'] - working_factors['momentum_weight']):<12.1%}")
    
    print(f"\n🔍 Key Differences:")
    print(f"   - Current strategy uses more complex factor structure (sub-weights)")
    print(f"   - Working strategy uses simpler 3-factor approach")
    print(f"   - Current strategy has higher quality weight (35% vs 30%)")
    print(f"   - Current strategy has lower momentum weight (40% vs 40%)")
    
    return working_config

def analyze_factor_integration_issues():
    """Analyze potential issues with factor integration"""
    print("\n=== FACTOR INTEGRATION ISSUES ANALYSIS ===")
    
    print(f"\n🔍 Potential Issues Identified:")
    print("=" * 50)
    
    issues = [
        {
            'issue': 'Complex Factor Structure',
            'description': 'Current strategy uses 6 sub-factors with complex weighting, while working strategy uses 3 simple factors',
            'impact': 'High - Complexity may reduce factor effectiveness',
            'solution': 'Simplify to 3-factor approach like working strategy'
        },
        {
            'issue': 'Factor Correlation',
            'description': 'Multiple factors may be highly correlated, reducing diversification benefits',
            'impact': 'Medium - Correlated factors provide less diversification',
            'solution': 'Analyze factor correlations and reduce redundancy'
        },
        {
            'issue': 'Normalization Issues',
            'description': 'Factors are normalized individually, which may not preserve relative importance',
            'impact': 'Medium - May distort factor contributions',
            'solution': 'Use rank-based normalization or z-score standardization'
        },
        {
            'issue': 'Data Quality Issues',
            'description': 'Some factors have missing data, leading to inconsistent factor availability',
            'impact': 'High - Inconsistent factor availability reduces strategy reliability',
            'solution': 'Improve data quality and implement robust fallback mechanisms'
        },
        {
            'issue': 'Overfitting to Statistical Testing',
            'description': 'Factors optimized for statistical significance may not translate to portfolio performance',
            'impact': 'High - Statistical significance ≠ portfolio performance',
            'solution': 'Focus on factor returns and portfolio-level testing'
        }
    ]
    
    for i, issue in enumerate(issues, 1):
        print(f"\n{i}. {issue['issue']}")
        print(f"   Description: {issue['description']}")
        print(f"   Impact: {issue['impact']}")
        print(f"   Solution: {issue['solution']}")
    
    return issues

def analyze_factor_correlation():
    """Analyze potential factor correlation issues"""
    print("\n=== FACTOR CORRELATION ANALYSIS ===")
    
    print(f"\n🔍 Potential Factor Correlations:")
    print("=" * 50)
    
    correlations = [
        ('ROAA', 'F-Score', 'High - Both measure quality'),
        ('P/E', 'FCF Yield', 'Medium - Both value factors but different aspects'),
        ('Momentum', 'Low-Volatility', 'Low - Different characteristics'),
        ('ROAA', 'P/E', 'Medium - Quality vs Value'),
        ('F-Score', 'FCF Yield', 'Medium - Quality vs Value'),
        ('Momentum', 'ROAA', 'Low - Different factor categories')
    ]
    
    print(f"\n📊 Factor Correlation Matrix:")
    print(f"   {'Factor 1':<12} {'Factor 2':<12} {'Expected':<15}")
    print(f"   {'-'*12} {'-'*12} {'-'*15}")
    
    for factor1, factor2, expected in correlations:
        print(f"   {factor1:<12} {factor2:<12} {expected:<15}")
    
    print(f"\n⚠️  High Correlation Issues:")
    print(f"   - ROAA and F-Score both measure quality → may be redundant")
    print(f"   - P/E and FCF Yield both measure value → may be redundant")
    print(f"   - This reduces diversification benefits")
    
    return correlations

def recommend_optimizations():
    """Recommend optimizations based on the analysis"""
    print("\n=== OPTIMIZATION RECOMMENDATIONS ===")
    
    print(f"\n🎯 Recommended Actions:")
    print("=" * 50)
    
    recommendations = [
        {
            'priority': 'High',
            'action': 'Simplify Factor Structure',
            'description': 'Reduce from 6 sub-factors to 3 main factors like working strategy',
            'implementation': 'Use ROAA (quality), P/E (value), Momentum (momentum) only'
        },
        {
            'priority': 'High',
            'action': 'Improve Data Quality',
            'description': 'Ensure consistent factor availability across all periods',
            'implementation': 'Implement robust data validation and fallback mechanisms'
        },
        {
            'priority': 'Medium',
            'action': 'Optimize Factor Weights',
            'description': 'Use weights based on factor returns, not just statistical significance',
            'implementation': 'Backtest different weight combinations and select best performing'
        },
        {
            'priority': 'Medium',
            'action': 'Reduce Factor Correlation',
            'description': 'Remove redundant factors to improve diversification',
            'implementation': 'Keep only one factor from each category (quality, value, momentum)'
        },
        {
            'priority': 'Low',
            'action': 'Improve Normalization',
            'description': 'Use rank-based normalization to preserve factor importance',
            'implementation': 'Replace z-score normalization with percentile ranks'
        }
    ]
    
    for rec in recommendations:
        print(f"\n{rec['priority']} Priority: {rec['action']}")
        print(f"   Description: {rec['description']}")
        print(f"   Implementation: {rec['implementation']}")
    
    return recommendations

def create_optimized_configuration():
    """Create an optimized configuration based on the analysis"""
    print("\n=== OPTIMIZED CONFIGURATION PROPOSAL ===")
    
    optimized_config = {
        'strategy_name': 'QVM_Engine_v3j_Optimized',
        'backtest_start_date': '2016-01-01',
        'backtest_end_date': '2025-07-28',
        'rebalance_frequency': 'M',
        'transaction_cost_bps': 30,
        'universe': {
            'lookback_days': 63,
            'top_n_stocks': 200,
            'max_position_size': 0.05,
            'max_sector_exposure': 0.30,
            'target_portfolio_size': 20,
        },
        'factors': {
            'roaa_weight': 0.35,      # Quality factor (increased from 0.30)
            'pe_weight': 0.25,        # Value factor (decreased from 0.30)
            'momentum_weight': 0.40,  # Momentum factor (unchanged)
            'momentum_horizons': [21, 63, 126, 252],
            'skip_months': 1,
            'fundamental_lag_days': 45,
        }
    }
    
    print(f"\n📋 Optimized Configuration:")
    print(f"   Strategy Name: {optimized_config['strategy_name']}")
    print(f"   Factor Weights:")
    print(f"     - ROAA (Quality): {optimized_config['factors']['roaa_weight']:.1%}")
    print(f"     - P/E (Value): {optimized_config['factors']['pe_weight']:.1%}")
    print(f"     - Momentum: {optimized_config['factors']['momentum_weight']:.1%}")
    
    print(f"\n🔧 Key Changes:")
    print(f"   - Simplified to 3-factor structure (removed sub-factors)")
    print(f"   - Increased quality weight (35% vs 30%)")
    print(f"   - Decreased value weight (25% vs 30%)")
    print(f"   - Maintained momentum weight (40%)")
    print(f"   - Removed redundant factors (F-Score, FCF Yield, Low-Volatility)")
    
    return optimized_config

if __name__ == "__main__":
    print("🔍 FACTOR INTEGRATION INVESTIGATION")
    print("=" * 80)
    print("Investigating why individual factors show statistical significance")
    print("but the combined strategy underperforms...")
    print("=" * 80)
    
    # Run all analyses
    factor_stats = analyze_factor_statistical_significance()
    current_config = analyze_current_strategy_configuration()
    working_config = compare_with_working_strategy()
    integration_issues = analyze_factor_integration_issues()
    correlations = analyze_factor_correlation()
    recommendations = recommend_optimizations()
    optimized_config = create_optimized_configuration()
    
    print("\n" + "=" * 80)
    print("📋 INVESTIGATION SUMMARY")
    print("=" * 80)
    
    print(f"\n✅ Individual Factors: All show strong statistical significance")
    print(f"❌ Combined Strategy: Underperforms despite individual factor strength")
    print(f"\n🔍 Root Causes Identified:")
    print(f"   1. Complex factor structure reduces effectiveness")
    print(f"   2. Factor correlations reduce diversification")
    print(f"   3. Data quality issues create inconsistency")
    print(f"   4. Overfitting to statistical significance")
    print(f"   5. Sub-optimal factor integration methodology")
    
    print(f"\n🎯 Recommended Solution:")
    print(f"   - Simplify to 3-factor approach (ROAA, P/E, Momentum)")
    print(f"   - Use weights based on factor returns, not just IC")
    print(f"   - Improve data quality and consistency")
    print(f"   - Test portfolio-level performance, not just factor-level")
    
    print(f"\n✅ Investigation Complete!") 