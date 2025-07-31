#!/usr/bin/env python3
"""
Comprehensive Momentum Regime Analysis

This script performs a comprehensive analysis of momentum factor IC across
different regimes, including parameter optimization, weight optimization,
and transaction cost analysis.

Author: Factor Investing Team
Date: 2025-07-30
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add paths for imports
sys.path.append('../../../production')

try:
    from database.connection import get_engine
    from engine.qvm_engine_v2_enhanced import QVMEngineV2Enhanced
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

class ComprehensiveMomentumAnalyzer:
    """
    Comprehensive analyzer for momentum factor regime analysis.
    """
    
    def __init__(self):
        """Initialize the comprehensive analyzer."""
        self.results = {}
        self.engine = None
        
    def initialize_engine(self):
        """Initialize the QVM engine."""
        try:
            self.engine = QVMEngineV2Enhanced()
            print("✅ QVM Engine initialized")
        except Exception as e:
            print(f"❌ Failed to initialize engine: {e}")
            raise
    
    def run_comprehensive_analysis(self):
        """Run all analyses."""
        print("🚀 COMPREHENSIVE MOMENTUM REGIME ANALYSIS")
        print("=" * 80)
        
        # 1. Basic regime validation
        print("\n📊 1. Basic Regime Validation")
        print("-" * 40)
        regime_results = self.run_regime_validation()
        
        # 2. Parameter optimization
        print("\n📊 2. Parameter Optimization")
        print("-" * 40)
        param_results = self.run_parameter_optimization()
        
        # 3. Weight optimization
        print("\n📊 3. Weight Optimization")
        print("-" * 40)
        weight_results = self.run_weight_optimization()
        
        # 4. Transaction cost analysis
        print("\n📊 4. Transaction Cost Analysis")
        print("-" * 40)
        cost_results = self.run_transaction_cost_analysis()
        
        # 5. Comprehensive summary
        print("\n📊 5. Comprehensive Summary")
        print("-" * 40)
        self.generate_comprehensive_summary(regime_results, param_results, weight_results, cost_results)
        
        return {
            'regime_validation': regime_results,
            'parameter_optimization': param_results,
            'weight_optimization': weight_results,
            'transaction_cost_analysis': cost_results
        }
    
    def run_regime_validation(self):
        """Run basic regime validation tests."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "momentum_regime_validation", 
            "01_momentum_regime_validation_tests.py"
        )
        momentum_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(momentum_module)
        
        try:
            analyzer = momentum_module.MomentumICAnalysis()
            results = analyzer.run_complete_analysis()
            return results
        except Exception as e:
            print(f"❌ Regime validation failed: {e}")
            return None
    
    def run_parameter_optimization(self):
        """Run parameter optimization analysis."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "momentum_parameter_optimization", 
            "02_momentum_parameter_optimization.py"
        )
        param_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(param_module)
        
        try:
            optimizer = param_module.MomentumParameterOptimizer()
            optimizer.initialize_engine()
            results = optimizer.optimize_parameters()
            return results
        except Exception as e:
            print(f"❌ Parameter optimization failed: {e}")
            return None
    
    def run_weight_optimization(self):
        """Run weight optimization analysis."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "momentum_weight_optimization", 
            "03_momentum_weight_optimization.py"
        )
        weight_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(weight_module)
        
        try:
            optimizer = weight_module.MomentumWeightOptimizer()
            optimizer.initialize_engine()
            results = optimizer.optimize_weights()
            return results
        except Exception as e:
            print(f"❌ Weight optimization failed: {e}")
            return None
    
    def run_transaction_cost_analysis(self):
        """Run transaction cost analysis."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "momentum_transaction_cost_analysis", 
            "04_momentum_transaction_cost_analysis.py"
        )
        cost_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cost_module)
        
        try:
            analyzer = cost_module.MomentumTransactionCostAnalyzer()
            analyzer.initialize_engine()
            results = analyzer.test_rebalancing_frequency()
            return results
        except Exception as e:
            print(f"❌ Transaction cost analysis failed: {e}")
            return None
    
    def generate_comprehensive_summary(self, regime_results, param_results, weight_results, cost_results):
        """Generate comprehensive summary of all analyses."""
        print("\n📋 COMPREHENSIVE SUMMARY")
        print("=" * 60)
        
        # Regime validation summary
        if regime_results:
            print("\n🎯 REGIME VALIDATION SUMMARY:")
            for period_name in ['2016-2020', '2021-2025']:
                if period_name in regime_results:
                    period_data = regime_results[period_name]
                    summary_stats = period_data.get('summary_stats', {})
                    
                    print(f"  {period_name}:")
                    for horizon in ['1M', '3M', '6M', '12M']:
                        if horizon in summary_stats:
                            stats = summary_stats[horizon]
                            if stats['n_observations'] > 0:
                                print(f"    {horizon}: IC={stats['mean_ic']:.4f}, T-stat={stats['t_stat']:.3f}, Hit Rate={stats['hit_rate']:.1%}")
        
        # Parameter optimization summary
        if param_results and hasattr(param_results, 'get'):
            print("\n🎯 PARAMETER OPTIMIZATION SUMMARY:")
            if 'optimal_params' in param_results:
                for metric, (params, stats) in param_results['optimal_params'].items():
                    print(f"  {metric.replace('_', ' ').title()}: {params}")
                    print(f"    IC: {stats['mean_ic']:.4f}, T-stat: {stats['t_stat']:.3f}")
        
        # Weight optimization summary
        if weight_results and hasattr(weight_results, 'get'):
            print("\n🎯 WEIGHT OPTIMIZATION SUMMARY:")
            if 'optimal_weights' in weight_results:
                for metric, (weights, stats) in weight_results['optimal_weights'].items():
                    print(f"  {metric.replace('_', ' ').title()}: {weights}")
                    print(f"    IC: {stats['mean_ic']:.4f}, T-stat: {stats['t_stat']:.3f}")
        
        # Transaction cost summary
        if cost_results:
            print("\n🎯 TRANSACTION COST SUMMARY:")
            # Find best frequency for typical cost (15 bps)
            best_freq = None
            best_ic = -1
            
            for freq_name, freq_results in cost_results.items():
                if 0.0015 in freq_results:  # 15 bps
                    ic = freq_results[0.0015]['mean_ic']
                    if ic > best_ic:
                        best_ic = ic
                        best_freq = freq_name
            
            if best_freq:
                print(f"  Optimal frequency for 15 bps: {best_freq} (IC: {best_ic:.4f})")
    
    def generate_comprehensive_report(self, all_results):
        """Generate comprehensive report."""
        report_lines = []
        report_lines.append("=" * 100)
        report_lines.append("COMPREHENSIVE MOMENTUM REGIME ANALYSIS REPORT")
        report_lines.append("=" * 100)
        report_lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Executive summary
        report_lines.append("EXECUTIVE SUMMARY")
        report_lines.append("-" * 30)
        report_lines.append("This report presents a comprehensive analysis of momentum factor")
        report_lines.append("Information Coefficient (IC) across different market regimes.")
        report_lines.append("")
        
        # Key findings
        report_lines.append("KEY FINDINGS:")
        report_lines.append("-" * 15)
        
        # Add findings based on results
        if all_results.get('regime_validation'):
            report_lines.append("• Regime validation shows differences between pre- and post-COVID periods")
        
        if all_results.get('parameter_optimization'):
            report_lines.append("• Parameter optimization identifies optimal lookback periods and weights")
        
        if all_results.get('weight_optimization'):
            report_lines.append("• Weight optimization reveals optimal momentum factor composition")
        
        if all_results.get('transaction_cost_analysis'):
            report_lines.append("• Transaction cost analysis determines optimal rebalancing frequency")
        
        report_lines.append("")
        
        # Detailed results sections
        for analysis_name, results in all_results.items():
            if results:
                report_lines.append(f"{analysis_name.upper().replace('_', ' ')}")
                report_lines.append("-" * 50)
                report_lines.append("Results available - see individual reports for details.")
                report_lines.append("")
        
        # Recommendations
        report_lines.append("RECOMMENDATIONS")
        report_lines.append("-" * 20)
        report_lines.append("1. Use regime-aware momentum factor implementation")
        report_lines.append("2. Apply optimal parameters identified in optimization")
        report_lines.append("3. Consider transaction costs in rebalancing decisions")
        report_lines.append("4. Monitor regime changes and adjust accordingly")
        report_lines.append("")
        
        return "\n".join(report_lines)

def main():
    """Main execution function."""
    print("🚀 COMPREHENSIVE MOMENTUM REGIME ANALYSIS")
    print("=" * 80)
    
    try:
        analyzer = ComprehensiveMomentumAnalyzer()
        analyzer.initialize_engine()
        
        # Run comprehensive analysis
        all_results = analyzer.run_comprehensive_analysis()
        
        # Generate comprehensive report
        report = analyzer.generate_comprehensive_report(all_results)
        print("\n" + report)
        
        # Save comprehensive report
        with open('data/comprehensive_momentum_regime_report.txt', 'w') as f:
            f.write(report)
        
        print("\n✅ Comprehensive analysis completed successfully!")
        print("📄 Report saved to: data/comprehensive_momentum_regime_report.txt")
        
        return all_results
        
    except Exception as e:
        print(f"❌ Comprehensive analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main() 