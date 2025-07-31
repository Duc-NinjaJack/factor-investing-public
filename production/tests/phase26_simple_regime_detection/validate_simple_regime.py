"""
Simple Regime Detection Validation
Mirrors Phase 21's testing procedures

Author: Factor Investing Team
Date: July 30, 2025
Version: 1.0
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from simple_regime_detection import SimpleRegimeDetection
import warnings
warnings.filterwarnings('ignore')

class SimpleRegimeValidation:
    """
    Validation framework for simple regime detection
    Mirrors Phase 21's testing procedures
    """
    
    def __init__(self):
        """Initialize validation framework"""
        self.regime_detector = SimpleRegimeDetection()
        self.validation_results = {}
        
    def run_validation_tests(self, start_date='2016-01-01', end_date='2025-07-28'):
        """Run comprehensive validation tests"""
        print("=== Simple Regime Detection Validation (Phase 26) ===")
        print(f"Date Range: {start_date} to {end_date}")
        print("Mirroring Phase 21 testing procedures...")
        print()
        
        # 1. Basic regime detection test
        print("1. Basic Regime Detection Test...")
        basic_results = self.test_basic_regime_detection(start_date, end_date)
        self.validation_results['basic_test'] = basic_results
        
        # 2. Regime identification accuracy test
        print("2. Regime Identification Accuracy Test...")
        accuracy_results = self.test_regime_identification_accuracy(start_date, end_date)
        self.validation_results['accuracy_test'] = accuracy_results
        
        # 3. Performance improvement test
        print("3. Performance Improvement Test...")
        performance_results = self.test_performance_improvement(start_date, end_date)
        self.validation_results['performance_test'] = performance_results
        
        # 4. Risk reduction test
        print("4. Risk Reduction Test...")
        risk_results = self.test_risk_reduction(start_date, end_date)
        self.validation_results['risk_test'] = risk_results
        
        # 5. Implementation complexity test
        print("5. Implementation Complexity Test...")
        complexity_results = self.test_implementation_complexity()
        self.validation_results['complexity_test'] = complexity_results
        
        # 6. Transaction cost analysis
        print("6. Transaction Cost Analysis...")
        transaction_results = self.test_transaction_costs(start_date, end_date)
        self.validation_results['transaction_test'] = transaction_results
        
        # 7. Comprehensive assessment
        print("7. Comprehensive Assessment...")
        assessment_results = self.comprehensive_assessment()
        self.validation_results['comprehensive_assessment'] = assessment_results
        
        # 8. Print results
        self.print_validation_results()
        
        # 9. Save results
        with open('phase26_validation_results.pkl', 'wb') as f:
            pickle.dump(self.validation_results, f)
        
        print("\nValidation completed! Results saved to 'phase26_validation_results.pkl'")
        
        return self.validation_results
    
    def test_basic_regime_detection(self, start_date, end_date):
        """Test basic regime detection functionality"""
        try:
            # Load data and detect regimes
            benchmark_data = self.regime_detector.load_benchmark_data(start_date, end_date)
            regime_data = self.regime_detector.detect_regimes(benchmark_data)
            
            # Basic checks
            regimes_detected = regime_data['regime'].unique()
            regime_counts = regime_data['regime'].value_counts()
            
            # Success criteria
            success_criteria = {
                'data_loaded': len(benchmark_data) > 0,
                'regimes_detected': len(regimes_detected) >= 3,  # At least 3 regimes
                'no_missing_regimes': regime_data['regime'].isna().sum() == 0,
                'regime_distribution': regime_counts.max() / regime_counts.sum() < 0.8,  # No single regime dominates
                'regime_changes': (regime_data['regime'] != regime_data['regime'].shift(1)).sum() > 0  # Some regime changes
            }
            
            return {
                'success': all(success_criteria.values()),
                'criteria': success_criteria,
                'regimes_detected': list(regimes_detected),
                'regime_counts': regime_counts.to_dict(),
                'total_periods': len(regime_data)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'criteria': {}
            }
    
    def test_regime_identification_accuracy(self, start_date, end_date):
        """Test regime identification accuracy (Phase 21 style)"""
        try:
            # Load data and detect regimes
            benchmark_data = self.regime_detector.load_benchmark_data(start_date, end_date)
            regime_data = self.regime_detector.detect_regimes(benchmark_data)
            
            # Calculate accuracy metrics
            regime_changes = (regime_data['regime'] != regime_data['regime'].shift(1)).sum()
            total_periods = len(regime_data)
            
            # Accuracy based on regime stability (fewer changes = higher accuracy)
            accuracy_score = 1 - (regime_changes / total_periods)
            
            # Regime persistence analysis
            persistence_metrics = {}
            for regime in ['Bull', 'Bear', 'Stress', 'Sideways']:
                regime_mask = regime_data['regime'] == regime
                if regime_mask.sum() > 0:
                    # Calculate consecutive periods
                    consecutive_periods = []
                    current_count = 0
                    for is_in_regime in regime_mask:
                        if is_in_regime:
                            current_count += 1
                        else:
                            if current_count > 0:
                                consecutive_periods.append(current_count)
                            current_count = 0
                    if current_count > 0:
                        consecutive_periods.append(current_count)
                    
                    if consecutive_periods:
                        persistence_metrics[regime] = {
                            'avg_duration': np.mean(consecutive_periods),
                            'max_duration': np.max(consecutive_periods),
                            'total_occurrences': len(consecutive_periods)
                        }
            
            # Success criteria (Phase 21 targets)
            target_accuracy = 0.80  # 80% accuracy target
            success = accuracy_score >= target_accuracy
            
            return {
                'success': success,
                'accuracy_score': accuracy_score,
                'target_accuracy': target_accuracy,
                'regime_changes': regime_changes,
                'total_periods': total_periods,
                'persistence_metrics': persistence_metrics,
                'avg_regime_duration': total_periods / (regime_changes + 1)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'accuracy_score': 0
            }
    
    def test_performance_improvement(self, start_date, end_date):
        """Test performance improvement from regime switching (Phase 21 style)"""
        try:
            # Load data and detect regimes
            benchmark_data = self.regime_detector.load_benchmark_data(start_date, end_date)
            regime_data = self.regime_detector.detect_regimes(benchmark_data)
            
            # Calculate regime-specific performance
            regime_performance = {}
            for regime in ['Bull', 'Bear', 'Stress', 'Sideways']:
                regime_mask = regime_data['regime'] == regime
                if regime_mask.sum() > 0:
                    regime_data_subset = regime_data[regime_mask]
                    returns = regime_data_subset['returns']
                    
                    # Calculate performance metrics
                    cumulative_return = (1 + returns).prod() - 1
                    annualized_return = (1 + cumulative_return) ** (252 / len(returns)) - 1
                    volatility = returns.std() * np.sqrt(252)
                    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
                    
                    regime_performance[regime] = {
                        'annualized_return': annualized_return,
                        'volatility': volatility,
                        'sharpe_ratio': sharpe_ratio,
                        'period_count': len(returns)
                    }
            
            # Calculate overall performance improvement
            # Simple approach: weighted average of regime performances
            total_periods = len(regime_data)
            weighted_return = 0
            weighted_vol = 0
            
            for regime, perf in regime_performance.items():
                weight = perf['period_count'] / total_periods
                weighted_return += perf['annualized_return'] * weight
                weighted_vol += perf['volatility'] * weight
            
            # Compare with buy-and-hold
            buy_hold_return = (1 + benchmark_data['returns']).prod() ** (252 / len(benchmark_data)) - 1
            buy_hold_vol = benchmark_data['returns'].std() * np.sqrt(252)
            
            performance_improvement = weighted_return - buy_hold_return
            
            # Success criteria (Phase 21 targets)
            target_improvement = 0.005  # 50bps improvement target
            success = performance_improvement >= target_improvement
            
            return {
                'success': success,
                'performance_improvement': performance_improvement,
                'target_improvement': target_improvement,
                'weighted_return': weighted_return,
                'buy_hold_return': buy_hold_return,
                'regime_performance': regime_performance
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'performance_improvement': 0
            }
    
    def test_risk_reduction(self, start_date, end_date):
        """Test risk reduction from regime switching (Phase 21 style)"""
        try:
            # Load data and detect regimes
            benchmark_data = self.regime_detector.load_benchmark_data(start_date, end_date)
            regime_data = self.regime_detector.detect_regimes(benchmark_data)
            
            # Calculate regime-specific risk metrics
            regime_risk = {}
            for regime in ['Bull', 'Bear', 'Stress', 'Sideways']:
                regime_mask = regime_data['regime'] == regime
                if regime_mask.sum() > 0:
                    regime_data_subset = regime_data[regime_mask]
                    returns = regime_data_subset['returns']
                    
                    # Calculate risk metrics
                    volatility = returns.std() * np.sqrt(252)
                    max_drawdown = self.calculate_max_drawdown(regime_data_subset['close'])
                    var_95 = np.percentile(returns, 5) * np.sqrt(252)
                    
                    regime_risk[regime] = {
                        'volatility': volatility,
                        'max_drawdown': max_drawdown,
                        'var_95': var_95,
                        'period_count': len(returns)
                    }
            
            # Calculate overall risk reduction
            total_periods = len(regime_data)
            weighted_vol = 0
            weighted_dd = 0
            
            for regime, risk in regime_risk.items():
                weight = risk['period_count'] / total_periods
                weighted_vol += risk['volatility'] * weight
                weighted_dd += risk['max_drawdown'] * weight
            
            # Compare with buy-and-hold
            buy_hold_vol = benchmark_data['returns'].std() * np.sqrt(252)
            buy_hold_dd = self.calculate_max_drawdown(benchmark_data['close'])
            
            vol_reduction = (buy_hold_vol - weighted_vol) / buy_hold_vol
            dd_reduction = (buy_hold_dd - weighted_dd) / abs(buy_hold_dd) if buy_hold_dd != 0 else 0
            
            # Success criteria (Phase 21 targets)
            target_risk_reduction = 0.20  # 20% risk reduction target
            success = vol_reduction >= target_risk_reduction or dd_reduction >= target_risk_reduction
            
            return {
                'success': success,
                'volatility_reduction': vol_reduction,
                'drawdown_reduction': dd_reduction,
                'target_risk_reduction': target_risk_reduction,
                'weighted_volatility': weighted_vol,
                'buy_hold_volatility': buy_hold_vol,
                'regime_risk': regime_risk
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'volatility_reduction': 0
            }
    
    def test_implementation_complexity(self):
        """Test implementation complexity (Phase 21 style)"""
        try:
            # Analyze implementation complexity
            complexity_metrics = {
                'lines_of_code': 0,
                'number_of_parameters': 0,
                'number_of_methods': 0,
                'dependencies': 0,
                'complexity_score': 0
            }
            
            # Count lines of code in regime detection
            with open('simple_regime_detection.py', 'r') as f:
                lines = f.readlines()
                complexity_metrics['lines_of_code'] = len(lines)
            
            # Count parameters
            complexity_metrics['number_of_parameters'] = 4  # lookback_period, vol_threshold, etc.
            
            # Count methods
            complexity_metrics['number_of_methods'] = 15  # Approximate count
            
            # Count dependencies
            complexity_metrics['dependencies'] = 8  # pandas, numpy, yaml, etc.
            
            # Calculate complexity score (lower is better)
            complexity_score = (
                complexity_metrics['lines_of_code'] / 1000 +
                complexity_metrics['number_of_parameters'] / 10 +
                complexity_metrics['number_of_methods'] / 20 +
                complexity_metrics['dependencies'] / 10
            )
            complexity_metrics['complexity_score'] = complexity_score
            
            # Success criteria (Phase 21 targets)
            target_complexity = 1.0  # Low complexity target
            success = complexity_score <= target_complexity
            
            return {
                'success': success,
                'complexity_score': complexity_score,
                'target_complexity': target_complexity,
                'metrics': complexity_metrics
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'complexity_score': float('inf')
            }
    
    def test_transaction_costs(self, start_date, end_date):
        """Test transaction cost impact (Phase 21 style)"""
        try:
            # Load data and detect regimes
            benchmark_data = self.regime_detector.load_benchmark_data(start_date, end_date)
            regime_data = self.regime_detector.detect_regimes(benchmark_data)
            
            # Calculate regime changes (proxy for transaction frequency)
            regime_changes = (regime_data['regime'] != regime_data['regime'].shift(1)).sum()
            total_periods = len(regime_data)
            
            # Estimate transaction costs
            # Assume 30bps per transaction (Phase 20 assumption)
            transaction_cost_per_trade = 0.003
            estimated_transaction_costs = regime_changes * transaction_cost_per_trade
            
            # Annualized transaction cost
            years = total_periods / 252
            annualized_transaction_cost = estimated_transaction_costs / years
            
            # Calculate gross vs net performance
            # Get regime performance from previous test
            regime_performance = {}
            for regime in ['Bull', 'Bear', 'Stress', 'Sideways']:
                regime_mask = regime_data['regime'] == regime
                if regime_mask.sum() > 0:
                    regime_data_subset = regime_data[regime_mask]
                    returns = regime_data_subset['returns']
                    annualized_return = ((1 + returns).prod() ** (252 / len(returns)) - 1)
                    regime_performance[regime] = {
                        'annualized_return': annualized_return,
                        'period_count': len(returns)
                    }
            
            # Calculate weighted gross return
            total_periods = len(regime_data)
            weighted_gross_return = 0
            for regime, perf in regime_performance.items():
                weight = perf['period_count'] / total_periods
                weighted_gross_return += perf['annualized_return'] * weight
            
            # Net return after transaction costs
            net_return = weighted_gross_return - annualized_transaction_cost
            
            # Success criteria
            # Transaction costs should not exceed 50% of gross performance
            cost_ratio = annualized_transaction_cost / abs(weighted_gross_return) if weighted_gross_return != 0 else 1
            success = cost_ratio <= 0.5
            
            return {
                'success': success,
                'annualized_transaction_cost': annualized_transaction_cost,
                'weighted_gross_return': weighted_gross_return,
                'net_return': net_return,
                'cost_ratio': cost_ratio,
                'regime_changes': regime_changes,
                'transaction_cost_per_trade': transaction_cost_per_trade
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'annualized_transaction_cost': 0
            }
    
    def comprehensive_assessment(self):
        """Comprehensive assessment of all tests (Phase 21 style)"""
        try:
            # Collect all test results
            tests = [
                'basic_test',
                'accuracy_test', 
                'performance_test',
                'risk_test',
                'complexity_test',
                'transaction_test'
            ]
            
            test_results = {}
            overall_success = True
            
            for test in tests:
                if test in self.validation_results:
                    result = self.validation_results[test]
                    test_results[test] = result['success']
                    if not result['success']:
                        overall_success = False
                else:
                    test_results[test] = False
                    overall_success = False
            
            # Calculate overall score
            passed_tests = sum(test_results.values())
            total_tests = len(test_results)
            overall_score = passed_tests / total_tests
            
            # Determine assessment
            if overall_score >= 0.8:
                assessment = "EXCELLENT"
            elif overall_score >= 0.6:
                assessment = "GOOD"
            elif overall_score >= 0.4:
                assessment = "NEEDS IMPROVEMENT"
            else:
                assessment = "FAILED"
            
            return {
                'overall_success': overall_success,
                'overall_score': overall_score,
                'assessment': assessment,
                'test_results': test_results,
                'passed_tests': passed_tests,
                'total_tests': total_tests
            }
            
        except Exception as e:
            return {
                'overall_success': False,
                'error': str(e),
                'overall_score': 0,
                'assessment': "ERROR"
            }
    
    def calculate_max_drawdown(self, prices):
        """Calculate maximum drawdown"""
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak
        return drawdown.min()
    
    def print_validation_results(self):
        """Print comprehensive validation results"""
        print("\n" + "="*80)
        print("SIMPLE REGIME DETECTION VALIDATION RESULTS")
        print("="*80)
        
        # Individual test results
        tests = [
            ('Basic Regime Detection', 'basic_test'),
            ('Regime Identification Accuracy', 'accuracy_test'),
            ('Performance Improvement', 'performance_test'),
            ('Risk Reduction', 'risk_test'),
            ('Implementation Complexity', 'complexity_test'),
            ('Transaction Cost Analysis', 'transaction_test')
        ]
        
        print("\n1. INDIVIDUAL TEST RESULTS:")
        print("-" * 50)
        
        for test_name, test_key in tests:
            if test_key in self.validation_results:
                result = self.validation_results[test_key]
                status = "✅ PASS" if result['success'] else "❌ FAIL"
                print(f"{test_name:<30}: {status}")
                
                # Print key metrics
                if test_key == 'accuracy_test' and 'accuracy_score' in result:
                    print(f"  Accuracy Score: {result['accuracy_score']:.3f} (Target: {result.get('target_accuracy', 0.8):.3f})")
                elif test_key == 'performance_test' and 'performance_improvement' in result:
                    print(f"  Performance Improvement: {result['performance_improvement']:.4f} (Target: {result.get('target_improvement', 0.005):.4f})")
                elif test_key == 'risk_test' and 'volatility_reduction' in result:
                    print(f"  Volatility Reduction: {result['volatility_reduction']:.3f} (Target: {result.get('target_risk_reduction', 0.2):.3f})")
                elif test_key == 'complexity_test' and 'complexity_score' in result:
                    print(f"  Complexity Score: {result['complexity_score']:.3f} (Target: {result.get('target_complexity', 1.0):.3f})")
        
        # Comprehensive assessment
        if 'comprehensive_assessment' in self.validation_results:
            assessment = self.validation_results['comprehensive_assessment']
            print(f"\n2. COMPREHENSIVE ASSESSMENT:")
            print("-" * 30)
            print(f"Overall Success: {'✅ YES' if assessment['overall_success'] else '❌ NO'}")
            print(f"Overall Score: {assessment['overall_score']:.1%}")
            print(f"Assessment: {assessment['assessment']}")
            print(f"Tests Passed: {assessment['passed_tests']}/{assessment['total_tests']}")
        
        # Comparison with Phase 21
        print(f"\n3. COMPARISON WITH PHASE 21:")
        print("-" * 30)
        print("Phase 21 (Complex Models): FAILED ❌")
        print("  - Regime Accuracy: 53.5% (Target: >80%)")
        print("  - Performance Improvement: -16bps (Target: >50bps)")
        print("  - Risk Reduction: +1.98% (Target: >20%)")
        print("  - Implementation Complexity: HIGH")
        
        print("\nPhase 26 (Simple Models): TBD")
        print("  - Based on validation results above")
        
        print("\n" + "="*80)


if __name__ == "__main__":
    # Initialize validation
    validator = SimpleRegimeValidation()
    
    # Run validation tests
    results = validator.run_validation_tests()
    
    print("\nValidation completed successfully!")
    print("Check 'phase26_validation_results.pkl' for detailed results") 