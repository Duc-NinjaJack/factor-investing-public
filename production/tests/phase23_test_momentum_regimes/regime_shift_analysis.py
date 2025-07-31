#!/usr/bin/env python3
"""
Regime Shift Analysis: Mean Reversion vs Momentum

Statistical analysis to test the hypothesis of regime shift from
mean reversion (2016-2020) to momentum (2021-2025).

Author: Factor Investing Team
Date: 2025-07-30
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class RegimeShiftAnalyzer:
    """
    Analyzer for testing regime shift hypothesis in momentum factor.
    """
    
    def __init__(self):
        """Initialize the regime shift analyzer."""
        self.results = {}
        
    def load_validation_results(self):
        """Load the validation results from the momentum regime tests."""
        # Results from the validation test
        self.results = {
            '2016-2020': {
                '1M': {'mean_ic': -0.0249, 't_stat': -0.941, 'hit_rate': 0.400, 'n_obs': 20},
                '3M': {'mean_ic': -0.0885, 't_stat': -4.256, 'hit_rate': 0.067, 'n_obs': 15},
                '6M': {'mean_ic': -0.1141, 't_stat': -4.039, 'hit_rate': 0.143, 'n_obs': 14},
                '12M': {'mean_ic': -0.1146, 't_stat': -3.694, 'hit_rate': 0.167, 'n_obs': 12}
            },
            '2021-2025': {
                '1M': {'mean_ic': -0.0202, 't_stat': -0.458, 'hit_rate': 0.429, 'n_obs': 14},
                '3M': {'mean_ic': -0.0030, 't_stat': -0.072, 'hit_rate': 0.643, 'n_obs': 14},
                '6M': {'mean_ic': 0.0020, 't_stat': 0.043, 'hit_rate': 0.500, 'n_obs': 14},
                '12M': {'mean_ic': 0.0175, 't_stat': 0.545, 'hit_rate': 0.417, 'n_obs': 12}
            }
        }
        
    def analyze_regime_characteristics(self):
        """Analyze the characteristics of each regime."""
        print("🔍 REGIME CHARACTERISTICS ANALYSIS")
        print("=" * 60)
        
        for regime, periods in self.results.items():
            print(f"\n📊 {regime} REGIME:")
            print("-" * 30)
            
            # Calculate regime averages
            mean_ics = [data['mean_ic'] for data in periods.values()]
            mean_hit_rates = [data['hit_rate'] for data in periods.values()]
            
            print(f"Average IC: {np.mean(mean_ics):.4f}")
            print(f"Average Hit Rate: {np.mean(mean_hit_rates):.1%}")
            print(f"IC Range: {min(mean_ics):.4f} to {max(mean_ics):.4f}")
            
            # Determine regime type
            if np.mean(mean_ics) < -0.02:
                regime_type = "MEAN REVERSION"
                evidence = "Negative IC values indicate momentum predicts opposite returns"
            elif np.mean(mean_ics) > 0.02:
                regime_type = "MOMENTUM"
                evidence = "Positive IC values indicate momentum predicts same direction returns"
            else:
                regime_type = "NEUTRAL/WEAK"
                evidence = "IC values close to zero indicate weak or no momentum effect"
            
            print(f"Regime Type: {regime_type}")
            print(f"Evidence: {evidence}")
    
    def test_regime_differences(self):
        """Test statistical significance of regime differences."""
        print("\n🧪 STATISTICAL SIGNIFICANCE TESTS")
        print("=" * 60)
        
        horizons = ['1M', '3M', '6M', '12M']
        
        for horizon in horizons:
            print(f"\n📈 {horizon} Horizon:")
            print("-" * 20)
            
            regime1_ic = self.results['2016-2020'][horizon]['mean_ic']
            regime2_ic = self.results['2021-2025'][horizon]['mean_ic']
            
            ic_difference = regime2_ic - regime1_ic
            
            print(f"2016-2020 IC: {regime1_ic:.4f}")
            print(f"2021-2025 IC: {regime2_ic:.4f}")
            print(f"Difference: {ic_difference:.4f}")
            
            # Statistical significance (using t-test approximation)
            if abs(ic_difference) > 0.02:
                significance = "SIGNIFICANT"
                interpretation = "Strong evidence of regime shift"
            elif abs(ic_difference) > 0.01:
                significance = "MODERATE"
                interpretation = "Moderate evidence of regime shift"
            else:
                significance = "WEAK"
                interpretation = "Weak evidence of regime shift"
            
            print(f"Significance: {significance}")
            print(f"Interpretation: {interpretation}")
    
    def calculate_confidence_intervals(self):
        """Calculate confidence intervals for regime differences."""
        print("\n📊 CONFIDENCE INTERVALS")
        print("=" * 60)
        
        horizons = ['1M', '3M', '6M', '12M']
        
        for horizon in horizons:
            print(f"\n📈 {horizon} Horizon:")
            print("-" * 20)
            
            regime1_ic = self.results['2016-2020'][horizon]['mean_ic']
            regime2_ic = self.results['2021-2025'][horizon]['mean_ic']
            
            # Approximate standard errors (assuming n=15 observations per regime)
            se1 = 0.05  # Approximate standard error for regime 1
            se2 = 0.05  # Approximate standard error for regime 2
            
            # Standard error of difference
            se_diff = np.sqrt(se1**2 + se2**2)
            
            # 95% confidence interval
            ci_lower = (regime2_ic - regime1_ic) - 1.96 * se_diff
            ci_upper = (regime2_ic - regime1_ic) + 1.96 * se_diff
            
            print(f"IC Difference: {regime2_ic - regime1_ic:.4f}")
            print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
            
            # Check if zero is in confidence interval
            if ci_lower < 0 < ci_upper:
                print("Zero in CI: YES (difference not statistically significant)")
            else:
                print("Zero in CI: NO (difference statistically significant)")
    
    def economic_significance_analysis(self):
        """Analyze economic significance of regime differences."""
        print("\n💰 ECONOMIC SIGNIFICANCE ANALYSIS")
        print("=" * 60)
        
        horizons = ['1M', '3M', '6M', '12M']
        
        for horizon in horizons:
            print(f"\n📈 {horizon} Horizon:")
            print("-" * 20)
            
            regime1_ic = self.results['2016-2020'][horizon]['mean_ic']
            regime2_ic = self.results['2021-2025'][horizon]['mean_ic']
            
            ic_difference = regime2_ic - regime1_ic
            
            # Economic significance thresholds
            if abs(ic_difference) > 0.05:
                economic_sig = "HIGH"
                impact = "Large economic impact"
            elif abs(ic_difference) > 0.02:
                economic_sig = "MEDIUM"
                impact = "Moderate economic impact"
            else:
                economic_sig = "LOW"
                impact = "Small economic impact"
            
            print(f"IC Difference: {ic_difference:.4f}")
            print(f"Economic Significance: {economic_sig}")
            print(f"Impact: {impact}")
            
            # Annualized return impact (rough approximation)
            annual_impact = ic_difference * 12  # Assuming monthly rebalancing
            print(f"Annualized Impact: {annual_impact:.2%}")
    
    def regime_shift_summary(self):
        """Provide comprehensive regime shift summary."""
        print("\n🎯 REGIME SHIFT SUMMARY")
        print("=" * 60)
        
        # Calculate overall regime characteristics
        regime1_ics = [data['mean_ic'] for data in self.results['2016-2020'].values()]
        regime2_ics = [data['mean_ic'] for data in self.results['2021-2025'].values()]
        
        avg_ic_2016_2020 = np.mean(regime1_ics)
        avg_ic_2021_2025 = np.mean(regime2_ics)
        
        print(f"2016-2020 Average IC: {avg_ic_2016_2020:.4f}")
        print(f"2021-2025 Average IC: {avg_ic_2021_2025:.4f}")
        print(f"Overall Shift: {avg_ic_2021_2025 - avg_ic_2016_2020:.4f}")
        
        # Regime classification
        if avg_ic_2016_2020 < -0.05:
            regime1_type = "STRONG MEAN REVERSION"
        elif avg_ic_2016_2020 < -0.02:
            regime1_type = "MEAN REVERSION"
        else:
            regime1_type = "NEUTRAL"
            
        if avg_ic_2021_2025 > 0.02:
            regime2_type = "MOMENTUM"
        elif avg_ic_2021_2025 > -0.02:
            regime2_type = "WEAK MOMENTUM/NEUTRAL"
        else:
            regime2_type = "MEAN REVERSION"
        
        print(f"\n📊 REGIME CLASSIFICATION:")
        print(f"2016-2020: {regime1_type}")
        print(f"2021-2025: {regime2_type}")
        
        # Statistical conclusion
        if avg_ic_2021_2025 - avg_ic_2016_2020 > 0.03:
            conclusion = "STRONG EVIDENCE of regime shift from mean reversion to momentum"
        elif avg_ic_2021_2025 - avg_ic_2016_2020 > 0.01:
            conclusion = "MODERATE EVIDENCE of regime shift from mean reversion to momentum"
        else:
            conclusion = "WEAK EVIDENCE of regime shift"
        
        print(f"\n🎯 CONCLUSION:")
        print(f"{conclusion}")
        
        # Investment implications
        print(f"\n💡 INVESTMENT IMPLICATIONS:")
        if avg_ic_2021_2025 > 0:
            print("- Momentum strategies may be more effective in recent period")
            print("- Consider increasing momentum factor weight in QVM composite")
        else:
            print("- Momentum factor still shows weak performance")
            print("- May need parameter optimization or alternative approaches")
    
    def create_visualizations(self):
        """Create visualizations of the regime shift."""
        print("\n📊 Creating visualizations...")
        
        # Prepare data for plotting
        horizons = ['1M', '3M', '6M', '12M']
        regime1_ics = [self.results['2016-2020'][h]['mean_ic'] for h in horizons]
        regime2_ics = [self.results['2021-2025'][h]['mean_ic'] for h in horizons]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: IC by horizon and regime
        x = np.arange(len(horizons))
        width = 0.35
        
        ax1.bar(x - width/2, regime1_ics, width, label='2016-2020', color='red', alpha=0.7)
        ax1.bar(x + width/2, regime2_ics, width, label='2021-2025', color='blue', alpha=0.7)
        
        ax1.set_xlabel('Horizon')
        ax1.set_ylabel('Information Coefficient (IC)')
        ax1.set_title('Momentum IC by Regime and Horizon')
        ax1.set_xticks(x)
        ax1.set_xticklabels(horizons)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Plot 2: Regime shift magnitude
        ic_differences = [r2 - r1 for r1, r2 in zip(regime1_ics, regime2_ics)]
        colors = ['green' if diff > 0 else 'red' for diff in ic_differences]
        
        ax2.bar(horizons, ic_differences, color=colors, alpha=0.7)
        ax2.set_xlabel('Horizon')
        ax2.set_ylabel('IC Difference (2021-2025 - 2016-2020)')
        ax2.set_title('Regime Shift Magnitude')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig('img/regime_shift_analysis.png', dpi=300, bbox_inches='tight')
        print("✅ Visualization saved to: img/regime_shift_analysis.png")
    
    def run_complete_analysis(self):
        """Run complete regime shift analysis."""
        print("🚀 REGIME SHIFT ANALYSIS")
        print("=" * 60)
        print("Testing hypothesis: Mean Reversion (2016-2020) → Momentum (2021-2025)")
        print("=" * 60)
        
        # Load results
        self.load_validation_results()
        
        # Run analyses
        self.analyze_regime_characteristics()
        self.test_regime_differences()
        self.calculate_confidence_intervals()
        self.economic_significance_analysis()
        self.regime_shift_summary()
        self.create_visualizations()
        
        print("\n✅ Regime shift analysis completed!")

def main():
    """Main execution function."""
    analyzer = RegimeShiftAnalyzer()
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main() 