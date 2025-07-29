#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5B VND Quick Validation
=======================
Component: Alternative Threshold Investigation
Purpose: Quick validation of 5B VND liquidity threshold
Author: Duc Nguyen, Principal Quantitative Strategist
Date Created: January 2025
Status: ALTERNATIVE THRESHOLD INVESTIGATION

This script performs quick validation of 5B VND threshold:
- Compares universe size vs 10B VND baseline
- Analyzes QVM score impact
- Evaluates average ADTV characteristics
- Provides initial assessment for further investigation

Based on 3B VND analysis findings:
- 3B VND showed -10.77% return decline
- Real data validation is critical
- Need conservative approach for alternatives
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FiveBVNDQuickValidator:
    """
    Quick validation for 5B VND liquidity threshold.
    """
    
    def __init__(self):
        """Initialize the validator."""
        self.thresholds = {
            '10B_VND': 10_000_000_000,
            '5B_VND': 5_000_000_000
        }
        
        logger.info("5B VND Quick Validator initialized")
    
    def load_data(self):
        """Load data for analysis."""
        logger.info("Loading data for 5B VND quick validation...")
        
        # Load unrestricted universe data
        with open('unrestricted_universe_data.pkl', 'rb') as f:
            data = pickle.load(f)
        
        factor_scores = data['factor_scores']
        adtv_data = data['adtv']
        
        logger.info(f"✅ Data loaded successfully")
        logger.info(f"   - Factor scores: {factor_scores.shape}")
        logger.info(f"   - ADTV data: {adtv_data.shape}")
        
        return factor_scores, adtv_data
    
    def analyze_universe_expansion(self, factor_scores, adtv_data):
        """Analyze universe expansion for 5B VND threshold."""
        logger.info("Analyzing universe expansion for 5B VND threshold...")
        
        # Get most recent data
        recent_date = factor_scores['calculation_date'].max()
        recent_factors = factor_scores[factor_scores['calculation_date'] == recent_date]
        
        # Get ADTV for recent date
        recent_adtv = adtv_data.loc[recent_date]
        
        # Merge data
        merged_data = recent_factors.merge(
            recent_adtv.reset_index().rename(columns={0: 'adtv'}),
            left_on='ticker', right_on='ticker', how='inner'
        )
        
        # The ADTV column is named with the timestamp, so we need to rename it
        adtv_column = recent_date
        merged_data = merged_data.rename(columns={adtv_column: 'adtv'})
        
        # Apply liquidity filters
        universe_10b = merged_data[merged_data['adtv'] >= self.thresholds['10B_VND']]
        universe_5b = merged_data[merged_data['adtv'] >= self.thresholds['5B_VND']]
        
        # Calculate metrics
        expansion_ratio = len(universe_5b) / len(universe_10b)
        additional_stocks = len(universe_5b) - len(universe_10b)
        
        # Calculate average metrics
        avg_adtv_10b = universe_10b['adtv'].mean()
        avg_adtv_5b = universe_5b['adtv'].mean()
        
        avg_qvm_10b = universe_10b['qvm_composite_score'].mean()
        avg_qvm_5b = universe_5b['qvm_composite_score'].mean()
        
        qvm_impact = avg_qvm_5b - avg_adtv_10b
        
        logger.info(f"✅ Universe expansion analysis complete")
        logger.info(f"   10B VND: {len(universe_10b)} stocks, {avg_adtv_10b/1e9:.1f}B VND avg ADTV")
        logger.info(f"   5B VND:  {len(universe_5b)} stocks, {avg_adtv_5b/1e9:.1f}B VND avg ADTV")
        logger.info(f"   Expansion: {expansion_ratio:.2f}x (+{additional_stocks} stocks)")
        
        return {
            'universe_10b': universe_10b,
            'universe_5b': universe_5b,
            'expansion_ratio': expansion_ratio,
            'additional_stocks': additional_stocks,
            'avg_adtv_10b': avg_adtv_10b,
            'avg_adtv_5b': avg_adtv_5b,
            'avg_qvm_10b': avg_qvm_10b,
            'avg_qvm_5b': avg_qvm_5b,
            'qvm_impact': qvm_impact
        }
    
    def analyze_liquidity_distribution(self, merged_data):
        """Analyze liquidity distribution characteristics."""
        logger.info("Analyzing liquidity distribution...")
        
        # Create liquidity buckets
        liquidity_buckets = {
            '5-10B VND': (5e9, 10e9),
            '10-20B VND': (10e9, 20e9),
            '20-50B VND': (20e9, 50e9),
            '50-100B VND': (50e9, 100e9),
            '100B+ VND': (100e9, float('inf'))
        }
        
        bucket_analysis = {}
        
        for bucket_name, (min_adtv, max_adtv) in liquidity_buckets.items():
            bucket_data = merged_data[
                (merged_data['adtv'] >= min_adtv) & 
                (merged_data['adtv'] < max_adtv)
            ]
            
            if len(bucket_data) > 0:
                bucket_analysis[bucket_name] = {
                    'count': len(bucket_data),
                    'avg_qvm': bucket_data['qvm_composite_score'].mean(),
                    'avg_adtv': bucket_data['adtv'].mean(),
                    'std_qvm': bucket_data['qvm_composite_score'].std()
                }
        
        return bucket_analysis
    
    def create_visualizations(self, analysis_results, bucket_analysis):
        """Create visualization of analysis results."""
        logger.info("Creating visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Universe Size Comparison
        ax1 = axes[0, 0]
        thresholds = ['10B VND', '5B VND']
        sizes = [len(analysis_results['universe_10b']), len(analysis_results['universe_5b'])]
        colors = ['#2E86AB', '#A23B72']
        
        bars = ax1.bar(thresholds, sizes, color=colors, alpha=0.7)
        ax1.set_title('Universe Size Comparison (5B vs 10B VND)', fontweight='bold')
        ax1.set_ylabel('Number of Stocks')
        
        # Add value labels on bars
        for bar, size in zip(bars, sizes):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{size}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Average ADTV Comparison
        ax2 = axes[0, 1]
        avg_adtv_values = [analysis_results['avg_adtv_10b']/1e9, analysis_results['avg_adtv_5b']/1e9]
        
        bars = ax2.bar(thresholds, avg_adtv_values, color=colors, alpha=0.7)
        ax2.set_title('Average ADTV Comparison (5B vs 10B VND)', fontweight='bold')
        ax2.set_ylabel('Average ADTV (Billion VND)')
        
        # Add value labels on bars
        for bar, value in zip(bars, avg_adtv_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{value:.1f}B', ha='center', va='bottom', fontweight='bold')
        
        # 3. QVM Score Comparison
        ax3 = axes[1, 0]
        qvm_values = [analysis_results['avg_qvm_10b'], analysis_results['avg_qvm_5b']]
        
        bars = ax3.bar(thresholds, qvm_values, color=colors, alpha=0.7)
        ax3.set_title('Average QVM Score Comparison (5B vs 10B VND)', fontweight='bold')
        ax3.set_ylabel('Average QVM Score')
        
        # Add value labels on bars
        for bar, value in zip(bars, qvm_values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Liquidity Bucket Analysis
        ax4 = axes[1, 1]
        bucket_names = list(bucket_analysis.keys())
        bucket_counts = [bucket_analysis[name]['count'] for name in bucket_names]
        bucket_qvm = [bucket_analysis[name]['avg_qvm'] for name in bucket_names]
        
        # Create scatter plot
        scatter = ax4.scatter(bucket_counts, bucket_qvm, s=100, alpha=0.7, c='#2E86AB')
        ax4.set_title('Liquidity Bucket Analysis', fontweight='bold')
        ax4.set_xlabel('Number of Stocks')
        ax4.set_ylabel('Average QVM Score')
        
        # Add labels for each bucket
        for i, name in enumerate(bucket_names):
            ax4.annotate(name, (bucket_counts[i], bucket_qvm[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('5b_vnd_quick_validation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("✅ Visualizations saved to 5b_vnd_quick_validation.png")
    
    def generate_report(self, analysis_results, bucket_analysis):
        """Generate comprehensive report."""
        logger.info("Generating 5B VND quick validation report...")
        
        report = []
        report.append("# 5B VND Liquidity Threshold Quick Validation")
        report.append("")
        report.append("**Date:** " + datetime.now().strftime("%Y-%m-%d"))
        report.append("**Purpose:** Quick validation of 5B VND liquidity threshold")
        report.append("**Context:** Alternative threshold investigation following 3B VND rejection")
        report.append("")
        
        # Executive Summary
        report.append("## 🎯 Executive Summary")
        report.append("")
        
        expansion_ratio = analysis_results['expansion_ratio']
        additional_stocks = analysis_results['additional_stocks']
        qvm_impact = analysis_results['qvm_impact']
        
        report.append(f"- **Universe Expansion:** {expansion_ratio:.2f}x ({additional_stocks} additional stocks)")
        report.append(f"- **QVM Score Impact:** {qvm_impact:+.3f}")
        report.append(f"- **Average ADTV:** {analysis_results['avg_adtv_5b']/1e9:.1f}B VND")
        report.append("")
        
        # Detailed Analysis
        report.append("## 📊 Detailed Analysis")
        report.append("")
        
        report.append("### Universe Comparison")
        report.append("")
        report.append("| Metric | 10B VND | 5B VND | Change |")
        report.append("|--------|---------|--------|--------|")
        report.append(f"| Universe Size | {len(analysis_results['universe_10b'])} | {len(analysis_results['universe_5b'])} | +{additional_stocks} |")
        report.append(f"| Average ADTV | {analysis_results['avg_adtv_10b']/1e9:.1f}B VND | {analysis_results['avg_adtv_5b']/1e9:.1f}B VND | {analysis_results['avg_adtv_5b'] - analysis_results['avg_adtv_10b']:+.1f}B VND |")
        report.append(f"| Average QVM | {analysis_results['avg_qvm_10b']:.3f} | {analysis_results['avg_qvm_5b']:.3f} | {qvm_impact:+.3f} |")
        report.append("")
        
        # Liquidity Bucket Analysis
        report.append("### Liquidity Bucket Analysis")
        report.append("")
        report.append("| Bucket | Count | Avg QVM | Avg ADTV |")
        report.append("|--------|-------|---------|----------|")
        
        for bucket_name, data in bucket_analysis.items():
            report.append(f"| {bucket_name} | {data['count']} | {data['avg_qvm']:.3f} | {data['avg_adtv']/1e9:.1f}B VND |")
        
        report.append("")
        
        # Assessment
        report.append("## 🎯 Assessment")
        report.append("")
        
        # Decision criteria
        expansion_acceptable = expansion_ratio < 1.5
        qvm_acceptable = qvm_impact >= -0.01  # Allow slight degradation
        adtv_acceptable = analysis_results['avg_adtv_5b'] >= 50e9  # At least 50B VND average
        
        if expansion_acceptable and qvm_acceptable and adtv_acceptable:
            report.append("✅ **PRELIMINARY APPROVAL**")
            report.append("- Universe expansion within acceptable range")
            report.append("- QVM score impact acceptable")
            report.append("- Average ADTV adequate")
            report.append("- **Recommendation:** Proceed to real data backtesting")
        elif expansion_acceptable and adtv_acceptable:
            report.append("⚠️ **CONDITIONAL APPROVAL**")
            report.append("- Universe expansion acceptable")
            report.append("- Average ADTV adequate")
            report.append("- QVM score impact needs monitoring")
            report.append("- **Recommendation:** Proceed with caution to real data backtesting")
        else:
            report.append("❌ **PRELIMINARY REJECTION**")
            report.append("- One or more criteria not met")
            report.append("- **Recommendation:** Consider other thresholds")
        
        report.append("")
        
        # Next Steps
        report.append("## 📋 Next Steps")
        report.append("")
        
        if expansion_acceptable and qvm_acceptable and adtv_acceptable:
            report.append("1. **Proceed to Real Data Backtesting**")
            report.append("   - Run full backtesting with real price data")
            report.append("   - Compare performance vs 10B VND baseline")
            report.append("   - Validate quick validation results")
        else:
            report.append("1. **Investigate Other Thresholds**")
            report.append("   - Test 7B VND threshold")
            report.append("   - Test 8B VND threshold")
            report.append("   - Compare results across thresholds")
        
        report.append("2. **Document Findings**")
        report.append("3. **Update Investigation Roadmap**")
        report.append("")
        
        report_text = "\n".join(report)
        
        # Save report
        with open('5b_vnd_quick_validation_report.md', 'w') as f:
            f.write(report_text)
        
        logger.info("✅ 5B VND quick validation report saved to 5b_vnd_quick_validation_report.md")
        return report_text
    
    def run_complete_validation(self):
        """Run the complete 5B VND quick validation."""
        logger.info("🚀 Starting 5B VND quick validation...")
        
        try:
            # Load data
            factor_scores, adtv_data = self.load_data()
            
            # Analyze universe expansion
            analysis_results = self.analyze_universe_expansion(factor_scores, adtv_data)
            
            # Analyze liquidity distribution
            merged_data = pd.concat([analysis_results['universe_10b'], analysis_results['universe_5b']]).drop_duplicates()
            bucket_analysis = self.analyze_liquidity_distribution(merged_data)
            
            # Create visualizations
            self.create_visualizations(analysis_results, bucket_analysis)
            
            # Generate report
            report = self.generate_report(analysis_results, bucket_analysis)
            
            # Save results
            results = {
                'analysis_results': analysis_results,
                'bucket_analysis': bucket_analysis,
                'report': report
            }
            
            # Save to pickle
            with open('5b_vnd_quick_validation_results.pkl', 'wb') as f:
                pickle.dump(results, f)
            
            logger.info("✅ Complete 5B VND quick validation finished successfully!")
            logger.info("📊 Results saved to:")
            logger.info("   - 5b_vnd_quick_validation.png")
            logger.info("   - 5b_vnd_quick_validation_report.md")
            logger.info("   - 5b_vnd_quick_validation_results.pkl")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ 5B VND quick validation failed: {e}")
            raise


def main():
    """Main execution function."""
    print("🔬 5B VND Quick Validation")
    print("=" * 40)
    
    # Initialize validator
    validator = FiveBVNDQuickValidator()
    
    # Run complete validation
    results = validator.run_complete_validation()
    
    print("\n✅ 5B VND quick validation completed successfully!")
    print("📊 Check the generated files for detailed results.")
    
    # Print key results
    analysis_results = results['analysis_results']
    
    print(f"\n📈 Key Results (5B VND):")
    print(f"   Universe Expansion: {analysis_results['expansion_ratio']:.2f}x (+{analysis_results['additional_stocks']} stocks)")
    print(f"   QVM Score Impact: {analysis_results['qvm_impact']:+.3f}")
    print(f"   Average ADTV: {analysis_results['avg_adtv_5b']/1e9:.1f}B VND")
    
    # Assessment
    expansion_acceptable = analysis_results['expansion_ratio'] < 1.5
    qvm_acceptable = analysis_results['qvm_impact'] >= -0.01
    adtv_acceptable = analysis_results['avg_adtv_5b'] >= 50e9
    
    if expansion_acceptable and qvm_acceptable and adtv_acceptable:
        print(f"   Assessment: ✅ PRELIMINARY APPROVAL")
    elif expansion_acceptable and adtv_acceptable:
        print(f"   Assessment: ⚠️ CONDITIONAL APPROVAL")
    else:
        print(f"   Assessment: ❌ PRELIMINARY REJECTION")


if __name__ == "__main__":
    main()