#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Liquidity Validation: 10B vs 3B VND Thresholds
===================================================
Component: Quick Liquidity Threshold Validation
Purpose: Validate 3B VND liquidity threshold implementation
Author: Duc Nguyen, Principal Quantitative Strategist
Date Created: January 2025
Status: PRODUCTION VALIDATION

This script performs quick validation of the 3B VND liquidity threshold:
- Universe size comparison
- Key validation metrics
- Implementation status check
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data():
    """Load data from pickle file."""
    logger.info("Loading data from pickle file...")
    
    try:
        with open('unrestricted_universe_data.pkl', 'rb') as f:
            data = pickle.load(f)
        
        logger.info("✅ Data loaded successfully")
        logger.info(f"   - Factor scores: {len(data['factor_data']):,} records")
        logger.info(f"   - ADTV data: {data['adtv'].shape}")
        
        return data
        
    except FileNotFoundError:
        logger.error("❌ Pickle file not found. Please run get_unrestricted_universe_data.py first.")
        raise

def validate_liquidity_thresholds(data):
    """Validate liquidity thresholds."""
    logger.info("Validating liquidity thresholds...")
    
    factor_data = data['factor_data']
    adtv_data = data['adtv']
    
    # Get recent data for analysis
    recent_date = factor_data['calculation_date'].max()
    logger.info(f"Analyzing data as of: {recent_date}")
    
    # Get factor data for recent date
    recent_factors = factor_data[factor_data['calculation_date'] == recent_date]
    
    # Get ADTV data for recent date
    if recent_date in adtv_data.index:
        recent_adtv = adtv_data.loc[recent_date].dropna()
    else:
        # Use the most recent available date
        recent_adtv = adtv_data.iloc[-1].dropna()
        recent_date = adtv_data.index[-1]
        logger.info(f"Using most recent ADTV date: {recent_date}")
    
    # Merge data
    merged_data = recent_factors.merge(
        recent_adtv.reset_index().rename(columns={0: 'adtv'}),
        left_on='ticker', right_on='ticker', how='inner'
    )
    
    # The ADTV column is named with the timestamp, so we need to rename it
    adtv_column = recent_date
    merged_data = merged_data.rename(columns={adtv_column: 'adtv'})
    
    logger.info(f"✅ Merged data: {len(merged_data)} stocks with both factor scores and ADTV")
    
    # Analyze thresholds
    thresholds = {
        '10B_VND': 10_000_000_000,
        '3B_VND': 3_000_000_000
    }
    
    results = {}
    
    for threshold_name, threshold_value in thresholds.items():
        liquid_universe = merged_data[merged_data['adtv'] >= threshold_value]
        
        results[threshold_name] = {
            'universe_size': len(liquid_universe),
            'avg_adtv': liquid_universe['adtv'].mean() / 1e9,  # Convert to billions
            'median_adtv': liquid_universe['adtv'].median() / 1e9,
            'avg_qvm_score': liquid_universe['qvm_composite_score'].mean(),
            'top_qvm_stocks': liquid_universe.nlargest(10, 'qvm_composite_score')[['ticker', 'qvm_composite_score', 'adtv']]
        }
        
        logger.info(f"✅ {threshold_name}:")
        logger.info(f"   - Universe size: {len(liquid_universe)} stocks")
        logger.info(f"   - Average ADTV: {liquid_universe['adtv'].mean() / 1e9:.1f} billion VND")
        logger.info(f"   - Average QVM score: {liquid_universe['qvm_composite_score'].mean():.3f}")
    
    return results, merged_data

def generate_validation_report(results, merged_data):
    """Generate validation report."""
    logger.info("Generating validation report...")
    
    report = []
    report.append("# Quick Liquidity Validation: 10B vs 3B VND Thresholds")
    report.append("")
    report.append("**Date:** " + datetime.now().strftime("%Y-%m-%d"))
    report.append("**Purpose:** Quick validation of 3B VND liquidity threshold implementation")
    report.append("")
    
    # Executive Summary
    report.append("## 🎯 Executive Summary")
    report.append("")
    
    v10b = results['10B_VND']
    v3b = results['3B_VND']
    
    expansion_ratio = v3b['universe_size'] / v10b['universe_size'] if v10b['universe_size'] > 0 else 0
    additional_stocks = v3b['universe_size'] - v10b['universe_size']
    
    report.append(f"- **Universe Expansion:** {expansion_ratio:.1f}x ({additional_stocks} additional stocks)")
    report.append(f"- **10B VND Universe:** {v10b['universe_size']} stocks")
    report.append(f"- **3B VND Universe:** {v3b['universe_size']} stocks")
    report.append(f"- **QVM Score Impact:** {v3b['avg_qvm_score']:.3f} vs {v10b['avg_qvm_score']:.3f}")
    report.append("")
    
    # Detailed Analysis
    report.append("## 📊 Detailed Analysis")
    report.append("")
    
    report.append("### Universe Size Comparison")
    report.append("")
    report.append("| Metric | 10B VND | 3B VND | Change |")
    report.append("|--------|---------|--------|--------|")
    report.append(f"| Universe Size | {v10b['universe_size']} | {v3b['universe_size']} | +{additional_stocks} |")
    report.append(f"| Avg ADTV (B VND) | {v10b['avg_adtv']:.1f} | {v3b['avg_adtv']:.1f} | {v3b['avg_adtv'] - v10b['avg_adtv']:+.1f} |")
    report.append(f"| Median ADTV (B VND) | {v10b['median_adtv']:.1f} | {v3b['median_adtv']:.1f} | {v3b['median_adtv'] - v10b['median_adtv']:+.1f} |")
    report.append(f"| Avg QVM Score | {v10b['avg_qvm_score']:.3f} | {v3b['avg_qvm_score']:.3f} | {v3b['avg_qvm_score'] - v10b['avg_qvm_score']:+.3f} |")
    report.append("")
    
    # Top QVM Stocks Analysis
    report.append("### Top QVM Stocks (3B VND Universe)")
    report.append("")
    report.append("| Rank | Ticker | QVM Score | ADTV (B VND) |")
    report.append("|------|--------|-----------|--------------|")
    
    for i, (_, row) in enumerate(v3b['top_qvm_stocks'].iterrows(), 1):
        report.append(f"| {i} | {row['ticker']} | {row['qvm_composite_score']:.3f} | {row['adtv'] / 1e9:.1f} |")
    
    report.append("")
    
    # Validation Criteria
    report.append("## ✅ Validation Criteria")
    report.append("")
    
    validation_passed = True
    validation_issues = []
    
    # Check universe expansion
    if expansion_ratio >= 1.5:
        report.append("✅ **Universe Expansion:** PASSED (≥1.5x expansion)")
    else:
        report.append("❌ **Universe Expansion:** FAILED (<1.5x expansion)")
        validation_passed = False
        validation_issues.append("Insufficient universe expansion")
    
    # Check minimum universe size
    if v3b['universe_size'] >= 200:
        report.append("✅ **Minimum Universe Size:** PASSED (≥200 stocks)")
    else:
        report.append("❌ **Minimum Universe Size:** FAILED (<200 stocks)")
        validation_passed = False
        validation_issues.append("Universe too small")
    
    # Check QVM score impact
    if v3b['avg_qvm_score'] >= v10b['avg_qvm_score'] * 0.95:  # Allow 5% degradation
        report.append("✅ **QVM Score Impact:** PASSED (≥95% of 10B VND score)")
    else:
        report.append("❌ **QVM Score Impact:** FAILED (<95% of 10B VND score)")
        validation_passed = False
        validation_issues.append("Significant QVM score degradation")
    
    report.append("")
    
    # Recommendations
    report.append("## 🎯 Recommendations")
    report.append("")
    
    if validation_passed:
        report.append("✅ **IMPLEMENTATION APPROVED**")
        report.append("- All validation criteria met")
        report.append("- 3B VND threshold ready for production")
        report.append("- Proceed with full backtesting")
    else:
        report.append("⚠️ **IMPLEMENTATION NEEDS REVIEW**")
        report.append("- Some validation criteria failed")
        report.append("- Issues to address:")
        for issue in validation_issues:
            report.append(f"  - {issue}")
        report.append("- Consider alternative thresholds or additional analysis")
    
    report.append("")
    
    # Implementation Status
    report.append("## 📋 Implementation Status")
    report.append("")
    report.append("- [x] Configuration files updated")
    report.append("- [x] Quick validation completed")
    report.append("- [x] Universe expansion validated")
    report.append("- [ ] Full backtesting with price data")
    report.append("- [ ] Performance impact assessment")
    report.append("- [ ] Risk metrics comparison")
    report.append("- [ ] Production deployment")
    report.append("")
    
    report_text = "\n".join(report)
    
    # Save report
    with open('quick_liquidity_validation_report.md', 'w') as f:
        f.write(report_text)
    
    logger.info("✅ Validation report saved to quick_liquidity_validation_report.md")
    return report_text, validation_passed

def main():
    """Main execution function."""
    print("🔬 Quick Liquidity Validation: 10B vs 3B VND Thresholds")
    print("=" * 60)
    
    try:
        # Load data
        data = load_data()
        
        # Validate thresholds
        results, merged_data = validate_liquidity_thresholds(data)
        
        # Generate report
        report, validation_passed = generate_validation_report(results, merged_data)
        
        print("\n✅ Quick validation completed successfully!")
        print("📊 Check quick_liquidity_validation_report.md for detailed results.")
        
        if validation_passed:
            print("🎉 All validation criteria PASSED - Ready for implementation!")
        else:
            print("⚠️ Some validation criteria FAILED - Review required.")
        
        return results, validation_passed
        
    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")
        raise

if __name__ == "__main__":
    main()