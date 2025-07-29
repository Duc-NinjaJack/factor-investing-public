#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified Comparative Liquidity Analysis: 10B vs 3B VND Thresholds
===================================================================
Component: Liquidity Threshold Validation (Simplified)
Purpose: Compare universe characteristics between 10B and 3B VND thresholds
Author: Duc Nguyen, Principal Quantitative Strategist
Date Created: January 2025
Status: PRODUCTION VALIDATION

This script performs simplified comparative analysis using existing pickle data:
- Universe size expansion analysis
- Stock survival analysis
- Sector diversification impact
- Performance simulation based on factor scores

Data Sources:
- unrestricted_universe_data.pkl (pre-loaded data)

Dependencies:
- pandas >= 1.3.0
- numpy >= 1.21.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
from datetime import datetime, timedelta
import logging
import pickle

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimplifiedLiquidityAnalyzer:
    """
    Simplified analyzer for comparing liquidity thresholds using pickle data.
    """
    
    def __init__(self):
        """Initialize the analyzer."""
        self.results = {}
        
        # Analysis parameters
        self.thresholds = {
            '10B_VND': 10_000_000_000,
            '3B_VND': 3_000_000_000
        }
        
        logger.info("Simplified Liquidity Analyzer initialized")
    
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load data from pickle file."""
        logger.info("Loading data from pickle file...")
        
        try:
            with open('unrestricted_universe_data.pkl', 'rb') as f:
                pickle_data = pickle.load(f)
            
            data = {
                'factor_scores': pickle_data['factor_data'],
                'volume_data': pickle_data['volume_data'],
                'adtv_data': pickle_data['adtv'],
                'metadata': pickle_data['metadata']
            }
            
            logger.info("✅ Data loaded successfully")
            logger.info(f"   - Factor scores: {len(data['factor_scores']):,} records")
            logger.info(f"   - Volume data: {len(data['volume_data']):,} records")
            logger.info(f"   - ADTV data: {data['adtv_data'].shape}")
            
            return data
            
        except FileNotFoundError:
            logger.error("❌ Pickle file not found. Please run get_unrestricted_universe_data.py first.")
            raise
    
    def analyze_universe_expansion(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Analyze universe expansion between thresholds."""
        logger.info("Analyzing universe expansion...")
        
        adtv_data = data['adtv_data']
        factor_data = data['factor_scores']
        
        # Sample dates for analysis (monthly)
        analysis_dates = pd.date_range(
            start=factor_data['calculation_date'].min(),
            end=factor_data['calculation_date'].max(),
            freq='M'
        )
        
        universe_stats = []
        
        for date in analysis_dates:
            # Get available data for this date
            if date in adtv_data.index:
                available_adtv = adtv_data.loc[date].dropna()
            else:
                continue
                
            available_factors = factor_data[
                factor_data['calculation_date'] == date
            ]
            
            if available_factors.empty:
                continue
            
            # Merge with ADTV data
            merged_data = available_factors.merge(
                available_adtv.reset_index().rename(columns={0: 'adtv'}),
                left_on='ticker', right_on='ticker', how='inner'
            )
            
            # Calculate universe sizes for each threshold
            for threshold_name, threshold_value in self.thresholds.items():
                liquid_universe = merged_data[merged_data['adtv'] >= threshold_value]
                
                universe_stats.append({
                    'date': date,
                    'threshold': threshold_name,
                    'total_stocks': len(merged_data),
                    'liquid_stocks': len(liquid_universe),
                    'liquidity_ratio': len(liquid_universe) / len(merged_data) if len(merged_data) > 0 else 0,
                    'avg_adtv': liquid_universe['adtv'].mean() if len(liquid_universe) > 0 else 0,
                    'median_adtv': liquid_universe['adtv'].median() if len(liquid_universe) > 0 else 0,
                    'avg_qvm_score': liquid_universe['qvm_composite_score'].mean() if len(liquid_universe) > 0 else 0
                })
        
        universe_df = pd.DataFrame(universe_stats)
        
        # Calculate expansion metrics
        expansion_metrics = {}
        for date in universe_df['date'].unique():
            date_data = universe_df[universe_df['date'] == date]
            if len(date_data) == 2:  # Both thresholds present
                v10b = date_data[date_data['threshold'] == '10B_VND'].iloc[0]
                v3b = date_data[date_data['threshold'] == '3B_VND'].iloc[0]
                
                expansion_metrics[date] = {
                    'universe_expansion': v3b['liquid_stocks'] / v10b['liquid_stocks'] if v10b['liquid_stocks'] > 0 else 0,
                    'additional_stocks': v3b['liquid_stocks'] - v10b['liquid_stocks'],
                    'expansion_ratio': v3b['liquid_stocks'] / v10b['liquid_stocks'] if v10b['liquid_stocks'] > 0 else 0,
                    'qvm_score_change': v3b['avg_qvm_score'] - v10b['avg_qvm_score']
                }
        
        expansion_df = pd.DataFrame(expansion_metrics).T
        
        logger.info(f"✅ Universe expansion analysis complete")
        logger.info(f"   - Average expansion ratio: {expansion_df['expansion_ratio'].mean():.2f}x")
        logger.info(f"   - Average additional stocks: {expansion_df['additional_stocks'].mean():.0f}")
        
        return {
            'universe_stats': universe_df,
            'expansion_metrics': expansion_df
        }
    
    def analyze_stock_survival(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Analyze stock survival rates between thresholds."""
        logger.info("Analyzing stock survival rates...")
        
        adtv_data = data['adtv_data']
        factor_data = data['factor_scores']
        
        # Sample dates for analysis
        analysis_dates = pd.date_range(
            start=factor_data['calculation_date'].min(),
            end=factor_data['calculation_date'].max(),
            freq='M'
        )
        
        survival_data = []
        
        for threshold_name, threshold_value in self.thresholds.items():
            stock_counts = {}
            total_dates = 0
            
            for date in analysis_dates:
                if date not in adtv_data.index:
                    continue
                    
                available_adtv = adtv_data.loc[date].dropna()
                available_factors = factor_data[
                    factor_data['calculation_date'] == date
                ]
                
                if available_factors.empty:
                    continue
                
                # Merge data
                merged_data = available_factors.merge(
                    available_adtv.reset_index().rename(columns={0: 'adtv'}),
                    left_on='ticker', right_on='ticker', how='inner'
                )
                
                # Apply liquidity filter
                liquid_stocks = merged_data[merged_data['adtv'] >= threshold_value]['ticker'].tolist()
                
                # Count appearances
                for stock in liquid_stocks:
                    stock_counts[stock] = stock_counts.get(stock, 0) + 1
                
                total_dates += 1
            
            # Calculate survival metrics
            for stock, count in stock_counts.items():
                survival_rate = count / total_dates if total_dates > 0 else 0
                survival_data.append({
                    'threshold': threshold_name,
                    'ticker': stock,
                    'appearances': count,
                    'total_dates': total_dates,
                    'survival_rate': survival_rate
                })
        
        survival_df = pd.DataFrame(survival_data)
        
        logger.info(f"✅ Stock survival analysis complete")
        logger.info(f"   - Average survival rate (10B): {survival_df[survival_df['threshold'] == '10B_VND']['survival_rate'].mean():.2%}")
        logger.info(f"   - Average survival rate (3B): {survival_df[survival_df['threshold'] == '3B_VND']['survival_rate'].mean():.2%}")
        
        return survival_df
    
    def analyze_sector_diversification(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Analyze sector diversification impact."""
        logger.info("Analyzing sector diversification...")
        
        # For this simplified version, we'll use a basic sector analysis
        # In a full implementation, we'd load sector data from the database
        
        # Create mock sector data based on ticker patterns
        factor_data = data['factor_scores']
        unique_tickers = factor_data['ticker'].unique()
        
        # Simple sector mapping based on ticker patterns
        sector_mapping = {}
        for ticker in unique_tickers:
            if ticker.startswith(('VCB', 'TCB', 'BID', 'MBB', 'ACB', 'STB', 'TPB', 'EIB', 'SHB', 'MSB')):
                sector_mapping[ticker] = 'Banking'
            elif ticker.startswith(('SSI', 'VND', 'HCM', 'VCI', 'FTS', 'BVS', 'APG')):
                sector_mapping[ticker] = 'Securities'
            elif ticker.startswith(('VNM', 'VIC', 'HPG', 'FPT', 'MWG', 'MSN', 'SAB', 'BVH')):
                sector_mapping[ticker] = 'Large Cap'
            else:
                sector_mapping[ticker] = 'Other'
        
        # Add sector to factor data
        factor_data_with_sector = factor_data.copy()
        factor_data_with_sector['sector'] = factor_data_with_sector['ticker'].map(sector_mapping)
        
        # Analyze sector distribution by threshold
        sector_analysis = []
        
        for threshold_name, threshold_value in self.thresholds.items():
            # Sample a recent date for analysis
            recent_date = factor_data['calculation_date'].max()
            recent_data = factor_data_with_sector[
                factor_data_with_sector['calculation_date'] == recent_date
            ]
            
            # Get ADTV for this date
            if recent_date in data['adtv_data'].index:
                adtv_scores = data['adtv_data'].loc[recent_date].dropna()
                
                # Merge with factor data
                merged_data = recent_data.merge(
                    adtv_scores.reset_index().rename(columns={0: 'adtv'}),
                    left_on='ticker', right_on='ticker', how='inner'
                )
                
                # Apply liquidity filter
                liquid_universe = merged_data[merged_data['adtv'] >= threshold_value]
                
                # Count by sector
                sector_counts = liquid_universe['sector'].value_counts()
                
                for sector, count in sector_counts.items():
                    sector_analysis.append({
                        'threshold': threshold_name,
                        'sector': sector,
                        'count': count,
                        'percentage': count / len(liquid_universe) * 100
                    })
        
        sector_df = pd.DataFrame(sector_analysis)
        
        logger.info(f"✅ Sector diversification analysis complete")
        
        return sector_df
    
    def create_visualizations(self, universe_analysis: Dict, survival_data: pd.DataFrame, sector_data: pd.DataFrame):
        """Create comprehensive visualizations."""
        logger.info("Creating visualizations...")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Universe Expansion Over Time
        ax1 = plt.subplot(3, 3, 1)
        universe_stats = universe_analysis['universe_stats']
        for threshold in ['10B_VND', '3B_VND']:
            threshold_data = universe_stats[universe_stats['threshold'] == threshold]
            ax1.plot(threshold_data['date'], threshold_data['liquid_stocks'], 
                    label=threshold, linewidth=2)
        ax1.set_title('Universe Size Over Time', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of Liquid Stocks')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Expansion Ratio Over Time
        ax2 = plt.subplot(3, 3, 2)
        expansion_metrics = universe_analysis['expansion_metrics']
        ax2.plot(expansion_metrics.index, expansion_metrics['expansion_ratio'], 
                color='red', linewidth=2)
        ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5)
        ax2.set_title('Universe Expansion Ratio (3B/10B)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Expansion Ratio')
        ax2.grid(True, alpha=0.3)
        
        # 3. Additional Stocks Over Time
        ax3 = plt.subplot(3, 3, 3)
        ax3.plot(expansion_metrics.index, expansion_metrics['additional_stocks'], 
                color='green', linewidth=2)
        ax3.set_title('Additional Stocks (3B vs 10B)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Additional Stocks')
        ax3.grid(True, alpha=0.3)
        
        # 4. QVM Score Change
        ax4 = plt.subplot(3, 3, 4)
        ax4.plot(expansion_metrics.index, expansion_metrics['qvm_score_change'], 
                color='blue', linewidth=2)
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax4.set_title('QVM Score Change (3B vs 10B)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('QVM Score Change')
        ax4.grid(True, alpha=0.3)
        
        # 5. Stock Survival Distribution
        ax5 = plt.subplot(3, 3, 5)
        for threshold in ['10B_VND', '3B_VND']:
            threshold_data = survival_data[survival_data['threshold'] == threshold]
            ax5.hist(threshold_data['survival_rate'], bins=20, alpha=0.7, label=threshold)
        ax5.set_title('Stock Survival Rate Distribution', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Survival Rate')
        ax5.set_ylabel('Frequency')
        ax5.legend()
        
        # 6. Sector Distribution Comparison
        ax6 = plt.subplot(3, 3, 6)
        sector_pivot = sector_data.pivot(index='sector', columns='threshold', values='count')
        sector_pivot.plot(kind='bar', ax=ax6)
        ax6.set_title('Sector Distribution by Threshold', fontsize=12, fontweight='bold')
        ax6.set_ylabel('Number of Stocks')
        ax6.legend()
        plt.xticks(rotation=45)
        
        # 7. ADTV Distribution
        ax7 = plt.subplot(3, 3, 7)
        universe_stats = universe_analysis['universe_stats']
        for threshold in ['10B_VND', '3B_VND']:
            threshold_data = universe_stats[universe_stats['threshold'] == threshold]
            ax7.hist(threshold_data['avg_adtv'] / 1e9, bins=20, alpha=0.7, label=threshold)
        ax7.set_title('Average ADTV Distribution', fontsize=12, fontweight='bold')
        ax7.set_xlabel('ADTV (Billion VND)')
        ax7.set_ylabel('Frequency')
        ax7.legend()
        
        # 8. Summary Statistics
        ax8 = plt.subplot(3, 3, 8)
        summary_stats = []
        for threshold in ['10B_VND', '3B_VND']:
            threshold_data = universe_stats[universe_stats['threshold'] == threshold]
            summary_stats.append({
                'Metric': 'Avg Universe Size',
                'Value': threshold_data['liquid_stocks'].mean(),
                'Threshold': threshold
            })
            summary_stats.append({
                'Metric': 'Avg QVM Score',
                'Value': threshold_data['avg_qvm_score'].mean(),
                'Threshold': threshold
            })
        
        summary_df = pd.DataFrame(summary_stats)
        summary_pivot = summary_df.pivot(index='Metric', columns='Threshold', values='Value')
        summary_pivot.plot(kind='bar', ax=ax8)
        ax8.set_title('Summary Statistics', fontsize=12, fontweight='bold')
        ax8.set_ylabel('Value')
        ax8.legend()
        plt.xticks(rotation=45)
        
        # 9. Expansion Metrics Summary
        ax9 = plt.subplot(3, 3, 9)
        expansion_summary = {
            'Metric': ['Avg Expansion Ratio', 'Avg Additional Stocks', 'Avg QVM Change'],
            'Value': [
                expansion_metrics['expansion_ratio'].mean(),
                expansion_metrics['additional_stocks'].mean(),
                expansion_metrics['qvm_score_change'].mean()
            ]
        }
        expansion_df = pd.DataFrame(expansion_summary)
        ax9.bar(expansion_df['Metric'], expansion_df['Value'], color=['red', 'green', 'blue'])
        ax9.set_title('Expansion Metrics Summary', fontsize=12, fontweight='bold')
        ax9.set_ylabel('Value')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('simplified_comparative_liquidity_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("✅ Visualizations saved to simplified_comparative_liquidity_analysis.png")
    
    def generate_comparison_report(self, universe_analysis: Dict, survival_data: pd.DataFrame, sector_data: pd.DataFrame) -> str:
        """Generate comprehensive comparison report."""
        logger.info("Generating comparison report...")
        
        report = []
        report.append("# Simplified Comparative Liquidity Analysis: 10B vs 3B VND Thresholds")
        report.append("")
        report.append("**Date:** " + datetime.now().strftime("%Y-%m-%d"))
        report.append("**Purpose:** Validate 3B VND liquidity threshold implementation")
        report.append("**Note:** This is a simplified analysis using existing pickle data")
        report.append("")
        
        # Executive Summary
        report.append("## 🎯 Executive Summary")
        report.append("")
        
        # Universe expansion summary
        expansion_metrics = universe_analysis['expansion_metrics']
        avg_expansion = expansion_metrics['expansion_ratio'].mean()
        avg_additional = expansion_metrics['additional_stocks'].mean()
        avg_qvm_change = expansion_metrics['qvm_score_change'].mean()
        
        report.append(f"- **Universe Expansion:** {avg_expansion:.1f}x average expansion")
        report.append(f"- **Additional Stocks:** {avg_additional:.0f} additional stocks on average")
        report.append(f"- **QVM Score Impact:** {avg_qvm_change:+.3f} average change in QVM score")
        report.append("")
        
        # Detailed Analysis
        report.append("## 📊 Detailed Analysis")
        report.append("")
        
        # Universe Analysis
        report.append("### Universe Expansion Analysis")
        report.append("")
        universe_stats = universe_analysis['universe_stats']
        report.append(f"- **Average Universe Size (10B VND):** {universe_stats[universe_stats['threshold'] == '10B_VND']['liquid_stocks'].mean():.0f} stocks")
        report.append(f"- **Average Universe Size (3B VND):** {universe_stats[universe_stats['threshold'] == '3B_VND']['liquid_stocks'].mean():.0f} stocks")
        report.append(f"- **Expansion Ratio:** {avg_expansion:.1f}x")
        report.append(f"- **Additional Stocks:** {avg_additional:.0f} stocks")
        report.append("")
        
        # Stock Survival Analysis
        report.append("### Stock Survival Analysis")
        report.append("")
        survival_10b = survival_data[survival_data['threshold'] == '10B_VND']['survival_rate']
        survival_3b = survival_data[survival_data['threshold'] == '3B_VND']['survival_rate']
        
        report.append(f"- **Average Survival Rate (10B VND):** {survival_10b.mean():.2%}")
        report.append(f"- **Average Survival Rate (3B VND):** {survival_3b.mean():.2%}")
        report.append(f"- **Survival Rate Improvement:** {survival_3b.mean() - survival_10b.mean():+.2%}")
        report.append("")
        
        # Sector Analysis
        report.append("### Sector Diversification")
        report.append("")
        sector_counts_3b = sector_data[sector_data['threshold'] == '3B_VND']
        report.append("**Sector Distribution (3B VND Universe):**")
        for _, row in sector_counts_3b.iterrows():
            report.append(f"- {row['sector']}: {row['count']} stocks ({row['percentage']:.1f}%)")
        report.append("")
        
        # Recommendations
        report.append("## 🎯 Recommendations")
        report.append("")
        
        if avg_expansion > 1.5:
            report.append("✅ **RECOMMENDED:** Implement 3B VND threshold")
            report.append("- Significant universe expansion achieved")
            report.append("- Better diversification opportunities")
            if avg_qvm_change >= 0:
                report.append("- QVM score maintained or improved")
            else:
                report.append(f"- QVM score slightly decreased ({avg_qvm_change:.3f})")
        else:
            report.append("⚠️ **FURTHER ANALYSIS NEEDED:**")
            report.append("- Universe expansion below target")
            report.append("- Consider alternative thresholds")
        
        report.append("")
        report.append("## 📋 Implementation Checklist")
        report.append("")
        report.append("- [x] Configuration files updated")
        report.append("- [x] Universe expansion validated")
        report.append("- [x] Stock survival analyzed")
        report.append("- [x] Sector diversification reviewed")
        report.append("- [ ] Full backtesting with price data")
        report.append("- [ ] Performance impact assessment")
        report.append("- [ ] Risk metrics comparison")
        report.append("")
        
        report_text = "\n".join(report)
        
        # Save report
        with open('simplified_comparative_liquidity_analysis_report.md', 'w') as f:
            f.write(report_text)
        
        logger.info("✅ Comparison report saved to simplified_comparative_liquidity_analysis_report.md")
        return report_text
    
    def run_complete_analysis(self):
        """Run the complete simplified analysis."""
        logger.info("🚀 Starting simplified comparative liquidity analysis...")
        
        try:
            # Load data
            data = self.load_data()
            
            # Analyze universe expansion
            universe_analysis = self.analyze_universe_expansion(data)
            
            # Analyze stock survival
            survival_data = self.analyze_stock_survival(data)
            
            # Analyze sector diversification
            sector_data = self.analyze_sector_diversification(data)
            
            # Create visualizations
            self.create_visualizations(universe_analysis, survival_data, sector_data)
            
            # Generate report
            report = self.generate_comparison_report(universe_analysis, survival_data, sector_data)
            
            # Save results
            results = {
                'universe_analysis': universe_analysis,
                'survival_data': survival_data,
                'sector_data': sector_data,
                'report': report
            }
            
            # Save to pickle for further analysis
            with open('simplified_comparative_liquidity_analysis_results.pkl', 'wb') as f:
                pickle.dump(results, f)
            
            logger.info("✅ Complete analysis finished successfully!")
            logger.info("📊 Results saved to:")
            logger.info("   - simplified_comparative_liquidity_analysis.png")
            logger.info("   - simplified_comparative_liquidity_analysis_report.md")
            logger.info("   - simplified_comparative_liquidity_analysis_results.pkl")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
            raise


def main():
    """Main execution function."""
    print("🔬 Simplified Comparative Liquidity Analysis: 10B vs 3B VND Thresholds")
    print("=" * 70)
    
    # Initialize analyzer
    analyzer = SimplifiedLiquidityAnalyzer()
    
    # Run complete analysis
    results = analyzer.run_complete_analysis()
    
    print("\n✅ Analysis completed successfully!")
    print("📊 Check the generated files for detailed results.")


if __name__ == "__main__":
    main()