#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pickle vs Real Data Analysis
============================
Component: Data Discrepancy Investigation
Purpose: Analyze discrepancies between pickle data and real database data
Author: Duc Nguyen, Principal Quantitative Strategist
Date Created: January 2025
Status: DATA VALIDATION

This script analyzes discrepancies between:
- Pickle data (used in simplified backtesting)
- Real database data (used in full backtesting)

Key areas of investigation:
- Factor score differences
- ADTV calculation differences
- Date alignment issues
- Data completeness differences
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sqlalchemy import create_engine
import yaml
from datetime import datetime, timedelta
import warnings
import logging
import random

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PickleVsRealDataAnalyzer:
    """
    Analyzer for comparing pickle data vs real database data.
    """
    
    def __init__(self, config_path: str = "../../../config/database.yml"):
        """Initialize the analyzer."""
        self.config_path = config_path
        self.engine = self._create_database_engine()
        
        logger.info("Pickle vs Real Data Analyzer initialized")
    
    def _create_database_engine(self):
        """Create database engine."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            db_config = config['production']
            connection_string = (
                f"mysql+pymysql://{db_config['username']}:{db_config['password']}"
                f"@{db_config['host']}/{db_config['schema_name']}"
            )
            return create_engine(connection_string, pool_recycle=3600)
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def load_pickle_data(self):
        """Load data from pickle file."""
        logger.info("Loading pickle data...")
        
        with open('data/unrestricted_universe_data.pkl', 'rb') as f:
            pickle_data = pickle.load(f)
        
        factor_scores = pickle_data['factor_data']
        adtv_data = pickle_data['adtv']
        
        logger.info(f"✅ Pickle data loaded")
        logger.info(f"   - Factor scores: {factor_scores.shape}")
        logger.info(f"   - ADTV data: {adtv_data.shape}")
        
        return factor_scores, adtv_data
    
    def load_real_data(self, sample_dates=None, sample_tickers=None):
        """Load real data from database."""
        logger.info("Loading real data from database...")
        
        # Load factor scores
        factor_query = """
        SELECT date, ticker, QVM_Composite
        FROM factor_scores_qvm
        WHERE date >= '2018-01-01'
        """
        
        conditions = []
        if sample_dates:
            date_list = ','.join([f"'{date}'" for date in sample_dates])
            conditions.append(f"date IN ({date_list})")
        
        if sample_tickers:
            ticker_list = ','.join([f"'{ticker}'" for ticker in sample_tickers])
            conditions.append(f"ticker IN ({ticker_list})")
        
        if conditions:
            factor_query += " AND " + " AND ".join(conditions)
        
        factor_query += " ORDER BY date, ticker"
        
        factor_scores_db = pd.read_sql(factor_query, self.engine)
        factor_scores_db['date'] = pd.to_datetime(factor_scores_db['date'])
        
        # Load price data for ADTV calculation
        price_query = """
        SELECT trading_date, ticker, close_price_adjusted, total_volume
        FROM vcsc_daily_data_complete
        WHERE trading_date >= '2018-01-01'
        """
        
        conditions = []
        if sample_dates:
            date_list = ','.join([f"'{date}'" for date in sample_dates])
            conditions.append(f"trading_date IN ({date_list})")
        
        if sample_tickers:
            ticker_list = ','.join([f"'{ticker}'" for ticker in sample_tickers])
            conditions.append(f"ticker IN ({ticker_list})")
        
        if conditions:
            price_query += " AND " + " AND ".join(conditions)
        
        price_query += " ORDER BY trading_date, ticker"
        
        price_data_db = pd.read_sql(price_query, self.engine)
        price_data_db['trading_date'] = pd.to_datetime(price_data_db['trading_date'])
        
        # Calculate ADTV (63-day rolling average)
        price_data_db['adtv'] = price_data_db['close_price_adjusted'] * price_data_db['total_volume']
        adtv_db = price_data_db.groupby('ticker')['adtv'].rolling(window=63, min_periods=30).mean().reset_index()
        adtv_db = adtv_db.rename(columns={'level_1': 'trading_date'})
        adtv_db = adtv_db.pivot(index='trading_date', columns='ticker', values='adtv')
        
        logger.info(f"✅ Real data loaded")
        logger.info(f"   - Factor scores: {factor_scores_db.shape}")
        logger.info(f"   - Price data: {price_data_db.shape}")
        logger.info(f"   - ADTV data: {adtv_db.shape}")
        
        return factor_scores_db, adtv_db
    
    def sample_random_data(self, factor_scores_pickle, adtv_pickle, n_samples=10):
        """Sample random dates and tickers for comparison."""
        logger.info(f"Sampling {n_samples} random data points for comparison...")
        
        # Get random dates
        available_dates = factor_scores_pickle['calculation_date'].unique()
        sample_dates = random.sample(list(available_dates), min(n_samples, len(available_dates)))
        
        # Get random tickers
        available_tickers = factor_scores_pickle['ticker'].unique()
        sample_tickers = random.sample(list(available_tickers), min(n_samples, len(available_tickers)))
        
        logger.info(f"✅ Sampled {len(sample_dates)} dates and {len(sample_tickers)} tickers")
        
        return sample_dates, sample_tickers
    
    def compare_factor_scores(self, factor_scores_pickle, factor_scores_db, sample_dates, sample_tickers):
        """Compare factor scores between pickle and database."""
        logger.info("Comparing factor scores...")
        
        comparisons = []
        
        for date in sample_dates[:5]:  # Compare first 5 dates
            for ticker in sample_tickers[:5]:  # Compare first 5 tickers
                # Get pickle data
                pickle_value = factor_scores_pickle[
                    (factor_scores_pickle['calculation_date'] == date) & 
                    (factor_scores_pickle['ticker'] == ticker)
                ]['qvm_composite_score'].values
                
                # Get database data
                db_value = factor_scores_db[
                    (factor_scores_db['date'] == date) & 
                    (factor_scores_db['ticker'] == ticker)
                ]['QVM_Composite'].values
                
                pickle_score = pickle_value[0] if len(pickle_value) > 0 else None
                db_score = db_value[0] if len(db_value) > 0 else None
                
                if pickle_score is not None and db_score is not None:
                    difference = abs(pickle_score - db_score)
                    percent_diff = (difference / abs(db_score)) * 100 if db_score != 0 else 0
                    
                    comparisons.append({
                        'date': date,
                        'ticker': ticker,
                        'pickle_score': pickle_score,
                        'db_score': db_score,
                        'difference': difference,
                        'percent_diff': percent_diff
                    })
        
        return pd.DataFrame(comparisons)
    
    def compare_adtv_data(self, adtv_pickle, adtv_db, sample_dates, sample_tickers):
        """Compare ADTV data between pickle and database."""
        logger.info("Comparing ADTV data...")
        
        comparisons = []
        
        for date in sample_dates[:5]:  # Compare first 5 dates
            for ticker in sample_tickers[:5]:  # Compare first 5 tickers
                # Get pickle data
                try:
                    pickle_value = adtv_pickle.loc[date, ticker] if ticker in adtv_pickle.columns else None
                except:
                    pickle_value = None
                
                # Get database data
                try:
                    db_value = adtv_db.loc[date, ticker] if ticker in adtv_db.columns else None
                except:
                    db_value = None
                
                if pickle_value is not None and db_value is not None and not pd.isna(pickle_value) and not pd.isna(db_value):
                    difference = abs(pickle_value - db_value)
                    percent_diff = (difference / abs(db_value)) * 100 if db_value != 0 else 0
                    
                    comparisons.append({
                        'date': date,
                        'ticker': ticker,
                        'pickle_adtv': pickle_value,
                        'db_adtv': db_value,
                        'difference': difference,
                        'percent_diff': percent_diff
                    })
        
        return pd.DataFrame(comparisons)
    
    def analyze_data_completeness(self, factor_scores_pickle, adtv_pickle, factor_scores_db, adtv_db):
        """Analyze data completeness differences."""
        logger.info("Analyzing data completeness...")
        
        # Date ranges
        pickle_dates = factor_scores_pickle['calculation_date'].unique()
        db_dates = factor_scores_db['date'].unique()
        
        # Ticker coverage
        pickle_tickers = factor_scores_pickle['ticker'].unique()
        db_tickers = factor_scores_db['ticker'].unique()
        
        # ADTV coverage
        adtv_pickle_tickers = adtv_pickle.columns.tolist()
        adtv_db_tickers = adtv_db.columns.tolist()
        
        completeness_analysis = {
            'date_ranges': {
                'pickle_start': min(pickle_dates),
                'pickle_end': max(pickle_dates),
                'db_start': min(db_dates),
                'db_end': max(db_dates),
                'pickle_count': len(pickle_dates),
                'db_count': len(db_dates)
            },
            'ticker_coverage': {
                'pickle_factor_tickers': len(pickle_tickers),
                'db_factor_tickers': len(db_tickers),
                'pickle_adtv_tickers': len(adtv_pickle_tickers),
                'db_adtv_tickers': len(adtv_db_tickers),
                'common_factor_tickers': len(set(pickle_tickers) & set(db_tickers)),
                'common_adtv_tickers': len(set(adtv_pickle_tickers) & set(adtv_db_tickers))
            }
        }
        
        return completeness_analysis
    
    def create_discrepancy_visualizations(self, factor_comparisons, adtv_comparisons, completeness_analysis):
        """Create visualizations of discrepancies."""
        logger.info("Creating discrepancy visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Factor Score Differences
        if len(factor_comparisons) > 0:
            ax1 = axes[0, 0]
            ax1.scatter(factor_comparisons['db_score'], factor_comparisons['pickle_score'], alpha=0.7)
            ax1.plot([factor_comparisons['db_score'].min(), factor_comparisons['db_score'].max()], 
                    [factor_comparisons['db_score'].min(), factor_comparisons['db_score'].max()], 
                    'r--', alpha=0.5)
            ax1.set_title('Factor Score Comparison (Pickle vs Database)', fontweight='bold')
            ax1.set_xlabel('Database Score')
            ax1.set_ylabel('Pickle Score')
            ax1.grid(True, alpha=0.3)
        else:
            ax1 = axes[0, 0]
            ax1.text(0.5, 0.5, 'No factor score data available', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Factor Score Comparison', fontweight='bold')
        
        # 2. ADTV Differences
        if len(adtv_comparisons) > 0:
            ax2 = axes[0, 1]
            ax2.scatter(adtv_comparisons['db_adtv']/1e9, adtv_comparisons['pickle_adtv']/1e9, alpha=0.7)
            ax2.plot([adtv_comparisons['db_adtv'].min()/1e9, adtv_comparisons['db_adtv'].max()/1e9], 
                    [adtv_comparisons['db_adtv'].min()/1e9, adtv_comparisons['db_adtv'].max()/1e9], 
                    'r--', alpha=0.5)
            ax2.set_title('ADTV Comparison (Pickle vs Database)', fontweight='bold')
            ax2.set_xlabel('Database ADTV (Billion VND)')
            ax2.set_ylabel('Pickle ADTV (Billion VND)')
            ax2.grid(True, alpha=0.3)
        else:
            ax2 = axes[0, 1]
            ax2.text(0.5, 0.5, 'No ADTV data available', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('ADTV Comparison', fontweight='bold')
        
        # 3. Percentage Differences
        ax3 = axes[1, 0]
        if len(factor_comparisons) > 0 and len(adtv_comparisons) > 0:
            data = pd.concat([
                factor_comparisons['percent_diff'].rename('Factor Score'),
                adtv_comparisons['percent_diff'].rename('ADTV')
            ], axis=1)
            data.boxplot(ax=ax3)
            ax3.set_title('Percentage Differences Distribution', fontweight='bold')
            ax3.set_ylabel('Percentage Difference (%)')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No comparison data available', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Percentage Differences', fontweight='bold')
        
        # 4. Data Completeness Summary
        ax4 = axes[1, 1]
        ax4.axis('tight')
        ax4.axis('off')
        
        completeness_data = [
            ['Metric', 'Pickle', 'Database', 'Difference'],
            ['Date Count', completeness_analysis['date_ranges']['pickle_count'], 
             completeness_analysis['date_ranges']['db_count'],
             completeness_analysis['date_ranges']['pickle_count'] - completeness_analysis['date_ranges']['db_count']],
            ['Factor Tickers', completeness_analysis['ticker_coverage']['pickle_factor_tickers'],
             completeness_analysis['ticker_coverage']['db_factor_tickers'],
             completeness_analysis['ticker_coverage']['pickle_factor_tickers'] - completeness_analysis['ticker_coverage']['db_factor_tickers']],
            ['ADTV Tickers', completeness_analysis['ticker_coverage']['pickle_adtv_tickers'],
             completeness_analysis['ticker_coverage']['db_adtv_tickers'],
             completeness_analysis['ticker_coverage']['pickle_adtv_tickers'] - completeness_analysis['ticker_coverage']['db_adtv_tickers']]
        ]
        
        table = ax4.table(cellText=completeness_data[1:],
                         colLabels=completeness_data[0],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax4.set_title('Data Completeness Comparison', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('img/pickle_vs_real_data_discrepancies.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("✅ Discrepancy visualizations saved to img/pickle_vs_real_data_discrepancies.png")
    
    def generate_discrepancy_report(self, factor_comparisons, adtv_comparisons, completeness_analysis):
        """Generate comprehensive discrepancy report."""
        logger.info("Generating discrepancy report...")
        
        report = []
        report.append("# Pickle vs Real Data Discrepancy Analysis")
        report.append("")
        report.append("**Date:** " + datetime.now().strftime("%Y-%m-%d"))
        report.append("**Purpose:** Analyze discrepancies between pickle data and real database data")
        report.append("**Context:** Investigation of simplified vs real data backtesting differences")
        report.append("")
        
        # Executive Summary
        report.append("## 🎯 Executive Summary")
        report.append("")
        
        if len(factor_comparisons) > 0:
            avg_factor_diff = factor_comparisons['percent_diff'].mean()
            max_factor_diff = factor_comparisons['percent_diff'].max()
            report.append(f"- **Factor Score Differences:** Average {avg_factor_diff:.2f}%, Max {max_factor_diff:.2f}%")
        
        if len(adtv_comparisons) > 0:
            avg_adtv_diff = adtv_comparisons['percent_diff'].mean()
            max_adtv_diff = adtv_comparisons['percent_diff'].max()
            report.append(f"- **ADTV Differences:** Average {avg_adtv_diff:.2f}%, Max {max_adtv_diff:.2f}%")
        
        date_diff = completeness_analysis['date_ranges']['pickle_count'] - completeness_analysis['date_ranges']['db_count']
        ticker_diff = completeness_analysis['ticker_coverage']['pickle_factor_tickers'] - completeness_analysis['ticker_coverage']['db_factor_tickers']
        report.append(f"- **Data Completeness:** {date_diff:+d} dates, {ticker_diff:+d} tickers")
        report.append("")
        
        # Detailed Analysis
        report.append("## 📊 Detailed Analysis")
        report.append("")
        
        # Factor Score Comparison
        if len(factor_comparisons) > 0:
            report.append("### Factor Score Comparison")
            report.append("")
            report.append("| Date | Ticker | Pickle Score | DB Score | Difference | % Diff |")
            report.append("|------|--------|--------------|----------|------------|--------|")
            
            for _, row in factor_comparisons.head(10).iterrows():
                report.append(f"| {row['date'].strftime('%Y-%m-%d')} | {row['ticker']} | {row['pickle_score']:.4f} | {row['db_score']:.4f} | {row['difference']:.4f} | {row['percent_diff']:.2f}% |")
            report.append("")
        
        # ADTV Comparison
        if len(adtv_comparisons) > 0:
            report.append("### ADTV Comparison")
            report.append("")
            report.append("| Date | Ticker | Pickle ADTV | DB ADTV | Difference | % Diff |")
            report.append("|------|--------|-------------|---------|------------|--------|")
            
            for _, row in adtv_comparisons.head(10).iterrows():
                report.append(f"| {row['date'].strftime('%Y-%m-%d')} | {row['ticker']} | {row['pickle_adtv']/1e9:.1f}B | {row['db_adtv']/1e9:.1f}B | {row['difference']/1e9:.1f}B | {row['percent_diff']:.2f}% |")
            report.append("")
        
        # Data Completeness
        report.append("### Data Completeness Analysis")
        report.append("")
        report.append("#### Date Ranges")
        report.append(f"- **Pickle:** {completeness_analysis['date_ranges']['pickle_start']} to {completeness_analysis['date_ranges']['pickle_end']} ({completeness_analysis['date_ranges']['pickle_count']} dates)")
        report.append(f"- **Database:** {completeness_analysis['date_ranges']['db_start']} to {completeness_analysis['date_ranges']['db_end']} ({completeness_analysis['date_ranges']['db_count']} dates)")
        report.append("")
        
        report.append("#### Ticker Coverage")
        report.append(f"- **Factor Scores:** Pickle {completeness_analysis['ticker_coverage']['pickle_factor_tickers']} vs DB {completeness_analysis['ticker_coverage']['db_factor_tickers']} tickers")
        report.append(f"- **ADTV Data:** Pickle {completeness_analysis['ticker_coverage']['pickle_adtv_tickers']} vs DB {completeness_analysis['ticker_coverage']['db_adtv_tickers']} tickers")
        report.append(f"- **Common Factor Tickers:** {completeness_analysis['ticker_coverage']['common_factor_tickers']}")
        report.append(f"- **Common ADTV Tickers:** {completeness_analysis['ticker_coverage']['common_adtv_tickers']}")
        report.append("")
        
        # Implications
        report.append("## 🔍 Implications for Backtesting")
        report.append("")
        
        if len(factor_comparisons) > 0 and len(adtv_comparisons) > 0:
            avg_factor_diff = factor_comparisons['percent_diff'].mean()
            avg_adtv_diff = adtv_comparisons['percent_diff'].mean()
            
            if avg_factor_diff > 5 or avg_adtv_diff > 5:
                report.append("❌ **SIGNIFICANT DISCREPANCIES DETECTED**")
                report.append("- Data differences may explain simplified vs real backtesting results")
                report.append("- Pickle data may be outdated or calculated differently")
                report.append("- Real data validation is critical for accurate results")
            elif avg_factor_diff > 1 or avg_adtv_diff > 1:
                report.append("⚠️ **MODERATE DISCREPANCIES DETECTED**")
                report.append("- Some differences exist but may not be critical")
                report.append("- Real data validation still recommended")
                report.append("- Monitor for systematic biases")
            else:
                report.append("✅ **MINIMAL DISCREPANCIES DETECTED**")
                report.append("- Data sources appear consistent")
                report.append("- Simplified backtesting differences likely due to methodology")
                report.append("- Real data validation confirms methodology impact")
        else:
            report.append("❓ **INSUFFICIENT DATA FOR COMPARISON**")
            report.append("- Limited overlap between pickle and database data")
            report.append("- May indicate different data sources or time periods")
            report.append("- Further investigation required")
        
        report.append("")
        
        # Recommendations
        report.append("## 📋 Recommendations")
        report.append("")
        
        report.append("1. **Data Source Alignment**")
        report.append("   - Ensure pickle data uses same source as database")
        report.append("   - Verify calculation methodologies are identical")
        report.append("   - Update pickle data if outdated")
        
        report.append("2. **Validation Framework**")
        report.append("   - Always validate simplified results with real data")
        report.append("   - Use consistent data sources across all analyses")
        report.append("   - Document data source differences")
        
        report.append("3. **Methodology Review**")
        report.append("   - Review simplified backtesting assumptions")
        report.append("   - Consider real market dynamics in simulations")
        report.append("   - Implement more realistic transaction cost models")
        
        report.append("")
        
        report_text = "\n".join(report)
        
        # Save report
        with open('pickle_vs_real_data_discrepancy_report.md', 'w') as f:
            f.write(report_text)
        
        logger.info("✅ Discrepancy report saved to pickle_vs_real_data_discrepancy_report.md")
        return report_text
    
    def run_complete_analysis(self):
        """Run the complete discrepancy analysis."""
        logger.info("🚀 Starting pickle vs real data discrepancy analysis...")
        
        try:
            # Load pickle data
            factor_scores_pickle, adtv_pickle = self.load_pickle_data()
            
            # Sample random data
            sample_dates, sample_tickers = self.sample_random_data(factor_scores_pickle, adtv_pickle, n_samples=20)
            
            # Load real data
            factor_scores_db, adtv_db = self.load_real_data(sample_dates, sample_tickers)
            
            # Compare data
            factor_comparisons = self.compare_factor_scores(factor_scores_pickle, factor_scores_db, sample_dates, sample_tickers)
            adtv_comparisons = self.compare_adtv_data(adtv_pickle, adtv_db, sample_dates, sample_tickers)
            
            # Analyze completeness
            completeness_analysis = self.analyze_data_completeness(factor_scores_pickle, adtv_pickle, factor_scores_db, adtv_db)
            
            # Create visualizations
            self.create_discrepancy_visualizations(factor_comparisons, adtv_comparisons, completeness_analysis)
            
            # Generate report
            report = self.generate_discrepancy_report(factor_comparisons, adtv_comparisons, completeness_analysis)
            
            # Save results
            results = {
                'factor_comparisons': factor_comparisons,
                'adtv_comparisons': adtv_comparisons,
                'completeness_analysis': completeness_analysis,
                'report': report
            }
            
            # Save to pickle
            with open('data/pickle_vs_real_data_analysis_results.pkl', 'wb') as f:
                pickle.dump(results, f)
            
            logger.info("✅ Complete discrepancy analysis finished successfully!")
            logger.info("📊 Results saved to:")
            logger.info("   - img/pickle_vs_real_data_discrepancies.png")
            logger.info("   - pickle_vs_real_data_discrepancy_report.md")
            logger.info("   - data/pickle_vs_real_data_analysis_results.pkl")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Discrepancy analysis failed: {e}")
            raise


def main():
    """Main execution function."""
    print("🔬 Pickle vs Real Data Discrepancy Analysis")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = PickleVsRealDataAnalyzer()
    
    # Run complete analysis
    results = analyzer.run_complete_analysis()
    
    print("\n✅ Discrepancy analysis completed successfully!")
    print("📊 Check the generated files for detailed results.")
    
    # Print key results
    factor_comparisons = results['factor_comparisons']
    adtv_comparisons = results['adtv_comparisons']
    
    print(f"\n📈 Key Findings:")
    if len(factor_comparisons) > 0:
        avg_factor_diff = factor_comparisons['percent_diff'].mean()
        print(f"   Factor Score Differences: {avg_factor_diff:.2f}% average")
    
    if len(adtv_comparisons) > 0:
        avg_adtv_diff = adtv_comparisons['percent_diff'].mean()
        print(f"   ADTV Differences: {avg_adtv_diff:.2f}% average")
    
    completeness = results['completeness_analysis']
    print(f"   Data Completeness: {completeness['date_ranges']['pickle_count']} vs {completeness['date_ranges']['db_count']} dates")


if __name__ == "__main__":
    main()