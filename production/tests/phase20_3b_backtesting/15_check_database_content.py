#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Database Content Check
======================
Component: Database Investigation
Purpose: Check actual content of database tables without filters
Author: Duc Nguyen, Principal Quantitative Strategist
Date Created: January 2025
Status: DATABASE INVESTIGATION

This script checks the actual content of database tables:
- factor_scores_qvm
- vcsc_daily_data_complete
- etf_history

To understand what data is actually available without any filters.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import yaml
from datetime import datetime
import warnings
import logging

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatabaseContentChecker:
    """
    Checker for database content without filters.
    """
    
    def __init__(self, config_path: str = "../../../config/database.yml"):
        """Initialize the checker."""
        self.config_path = config_path
        self.engine = self._create_database_engine()
        
        logger.info("Database Content Checker initialized")
    
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
    
    def check_factor_scores_table(self):
        """Check factor_scores_qvm table content."""
        logger.info("Checking factor_scores_qvm table...")
        
        # Check total records
        count_query = "SELECT COUNT(*) as total_records FROM factor_scores_qvm"
        total_count = pd.read_sql(count_query, self.engine).iloc[0]['total_records']
        
        # Check date range
        date_range_query = """
        SELECT MIN(date) as min_date, MAX(date) as max_date, 
               COUNT(DISTINCT date) as unique_dates
        FROM factor_scores_qvm
        """
        date_range = pd.read_sql(date_range_query, self.engine)
        
        # Check ticker coverage
        ticker_query = """
        SELECT COUNT(DISTINCT ticker) as unique_tickers,
               COUNT(DISTINCT date) as unique_dates
        FROM factor_scores_qvm
        """
        ticker_info = pd.read_sql(ticker_query, self.engine)
        
        # Get sample of recent data
        recent_query = """
        SELECT date, ticker, QVM_Composite
        FROM factor_scores_qvm
        WHERE date >= '2024-01-01'
        ORDER BY date DESC, ticker
        LIMIT 100
        """
        recent_data = pd.read_sql(recent_query, self.engine)
        
        # Get unique tickers in recent data
        recent_tickers = recent_data['ticker'].unique()
        
        logger.info(f"✅ Factor scores table analysis complete")
        logger.info(f"   - Total records: {total_count:,}")
        logger.info(f"   - Date range: {date_range.iloc[0]['min_date']} to {date_range.iloc[0]['max_date']}")
        logger.info(f"   - Unique dates: {date_range.iloc[0]['unique_dates']}")
        logger.info(f"   - Unique tickers: {ticker_info.iloc[0]['unique_tickers']}")
        logger.info(f"   - Recent tickers (2024+): {len(recent_tickers)}")
        
        return {
            'total_records': total_count,
            'date_range': date_range,
            'ticker_info': ticker_info,
            'recent_data': recent_data,
            'recent_tickers': recent_tickers
        }
    
    def check_price_data_table(self):
        """Check vcsc_daily_data_complete table content."""
        logger.info("Checking vcsc_daily_data_complete table...")
        
        # Check total records
        count_query = "SELECT COUNT(*) as total_records FROM vcsc_daily_data_complete"
        total_count = pd.read_sql(count_query, self.engine).iloc[0]['total_records']
        
        # Check date range
        date_range_query = """
        SELECT MIN(trading_date) as min_date, MAX(trading_date) as max_date,
               COUNT(DISTINCT trading_date) as unique_dates
        FROM vcsc_daily_data_complete
        """
        date_range = pd.read_sql(date_range_query, self.engine)
        
        # Check ticker coverage
        ticker_query = """
        SELECT COUNT(DISTINCT ticker) as unique_tickers,
               COUNT(DISTINCT trading_date) as unique_dates
        FROM vcsc_daily_data_complete
        """
        ticker_info = pd.read_sql(ticker_query, self.engine)
        
        # Get sample of recent data
        recent_query = """
        SELECT trading_date, ticker, close_price_adjusted, total_volume
        FROM vcsc_daily_data_complete
        WHERE trading_date >= '2024-01-01'
        ORDER BY trading_date DESC, ticker
        LIMIT 100
        """
        recent_data = pd.read_sql(recent_query, self.engine)
        
        # Get unique tickers in recent data
        recent_tickers = recent_data['ticker'].unique()
        
        logger.info(f"✅ Price data table analysis complete")
        logger.info(f"   - Total records: {total_count:,}")
        logger.info(f"   - Date range: {date_range.iloc[0]['min_date']} to {date_range.iloc[0]['max_date']}")
        logger.info(f"   - Unique dates: {date_range.iloc[0]['unique_dates']}")
        logger.info(f"   - Unique tickers: {ticker_info.iloc[0]['unique_tickers']}")
        logger.info(f"   - Recent tickers (2024+): {len(recent_tickers)}")
        
        return {
            'total_records': total_count,
            'date_range': date_range,
            'ticker_info': ticker_info,
            'recent_data': recent_data,
            'recent_tickers': recent_tickers
        }
    
    def check_etf_history_table(self):
        """Check etf_history table content."""
        logger.info("Checking etf_history table...")
        
        # Check total records
        count_query = "SELECT COUNT(*) as total_records FROM etf_history"
        total_count = pd.read_sql(count_query, self.engine).iloc[0]['total_records']
        
        # Check date range
        date_range_query = """
        SELECT MIN(date) as min_date, MAX(date) as max_date,
               COUNT(DISTINCT date) as unique_dates
        FROM etf_history
        """
        date_range = pd.read_sql(date_range_query, self.engine)
        
        # Check ticker coverage
        ticker_query = """
        SELECT COUNT(DISTINCT ticker) as unique_tickers,
               COUNT(DISTINCT date) as unique_dates
        FROM etf_history
        """
        ticker_info = pd.read_sql(ticker_query, self.engine)
        
        # Get unique tickers
        tickers_query = "SELECT DISTINCT ticker FROM etf_history ORDER BY ticker"
        tickers = pd.read_sql(tickers_query, self.engine)
        
        logger.info(f"✅ ETF history table analysis complete")
        logger.info(f"   - Total records: {total_count:,}")
        logger.info(f"   - Date range: {date_range.iloc[0]['min_date']} to {date_range.iloc[0]['max_date']}")
        logger.info(f"   - Unique dates: {date_range.iloc[0]['unique_dates']}")
        logger.info(f"   - Unique tickers: {ticker_info.iloc[0]['unique_tickers']}")
        logger.info(f"   - Available tickers: {list(tickers['ticker'])}")
        
        return {
            'total_records': total_count,
            'date_range': date_range,
            'ticker_info': ticker_info,
            'tickers': tickers
        }
    
    def compare_with_pickle_data(self):
        """Compare database content with pickle data."""
        logger.info("Comparing database content with pickle data...")
        
        import pickle
        
        # Load pickle data
        with open('unrestricted_universe_data.pkl', 'rb') as f:
            pickle_data = pickle.load(f)
        
        factor_data = pickle_data['factor_data']
        adtv_data = pickle_data['adtv']
        
        # Get pickle statistics
        pickle_dates = factor_data['calculation_date'].unique()
        pickle_tickers = factor_data['ticker'].unique()
        
        logger.info(f"✅ Pickle data analysis")
        logger.info(f"   - Factor data shape: {factor_data.shape}")
        logger.info(f"   - ADTV data shape: {adtv_data.shape}")
        logger.info(f"   - Date range: {min(pickle_dates)} to {max(pickle_dates)}")
        logger.info(f"   - Unique dates: {len(pickle_dates)}")
        logger.info(f"   - Unique tickers: {len(pickle_tickers)}")
        
        return {
            'factor_data_shape': factor_data.shape,
            'adtv_data_shape': adtv_data.shape,
            'date_range': (min(pickle_dates), max(pickle_dates)),
            'unique_dates': len(pickle_dates),
            'unique_tickers': len(pickle_tickers)
        }
    
    def create_database_summary_report(self, factor_results, price_results, etf_results, pickle_results):
        """Create comprehensive database summary report."""
        logger.info("Creating database summary report...")
        
        report = []
        report.append("# Database Content Analysis")
        report.append("")
        report.append("**Date:** " + datetime.now().strftime("%Y-%m-%d"))
        report.append("**Purpose:** Analyze actual database content without filters")
        report.append("**Context:** Investigation of limited database data in discrepancy analysis")
        report.append("")
        
        # Executive Summary
        report.append("## 🎯 Executive Summary")
        report.append("")
        report.append(f"- **Factor Scores Table:** {factor_results['total_records']:,} records, {factor_results['ticker_info'].iloc[0]['unique_tickers']} tickers")
        report.append(f"- **Price Data Table:** {price_results['total_records']:,} records, {price_results['ticker_info'].iloc[0]['unique_tickers']} tickers")
        report.append(f"- **ETF History Table:** {etf_results['total_records']:,} records, {etf_results['ticker_info'].iloc[0]['unique_tickers']} tickers")
        report.append(f"- **Pickle Data:** {pickle_results['factor_data_shape'][0]:,} records, {pickle_results['unique_tickers']} tickers")
        report.append("")
        
        # Detailed Analysis
        report.append("## 📊 Detailed Analysis")
        report.append("")
        
        # Factor Scores Table
        report.append("### Factor Scores Table (factor_scores_qvm)")
        report.append("")
        report.append("| Metric | Value |")
        report.append("|--------|-------|")
        report.append(f"| Total Records | {factor_results['total_records']:,} |")
        report.append(f"| Date Range | {factor_results['date_range'].iloc[0]['min_date']} to {factor_results['date_range'].iloc[0]['max_date']} |")
        report.append(f"| Unique Dates | {factor_results['date_range'].iloc[0]['unique_dates']} |")
        report.append(f"| Unique Tickers | {factor_results['ticker_info'].iloc[0]['unique_tickers']} |")
        report.append("")
        
        # Price Data Table
        report.append("### Price Data Table (vcsc_daily_data_complete)")
        report.append("")
        report.append("| Metric | Value |")
        report.append("|--------|-------|")
        report.append(f"| Total Records | {price_results['total_records']:,} |")
        report.append(f"| Date Range | {price_results['date_range'].iloc[0]['min_date']} to {price_results['date_range'].iloc[0]['max_date']} |")
        report.append(f"| Unique Dates | {price_results['date_range'].iloc[0]['unique_dates']} |")
        report.append(f"| Unique Tickers | {price_results['ticker_info'].iloc[0]['unique_tickers']} |")
        report.append("")
        
        # ETF History Table
        report.append("### ETF History Table (etf_history)")
        report.append("")
        report.append("| Metric | Value |")
        report.append("|--------|-------|")
        report.append(f"| Total Records | {etf_results['total_records']:,} |")
        report.append(f"| Date Range | {etf_results['date_range'].iloc[0]['min_date']} to {etf_results['date_range'].iloc[0]['max_date']} |")
        report.append(f"| Unique Dates | {etf_results['date_range'].iloc[0]['unique_dates']} |")
        report.append(f"| Unique Tickers | {etf_results['ticker_info'].iloc[0]['unique_tickers']} |")
        report.append(f"| Available Tickers | {list(etf_results['tickers']['ticker'])} |")
        report.append("")
        
        # Pickle Data
        report.append("### Pickle Data (unrestricted_universe_data.pkl)")
        report.append("")
        report.append("| Metric | Value |")
        report.append("|--------|-------|")
        report.append(f"| Factor Data Shape | {pickle_results['factor_data_shape']} |")
        report.append(f"| ADTV Data Shape | {pickle_results['adtv_data_shape']} |")
        report.append(f"| Date Range | {pickle_results['date_range'][0]} to {pickle_results['date_range'][1]} |")
        report.append(f"| Unique Dates | {pickle_results['unique_dates']} |")
        report.append(f"| Unique Tickers | {pickle_results['unique_tickers']} |")
        report.append("")
        
        # Comparison
        report.append("## 🔍 Comparison Analysis")
        report.append("")
        report.append("### Database vs Pickle Coverage")
        report.append("")
        report.append("| Metric | Database | Pickle | Ratio |")
        report.append("|--------|----------|--------|-------|")
        report.append(f"| Factor Records | {factor_results['total_records']:,} | {pickle_results['factor_data_shape'][0]:,} | {factor_results['total_records']/pickle_results['factor_data_shape'][0]:.2f} |")
        report.append(f"| Unique Tickers | {factor_results['ticker_info'].iloc[0]['unique_tickers']} | {pickle_results['unique_tickers']} | {factor_results['ticker_info'].iloc[0]['unique_tickers']/pickle_results['unique_tickers']:.2f} |")
        report.append(f"| Unique Dates | {factor_results['date_range'].iloc[0]['unique_dates']} | {pickle_results['unique_dates']} | {factor_results['date_range'].iloc[0]['unique_dates']/pickle_results['unique_dates']:.2f} |")
        report.append("")
        
        # Key Findings
        report.append("## 🎯 Key Findings")
        report.append("")
        
        if factor_results['total_records'] > 1000000:
            report.append("✅ **Database contains substantial data**")
            report.append(f"- Factor scores: {factor_results['total_records']:,} records")
            report.append(f"- Price data: {price_results['total_records']:,} records")
            report.append("- Previous analysis was limited by sampling filters")
        else:
            report.append("❌ **Database contains limited data**")
            report.append("- May indicate data source issues")
            report.append("- Need to investigate data pipeline")
        
        report.append("")
        
        # Recommendations
        report.append("## 📋 Recommendations")
        report.append("")
        report.append("1. **Re-run Discrepancy Analysis**")
        report.append("   - Remove sampling filters")
        report.append("   - Use full database content")
        report.append("   - Compare with pickle data properly")
        
        report.append("2. **Investigate Data Pipeline**")
        report.append("   - Check if database is production or test")
        report.append("   - Verify data loading processes")
        report.append("   - Ensure data completeness")
        
        report.append("3. **Update Backtesting Framework**")
        report.append("   - Use full database content")
        report.append("   - Apply proper filters")
        report.append("   - Validate data quality")
        
        report.append("")
        
        report_text = "\n".join(report)
        
        # Save report
        with open('database_content_analysis_report.md', 'w') as f:
            f.write(report_text)
        
        logger.info("✅ Database content analysis report saved to database_content_analysis_report.md")
        return report_text
    
    def run_complete_analysis(self):
        """Run the complete database content analysis."""
        logger.info("🚀 Starting database content analysis...")
        
        try:
            # Check all tables
            factor_results = self.check_factor_scores_table()
            price_results = self.check_price_data_table()
            etf_results = self.check_etf_history_table()
            pickle_results = self.compare_with_pickle_data()
            
            # Generate report
            report = self.create_database_summary_report(factor_results, price_results, etf_results, pickle_results)
            
            # Save results
            results = {
                'factor_results': factor_results,
                'price_results': price_results,
                'etf_results': etf_results,
                'pickle_results': pickle_results,
                'report': report
            }
            
            logger.info("✅ Complete database content analysis finished successfully!")
            logger.info("📊 Results saved to database_content_analysis_report.md")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Database content analysis failed: {e}")
            raise


def main():
    """Main execution function."""
    print("🔬 Database Content Analysis")
    print("=" * 40)
    
    # Initialize checker
    checker = DatabaseContentChecker()
    
    # Run complete analysis
    results = checker.run_complete_analysis()
    
    print("\n✅ Database content analysis completed successfully!")
    print("📊 Check database_content_analysis_report.md for detailed results.")
    
    # Print key results
    factor_results = results['factor_results']
    price_results = results['price_results']
    pickle_results = results['pickle_results']
    
    print(f"\n📈 Key Findings:")
    print(f"   Factor Scores: {factor_results['total_records']:,} records, {factor_results['ticker_info'].iloc[0]['unique_tickers']} tickers")
    print(f"   Price Data: {price_results['total_records']:,} records, {price_results['ticker_info'].iloc[0]['unique_tickers']} tickers")
    print(f"   Pickle Data: {pickle_results['factor_data_shape'][0]:,} records, {pickle_results['unique_tickers']} tickers")


if __name__ == "__main__":
    main()