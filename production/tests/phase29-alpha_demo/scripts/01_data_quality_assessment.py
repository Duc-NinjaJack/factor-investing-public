#!/usr/bin/env python3
"""
Data Quality Assessment for Comprehensive Multi-Factor Strategy
==============================================================

This script assesses the quality and coverage of different data sources:
1. vcsc_daily_data_complete - Primary price and market data
2. intermediary_calculations_enhanced - Non-financial fundamental data
3. intermediary_calculations_banking - Banking sector data
4. intermediary_calculations_securities - Securities sector data
5. fundamental_values - Raw fundamental data

The goal is to determine the best data sources for factor calculation with maximum coverage.
"""

# %% [markdown]
# # IMPORTS AND SETUP

# %%
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Database connectivity
from sqlalchemy import create_engine, text

# Add Project Root to Python Path
try:
    current_path = Path.cwd()
    while not (current_path / 'production').is_dir():
        if current_path.parent == current_path:
            raise FileNotFoundError("Could not find the 'production' directory.")
        current_path = current_path.parent
    
    project_root = current_path
    
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    from production.database.connection import get_database_manager
    print(f"✅ Successfully imported production modules.")
    print(f"   - Project Root set to: {project_root}")

except (ImportError, FileNotFoundError) as e:
    print(f"❌ ERROR: Could not import production modules. Please check your directory structure.")
    print(f"   - Final Path Searched: {project_root}")
    print(f"   - Error: {e}")
    raise

# %% [markdown]
# # DATABASE CONNECTION

# %%
def create_db_connection():
    """Create database connection."""
    try:
        db_manager = get_database_manager()
        engine = db_manager.get_engine()
        print("✅ Database connection established")
        return engine
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        raise

# %% [markdown]
# # DATA QUALITY ASSESSMENT FUNCTIONS

# %%
def assess_vcsc_data_coverage(engine):
    """Assess coverage of vcsc_daily_data_complete table."""
    print("\n📊 Assessing VNSC Daily Data Coverage...")
    
    # Check date range
    date_range_query = text("""
        SELECT 
            MIN(trading_date) as start_date,
            MAX(trading_date) as end_date,
            COUNT(DISTINCT trading_date) as trading_days,
            COUNT(DISTINCT ticker) as unique_tickers,
            COUNT(*) as total_records
        FROM vcsc_daily_data_complete
    """)
    
    date_range = pd.read_sql(date_range_query, engine)
    print(f"   📅 Date Range: {date_range['start_date'].iloc[0]} to {date_range['end_date'].iloc[0]}")
    print(f"   📊 Trading Days: {date_range['trading_days'].iloc[0]:,}")
    print(f"   🏢 Unique Tickers: {date_range['unique_tickers'].iloc[0]:,}")
    print(f"   📈 Total Records: {date_range['total_records'].iloc[0]:,}")
    
    # Check data completeness by year
    yearly_coverage_query = text("""
        SELECT 
            YEAR(trading_date) as year,
            COUNT(DISTINCT trading_date) as trading_days,
            COUNT(DISTINCT ticker) as unique_tickers,
            COUNT(*) as total_records,
            AVG(market_cap) as avg_market_cap,
            AVG(total_volume) as avg_volume
        FROM vcsc_daily_data_complete
        GROUP BY YEAR(trading_date)
        ORDER BY year
    """)
    
    yearly_coverage = pd.read_sql(yearly_coverage_query, engine)
    print(f"\n   📊 Yearly Coverage:")
    for _, row in yearly_coverage.iterrows():
        print(f"      {row['year']}: {row['trading_days']} days, {row['unique_tickers']} tickers, {row['total_records']:,} records")
    
    # Check key columns for data quality
    quality_query = text("""
        SELECT 
            COUNT(*) as total_records,
            COUNT(close_price_adjusted) as close_price_count,
            COUNT(market_cap) as market_cap_count,
            COUNT(total_volume) as volume_count,
            COUNT(total_value) as value_count,
            COUNT(foreign_net_value_matched) as foreign_flow_count
        FROM vcsc_daily_data_complete
        WHERE trading_date >= '2016-01-01'
    """)
    
    quality = pd.read_sql(quality_query, engine)
    print(f"\n   🔍 Data Quality (2016+):")
    print(f"      Close Price: {quality['close_price_count'].iloc[0]:,} / {quality['total_records'].iloc[0]:,} ({quality['close_price_count'].iloc[0]/quality['total_records'].iloc[0]*100:.1f}%)")
    print(f"      Market Cap: {quality['market_cap_count'].iloc[0]:,} / {quality['total_records'].iloc[0]:,} ({quality['market_cap_count'].iloc[0]/quality['total_records'].iloc[0]*100:.1f}%)")
    print(f"      Volume: {quality['volume_count'].iloc[0]:,} / {quality['total_records'].iloc[0]:,} ({quality['volume_count'].iloc[0]/quality['total_records'].iloc[0]*100:.1f}%)")
    print(f"      Value: {quality['value_count'].iloc[0]:,} / {quality['total_records'].iloc[0]:,} ({quality['value_count'].iloc[0]/quality['total_records'].iloc[0]*100:.1f}%)")
    print(f"      Foreign Flow: {quality['foreign_flow_count'].iloc[0]:,} / {quality['total_records'].iloc[0]:,} ({quality['foreign_flow_count'].iloc[0]/quality['total_records'].iloc[0]*100:.1f}%)")
    
    return {
        'date_range': date_range,
        'yearly_coverage': yearly_coverage,
        'quality': quality
    }

def assess_intermediary_table_coverage(engine, table_name):
    """Assess coverage of intermediary tables."""
    print(f"\n📊 Assessing {table_name} Coverage...")
    
    # Check date range
    date_range_query = text(f"""
        SELECT 
            MIN(year) as start_year,
            MAX(year) as end_year,
            COUNT(DISTINCT year) as years,
            COUNT(DISTINCT ticker) as unique_tickers,
            COUNT(*) as total_records
        FROM {table_name}
    """)
    
    date_range = pd.read_sql(date_range_query, engine)
    print(f"   📅 Year Range: {date_range['start_year'].iloc[0]} to {date_range['end_year'].iloc[0]}")
    print(f"   📊 Years: {date_range['years'].iloc[0]}")
    print(f"   🏢 Unique Tickers: {date_range['unique_tickers'].iloc[0]:,}")
    print(f"   📈 Total Records: {date_range['total_records'].iloc[0]:,}")
    
    # Check yearly coverage
    yearly_coverage_query = text(f"""
        SELECT 
            year,
            COUNT(DISTINCT ticker) as unique_tickers,
            COUNT(*) as total_records
        FROM {table_name}
        GROUP BY year
        ORDER BY year
    """)
    
    yearly_coverage = pd.read_sql(yearly_coverage_query, engine)
    print(f"\n   📊 Yearly Coverage:")
    for _, row in yearly_coverage.iterrows():
        print(f"      {row['year']}: {row['unique_tickers']} tickers, {row['total_records']} records")
    
    # Check key metrics availability
    if table_name == 'intermediary_calculations_enhanced':
        metrics_query = text(f"""
            SELECT 
                COUNT(*) as total_records,
                COUNT(NetProfit_TTM) as net_profit_count,
                COUNT(Revenue_TTM) as revenue_count,
                COUNT(AvgTotalAssets) as assets_count,
                COUNT(FCF_TTM) as fcf_count,
                COUNT(AvgTotalDebt) as debt_count
            FROM {table_name}
            WHERE year >= 2016
        """)
    elif table_name == 'intermediary_calculations_banking':
        metrics_query = text(f"""
            SELECT 
                COUNT(*) as total_records,
                COUNT(NetProfit_TTM) as net_profit_count,
                COUNT(TotalOperatingIncome_TTM) as revenue_count,
                COUNT(AvgTotalAssets) as assets_count,
                COUNT(ROAA) as roaa_count,
                COUNT(NIM) as nim_count
            FROM {table_name}
            WHERE year >= 2016
        """)
    elif table_name == 'intermediary_calculations_securities':
        metrics_query = text(f"""
            SELECT 
                COUNT(*) as total_records,
                COUNT(NetProfit_TTM) as net_profit_count,
                COUNT(TotalOperatingRevenue_TTM) as revenue_count,
                COUNT(AvgTotalAssets) as assets_count,
                COUNT(ROAA) as roaa_count,
                COUNT(BrokerageRatio) as brokerage_count
            FROM {table_name}
            WHERE year >= 2016
        """)
    
    metrics = pd.read_sql(metrics_query, engine)
    print(f"\n   🔍 Key Metrics Availability (2016+):")
    for col in metrics.columns:
        if col != 'total_records':
            count = metrics[col].iloc[0]
            total = metrics['total_records'].iloc[0]
            percentage = count/total*100 if total > 0 else 0
            print(f"      {col}: {count:,} / {total:,} ({percentage:.1f}%)")
    
    return {
        'date_range': date_range,
        'yearly_coverage': yearly_coverage,
        'metrics': metrics
    }

def assess_fundamental_values_coverage(engine):
    """Assess coverage of raw fundamental_values table."""
    print("\n📊 Assessing Fundamental Values Coverage...")
    
    # Check date range
    date_range_query = text("""
        SELECT 
            MIN(year) as start_year,
            MAX(year) as end_year,
            COUNT(DISTINCT year) as years,
            COUNT(DISTINCT ticker) as unique_tickers,
            COUNT(DISTINCT item_id) as unique_items,
            COUNT(*) as total_records
        FROM fundamental_values
    """)
    
    date_range = pd.read_sql(date_range_query, engine)
    print(f"   📅 Year Range: {date_range['start_year'].iloc[0]} to {date_range['end_year'].iloc[0]}")
    print(f"   📊 Years: {date_range['years'].iloc[0]}")
    print(f"   🏢 Unique Tickers: {date_range['unique_tickers'].iloc[0]:,}")
    print(f"   📋 Unique Items: {date_range['unique_items'].iloc[0]:,}")
    print(f"   📈 Total Records: {date_range['total_records'].iloc[0]:,}")
    
    # Check yearly coverage
    yearly_coverage_query = text("""
        SELECT 
            year,
            COUNT(DISTINCT ticker) as unique_tickers,
            COUNT(DISTINCT item_id) as unique_items,
            COUNT(*) as total_records
        FROM fundamental_values
        GROUP BY year
        ORDER BY year
    """)
    
    yearly_coverage = pd.read_sql(yearly_coverage_query, engine)
    print(f"\n   📊 Yearly Coverage:")
    for _, row in yearly_coverage.iterrows():
        print(f"      {row['year']}: {row['unique_tickers']} tickers, {row['unique_items']} items, {row['total_records']:,} records")
    
    # Check statement type distribution
    statement_query = text("""
        SELECT 
            statement_type,
            COUNT(DISTINCT ticker) as unique_tickers,
            COUNT(DISTINCT item_id) as unique_items,
            COUNT(*) as total_records
        FROM fundamental_values
        WHERE year >= 2016
        GROUP BY statement_type
    """)
    
    statement_dist = pd.read_sql(statement_query, engine)
    print(f"\n   📋 Statement Type Distribution (2016+):")
    for _, row in statement_dist.iterrows():
        print(f"      {row['statement_type']}: {row['unique_tickers']} tickers, {row['unique_items']} items, {row['total_records']:,} records")
    
    return {
        'date_range': date_range,
        'yearly_coverage': yearly_coverage,
        'statement_dist': statement_dist
    }

# %% [markdown]
# # MAIN ASSESSMENT EXECUTION

# %%
def main():
    """Run comprehensive data quality assessment."""
    print("🔍 COMPREHENSIVE DATA QUALITY ASSESSMENT")
    print("=" * 60)
    
    try:
        # Connect to database
        engine = create_db_connection()
        
        # Assess VNSC data coverage
        vcsc_data = assess_vcsc_data_coverage(engine)
        
        # Assess intermediary tables
        enhanced_data = assess_intermediary_table_coverage(engine, 'intermediary_calculations_enhanced')
        banking_data = assess_intermediary_table_coverage(engine, 'intermediary_calculations_banking')
        securities_data = assess_intermediary_table_coverage(engine, 'intermediary_calculations_securities')
        
        # Assess raw fundamental data
        fundamental_data = assess_fundamental_values_coverage(engine)
        
        # Generate summary report
        print("\n" + "=" * 60)
        print("📋 DATA QUALITY ASSESSMENT SUMMARY")
        print("=" * 60)
        
        print(f"\n📊 VNSC Daily Data:")
        print(f"   - Coverage: {vcsc_data['date_range']['start_date'].iloc[0]} to {vcsc_data['date_range']['end_date'].iloc[0]}")
        print(f"   - Tickers: {vcsc_data['date_range']['unique_tickers'].iloc[0]:,}")
        print(f"   - Records: {vcsc_data['date_range']['total_records'].iloc[0]:,}")
        
        print(f"\n📊 Intermediary Tables:")
        print(f"   - Enhanced: {enhanced_data['date_range']['start_year'].iloc[0]}-{enhanced_data['date_range']['end_year'].iloc[0]} ({enhanced_data['date_range']['unique_tickers'].iloc[0]:,} tickers)")
        print(f"   - Banking: {banking_data['date_range']['start_year'].iloc[0]}-{banking_data['date_range']['end_year'].iloc[0]} ({banking_data['date_range']['unique_tickers'].iloc[0]:,} tickers)")
        print(f"   - Securities: {securities_data['date_range']['start_year'].iloc[0]}-{securities_data['date_range']['end_year'].iloc[0]} ({securities_data['date_range']['unique_tickers'].iloc[0]:,} tickers)")
        
        print(f"\n📊 Raw Fundamental Data:")
        print(f"   - Coverage: {fundamental_data['date_range']['start_year'].iloc[0]}-{fundamental_data['date_range']['end_year'].iloc[0]}")
        print(f"   - Tickers: {fundamental_data['date_range']['unique_tickers'].iloc[0]:,}")
        print(f"   - Items: {fundamental_data['date_range']['unique_items'].iloc[0]:,}")
        print(f"   - Records: {fundamental_data['date_range']['total_records'].iloc[0]:,}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"   1. Use VNSC daily data for price/market data (best coverage)")
        print(f"   2. Use raw fundamental_values for factor calculation (most comprehensive)")
        print(f"   3. Calculate metrics directly from raw data for maximum coverage")
        print(f"   4. Implement sector-specific logic for different financial statements")
        
        print(f"\n✅ Data quality assessment complete!")
        
    except Exception as e:
        print(f"❌ Error during assessment: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
