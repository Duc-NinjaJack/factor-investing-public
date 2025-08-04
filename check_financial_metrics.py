#!/usr/bin/env python3
"""
Check financial metrics data availability.
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import sys
import os

# Add project root to path
project_root = "/Users/raymond/Documents/Projects/factor-investing-public"
sys.path.append(project_root)

from production.database.connection import DatabaseManager

def create_db_connection():
    """Create database connection."""
    try:
        db_manager = DatabaseManager()
        engine = db_manager.get_engine()
        print("✅ Database connection established successfully.")
        return engine
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return None

def check_financial_metrics_data(engine):
    """Check what financial metrics data is available."""
    print("\n🔍 Checking financial metrics data availability...")
    
    # Check table structure
    structure_query = text("""
        DESCRIBE financial_metrics
    """)
    
    try:
        structure = pd.read_sql(structure_query, engine)
        print("   - Table structure:")
        print(structure.to_string())
    except Exception as e:
        print(f"   ❌ Error checking table structure: {e}")
        return
    
    # Check data availability by year
    year_query = text("""
        SELECT 
            YEAR(Date) as year,
            COUNT(*) as total_records,
            COUNT(CASE WHEN PE IS NOT NULL AND PE > 0 THEN 1 END) as pe_records,
            COUNT(CASE WHEN PB IS NOT NULL AND PB > 0 THEN 1 END) as pb_records,
            COUNT(CASE WHEN EPS IS NOT NULL THEN 1 END) as eps_records,
            COUNT(CASE WHEN MarketCapitalization IS NOT NULL AND MarketCapitalization > 0 THEN 1 END) as market_cap_records
        FROM financial_metrics 
        GROUP BY YEAR(Date)
        ORDER BY year
    """)
    
    try:
        year_data = pd.read_sql(year_query, engine)
        print("\n   - Data availability by year:")
        print(year_data.to_string())
    except Exception as e:
        print(f"   ❌ Error checking year data: {e}")
    
    # Check sample data for recent years
    sample_query = text("""
        SELECT 
            ticker,
            Date,
            PE,
            PB,
            EPS,
            MarketCapitalization / 1e9 as market_cap_bn
        FROM financial_metrics 
        WHERE Date >= '2020-01-01'
        AND PE IS NOT NULL AND PE > 0
        ORDER BY Date DESC, ticker
        LIMIT 20
    """)
    
    try:
        sample_data = pd.read_sql(sample_query, engine)
        print(f"\n   - Sample recent data (PE > 0):")
        print(sample_data.to_string())
    except Exception as e:
        print(f"   ❌ Error checking sample data: {e}")
    
    # Check what tickers have data
    ticker_query = text("""
        SELECT 
            ticker,
            COUNT(*) as total_records,
            COUNT(CASE WHEN PE IS NOT NULL AND PE > 0 THEN 1 END) as pe_records,
            MIN(Date) as first_date,
            MAX(Date) as last_date
        FROM financial_metrics 
        GROUP BY ticker
        HAVING COUNT(CASE WHEN PE IS NOT NULL AND PE > 0 THEN 1 END) > 0
        ORDER BY pe_records DESC
        LIMIT 10
    """)
    
    try:
        ticker_data = pd.read_sql(ticker_query, engine)
        print(f"\n   - Top tickers with P/E data:")
        print(ticker_data.to_string())
    except Exception as e:
        print(f"   ❌ Error checking ticker data: {e}")

def check_alternative_data_sources(engine):
    """Check if there are alternative sources for P/E data."""
    print("\n🔍 Checking alternative data sources...")
    
    # Check if there's a different table with P/E data
    tables_query = text("""
        SHOW TABLES LIKE '%pe%'
    """)
    
    try:
        pe_tables = pd.read_sql(tables_query, engine)
        print("   - Tables with 'pe' in name:")
        print(pe_tables.to_string())
    except Exception as e:
        print(f"   ❌ Error checking PE tables: {e}")
    
    # Check if there's a different table with valuation data
    valuation_tables_query = text("""
        SHOW TABLES LIKE '%valuation%'
    """)
    
    try:
        valuation_tables = pd.read_sql(valuation_tables_query, engine)
        print("   - Tables with 'valuation' in name:")
        print(valuation_tables.to_string())
    except Exception as e:
        print(f"   ❌ Error checking valuation tables: {e}")
    
    # Check if there's a different table with market data
    market_tables_query = text("""
        SHOW TABLES LIKE '%market%'
    """)
    
    try:
        market_tables = pd.read_sql(market_tables_query, engine)
        print("   - Tables with 'market' in name:")
        print(market_tables.to_string())
    except Exception as e:
        print(f"   ❌ Error checking market tables: {e}")

def main():
    """Main function."""
    print("🔍 FINANCIAL METRICS DATA CHECK")
    print("=" * 50)
    
    # Create database connection
    engine = create_db_connection()
    if not engine:
        return
    
    # Check financial metrics data
    check_financial_metrics_data(engine)
    
    # Check alternative sources
    check_alternative_data_sources(engine)

if __name__ == "__main__":
    main() 