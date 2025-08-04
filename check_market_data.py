#!/usr/bin/env python3
"""
Check market data tables for P/E information.
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

def check_equity_market_data(engine):
    """Check equity_market_data table."""
    print("\n🔍 Checking equity_market_data table...")
    
    # Check structure
    structure_query = text("""
        DESCRIBE equity_market_data
    """)
    
    try:
        structure = pd.read_sql(structure_query, engine)
        print("   - Table structure:")
        print(structure.to_string())
    except Exception as e:
        print(f"   ❌ Error checking structure: {e}")
        return
    
    # Check data availability
    data_query = text("""
        SELECT 
            COUNT(*) as total_records,
            COUNT(DISTINCT ticker) as unique_tickers,
            MIN(date) as first_date,
            MAX(date) as last_date
        FROM equity_market_data
    """)
    
    try:
        data_info = pd.read_sql(data_query, engine)
        print("\n   - Data overview:")
        print(data_info.to_string())
    except Exception as e:
        print(f"   ❌ Error checking data: {e}")
    
    # Check sample data
    sample_query = text("""
        SELECT *
        FROM equity_market_data
        WHERE date >= '2020-01-01'
        ORDER BY date DESC, ticker
        LIMIT 10
    """)
    
    try:
        sample_data = pd.read_sql(sample_query, engine)
        print(f"\n   - Sample data:")
        print(sample_data.to_string())
    except Exception as e:
        print(f"   ❌ Error checking sample: {e}")

def check_historical_market_cap(engine):
    """Check historical_daily_market_cap table."""
    print("\n🔍 Checking historical_daily_market_cap table...")
    
    # Check structure
    structure_query = text("""
        DESCRIBE historical_daily_market_cap
    """)
    
    try:
        structure = pd.read_sql(structure_query, engine)
        print("   - Table structure:")
        print(structure.to_string())
    except Exception as e:
        print(f"   ❌ Error checking structure: {e}")
        return
    
    # Check data availability
    data_query = text("""
        SELECT 
            COUNT(*) as total_records,
            COUNT(DISTINCT ticker) as unique_tickers,
            MIN(date) as first_date,
            MAX(date) as last_date
        FROM historical_daily_market_cap
    """)
    
    try:
        data_info = pd.read_sql(data_query, engine)
        print("\n   - Data overview:")
        print(data_info.to_string())
    except Exception as e:
        print(f"   ❌ Error checking data: {e}")
    
    # Check sample data
    sample_query = text("""
        SELECT *
        FROM historical_daily_market_cap
        WHERE date >= '2020-01-01'
        ORDER BY date DESC, ticker
        LIMIT 10
    """)
    
    try:
        sample_data = pd.read_sql(sample_query, engine)
        print(f"\n   - Sample data:")
        print(sample_data.to_string())
    except Exception as e:
        print(f"   ❌ Error checking sample: {e}")

def check_precalculated_market_cap(engine):
    """Check precalculated_quarterly_market_cap table."""
    print("\n🔍 Checking precalculated_quarterly_market_cap table...")
    
    # Check structure
    structure_query = text("""
        DESCRIBE precalculated_quarterly_market_cap
    """)
    
    try:
        structure = pd.read_sql(structure_query, engine)
        print("   - Table structure:")
        print(structure.to_string())
    except Exception as e:
        print(f"   ❌ Error checking structure: {e}")
        return
    
    # Check data availability
    data_query = text("""
        SELECT 
            COUNT(*) as total_records,
            COUNT(DISTINCT ticker) as unique_tickers,
            MIN(date) as first_date,
            MAX(date) as last_date
        FROM precalculated_quarterly_market_cap
    """)
    
    try:
        data_info = pd.read_sql(data_query, engine)
        print("\n   - Data overview:")
        print(data_info.to_string())
    except Exception as e:
        print(f"   ❌ Error checking data: {e}")
    
    # Check sample data
    sample_query = text("""
        SELECT *
        FROM precalculated_quarterly_market_cap
        WHERE date >= '2020-01-01'
        ORDER BY date DESC, ticker
        LIMIT 10
    """)
    
    try:
        sample_data = pd.read_sql(sample_query, engine)
        print(f"\n   - Sample data:")
        print(sample_data.to_string())
    except Exception as e:
        print(f"   ❌ Error checking sample: {e}")

def check_equity_history_with_market_cap(engine):
    """Check equity_history_with_market_cap table."""
    print("\n🔍 Checking equity_history_with_market_cap table...")
    
    # Check structure
    structure_query = text("""
        DESCRIBE equity_history_with_market_cap
    """)
    
    try:
        structure = pd.read_sql(structure_query, engine)
        print("   - Table structure:")
        print(structure.to_string())
    except Exception as e:
        print(f"   ❌ Error checking structure: {e}")
        return
    
    # Check data availability
    data_query = text("""
        SELECT 
            COUNT(*) as total_records,
            COUNT(DISTINCT ticker) as unique_tickers,
            MIN(date) as first_date,
            MAX(date) as last_date
        FROM equity_history_with_market_cap
    """)
    
    try:
        data_info = pd.read_sql(data_query, engine)
        print("\n   - Data overview:")
        print(data_info.to_string())
    except Exception as e:
        print(f"   ❌ Error checking data: {e}")
    
    # Check sample data
    sample_query = text("""
        SELECT *
        FROM equity_history_with_market_cap
        WHERE date >= '2020-01-01'
        ORDER BY date DESC, ticker
        LIMIT 10
    """)
    
    try:
        sample_data = pd.read_sql(sample_query, engine)
        print(f"\n   - Sample data:")
        print(sample_data.to_string())
    except Exception as e:
        print(f"   ❌ Error checking sample: {e}")

def main():
    """Main function."""
    print("🔍 MARKET DATA TABLES CHECK")
    print("=" * 50)
    
    # Create database connection
    engine = create_db_connection()
    if not engine:
        return
    
    # Check all market data tables
    check_equity_market_data(engine)
    check_historical_market_cap(engine)
    check_precalculated_market_cap(engine)
    check_equity_history_with_market_cap(engine)

if __name__ == "__main__":
    main() 