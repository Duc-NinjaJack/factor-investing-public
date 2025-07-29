#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check Benchmark Data
===================
Purpose: Check what benchmark data is available
"""

import yaml
from sqlalchemy import create_engine
import pandas as pd

def check_benchmark_data():
    """Check what benchmark data is available."""
    
    # Load database config
    with open('../../../config/database.yml', 'r') as file:
        config = yaml.safe_load(file)
    
    db_config = config['production']
    connection_string = (
        f"mysql+pymysql://{db_config['username']}:{db_config['password']}"
        f"@{db_config['host']}/{db_config['schema_name']}"
    )
    engine = create_engine(connection_string, pool_recycle=3600)
    
    print("🔍 Checking Benchmark Data...")
    print("=" * 50)
    
    # Check for index/ETF data in vcsc_daily_data_complete
    print("\n📊 Checking vcsc_daily_data_complete for index/ETF data:")
    try:
        # Check for any ticker containing 'INDEX'
        result = pd.read_sql("SELECT DISTINCT ticker FROM vcsc_daily_data_complete WHERE ticker LIKE '%INDEX%' LIMIT 10", engine)
        print(f"INDEX tickers: {result['ticker'].tolist()}")
        
        # Check for any ticker containing 'ETF'
        result = pd.read_sql("SELECT DISTINCT ticker FROM vcsc_daily_data_complete WHERE ticker LIKE '%ETF%' LIMIT 10", engine)
        print(f"ETF tickers: {result['ticker'].tolist()}")
        
        # Check for VNINDEX specifically
        result = pd.read_sql("SELECT COUNT(*) as count FROM vcsc_daily_data_complete WHERE ticker = 'VNINDEX'", engine)
        print(f"VNINDEX records: {result['count'].iloc[0]}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    # Check etf_history table
    print("\n📊 Checking etf_history table:")
    try:
        result = pd.read_sql("SELECT DISTINCT ticker FROM etf_history LIMIT 10", engine)
        print(f"ETF tickers: {result['ticker'].tolist()}")
        
        # Check for VNINDEX
        result = pd.read_sql("SELECT COUNT(*) as count FROM etf_history WHERE ticker = 'VNINDEX'", engine)
        print(f"VNINDEX records: {result['count'].iloc[0]}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    # Check equity_history table
    print("\n📊 Checking equity_history table:")
    try:
        result = pd.read_sql("SELECT DISTINCT ticker FROM equity_history WHERE ticker LIKE '%INDEX%' OR ticker LIKE '%ETF%' LIMIT 10", engine)
        print(f"Index/ETF tickers: {result['ticker'].tolist()}")
        
        # Check for VNINDEX
        result = pd.read_sql("SELECT COUNT(*) as count FROM equity_history WHERE ticker = 'VNINDEX'", engine)
        print(f"VNINDEX records: {result['count'].iloc[0]}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_benchmark_data()