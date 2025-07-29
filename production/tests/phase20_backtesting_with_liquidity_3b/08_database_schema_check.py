#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Database Schema Check
====================
Purpose: Check actual database schema to understand available columns
"""

import yaml
from sqlalchemy import create_engine, text
import pandas as pd

def check_database_schema():
    """Check the actual database schema."""
    
    # Load database config
    with open('../../../config/database.yml', 'r') as file:
        config = yaml.safe_load(file)
    
    db_config = config['production']
    connection_string = (
        f"mysql+pymysql://{db_config['username']}:{db_config['password']}"
        f"@{db_config['host']}/{db_config['schema_name']}"
    )
    engine = create_engine(connection_string, pool_recycle=3600)
    
    print("🔍 Checking Database Schema...")
    print("=" * 50)
    
    # Check equity_history table
    print("\n📊 equity_history table:")
    try:
        with engine.connect() as conn:
            result = conn.execute(text("DESCRIBE equity_history"))
            columns = [row[0] for row in result]
            print(f"Columns: {columns}")
        
        # Check sample data
        sample = pd.read_sql("SELECT * FROM equity_history LIMIT 5", engine)
        print(f"Sample data shape: {sample.shape}")
        print(f"Sample columns: {sample.columns.tolist()}")
        
    except Exception as e:
        print(f"Error with equity_history: {e}")
    
    # Check etf_history table
    print("\n📊 etf_history table:")
    try:
        with engine.connect() as conn:
            result = conn.execute(text("DESCRIBE etf_history"))
            columns = [row[0] for row in result]
            print(f"Columns: {columns}")
        
        # Check sample data
        sample = pd.read_sql("SELECT * FROM etf_history LIMIT 5", engine)
        print(f"Sample data shape: {sample.shape}")
        print(f"Sample columns: {sample.columns.tolist()}")
        
    except Exception as e:
        print(f"Error with etf_history: {e}")
    
    # Check vcsc_daily_data_complete table
    print("\n📊 vcsc_daily_data_complete table:")
    try:
        with engine.connect() as conn:
            result = conn.execute(text("DESCRIBE vcsc_daily_data_complete"))
            columns = [row[0] for row in result]
            print(f"Columns: {columns}")
        
        # Check sample data
        sample = pd.read_sql("SELECT * FROM vcsc_daily_data_complete LIMIT 5", engine)
        print(f"Sample data shape: {sample.shape}")
        print(f"Sample columns: {sample.columns.tolist()}")
        
    except Exception as e:
        print(f"Error with vcsc_daily_data_complete: {e}")
    
    # Check factor_scores_qvm table
    print("\n📊 factor_scores_qvm table:")
    try:
        with engine.connect() as conn:
            result = conn.execute(text("DESCRIBE factor_scores_qvm"))
            columns = [row[0] for row in result]
            print(f"Columns: {columns}")
        
        # Check sample data
        sample = pd.read_sql("SELECT * FROM factor_scores_qvm LIMIT 5", engine)
        print(f"Sample data shape: {sample.shape}")
        print(f"Sample columns: {sample.columns.tolist()}")
        
    except Exception as e:
        print(f"Error with factor_scores_qvm: {e}")

if __name__ == "__main__":
    check_database_schema()