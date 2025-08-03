#!/usr/bin/env python3
"""
Test script to check database schema and available data
"""

import sys
import os
import pandas as pd

# Add the necessary paths to import modules
sys.path.append(os.path.join(os.path.dirname('__file__'), '..', '..', 'engine'))

from qvm_engine_v2_enhanced import QVMEngineV2Enhanced

def main():
    print("Testing database schema...")
    
    # Initialize engine
    engine = QVMEngineV2Enhanced()
    print("✅ Engine initialized")
    
    # Check available tables
    print("\n1. Checking available tables...")
    try:
        tables_query = "SHOW TABLES LIKE '%intermediary%'"
        tables_result = pd.read_sql(tables_query, engine.engine)
        print("Available intermediary tables:")
        for table in tables_result.iloc[:, 0]:
            print(f"  - {table}")
    except Exception as e:
        print(f"Error checking tables: {e}")
    
    # Check vcsc_daily_data_complete columns
    print("\n2. Checking vcsc_daily_data_complete columns...")
    try:
        schema_query = "DESCRIBE vcsc_daily_data_complete"
        schema_result = pd.read_sql(schema_query, engine.engine)
        print(f"Total columns: {len(schema_result)}")
        print("Sample columns:")
        for i, col in enumerate(schema_result['Field'].tolist()[:20]):
            print(f"  {i+1:2d}. {col}")
    except Exception as e:
        print(f"Error checking vcsc_daily_data_complete: {e}")
    
    # Check if intermediary tables exist
    print("\n3. Checking intermediary tables...")
    intermediary_tables = ['intermediary_calculations_enhanced', 'intermediary_calculations_banking', 'intermediary_calculations_securities']
    
    for table in intermediary_tables:
        try:
            check_query = f"SELECT COUNT(*) as count FROM {table} LIMIT 1"
            check_result = pd.read_sql(check_query, engine.engine)
            print(f"  ✅ {table}: {check_result.iloc[0]['count']} records available")
            
            # Check columns in this table
            table_schema_query = f"DESCRIBE {table}"
            table_schema = pd.read_sql(table_schema_query, engine.engine)
            print(f"    Columns: {len(table_schema)}")
            
            # Show all columns
            print(f"    All columns in {table}:")
            for i, col in enumerate(table_schema['Field'].tolist()):
                print(f"      {i+1:2d}. {col}")
            
            # Show some key columns
            key_columns = ['ticker', 'year', 'quarter', 'NetProfit_TTM', 'Revenue_TTM', 'AvgTotalAssets']
            available_key_columns = [col for col in key_columns if col in table_schema['Field'].tolist()]
            print(f"    Key columns available: {available_key_columns}")
            
        except Exception as e:
            print(f"  ❌ {table}: {str(e)[:50]}...")

if __name__ == "__main__":
    main() 