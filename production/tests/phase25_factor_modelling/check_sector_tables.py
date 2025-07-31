"""
Check Sector Tables
==================
Check what sector information is available in the database.
"""

import pandas as pd
import sys
import os

# Add the parent directory to the path to import database modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'database'))
from connection import get_engine

def check_sector_information():
    """Check what sector information is available."""
    engine = get_engine()
    
    print("Checking sector information in database...")
    
    # Check what tables exist
    tables_query = """
    SHOW TABLES LIKE '%sector%'
    """
    
    try:
        tables_df = pd.read_sql(tables_query, engine)
        print("Tables with 'sector' in name:")
        print(tables_df)
    except Exception as e:
        print(f"Error checking sector tables: {e}")
    
    # Check intermediary_calculations_banking_cleaned schema
    schema_query = """
    DESCRIBE intermediary_calculations_banking_cleaned
    """
    
    try:
        schema_df = pd.read_sql(schema_query, engine)
        print("\nintermediary_calculations_banking_cleaned schema:")
        print(schema_df)
    except Exception as e:
        print(f"Error checking schema: {e}")
    
    # Check if there are other intermediary tables
    other_tables_query = """
    SHOW TABLES LIKE 'intermediary%'
    """
    
    try:
        other_tables_df = pd.read_sql(other_tables_query, engine)
        print("\nOther intermediary tables:")
        print(other_tables_df)
    except Exception as e:
        print(f"Error checking other tables: {e}")
    
    # Check if there's a separate sectors table
    sectors_table_query = """
    SHOW TABLES LIKE '%sector%'
    """
    
    try:
        sectors_df = pd.read_sql(sectors_table_query, engine)
        print("\nSector-related tables:")
        print(sectors_df)
    except Exception as e:
        print(f"Error checking sector tables: {e}")
    
    # Check if there's sector information in vcsc_daily_data_complete
    vcsc_schema_query = """
    DESCRIBE vcsc_daily_data_complete
    """
    
    try:
        vcsc_schema_df = pd.read_sql(vcsc_schema_query, engine)
        print("\nvcsc_daily_data_complete schema:")
        print(vcsc_schema_df)
    except Exception as e:
        print(f"Error checking vcsc schema: {e}")
    
    # Check if there are any other tables with sector information
    all_tables_query = """
    SHOW TABLES
    """
    
    try:
        all_tables_df = pd.read_sql(all_tables_query, engine)
        print("\nAll tables in database:")
        print(all_tables_df)
    except Exception as e:
        print(f"Error checking all tables: {e}")

if __name__ == "__main__":
    check_sector_information() 