"""
Check Intermediary Sectors
=========================
Check what sector information is available in different intermediary tables.
"""

import pandas as pd
import sys
import os

# Add the parent directory to the path to import database modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'database'))
from connection import get_engine

def check_intermediary_sectors():
    """Check sector information in different intermediary tables."""
    engine = get_engine()
    
    print("Checking sector information in intermediary tables...")
    
    # List of intermediary tables to check
    intermediary_tables = [
        'intermediary_calculations_banking_cleaned',
        'intermediary_calculations_securities_cleaned',
        'intermediary_calculations_enhanced'
    ]
    
    for table in intermediary_tables:
        print(f"\n{'='*60}")
        print(f"Checking table: {table}")
        print(f"{'='*60}")
        
        # Check schema
        schema_query = f"DESCRIBE {table}"
        
        try:
            schema_df = pd.read_sql(schema_query, engine)
            print(f"\nSchema for {table}:")
            print(schema_df)
            
            # Check if there's a sector column
            if 'sector' in schema_df['Field'].values:
                print(f"\n✅ {table} has sector column!")
                
                # Get sample sector data
                sector_query = f"SELECT DISTINCT sector FROM {table} WHERE sector IS NOT NULL AND sector != '' LIMIT 10"
                sector_df = pd.read_sql(sector_query, engine)
                print(f"\nSample sectors in {table}:")
                print(sector_df)
                
                # Count sectors
                count_query = f"SELECT COUNT(DISTINCT sector) as sector_count FROM {table} WHERE sector IS NOT NULL AND sector != ''"
                count_df = pd.read_sql(count_query, engine)
                print(f"\nTotal unique sectors in {table}: {count_df['sector_count'].iloc[0]}")
                
            else:
                print(f"\n❌ {table} does not have sector column")
                
                # Check what columns might indicate sector
                sector_like_columns = schema_df[schema_df['Field'].str.contains('sector|industry|type', case=False, na=False)]
                if len(sector_like_columns) > 0:
                    print(f"\nPotential sector-related columns in {table}:")
                    print(sector_like_columns)
                
        except Exception as e:
            print(f"Error checking {table}: {e}")
    
    # Check if there's a separate sector mapping table
    print(f"\n{'='*60}")
    print("Checking for sector mapping tables...")
    print(f"{'='*60}")
    
    sector_tables_query = "SHOW TABLES LIKE '%sector%'"
    
    try:
        sector_tables_df = pd.read_sql(sector_tables_query, engine)
        print("\nSector-related tables:")
        print(sector_tables_df)
        
        if len(sector_tables_df) > 0:
            for _, row in sector_tables_df.iterrows():
                table_name = row.iloc[0]
                print(f"\nChecking table: {table_name}")
                
                # Get sample data
                sample_query = f"SELECT * FROM {table_name} LIMIT 5"
                sample_df = pd.read_sql(sample_query, engine)
                print(f"Sample data from {table_name}:")
                print(sample_df)
                
    except Exception as e:
        print(f"Error checking sector tables: {e}")
    
    # Check master_info table for sector information
    print(f"\n{'='*60}")
    print("Checking master_info table...")
    print(f"{'='*60}")
    
    master_info_schema_query = "DESCRIBE master_info"
    
    try:
        master_schema_df = pd.read_sql(master_info_schema_query, engine)
        print("\nMaster info schema:")
        print(master_schema_df)
        
        # Check if there's sector information
        if 'sector' in master_schema_df['Field'].values:
            print("\n✅ master_info has sector column!")
            
            # Get sample sector data
            master_sector_query = "SELECT DISTINCT sector FROM master_info WHERE sector IS NOT NULL AND sector != '' LIMIT 10"
            master_sector_df = pd.read_sql(master_sector_query, engine)
            print("\nSample sectors in master_info:")
            print(master_sector_df)
            
        else:
            print("\n❌ master_info does not have sector column")
            
    except Exception as e:
        print(f"Error checking master_info: {e}")

if __name__ == "__main__":
    check_intermediary_sectors() 