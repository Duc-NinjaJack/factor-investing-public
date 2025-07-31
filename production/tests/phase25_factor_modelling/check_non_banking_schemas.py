"""
Check Non-Banking Schemas
=========================
Check the actual schemas of enhanced and securities tables to see what columns are available for non-banking sectors.
"""

import pandas as pd
import sys
import os

# Add the parent directory to the path to import database modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'database'))
from connection import get_engine

def check_non_banking_schemas():
    """Check schemas of enhanced and securities tables for non-banking sectors."""
    engine = get_engine()
    
    print("Checking schemas for non-banking sector analysis...")
    
    # Check enhanced table schema
    print(f"\n{'='*60}")
    print("INTERMEDIARY_CALCULATIONS_ENHANCED SCHEMA")
    print(f"{'='*60}")
    
    enhanced_schema_query = "DESCRIBE intermediary_calculations_enhanced"
    
    try:
        enhanced_schema_df = pd.read_sql(enhanced_schema_query, engine)
        print("\nEnhanced table schema:")
        print(enhanced_schema_df)
        
        # Look for profit-related columns
        profit_columns = enhanced_schema_df[enhanced_schema_df['Field'].str.contains('profit|income|revenue|earnings', case=False, na=False)]
        print(f"\nProfit-related columns in enhanced table:")
        print(profit_columns)
        
        # Look for asset/equity columns
        asset_columns = enhanced_schema_df[enhanced_schema_df['Field'].str.contains('asset|equity|capital', case=False, na=False)]
        print(f"\nAsset/Equity-related columns in enhanced table:")
        print(asset_columns)
        
    except Exception as e:
        print(f"Error checking enhanced schema: {e}")
    
    # Check securities table schema
    print(f"\n{'='*60}")
    print("INTERMEDIARY_CALCULATIONS_SECURITIES_CLEANED SCHEMA")
    print(f"{'='*60}")
    
    securities_schema_query = "DESCRIBE intermediary_calculations_securities_cleaned"
    
    try:
        securities_schema_df = pd.read_sql(securities_schema_query, engine)
        print("\nSecurities table schema:")
        print(securities_schema_df)
        
        # Look for profit-related columns
        profit_columns = securities_schema_df[securities_schema_df['Field'].str.contains('profit|income|revenue|earnings', case=False, na=False)]
        print(f"\nProfit-related columns in securities table:")
        print(profit_columns)
        
        # Look for asset/equity columns
        asset_columns = securities_schema_df[securities_schema_df['Field'].str.contains('asset|equity|capital', case=False, na=False)]
        print(f"\nAsset/Equity-related columns in securities table:")
        print(asset_columns)
        
    except Exception as e:
        print(f"Error checking securities schema: {e}")
    
    # Check what sectors have data in enhanced table
    print(f"\n{'='*60}")
    print("CHECKING SECTOR DATA AVAILABILITY")
    print(f"{'='*60}")
    
    # Get sample data from enhanced table
    enhanced_sample_query = """
    SELECT ticker, year, quarter, calc_date
    FROM intermediary_calculations_enhanced
    WHERE quarter = (
        SELECT MAX(quarter) 
        FROM intermediary_calculations_enhanced
    )
    LIMIT 10
    """
    
    try:
        enhanced_sample_df = pd.read_sql(enhanced_sample_query, engine)
        print("\nSample data from enhanced table:")
        print(enhanced_sample_df)
        
        # Check if these tickers have sector info
        if len(enhanced_sample_df) > 0:
            tickers = enhanced_sample_df['ticker'].tolist()
            ticker_list = ','.join([f"'{ticker}'" for ticker in tickers])
            
            sector_query = f"""
            SELECT ticker, sector, industry
            FROM master_info
            WHERE ticker IN ({ticker_list})
            """
            
            sector_df = pd.read_sql(sector_query, engine)
            print(f"\nSector information for sample tickers:")
            print(sector_df)
            
    except Exception as e:
        print(f"Error checking enhanced sample data: {e}")
    
    # Check what sectors have data in securities table
    securities_sample_query = """
    SELECT ticker, year, quarter, calc_date
    FROM intermediary_calculations_securities_cleaned
    WHERE quarter = (
        SELECT MAX(quarter) 
        FROM intermediary_calculations_securities_cleaned
    )
    LIMIT 10
    """
    
    try:
        securities_sample_df = pd.read_sql(securities_sample_query, engine)
        print(f"\nSample data from securities table:")
        print(securities_sample_df)
        
        # Check if these tickers have sector info
        if len(securities_sample_df) > 0:
            tickers = securities_sample_df['ticker'].tolist()
            ticker_list = ','.join([f"'{ticker}'" for ticker in tickers])
            
            sector_query = f"""
            SELECT ticker, sector, industry
            FROM master_info
            WHERE ticker IN ({ticker_list})
            """
            
            sector_df = pd.read_sql(sector_query, engine)
            print(f"\nSector information for securities sample tickers:")
            print(sector_df)
            
    except Exception as e:
        print(f"Error checking securities sample data: {e}")

if __name__ == "__main__":
    check_non_banking_schemas() 