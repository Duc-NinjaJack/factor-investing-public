#!/usr/bin/env python3
"""
Check table structure to find correct column names
"""

import sys
sys.path.append('../../../production/database')

from connection import get_database_manager
import pandas as pd

def main():
    try:
        db_manager = get_database_manager()
        engine = db_manager.get_engine()
        
        # Check table structure
        result = pd.read_sql('DESCRIBE vcsc_daily_data_complete', engine)
        print("Table structure:")
        print(result)
        
        # Check sample data
        sample = pd.read_sql('SELECT * FROM vcsc_daily_data_complete LIMIT 5', engine)
        print("\nSample data:")
        print(sample)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 