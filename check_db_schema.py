#!/usr/bin/env python3

from production.database.connection import get_database_manager
from sqlalchemy import text

def check_schema():
    db = get_database_manager()
    engine = db.get_engine()
    
    with engine.connect() as conn:
        # Check vcsc_daily_data_complete table
        result = conn.execute(text('DESCRIBE vcsc_daily_data_complete'))
        print('Columns in vcsc_daily_data_complete:')
        for row in result:
            print(f"  {row[0]} - {row[1]}")
        
        print('\nColumns in intermediary_calculations_enhanced:')
        result = conn.execute(text('DESCRIBE intermediary_calculations_enhanced'))
        for row in result:
            print(f"  {row[0]} - {row[1]}")
        
        print('\nSample data from intermediary_calculations_enhanced:')
        result = conn.execute(text('SELECT * FROM intermediary_calculations_enhanced LIMIT 3'))
        for row in result:
            print(f"  {row}")

if __name__ == "__main__":
    check_schema() 