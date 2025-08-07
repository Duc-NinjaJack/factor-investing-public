#!/usr/bin/env python3

import sys
from pathlib import Path

# Add Project Root to Python Path
current_path = Path.cwd()
while not (current_path / 'production').is_dir():
    if current_path.parent == current_path:
        raise FileNotFoundError("Could not find the 'production' directory.")
    current_path = current_path.parent

project_root = current_path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from production.database.connection import get_database_manager
from sqlalchemy import text

def check_table_structure(table_name):
    try:
        db_manager = get_database_manager()
        engine = db_manager.get_engine()
        
        with engine.connect() as conn:
            # Check table structure
            result = conn.execute(text(f'DESCRIBE {table_name}'))
            columns = [row[0] for row in result]
            
            print(f'\n{table_name} columns:')
            for col in columns:
                print(f'  - {col}')
            
            # Check sample data
            result = conn.execute(text(f'SELECT * FROM {table_name} LIMIT 3'))
            rows = result.fetchall()
            
            if rows:
                print(f'\n{table_name} sample data (first 3 rows):')
                for i, row in enumerate(rows):
                    print(f'  Row {i+1}: {row}')
            else:
                print(f'\n{table_name} has no data')
                
    except Exception as e:
        print(f"Error checking {table_name}: {e}")

def main():
    tables_to_check = [
        'financial_metrics',
        'financial_metrics_v2', 
        'precalculated_metrics',
        'wong_api_daily_financial_info'
    ]
    
    for table in tables_to_check:
        check_table_structure(table)

if __name__ == "__main__":
    main() 