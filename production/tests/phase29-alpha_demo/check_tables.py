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

def main():
    try:
        db_manager = get_database_manager()
        engine = db_manager.get_engine()
        
        with engine.connect() as conn:
            result = conn.execute(text('SHOW TABLES'))
            tables = [row[0] for row in result]
            
            print('Available tables:')
            for table in sorted(tables):
                print(f'  - {table}')
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
