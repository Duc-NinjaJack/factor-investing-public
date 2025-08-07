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
            # Check available dates in wong_api_daily_financial_info
            result = conn.execute(text("""
                SELECT 
                    MIN(data_date) as earliest_date,
                    MAX(data_date) as latest_date,
                    COUNT(DISTINCT data_date) as unique_dates,
                    COUNT(DISTINCT ticker) as unique_tickers
                FROM wong_api_daily_financial_info
            """))
            stats = result.fetchone()
            
            print('wong_api_daily_financial_info statistics:')
            print(f'  Earliest date: {stats[0]}')
            print(f'  Latest date: {stats[1]}')
            print(f'  Unique dates: {stats[2]}')
            print(f'  Unique tickers: {stats[3]}')
            
            # Check sample dates
            result = conn.execute(text("""
                SELECT DISTINCT data_date 
                FROM wong_api_daily_financial_info 
                ORDER BY data_date DESC 
                LIMIT 10
            """))
            recent_dates = [row[0] for row in result]
            
            print(f'\nRecent dates available:')
            for date in recent_dates:
                print(f'  - {date}')
            
            # Check for specific stocks
            result = conn.execute(text("""
                SELECT ticker, COUNT(*) as records
                FROM wong_api_daily_financial_info
                WHERE ticker IN ('HPG', 'VNM', 'VCB', 'TCB', 'FPT')
                GROUP BY ticker
                ORDER BY records DESC
            """))
            
            print(f'\nSample stocks data availability:')
            for row in result:
                print(f'  - {row[0]}: {row[1]} records')
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
