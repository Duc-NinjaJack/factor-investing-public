#!/usr/bin/env python3
"""
Test script to verify database connection and table access.
"""

import sys
from pathlib import Path

# Add project root to path
current_path = Path.cwd()
while not (current_path / 'production').is_dir():
    if current_path.parent == current_path:
        raise FileNotFoundError("Could not find the 'production' directory.")
    current_path = current_path.parent

if str(current_path) not in sys.path:
    sys.path.insert(0, str(current_path))

from production.database.connection import get_database_manager
from sqlalchemy import text
import pandas as pd

def test_database_connection():
    """Test database connection and basic queries."""
    print("🔍 Testing Database Connection and Table Access")
    print("=" * 60)
    
    try:
        # Initialize database connection
        db_manager = get_database_manager()
        engine = db_manager.get_engine()
        print("✅ Database connection established")
        
        with engine.connect() as conn:
            # Test 1: Check if vcsc_daily_data table exists and has data
            print("\n📊 Test 1: Checking vcsc_daily_data table")
            result = conn.execute(text("""
                SELECT COUNT(*) as total_rows, 
                       MIN(trading_date) as earliest_date,
                       MAX(trading_date) as latest_date
                FROM vcsc_daily_data
            """))
            row = result.fetchone()
            print(f"   - Total rows: {row[0]:,}")
            print(f"   - Date range: {row[1]} to {row[2]}")
            
            # Test 2: Check for VNM benchmark data
            print("\n📊 Test 2: Checking VNM benchmark data")
            result = conn.execute(text("""
                SELECT COUNT(*) as vnm_rows,
                       MIN(trading_date) as earliest_vnm,
                       MAX(trading_date) as latest_vnm
                FROM vcsc_daily_data 
                WHERE ticker = 'VNM'
            """))
            row = result.fetchone()
            print(f"   - VNM rows: {row[0]:,}")
            print(f"   - VNM date range: {row[1]} to {row[2]}")
            
            # Test 3: Check sample price data
            print("\n📊 Test 3: Sample price data")
            result = conn.execute(text("""
                SELECT ticker, trading_date, close_price, total_volume
                FROM vcsc_daily_data 
                WHERE trading_date >= '2025-01-01'
                ORDER BY trading_date DESC, ticker
                LIMIT 5
            """))
            for row in result.fetchall():
                print(f"   - {row[0]}: {row[1]} | Close: {row[2]} | Volume: {row[3]:,}")
            
            # Test 4: Test the exact query from the fixed code
            print("\n📊 Test 4: Testing fixed code query")
            query = """
            SELECT 
                ticker,
                trading_date as date,
                close_price as close,
                total_volume as volume
            FROM vcsc_daily_data 
            WHERE trading_date >= '2025-01-01' AND trading_date <= '2025-01-10'
            ORDER BY trading_date, ticker
            LIMIT 10
            """
            
            result = conn.execute(text(query))
            rows = result.fetchall()
            print(f"   - Query returned {len(rows)} rows")
            if rows:
                print(f"   - Sample: {rows[0]}")
        
        print("\n✅ All database tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Database test failed: {e}")
        return False

if __name__ == "__main__":
    test_database_connection() 