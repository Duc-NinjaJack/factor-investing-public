#!/usr/bin/env python3
"""
Check Fundamental Values Structure
=================================

Check the actual structure of the fundamental_values table.
"""

import sys
import pandas as pd
from pathlib import Path
from sqlalchemy import text

# Add project root to path
current_path = Path.cwd()
while not (current_path / 'production').is_dir():
    if current_path.parent == current_path:
        raise FileNotFoundError("Could not find the 'production' directory.")
    current_path = current_path.parent

project_root = current_path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from production.database.connection import get_database_manager

def check_fundamental_values_structure():
    """Check the structure of fundamental_values table."""
    print("🔍 Checking Fundamental Values Table Structure")
    print("=" * 60)
    
    # Connect to database
    db_manager = get_database_manager()
    engine = db_manager.get_engine()
    
    # 1. Check table structure
    print("\n📋 Step 1: Table Structure")
    print("-" * 40)
    
    structure_query = text("""
        DESCRIBE fundamental_values
    """)
    
    structure_df = pd.read_sql(structure_query, engine)
    print("   Table columns:")
    for _, row in structure_df.iterrows():
        print(f"     {row['Field']}: {row['Type']} ({row['Null']})")
    
    # 2. Check sample data
    print("\n📊 Step 2: Sample Data")
    print("-" * 40)
    
    sample_query = text("""
        SELECT * FROM fundamental_values 
        LIMIT 5
    """)
    
    sample_df = pd.read_sql(sample_query, engine)
    print(f"   Sample data shape: {sample_df.shape}")
    print(f"   Columns: {list(sample_df.columns)}")
    
    if not sample_df.empty:
        print("   Sample rows:")
        for _, row in sample_df.iterrows():
            print(f"     {row.to_dict()}")
    
    # 3. Check if there's a different column for statement type
    print("\n🔍 Step 3: Looking for Statement Type Column")
    print("-" * 40)
    
    # Check for common variations
    possible_columns = ['statement_type', 'statement', 'type', 'stmt_type', 'financial_type']
    
    for col in possible_columns:
        try:
            check_query = text(f"""
                SELECT {col} FROM fundamental_values LIMIT 1
            """)
            pd.read_sql(check_query, engine)
            print(f"   ✅ Found column: {col}")
        except Exception as e:
            print(f"   ❌ Column '{col}' not found: {str(e)[:50]}...")
    
    # 4. Check what distinguishes PL vs BS data
    print("\n🔍 Step 4: Checking Item IDs")
    print("-" * 40)
    
    item_query = text("""
        SELECT 
            item_id,
            COUNT(*) as count,
            MIN(ticker) as sample_ticker,
            MIN(year) as min_year,
            MAX(year) as max_year
        FROM fundamental_values 
        GROUP BY item_id
        ORDER BY item_id
        LIMIT 10
    """)
    
    item_df = pd.read_sql(item_query, engine)
    print(f"   Item ID distribution:")
    for _, row in item_df.iterrows():
        print(f"     Item {row['item_id']}: {row['count']} records, Sample: {row['sample_ticker']}, Years: {row['min_year']}-{row['max_year']}")

if __name__ == "__main__":
    check_fundamental_values_structure() 