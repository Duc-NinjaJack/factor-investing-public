#!/usr/bin/env python3
"""
Check the structure of intermediary_calculations_enhanced table
"""

import sys
from pathlib import Path
import pandas as pd

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

def check_table_structure():
    """Check the structure of intermediary_calculations_enhanced table."""
    try:
        # Get database connection
        db_manager = get_database_manager()
        engine = db_manager.get_engine()
        
        print("🔍 Checking intermediary_calculations_enhanced table structure...")
        
        # Get table columns
        query = """
        SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_NAME = 'intermediary_calculations_enhanced'
        ORDER BY ORDINAL_POSITION
        """
        
        columns_df = pd.read_sql(query, engine)
        print(f"\n📊 Table Columns ({len(columns_df)} columns):")
        for _, row in columns_df.iterrows():
            print(f"   - {row['COLUMN_NAME']}: {row['DATA_TYPE']} ({'NULL' if row['IS_NULLABLE'] == 'YES' else 'NOT NULL'})")
        
        # Get sample data
        print(f"\n📋 Sample Data (first 3 rows):")
        sample_df = pd.read_sql("SELECT * FROM intermediary_calculations_enhanced LIMIT 3", engine)
        print(sample_df.to_string())
        
        # Get row count
        count_df = pd.read_sql("SELECT COUNT(*) as total_rows FROM intermediary_calculations_enhanced", engine)
        print(f"\n📈 Total Rows: {count_df['total_rows'].iloc[0]:,}")
        
        # Check for specific columns
        available_columns = columns_df['COLUMN_NAME'].tolist()
        required_columns = ['ticker', 'year', 'quarter', 'roaa', 'pe_ratio']
        
        print(f"\n🔍 Checking required columns:")
        for col in required_columns:
            status = "✅" if col in available_columns else "❌"
            print(f"   {status} {col}")
        
        return available_columns
        
    except Exception as e:
        print(f"❌ Error checking table structure: {e}")
        return []

if __name__ == "__main__":
    check_table_structure() 