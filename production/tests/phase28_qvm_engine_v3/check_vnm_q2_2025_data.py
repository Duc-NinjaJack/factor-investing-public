#!/usr/bin/env python3
"""
Check VNM Q2 2025 Data
======================

Simple script to check what data is available for VNM Q2 2025.
"""

import sys
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine, text

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

def check_vnm_q2_2025_data():
    """Check VNM Q2 2025 data."""
    print("🔍 Checking VNM Q2 2025 Data")
    print("=" * 50)
    
    # Connect to database
    db_manager = get_database_manager()
    engine = db_manager.get_engine()
    
    # Check if Q2 2025 data exists
    check_query = text("""
        SELECT COUNT(*) as count
        FROM fundamental_values fv
        WHERE fv.ticker = 'VNM'
        AND fv.year = 2025 AND fv.quarter = 2
    """)
    
    count_result = pd.read_sql(check_query, engine)
    count = count_result.iloc[0]['count']
    
    print(f"📊 VNM Q2 2025 data count: {count}")
    
    if count > 0:
        # Get all Q2 2025 data
        data_query = text("""
            SELECT 
                fv.item_id,
                fv.statement_type,
                fv.value,
                fv.value / 6222444702.01 as value_bn
            FROM fundamental_values fv
            WHERE fv.ticker = 'VNM'
            AND fv.year = 2025 AND fv.quarter = 2
            ORDER BY ABS(fv.value) DESC
        """)
        
        data = pd.read_sql(data_query, engine)
        
        print(f"\n📋 All VNM Q2 2025 Data:")
        print("-" * 60)
        
        for _, row in data.iterrows():
            item_id = int(row['item_id'])
            stmt_type = row['statement_type']
            value = row['value']
            value_bn = row['value_bn']
            
            print(f"   Item {item_id:3d} ({stmt_type}): {value:15,.0f} → {value_bn:8.2f} bn VND")
        
        # Check for negative values (potential deductions)
        negative_data = data[data['value'] < 0]
        if not negative_data.empty:
            print(f"\n🔴 Negative Values (Potential Deductions):")
            print("-" * 50)
            for _, row in negative_data.iterrows():
                item_id = int(row['item_id'])
                stmt_type = row['statement_type']
                value = row['value']
                value_bn = row['value_bn']
                
                print(f"   Item {item_id:3d} ({stmt_type}): {value:15,.0f} → {value_bn:8.2f} bn VND")
        
        # Check for large positive values (potential sales)
        large_positive = data[data['value'] > 1000000000000]  # > 1 trillion
        if not large_positive.empty:
            print(f"\n🟢 Large Positive Values (Potential Sales):")
            print("-" * 50)
            for _, row in large_positive.iterrows():
                item_id = int(row['item_id'])
                stmt_type = row['statement_type']
                value = row['value']
                value_bn = row['value_bn']
                
                print(f"   Item {item_id:3d} ({stmt_type}): {value:15,.0f} → {value_bn:8.2f} bn VND")
    else:
        print("❌ No Q2 2025 data found for VNM")
        
        # Check what quarters are available
        quarters_query = text("""
            SELECT DISTINCT year, quarter
            FROM fundamental_values fv
            WHERE fv.ticker = 'VNM'
            ORDER BY year DESC, quarter DESC
            LIMIT 10
        """)
        
        quarters_data = pd.read_sql(quarters_query, engine)
        
        print(f"\n📅 Available quarters for VNM:")
        for _, row in quarters_data.iterrows():
            year = int(row['year'])
            quarter = int(row['quarter'])
            print(f"   {year}Q{quarter}")

if __name__ == "__main__":
    check_vnm_q2_2025_data() 