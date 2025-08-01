#!/usr/bin/env python3
"""
Check VNM Sales Breakdown
=========================

This script checks if VNM's Net Sales is stored as separate
Sales and Sales deductions items in the database.
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

def check_vnm_sales_breakdown():
    """Check VNM's sales breakdown in the database."""
    print("🔍 Checking VNM Sales Breakdown")
    print("=" * 50)
    
    # Connect to database
    db_manager = get_database_manager()
    engine = db_manager.get_engine()
    
    # Known values from VNM financial statements (Q2 2025)
    known_sales = 16744.61  # bn VND
    known_sales_deductions = -20.01  # bn VND
    known_net_sales = 16724.60  # bn VND
    
    print(f"📋 Known Values from VNM Financial Statements (Q2 2025):")
    print(f"   Sales: {known_sales:.2f} bn VND")
    print(f"   Sales deductions: {known_sales_deductions:.2f} bn VND")
    print(f"   Net Sales: {known_net_sales:.2f} bn VND")
    
    # Test date
    test_date = pd.Timestamp('2025-01-29')
    lag_days = 45
    lag_date = test_date - pd.Timedelta(days=lag_days)
    lag_year = lag_date.year
    lag_quarter = ((lag_date.month - 1) // 3) + 1
    
    print(f"\n🔍 Looking for Sales-related items in VNM database:")
    print(f"   Test Date: {test_date.date()}")
    print(f"   Lag Date: {lag_date.date()}")
    print(f"   Lag Year/Quarter: {lag_year}Q{lag_quarter}")
    
    # Get all PL items for VNM with their latest values
    pl_query = text("""
        SELECT 
            fv.item_id,
            fv.year,
            fv.quarter,
            fv.value,
            ROW_NUMBER() OVER (PARTITION BY fv.item_id ORDER BY fv.year DESC, fv.quarter DESC) as rn
        FROM fundamental_values fv
        WHERE fv.ticker = 'VNM'
        AND fv.statement_type = 'PL'
        AND (fv.year < :lag_year OR (fv.year = :lag_year AND fv.quarter <= :lag_quarter))
    """)
    
    pl_data = pd.read_sql(pl_query, engine, params={
        'lag_year': lag_year,
        'lag_quarter': lag_quarter
    })
    
    # Get latest values for each item
    latest_values = pl_data[pl_data['rn'] == 1].copy()
    
    print(f"\n📊 Latest PL Item Values for VNM:")
    print("-" * 60)
    
    # Look for items that could be Sales or Sales deductions
    sales_candidates = []
    deductions_candidates = []
    
    for _, row in latest_values.iterrows():
        item_id = int(row['item_id'])
        value = row['value']
        year = int(row['year'])
        quarter = int(row['quarter'])
        
        print(f"   Item {item_id:3d}: {value:12.0f} bn VND ({year}Q{quarter})")
        
        # Check if this could be Sales (should be positive and large)
        if value > 0 and abs(value - known_sales) < known_sales * 0.1:  # Within 10%
            sales_candidates.append((item_id, value, abs(value - known_sales)))
        
        # Check if this could be Sales deductions (should be negative and small)
        if value < 0 and abs(value - known_sales_deductions) < abs(known_sales_deductions) * 0.5:  # Within 50%
            deductions_candidates.append((item_id, value, abs(value - known_sales_deductions)))
    
    print(f"\n🎯 Sales Candidates (close to {known_sales:.2f} bn VND):")
    if sales_candidates:
        for item_id, value, diff in sorted(sales_candidates, key=lambda x: x[2]):
            print(f"   Item {item_id}: {value:.0f} bn VND (diff: {diff:.0f} bn VND)")
    else:
        print("   No candidates found")
    
    print(f"\n🎯 Sales Deductions Candidates (close to {known_sales_deductions:.2f} bn VND):")
    if deductions_candidates:
        for item_id, value, diff in sorted(deductions_candidates, key=lambda x: x[2]):
            print(f"   Item {item_id}: {value:.0f} bn VND (diff: {diff:.0f} bn VND)")
    else:
        print("   No candidates found")
    
    # Check if we can calculate Net Sales from any combination
    print(f"\n🧮 Checking if Net Sales can be calculated from combinations:")
    print("-" * 60)
    
    # Look for combinations that might give us Net Sales
    for i, row1 in latest_values.iterrows():
        for j, row2 in latest_values.iterrows():
            if i != j:
                item1_id = int(row1['item_id'])
                item2_id = int(row2['item_id'])
                value1 = row1['value']
                value2 = row2['value']
                
                # Try different combinations
                combinations = [
                    (value1 + value2, f"Item {item1_id} + Item {item2_id}"),
                    (value1 - value2, f"Item {item1_id} - Item {item2_id}"),
                    (value2 - value1, f"Item {item2_id} - Item {item1_id}")
                ]
                
                for result, description in combinations:
                    if abs(result - known_net_sales) < known_net_sales * 0.05:  # Within 5%
                        print(f"   ✅ {description} = {result:.0f} bn VND")
                        print(f"       (Target: {known_net_sales:.0f} bn VND, Diff: {abs(result - known_net_sales):.0f} bn VND)")
                        print()
    
    # Check if any single item is close to Net Sales
    print(f"\n🎯 Single Items Close to Net Sales ({known_net_sales:.2f} bn VND):")
    print("-" * 60)
    
    net_sales_candidates = []
    for _, row in latest_values.iterrows():
        item_id = int(row['item_id'])
        value = row['value']
        diff = abs(value - known_net_sales)
        percentage_diff = (diff / known_net_sales) * 100
        
        if percentage_diff < 50:  # Within 50%
            net_sales_candidates.append((item_id, value, diff, percentage_diff))
    
    for item_id, value, diff, percentage_diff in sorted(net_sales_candidates, key=lambda x: x[3]):
        print(f"   Item {item_id}: {value:.0f} bn VND (diff: {diff:.0f} bn VND, {percentage_diff:.1f}%)")

if __name__ == "__main__":
    check_vnm_sales_breakdown() 