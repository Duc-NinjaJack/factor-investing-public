#!/usr/bin/env python3
"""
Find Conversion Factor
======================

This script finds the correct conversion factor between database values
and financial statement values.
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

def find_conversion_factor():
    """Find the correct conversion factor."""
    print("🔍 Finding Conversion Factor")
    print("=" * 50)
    
    # Connect to database
    db_manager = get_database_manager()
    engine = db_manager.get_engine()
    
    # Known values from VNM financial statements (Q2 2025)
    known_net_profit = 2488.58  # bn VND
    known_net_sales = 16724.60  # bn VND
    known_total_assets = 55282.66  # bn VND
    
    print(f"📋 Known Values from VNM Financial Statements (Q2 2025):")
    print(f"   Net Profit: {known_net_profit:.2f} bn VND")
    print(f"   Net Sales: {known_net_sales:.2f} bn VND")
    print(f"   Total Assets: {known_total_assets:.2f} bn VND")
    
    # Test date
    test_date = pd.Timestamp('2025-01-29')
    lag_days = 45
    lag_date = test_date - pd.Timedelta(days=lag_days)
    lag_year = lag_date.year
    lag_quarter = ((lag_date.month - 1) // 3) + 1
    
    # Get latest database values for key items
    db_query = text("""
        SELECT 
            fv.item_id,
            fv.value,
            fv.year,
            fv.quarter
        FROM fundamental_values fv
        WHERE fv.ticker = 'VNM'
        AND fv.item_id IN (1, 2, 3, 4, 15)
        AND fv.statement_type = 'PL'
        AND (fv.year < :lag_year OR (fv.year = :lag_year AND fv.quarter <= :lag_quarter))
        ORDER BY fv.item_id, fv.year DESC, fv.quarter DESC
    """)
    
    db_data = pd.read_sql(db_query, engine, params={
        'lag_year': lag_year,
        'lag_quarter': lag_quarter
    })
    
    # Get latest value for each item
    latest_values = {}
    for _, row in db_data.iterrows():
        item_id = int(row['item_id'])
        if item_id not in latest_values:
            latest_values[item_id] = row['value']
    
    print(f"\n📊 Latest Database Values for VNM:")
    print("-" * 50)
    for item_id, value in sorted(latest_values.items()):
        print(f"   Item {item_id}: {value:,.0f}")
    
    # Calculate conversion factors
    print(f"\n🧮 Calculating Conversion Factors:")
    print("-" * 50)
    
    # For Net Profit (Item 1)
    if 1 in latest_values:
        db_net_profit = latest_values[1]
        conversion_factor_net_profit = db_net_profit / known_net_profit
        print(f"   Net Profit (Item 1):")
        print(f"     Database: {db_net_profit:,.0f}")
        print(f"     Financial Statement: {known_net_profit:.2f} bn VND")
        print(f"     Conversion Factor: {conversion_factor_net_profit:.2f}")
        print()
    
    # For Net Sales candidates
    sales_candidates = [2, 3, 4, 15]
    for item_id in sales_candidates:
        if item_id in latest_values:
            db_value = latest_values[item_id]
            conversion_factor = db_value / known_net_sales
            print(f"   Item {item_id} (Net Sales candidate):")
            print(f"     Database: {db_value:,.0f}")
            print(f"     Financial Statement: {known_net_sales:.2f} bn VND")
            print(f"     Conversion Factor: {conversion_factor:.2f}")
            print()
    
    # Test different conversion factors
    print(f"\n🧪 Testing Different Conversion Factors:")
    print("-" * 50)
    
    # Try the conversion factor from Net Profit
    if 1 in latest_values:
        conversion_factor = conversion_factor_net_profit
        
        print(f"   Using conversion factor from Net Profit: {conversion_factor:.2f}")
        print(f"   (Database values / {conversion_factor:.2f} = Financial Statement values)")
        print()
        
        for item_id in [1, 2, 3, 4, 15]:
            if item_id in latest_values:
                db_value = latest_values[item_id]
                converted_value = db_value / conversion_factor
                
                if item_id == 1:
                    target = known_net_profit
                    label = "Net Profit"
                else:
                    target = known_net_sales
                    label = "Net Sales"
                
                diff = abs(converted_value - target)
                percentage_diff = (diff / target) * 100
                
                print(f"   Item {item_id} ({label}):")
                print(f"     Converted: {converted_value:.2f} bn VND")
                print(f"     Target: {target:.2f} bn VND")
                print(f"     Difference: {diff:.2f} bn VND ({percentage_diff:.1f}%)")
                print()
    
    # Find the best conversion factor for Net Sales
    print(f"\n🎯 Best Conversion Factor for Net Sales:")
    print("-" * 50)
    
    best_factor = None
    best_item = None
    best_diff = float('inf')
    
    for item_id in sales_candidates:
        if item_id in latest_values:
            db_value = latest_values[item_id]
            conversion_factor = db_value / known_net_sales
            diff = abs(conversion_factor - conversion_factor_net_profit)
            
            print(f"   Item {item_id}: Conversion factor = {conversion_factor:.2f}")
            print(f"     Difference from Net Profit factor: {diff:.2f}")
            
            if diff < best_diff:
                best_diff = diff
                best_factor = conversion_factor
                best_item = item_id
    
    if best_factor:
        print(f"\n   🏆 Best factor: {best_factor:.2f} (from Item {best_item})")
        print(f"   This factor is closest to the Net Profit factor: {conversion_factor_net_profit:.2f}")

if __name__ == "__main__":
    find_conversion_factor() 