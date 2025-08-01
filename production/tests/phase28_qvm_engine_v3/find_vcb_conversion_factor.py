#!/usr/bin/env python3
"""
Find VCB Conversion Factor
==========================

This script finds the correct conversion factor for VCB using
the financial statement data.
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

def find_vcb_conversion_factor():
    """Find the correct conversion factor for VCB."""
    print("🔍 Finding VCB Conversion Factor")
    print("=" * 50)
    
    # Connect to database
    db_manager = get_database_manager()
    engine = db_manager.get_engine()
    
    # Known values from VCB financial statements (Q2 2025)
    known_net_profit = 8837.37  # bn VND
    known_total_operating_income = 17868.24  # bn VND
    known_total_assets = 2217941.10  # bn VND
    
    print(f"📋 Known Values from VCB Financial Statements (Q2 2025):")
    print(f"   Net Profit: {known_net_profit:.2f} bn VND")
    print(f"   Total Operating Income: {known_total_operating_income:.2f} bn VND")
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
        WHERE fv.ticker = 'VCB'
        AND fv.item_id IN (1, 13, 2)
        AND fv.statement_type IN ('PL', 'BS')
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
    
    print(f"\n📊 Latest Database Values for VCB:")
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
    
    # For Total Operating Income (Item 13)
    if 13 in latest_values:
        db_operating_income = latest_values[13]
        conversion_factor_operating_income = db_operating_income / known_total_operating_income
        print(f"   Total Operating Income (Item 13):")
        print(f"     Database: {db_operating_income:,.0f}")
        print(f"     Financial Statement: {known_total_operating_income:.2f} bn VND")
        print(f"     Conversion Factor: {conversion_factor_operating_income:.2f}")
        print()
    
    # For Total Assets (Item 2)
    if 2 in latest_values:
        db_total_assets = latest_values[2]
        conversion_factor_total_assets = db_total_assets / known_total_assets
        print(f"   Total Assets (Item 2):")
        print(f"     Database: {db_total_assets:,.0f}")
        print(f"     Financial Statement: {known_total_assets:.2f} bn VND")
        print(f"     Conversion Factor: {conversion_factor_total_assets:.2f}")
        print()
    
    # Find the best conversion factor
    print(f"\n🎯 Best Conversion Factor:")
    print("-" * 50)
    
    conversion_factors = []
    if 1 in latest_values:
        conversion_factors.append(('Net Profit', conversion_factor_net_profit))
    if 13 in latest_values:
        conversion_factors.append(('Total Operating Income', conversion_factor_operating_income))
    if 2 in latest_values:
        conversion_factors.append(('Total Assets', conversion_factor_total_assets))
    
    # Calculate average and standard deviation
    if conversion_factors:
        factors = [cf[1] for cf in conversion_factors]
        avg_factor = sum(factors) / len(factors)
        
        print(f"   Conversion Factors:")
        for name, factor in conversion_factors:
            print(f"     {name}: {factor:.2f}")
        
        print(f"   Average: {avg_factor:.2f}")
        
        # Find the most consistent factor
        best_factor = min(conversion_factors, key=lambda x: abs(x[1] - avg_factor))
        print(f"   🏆 Best (closest to average): {best_factor[0]} = {best_factor[1]:.2f}")
        
        # Test the best factor
        print(f"\n🧪 Testing with Best Factor ({best_factor[1]:.2f}):")
        print("-" * 50)
        
        for item_id in [1, 13, 2]:
            if item_id in latest_values:
                db_value = latest_values[item_id]
                converted_value = db_value / best_factor[1]
                
                if item_id == 1:
                    target = known_net_profit
                    label = "Net Profit"
                elif item_id == 13:
                    target = known_total_operating_income
                    label = "Total Operating Income"
                else:
                    target = known_total_assets
                    label = "Total Assets"
                
                diff = abs(converted_value - target)
                percentage_diff = (diff / target) * 100
                
                print(f"   Item {item_id} ({label}):")
                print(f"     Converted: {converted_value:.2f} bn VND")
                print(f"     Target: {target:.2f} bn VND")
                print(f"     Difference: {diff:.2f} bn VND ({percentage_diff:.1f}%)")
                print()

if __name__ == "__main__":
    find_vcb_conversion_factor() 