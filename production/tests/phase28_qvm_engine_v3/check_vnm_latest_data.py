#!/usr/bin/env python3
"""
Check VNM Latest Data
=====================

Check VNM's latest available data and compare with financial statements.
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

def check_vnm_latest_data():
    """Check VNM's latest available data."""
    print("🔍 Checking VNM Latest Available Data")
    print("=" * 50)
    
    # Connect to database
    db_manager = get_database_manager()
    engine = db_manager.get_engine()
    
    # Get latest available quarter
    latest_query = text("""
        SELECT MAX(year) as max_year, MAX(quarter) as max_quarter
        FROM fundamental_values fv
        WHERE fv.ticker = 'VNM'
    """)
    
    latest_result = pd.read_sql(latest_query, engine)
    max_year = int(latest_result.iloc[0]['max_year'])
    max_quarter = int(latest_result.iloc[0]['max_quarter'])
    
    print(f"📅 Latest available quarter: {max_year}Q{max_quarter}")
    
    # Get all data for the latest quarter
    data_query = text("""
        SELECT 
            fv.item_id,
            fv.statement_type,
            fv.value,
            fv.value / 6222444702.01 as value_bn
        FROM fundamental_values fv
        WHERE fv.ticker = 'VNM'
        AND fv.year = :max_year AND fv.quarter = :max_quarter
        ORDER BY ABS(fv.value) DESC
    """)
    
    data = pd.read_sql(data_query, engine, params={'max_year': max_year, 'max_quarter': max_quarter})
    
    print(f"\n📋 All VNM {max_year}Q{max_quarter} Data:")
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
    
    # Compare with financial statement data
    print(f"\n📊 Comparison with Financial Statements:")
    print("-" * 50)
    
    if max_year == 2025 and max_quarter == 1:
        print(f"   Database: {max_year}Q{max_quarter}")
        print(f"   Financial Statement: Q2 2025 (not in database yet)")
        print(f"   Note: Database is 1 quarter behind financial statements")
        
        # Get Net Profit for comparison
        net_profit_query = text("""
            SELECT fv.value / 6222444702.01 as net_profit_bn
            FROM fundamental_values fv
            WHERE fv.ticker = 'VNM'
            AND fv.item_id = 1
            AND fv.statement_type = 'PL'
            AND fv.year = :max_year AND fv.quarter = :max_quarter
        """)
        
        net_profit_result = pd.read_sql(net_profit_query, engine, params={'max_year': max_year, 'max_quarter': max_quarter})
        
        if not net_profit_result.empty:
            db_net_profit = net_profit_result.iloc[0]['net_profit_bn']
            print(f"\n   Net Profit Comparison:")
            print(f"     Database ({max_year}Q{max_quarter}): {db_net_profit:.2f} bn VND")
            print(f"     Financial Statement (Q2 2025): 2,488.58 bn VND")
            print(f"     Note: Different quarters, so direct comparison not meaningful")
    
    # Check if we can find a reasonable sales proxy
    print(f"\n🎯 Finding Sales Proxy:")
    print("-" * 30)
    
    # Look for the largest positive PL item
    pl_data = data[data['statement_type'] == 'PL']
    if not pl_data.empty:
        largest_pl = pl_data.loc[pl_data['value'].idxmax()]
        largest_item_id = int(largest_pl['item_id'])
        largest_value = largest_pl['value_bn']
        
        print(f"   Largest PL Item: {largest_item_id} = {largest_value:.2f} bn VND")
        print(f"   This could be a reasonable sales proxy for {max_year}Q{max_quarter}")
        
        # Calculate Net Margin if we have Net Profit
        net_profit_data = data[(data['item_id'] == 1) & (data['statement_type'] == 'PL')]
        if not net_profit_data.empty:
            net_profit = net_profit_data.iloc[0]['value_bn']
            net_margin = net_profit / largest_value if largest_value > 0 else None
            
            if net_margin:
                print(f"   Net Profit: {net_profit:.2f} bn VND")
                print(f"   Net Margin: {net_margin:.4f} ({net_margin*100:.2f}%)")
                
                if 0.05 <= net_margin <= 0.30:
                    print(f"   ✅ Net Margin is reasonable (5-30% expected)")
                else:
                    print(f"   ⚠️ Net Margin is outside expected range (5-30%)")

if __name__ == "__main__":
    check_vnm_latest_data() 