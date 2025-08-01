#!/usr/bin/env python3
"""
Find VNM Sales Components
=========================

This script finds the separate Sales and Sales deductions items
in the database and calculates Net Sales.
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

def find_vnm_sales_components():
    """Find VNM sales components in the database."""
    print("🔍 Finding VNM Sales Components")
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
    
    # Get all PL items for VNM to find Sales and Sales deductions
    query = text("""
        SELECT 
            fv.item_id,
            fv.year,
            fv.quarter,
            fv.value,
            fv.value / 6222444702.01 as value_bn
        FROM fundamental_values fv
        WHERE fv.ticker = 'VNM'
        AND fv.statement_type = 'PL'
        AND fv.year = 2025 AND fv.quarter = 2
        ORDER BY ABS(fv.value) DESC
    """)
    
    data = pd.read_sql(query, engine)
    
    print(f"\n📊 All PL Items for VNM (Q2 2025) - Sorted by Absolute Value:")
    print("-" * 80)
    
    # Look for Sales and Sales deductions candidates
    sales_candidates = []
    deductions_candidates = []
    
    for _, row in data.iterrows():
        item_id = int(row['item_id'])
        value = row['value']
        value_bn = row['value_bn']
        
        print(f"   Item {item_id:3d}: {value:15,.0f} → {value_bn:8.2f} bn VND")
        
        # Check if this could be Sales (should be positive and large)
        if value > 0 and abs(value_bn - known_sales) < known_sales * 0.2:  # Within 20%
            sales_candidates.append((item_id, value_bn, abs(value_bn - known_sales)))
        
        # Check if this could be Sales deductions (should be negative and small)
        if value < 0 and abs(value_bn - known_sales_deductions) < abs(known_sales_deductions) * 2:  # Within 200%
            deductions_candidates.append((item_id, value_bn, abs(value_bn - known_sales_deductions)))
    
    print(f"\n🎯 Sales Candidates (close to {known_sales:.2f} bn VND):")
    if sales_candidates:
        for item_id, value, diff in sorted(sales_candidates, key=lambda x: x[2]):
            print(f"   Item {item_id}: {value:.2f} bn VND (diff: {diff:.2f} bn VND)")
    else:
        print("   No candidates found")
    
    print(f"\n🎯 Sales Deductions Candidates (close to {known_sales_deductions:.2f} bn VND):")
    if deductions_candidates:
        for item_id, value, diff in sorted(deductions_candidates, key=lambda x: x[2]):
            print(f"   Item {item_id}: {value:.2f} bn VND (diff: {diff:.2f} bn VND)")
    else:
        print("   No candidates found")
    
    # Try to calculate Net Sales from combinations
    print(f"\n🧮 Calculating Net Sales from combinations:")
    print("-" * 60)
    
    # Get all positive and negative items
    positive_items = data[data['value'] > 0].head(10)  # Top 10 positive items
    negative_items = data[data['value'] < 0].head(10)  # Top 10 negative items
    
    print(f"   Testing combinations of positive and negative items:")
    
    best_combination = None
    best_diff = float('inf')
    
    for _, pos_row in positive_items.iterrows():
        for _, neg_row in negative_items.iterrows():
            pos_item_id = int(pos_row['item_id'])
            neg_item_id = int(neg_row['item_id'])
            pos_value = pos_row['value_bn']
            neg_value = neg_row['value_bn']
            
            # Calculate Net Sales = Sales - Sales deductions
            net_sales = pos_value - neg_value
            
            diff = abs(net_sales - known_net_sales)
            percentage_diff = (diff / known_net_sales) * 100
            
            if percentage_diff < 10:  # Within 10%
                print(f"     Item {pos_item_id} - Item {neg_item_id} = {net_sales:.2f} bn VND")
                print(f"       ({pos_value:.2f} - {neg_value:.2f} = {net_sales:.2f})")
                print(f"       Target: {known_net_sales:.2f} bn VND, Diff: {diff:.2f} bn VND ({percentage_diff:.1f}%)")
                print()
                
                if diff < best_diff:
                    best_diff = diff
                    best_combination = (pos_item_id, neg_item_id, pos_value, neg_value, net_sales)
    
    if best_combination:
        pos_id, neg_id, pos_val, neg_val, net_sales = best_combination
        print(f"🏆 Best Combination:")
        print(f"   Sales: Item {pos_id} = {pos_val:.2f} bn VND")
        print(f"   Sales Deductions: Item {neg_id} = {neg_val:.2f} bn VND")
        print(f"   Net Sales: {pos_val:.2f} - {neg_val:.2f} = {net_sales:.2f} bn VND")
        print(f"   Target: {known_net_sales:.2f} bn VND")
        print(f"   Difference: {abs(net_sales - known_net_sales):.2f} bn VND")
        
        # Check if this gives reasonable ratios
        print(f"\n📊 Testing Ratios with Best Combination:")
        print("-" * 50)
        
        # Get Net Profit for Q2 2025
        net_profit_query = text("""
            SELECT fv.value / 6222444702.01 as net_profit_bn
            FROM fundamental_values fv
            WHERE fv.ticker = 'VNM'
            AND fv.item_id = 1
            AND fv.statement_type = 'PL'
            AND fv.year = 2025 AND fv.quarter = 2
        """)
        
        net_profit_result = pd.read_sql(net_profit_query, engine)
        
        if not net_profit_result.empty:
            net_profit = net_profit_result.iloc[0]['net_profit_bn']
            
            # Calculate ratios
            net_margin = net_profit / net_sales if net_sales > 0 else None
            print(f"   Net Profit: {net_profit:.2f} bn VND")
            print(f"   Net Sales: {net_sales:.2f} bn VND")
            if net_margin:
                print(f"   Net Margin: {net_margin:.4f} ({net_margin*100:.2f}%)")
                
                if 0.05 <= net_margin <= 0.30:
                    print(f"   ✅ Net Margin is reasonable (5-30% expected)")
                else:
                    print(f"   ❌ Net Margin is outside expected range (5-30%)")
    
    # Check if any single item is close to Net Sales
    print(f"\n🎯 Single Items Close to Net Sales ({known_net_sales:.2f} bn VND):")
    print("-" * 60)
    
    net_sales_candidates = []
    for _, row in data.iterrows():
        item_id = int(row['item_id'])
        value_bn = row['value_bn']
        diff = abs(value_bn - known_net_sales)
        percentage_diff = (diff / known_net_sales) * 100
        
        if percentage_diff < 50:  # Within 50%
            net_sales_candidates.append((item_id, value_bn, diff, percentage_diff))
    
    for item_id, value, diff, percentage_diff in sorted(net_sales_candidates, key=lambda x: x[3]):
        print(f"   Item {item_id}: {value:.2f} bn VND (diff: {diff:.2f} bn VND, {percentage_diff:.1f}%)")

if __name__ == "__main__":
    find_vnm_sales_components() 