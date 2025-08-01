#!/usr/bin/env python3
"""
Identify the correct item IDs by analyzing data patterns in fundamental_values table
"""

import pandas as pd
import sys
import os
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

def identify_fundamental_item_ids():
    """Identify the correct item IDs by analyzing data patterns"""
    
    # Get database connection
    db_manager = get_database_manager()
    engine = db_manager.get_engine()
    
    print("🔍 IDENTIFYING FUNDAMENTAL ITEM IDs BY DATA PATTERNS")
    print("="*60)
    
    # 1. Get sample data for a well-known stock (VNM) to identify patterns
    print("\n1. ANALYZING VNM DATA PATTERNS")
    print("-" * 40)
    
    vnm_query = text("""
        SELECT 
            fv.item_id,
            fv.year,
            fv.quarter,
            fv.statement_type,
            fv.value,
            fi.item_name_en,
            fi.item_name_vn
        FROM fundamental_values fv
        LEFT JOIN fundamental_items fi ON fv.item_id = fi.item_id
        WHERE fv.ticker = 'VNM'
        AND fv.year = 2024
        ORDER BY fv.quarter, fv.statement_type, fv.item_id
        LIMIT 50
    """)
    
    vnm_df = pd.read_sql(vnm_query, engine)
    print(f"   VNM 2024 data sample:")
    for _, row in vnm_df.iterrows():
        name = row['item_name_en'] if row['item_name_en'] != 'UNMAPPED' else f"Item_{row['item_id']}"
        print(f"     - {row['statement_type']} Q{row['quarter']}: {name} = {row['value']:,.0f}")
    
    # 2. Find items with the highest values (likely Revenue or TotalAssets)
    print("\n2. FINDING HIGHEST VALUE ITEMS (LIKELY REVENUE/ASSETS)")
    print("-" * 40)
    
    high_value_query = text("""
        SELECT 
            fv.item_id,
            fv.statement_type,
            AVG(fv.value) as avg_value,
            COUNT(*) as record_count,
            fi.item_name_en,
            fi.item_name_vn
        FROM fundamental_values fv
        LEFT JOIN fundamental_items fi ON fv.item_id = fi.item_id
        WHERE fv.year = 2024
        AND fv.value > 0
        GROUP BY fv.item_id, fv.statement_type, fi.item_name_en, fi.item_name_vn
        ORDER BY avg_value DESC
        LIMIT 20
    """)
    
    high_value_df = pd.read_sql(high_value_query, engine)
    print(f"   Highest value items in 2024:")
    for _, row in high_value_df.iterrows():
        name = row['item_name_en'] if row['item_name_en'] != 'UNMAPPED' else f"Item_{row['item_id']}"
        print(f"     - {row['statement_type']}: {name} = {row['avg_value']:,.0f} (avg, {row['record_count']} records)")
    
    # 3. Find items with moderate positive values (likely NetProfit)
    print("\n3. FINDING MODERATE POSITIVE VALUE ITEMS (LIKELY NETPROFIT)")
    print("-" * 40)
    
    profit_query = text("""
        SELECT 
            fv.item_id,
            fv.statement_type,
            AVG(fv.value) as avg_value,
            COUNT(*) as record_count,
            fi.item_name_en,
            fi.item_name_vn
        FROM fundamental_values fv
        LEFT JOIN fundamental_items fi ON fv.item_id = fi.item_id
        WHERE fv.year = 2024
        AND fv.value > 0
        AND fv.value < 1000000000000  -- Less than 1 trillion
        AND fv.statement_type = 'PL'
        GROUP BY fv.item_id, fv.statement_type, fi.item_name_en, fi.item_name_vn
        ORDER BY avg_value DESC
        LIMIT 15
    """)
    
    profit_df = pd.read_sql(profit_query, engine)
    print(f"   Moderate positive value PL items in 2024:")
    for _, row in profit_df.iterrows():
        name = row['item_name_en'] if row['item_name_en'] != 'UNMAPPED' else f"Item_{row['item_id']}"
        print(f"     - {name} = {row['avg_value']:,.0f} (avg, {row['record_count']} records)")
    
    # 4. Test specific item IDs for VNM to see if they make sense
    print("\n4. TESTING SPECIFIC ITEM IDs FOR VNM")
    print("-" * 40)
    
    # Get the top items from each category
    if not high_value_df.empty and not profit_df.empty:
        top_revenue_id = high_value_df.iloc[0]['item_id']
        top_profit_id = profit_df.iloc[0]['item_id']
        
        test_query = text("""
            SELECT 
                fv.item_id,
                fv.year,
                fv.quarter,
                fv.statement_type,
                fv.value,
                fi.item_name_en,
                fi.item_name_vn
            FROM fundamental_values fv
            LEFT JOIN fundamental_items fi ON fv.item_id = fi.item_id
            WHERE fv.ticker = 'VNM'
            AND fv.item_id IN (%s, %s)
            AND fv.year BETWEEN 2020 AND 2024
            ORDER BY fv.year, fv.quarter, fv.item_id
        """ % (top_revenue_id, top_profit_id))
        
        test_df = pd.read_sql(test_query, engine)
        print(f"   VNM historical data for top items:")
        for _, row in test_df.iterrows():
            name = row['item_name_en'] if row['item_name_en'] != 'UNMAPPED' else f"Item_{row['item_id']}"
            print(f"     - {row['year']}Q{row['quarter']} {row['statement_type']}: {name} = {row['value']:,.0f}")
    
    # 5. Summary and recommendations
    print("\n5. SUMMARY AND RECOMMENDATIONS")
    print("-" * 40)
    
    if not high_value_df.empty and not profit_df.empty:
        print("   ✅ SUCCESS: Identified potential fundamental items")
        print(f"   Recommended Revenue ID: {high_value_df.iloc[0]['item_id']} (avg: {high_value_df.iloc[0]['avg_value']:,.0f})")
        print(f"   Recommended NetProfit ID: {profit_df.iloc[0]['item_id']} (avg: {profit_df.iloc[0]['avg_value']:,.0f})")
        print("   ")
        print("   🎯 NEXT STEPS:")
        print("   1. Use these item IDs in the strategy")
        print("   2. Calculate TTM values from quarterly data")
        print("   3. Apply 45-day lag for announcement delay")
        print("   4. Test the updated strategy")
    else:
        print("   ❌ ISSUE: Could not identify fundamental items")
        print("   Need to investigate further")

if __name__ == "__main__":
    identify_fundamental_item_ids() 