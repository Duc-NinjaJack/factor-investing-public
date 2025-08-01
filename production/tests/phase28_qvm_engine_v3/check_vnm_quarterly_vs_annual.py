#!/usr/bin/env python3
"""
Check VNM Quarterly vs Annual
=============================

This script checks if VNM's database values are quarterly or annual data
by comparing with the known Q2 2025 sales figure.
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

def check_vnm_quarterly_vs_annual():
    """Check if VNM database values are quarterly or annual."""
    print("🔍 Checking VNM Quarterly vs Annual Data")
    print("=" * 50)
    
    # Connect to database
    db_manager = get_database_manager()
    engine = db_manager.get_engine()
    
    # Known values from VNM financial statements (Q2 2025)
    known_q2_2025_sales = 16744.61  # bn VND (Q2 2025)
    known_q2_2025_net_profit = 2488.58  # bn VND (Q2 2025)
    
    print(f"📋 Known Values from VNM Financial Statements (Q2 2025):")
    print(f"   Sales (Q2 2025): {known_q2_2025_sales:.2f} bn VND")
    print(f"   Net Profit (Q2 2025): {known_q2_2025_net_profit:.2f} bn VND")
    
    # Test date
    test_date = pd.Timestamp('2025-01-29')
    lag_days = 45
    lag_date = test_date - pd.Timedelta(days=lag_days)
    lag_year = lag_date.year
    lag_quarter = ((lag_date.month - 1) // 3) + 1
    
    print(f"\n🔍 Testing different scenarios:")
    print(f"   Test Date: {test_date.date()}")
    print(f"   Lag Date: {lag_date.date()}")
    print(f"   Lag Year/Quarter: {lag_year}Q{lag_quarter}")
    
    # Get VNM data for different quarters and years
    query = text("""
        SELECT 
            fv.item_id,
            fv.year,
            fv.quarter,
            fv.value,
            fv.statement_type
        FROM fundamental_values fv
        WHERE fv.ticker = 'VNM'
        AND fv.item_id IN (1, 2, 3, 4, 15)  -- NetProfit, Sales candidates
        AND fv.statement_type = 'PL'
        AND fv.year >= 2024
        ORDER BY fv.year DESC, fv.quarter DESC, fv.item_id
    """)
    
    data = pd.read_sql(query, engine)
    
    print(f"\n📊 VNM Database Values (2024-2025):")
    print("-" * 80)
    
    # Group by item_id and show recent quarters
    for item_id in [1, 2, 3, 4, 15]:
        item_data = data[data['item_id'] == item_id].head(8)  # Last 8 quarters
        
        if not item_data.empty:
            print(f"\n   Item {item_id}:")
            for _, row in item_data.iterrows():
                year = int(row['year'])
                quarter = int(row['quarter'])
                value = row['value']
                
                # Convert using the non-bank factor
                converted_value = value / 6222444702.01
                
                print(f"     {year}Q{quarter}: {value:,.0f} → {converted_value:.2f} bn VND")
                
                # Check if this matches Q2 2025 sales
                if year == 2025 and quarter == 2:
                    if abs(converted_value - known_q2_2025_sales) < known_q2_2025_sales * 0.1:  # Within 10%
                        print(f"       ✅ MATCHES Q2 2025 Sales!")
                    elif abs(converted_value - known_q2_2025_sales * 4) < known_q2_2025_sales * 0.1:  # Within 10% of annual
                        print(f"       ✅ MATCHES Annual Sales (Q2 × 4)!")
    
    # Test TTM calculation
    print(f"\n🧪 Testing TTM Calculation:")
    print("-" * 50)
    
    # Get last 4 quarters for TTM
    ttm_query = text("""
        WITH quarterly_data AS (
            SELECT 
                fv.item_id,
                fv.year,
                fv.quarter,
                fv.value / 6222444702.01 as value_bn,
                ROW_NUMBER() OVER (PARTITION BY fv.item_id ORDER BY fv.year DESC, fv.quarter DESC) as rn
            FROM fundamental_values fv
            WHERE fv.ticker = 'VNM'
            AND fv.item_id IN (1, 2, 3, 4, 15)
            AND fv.statement_type = 'PL'
        )
        SELECT 
            item_id,
            SUM(value_bn) as ttm_value,
            COUNT(*) as quarters_count,
            GROUP_CONCAT(CONCAT(year, 'Q', quarter) ORDER BY year DESC, quarter DESC) as quarters
        FROM quarterly_data
        WHERE rn <= 4
        GROUP BY item_id
        ORDER BY item_id
    """)
    
    ttm_data = pd.read_sql(ttm_query, engine)
    
    print(f"   TTM Values (Last 4 Quarters):")
    for _, row in ttm_data.iterrows():
        item_id = int(row['item_id'])
        ttm_value = row['ttm_value']
        quarters_count = row['quarters_count']
        quarters = row['quarters']
        
        print(f"     Item {item_id}: {ttm_value:.2f} bn VND ({quarters_count} quarters: {quarters})")
        
        # Check if TTM matches annual sales
        if abs(ttm_value - known_q2_2025_sales * 4) < known_q2_2025_sales * 0.2:  # Within 20%
            print(f"       ✅ TTM matches Annual Sales (Q2 × 4)!")
        elif abs(ttm_value - known_q2_2025_sales) < known_q2_2025_sales * 0.2:  # Within 20%
            print(f"       ✅ TTM matches Quarterly Sales!")
    
    # Test different conversion scenarios
    print(f"\n🧮 Testing Different Conversion Scenarios:")
    print("-" * 60)
    
    # Get Q2 2025 data specifically
    q2_2025_query = text("""
        SELECT 
            fv.item_id,
            fv.value,
            fv.value / 6222444702.01 as converted_bn
        FROM fundamental_values fv
        WHERE fv.ticker = 'VNM'
        AND fv.item_id IN (1, 2, 3, 4, 15)
        AND fv.statement_type = 'PL'
        AND fv.year = 2025 AND fv.quarter = 2
        ORDER BY fv.item_id
    """)
    
    q2_2025_data = pd.read_sql(q2_2025_query, engine)
    
    print(f"   Q2 2025 Values:")
    for _, row in q2_2025_data.iterrows():
        item_id = int(row['item_id'])
        value = row['value']
        converted = row['converted_bn']
        
        print(f"     Item {item_id}: {value:,.0f} → {converted:.2f} bn VND")
        
        # Check different scenarios
        if abs(converted - known_q2_2025_sales) < known_q2_2025_sales * 0.1:
            print(f"       ✅ MATCHES Q2 2025 Sales!")
        elif abs(converted - known_q2_2025_sales * 4) < known_q2_2025_sales * 0.1:
            print(f"       ✅ MATCHES Annual Sales (Q2 × 4)!")
        elif abs(converted - known_q2_2025_sales / 4) < known_q2_2025_sales * 0.1:
            print(f"       ✅ MATCHES Monthly Sales (Q2 / 4)!")
    
    # Check if database stores annualized values
    print(f"\n🎯 Conclusion:")
    print("-" * 30)
    print(f"   If database stores QUARTERLY data:")
    print(f"     - Item 2 should be close to {known_q2_2025_sales:.2f} bn VND")
    print(f"     - TTM should be close to {known_q2_2025_sales * 4:.2f} bn VND")
    print(f"   If database stores ANNUAL data:")
    print(f"     - Item 2 should be close to {known_q2_2025_sales * 4:.2f} bn VND")
    print(f"     - TTM should be close to {known_q2_2025_sales * 4:.2f} bn VND")

if __name__ == "__main__":
    check_vnm_quarterly_vs_annual() 