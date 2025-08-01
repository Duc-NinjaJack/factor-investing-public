#!/usr/bin/env python3
"""
Map VCSC Items to Database Item IDs
==================================

This script maps the VCSC financial statement items to the actual item_ids
in the fundamental_values database table.
"""

import sys
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine, text
import json

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

def load_vcsc_mappings():
    """Load VCSC financial statement mappings."""
    mappings_path = project_root / 'production' / 'database' / 'mappings'
    
    # Load corporate mappings
    with open(mappings_path / 'corp_code_name_mapping.json', 'r') as f:
        corp_mappings = json.load(f)
    
    # Load bank mappings
    with open(mappings_path / 'bank_code_name_mapping.json', 'r') as f:
        bank_mappings = json.load(f)
    
    return corp_mappings, bank_mappings

def map_vcsc_items_to_database():
    """Map VCSC items to database item_ids."""
    print("🔍 Mapping VCSC Items to Database Item IDs")
    print("=" * 60)
    
    # Load mappings
    corp_mappings, bank_mappings = load_vcsc_mappings()
    
    # Connect to database
    db_manager = get_database_manager()
    engine = db_manager.get_engine()
    
    # Test ticker
    test_ticker = 'TCB'
    
    print(f"📊 Analyzing data for ticker: {test_ticker}")
    
    # Get all available item_ids for this ticker
    print(f"\n1️⃣ Available item_ids for {test_ticker}:")
    print("-" * 50)
    
    item_query = text("""
        SELECT DISTINCT 
            fv.item_id,
            fv.statement_type,
            COUNT(*) as count,
            MIN(fv.value) as min_value,
            MAX(fv.value) as max_value,
            AVG(fv.value) as avg_value
        FROM fundamental_values fv
        WHERE fv.ticker = :ticker
        GROUP BY fv.item_id, fv.statement_type
        ORDER BY fv.item_id
    """)
    
    item_data = pd.read_sql(item_query, engine, params={'ticker': test_ticker})
    print(f"Available item_ids for {test_ticker}:")
    print(item_data.to_string(index=False))
    
    # Based on the mappings, let's identify the correct items:
    print(f"\n2️⃣ Mapping Analysis:")
    print("-" * 50)
    
    # For banks, we need:
    # - Net Profit: isa21 (Net profit/(loss) after tax)
    # - Total Operating Income: isa13 (Total Operating Income) 
    # - Total Assets: bsa39 (TOTAL LIABILITIES - this seems wrong, should be Total Assets)
    
    # Let's check what the largest items are (likely to be Revenue/Total Assets)
    print(f"\n3️⃣ Largest Items (likely Revenue/Total Assets):")
    print("-" * 50)
    
    large_items_query = text("""
        SELECT 
            fv.item_id,
            fv.statement_type,
            COUNT(*) as count,
            MIN(fv.value) as min_value,
            MAX(fv.value) as max_value,
            AVG(fv.value) as avg_value
        FROM fundamental_values fv
        WHERE fv.ticker = :ticker
        AND fv.value > 0
        GROUP BY fv.item_id, fv.statement_type
        HAVING count >= 4
        ORDER BY avg_value DESC
        LIMIT 10
    """)
    
    large_items = pd.read_sql(large_items_query, engine, params={'ticker': test_ticker})
    print("Largest items (likely Revenue/Total Assets):")
    print(large_items.to_string(index=False))
    
    # Let's check what the medium items are (likely Net Profit)
    print(f"\n4️⃣ Medium Items (likely Net Profit):")
    print("-" * 50)
    
    medium_items_query = text("""
        SELECT 
            fv.item_id,
            fv.statement_type,
            COUNT(*) as count,
            MIN(fv.value) as min_value,
            MAX(fv.value) as max_value,
            AVG(fv.value) as avg_value
        FROM fundamental_values fv
        WHERE fv.ticker = :ticker
        AND fv.value > 0
        GROUP BY fv.item_id, fv.statement_type
        HAVING count >= 4
        ORDER BY avg_value DESC
        LIMIT 20
    """)
    
    medium_items = pd.read_sql(medium_items_query, engine, params={'ticker': test_ticker})
    print("Medium items (likely Net Profit):")
    print(medium_items.to_string(index=False))
    
    # Based on the data, let's make educated guesses:
    print(f"\n5️⃣ Educated Guesses:")
    print("-" * 50)
    
    # Looking at the data:
    # - Item 2 (BS) has the largest values (~4.2e14) - likely Total Assets
    # - Item 1 (PL) has medium values (~4.1e12) - likely Net Profit  
    # - Item 2 (PL) has smaller values (~1.0e12) - likely Revenue or Operating Income
    
    print("Based on the data analysis:")
    print("   - Total Assets: Item 2 (BS) - largest values (~4.2e14)")
    print("   - Net Profit: Item 1 (PL) - medium values (~4.1e12)")
    print("   - Revenue: Item 2 (PL) - smaller values (~1.0e12)")
    
    # Test this combination
    print(f"\n6️⃣ Testing Current Combination:")
    print("-" * 50)
    
    test_combinations = [
        (1, 2, 2, "Current: NetProfit(1) + Revenue(2) + TotalAssets(2)"),
        (1, 2, 2, "Alternative: Same but different logic"),
    ]
    
    for netprofit_id, revenue_id, totalassets_id, description in test_combinations:
        print(f"\n🔍 Testing: {description}")
        
        # Check if these items exist
        check_query = text("""
            SELECT 
                fv.item_id,
                fv.statement_type,
                COUNT(*) as count,
                AVG(fv.value) as avg_value
            FROM fundamental_values fv
            WHERE fv.ticker = :ticker
            AND (
                (fv.item_id = :netprofit_id AND fv.statement_type = 'PL')
                OR (fv.item_id = :revenue_id AND fv.statement_type = 'PL')
                OR (fv.item_id = :totalassets_id AND fv.statement_type = 'BS')
            )
            GROUP BY fv.item_id, fv.statement_type
        """)
        
        check_result = pd.read_sql(check_query, engine, params={
            'ticker': test_ticker,
            'netprofit_id': netprofit_id,
            'revenue_id': revenue_id,
            'totalassets_id': totalassets_id
        })
        
        if not check_result.empty:
            print(f"   ✅ Items found:")
            for _, row in check_result.iterrows():
                print(f"      Item {row['item_id']} ({row['statement_type']}): {row['count']} records, avg={row['avg_value']:.0f}")
        else:
            print(f"   ❌ No items found")

if __name__ == "__main__":
    map_vcsc_items_to_database() 