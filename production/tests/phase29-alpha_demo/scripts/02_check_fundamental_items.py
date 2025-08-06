#!/usr/bin/env python3
"""
Check Fundamental Items for Factor Calculation
==============================================

This script checks what item IDs are actually available in the fundamental_values table
to ensure the factor calculator uses the correct item mappings.
"""

# %% [markdown]
# # IMPORTS AND SETUP

# %%
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Database connectivity
from sqlalchemy import create_engine, text

# Add Project Root to Python Path
try:
    current_path = Path.cwd()
    while not (current_path / 'production').is_dir():
        if current_path.parent == current_path:
            raise FileNotFoundError("Could not find the 'production' directory.")
        current_path = current_path.parent
    
    project_root = current_path
    
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    from production.database.connection import get_database_manager
    print(f"✅ Successfully imported production modules.")
    print(f"   - Project Root set to: {project_root}")

except (ImportError, FileNotFoundError) as e:
    print(f"❌ ERROR: Could not import production modules. Please check your directory structure.")
    print(f"   - Final Path Searched: {project_root}")
    print(f"   - Error: {e}")
    raise

# %% [markdown]
# # CHECK FUNDAMENTAL ITEMS

# %%
def check_fundamental_items():
    """Check what item IDs are available in fundamental_values."""
    print("🔍 Checking Fundamental Items...")
    
    try:
        # Create database connection
        db_manager = get_database_manager()
        engine = db_manager.get_engine()
        
        # Check all available item IDs
        query = text("""
            SELECT 
                item_id,
                COUNT(*) as record_count,
                COUNT(DISTINCT ticker) as ticker_count,
                MIN(year) as min_year,
                MAX(year) as max_year
            FROM fundamental_values
            WHERE year >= 2020
            GROUP BY item_id
            ORDER BY item_id
        """)
        
        items = pd.read_sql(query, engine)
        print(f"📊 Found {len(items)} unique item IDs")
        
        # Show item distribution by statement type
        statement_query = text("""
            SELECT 
                statement_type,
                COUNT(DISTINCT item_id) as unique_items,
                COUNT(*) as total_records,
                COUNT(DISTINCT ticker) as unique_tickers
            FROM fundamental_values
            WHERE year >= 2020
            GROUP BY statement_type
            ORDER BY statement_type
        """)
        
        statements = pd.read_sql(statement_query, engine)
        print(f"\n📋 Statement Type Distribution:")
        for _, row in statements.iterrows():
            print(f"   {row['statement_type']}: {row['unique_items']} items, {row['total_records']:,} records, {row['unique_tickers']} tickers")
        
        # Show top items by record count
        print(f"\n📊 Top 20 Items by Record Count:")
        top_items = items.nlargest(20, 'record_count')
        for _, row in top_items.iterrows():
            print(f"   Item {row['item_id']}: {row['record_count']:,} records, {row['ticker_count']} tickers")
        
        # Check for specific key items
        key_items = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        print(f"\n🔍 Checking Key Items:")
        for item_id in key_items:
            item_data = items[items['item_id'] == item_id]
            if not item_data.empty:
                row = item_data.iloc[0]
                print(f"   Item {item_id}: {row['record_count']:,} records, {row['ticker_count']} tickers")
            else:
                print(f"   Item {item_id}: NOT FOUND")
        
        # Check for net profit items
        print(f"\n💰 Checking Net Profit Items:")
        net_profit_items = items[items['item_id'].isin([42, 43, 44, 45])]
        for _, row in net_profit_items.iterrows():
            print(f"   Item {row['item_id']}: {row['record_count']:,} records, {row['ticker_count']} tickers")
        
        # Check for total assets items
        print(f"\n🏢 Checking Total Assets Items:")
        total_assets_items = items[items['item_id'].isin([101, 102, 103, 104, 105])]
        for _, row in total_assets_items.iterrows():
            print(f"   Item {row['item_id']}: {row['record_count']:,} records, {row['ticker_count']} tickers")
        
        # Check for revenue items
        print(f"\n📈 Checking Revenue Items:")
        revenue_items = items[items['item_id'].isin([1, 2, 3, 4, 5])]
        for _, row in revenue_items.iterrows():
            print(f"   Item {row['item_id']}: {row['record_count']:,} records, {row['ticker_count']} tickers")
        
        # Check for operating cash flow items
        print(f"\n💸 Checking Operating Cash Flow Items:")
        ocf_items = items[items['item_id'].isin([401, 402, 403, 404, 405])]
        for _, row in ocf_items.iterrows():
            print(f"   Item {row['item_id']}: {row['record_count']:,} records, {row['ticker_count']} tickers")
        
        # Check for capital expenditure items
        print(f"\n🏗️ Checking Capital Expenditure Items:")
        capex_items = items[items['item_id'].isin([406, 407])]
        for _, row in capex_items.iterrows():
            print(f"   Item {row['item_id']}: {row['record_count']:,} records, {row['ticker_count']} tickers")
        
        # Find actual item IDs that exist
        print(f"\n✅ Available Item IDs Summary:")
        print(f"   - Total items: {len(items)}")
        print(f"   - Items with >1000 records: {len(items[items['record_count'] > 1000])}")
        print(f"   - Items with >100 tickers: {len(items[items['ticker_count'] > 100])}")
        
        # Show items with high coverage
        high_coverage = items[items['ticker_count'] > 100].sort_values('ticker_count', ascending=False)
        print(f"\n📊 Items with High Coverage (>100 tickers):")
        for _, row in high_coverage.head(20).iterrows():
            print(f"   Item {row['item_id']}: {row['ticker_count']} tickers, {row['record_count']:,} records")
        
        return items, statements
        
    except Exception as e:
        print(f"❌ Error checking fundamental items: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# %% [markdown]
# # MAIN EXECUTION

# %%
if __name__ == "__main__":
    items, statements = check_fundamental_items()
    
    if items is not None:
        print(f"\n✅ Fundamental items check completed!")
        print(f"   - Found {len(items)} unique item IDs")
        print(f"   - Data available from {items['min_year'].min()} to {items['max_year'].max()}")
    else:
        print(f"\n❌ Fundamental items check failed!")
