#!/usr/bin/env python3
"""
Find Net Profit and Key Financial Items
=======================================

This script searches for the actual item IDs that represent net profit and other
key financial metrics in the fundamental_values table.
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
# # FIND KEY FINANCIAL ITEMS

# %%
def find_net_profit_items():
    """Find item IDs that represent net profit."""
    print("🔍 Finding Net Profit Items...")
    
    try:
        # Create database connection
        db_manager = get_database_manager()
        engine = db_manager.get_engine()
        
        # Search for items that might represent net profit
        # Look for items with high negative values (expenses) and high positive values (income)
        
        # Check items with high positive values (likely income/profit)
        income_query = text("""
            SELECT 
                item_id,
                statement_type,
                COUNT(*) as record_count,
                COUNT(DISTINCT ticker) as ticker_count,
                AVG(value) as avg_value,
                MIN(value) as min_value,
                MAX(value) as max_value,
                SUM(CASE WHEN value > 0 THEN 1 ELSE 0 END) as positive_count,
                SUM(CASE WHEN value < 0 THEN 1 ELSE 0 END) as negative_count
            FROM fundamental_values
            WHERE year >= 2020
            AND statement_type = 'PL'
            GROUP BY item_id, statement_type
            HAVING COUNT(*) > 1000
            ORDER BY AVG(value) DESC
            LIMIT 20
        """)
        
        income_items = pd.read_sql(income_query, engine)
        print(f"\n💰 Top Income/Profit Items (PL):")
        for _, row in income_items.iterrows():
            print(f"   Item {row['item_id']}: {row['ticker_count']} tickers, avg={row['avg_value']:,.0f}, pos={row['positive_count']}, neg={row['negative_count']}")
        
        # Check items with high negative values (likely expenses)
        expense_query = text("""
            SELECT 
                item_id,
                statement_type,
                COUNT(*) as record_count,
                COUNT(DISTINCT ticker) as ticker_count,
                AVG(value) as avg_value,
                MIN(value) as min_value,
                MAX(value) as max_value,
                SUM(CASE WHEN value > 0 THEN 1 ELSE 0 END) as positive_count,
                SUM(CASE WHEN value < 0 THEN 1 ELSE 0 END) as negative_count
            FROM fundamental_values
            WHERE year >= 2020
            AND statement_type = 'PL'
            GROUP BY item_id, statement_type
            HAVING COUNT(*) > 1000
            ORDER BY AVG(value) ASC
            LIMIT 20
        """)
        
        expense_items = pd.read_sql(expense_query, engine)
        print(f"\n💸 Top Expense Items (PL):")
        for _, row in expense_items.iterrows():
            print(f"   Item {row['item_id']}: {row['ticker_count']} tickers, avg={row['avg_value']:,.0f}, pos={row['positive_count']}, neg={row['negative_count']}")
        
        # Look for items that appear at the bottom of income statements (likely net profit)
        # Check items with high coverage and positive values
        net_profit_candidates = income_items[income_items['positive_count'] > income_items['negative_count']]
        
        print(f"\n🎯 Net Profit Candidates:")
        for _, row in net_profit_candidates.head(10).iterrows():
            print(f"   Item {row['item_id']}: {row['ticker_count']} tickers, avg={row['avg_value']:,.0f}")
        
        # Check specific ranges for net profit items
        ranges_to_check = [
            (40, 50),   # Expected range
            (50, 60),   # Alternative range
            (60, 70),   # Another alternative
            (100, 110), # Balance sheet range
            (200, 210), # Another balance sheet range
            (300, 310)  # Equity range
        ]
        
        for start_id, end_id in ranges_to_check:
            range_query = text(f"""
                SELECT 
                    item_id,
                    statement_type,
                    COUNT(*) as record_count,
                    COUNT(DISTINCT ticker) as ticker_count,
                    AVG(value) as avg_value,
                    MIN(value) as min_value,
                    MAX(value) as max_value
                FROM fundamental_values
                WHERE year >= 2020
                AND item_id BETWEEN {start_id} AND {end_id}
                GROUP BY item_id, statement_type
                ORDER BY item_id
            """)
            
            range_items = pd.read_sql(range_query, engine)
            if not range_items.empty:
                print(f"\n🔍 Items {start_id}-{end_id}:")
                for _, row in range_items.iterrows():
                    print(f"   Item {row['item_id']} ({row['statement_type']}): {row['ticker_count']} tickers, avg={row['avg_value']:,.0f}")
        
        # Check for items with "net" or "profit" in their descriptions (if available)
        # Since we don't have descriptions, let's look for patterns
        
        # Look for items that are likely to be net profit based on their position and values
        # Items with high positive values and high coverage are likely net profit
        
        # Get sample data for top income items to understand the structure
        print(f"\n📋 Sample Data for Top Income Items:")
        for item_id in income_items['item_id'].head(5):
            sample_query = text(f"""
                SELECT ticker, year, quarter, value
                FROM fundamental_values
                WHERE item_id = {item_id}
                AND year = 2020
                AND value > 0
                ORDER BY value DESC
                LIMIT 5
            """)
            
            sample_data = pd.read_sql(sample_query, engine)
            if not sample_data.empty:
                print(f"\n   Item {item_id} sample values:")
                for _, row in sample_data.iterrows():
                    print(f"      {row['ticker']} {row['year']}Q{row['quarter']}: {row['value']:,.0f}")
        
        return income_items, expense_items
        
    except Exception as e:
        print(f"❌ Error finding net profit items: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def find_balance_sheet_items():
    """Find key balance sheet items."""
    print("\n🏢 Finding Balance Sheet Items...")
    
    try:
        db_manager = get_database_manager()
        engine = db_manager.get_engine()
        
        # Find total assets items
        assets_query = text("""
            SELECT 
                item_id,
                COUNT(*) as record_count,
                COUNT(DISTINCT ticker) as ticker_count,
                AVG(value) as avg_value,
                MIN(value) as min_value,
                MAX(value) as max_value
            FROM fundamental_values
            WHERE year >= 2020
            AND statement_type = 'BS'
            AND item_id BETWEEN 100 AND 200
            GROUP BY item_id
            HAVING COUNT(*) > 1000
            ORDER BY AVG(value) DESC
            LIMIT 10
        """)
        
        assets_items = pd.read_sql(assets_query, engine)
        print(f"\n📊 Top Assets Items:")
        for _, row in assets_items.iterrows():
            print(f"   Item {row['item_id']}: {row['ticker_count']} tickers, avg={row['avg_value']:,.0f}")
        
        # Find total liabilities items
        liabilities_query = text("""
            SELECT 
                item_id,
                COUNT(*) as record_count,
                COUNT(DISTINCT ticker) as ticker_count,
                AVG(value) as avg_value,
                MIN(value) as min_value,
                MAX(value) as max_value
            FROM fundamental_values
            WHERE year >= 2020
            AND statement_type = 'BS'
            AND item_id BETWEEN 200 AND 300
            GROUP BY item_id
            HAVING COUNT(*) > 1000
            ORDER BY AVG(value) DESC
            LIMIT 10
        """)
        
        liabilities_items = pd.read_sql(liabilities_query, engine)
        print(f"\n📊 Top Liabilities Items:")
        for _, row in liabilities_items.iterrows():
            print(f"   Item {row['item_id']}: {row['ticker_count']} tickers, avg={row['avg_value']:,.0f}")
        
        # Find total equity items
        equity_query = text("""
            SELECT 
                item_id,
                COUNT(*) as record_count,
                COUNT(DISTINCT ticker) as ticker_count,
                AVG(value) as avg_value,
                MIN(value) as min_value,
                MAX(value) as max_value
            FROM fundamental_values
            WHERE year >= 2020
            AND statement_type = 'BS'
            AND item_id BETWEEN 300 AND 400
            GROUP BY item_id
            HAVING COUNT(*) > 1000
            ORDER BY AVG(value) DESC
            LIMIT 10
        """)
        
        equity_items = pd.read_sql(equity_query, engine)
        print(f"\n📊 Top Equity Items:")
        for _, row in equity_items.iterrows():
            print(f"   Item {row['item_id']}: {row['ticker_count']} tickers, avg={row['avg_value']:,.0f}")
        
        return assets_items, liabilities_items, equity_items
        
    except Exception as e:
        print(f"❌ Error finding balance sheet items: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

# %% [markdown]
# # MAIN EXECUTION

# %%
if __name__ == "__main__":
    income_items, expense_items = find_net_profit_items()
    assets_items, liabilities_items, equity_items = find_balance_sheet_items()
    
    if income_items is not None:
        print(f"\n✅ Net profit items search completed!")
        print(f"   - Found {len(income_items)} potential income items")
        print(f"   - Found {len(expense_items)} potential expense items")
        
        # Suggest the most likely net profit items
        print(f"\n💡 SUGGESTED NET PROFIT ITEMS:")
        print(f"   Based on high coverage and positive values, the most likely net profit items are:")
        for _, row in income_items.head(3).iterrows():
            print(f"   - Item {row['item_id']}: {row['ticker_count']} tickers, avg={row['avg_value']:,.0f}")
    else:
        print(f"\n❌ Net profit items search failed!")
