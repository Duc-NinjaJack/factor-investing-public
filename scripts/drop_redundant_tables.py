#!/usr/bin/env python3
"""
Drop Redundant Tables in alphabeta Database
==========================================

This script safely drops redundant tables identified by the analysis.
It includes safety checks and confirmation prompts.

Usage:
    python scripts/drop_redundant_tables.py
"""

import sys
import os
import pandas as pd
from typing import List, Dict, Any
from datetime import datetime

# Add the parent directory to the path to import database modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from production.database.utils import get_engine, execute_query, backup_table

def get_tables_to_drop() -> Dict[str, List[str]]:
    """Define tables that can be safely dropped based on analysis."""
    return {
        'backup_tables': [
            'factor_values_backup_20250626_213512',
            'factor_scores_qvm_backup_20250721', 
            'master_info_backup_20250621'
        ],
        'legacy_tables': [
            'factor_values_legacy_20250625',
            'factor_values_temp'
        ],
        'duplicate_tables': [
            'factor_values_legacy'  # This appears to be superseded by factor_values
        ]
    }

def get_table_sizes(engine, table_names: List[str]) -> Dict[str, float]:
    """Get sizes of specified tables."""
    if not table_names:
        return {}
    
    # Build query with table names directly (safe since these are predefined)
    table_list = "', '".join(table_names)
    query = f"""
    SELECT 
        table_name,
        ROUND(((data_length + index_length) / 1024 / 1024), 2) AS 'size_mb'
    FROM information_schema.tables 
    WHERE table_schema = DATABASE()
    AND table_name IN ('{table_list}')
    """
    
    result = execute_query(query, engine=engine)
    return dict(zip(result['TABLE_NAME'], result['size_mb']))

def check_table_exists(engine, table_name: str) -> bool:
    """Check if a table exists."""
    query = f"""
    SELECT COUNT(*) as count
    FROM information_schema.tables 
    WHERE table_schema = DATABASE()
    AND table_name = '{table_name}'
    """
    result = execute_query(query, engine=engine)
    return result['count'].iloc[0] > 0

def drop_table_safely(engine, table_name: str, dry_run: bool = True) -> bool:
    """Safely drop a table with backup option."""
    try:
        if not check_table_exists(engine, table_name):
            print(f"  ⚠️  Table {table_name} does not exist, skipping...")
            return True
        
        if dry_run:
            print(f"  🔍 [DRY RUN] Would drop table: {table_name}")
            return True
        
        # Create backup before dropping (optional safety measure)
        backup_name = f"{table_name}_pre_drop_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"  💾 Creating backup: {backup_name}")
        backup_table(table_name, backup_name, engine)
        
        # Drop the table
        drop_query = f"DROP TABLE {table_name}"
        execute_query(drop_query, engine=engine)
        print(f"  ✅ Successfully dropped table: {table_name}")
        return True
        
    except Exception as e:
        print(f"  ❌ Failed to drop table {table_name}: {e}")
        return False

def main():
    """Main function to drop redundant tables."""
    print("🗑️  Dropping Redundant Tables in alphabeta Database")
    print("=" * 60)
    
    engine = get_engine()
    
    # Get tables to drop
    tables_to_drop = get_tables_to_drop()
    
    # Calculate total potential savings
    all_tables = []
    for category, tables in tables_to_drop.items():
        all_tables.extend(tables)
    
    if not all_tables:
        print("✅ No redundant tables identified for dropping.")
        return
    
    # Get sizes of tables to be dropped
    table_sizes = get_table_sizes(engine, all_tables)
    total_savings = sum(table_sizes.values())
    
    print(f"📊 Tables identified for dropping:")
    print(f"  • Backup tables: {len(tables_to_drop['backup_tables'])}")
    print(f"  • Legacy tables: {len(tables_to_drop['legacy_tables'])}")
    print(f"  • Duplicate tables: {len(tables_to_drop['duplicate_tables'])}")
    print(f"  • Total potential savings: {total_savings:.2f} MB")
    print()
    
    # Show detailed breakdown
    for category, tables in tables_to_drop.items():
        if tables:
            print(f"📋 {category.upper().replace('_', ' ')}:")
            category_size = 0
            for table in tables:
                size = table_sizes.get(table, 0)
                category_size += size
                print(f"  • {table} ({size:.2f} MB)")
            print(f"  Category total: {category_size:.2f} MB")
            print()
    
    # Ask for confirmation
    print("⚠️  WARNING: This will permanently delete tables from the database!")
    print("   Make sure you have backups if needed.")
    print()
    
    response = input("Do you want to proceed with dropping these tables? (yes/no): ").lower().strip()
    if response not in ['yes', 'y']:
        print("❌ Operation cancelled.")
        return
    
    # Ask for dry run first
    print("\n🔍 Performing dry run first...")
    dry_run = True
    success_count = 0
    total_count = len(all_tables)
    
    for category, tables in tables_to_drop.items():
        if tables:
            print(f"\n📋 Processing {category}:")
            for table in tables:
                if drop_table_safely(engine, table, dry_run):
                    success_count += 1
    
    print(f"\n📊 Dry run completed: {success_count}/{total_count} tables would be dropped")
    
    # Ask for actual execution
    response = input("\nDo you want to proceed with the actual deletion? (yes/no): ").lower().strip()
    if response not in ['yes', 'y']:
        print("❌ Actual deletion cancelled.")
        return
    
    # Perform actual deletion
    print("\n🗑️  Performing actual deletion...")
    dry_run = False
    success_count = 0
    failed_tables = []
    
    for category, tables in tables_to_drop.items():
        if tables:
            print(f"\n📋 Processing {category}:")
            for table in tables:
                if drop_table_safely(engine, table, dry_run):
                    success_count += 1
                else:
                    failed_tables.append(table)
    
    # Summary
    print(f"\n📋 FINAL SUMMARY:")
    print(f"  • Successfully dropped: {success_count}/{total_count} tables")
    print(f"  • Failed to drop: {len(failed_tables)} tables")
    if failed_tables:
        print(f"  • Failed tables: {', '.join(failed_tables)}")
    
    actual_savings = sum(table_sizes.get(table, 0) for table in all_tables if table not in failed_tables)
    print(f"  • Actual space saved: {actual_savings:.2f} MB")
    
    if success_count > 0:
        print(f"\n✅ Successfully freed up {actual_savings:.2f} MB of database space!")
    else:
        print(f"\n❌ No tables were dropped.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error during table dropping: {e}")
        sys.exit(1) 