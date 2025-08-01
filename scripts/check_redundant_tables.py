#!/usr/bin/env python3
"""
Check Redundant Tables in alphabeta Database
===========================================

This script analyzes the alphabeta database to identify redundant tables
that can be dropped to save space. It looks for:

1. Tables with similar names that might be duplicates
2. Backup tables (with _backup_ in name)
3. Old version tables (with _v1, _v2, etc.)
4. Test tables (with _test in name)
5. Tables with very similar schemas
6. Tables with overlapping data

Usage:
    python scripts/check_redundant_tables.py
"""

import sys
import os
import pandas as pd
from typing import Dict, List, Tuple, Any
import re

# Add the parent directory to the path to import database modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from production.database.utils import get_engine, execute_query, get_database_stats, get_table_info, get_table_row_count, get_table_date_range

def get_all_tables(engine) -> pd.DataFrame:
    """Get all tables in the database with their sizes."""
    query = """
    SELECT 
        table_name,
        ROUND(((data_length + index_length) / 1024 / 1024), 2) AS 'size_mb',
        table_rows,
        ROUND((data_length / 1024 / 1024), 2) AS 'data_size_mb',
        ROUND((index_length / 1024 / 1024), 2) AS 'index_size_mb'
    FROM information_schema.tables 
    WHERE table_schema = DATABASE()
    ORDER BY (data_length + index_length) DESC
    """
    result = execute_query(query, engine=engine)
    # Print column names for debugging
    print(f"Debug: Column names in result: {list(result.columns)}")
    return result

def identify_backup_tables(tables_df: pd.DataFrame) -> List[str]:
    """Identify backup tables (containing _backup_ in name)."""
    backup_tables = []
    for _, row in tables_df.iterrows():
        table_name = row['TABLE_NAME']
        if '_backup_' in table_name:
            backup_tables.append(table_name)
    return backup_tables

def identify_version_tables(tables_df: pd.DataFrame) -> List[str]:
    """Identify version tables (containing _v1, _v2, etc.)."""
    version_tables = []
    for _, row in tables_df.iterrows():
        table_name = row['TABLE_NAME']
        if re.search(r'_v\d+', table_name):
            version_tables.append(table_name)
    return version_tables

def identify_test_tables(tables_df: pd.DataFrame) -> List[str]:
    """Identify test tables (containing _test in name)."""
    test_tables = []
    for _, row in tables_df.iterrows():
        table_name = row['TABLE_NAME']
        if '_test' in table_name:
            test_tables.append(table_name)
    return test_tables

def identify_similar_named_tables(tables_df: pd.DataFrame) -> List[List[str]]:
    """Identify tables with similar names that might be duplicates."""
    similar_groups = []
    table_names = tables_df['TABLE_NAME'].tolist()
    
    # Group by base name (removing version suffixes, backup suffixes, etc.)
    base_names = {}
    for table_name in table_names:
        # Remove common suffixes
        base_name = re.sub(r'_backup_\d{8}_\d{6}', '', table_name)
        base_name = re.sub(r'_v\d+', '', base_name)
        base_name = re.sub(r'_test', '', base_name)
        base_name = re.sub(r'_old', '', base_name)
        base_name = re.sub(r'_new', '', base_name)
        
        if base_name not in base_names:
            base_names[base_name] = []
        base_names[base_name].append(table_name)
    
    # Find groups with multiple tables
    for base_name, tables in base_names.items():
        if len(tables) > 1:
            similar_groups.append(tables)
    
    return similar_groups

def compare_table_schemas(table1: str, table2: str, engine) -> Dict[str, Any]:
    """Compare schemas of two tables."""
    try:
        schema1 = get_table_info(table1, engine)
        schema2 = get_table_info(table2, engine)
        
        # Get column names
        cols1 = set(schema1['COLUMN_NAME'].tolist())
        cols2 = set(schema2['COLUMN_NAME'].tolist())
        
        # Calculate similarity metrics
        common_cols = cols1.intersection(cols2)
        total_cols = cols1.union(cols2)
        similarity = len(common_cols) / len(total_cols) if total_cols else 0
        
        return {
            'table1': table1,
            'table2': table2,
            'common_columns': len(common_cols),
            'total_columns': len(total_cols),
            'similarity_score': similarity,
            'common_columns_list': list(common_cols),
            'unique_to_table1': list(cols1 - cols2),
            'unique_to_table2': list(cols2 - cols1)
        }
    except Exception as e:
        return {
            'table1': table1,
            'table2': table2,
            'error': str(e)
        }

def find_schema_similar_tables(tables_df: pd.DataFrame, engine, similarity_threshold: float = 0.8) -> List[Dict[str, Any]]:
    """Find tables with very similar schemas."""
    similar_schemas = []
    table_names = tables_df['TABLE_NAME'].tolist()
    
    # Focus on intermediary tables and factor tables
    target_tables = [t for t in table_names if any(keyword in t.lower() for keyword in 
                   ['intermediary', 'factor', 'calculation', 'score'])]
    
    print(f"Comparing schemas for {len(target_tables)} target tables...")
    
    for i, table1 in enumerate(target_tables):
        for j, table2 in enumerate(target_tables[i+1:], i+1):
            comparison = compare_table_schemas(table1, table2, engine)
            
            if 'error' not in comparison and comparison['similarity_score'] >= similarity_threshold:
                similar_schemas.append(comparison)
    
    return similar_schemas

def check_table_data_overlap(table1: str, table2: str, engine) -> Dict[str, Any]:
    """Check if two tables have overlapping data."""
    try:
        # Get row counts
        count1 = get_table_row_count(table1, engine)
        count2 = get_table_row_count(table2, engine)
        
        # Check if they have similar row counts (within 10%)
        count_similarity = min(count1, count2) / max(count1, count2) if max(count1, count2) > 0 else 0
        
        # Check if they have similar date ranges (if they have date columns)
        date_range1 = get_table_date_range(table1, 'calc_date', engine) if 'calc_date' in get_table_info(table1, engine)['COLUMN_NAME'].tolist() else None
        date_range2 = get_table_date_range(table2, 'calc_date', engine) if 'calc_date' in get_table_info(table2, engine)['COLUMN_NAME'].tolist() else None
        
        return {
            'table1': table1,
            'table2': table2,
            'count1': count1,
            'count2': count2,
            'count_similarity': count_similarity,
            'date_range1': date_range1,
            'date_range2': date_range2,
            'potential_overlap': count_similarity > 0.9
        }
    except Exception as e:
        return {
            'table1': table1,
            'table2': table2,
            'error': str(e)
        }

def analyze_redundant_tables():
    """Main function to analyze redundant tables."""
    print("🔍 Analyzing alphabeta database for redundant tables...")
    print("=" * 60)
    
    engine = get_engine()
    
    # Get all tables
    tables_df = get_all_tables(engine)
    print(f"📊 Found {len(tables_df)} tables in database")
    print(f"📊 Total database size: {tables_df['size_mb'].sum():.2f} MB")
    print()
    
    # 1. Identify backup tables
    backup_tables = identify_backup_tables(tables_df)
    if backup_tables:
        print("🔴 BACKUP TABLES (can be dropped):")
        backup_size = 0
        for table in backup_tables:
            size = tables_df[tables_df['TABLE_NAME'] == table]['size_mb'].iloc[0]
            backup_size += size
            print(f"  • {table} ({size:.2f} MB)")
        print(f"  Total backup tables size: {backup_size:.2f} MB")
        print()
    
    # 2. Identify version tables
    version_tables = identify_version_tables(tables_df)
    if version_tables:
        print("🟡 VERSION TABLES (review for redundancy):")
        version_size = 0
        for table in version_tables:
            size = tables_df[tables_df['TABLE_NAME'] == table]['size_mb'].iloc[0]
            version_size += size
            print(f"  • {table} ({size:.2f} MB)")
        print(f"  Total version tables size: {version_size:.2f} MB")
        print()
    
    # 3. Identify test tables
    test_tables = identify_test_tables(tables_df)
    if test_tables:
        print("🟠 TEST TABLES (can be dropped):")
        test_size = 0
        for table in test_tables:
            size = tables_df[tables_df['TABLE_NAME'] == table]['size_mb'].iloc[0]
            test_size += size
            print(f"  • {table} ({size:.2f} MB)")
        print(f"  Total test tables size: {test_size:.2f} MB")
        print()
    
    # 4. Identify similar named tables
    similar_groups = identify_similar_named_tables(tables_df)
    if similar_groups:
        print("🟢 SIMILAR NAMED TABLES (potential duplicates):")
        for group in similar_groups:
            group_size = 0
            print(f"  Group: {group[0].split('_')[0] if '_' in group[0] else group[0]}")
            for table in group:
                size = tables_df[tables_df['TABLE_NAME'] == table]['size_mb'].iloc[0]
                group_size += size
                print(f"    • {table} ({size:.2f} MB)")
            print(f"    Group total: {group_size:.2f} MB")
        print()
    
    # 5. Find schema similar tables
    print("🔍 Analyzing table schemas for similarities...")
    similar_schemas = find_schema_similar_tables(tables_df, engine)
    if similar_schemas:
        print("🟣 SCHEMA SIMILAR TABLES (high similarity score):")
        for comparison in similar_schemas:
            print(f"  • {comparison['table1']} vs {comparison['table2']}")
            print(f"    Similarity: {comparison['similarity_score']:.2%}")
            print(f"    Common columns: {comparison['common_columns']}/{comparison['total_columns']}")
            if comparison['unique_to_table1']:
                print(f"    Unique to {comparison['table1']}: {comparison['unique_to_table1']}")
            if comparison['unique_to_table2']:
                print(f"    Unique to {comparison['table2']}: {comparison['unique_to_table2']}")
            print()
    
    # 6. Check data overlap for similar tables
    print("🔍 Checking data overlap for similar tables...")
    overlap_checks = []
    for comparison in similar_schemas[:5]:  # Limit to first 5 to avoid too many queries
        overlap = check_table_data_overlap(comparison['table1'], comparison['table2'], engine)
        if 'error' not in overlap and overlap['potential_overlap']:
            overlap_checks.append(overlap)
    
    if overlap_checks:
        print("🟤 TABLES WITH DATA OVERLAP:")
        for overlap in overlap_checks:
            print(f"  • {overlap['table1']} vs {overlap['table2']}")
            print(f"    Row counts: {overlap['count1']} vs {overlap['count2']}")
            print(f"    Count similarity: {overlap['count_similarity']:.2%}")
            print()
    
    # Summary
    total_potential_savings = 0
    if backup_tables:
        total_potential_savings += sum(tables_df[tables_df['TABLE_NAME'].isin(backup_tables)]['size_mb'])
    if test_tables:
        total_potential_savings += sum(tables_df[tables_df['TABLE_NAME'].isin(test_tables)]['size_mb'])
    
    print("📋 SUMMARY:")
    print(f"  • Backup tables: {len(backup_tables)}")
    print(f"  • Version tables: {len(version_tables)}")
    print(f"  • Test tables: {len(test_tables)}")
    print(f"  • Similar named groups: {len(similar_groups)}")
    print(f"  • Schema similar pairs: {len(similar_schemas)}")
    print(f"  • Data overlap pairs: {len(overlap_checks)}")
    print(f"  • Potential space savings: {total_potential_savings:.2f} MB")
    
    return {
        'backup_tables': backup_tables,
        'version_tables': version_tables,
        'test_tables': test_tables,
        'similar_groups': similar_groups,
        'similar_schemas': similar_schemas,
        'overlap_checks': overlap_checks,
        'total_potential_savings': total_potential_savings
    }

if __name__ == "__main__":
    try:
        results = analyze_redundant_tables()
        print("\n✅ Analysis complete!")
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        sys.exit(1) 