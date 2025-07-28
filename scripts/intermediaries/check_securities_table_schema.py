#!/usr/bin/env python3
"""
Check Securities Table Schema
============================
Examines the current schema of intermediary_calculations_securities_cleaned table
to identify missing expense-related TTM columns.

Author: Database Schema Checker
Date: July 23, 2025
"""

import sys
import pymysql
import yaml
from pathlib import Path
from tabulate import tabulate

# Setup project path
def find_project_root():
    current = Path(__file__).parent
    while current != current.parent:
        if (current / 'config' / 'database.yml').exists():
            return current
        current = current.parent
    raise FileNotFoundError("Could not find project root with config/database.yml")

project_root = find_project_root()

def connect_to_database():
    """Create database connection"""
    try:
        config_path = project_root / 'config' / 'database.yml'
        with open(config_path, 'r') as f:
            db_config = yaml.safe_load(f)['production']
        
        connection = pymysql.connect(
            host=db_config['host'],
            user=db_config['username'],
            password=db_config['password'],
            database=db_config['schema_name'],
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        return connection
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

def check_table_schema():
    """Check the schema of intermediary_calculations_securities_cleaned table"""
    connection = connect_to_database()
    if not connection:
        return
    
    try:
        with connection.cursor() as cursor:
            # Check if table exists
            cursor.execute("""
                SELECT COUNT(*) as count 
                FROM information_schema.tables 
                WHERE table_schema = 'alphabeta' 
                AND table_name = 'intermediary_calculations_securities_cleaned'
            """)
            table_exists = cursor.fetchone()['count'] > 0
            
            if not table_exists:
                print("❌ Table 'intermediary_calculations_securities_cleaned' does not exist!")
                return
            
            print("✅ Table 'intermediary_calculations_securities_cleaned' exists")
            print("\n" + "="*80)
            print("CURRENT TABLE SCHEMA")
            print("="*80)
            
            # Get table schema
            cursor.execute("DESCRIBE intermediary_calculations_securities_cleaned")
            columns = cursor.fetchall()
            
            # Format schema info
            schema_data = []
            for col in columns:
                schema_data.append([
                    col['Field'],
                    col['Type'],
                    col['Null'],
                    col['Key'],
                    col['Default'],
                    col['Extra']
                ])
            
            print(tabulate(schema_data, 
                          headers=['Column', 'Type', 'Null', 'Key', 'Default', 'Extra'],
                          tablefmt='grid'))
            
            # Check for expense-related TTM columns specifically
            print("\n" + "="*80)
            print("EXPENSE TTM COLUMNS ANALYSIS")
            print("="*80)
            
            required_expense_columns = [
                'OperatingExpenses_TTM',
                'BrokerageExpenses_TTM',
                'AdvisoryExpenses_TTM',
                'CustodyServiceExpenses_TTM',
                'OtherOperatingExpenses_TTM',
                'TotalOperatingRevenue_TTM'
            ]
            
            existing_columns = [col['Field'] for col in columns]
            
            print("\nREQUIRED EXPENSE TTM COLUMNS:")
            print("-" * 50)
            
            missing_columns = []
            existing_expense_columns = []
            
            for col in required_expense_columns:
                if col in existing_columns:
                    print(f"✅ {col} - EXISTS")
                    existing_expense_columns.append(col)
                else:
                    print(f"❌ {col} - MISSING")
                    missing_columns.append(col)
            
            # Look for any expense-related columns that do exist
            print("\nEXISTING EXPENSE-RELATED COLUMNS:")
            print("-" * 50)
            expense_related = [col for col in existing_columns if 'expense' in col.lower() or 'cost' in col.lower()]
            
            if expense_related:
                for col in expense_related:
                    print(f"📋 {col}")
            else:
                print("❌ No expense-related columns found")
            
            # Look for TTM columns
            print("\nEXISTING TTM COLUMNS:")
            print("-" * 50)
            ttm_columns = [col for col in existing_columns if 'TTM' in col]
            
            if ttm_columns:
                for col in ttm_columns:
                    print(f"📋 {col}")
            else:
                print("❌ No TTM columns found")
            
            # Summary
            print("\n" + "="*80)
            print("SUMMARY")
            print("="*80)
            print(f"Total columns in table: {len(existing_columns)}")
            print(f"Required expense TTM columns: {len(required_expense_columns)}")
            print(f"Missing expense TTM columns: {len(missing_columns)}")
            print(f"Existing expense TTM columns: {len(existing_expense_columns)}")
            
            if missing_columns:
                print(f"\n❌ MISSING COLUMNS THAT NEED TO BE ADDED:")
                for col in missing_columns:
                    print(f"   - {col}")
                    
                print(f"\n🔧 SUGGESTED ALTER TABLE STATEMENTS:")
                for col in missing_columns:
                    print(f"   ALTER TABLE intermediary_calculations_securities_cleaned ADD COLUMN {col} DECIMAL(20,2) DEFAULT NULL;")
            else:
                print("\n✅ All required expense TTM columns are present!")
            
    except Exception as e:
        print(f"Error checking table schema: {e}")
    finally:
        connection.close()

if __name__ == "__main__":
    print("Securities Table Schema Checker")
    print("=" * 50)
    check_table_schema()