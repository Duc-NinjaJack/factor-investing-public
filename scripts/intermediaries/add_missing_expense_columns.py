#!/usr/bin/env python3
"""
Add Missing Expense Columns to Securities Table
===============================================
Adds the missing expense TTM columns to intermediary_calculations_securities_cleaned
table so that the calculator can function properly.

Missing columns to add:
- OperatingExpenses_TTM
- BrokerageExpenses_TTM  
- AdvisoryExpenses_TTM
- CustodyServiceExpenses_TTM
- OtherOperatingExpenses_TTM

Author: Database Schema Fixer
Date: July 23, 2025
"""

import sys
import pymysql
import yaml
from pathlib import Path

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

def add_missing_columns():
    """Add missing expense TTM columns to the securities table"""
    connection = connect_to_database()
    if not connection:
        return
    
    # Define the missing columns to add
    missing_columns = [
        'OperatingExpenses_TTM',
        'BrokerageExpenses_TTM',
        'AdvisoryExpenses_TTM',
        'CustodyServiceExpenses_TTM',
        'OtherOperatingExpenses_TTM'
    ]
    
    try:
        with connection.cursor() as cursor:
            print("Adding missing expense TTM columns to intermediary_calculations_securities_cleaned...")
            print("=" * 80)
            
            # Add each missing column
            for column in missing_columns:
                try:
                    alter_sql = f"""
                    ALTER TABLE intermediary_calculations_securities_cleaned 
                    ADD COLUMN {column} DECIMAL(30,2) DEFAULT NULL
                    """
                    
                    print(f"Adding column: {column}")
                    cursor.execute(alter_sql)
                    print(f"✅ Successfully added {column}")
                    
                except pymysql.err.OperationalError as e:
                    if "Duplicate column name" in str(e):
                        print(f"⚠️  Column {column} already exists, skipping...")
                    else:
                        print(f"❌ Error adding {column}: {e}")
                        raise
                except Exception as e:
                    print(f"❌ Error adding {column}: {e}")
                    raise
            
            # Commit the changes
            connection.commit()
            print("\n" + "=" * 80)
            print("✅ All missing columns have been successfully added!")
            print("=" * 80)
            
            # Verify the additions by checking the schema
            print("\nVerifying additions...")
            cursor.execute("DESCRIBE intermediary_calculations_securities_cleaned")
            columns = cursor.fetchall()
            
            existing_columns = [col['Field'] for col in columns]
            
            print("\nVerification Results:")
            print("-" * 50)
            for column in missing_columns:
                if column in existing_columns:
                    print(f"✅ {column} - CONFIRMED")
                else:
                    print(f"❌ {column} - MISSING")
            
            print(f"\nTotal columns in table: {len(existing_columns)}")
            
    except Exception as e:
        print(f"❌ Error during column addition: {e}")
        connection.rollback()
    finally:
        connection.close()

if __name__ == "__main__":
    print("Securities Table Schema Fixer")
    print("=" * 50)
    
    # Get user confirmation
    response = input("This will add 5 missing expense TTM columns to the securities table. Continue? (y/N): ")
    if response.lower() != 'y':
        print("Operation cancelled.")
        sys.exit(0)
    
    add_missing_columns()
    print("\nSchema update completed. The calculator should now work properly.")