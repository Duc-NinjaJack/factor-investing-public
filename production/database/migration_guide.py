"""
Database Migration Guide

This script demonstrates how to migrate existing code to use the new common
database connection module. It provides examples of before/after code patterns
and migration utilities.

Author: Factor Investing Team
Date: 2025-07-30
"""

import pandas as pd
from typing import Dict, Any, Optional
from pathlib import Path

from .connection import get_engine, get_connection, DatabaseManager
from .utils import execute_query, get_table_info, get_ticker_list

class DatabaseMigrationGuide:
    """
    Guide for migrating existing code to use the common database connection.
    """
    
    def __init__(self):
        self.examples = self._load_migration_examples()
    
    def _load_migration_examples(self) -> Dict[str, Dict[str, str]]:
        """Load migration examples."""
        return {
            'sqlalchemy_engine': {
                'before': '''
# OLD WAY - Direct SQLAlchemy engine creation
from sqlalchemy import create_engine, text
import yaml

# Load config manually
with open('config/database.yml', 'r') as f:
    db_config = yaml.safe_load(f)['production']

# Create engine manually
engine = create_engine(
    f"mysql+pymysql://{db_config['username']}:{db_config['password']}@"
    f"{db_config['host']}/{db_config['schema_name']}"
)

# Use engine
with engine.connect() as conn:
    result = conn.execute(text("SELECT * FROM master_info"))
    data = result.fetchall()
''',
                'after': '''
# NEW WAY - Common database connection
from production.database import get_engine

# Get engine automatically
engine = get_engine()

# Use engine (same as before)
with engine.connect() as conn:
    result = conn.execute(text("SELECT * FROM master_info"))
    data = result.fetchall()
'''
            },
            'pymysql_connection': {
                'before': '''
# OLD WAY - Direct PyMySQL connection
import pymysql
import yaml
from pathlib import Path

# Load config manually
config_path = Path(__file__).parent.parent.parent / 'config' / 'database.yml'
with open(config_path, 'r') as f:
    db_config = yaml.safe_load(f)['production']

# Create connection manually
connection = pymysql.connect(
    host=db_config['host'],
    user=db_config['username'],
    password=db_config['password'],
    database=db_config['schema_name'],
    charset='utf8mb4',
    cursorclass=pymysql.cursors.DictCursor
)

# Use connection
with connection.cursor() as cursor:
    cursor.execute("SELECT * FROM master_info")
    data = cursor.fetchall()
''',
                'after': '''
# NEW WAY - Common database connection
from production.database import get_connection

# Get connection automatically
connection = get_connection()

# Use connection (same as before)
with connection.cursor() as cursor:
    cursor.execute("SELECT * FROM master_info")
    data = cursor.fetchall()
'''
            },
            'query_execution': {
                'before': '''
# OLD WAY - Manual query execution
import pandas as pd
from sqlalchemy import text

# Execute query manually
query = "SELECT ticker, sector FROM master_info WHERE ticker IS NOT NULL"
df = pd.read_sql(text(query), engine)

# Execute with parameters
params = {'sector': 'Banking'}
query = "SELECT ticker FROM master_info WHERE sector = :sector"
df = pd.read_sql(text(query), engine, params=params)
''',
                'after': '''
# NEW WAY - Utility function
from production.database.utils import execute_query

# Execute query with utility function
query = "SELECT ticker, sector FROM master_info WHERE ticker IS NOT NULL"
df = execute_query(query)

# Execute with parameters
params = {'sector': 'Banking'}
query = "SELECT ticker FROM master_info WHERE sector = :sector"
df = execute_query(query, params=params)
'''
            },
            'common_operations': {
                'before': '''
# OLD WAY - Manual common operations
import pandas as pd
from sqlalchemy import text

# Get ticker list
query = "SELECT ticker FROM master_info WHERE ticker IS NOT NULL"
tickers_df = pd.read_sql(text(query), engine)
tickers = tickers_df['ticker'].tolist()

# Get sector mapping
query = "SELECT ticker, sector FROM master_info WHERE ticker IS NOT NULL"
sector_df = pd.read_sql(text(query), engine)
sector_df.loc[sector_df['sector'] == 'Banks', 'sector'] = 'Banking'

# Get table info
query = """
SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE
FROM INFORMATION_SCHEMA.COLUMNS 
WHERE TABLE_SCHEMA = DATABASE() 
AND TABLE_NAME = 'master_info'
"""
table_info = pd.read_sql(text(query), engine)
''',
                'after': '''
# NEW WAY - Utility functions
from production.database.utils import get_ticker_list, get_sector_mapping, get_table_info

# Get ticker list
tickers = get_ticker_list()

# Get sector mapping (with automatic Banks->Banking fix)
sector_df = get_sector_mapping()

# Get table info
table_info = get_table_info('master_info')
'''
            }
        }
    
    def show_migration_examples(self):
        """Display all migration examples."""
        print("=" * 80)
        print("DATABASE MIGRATION GUIDE")
        print("=" * 80)
        
        for example_name, example in self.examples.items():
            print(f"\n{example_name.upper().replace('_', ' ')}")
            print("-" * 60)
            print("BEFORE (Old Way):")
            print(example['before'])
            print("\nAFTER (New Way):")
            print(example['after'])
            print("\n" + "=" * 80)
    
    def generate_migration_script(self, file_path: str) -> str:
        """
        Generate a migration script for a specific file.
        
        Args:
            file_path: Path to the file to migrate
            
        Returns:
            Migration script content
        """
        # This would analyze the file and generate specific migration code
        # For now, return a template
        return f'''
# Migration script for {file_path}
# Generated on {pd.Timestamp.now()}

# Step 1: Add imports
from production.database import get_engine, get_connection
from production.database.utils import execute_query, get_ticker_list, get_sector_mapping

# Step 2: Replace engine creation
# OLD: engine = create_engine(...)
# NEW: engine = get_engine()

# Step 3: Replace connection creation  
# OLD: connection = pymysql.connect(...)
# NEW: connection = get_connection()

# Step 4: Replace query execution
# OLD: pd.read_sql(text(query), engine)
# NEW: execute_query(query)

# Step 5: Replace common operations
# OLD: Manual queries for common operations
# NEW: Use utility functions like get_ticker_list(), get_sector_mapping()
'''

def migrate_file(file_path: str, backup: bool = True) -> bool:
    """
    Migrate a single file to use the new database connection.
    
    Args:
        file_path: Path to the file to migrate
        backup: Whether to create a backup before migration
        
    Returns:
        True if migration successful, False otherwise
    """
    try:
        file_path = Path(file_path)
        
        if not file_path.exists():
            print(f"File not found: {file_path}")
            return False
        
        # Read original file
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Create backup if requested
        if backup:
            backup_path = file_path.with_suffix(f'{file_path.suffix}.backup')
            with open(backup_path, 'w') as f:
                f.write(content)
            print(f"Backup created: {backup_path}")
        
        # Perform migrations
        new_content = content
        
        # Replace imports
        if 'from sqlalchemy import create_engine' in content:
            new_content = new_content.replace(
                'from sqlalchemy import create_engine',
                'from production.database import get_engine'
            )
        
        if 'import pymysql' in content and 'pymysql.connect(' in content:
            new_content = new_content.replace(
                'import pymysql',
                'import pymysql\nfrom production.database import get_connection'
            )
        
        # Replace engine creation patterns
        # This is a simplified example - real migration would be more complex
        if 'create_engine(' in content:
            print("Warning: Manual review needed for create_engine() calls")
        
        # Write migrated file
        with open(file_path, 'w') as f:
            f.write(new_content)
        
        print(f"Migration completed: {file_path}")
        return True
        
    except Exception as e:
        print(f"Migration failed for {file_path}: {e}")
        return False

def test_migration():
    """Test the new database connection system."""
    print("Testing new database connection system...")
    
    try:
        # Test engine
        print("1. Testing SQLAlchemy engine...")
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1 as test"))
            print(f"   ✅ Engine test passed: {result.fetchone()}")
        
        # Test connection
        print("2. Testing PyMySQL connection...")
        connection = get_connection()
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1 as test")
            result = cursor.fetchone()
            print(f"   ✅ Connection test passed: {result}")
        
        # Test utility functions
        print("3. Testing utility functions...")
        tickers = get_ticker_list()
        print(f"   ✅ get_ticker_list(): {len(tickers)} tickers")
        
        sector_df = get_sector_mapping()
        print(f"   ✅ get_sector_mapping(): {len(sector_df)} rows")
        
        table_info = get_table_info('master_info')
        print(f"   ✅ get_table_info(): {len(table_info)} columns")
        
        print("\n✅ All tests passed! Migration system is ready.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False

def show_usage_examples():
    """Show usage examples for the new database system."""
    print("\n" + "=" * 80)
    print("USAGE EXAMPLES")
    print("=" * 80)
    
    examples = [
        {
            'title': 'Basic Engine Usage',
            'code': '''
from production.database import get_engine
from sqlalchemy import text

engine = get_engine()
with engine.connect() as conn:
    result = conn.execute(text("SELECT * FROM master_info LIMIT 5"))
    data = result.fetchall()
'''
        },
        {
            'title': 'Basic Connection Usage',
            'code': '''
from production.database import get_connection

connection = get_connection()
with connection.cursor() as cursor:
    cursor.execute("SELECT * FROM master_info LIMIT 5")
    data = cursor.fetchall()
'''
        },
        {
            'title': 'Utility Functions',
            'code': '''
from production.database.utils import execute_query, get_ticker_list, get_sector_mapping

# Execute custom query
df = execute_query("SELECT * FROM master_info WHERE sector = 'Banking'")

# Get common data
tickers = get_ticker_list()
sector_df = get_sector_mapping()
'''
        },
        {
            'title': 'Advanced Usage with DatabaseManager',
            'code': '''
from production.database import DatabaseManager

# Create manager with custom settings
db_manager = DatabaseManager(
    environment='production',
    enable_pooling=True,
    pool_size=20
)

# Use context managers
with db_manager.get_engine_context() as engine:
    # Use engine
    pass

with db_manager.get_connection_context() as connection:
    # Use connection
    pass
'''
        }
    ]
    
    for example in examples:
        print(f"\n{example['title']}")
        print("-" * 40)
        print(example['code'])

if __name__ == "__main__":
    # Show migration guide
    guide = DatabaseMigrationGuide()
    guide.show_migration_examples()
    
    # Show usage examples
    show_usage_examples()
    
    # Test the system
    print("\n" + "=" * 80)
    print("TESTING MIGRATION SYSTEM")
    print("=" * 80)
    test_migration()