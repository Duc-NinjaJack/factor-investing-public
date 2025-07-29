# Common Database Connection System

## Overview

This module provides a unified database connection interface for all codes in the factor investing project. It supports both SQLAlchemy engines and PyMySQL connections with proper configuration management, connection pooling, and error handling.

## Features

- **Unified Interface**: Single interface for both SQLAlchemy and PyMySQL connections
- **Automatic Configuration**: Loads database configuration from `config/database.yml`
- **Connection Pooling**: Efficient connection management with pooling
- **Environment Support**: Production and development environment configurations
- **Error Handling**: Comprehensive error handling and logging
- **Utility Functions**: Common database operations as utility functions
- **Context Managers**: Safe connection handling with context managers

## Quick Start

### Basic Usage

```python
# Get SQLAlchemy engine
from production.database import get_engine
engine = get_engine()

# Get PyMySQL connection
from production.database import get_connection
connection = get_connection()

# Use utility functions
from production.database.utils import execute_query, get_ticker_list
df = execute_query("SELECT * FROM master_info")
tickers = get_ticker_list()
```

### Advanced Usage

```python
from production.database import DatabaseManager

# Create manager with custom settings
db_manager = DatabaseManager(
    environment='production',
    enable_pooling=True,
    pool_size=20
)

# Use context managers
with db_manager.get_engine_context() as engine:
    # Use engine safely
    pass

with db_manager.get_connection_context() as connection:
    # Use connection safely
    pass
```

## Installation

The module is part of the production package and requires the following dependencies:

```bash
pip install sqlalchemy pymysql pyyaml pandas numpy
```

## Configuration

The system automatically loads configuration from `config/database.yml`:

```yaml
production:
  host: localhost
  schema_name: alphabeta
  username: root
  password: "12345678"

development:
  host: localhost
  schema_name: alphabeta
  username: root
  password: "12345678"
```

## API Reference

### Core Functions

#### `get_engine(environment='production', **kwargs)`
Get SQLAlchemy engine with connection pooling.

**Parameters:**
- `environment`: Environment to use ('production' or 'development')
- `**kwargs`: Additional arguments for DatabaseManager

**Returns:**
- SQLAlchemy engine

#### `get_connection(environment='production', **kwargs)`
Get PyMySQL connection.

**Parameters:**
- `environment`: Environment to use ('production' or 'development')
- `**kwargs`: Additional arguments for DatabaseManager

**Returns:**
- PyMySQL connection

#### `test_connection(environment='production', **kwargs)`
Test database connectivity.

**Parameters:**
- `environment`: Environment to use ('production' or 'development')
- `**kwargs`: Additional arguments for DatabaseManager

**Returns:**
- True if connection successful, False otherwise

### DatabaseManager Class

#### `DatabaseManager(config_path=None, environment='production', enable_pooling=True, pool_size=10, max_overflow=20, pool_timeout=30, pool_recycle=3600)`

**Parameters:**
- `config_path`: Path to database configuration file
- `environment`: Environment to use ('production' or 'development')
- `enable_pooling`: Whether to enable connection pooling
- `pool_size`: Number of connections to maintain in pool
- `max_overflow`: Maximum number of connections beyond pool_size
- `pool_timeout`: Timeout for getting connection from pool
- `pool_recycle`: Recycle connections after this many seconds

#### Methods

- `get_engine(force_new=False)`: Get SQLAlchemy engine
- `get_connection(force_new=False)`: Get PyMySQL connection
- `get_engine_context()`: Context manager for SQLAlchemy engine
- `get_connection_context()`: Context manager for PyMySQL connection
- `test_connection()`: Test database connectivity
- `close_all_connections()`: Close all cached connections
- `get_config()`: Get current database configuration

### Utility Functions

#### `execute_query(query, params=None, engine=None, return_dataframe=True)`
Execute a SQL query and return results.

**Parameters:**
- `query`: SQL query string
- `params`: Query parameters (for parameterized queries)
- `engine`: SQLAlchemy engine (if None, uses default)
- `return_dataframe`: Whether to return pandas DataFrame or list of dicts

**Returns:**
- Query results as DataFrame or list of dictionaries

#### `get_ticker_list(table_name='master_info', active_only=True, engine=None)`
Get list of tickers from master_info table.

**Parameters:**
- `table_name`: Name of the table (default: master_info)
- `active_only`: Whether to return only active tickers
- `engine`: SQLAlchemy engine (if None, uses default)

**Returns:**
- List of ticker symbols

#### `get_sector_mapping(engine=None)`
Get sector mapping for all tickers.

**Parameters:**
- `engine`: SQLAlchemy engine (if None, uses default)

**Returns:**
- DataFrame with ticker and sector columns

#### `get_price_data(tickers, start_date, end_date, table_name='vcsc_daily_data_complete', engine=None)`
Get price data for specified tickers and date range.

**Parameters:**
- `tickers`: List of ticker symbols
- `start_date`: Start date (YYYY-MM-DD)
- `end_date`: End date (YYYY-MM-DD)
- `table_name`: Name of the price data table
- `engine`: SQLAlchemy engine (if None, uses default)

**Returns:**
- DataFrame with price data

#### `get_factor_scores(tickers, rebalance_date, table_name='factor_scores_qvm', engine=None)`
Get factor scores for specified tickers and rebalance date.

**Parameters:**
- `tickers`: List of ticker symbols
- `rebalance_date`: Rebalance date (YYYY-MM-DD)
- `table_name`: Name of the factor scores table
- `engine`: SQLAlchemy engine (if None, uses default)

**Returns:**
- DataFrame with factor scores

#### `get_liquid_universe(analysis_date, adtv_threshold=10.0, lookback_days=63, top_n=200, min_trading_coverage=0.6, engine=None)`
Get liquid universe based on ADTV criteria.

**Parameters:**
- `analysis_date`: Analysis date (YYYY-MM-DD)
- `adtv_threshold`: ADTV threshold in billions VND
- `lookback_days`: Number of days to look back for ADTV calculation
- `top_n`: Maximum number of stocks to return
- `min_trading_coverage`: Minimum trading coverage requirement
- `engine`: SQLAlchemy engine (if None, uses default)

**Returns:**
- DataFrame with liquid universe information

## Migration Guide

### From Direct SQLAlchemy Usage

**Before:**
```python
from sqlalchemy import create_engine, text
import yaml

with open('config/database.yml', 'r') as f:
    db_config = yaml.safe_load(f)['production']

engine = create_engine(
    f"mysql+pymysql://{db_config['username']}:{db_config['password']}@"
    f"{db_config['host']}/{db_config['schema_name']}"
)
```

**After:**
```python
from production.database import get_engine

engine = get_engine()
```

### From Direct PyMySQL Usage

**Before:**
```python
import pymysql
import yaml

with open('config/database.yml', 'r') as f:
    db_config = yaml.safe_load(f)['production']

connection = pymysql.connect(
    host=db_config['host'],
    user=db_config['username'],
    password=db_config['password'],
    database=db_config['schema_name']
)
```

**After:**
```python
from production.database import get_connection

connection = get_connection()
```

### From Manual Query Execution

**Before:**
```python
import pandas as pd
from sqlalchemy import text

query = "SELECT ticker, sector FROM master_info WHERE ticker IS NOT NULL"
df = pd.read_sql(text(query), engine)
```

**After:**
```python
from production.database.utils import execute_query

query = "SELECT ticker, sector FROM master_info WHERE ticker IS NOT NULL"
df = execute_query(query)
```

## Examples

### Basic Database Operations

```python
from production.database import get_engine, get_connection
from production.database.utils import execute_query, get_ticker_list, get_sector_mapping

# Get engine and connection
engine = get_engine()
connection = get_connection()

# Execute custom query
df = execute_query("SELECT * FROM master_info WHERE sector = 'Banking'")

# Get common data
tickers = get_ticker_list()
sector_df = get_sector_mapping()

# Get price data
price_df = get_price_data(
    tickers=['VNM', 'VCB', 'TCB'],
    start_date='2024-01-01',
    end_date='2024-12-31'
)
```

### Advanced Usage with Context Managers

```python
from production.database import DatabaseManager

# Create manager
db_manager = DatabaseManager(
    environment='production',
    enable_pooling=True,
    pool_size=20
)

# Use context managers for safe connection handling
with db_manager.get_engine_context() as engine:
    # Execute queries with engine
    df = execute_query("SELECT COUNT(*) FROM master_info", engine=engine)

with db_manager.get_connection_context() as connection:
    # Execute queries with connection
    with connection.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM master_info")
        count = cursor.fetchone()
```

### Batch Operations

```python
from production.database.utils import batch_query_executor

# Execute multiple queries in batches
queries = [
    "SELECT COUNT(*) FROM master_info",
    "SELECT COUNT(*) FROM vcsc_daily_data_complete",
    "SELECT COUNT(*) FROM factor_scores_qvm"
]

results = batch_query_executor(queries, batch_size=10)
```

### Table Management

```python
from production.database.utils import (
    get_table_info, 
    get_table_row_count, 
    create_table_if_not_exists,
    backup_table
)

# Get table information
table_info = get_table_info('master_info')
row_count = get_table_row_count('master_info')

# Create table if not exists
create_sql = """
CREATE TABLE IF NOT EXISTS test_table (
    id INT PRIMARY KEY,
    name VARCHAR(100)
)
"""
create_table_if_not_exists('test_table', create_sql)

# Backup table
backup_name = backup_table('master_info')
```

## Error Handling

The system provides comprehensive error handling:

```python
from production.database import DatabaseConnectionError, DatabaseConfigError

try:
    engine = get_engine()
except DatabaseConfigError as e:
    print(f"Configuration error: {e}")
except DatabaseConnectionError as e:
    print(f"Connection error: {e}")
```

## Best Practices

1. **Use Context Managers**: Always use context managers for safe connection handling
2. **Reuse Connections**: The system caches connections, so reuse them when possible
3. **Handle Errors**: Always handle database errors appropriately
4. **Use Utility Functions**: Use provided utility functions for common operations
5. **Test Connections**: Test connections before running critical operations

## Testing

Test the database connection system:

```python
from production.database import test_connection

if test_connection():
    print("Database connection successful!")
else:
    print("Database connection failed!")
```

## Troubleshooting

### Common Issues

1. **Configuration File Not Found**
   - Ensure `config/database.yml` exists
   - Check file permissions

2. **Connection Failed**
   - Verify database server is running
   - Check credentials in configuration file
   - Ensure network connectivity

3. **Import Errors**
   - Install required dependencies: `pip install sqlalchemy pymysql pyyaml`
   - Check Python path includes the production directory

### Debug Mode

Enable debug logging:

```python
import logging
logging.getLogger('production.database').setLevel(logging.DEBUG)
```

## Contributing

When adding new features to the database system:

1. Follow the existing code structure
2. Add comprehensive docstrings
3. Include error handling
4. Add tests for new functionality
5. Update this documentation

## License

This module is part of the factor investing project and follows the same license terms.