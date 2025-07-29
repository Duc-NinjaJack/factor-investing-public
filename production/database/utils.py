"""
Database Utilities

Common database operations and utilities for the factor investing project.
This module provides helper functions for common database tasks.

Author: Factor Investing Team
Date: 2025-07-30
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
from sqlalchemy import text
from sqlalchemy.engine import Engine
import pymysql
from pymysql.cursors import DictCursor

from .connection import get_engine, get_connection, DatabaseManager

def execute_query(query: str, 
                 params: Optional[Dict[str, Any]] = None,
                 engine: Optional[Engine] = None,
                 return_dataframe: bool = True) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Execute a SQL query and return results.
    
    Args:
        query: SQL query string
        params: Query parameters (for parameterized queries)
        engine: SQLAlchemy engine (if None, uses default)
        return_dataframe: Whether to return pandas DataFrame or list of dicts
        
    Returns:
        Query results as DataFrame or list of dictionaries
    """
    if engine is None:
        engine = get_engine()
    
    try:
        if return_dataframe:
            if params:
                df = pd.read_sql(text(query), engine, params=params)
            else:
                df = pd.read_sql(text(query), engine)
            return df
        else:
            with engine.connect() as conn:
                if params:
                    result = conn.execute(text(query), params)
                else:
                    result = conn.execute(text(query))
                return [dict(row) for row in result]
    except Exception as e:
        raise Exception(f"Query execution failed: {e}")

def execute_pymysql_query(query: str,
                         params: Optional[Dict[str, Any]] = None,
                         connection: Optional[pymysql.Connection] = None,
                         return_dataframe: bool = True) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Execute a SQL query using PyMySQL and return results.
    
    Args:
        query: SQL query string
        params: Query parameters (for parameterized queries)
        connection: PyMySQL connection (if None, uses default)
        return_dataframe: Whether to return pandas DataFrame or list of dicts
        
    Returns:
        Query results as DataFrame or list of dictionaries
    """
    if connection is None:
        connection = get_connection()
    
    try:
        with connection.cursor() as cursor:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            if query.strip().upper().startswith('SELECT'):
                results = cursor.fetchall()
                if return_dataframe:
                    return pd.DataFrame(results)
                else:
                    return results
            else:
                connection.commit()
                return cursor.rowcount
    except Exception as e:
        connection.rollback()
        raise Exception(f"PyMySQL query execution failed: {e}")

def get_table_info(table_name: str, engine: Optional[Engine] = None) -> pd.DataFrame:
    """
    Get table schema information.
    
    Args:
        table_name: Name of the table
        engine: SQLAlchemy engine (if None, uses default)
        
    Returns:
        DataFrame with column information
    """
    query = f"""
    SELECT 
        COLUMN_NAME,
        DATA_TYPE,
        IS_NULLABLE,
        COLUMN_DEFAULT,
        COLUMN_COMMENT
    FROM INFORMATION_SCHEMA.COLUMNS 
    WHERE TABLE_SCHEMA = DATABASE() 
    AND TABLE_NAME = '{table_name}'
    ORDER BY ORDINAL_POSITION
    """
    
    return execute_query(query, engine=engine)

def get_table_row_count(table_name: str, engine: Optional[Engine] = None) -> int:
    """
    Get the number of rows in a table.
    
    Args:
        table_name: Name of the table
        engine: SQLAlchemy engine (if None, uses default)
        
    Returns:
        Number of rows
    """
    query = f"SELECT COUNT(*) as row_count FROM {table_name}"
    result = execute_query(query, engine=engine)
    return result['row_count'].iloc[0]

def get_table_date_range(table_name: str, 
                        date_column: str,
                        engine: Optional[Engine] = None) -> Dict[str, Any]:
    """
    Get the date range for a table.
    
    Args:
        table_name: Name of the table
        date_column: Name of the date column
        engine: SQLAlchemy engine (if None, uses default)
        
    Returns:
        Dictionary with min_date and max_date
    """
    query = f"""
    SELECT 
        MIN({date_column}) as min_date,
        MAX({date_column}) as max_date,
        COUNT(DISTINCT {date_column}) as unique_dates
    FROM {table_name}
    """
    
    result = execute_query(query, engine=engine)
    return {
        'min_date': result['min_date'].iloc[0],
        'max_date': result['max_date'].iloc[0],
        'unique_dates': result['unique_dates'].iloc[0]
    }

def get_ticker_list(table_name: str = 'master_info',
                   active_only: bool = True,
                   engine: Optional[Engine] = None) -> List[str]:
    """
    Get list of tickers from master_info table.
    
    Args:
        table_name: Name of the table (default: master_info)
        active_only: Whether to return only active tickers (ignored if status column doesn't exist)
        engine: SQLAlchemy engine (if None, uses default)
        
    Returns:
        List of ticker symbols
    """
    # Check if status column exists
    try:
        if active_only:
            query = f"SELECT ticker FROM {table_name} WHERE ticker IS NOT NULL AND status = 'active'"
        else:
            query = f"SELECT ticker FROM {table_name} WHERE ticker IS NOT NULL"
    except:
        # If status column doesn't exist, just get all tickers
        query = f"SELECT ticker FROM {table_name} WHERE ticker IS NOT NULL"
    
    result = execute_query(query, engine=engine)
    return result['ticker'].tolist()

def get_sector_mapping(engine: Optional[Engine] = None) -> pd.DataFrame:
    """
    Get sector mapping for all tickers.
    
    Args:
        engine: SQLAlchemy engine (if None, uses default)
        
    Returns:
        DataFrame with ticker and sector columns
    """
    query = """
    SELECT ticker, sector
    FROM master_info
    WHERE ticker IS NOT NULL
    """
    
    result = execute_query(query, engine=engine)
    
    # Fix Banks -> Banking sector name
    result.loc[result['sector'] == 'Banks', 'sector'] = 'Banking'
    
    return result

def get_price_data(tickers: List[str],
                   start_date: str,
                   end_date: str,
                   table_name: str = 'vcsc_daily_data_complete',
                   engine: Optional[Engine] = None) -> pd.DataFrame:
    """
    Get price data for specified tickers and date range.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        table_name: Name of the price data table
        engine: SQLAlchemy engine (if None, uses default)
        
    Returns:
        DataFrame with price data
    """
    ticker_list = "', '".join(tickers)
    
    # Try to determine the correct column names
    try:
        # First check what columns exist
        columns_query = f"DESCRIBE {table_name}"
        columns_df = execute_query(columns_query, engine=engine)
        columns = columns_df['Field'].tolist()
        
        # Determine date and price columns
        date_col = 'date' if 'date' in columns else 'trading_date'
        price_col = 'close_price_adjusted' if 'close_price_adjusted' in columns else 'close_price'
        
        query = f"""
        SELECT {date_col} as date, ticker, {price_col} as close_price
        FROM {table_name}
        WHERE ticker IN ('{ticker_list}')
        AND {date_col} BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY {date_col}, ticker
        """
        
        return execute_query(query, engine=engine)
        
    except Exception as e:
        # Fallback to basic query
        query = f"""
        SELECT * FROM {table_name}
        WHERE ticker IN ('{ticker_list}')
        LIMIT 10
        """
        return execute_query(query, engine=engine)

def get_factor_scores(tickers: List[str],
                     rebalance_date: str,
                     table_name: str = 'factor_scores_qvm',
                     engine: Optional[Engine] = None) -> pd.DataFrame:
    """
    Get factor scores for specified tickers and rebalance date.
    
    Args:
        tickers: List of ticker symbols
        rebalance_date: Rebalance date (YYYY-MM-DD)
        table_name: Name of the factor scores table
        engine: SQLAlchemy engine (if None, uses default)
        
    Returns:
        DataFrame with factor scores
    """
    ticker_list = "', '".join(tickers)
    query = f"""
    SELECT ticker, Quality_Composite, Value_Composite, Momentum_Composite, QVM_Composite
    FROM {table_name}
    WHERE ticker IN ('{ticker_list}')
    AND rebalance_date = '{rebalance_date}'
    ORDER BY ticker
    """
    
    return execute_query(query, engine=engine)

def get_benchmark_data(start_date: str,
                      end_date: str,
                      table_name: str = 'etf_history',
                      ticker: str = 'VNINDEX',
                      engine: Optional[Engine] = None) -> pd.DataFrame:
    """
    Get benchmark data for specified date range.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        table_name: Name of the benchmark data table
        ticker: Benchmark ticker symbol
        engine: SQLAlchemy engine (if None, uses default)
        
    Returns:
        DataFrame with benchmark data
    """
    query = f"""
    SELECT date, close_price
    FROM {table_name}
    WHERE ticker = '{ticker}'
    AND date BETWEEN '{start_date}' AND '{end_date}'
    ORDER BY date
    """
    
    return execute_query(query, engine=engine)

def get_liquid_universe(analysis_date: str,
                       adtv_threshold: float = 10.0,
                       lookback_days: int = 63,
                       top_n: int = 200,
                       min_trading_coverage: float = 0.6,
                       engine: Optional[Engine] = None) -> pd.DataFrame:
    """
    Get liquid universe based on ADTV criteria.
    
    Args:
        analysis_date: Analysis date (YYYY-MM-DD)
        adtv_threshold: ADTV threshold in billions VND
        lookback_days: Number of days to look back for ADTV calculation
        top_n: Maximum number of stocks to return
        min_trading_coverage: Minimum trading coverage requirement
        engine: SQLAlchemy engine (if None, uses default)
        
    Returns:
        DataFrame with liquid universe information
    """
    # Use the correct column names based on actual schema
    query = f"""
    WITH adtv_calc AS (
        SELECT 
            ticker,
            AVG(total_volume * close_price_adjusted) / 1e9 as adtv_bn_vnd,
            COUNT(*) / {lookback_days} as trading_coverage
        FROM vcsc_daily_data_complete
        WHERE trading_date BETWEEN DATE_SUB('{analysis_date}', INTERVAL {lookback_days} DAY) 
                      AND '{analysis_date}'
        GROUP BY ticker
        HAVING trading_coverage >= {min_trading_coverage}
    )
    SELECT 
        a.ticker,
        a.adtv_bn_vnd,
        a.trading_coverage,
        m.sector
    FROM adtv_calc a
    JOIN master_info m ON a.ticker = m.ticker
    WHERE a.adtv_bn_vnd >= {adtv_threshold}
    ORDER BY a.adtv_bn_vnd DESC
    LIMIT {top_n}
    """
    
    return execute_query(query, engine=engine)

def batch_query_executor(queries: List[str],
                        params_list: Optional[List[Dict[str, Any]]] = None,
                        engine: Optional[Engine] = None,
                        batch_size: int = 100) -> List[pd.DataFrame]:
    """
    Execute multiple queries in batches.
    
    Args:
        queries: List of SQL queries
        params_list: List of parameter dictionaries (optional)
        engine: SQLAlchemy engine (if None, uses default)
        batch_size: Number of queries to execute in each batch
        
    Returns:
        List of DataFrames with results
    """
    if engine is None:
        engine = get_engine()
    
    results = []
    
    for i in range(0, len(queries), batch_size):
        batch_queries = queries[i:i + batch_size]
        batch_params = params_list[i:i + batch_size] if params_list else [None] * len(batch_queries)
        
        batch_results = []
        for query, params in zip(batch_queries, batch_params):
            try:
                result = execute_query(query, params=params, engine=engine)
                batch_results.append(result)
            except Exception as e:
                print(f"Query failed: {e}")
                batch_results.append(pd.DataFrame())
        
        results.extend(batch_results)
    
    return results

def create_table_if_not_exists(table_name: str,
                              create_sql: str,
                              engine: Optional[Engine] = None) -> bool:
    """
    Create a table if it doesn't exist.
    
    Args:
        table_name: Name of the table
        create_sql: CREATE TABLE SQL statement
        engine: SQLAlchemy engine (if None, uses default)
        
    Returns:
        True if table was created or already exists, False otherwise
    """
    if engine is None:
        engine = get_engine()
    
    try:
        # Check if table exists
        check_query = f"""
        SELECT COUNT(*) as table_exists
        FROM information_schema.tables 
        WHERE table_schema = DATABASE() 
        AND table_name = '{table_name}'
        """
        
        result = execute_query(check_query, engine=engine)
        table_exists = result['table_exists'].iloc[0] > 0
        
        if not table_exists:
            execute_query(create_sql, engine=engine)
            return True
        else:
            return True
            
    except Exception as e:
        print(f"Failed to create table {table_name}: {e}")
        return False

def insert_dataframe_to_table(df: pd.DataFrame,
                             table_name: str,
                             if_exists: str = 'append',
                             engine: Optional[Engine] = None,
                             chunk_size: int = 1000) -> bool:
    """
    Insert DataFrame to database table.
    
    Args:
        df: DataFrame to insert
        table_name: Name of the target table
        if_exists: How to behave if table exists ('fail', 'replace', 'append')
        engine: SQLAlchemy engine (if None, uses default)
        chunk_size: Number of rows to insert in each chunk
        
    Returns:
        True if successful, False otherwise
    """
    if engine is None:
        engine = get_engine()
    
    try:
        df.to_sql(table_name, engine, if_exists=if_exists, index=False, chunksize=chunk_size)
        return True
    except Exception as e:
        print(f"Failed to insert DataFrame to {table_name}: {e}")
        return False

def backup_table(table_name: str,
                 backup_suffix: str = None,
                 engine: Optional[Engine] = None) -> str:
    """
    Create a backup of a table.
    
    Args:
        table_name: Name of the table to backup
        backup_suffix: Suffix for backup table name (if None, uses timestamp)
        engine: SQLAlchemy engine (if None, uses default)
        
    Returns:
        Name of the backup table
    """
    if engine is None:
        engine = get_engine()
    
    if backup_suffix is None:
        from datetime import datetime
        backup_suffix = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    backup_table_name = f"{table_name}_backup_{backup_suffix}"
    
    try:
        backup_query = f"CREATE TABLE {backup_table_name} AS SELECT * FROM {table_name}"
        execute_query(backup_query, engine=engine)
        return backup_table_name
    except Exception as e:
        print(f"Failed to backup table {table_name}: {e}")
        return None

def get_database_stats(engine: Optional[Engine] = None) -> Dict[str, Any]:
    """
    Get database statistics.
    
    Args:
        engine: SQLAlchemy engine (if None, uses default)
        
    Returns:
        Dictionary with database statistics
    """
    if engine is None:
        engine = get_engine()
    
    try:
        # Get table sizes
        size_query = """
        SELECT 
            table_name,
            ROUND(((data_length + index_length) / 1024 / 1024), 2) AS 'size_mb',
            table_rows
        FROM information_schema.tables 
        WHERE table_schema = DATABASE()
        ORDER BY (data_length + index_length) DESC
        """
        
        table_sizes = execute_query(size_query, engine=engine)
        
        # Get total database size
        total_size_query = """
        SELECT 
            ROUND(SUM(data_length + index_length) / 1024 / 1024, 2) AS total_size_mb
        FROM information_schema.tables 
        WHERE table_schema = DATABASE()
        """
        
        total_size = execute_query(total_size_query, engine=engine)
        
        return {
            'table_sizes': table_sizes,
            'total_size_mb': total_size['total_size_mb'].iloc[0],
            'table_count': len(table_sizes)
        }
        
    except Exception as e:
        print(f"Failed to get database stats: {e}")
        return {}