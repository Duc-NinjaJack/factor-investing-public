# production/utils/db.py

"""
Aureus Sigma Capital - Database Utility Module
===============================================
Purpose:
    Handles all database interactions, including establishing connections
    and loading standardized datasets for backtesting.
"""
import pandas as pd
import yaml
from sqlalchemy import create_engine, text
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def create_db_connection(project_root_path: Path) -> create_engine:
    """Establishes a robust, production-ready database connection."""
    try:
        config_path = project_root_path / 'config' / 'database.yml'
        with open(config_path, 'r') as f:
            db_config = yaml.safe_load(f)['production']
        connection_string = (
            f"mysql+pymysql://{db_config['username']}:{db_config['password']}"
            f"@{db_config['host']}/{db_config['schema_name']}"
        )
        engine = create_engine(connection_string, pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1")) # Test connection
        logger.info(f"Database connection established successfully to schema '{db_config['schema_name']}'.")
        return engine
    except Exception as e:
        logger.error(f"FAILED to connect to the database: {e}")
        raise ConnectionError(f"Database connection failed: {e}")

def load_all_data_for_backtest(config: dict, db_engine: create_engine) -> tuple:
    """
    Loads and prepares all necessary data from the database for a backtest.
    """
    start_date_str = config['backtest_start_date']
    end_date_str = config['backtest_end_date']
    db_version = config['signal']['db_strategy_version']

    buffer_start_date = pd.Timestamp(start_date_str) - pd.DateOffset(months=3)
    end_date = pd.Timestamp(end_date_str)

    logger.info(f"Loading all data for period: {buffer_start_date.date()} to {end_date.date()}...")

    # 1. Load Factor Scores
    factor_query = text("""
        SELECT date, ticker, Quality_Composite, Value_Composite, Momentum_Composite, Defensive_Composite, QVM_Composite
        FROM factor_scores_qvm
        WHERE date BETWEEN :start_date AND :end_date
          AND strategy_version = :strategy_version
    """)
    db_params = {'start_date': buffer_start_date, 'end_date': end_date, 'strategy_version': db_version}
    factor_data = pd.read_sql(factor_query, db_engine, params=db_params, parse_dates=['date'])
    if factor_data.empty:
        raise ValueError(f"CRITICAL ERROR: No factor scores found for strategy_version='{db_version}'.")
    logger.info(f"Loaded {len(factor_data):,} factor score rows for version '{db_version}'.")

    # 2. Load Daily Price History
    price_query = text("SELECT date, ticker, close FROM equity_history WHERE date BETWEEN :start_date AND :end_date")
    price_data = pd.read_sql(price_query, db_engine, params=db_params, parse_dates=['date'])
    logger.info(f"Loaded {len(price_data):,} daily price records.")

    # 3. Load Benchmark (VN-Index) History
    benchmark_query = text("SELECT date, close FROM etf_history WHERE ticker = 'VNINDEX' AND date BETWEEN :start_date AND :end_date")
    benchmark_data = pd.read_sql(benchmark_query, db_engine, params=db_params, parse_dates=['date'])
    logger.info(f"Loaded {len(benchmark_data):,} VN-Index records.")

    # --- Data Preparation ---
    logger.info("Preparing data for backtesting engine...")
    price_data['return'] = price_data.groupby('ticker')['close'].pct_change()
    daily_returns_matrix = price_data.pivot(index='date', columns='ticker', values='return')
    logger.info(f"Created daily returns matrix with shape: {daily_returns_matrix.shape}")

    benchmark_returns = benchmark_data.set_index('date')['close'].pct_change().rename('VN-Index')
    logger.info(f"Prepared benchmark returns series with {len(benchmark_data)} data points.")

    return factor_data, daily_returns_matrix, benchmark_returns
