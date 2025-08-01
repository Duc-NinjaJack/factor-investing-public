"""
Aureus Sigma Capital - Universe Construction Module
===================================================
Purpose:
    Defines and constructs investable universes based on systematic,
    point-in-time correct rules. This module is the single source of
    truth for universe definitions.

Author: Duc Nguyen, Quantitative Finance Expert
Date Created: July 28, 2025
Version: 1.2 (Fully Refactored for Production)

Changelog (v1.1 -> v1.2):
-   CRITICAL FIX: Eliminated look-ahead bias by shifting all data windows to end at T-1.
-   EFFICIENCY: Refactored to use a single internal worker function (_construct_liquid_universe_df)
    to eliminate redundant database queries.
-   ROBUSTNESS: Replaced all print statements with a standard logging framework.
-   ROBUSTNESS: Added explicit handling for edge cases (e.g., empty query results).
"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging

# Set up a logger for this module
logger = logging.getLogger(__name__)


def _construct_liquid_universe_df(
    analysis_date: pd.Timestamp,
    engine,
    config: Dict
) -> pd.DataFrame:
    """
    Internal worker function to construct the liquid universe DataFrame.
    This is the core logic engine that performs the heavy lifting.
    """
    # --- CRITICAL FIX v1.2: Prevent Look-Ahead Bias ---
    # The decision for day T must only use data available up to T-1 close.
    end_date = analysis_date - timedelta(days=1)
    start_date = end_date - timedelta(days=config['lookback_days'] - 1)
    # --- END FIX ---

    logger.info(f"Constructing universe for {analysis_date.date()} using data from {start_date.date()} to {end_date.date()}.")

    # Step 1: Get all potentially active tickers in the period
    ticker_query = text("""
        SELECT DISTINCT ticker FROM vcsc_daily_data_complete
        WHERE trading_date BETWEEN :start_date AND :end_date AND total_value > 0
        ORDER BY ticker
    """)
    with engine.connect() as conn:
        all_tickers = pd.read_sql(ticker_query, conn, params={
            'start_date': start_date, 'end_date': end_date
        })['ticker'].tolist()

    if not all_tickers:
        logger.warning("No active tickers found in the lookback window.")
        return pd.DataFrame()

    logger.info(f"Found {len(all_tickers)} potentially active tickers. Calculating liquidity metrics in batches.")

    # Step 2: Process tickers in batches to calculate ADTV and other metrics
    batch_size = 50
    all_results = []
    for i in range(0, len(all_tickers), batch_size):
        batch_tickers = all_tickers[i:i + batch_size]
        batch_query = text("""
            SELECT
                v.ticker,
                COUNT(v.trading_date) AS trading_days,
                AVG(v.total_value / 1e9) AS adtv_bn_vnd,
                (SELECT market_cap / 1e9 FROM vcsc_daily_data_complete
                 WHERE ticker = v.ticker AND trading_date <= :end_date
                 ORDER BY trading_date DESC LIMIT 1) AS last_market_cap_bn
            FROM vcsc_daily_data_complete v
            WHERE v.ticker IN :tickers
              AND v.trading_date BETWEEN :start_date AND :end_date
              AND v.total_value > 0
            GROUP BY v.ticker
        """)
        with engine.connect() as conn:
            result = conn.execute(batch_query, {
                'tickers': tuple(batch_tickers),
                'start_date': start_date,
                'end_date': end_date
            })
            all_results.extend(result.fetchall())

    if not all_results:
        logger.warning("No liquidity data found for any tickers in the period.")
        return pd.DataFrame()

    # Step 3: Filter, rank, and finalize the universe DataFrame
    df = pd.DataFrame(all_results, columns=['ticker', 'trading_days', 'adtv_bn_vnd', 'last_market_cap_bn'])
    min_trading_days = int(config['lookback_days'] * config['min_trading_coverage'])

    logger.info(f"Initial metrics calculated for {len(df)} tickers. Applying filters...")
    logger.info(f"  - Trading days threshold: >= {min_trading_days}")
    logger.info(f"  - ADTV threshold: >= {config['adtv_threshold_bn']:.1f}B VND")

    filtered_df = df[
        (df['trading_days'] >= min_trading_days) &
        (df['adtv_bn_vnd'] >= config['adtv_threshold_bn'])
    ].copy()

    logger.info(f"{len(filtered_df)} stocks passed filters. Selecting top {config['top_n']} by ADTV.")

    # Sort by ADTV and take top N
    universe_df = filtered_df.sort_values('adtv_bn_vnd', ascending=False).head(config['top_n'])
    return universe_df


def get_liquid_universe(
    analysis_date: pd.Timestamp,
    engine,
    config: Optional[Dict] = None
) -> List[str]:
    """
    Constructs the liquid universe for a given analysis date and returns a list of tickers.
    v1.2: This is now a lightweight wrapper around the core worker function.
    """
    default_config = {
        'lookback_days': 63, 'adtv_threshold_bn': 10.0,
        'top_n': 200, 'min_trading_coverage': 0.6
    }
    final_config = {**default_config, **(config or {})}

    universe_df = _construct_liquid_universe_df(analysis_date, engine, final_config)

    if universe_df.empty:
        return []

    tickers = universe_df['ticker'].tolist()
    logger.info(f"Final universe constructed with {len(tickers)} tickers.")
    return tickers


def get_liquid_universe_dataframe(
    analysis_date: pd.Timestamp,
    engine,
    config: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Constructs the liquid universe and returns a full DataFrame with metrics.
    v1.2: This is now a lightweight wrapper that adds sector context.
    """
    default_config = {
        'lookback_days': 63, 'adtv_threshold_bn': 10.0,
        'top_n': 200, 'min_trading_coverage': 0.6
    }
    final_config = {**default_config, **(config or {})}

    # Get the core universe DataFrame from the worker function
    universe_df = _construct_liquid_universe_df(analysis_date, engine, final_config)

    if universe_df.empty:
        logger.warning(f"Universe construction for {analysis_date.date()} yielded an empty DataFrame.")
        return pd.DataFrame()

    # Add sector information
    tickers = universe_df['ticker'].tolist()
    sector_query = text("SELECT ticker, sector FROM master_info WHERE ticker IN :tickers")
    with engine.connect() as conn:
        sector_map = pd.read_sql(sector_query, conn, params={'tickers': tuple(tickers)}).set_index('ticker')['sector']

    universe_df['sector'] = universe_df['ticker'].map(sector_map)
    universe_df = universe_df.sort_values('adtv_bn_vnd', ascending=False).reset_index(drop=True)
    universe_df['universe_rank'] = universe_df.index + 1
    universe_df['universe_date'] = analysis_date.date()

    logger.info(f"Final universe DataFrame constructed for {len(universe_df)} stocks with sector data.")
    return universe_df


def validate_universe_construction(
    tickers: List[str],
    analysis_date: pd.Timestamp,
    min_size: int = 50
) -> Dict:
    """
    Validates the constructed universe meets quality standards.
    """
    validation = {'is_valid': True, 'checks': {}, 'warnings': [], 'errors': []}
    size_check = len(tickers) >= min_size
    validation['checks']['minimum_size'] = {'pass': size_check, 'actual': len(tickers), 'required': min_size}
    if not size_check:
        validation['is_valid'] = False
        validation['errors'].append(f"Universe too small: {len(tickers)} < {min_size}")
    unique_tickers = len(set(tickers))
    duplicate_check = unique_tickers == len(tickers)
    validation['checks']['no_duplicates'] = {'pass': duplicate_check, 'unique_count': unique_tickers, 'total_count': len(tickers)}
    if not duplicate_check:
        validation['warnings'].append("Duplicate tickers found in universe")
    return validation


def get_quarterly_universe_dates(year: int) -> Dict[str, pd.Timestamp]:
    """
    Get standard quarterly universe refresh dates for a given year.
    """
    return {
        'Q1': pd.Timestamp(f'{year}-03-31'),
        'Q2': pd.Timestamp(f'{year}-06-30'),
        'Q3': pd.Timestamp(f'{year}-09-30'),
        'Q4': pd.Timestamp(f'{year}-12-31')
    }


if __name__ == "__main__":
    # This block is for direct execution testing, not for import.
    print("="*60)
    print("🧪 Universe Constructor Module (v1.2 - Refactored) Test")
    print("="*60)
    print("This module is intended for import into backtesting scripts.")
    print("To test, you would need a live database connection.")
    print("\nAvailable functions:")
    print("  - get_liquid_universe(analysis_date, engine, config)")
    print("  - get_liquid_universe_dataframe(analysis_date, engine, config)")
    print("  - validate_universe_construction(tickers, analysis_date, min_size)")