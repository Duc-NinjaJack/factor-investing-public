#!/usr/bin/env python3
"""
Unrestricted Universe Data Extraction for Liquidity Bucket Analysis
==================================================================

This script extracts the complete unrestricted universe data needed for:
1. Factor scores (Quality, Value, Momentum, QVM Composite)
2. Daily returns
3. Volume data for ADTV calculation
4. Price data for ADTV calculation

The data will be used to analyze performance by liquidity buckets:
- Below 1B VND
- 1-3B VND
- 3-5B VND
- 5-10B VND
- 10B+ VND
"""

import sys
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
import pickle
import pymysql
from sqlalchemy import create_engine, text

warnings.filterwarnings('ignore')

# Add production engine to path
project_root = Path.cwd()
while not (project_root / 'production').exists() and not (project_root / 'config').exists():
    if project_root.parent == project_root:
        raise FileNotFoundError("Could not find project root")
    project_root = project_root.parent

production_path = project_root / 'production'
if str(production_path) not in sys.path:
    sys.path.insert(0, str(production_path))

# Import engine configuration
from engine.qvm_engine_v2_enhanced import QVMEngineV2Enhanced

print("🔍 UNRESTRICTED UNIVERSE DATA EXTRACTION")
print("=" * 50)
print(f"Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Project Root: {project_root}")


def get_database_connection():
    """Get database connection using the engine's configuration."""
    engine_instance = QVMEngineV2Enhanced()
    return engine_instance.engine


def load_factor_data():
    """Load factor scores from qvm_v2.0_enhanced strategy."""
    print("\n📊 Loading factor scores...")

    factor_query = """
    SELECT
        date as calculation_date,
        ticker,
        Quality_Composite as quality_score,
        Value_Composite as value_score,
        Momentum_Composite as momentum_score,
        QVM_Composite as qvm_composite_score
    FROM factor_scores_qvm
    WHERE strategy_version = 'qvm_v2.0_enhanced'
    ORDER BY date, ticker
    """

    engine = get_database_connection()
    factor_data = pd.read_sql(factor_query, engine)

    print(f"✅ Loaded {len(factor_data):,} factor score records")
    print(f"    Date range: {factor_data['calculation_date'].min()} to "
          f"{factor_data['calculation_date'].max()}")
    print(f"    Unique tickers: {factor_data['ticker'].nunique()}")

    return factor_data


def load_volume_and_price_data():
    """Load volume and price data for ADTV calculation."""
    print("\n📈 Loading volume and price data...")

    # Get unique tickers from factor data first
    factor_data = load_factor_data()
    unique_tickers = factor_data['ticker'].unique()

    volume_query = """
    SELECT
        trading_date as date,
        ticker,
        close_price_adjusted,
        total_volume
    FROM vcsc_daily_data_complete
    WHERE trading_date >= '2016-01-01'
        AND trading_date <= '2025-07-25'
        AND close_price_adjusted IS NOT NULL
        AND close_price_adjusted > 0
        AND total_volume IS NOT NULL
        AND total_volume > 0
    ORDER BY trading_date, ticker
    """

    engine = get_database_connection()
    volume_data = pd.read_sql(volume_query, engine)
    volume_data['date'] = pd.to_datetime(volume_data['date'])

    # Filter to only tickers in our factor universe
    volume_data = volume_data[volume_data['ticker'].isin(unique_tickers)]

    print(f"✅ Loaded {len(volume_data):,} volume/price records")
    print(f"    Date range: {volume_data['date'].min().date()} to "
          f"{volume_data['date'].max().date()}")
    print(f"    Unique tickers: {volume_data['ticker'].nunique()}")

    return volume_data


def calculate_adtv(volume_data, lookback_days=63):
    """Calculate Average Daily Turnover (ADTV)."""
    print(f"\n🔄 Calculating {lookback_days}-day ADTV...")

    # Calculate daily turnover (price * volume)
    volume_data['daily_turnover'] = (volume_data['close_price_adjusted'] *
                                    volume_data['total_volume'])

    # Pivot to create ticker columns
    turnover_pivot = volume_data.pivot(index='date', columns='ticker',
                                     values='daily_turnover')

    # Calculate rolling average ADTV
    adtv = turnover_pivot.rolling(window=lookback_days, min_periods=30).mean()

    print(f"✅ ADTV calculated")
    print(f"    ADTV matrix shape: {adtv.shape}")
    print(f"    Date range: {adtv.index.min().date()} to {adtv.index.max().date()}")

    return adtv


def create_liquidity_buckets(adtv, date):
    """Create liquidity buckets for a specific date."""
    print(f"\n📊 Creating liquidity buckets for {date.date()}...")

    # Get ADTV for the specific date
    if date in adtv.index:
        date_adtv = adtv.loc[date].dropna()
    else:
        # Use the most recent available date
        available_dates = adtv.index[adtv.index <= date]
        if len(available_dates) == 0:
            raise ValueError(f"No ADTV data available for {date}")
        date_adtv = adtv.loc[available_dates[-1]].dropna()

    # Define liquidity buckets (in VND)
    buckets = {
        'below_1b': (0, 1_000_000_000),
        '1b_to_3b': (1_000_000_000, 3_000_000_000),
        '3b_to_5b': (3_000_000_000, 5_000_000_000),
        '5b_to_10b': (5_000_000_000, 10_000_000_000),
        'above_10b': (10_000_000_000, float('inf'))
    }

    bucket_stocks = {}
    for bucket_name, (min_adtv, max_adtv) in buckets.items():
        if max_adtv == float('inf'):
            mask = date_adtv >= min_adtv
        else:
            mask = (date_adtv >= min_adtv) & (date_adtv < max_adtv)

        bucket_stocks[bucket_name] = date_adtv[mask].index.tolist()
        print(f"    {bucket_name}: {len(bucket_stocks[bucket_name])} stocks")

    return bucket_stocks


def save_data_for_analysis():
    """Save all data for the liquidity bucket analysis."""
    print("\n💾 Saving data for analysis...")

    # Load all data
    factor_data = load_factor_data()
    volume_data = load_volume_and_price_data()
    adtv = calculate_adtv(volume_data)

    # Create data object
    unrestricted_data = {
        'factor_data': factor_data,
        'volume_data': volume_data,
        'adtv': adtv,
        'metadata': {
            'creation_date': datetime.now(),
            'factor_records': len(factor_data),
            'volume_records': len(volume_data),
            'adtv_shape': adtv.shape,
            'date_range': {
                'start': factor_data['calculation_date'].min(),
                'end': factor_data['calculation_date'].max()
            },
            'universe_size': factor_data['ticker'].nunique()
        }
    }

    # Save to file
    save_path = Path(__file__).parent / "data" / "unrestricted_universe_data.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(unrestricted_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"✅ Data saved to: {save_path}")
    print(f"    File size: {save_path.stat().st_size / 1024**2:.1f} MB")

    return unrestricted_data


if __name__ == "__main__":
    try:
        # Save all data
        data = save_data_for_analysis()

        # Test liquidity bucket creation
        test_date = pd.to_datetime("2023-12-31")
        buckets = create_liquidity_buckets(data['adtv'], test_date)

        print(f"\n🎯 DATA EXTRACTION COMPLETE")
        print(f"    Ready for liquidity bucket analysis")
        print(f"    Test date {test_date.date()}: "
              f"{sum(len(stocks) for stocks in buckets.values())} total stocks")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()