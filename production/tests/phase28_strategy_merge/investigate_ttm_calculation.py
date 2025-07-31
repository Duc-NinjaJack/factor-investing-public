#!/usr/bin/env python3
"""
TTM Calculation Investigation Script
===================================

This script traces through each phase of the TTM calculation to identify
exactly where the data is being lost in the fundamental data query.
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from sqlalchemy import create_engine, text

# Add project root to path
current_path = Path.cwd()
while not (current_path / 'production').is_dir():
    if current_path.parent == current_path:
        raise FileNotFoundError("Could not find the 'production' directory.")
    current_path = current_path.parent

project_root = current_path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from production.database.connection import get_database_manager

def create_db_connection():
    """Establishes a SQLAlchemy database engine connection."""
    try:
        db_manager = get_database_manager()
        engine = db_manager.get_engine()
        
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print(f"✅ Database connection established successfully.")
        return engine
    except Exception as e:
        print(f"❌ FAILED to connect to the database: {e}")
        return None

def investigate_ttm_calculation():
    """Investigate TTM calculation step by step."""
    
    # Configuration (same as strategy)
    config = {
        'backtest_start_date': '2020-01-01',
        'backtest_end_date': '2024-12-31',
        'factors': {
            'fundamental_lag_days': 45,
            'netprofit_item_id': 1501,
            'revenue_item_id': 10701,
            'totalassets_item_id': 107,
        }
    }
    
    # Test with a specific date and ticker
    test_date = pd.Timestamp('2021-06-30')  # Mid-2021
    test_ticker = 'TCB'  # Known to have data
    
    print("="*80)
    print("🔍 TTM CALCULATION INVESTIGATION")
    print("="*80)
    print(f"Test Date: {test_date}")
    print(f"Test Ticker: {test_ticker}")
    print(f"Item IDs: NetProfit={config['factors']['netprofit_item_id']}, "
          f"Revenue={config['factors']['revenue_item_id']}, "
          f"TotalAssets={config['factors']['totalassets_item_id']}")
    print()
    
    # Calculate lag date
    lag_days = config['factors']['fundamental_lag_days']
    lag_date = test_date - pd.Timedelta(days=lag_days)
    lag_year = lag_date.year
    lag_quarter = ((lag_date.month - 1) // 3) + 1
    
    print(f"📅 Lag Calculation:")
    print(f"   Original Date: {test_date}")
    print(f"   Lag Days: {lag_days}")
    print(f"   Lag Date: {lag_date}")
    print(f"   Lag Year: {lag_year}")
    print(f"   Lag Quarter: {lag_quarter}")
    print()
    
    engine = create_db_connection()
    if not engine:
        return
    
    # PHASE 1: Check basic quarterly data availability
    print("="*80)
    print("PHASE 1: BASIC QUARTERLY DATA AVAILABILITY")
    print("="*80)
    
    phase1_query = text("""
        SELECT 
            ticker,
            year,
            quarter,
            item_id,
            value,
            statement_type
        FROM fundamental_values
        WHERE ticker = :ticker
        AND item_id IN (:netprofit_id, :revenue_id, :totalassets_id)
        AND year BETWEEN 2020 AND 2024
        ORDER BY year DESC, quarter DESC, item_id
    """)
    
    phase1_params = {
        'ticker': test_ticker,
        'netprofit_id': config['factors']['netprofit_item_id'],
        'revenue_id': config['factors']['revenue_item_id'],
        'totalassets_id': config['factors']['totalassets_item_id']
    }
    
    phase1_data = pd.read_sql(phase1_query, engine, params=phase1_params)
    print(f"📊 Phase 1 Results:")
    print(f"   Records found: {len(phase1_data)}")
    if not phase1_data.empty:
        print(f"   Sample data:")
        print(phase1_data.head(10).to_string())
    else:
        print("   ❌ NO DATA FOUND - This is the root cause!")
        return
    print()
    
    # PHASE 2: Check data after lag filtering
    print("="*80)
    print("PHASE 2: DATA AFTER LAG FILTERING")
    print("="*80)
    
    phase2_query = text("""
        SELECT 
            ticker,
            year,
            quarter,
            item_id,
            value
        FROM fundamental_values
        WHERE ticker = :ticker
        AND item_id IN (:netprofit_id, :revenue_id, :totalassets_id)
        AND (year < :lag_year OR (year = :lag_year AND quarter <= :lag_quarter))
        ORDER BY year DESC, quarter DESC, item_id
    """)
    
    phase2_params = {
        'ticker': test_ticker,
        'netprofit_id': config['factors']['netprofit_item_id'],
        'revenue_id': config['factors']['revenue_item_id'],
        'totalassets_id': config['factors']['totalassets_item_id'],
        'lag_year': lag_year,
        'lag_quarter': lag_quarter
    }
    
    phase2_data = pd.read_sql(phase2_query, engine, params=phase2_params)
    print(f"📊 Phase 2 Results:")
    print(f"   Records after lag filtering: {len(phase2_data)}")
    if not phase2_data.empty:
        print(f"   Sample data:")
        print(phase2_data.head(10).to_string())
    else:
        print("   ❌ NO DATA FOUND - Lag filtering is too restrictive!")
        return
    print()
    
    # PHASE 3: Check ranking and top 4 quarters
    print("="*80)
    print("PHASE 3: RANKING AND TOP 4 QUARTERS")
    print("="*80)
    
    phase3_query = text("""
        WITH quarterly_data AS (
            SELECT 
                ticker,
                year,
                quarter,
                item_id,
                value
            FROM fundamental_values
            WHERE ticker = :ticker
            AND item_id IN (:netprofit_id, :revenue_id, :totalassets_id)
            AND (year < :lag_year OR (year = :lag_year AND quarter <= :lag_quarter))
        ),
        ranked_data AS (
            SELECT 
                ticker,
                year,
                quarter,
                item_id,
                value,
                ROW_NUMBER() OVER (PARTITION BY ticker, item_id ORDER BY year DESC, quarter DESC) as rn
            FROM quarterly_data
        )
        SELECT 
            ticker,
            year,
            quarter,
            item_id,
            value,
            rn
        FROM ranked_data
        WHERE rn <= 4
        ORDER BY item_id, rn
    """)
    
    phase3_params = {
        'ticker': test_ticker,
        'netprofit_id': config['factors']['netprofit_item_id'],
        'revenue_id': config['factors']['revenue_item_id'],
        'totalassets_id': config['factors']['totalassets_item_id'],
        'lag_year': lag_year,
        'lag_quarter': lag_quarter
    }
    
    phase3_data = pd.read_sql(phase3_query, engine, params=phase3_params)
    print(f"📊 Phase 3 Results:")
    print(f"   Records in top 4 quarters: {len(phase3_data)}")
    if not phase3_data.empty:
        print(f"   Sample data:")
        print(phase3_data.to_string())
        
        # Show breakdown by item_id
        print(f"\n   Breakdown by item_id:")
        for item_id in phase3_data['item_id'].unique():
            item_data = phase3_data[phase3_data['item_id'] == item_id]
            print(f"     Item {item_id}: {len(item_data)} records")
    else:
        print("   ❌ NO DATA FOUND - Ranking issue!")
        return
    print()
    
    # PHASE 4: Check TTM aggregation
    print("="*80)
    print("PHASE 4: TTM AGGREGATION")
    print("="*80)
    
    phase4_query = text("""
        WITH quarterly_data AS (
            SELECT 
                ticker,
                year,
                quarter,
                item_id,
                value
            FROM fundamental_values
            WHERE ticker = :ticker
            AND item_id IN (:netprofit_id, :revenue_id, :totalassets_id)
            AND (year < :lag_year OR (year = :lag_year AND quarter <= :lag_quarter))
        ),
        ranked_data AS (
            SELECT 
                ticker,
                year,
                quarter,
                item_id,
                value,
                ROW_NUMBER() OVER (PARTITION BY ticker, item_id ORDER BY year DESC, quarter DESC) as rn
            FROM quarterly_data
        ),
        ttm_calculations AS (
            SELECT 
                ticker,
                year,
                quarter,
                SUM(CASE WHEN item_id = :netprofit_id THEN value ELSE 0 END) as netprofit_ttm,
                SUM(CASE WHEN item_id = :revenue_id THEN value ELSE 0 END) as revenue_ttm,
                SUM(CASE WHEN item_id = :totalassets_id THEN value ELSE 0 END) as totalassets_ttm
            FROM ranked_data
            WHERE rn <= 4
            GROUP BY ticker, year, quarter
        )
        SELECT 
            ticker,
            year,
            quarter,
            netprofit_ttm,
            revenue_ttm,
            totalassets_ttm
        FROM ttm_calculations
        ORDER BY year DESC, quarter DESC
    """)
    
    phase4_params = {
        'ticker': test_ticker,
        'netprofit_id': config['factors']['netprofit_item_id'],
        'revenue_id': config['factors']['revenue_item_id'],
        'totalassets_id': config['factors']['totalassets_item_id'],
        'lag_year': lag_year,
        'lag_quarter': lag_quarter
    }
    
    phase4_data = pd.read_sql(phase4_query, engine, params=phase4_params)
    print(f"📊 Phase 4 Results:")
    print(f"   TTM records: {len(phase4_data)}")
    if not phase4_data.empty:
        print(f"   Sample TTM data:")
        print(phase4_data.to_string())
    else:
        print("   ❌ NO TTM DATA - Aggregation issue!")
        return
    print()
    
    # PHASE 5: Check final filtering and ratios
    print("="*80)
    print("PHASE 5: FINAL FILTERING AND RATIOS")
    print("="*80)
    
    phase5_query = text("""
        WITH quarterly_data AS (
            SELECT 
                ticker,
                year,
                quarter,
                item_id,
                value
            FROM fundamental_values
            WHERE ticker = :ticker
            AND item_id IN (:netprofit_id, :revenue_id, :totalassets_id)
            AND (year < :lag_year OR (year = :lag_year AND quarter <= :lag_quarter))
        ),
        ranked_data AS (
            SELECT 
                ticker,
                year,
                quarter,
                item_id,
                value,
                ROW_NUMBER() OVER (PARTITION BY ticker, item_id ORDER BY year DESC, quarter DESC) as rn
            FROM quarterly_data
        ),
        ttm_calculations AS (
            SELECT 
                ticker,
                year,
                quarter,
                SUM(CASE WHEN item_id = :netprofit_id THEN value ELSE 0 END) as netprofit_ttm,
                SUM(CASE WHEN item_id = :revenue_id THEN value ELSE 0 END) as revenue_ttm,
                SUM(CASE WHEN item_id = :totalassets_id THEN value ELSE 0 END) as totalassets_ttm
            FROM ranked_data
            WHERE rn <= 4
            GROUP BY ticker, year, quarter
        )
        SELECT 
            ttm.ticker,
            mi.sector,
            ttm.year,
            ttm.quarter,
            ttm.netprofit_ttm,
            ttm.revenue_ttm,
            ttm.totalassets_ttm,
            CASE 
                WHEN ttm.totalassets_ttm > 0 THEN ttm.netprofit_ttm / ttm.totalassets_ttm 
                ELSE NULL 
            END as roaa,
            CASE 
                WHEN ttm.revenue_ttm > 0 THEN (ttm.netprofit_ttm / ttm.revenue_ttm)
                ELSE NULL 
            END as net_margin,
            CASE 
                WHEN ttm.totalassets_ttm > 0 THEN ttm.revenue_ttm / ttm.totalassets_ttm 
                ELSE NULL 
            END as asset_turnover
        FROM ttm_calculations ttm
        LEFT JOIN master_info mi ON ttm.ticker = mi.ticker
        WHERE ttm.netprofit_ttm > 0 AND ttm.revenue_ttm > 0 AND ttm.totalassets_ttm > 0
        ORDER BY ttm.year DESC, ttm.quarter DESC
    """)
    
    phase5_params = {
        'ticker': test_ticker,
        'netprofit_id': config['factors']['netprofit_item_id'],
        'revenue_id': config['factors']['revenue_item_id'],
        'totalassets_id': config['factors']['totalassets_item_id'],
        'lag_year': lag_year,
        'lag_quarter': lag_quarter
    }
    
    phase5_data = pd.read_sql(phase5_query, engine, params=phase5_params)
    print(f"📊 Phase 5 Results:")
    print(f"   Final records: {len(phase5_data)}")
    if not phase5_data.empty:
        print(f"   Final data with ratios:")
        print(phase5_data.to_string())
    else:
        print("   ❌ NO FINAL DATA - Filtering issue!")
        print("   Let's check what the TTM values look like before filtering:")
        
        # Check TTM values before filtering
        check_ttm_query = text("""
            WITH quarterly_data AS (
                SELECT 
                    ticker,
                    year,
                    quarter,
                    item_id,
                    value
                FROM fundamental_values
                WHERE ticker = :ticker
                AND item_id IN (:netprofit_id, :revenue_id, :totalassets_id)
                AND (year < :lag_year OR (year = :lag_year AND quarter <= :lag_quarter))
            ),
            ranked_data AS (
                SELECT 
                    ticker,
                    year,
                    quarter,
                    item_id,
                    value,
                    ROW_NUMBER() OVER (PARTITION BY ticker, item_id ORDER BY year DESC, quarter DESC) as rn
                FROM quarterly_data
            ),
            ttm_calculations AS (
                SELECT 
                    ticker,
                    year,
                    quarter,
                    SUM(CASE WHEN item_id = :netprofit_id THEN value ELSE 0 END) as netprofit_ttm,
                    SUM(CASE WHEN item_id = :revenue_id THEN value ELSE 0 END) as revenue_ttm,
                    SUM(CASE WHEN item_id = :totalassets_id THEN value ELSE 0 END) as totalassets_ttm
                FROM ranked_data
                WHERE rn <= 4
                GROUP BY ticker, year, quarter
            )
            SELECT 
                ticker,
                year,
                quarter,
                netprofit_ttm,
                revenue_ttm,
                totalassets_ttm,
                netprofit_ttm > 0 as netprofit_positive,
                revenue_ttm > 0 as revenue_positive,
                totalassets_ttm > 0 as totalassets_positive
            FROM ttm_calculations
            ORDER BY year DESC, quarter DESC
        """)
        
        check_ttm_data = pd.read_sql(check_ttm_query, engine, params=phase5_params)
        print(f"   TTM values before filtering:")
        print(check_ttm_data.to_string())
    print()
    
    # PHASE 6: Test with multiple tickers
    print("="*80)
    print("PHASE 6: TEST WITH MULTIPLE TICKERS")
    print("="*80)
    
    # Get a list of tickers that have data
    ticker_query = text("""
        SELECT DISTINCT ticker
        FROM fundamental_values
        WHERE item_id IN (:netprofit_id, :revenue_id, :totalassets_id)
        AND year BETWEEN 2020 AND 2024
        LIMIT 10
    """)
    
    ticker_params = {
        'netprofit_id': config['factors']['netprofit_item_id'],
        'revenue_id': config['factors']['revenue_item_id'],
        'totalassets_id': config['factors']['totalassets_item_id']
    }
    
    tickers_df = pd.read_sql(ticker_query, engine, params=ticker_params)
    print(f"📊 Available tickers: {len(tickers_df)}")
    print(f"   Sample tickers: {tickers_df['ticker'].tolist()[:5]}")
    
    # Test with first few tickers
    for i, ticker in enumerate(tickers_df['ticker'].head(3)):
        print(f"\n   Testing ticker {i+1}: {ticker}")
        
        # Quick test with simplified query
        quick_query = text("""
            SELECT COUNT(*) as record_count
            FROM fundamental_values
            WHERE ticker = :ticker
            AND item_id IN (:netprofit_id, :revenue_id, :totalassets_id)
            AND (year < :lag_year OR (year = :lag_year AND quarter <= :lag_quarter))
        """)
        
        quick_params = {
            'ticker': ticker,
            'netprofit_id': config['factors']['netprofit_item_id'],
            'revenue_id': config['factors']['revenue_item_id'],
            'totalassets_id': config['factors']['totalassets_item_id'],
            'lag_year': lag_year,
            'lag_quarter': lag_quarter
        }
        
        quick_result = pd.read_sql(quick_query, engine, params=quick_params)
        print(f"     Records after lag filtering: {quick_result.iloc[0]['record_count']}")
    
    print("\n" + "="*80)
    print("🔍 INVESTIGATION SUMMARY")
    print("="*80)
    print("The investigation shows exactly where the TTM calculation is failing.")
    print("This will help identify the specific issue in the strategy's fundamental query.")

if __name__ == "__main__":
    investigate_ttm_calculation() 