#!/usr/bin/env python3
"""
Debug script to identify trading logic issues causing underperformance
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

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
from sqlalchemy import text

def test_trading_logic():
    """Test the trading logic step by step."""
    
    print("🔍 DEBUGGING TRADING LOGIC ISSUES")
    print("=" * 60)
    
    # Get database connection
    db_manager = get_database_manager()
    engine = db_manager.get_engine()
    
    # Test configuration
    config = {
        'backtest_start_date': '2016-01-01',
        'backtest_end_date': '2025-07-28',
        'universe': {
            'lookback_days': 63,
            'top_n_stocks': 200,
            'target_portfolio_size': 20,
            'max_position_size': 0.05,
        },
        'factors': {
            'value_weight': 0.33,
            'quality_weight': 0.33,
            'momentum_weight': 0.34,
            'value_factors': {
                'pe_weight': 0.5,
                'fcf_yield_weight': 0.5
            },
            'quality_factors': {
                'roaa_weight': 0.5,
                'fscore_weight': 0.5
            },
            'momentum_factors': {
                'momentum_weight': 0.5,
                'low_vol_weight': 0.5
            },
            'momentum_horizons': [21, 63, 126, 252],
            'fundamental_lag_days': 45,
        }
    }
    
    print(f"📊 Testing trading logic for period: {config['backtest_start_date']} to {config['backtest_end_date']}")
    
    # Test 1: Check universe construction
    print("\n🔍 Test 1: Universe Construction...")
    test_date = '2016-02-01'
    
    # Get universe for test date
    universe_query = text("""
        WITH daily_adtv AS (
            SELECT 
                trading_date,
                ticker,
                total_volume * close_price as adtv_vnd
            FROM vcsc_daily_data
            WHERE trading_date BETWEEN :start_date AND :test_date
        ),
        rolling_adtv AS (
            SELECT 
                trading_date,
                ticker,
                AVG(adtv_vnd) OVER (
                    PARTITION BY ticker 
                    ORDER BY trading_date 
                    ROWS BETWEEN 62 PRECEDING AND CURRENT ROW
                ) as avg_adtv_63d
            FROM daily_adtv
            WHERE adtv_vnd > 0
        ),
        ranked_universe AS (
            SELECT 
                trading_date,
                ticker,
                ROW_NUMBER() OVER (
                    PARTITION BY trading_date 
                    ORDER BY avg_adtv_63d DESC
                ) as rank_position
            FROM rolling_adtv
            WHERE avg_adtv_63d > 0
        )
        SELECT trading_date, ticker, rank_position
        FROM ranked_universe
        WHERE rank_position <= :top_n_stocks
        AND trading_date = :test_date
        ORDER BY rank_position
    """)
    
    buffer_start_date = pd.Timestamp(config['backtest_start_date']) - pd.Timedelta(days=config['universe']['lookback_days'] + 30)
    
    with engine.connect() as conn:
        universe_result = pd.read_sql(universe_query, conn, params={
            'start_date': buffer_start_date,
            'test_date': test_date,
            'top_n_stocks': config['universe']['top_n_stocks']
        })
    
    print(f"   ✅ Universe for {test_date}:")
    print(f"      - Found {len(universe_result)} stocks")
    if len(universe_result) > 0:
        print(f"      - Top 5 stocks: {universe_result['ticker'].head().tolist()}")
        test_universe = universe_result['ticker'].head(10).tolist()
    else:
        print(f"      - ⚠️  No universe found!")
        return
    
    # Test 2: Check fundamental data availability
    print(f"\n🔍 Test 2: Fundamental Data Availability...")
    lag_date = pd.Timestamp(test_date) - pd.Timedelta(days=config['factors']['fundamental_lag_days'])
    
    fundamental_query = text("""
        WITH fundamental_metrics AS (
            SELECT 
                fv.ticker,
                fv.year,
                fv.quarter,
                fv.item_id,
                fv.statement_type,
                SUM(fv.value / 1e9) as value_bn
            FROM fundamental_values fv
            WHERE fv.year BETWEEN YEAR(:start_date) AND YEAR(:end_date)
            AND fv.item_id IN (1, 2)
            GROUP BY fv.ticker, fv.year, fv.quarter, fv.item_id, fv.statement_type
        ),
        netprofit_ttm AS (
            SELECT 
                ticker,
                year,
                quarter,
                SUM(CASE WHEN item_id = 1 AND statement_type = 'PL' THEN value_bn ELSE 0 END) as netprofit_ttm
            FROM fundamental_metrics
            GROUP BY ticker, year, quarter
        ),
        totalassets_ttm AS (
            SELECT 
                ticker,
                year,
                quarter,
                SUM(CASE WHEN item_id = 2 AND statement_type = 'BS' THEN value_bn ELSE 0 END) as totalassets_ttm
            FROM fundamental_metrics
            GROUP BY ticker, year, quarter
        ),
        revenue_ttm AS (
            SELECT 
                ticker,
                year,
                quarter,
                SUM(CASE WHEN item_id = 2 AND statement_type = 'PL' THEN value_bn ELSE 0 END) as revenue_ttm
            FROM fundamental_metrics
            GROUP BY ticker, year, quarter
        )
        SELECT 
            np.ticker,
            np.year,
            np.quarter,
            np.netprofit_ttm,
            ta.totalassets_ttm,
            rv.revenue_ttm,
            CASE 
                WHEN ta.totalassets_ttm > 0 THEN np.netprofit_ttm / ta.totalassets_ttm 
                ELSE NULL 
            END as roaa,
            CASE 
                WHEN rv.revenue_ttm > 0 THEN np.netprofit_ttm / rv.revenue_ttm
                ELSE NULL 
            END as net_margin
        FROM netprofit_ttm np
        LEFT JOIN totalassets_ttm ta ON np.ticker = ta.ticker AND np.year = ta.year AND np.quarter = ta.quarter
        LEFT JOIN revenue_ttm rv ON np.ticker = rv.ticker AND np.year = rv.year AND np.quarter = rv.quarter
        WHERE np.ticker IN :test_universe
        AND np.netprofit_ttm > 0 
        AND ta.totalassets_ttm > 0
        AND rv.revenue_ttm > 0
        ORDER BY np.year DESC, np.quarter DESC
    """)
    
    with engine.connect() as conn:
        fundamental_result = pd.read_sql(fundamental_query, conn, params={
            'start_date': buffer_start_date,
            'end_date': config['backtest_end_date'],
            'test_universe': tuple(test_universe)
        })
    
    print(f"   ✅ Fundamental data:")
    print(f"      - Found {len(fundamental_result)} records")
    if len(fundamental_result) > 0:
        print(f"      - Sample ROAA: {fundamental_result['roaa'].head().tolist()}")
        print(f"      - Sample Net Margin: {fundamental_result['net_margin'].head().tolist()}")
    else:
        print(f"      - ⚠️  No fundamental data found!")
    
    # Test 3: Check momentum data availability
    print(f"\n🔍 Test 3: Momentum Data Availability...")
    momentum_query = text("""
        SELECT 
            trading_date,
            ticker,
            close_price
        FROM vcsc_daily_data
        WHERE ticker IN :test_universe
        AND trading_date BETWEEN :start_date AND :test_date
        ORDER BY ticker, trading_date
    """)
    
    with engine.connect() as conn:
        momentum_result = pd.read_sql(momentum_query, conn, params={
            'start_date': buffer_start_date,
            'test_date': test_date,
            'test_universe': tuple(test_universe)
        })
    
    print(f"   ✅ Momentum data:")
    print(f"      - Found {len(momentum_result)} price records")
    if len(momentum_result) > 0:
        # Calculate simple momentum for one stock
        sample_stock = test_universe[0]
        stock_data = momentum_result[momentum_result['ticker'] == sample_stock].sort_values('trading_date')
        if len(stock_data) >= 21:
            momentum_21d = (stock_data['close_price'].iloc[-1] / stock_data['close_price'].iloc[-21]) - 1
            print(f"      - {sample_stock} 21-day momentum: {momentum_21d:.2%}")
    
    # Test 4: Check P/E data availability
    print(f"\n🔍 Test 4: P/E Data Availability...")
    pe_query = text("""
        SELECT 
            ticker,
            PE,
            PB,
            Date
        FROM financial_metrics
        WHERE ticker IN :test_universe
        AND Date <= :lag_date
        ORDER BY Date DESC
        LIMIT 10
    """)
    
    with engine.connect() as conn:
        pe_result = pd.read_sql(pe_query, conn, params={
            'test_universe': tuple(test_universe),
            'lag_date': lag_date
        })
    
    print(f"   ✅ P/E data:")
    print(f"      - Found {len(pe_result)} records")
    if len(pe_result) > 0:
        print(f"      - Sample P/E: {pe_result['PE'].head().tolist()}")
        print(f"      - Sample P/B: {pe_result['PB'].head().tolist()}")
    else:
        print(f"      - ⚠️  No P/E data found!")
    
    # Test 5: Simulate factor calculation
    print(f"\n🔍 Test 5: Factor Calculation Simulation...")
    
    # Create mock factor data
    mock_factors = pd.DataFrame({
        'ticker': test_universe[:5],
        'quality_adjusted_pe': [10.5, 15.2, 8.7, 22.1, 12.3],
        'roaa': [0.08, 0.12, 0.06, 0.15, 0.09],
        'momentum_score': [0.05, -0.02, 0.08, -0.01, 0.03],
        'fcf_yield': [0.04, 0.06, 0.03, 0.08, 0.05],
        'fscore': [7, 8, 6, 9, 7],
        'low_vol_score': [0.02, 0.01, 0.03, 0.01, 0.02]
    })
    
    print(f"   ✅ Mock factor data created:")
    print(f"      - {len(mock_factors)} stocks with factors")
    print(f"      - P/E range: {mock_factors['quality_adjusted_pe'].min():.1f} - {mock_factors['quality_adjusted_pe'].max():.1f}")
    print(f"      - ROAA range: {mock_factors['roaa'].min():.1%} - {mock_factors['roaa'].max():.1%}")
    
    # Simulate composite score calculation
    mock_factors['composite_score'] = 0.0
    
    # Value score
    pe_normalized = (mock_factors['quality_adjusted_pe'] - mock_factors['quality_adjusted_pe'].mean()) / mock_factors['quality_adjusted_pe'].std()
    fcf_normalized = (mock_factors['fcf_yield'] - mock_factors['fcf_yield'].mean()) / mock_factors['fcf_yield'].std()
    value_score = (-pe_normalized * 0.5) + (fcf_normalized * 0.5)
    
    # Quality score
    roaa_normalized = (mock_factors['roaa'] - mock_factors['roaa'].mean()) / mock_factors['roaa'].std()
    fscore_normalized = (mock_factors['fscore'] - mock_factors['fscore'].mean()) / mock_factors['fscore'].std()
    quality_score = (roaa_normalized * 0.5) + (fscore_normalized * 0.5)
    
    # Momentum score
    momentum_normalized = (mock_factors['momentum_score'] - mock_factors['momentum_score'].mean()) / mock_factors['momentum_score'].std()
    low_vol_normalized = (mock_factors['low_vol_score'] - mock_factors['low_vol_score'].mean()) / mock_factors['low_vol_score'].std()
    momentum_score = (momentum_normalized * 0.5) + (low_vol_normalized * 0.5)
    
    # Composite score
    mock_factors['composite_score'] = (
        value_score * config['factors']['value_weight'] +
        quality_score * config['factors']['quality_weight'] +
        momentum_score * config['factors']['momentum_weight']
    )
    
    print(f"   ✅ Composite scores calculated:")
    print(f"      - Score range: {mock_factors['composite_score'].min():.3f} - {mock_factors['composite_score'].max():.3f}")
    print(f"      - Top stock: {mock_factors.loc[mock_factors['composite_score'].idxmax(), 'ticker']}")
    
    # Test 6: Entry criteria simulation
    print(f"\n🔍 Test 6: Entry Criteria Simulation...")
    
    # Apply entry criteria
    qualified = mock_factors.copy()
    qualified = qualified.dropna(subset=['composite_score'])
    
    # P/E filtering
    pe_median = qualified['quality_adjusted_pe'].median()
    pe_std = qualified['quality_adjusted_pe'].std()
    qualified = qualified[
        (qualified['quality_adjusted_pe'] > pe_median - 3 * pe_std) &
        (qualified['quality_adjusted_pe'] < pe_median + 3 * pe_std)
    ]
    
    print(f"   ✅ Entry criteria applied:")
    print(f"      - Before filtering: {len(mock_factors)} stocks")
    print(f"      - After filtering: {len(qualified)} stocks")
    print(f"      - P/E range: {qualified['quality_adjusted_pe'].min():.1f} - {qualified['quality_adjusted_pe'].max():.1f}")
    
    # Test 7: Portfolio construction simulation
    print(f"\n🔍 Test 7: Portfolio Construction Simulation...")
    
    if len(qualified) > 0:
        # Select top stocks
        target_size = config['universe']['target_portfolio_size']
        top_stocks = qualified.nlargest(min(target_size, len(qualified)), 'composite_score')
        
        # Calculate weights
        weights = pd.Series(1.0 / len(top_stocks), index=top_stocks['ticker'])
        regime_allocation = 1.0  # Assume full allocation
        weights = weights * regime_allocation
        
        print(f"   ✅ Portfolio constructed:")
        print(f"      - Portfolio size: {len(top_stocks)} stocks")
        print(f"      - Average weight: {weights.mean():.2%}")
        print(f"      - Total allocation: {weights.sum():.2%}")
        print(f"      - Top holdings: {weights.head().to_dict()}")
    else:
        print(f"   ⚠️  No qualified stocks for portfolio!")
    
    print("\n" + "=" * 60)
    print("🔍 DIAGNOSIS SUMMARY")
    print("=" * 60)
    
    issues_found = []
    
    if len(universe_result) == 0:
        issues_found.append("❌ No universe stocks found")
    elif len(universe_result) < 50:
        issues_found.append(f"⚠️  Small universe: {len(universe_result)} stocks")
    
    if len(fundamental_result) == 0:
        issues_found.append("❌ No fundamental data available")
    elif len(fundamental_result) < len(test_universe) * 0.5:
        issues_found.append(f"⚠️  Limited fundamental data: {len(fundamental_result)} records")
    
    if len(pe_result) == 0:
        issues_found.append("❌ No P/E data available")
    elif len(pe_result) < len(test_universe) * 0.5:
        issues_found.append(f"⚠️  Limited P/E data: {len(pe_result)} records")
    
    if len(qualified) == 0:
        issues_found.append("❌ No stocks pass entry criteria")
    elif len(qualified) < len(mock_factors) * 0.5:
        issues_found.append(f"⚠️  Restrictive entry criteria: {len(qualified)}/{len(mock_factors)} stocks pass")
    
    if len(issues_found) == 0:
        print("✅ Trading logic appears to be working correctly")
        print("   - Check other parts of the strategy")
    else:
        print("❌ Issues found in trading logic:")
        for issue in issues_found:
            print(f"   {issue}")

if __name__ == "__main__":
    test_trading_logic() 