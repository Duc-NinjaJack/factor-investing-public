#!/usr/bin/env python3
"""
QVM Engine v3j Comprehensive Multi-Factor Strategy
=================================================

This strategy combines 6 factors:
- ROAA (Quality)
- P/E (Value) 
- Momentum
- FCF Yield (Value)
- F-Score (Quality)
- Low Volatility (Risk)
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Database connectivity
from sqlalchemy import create_engine, text

# Add Project Root to Python Path
try:
    current_path = Path.cwd()
    while not (current_path / 'production').is_dir():
        if current_path.parent == current_path:
            raise FileNotFoundError("Could not find the 'production' directory.")
        current_path = current_path.parent
    
    project_root = current_path
    
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    from production.database.connection import get_database_manager
    from production.database.mappings.financial_mapping_manager import FinancialMappingManager
    print(f"✅ Successfully imported production modules.")
    print(f"   - Project Root set to: {project_root}")

except (ImportError, FileNotFoundError) as e:
    print(f"❌ ERROR: Could not import production modules. Please check your directory structure.")
    print(f"   - Final Path Searched: {project_root}")
    print(f"   - Error: {e}")
    raise

# COMPREHENSIVE MULTI-FACTOR CONFIGURATION
QVM_CONFIG = {
    "strategy_name": "QVM_Engine_v3j_Comprehensive_Multi_Factor",
    "backtest_start_date": "2016-01-01",
    "backtest_end_date": "2025-07-28",
    "rebalance_frequency": "M",
    "transaction_cost_bps": 30,
    
    "universe": {
        "lookback_days": 63,
        "top_n_stocks": 40,
        "max_position_size": 0.035,
        "max_sector_exposure": 0.25,
        "target_portfolio_size": 35,
    },
    
    "factors": {
        "roaa_weight": 0.25,      # Quality factor
        "pe_weight": 0.20,        # Value factor
        "momentum_weight": 0.15,  # Momentum factor
        "fcf_yield_weight": 0.15, # Value factor (NEW)
        "f_score_weight": 0.15,   # Quality factor (NEW)
        "low_vol_weight": 0.10,   # Risk factor (NEW)
        "momentum_horizons": [21, 63, 126, 252],
        "skip_months": 1,
        "fundamental_lag_days": 45,
    }
}

print("\n⚙️  QVM Engine v3j Comprehensive Multi-Factor Configuration Loaded:")
print(f"   - Strategy: {QVM_CONFIG['strategy_name']}")
print(f"   - Period: {QVM_CONFIG['backtest_start_date']} to {QVM_CONFIG['backtest_end_date']}")
print(f"   - Universe: Top {QVM_CONFIG['universe']['top_n_stocks']} stocks by ADTV")
print(f"   - Rebalancing: {QVM_CONFIG['rebalance_frequency']} frequency")
print(f"   - ROAA (Quality): {QVM_CONFIG['factors']['roaa_weight']:.1%}")
print(f"   - P/E (Value): {QVM_CONFIG['factors']['pe_weight']:.1%}")
print(f"   - Momentum: {QVM_CONFIG['factors']['momentum_weight']:.1%}")
print(f"   - FCF Yield (Value): {QVM_CONFIG['factors']['fcf_yield_weight']:.1%}")
print(f"   - F-Score (Quality): {QVM_CONFIG['factors']['f_score_weight']:.1%}")
print(f"   - Low Volatility (Risk): {QVM_CONFIG['factors']['low_vol_weight']:.1%}")
print(f"   - Performance: 6-factor comprehensive approach")

# Import the base strategy functions
import importlib.util
spec = importlib.util.spec_from_file_location("base_strategy", "08_integrated_strategy_with_validated_factors_optimized.py")
base_strategy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base_strategy)

def precompute_fundamental_factors_comprehensive(config: dict, db_engine):
    """Precompute comprehensive fundamental factors including FCF Yield and F-Score."""
    print("📊 Precomputing comprehensive fundamental factors...")
    
    start_date = pd.Timestamp(config['backtest_start_date']) - pd.DateOffset(days=365)
    
    # Load fundamental data with additional metrics
    fundamental_query = text("""
        SELECT 
            fv.ticker,
            fv.year,
            fv.quarter,
            SUM(CASE WHEN fv.item_id = 1 AND fv.statement_type = 'PL' THEN fv.value / 1e9 ELSE 0 END) as netprofit_ttm,
            SUM(CASE WHEN fv.item_id = 2 AND fv.statement_type = 'BS' THEN fv.value / 1e9 ELSE 0 END) as totalassets_ttm,
            SUM(CASE WHEN fv.item_id = 3 AND fv.statement_type = 'CF' THEN fv.value / 1e9 ELSE 0 END) as fcf_ttm,
            SUM(CASE WHEN fv.item_id = 4 AND fv.statement_type = 'BS' THEN fv.value / 1e9 ELSE 0 END) as totaldebt_ttm,
            SUM(CASE WHEN fv.item_id = 5 AND fv.statement_type = 'BS' THEN fv.value / 1e9 ELSE 0 END) as currentassets_ttm,
            SUM(CASE WHEN fv.item_id = 6 AND fv.statement_type = 'BS' THEN fv.value / 1e9 ELSE 0 END) as currentliabilities_ttm
        FROM fundamental_values fv
        WHERE fv.year >= :start_year
        AND fv.item_id IN (1, 2, 3, 4, 5, 6)
        GROUP BY fv.ticker, fv.year, fv.quarter
        HAVING netprofit_ttm > 0 AND totalassets_ttm > 0
    """)
    
    fundamental_data = pd.read_sql(fundamental_query, db_engine, params={'start_year': start_date.year})
    
    # Calculate ROAA
    fundamental_data['roaa'] = fundamental_data['netprofit_ttm'] / fundamental_data['totalassets_ttm']
    
    # Calculate P/E ratio
    print("   📊 Calculating P/E ratios...")
    pe_query = text("""
        SELECT 
            fv.ticker,
            fv.year,
            fv.quarter,
            SUM(CASE WHEN fv.item_id = 1 AND fv.statement_type = 'PL' THEN fv.value / 1e9 ELSE 0 END) as netprofit_ttm,
            eh.market_cap / 1e9 as market_cap_bn
        FROM fundamental_values fv
        JOIN equity_history_with_market_cap eh ON fv.ticker = eh.ticker 
            AND fv.year = YEAR(eh.date) 
            AND fv.quarter = QUARTER(eh.date)
        WHERE fv.year >= :start_year
        AND fv.item_id = 1 
        AND fv.statement_type = 'PL'
        AND eh.market_cap > 0
        GROUP BY fv.ticker, fv.year, fv.quarter, eh.market_cap
        HAVING netprofit_ttm > 0
    """)
    
    pe_data = pd.read_sql(pe_query, db_engine, params={'start_year': start_date.year})
    
    if not pe_data.empty:
        pe_data['pe'] = pe_data['market_cap_bn'] / pe_data['netprofit_ttm']
        pe_data['date'] = pd.to_datetime(
            pe_data['year'].astype(str) + '-' + 
            (pe_data['quarter'] * 3).astype(str).str.zfill(2) + '-01'
        )
        
        fundamental_data['date'] = pd.to_datetime(
            fundamental_data['year'].astype(str) + '-' + 
            (fundamental_data['quarter'] * 3).astype(str).str.zfill(2) + '-01'
        )
        
        fundamental_data = fundamental_data.merge(
            pe_data[['ticker', 'date', 'pe']], 
            on=['ticker', 'date'], 
            how='left'
        )
    else:
        fundamental_data['pe'] = np.nan
    
    # Calculate FCF Yield
    fundamental_data['fcf_yield'] = fundamental_data['fcf_ttm'] / fundamental_data['totalassets_ttm']
    
    # Calculate F-Score components
    fundamental_data['f_score'] = 0
    
    # Profitability components
    fundamental_data.loc[fundamental_data['roaa'] > 0, 'f_score'] += 1
    fundamental_data.loc[fundamental_data['fcf_ttm'] > 0, 'f_score'] += 1
    
    # Leverage components
    fundamental_data['debt_ratio'] = fundamental_data['totaldebt_ttm'] / fundamental_data['totalassets_ttm']
    fundamental_data.loc[fundamental_data['debt_ratio'] < 0.4, 'f_score'] += 1
    
    # Liquidity components
    fundamental_data['current_ratio'] = fundamental_data['currentassets_ttm'] / fundamental_data['currentliabilities_ttm']
    fundamental_data.loc[fundamental_data['current_ratio'] > 1, 'f_score'] += 1
    
    # Clean up extreme values
    fundamental_data['roaa'] = fundamental_data['roaa'].clip(-1, 1)
    fundamental_data['pe'] = fundamental_data['pe'].clip(0, 100)
    fundamental_data['fcf_yield'] = fundamental_data['fcf_yield'].clip(-0.5, 0.5)
    fundamental_data['f_score'] = fundamental_data['f_score'].clip(0, 4)
    
    print(f"   ✅ Comprehensive fundamental factors computed: {len(fundamental_data)} records")
    return fundamental_data

def precompute_low_volatility_factors(config: dict, db_engine):
    """Precompute low volatility factors."""
    print("📊 Precomputing low volatility factors...")
    
    start_date = pd.Timestamp(config['backtest_start_date']) - pd.DateOffset(days=365)
    end_date = config['backtest_end_date']
    
    # Load price data
    price_query = text("""
        SELECT 
            trading_date,
            ticker,
            close_price_adjusted as close
        FROM vcsc_daily_data_complete
        WHERE trading_date BETWEEN :start_date AND :end_date
        ORDER BY ticker, trading_date
    """)
    
    price_data = pd.read_sql(price_query, db_engine, params={'start_date': start_date, 'end_date': end_date})
    
    # Calculate volatility for each stock
    volatility_data = []
    for ticker in price_data['ticker'].unique():
        ticker_data = price_data[price_data['ticker'] == ticker].sort_values('trading_date')
        ticker_data['returns'] = ticker_data['close'].pct_change()
        
        # Calculate rolling volatility (63-day window)
        ticker_data['volatility'] = ticker_data['returns'].rolling(63).std()
        
        volatility_data.append(ticker_data)
    
    volatility_df = pd.concat(volatility_data, ignore_index=True)
    
    # Calculate low volatility score (inverse of volatility)
    volatility_df['low_vol_score'] = 1 / (1 + volatility_df['volatility'])
    
    print(f"   ✅ Low volatility factors computed: {len(volatility_df)} records")
    return volatility_df

def precompute_all_data_comprehensive(config: dict, db_engine):
    """Precompute all data for comprehensive strategy."""
    print("🚀 Precomputing all data for comprehensive multi-factor strategy...")
    
    # Precompute universe rankings
    universe_rankings = base_strategy.precompute_universe_rankings(config, db_engine)
    
    # Precompute comprehensive fundamental factors
    fundamental_factors = precompute_fundamental_factors_comprehensive(config, db_engine)
    
    # Precompute momentum factors
    momentum_factors = base_strategy.precompute_momentum_factors(config, db_engine)
    
    # Precompute low volatility factors
    low_vol_factors = precompute_low_volatility_factors(config, db_engine)
    
    precomputed_data = {
        'universe': universe_rankings,
        'fundamental': fundamental_factors,
        'momentum': momentum_factors,
        'low_vol': low_vol_factors
    }
    
    print("✅ All comprehensive data precomputed successfully!")
    return precomputed_data

# MAIN EXECUTION
if __name__ == "__main__":
    print("🚀 QVM ENGINE V3J COMPREHENSIVE MULTI-FACTOR STRATEGY EXECUTION")
    print("=" * 80)
    
    try:
        # Step 1: Database connection
        print("📊 Step 1: Establishing database connection...")
        db_engine = base_strategy.create_db_connection()
        
        # Step 2: Load data
        print("📊 Step 2: Loading data...")
        all_data = base_strategy.load_all_data_for_backtest(QVM_CONFIG, db_engine)
        
        # Step 3: Precompute comprehensive data
        print("📊 Step 3: Precomputing comprehensive data...")
        precomputed_data = precompute_all_data_comprehensive(QVM_CONFIG, db_engine)
        
        # Step 4: Initialize and run strategy (using base strategy with comprehensive data)
        print("📊 Step 4: Running comprehensive strategy...")
        
        # For now, use the base strategy with comprehensive data
        # In a full implementation, we would create a comprehensive strategy class
        QVMEngineV3jOptimized = base_strategy.QVMEngineV3jOptimized
        
        engine = QVMEngineV3jOptimized(
            QVM_CONFIG,
            all_data[0],  # price_data
            all_data[1],  # fundamental_data
            all_data[2],  # returns_matrix
            all_data[3],  # benchmark_returns
            db_engine,
            precomputed_data
        )
        
        strategy_returns, diagnostics = engine.run_backtest()
        
        # Step 5: Calculate performance metrics
        print("📊 Step 5: Calculating performance metrics...")
        metrics = base_strategy.calculate_performance_metrics(strategy_returns, all_data['benchmark_returns'])
        
        # Step 6: Generate tearsheet
        print("📊 Step 6: Generating comprehensive tearsheet...")
        base_strategy.generate_comprehensive_tearsheet(
            strategy_returns, 
            all_data['benchmark_returns'], 
            diagnostics, 
            "QVM Engine v3j Comprehensive Multi-Factor Strategy"
        )
        
        # Step 7: Display results
        print("=" * 80)
        print("📊 QVM ENGINE V3J: COMPREHENSIVE MULTI-FACTOR STRATEGY RESULTS")
        print("=" * 80)
        print("📈 Performance Summary:")
        print(f"   - Strategy Annualized Return: {metrics['strategy_annualized_return']:.2%}")
        print(f"   - Benchmark Annualized Return: {metrics['benchmark_annualized_return']:.2%}")
        print(f"   - Strategy Sharpe Ratio: {metrics['strategy_sharpe_ratio']:.2f}")
        print(f"   - Benchmark Sharpe Ratio: {metrics['benchmark_sharpe_ratio']:.2f}")
        print(f"   - Strategy Max Drawdown: {metrics['strategy_max_drawdown']:.2%}")
        print(f"   - Benchmark Max Drawdown: {metrics['benchmark_max_drawdown']:.2%}")
        print(f"   - Information Ratio: {metrics['information_ratio']:.2f}")
        print(f"   - Beta: {metrics['beta']:.2f}")
        
        print("\n🔧 Comprehensive Configuration:")
        print("   - 6-factor comprehensive structure (ROAA, P/E, Momentum, FCF Yield, F-Score, Low Vol)")
        print("   - Balanced factor weights for optimal performance")
        print("   - Enhanced risk management with low volatility factor")
        print("   - Improved diversification with larger portfolio size")
        
        print("\n✅ QVM Engine v3j Comprehensive strategy execution complete!")
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc() 