#!/usr/bin/env python3
"""
QVM Engine v3j Comprehensive Multi-Factor Strategy v2
====================================================

This strategy combines 6 factors using VNSC data for maximum coverage:
- ROAA (Quality) - from raw fundamental data
- P/E (Value) - from raw fundamental data + market data
- Momentum (4-horizon) - from VNSC daily data
- FCF Yield (Value) - from raw fundamental data
- F-Score (Quality) - from raw fundamental data
- Low Volatility (Risk) - from VNSC daily data

The strategy uses VNSC daily data and raw fundamental data for maximum
historical coverage and precise financial calculations.
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
current_path = Path.cwd()
while not (current_path / 'production').is_dir():
    if current_path.parent == current_path:
        raise FileNotFoundError("Could not find the 'production' directory.")
    current_path = current_path.parent

project_root = current_path

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import production modules
from production.database.connection import get_database_manager

# Import our custom components
sys.path.append(str(project_root / 'production' / 'tests' / 'phase29-alpha_demo' / 'components'))
from fundamental_factor_calculator import FundamentalFactorCalculator
from momentum_volatility_calculator import MomentumVolatilityCalculator

print(f"✅ Successfully imported production modules.")
print(f"   - Project Root set to: {project_root}")

# COMPREHENSIVE MULTI-FACTOR CONFIGURATION
QVM_CONFIG = {
    "strategy_name": "QVM_Engine_v3j_Comprehensive_Multi_Factor_v2",
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
        # Quality factors (1/3 total weight)
        "roaa_weight": 0.167,  # 0.5 * 1/3 = 0.167
        "f_score_weight": 0.167,  # 0.5 * 1/3 = 0.167
        
        # Value factors (1/3 total weight)
        "pe_weight": 0.167,  # 0.5 * 1/3 = 0.167
        "fcf_yield_weight": 0.167,  # 0.5 * 1/3 = 0.167
        
        # Momentum factors (1/3 total weight)
        "momentum_weight": 0.167,  # 0.5 * 1/3 = 0.167
        "low_vol_weight": 0.167,  # 0.5 * 1/3 = 0.167
        
        "momentum_horizons": [21, 63, 126, 252],
        "skip_months": 1,
        "fundamental_lag_days": 45,
    }
}

print("\n⚙️  QVM Engine v3j Comprehensive Multi-Factor v2 Configuration Loaded:")
print(f"   - Strategy: {QVM_CONFIG['strategy_name']}")
print(f"   - Period: {QVM_CONFIG['backtest_start_date']} to {QVM_CONFIG['backtest_end_date']}")
print(f"   - Universe: Top {QVM_CONFIG['universe']['top_n_stocks']} stocks by ADTV")
print(f"   - Rebalancing: {QVM_CONFIG['rebalance_frequency']} frequency")
print(f"   - Quality (1/3): ROAA 50% + F-Score 50%")
print(f"   - Value (1/3): P/E 50% + FCF Yield 50%")
print(f"   - Momentum (1/3): 4-Horizon 50% + Low Vol 50%")
print(f"   - Data Source: VNSC daily data + Raw fundamental data")

def create_db_connection():
    """Create database connection."""
    try:
        db_manager = get_database_manager()
        engine = db_manager.get_engine()
        print("✅ Database connection established")
        return engine
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        raise

def load_universe_data(config, db_engine):
    """Load universe data using VNSC daily data."""
    print("📊 Loading universe data...")
    
    start_date = pd.to_datetime(config['backtest_start_date']) - timedelta(days=config['universe']['lookback_days'])
    
    query = text("""
        SELECT 
            ticker,
            trading_date,
            close_price_adjusted as close,
            total_volume as volume,
            total_value as value,
            market_cap
        FROM vcsc_daily_data_complete
        WHERE trading_date >= :start_date
        AND close_price_adjusted > 0
        AND total_volume > 0
        ORDER BY ticker, trading_date
    """)
    
    universe_data = pd.read_sql(query, db_engine, params={'start_date': start_date})
    
    print(f"   ✅ Loaded {len(universe_data):,} universe records")
    print(f"   📊 Coverage: {universe_data['ticker'].nunique()} tickers")
    
    return universe_data

def calculate_universe_rankings(universe_data, config):
    """Calculate universe rankings based on average daily turnover."""
    print("📊 Calculating universe rankings...")
    
    # Calculate average daily turnover for each stock
    rankings = universe_data.groupby('ticker').agg({
        'volume': 'mean',
        'value': 'mean',
        'market_cap': 'mean',
        'trading_date': 'max'
    }).reset_index()
    
    # Calculate average daily turnover (volume * price)
    rankings['avg_daily_turnover'] = rankings['volume'] * rankings['market_cap'] / rankings['market_cap']
    
    # Sort by average daily turnover and get top N
    top_n = config['universe']['top_n_stocks']
    rankings = rankings.nlargest(top_n * 2, 'avg_daily_turnover')  # Get 2x for filtering
    
    # Add ranking
    rankings['ranking'] = range(1, len(rankings) + 1)
    
    print(f"   ✅ Calculated rankings for {len(rankings)} stocks")
    print(f"   📊 Top stock: {rankings.iloc[0]['ticker']} (turnover: {rankings.iloc[0]['avg_daily_turnover']:,.0f})")
    
    return rankings

def load_benchmark_data(config, db_engine):
    """Load benchmark data (VN-Index)."""
    print("📊 Loading benchmark data...")
    
    query = text("""
        SELECT 
            trading_date,
            close_price_adjusted as close
        FROM vcsc_daily_data_complete
        WHERE ticker = 'VNM'
        AND trading_date BETWEEN :start_date AND :end_date
        ORDER BY trading_date
    """)
    
    benchmark_data = pd.read_sql(query, db_engine, params={
        'start_date': config['backtest_start_date'],
        'end_date': config['backtest_end_date']
    })
    
    # Calculate benchmark returns
    benchmark_data['returns'] = benchmark_data['close'].pct_change()
    
    print(f"   ✅ Loaded {len(benchmark_data)} benchmark records")
    print(f"   📅 Period: {benchmark_data['trading_date'].min()} to {benchmark_data['trading_date'].max()}")
    
    return benchmark_data

def calculate_fundamental_factors(config, db_engine):
    """Calculate fundamental factors using raw data."""
    print("📊 Calculating fundamental factors...")
    
    # Initialize fundamental calculator
    fundamental_calc = FundamentalFactorCalculator(db_engine)
    
    # Calculate factors for the entire period
    fundamental_factors = fundamental_calc.calculate_all_factors(
        config['backtest_start_date'],
        config['backtest_end_date']
    )
    
    print(f"   ✅ Calculated fundamental factors for {len(fundamental_factors)} records")
    print(f"   📊 Coverage: {fundamental_factors['ticker'].nunique()} tickers")
    
    return fundamental_factors

def calculate_momentum_volatility_factors(config, db_engine):
    """Calculate momentum and volatility factors using VNSC data."""
    print("📊 Calculating momentum and volatility factors...")
    
    # Initialize momentum/volatility calculator
    momentum_vol_calc = MomentumVolatilityCalculator(db_engine)
    
    # Calculate factors for the entire period
    momentum_vol_factors = momentum_vol_calc.calculate_all_factors(
        config['backtest_start_date'],
        config['backtest_end_date']
    )
    
    print(f"   ✅ Calculated momentum/volatility factors for {len(momentum_vol_factors)} records")
    print(f"   📊 Coverage: {momentum_vol_factors['ticker'].nunique()} tickers")
    
    return momentum_vol_factors

def combine_all_factors(fundamental_factors, momentum_vol_factors, universe_rankings):
    """Combine all factors into a single dataset."""
    print("📊 Combining all factors...")
    
    # Get universe tickers
    universe_tickers = universe_rankings['ticker'].tolist()
    
    # Filter factors to universe
    fundamental_filtered = fundamental_factors[fundamental_factors['ticker'].isin(universe_tickers)]
    momentum_vol_filtered = momentum_vol_factors[momentum_vol_factors['ticker'].isin(universe_tickers)]
    
    # Convert fundamental date to datetime for merging
    fundamental_filtered['date'] = pd.to_datetime(fundamental_filtered['date'])
    
    # Merge fundamental and momentum/volatility factors
    combined_factors = fundamental_filtered.merge(
        momentum_vol_factors[['ticker', 'trading_date', 'composite_momentum', 'low_vol_score', 'momentum_vol_score']],
        left_on=['ticker', 'date'],
        right_on=['ticker', 'trading_date'],
        how='outer'
    )
    
    # Fill missing values
    combined_factors['roaa'] = combined_factors['roaa'].fillna(0)
    combined_factors['pe_ratio'] = combined_factors['pe_ratio'].fillna(50)
    combined_factors['fcf_yield'] = combined_factors['fcf_yield'].fillna(0)
    combined_factors['f_score'] = combined_factors['f_score'].fillna(0)
    combined_factors['composite_momentum'] = combined_factors['composite_momentum'].fillna(0)
    combined_factors['low_vol_score'] = combined_factors['low_vol_score'].fillna(0.5)
    
    print(f"   ✅ Combined factors for {len(combined_factors)} records")
    print(f"   📊 Coverage: {combined_factors['ticker'].nunique()} tickers")
    
    return combined_factors

def normalize_factor(factor_series):
    """Normalize factor to 0-1 range using winsorization and z-score."""
    if factor_series.empty or factor_series.isna().all():
        return pd.Series(0, index=factor_series.index)
    
    # Remove outliers using winsorization
    factor_clean = factor_series.copy()
    q1 = factor_clean.quantile(0.01)
    q99 = factor_clean.quantile(0.99)
    factor_clean = factor_clean.clip(q1, q99)
    
    # Calculate z-score
    mean_val = factor_clean.mean()
    std_val = factor_clean.std()
    
    if std_val == 0:
        return pd.Series(0.5, index=factor_series.index)
    
    z_scores = (factor_clean - mean_val) / std_val
    
    # Convert to 0-1 range using sigmoid
    normalized = 1 / (1 + np.exp(-z_scores))
    
    return normalized.fillna(0.5)

def calculate_composite_scores(combined_factors, config):
    """Calculate composite factor scores."""
    print("📊 Calculating composite scores...")
    
    # Normalize individual factors
    combined_factors['roaa_score'] = normalize_factor(combined_factors['roaa'])
    combined_factors['pe_score'] = normalize_factor(-combined_factors['pe_ratio'])  # Lower P/E is better
    combined_factors['fcf_yield_score'] = normalize_factor(combined_factors['fcf_yield'])
    combined_factors['f_score_score'] = normalize_factor(combined_factors['f_score'])
    combined_factors['momentum_score'] = normalize_factor(combined_factors['composite_momentum'])
    combined_factors['low_vol_score_final'] = normalize_factor(combined_factors['low_vol_score'])
    
    # Calculate composite scores by category
    # Quality factors (1/3 total weight)
    quality_score = (
        combined_factors['roaa_score'] * 0.5 +  # 50% of quality
        combined_factors['f_score_score'] * 0.5   # 50% of quality
    )
    
    # Value factors (1/3 total weight)
    value_score = (
        combined_factors['pe_score'] * 0.5 +      # 50% of value
        combined_factors['fcf_yield_score'] * 0.5  # 50% of value
    )
    
    # Momentum factors (1/3 total weight)
    momentum_score = (
        combined_factors['momentum_score'] * 0.5 +  # 50% of momentum (4-horizon average)
        combined_factors['low_vol_score_final'] * 0.5     # 50% of momentum (low vol)
    )
    
    # Final composite score: 1/3 Quality + 1/3 Value + 1/3 Momentum
    combined_factors['composite_score'] = (
        quality_score * (1/3) +
        value_score * (1/3) +
        momentum_score * (1/3)
    )
    
    print(f"   ✅ Calculated composite scores for {len(combined_factors)} records")
    print(f"   📊 Score range: {combined_factors['composite_score'].min():.4f} to {combined_factors['composite_score'].max():.4f}")
    
    return combined_factors

def test_strategy_components():
    """Test individual strategy components."""
    print("🧪 Testing Strategy Components...")
    
    try:
        # Test database connection
        print("   🔍 Testing database connection...")
        db_engine = create_db_connection()
        print("   ✅ Database connection successful")
        
        # Test universe data loading
        print("   🔍 Testing universe data loading...")
        universe_data = load_universe_data(QVM_CONFIG, db_engine)
        print(f"   ✅ Universe data loaded: {len(universe_data):,} records")
        
        # Test benchmark data loading
        print("   🔍 Testing benchmark data loading...")
        benchmark_data = load_benchmark_data(QVM_CONFIG, db_engine)
        print(f"   ✅ Benchmark data loaded: {len(benchmark_data)} records")
        
        # Test fundamental factor calculation (small period)
        print("   🔍 Testing fundamental factor calculation...")
        test_config = QVM_CONFIG.copy()
        test_config['backtest_start_date'] = "2020-01-01"
        test_config['backtest_end_date'] = "2020-12-31"
        
        fundamental_factors = calculate_fundamental_factors(test_config, db_engine)
        print(f"   ✅ Fundamental factors calculated: {len(fundamental_factors)} records")
        
        # Test momentum/volatility factor calculation (small period)
        print("   🔍 Testing momentum/volatility factor calculation...")
        momentum_vol_factors = calculate_momentum_volatility_factors(test_config, db_engine)
        print(f"   ✅ Momentum/volatility factors calculated: {len(momentum_vol_factors)} records")
        
        # Test factor combination
        print("   🔍 Testing factor combination...")
        universe_rankings = calculate_universe_rankings(universe_data, test_config)
        combined_factors = combine_all_factors(fundamental_factors, momentum_vol_factors, universe_rankings)
        print(f"   ✅ Factors combined: {len(combined_factors)} records")
        
        # Test composite score calculation
        print("   🔍 Testing composite score calculation...")
        combined_factors = calculate_composite_scores(combined_factors, test_config)
        print(f"   ✅ Composite scores calculated: {len(combined_factors)} records")
        
        print("✅ All strategy components tested successfully!")
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_strategy_components()
