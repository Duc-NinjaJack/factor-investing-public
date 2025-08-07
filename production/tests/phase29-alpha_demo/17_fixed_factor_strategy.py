#!/usr/bin/env python3

# %% [markdown]
# # QVM Engine v3j Long-Only Real Data Strategy - Version 17 (Fixed Factors)
# 
# This strategy uses REAL data with FIXED factor calculations and proper universe selection.
# 
# ## Key Fixes in Version 17:
# - **Fixed Factor Normalization**: Ranking + Z-score normalization for all factors
# - **Proper ADTV Threshold**: 10 billion VND minimum instead of top 200 ranking
# - **Rebalanced Factor Weights**: Reduced quality dominance, improved value representation
# - **Enhanced Regime Detection**: More stable regime classification

# %%
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add Project Root to Python Path
current_path = Path.cwd()
while not (current_path / 'production').is_dir():
    if current_path.parent == current_path:
        raise FileNotFoundError("Could not find the 'production' directory.")
    current_path = current_path.parent

project_root = current_path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from production.database.connection import get_database_manager

print("✅ QVM Engine v3j Long-Only Real Data Strategy - Version 17 (Fixed Factors)")

# %%
# FIXED STRATEGY CONFIGURATION - VERSION 17
QVM_CONFIG = {
    "strategy_name": "QVM_Engine_v3j_Long_Only_Real_Data_Fixed_Factors_v17",
    "backtest_start_date": "2016-01-01",
    "backtest_end_date": "2025-12-31",
    "rebalance_frequency": "M",
    "transaction_cost_bps": 30,
    "universe": {
        "lookback_days": 63,
        "adtv_threshold_bn": 10.0,  # 10 billion VND minimum ADTV (FIXED)
        "target_portfolio_size": 20,
    },
    "regime_detection": {
        "lookback_days": 30,
        "volatility_threshold": 0.75,
        "return_threshold": 0.25,
        "bull_return_threshold": 0.75,
        "min_regime_duration": 5,
    },
    "factor_weights": {
        "normal": {"quality": 0.30, "value": 0.40, "momentum": 0.30},  # Rebalanced - more value focus
        "stress": {"quality": 0.50, "value": 0.25, "momentum": 0.25, "allocation": 0.6},
        "bull": {"quality": 0.20, "value": 0.30, "momentum": 0.50, "allocation": 1.0},
    },
    "strategy_type": "long_only_fixed_factors_v17",
}

# %%
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

def load_real_price_data(db_engine, start_date, end_date):
    """Load real price data with proper ADTV calculation."""
    print(f"📊 Loading real price data from {start_date} to {end_date}...")
    
    query = f"""
    SELECT 
        trading_date as date,
        ticker,
        close_price_adjusted as close_price,
        total_volume as volume,
        market_cap,
        (close_price_adjusted * total_volume) as daily_value
    FROM vcsc_daily_data_complete
    WHERE trading_date >= '{start_date}' AND trading_date <= '{end_date}'
    ORDER BY trading_date, ticker
    """
    
    try:
        price_data = pd.read_sql(query, db_engine)
        price_data['date'] = pd.to_datetime(price_data['date'])
        price_data['return'] = price_data.groupby('ticker')['close_price'].pct_change()
        
        print(f"   ✅ Loaded {len(price_data):,} price records")
        return price_data
        
    except Exception as e:
        print(f"   ❌ Failed to load price data: {e}")
        return pd.DataFrame()

def load_real_factor_scores(db_engine, start_date, end_date):
    """Load real factor scores with proper component selection."""
    print(f"📊 Loading real factor scores from {start_date} to {end_date}...")
    
    query = f"""
    SELECT 
        date,
        ticker,
        Quality_Composite,
        Value_Composite,
        Momentum_Composite,
        QVM_Composite
    FROM factor_scores_qvm
    WHERE date >= '{start_date}' AND date <= '{end_date}'
    ORDER BY date, ticker
    """
    
    try:
        factor_data = pd.read_sql(query, db_engine)
        factor_data['date'] = pd.to_datetime(factor_data['date'])
        
        print(f"   ✅ Loaded {len(factor_data):,} factor records")
        return factor_data
        
    except Exception as e:
        print(f"   ❌ Failed to load factor data: {e}")
        return pd.DataFrame()

def load_benchmark_data(db_engine, start_date, end_date):
    """Load benchmark data for regime detection."""
    print(f"📊 Loading benchmark data from {start_date} to {end_date}...")
    
    query = f"""
    SELECT 
        date,
        close as close_price
    FROM etf_history
    WHERE ticker = 'VNINDEX' 
    AND date >= '{start_date}' AND date <= '{end_date}'
    ORDER BY date
    """
    
    try:
        benchmark_data = pd.read_sql(query, db_engine)
        benchmark_data['date'] = pd.to_datetime(benchmark_data['date'])
        benchmark_data['return'] = benchmark_data['close_price'].pct_change()
        benchmark_data['cumulative_return'] = (1 + benchmark_data['return']).cumprod()
        
        print(f"   ✅ Loaded {len(benchmark_data)} benchmark records")
        return benchmark_data
        
    except Exception as e:
        print(f"   ❌ Failed to load benchmark data: {e}")
        return pd.DataFrame()

# %%
def calculate_adtv_universe_fixed(price_data, config):
    """Calculate ADTV universe with FIXED 10B VND threshold."""
    print("📊 Calculating ADTV universe with FIXED threshold...")
    
    lookback_days = config['universe']['lookback_days']
    adtv_threshold_bn = config['universe']['adtv_threshold_bn']
    
    # Calculate rolling ADTV (Average Daily Trading Value)
    price_data = price_data.sort_values(['ticker', 'date'])
    price_data['adtv'] = price_data.groupby('ticker')['daily_value'].rolling(
        window=lookback_days, min_periods=lookback_days//2
    ).mean().reset_index(0, drop=True)
    
    # Convert to billion VND
    price_data['adtv_bn'] = price_data['adtv'] / 1e9
    
    # Calculate universe rankings for each date
    rankings = []
    unique_dates = price_data['date'].unique()
    
    for i, date in enumerate(unique_dates):
        if i % 100 == 0:
            print(f"   📅 Processing date {i+1}/{len(unique_dates)}: {date.strftime('%Y-%m-%d')}")
        
        date_data = price_data[price_data['date'] == date].copy()
        date_data = date_data.dropna(subset=['adtv_bn'])
        
        if len(date_data) > 0:
            # Apply 10B VND threshold (FIXED)
            liquid_stocks = date_data[date_data['adtv_bn'] >= adtv_threshold_bn].copy()
            
            if len(liquid_stocks) > 0:
                liquid_stocks['adtv_rank'] = liquid_stocks['adtv_bn'].rank(ascending=False)
                liquid_stocks['in_universe'] = True
                rankings.append(liquid_stocks[['ticker', 'date', 'adtv_bn', 'adtv_rank', 'in_universe']])
    
    if rankings:
        rankings_df = pd.concat(rankings, ignore_index=True)
        print(f"   ✅ Calculated rankings for {len(rankings_df):,} records")
        print(f"   📊 Average liquid stocks per date: {len(rankings_df) / len(unique_dates):.1f}")
        return rankings_df
    else:
        print("   ❌ No rankings calculated")
        return pd.DataFrame()

def normalize_factors_properly(factor_data, universe_data, config):
    """Normalize factors using ranking + z-score methodology (FIXED)."""
    print("📊 Normalizing factors with proper methodology...")
    
    # Merge factor data with universe data
    factor_universe = factor_data.merge(
        universe_data[['ticker', 'date', 'in_universe']], 
        on=['ticker', 'date'], 
        how='inner'
    )
    
    # Filter to universe stocks only
    factor_universe = factor_universe[factor_universe['in_universe'] == True].copy()
    
    normalized_factors = []
    
    for date in factor_universe['date'].unique():
        date_data = factor_universe[factor_universe['date'] == date].copy()
        
        if len(date_data) < 10:  # Need minimum stocks for normalization
            continue
        
        # Step 1: Calculate percentile ranks (0-1)
        for factor in ['Quality_Composite', 'Value_Composite', 'Momentum_Composite']:
            date_data[f'{factor}_rank'] = date_data[factor].rank(pct=True)
        
        # Step 2: Convert ranks to z-scores using inverse normal distribution
        for factor in ['Quality_Composite', 'Value_Composite', 'Momentum_Composite']:
            rank_col = f'{factor}_rank'
            # Convert percentile rank to z-score using inverse normal
            date_data[f'{factor}_normalized'] = norm.ppf(date_data[rank_col])
            # Handle edge cases (0 and 1 ranks)
            date_data.loc[date_data[rank_col] == 0, f'{factor}_normalized'] = -3.0
            date_data.loc[date_data[rank_col] == 1, f'{factor}_normalized'] = 3.0
        
        normalized_factors.append(date_data)
    
    if normalized_factors:
        result_df = pd.concat(normalized_factors, ignore_index=True)
        print(f"   ✅ Normalized factors for {len(result_df):,} records")
        
        # Print normalization statistics
        for factor in ['Quality_Composite', 'Value_Composite', 'Momentum_Composite']:
            norm_col = f'{factor}_normalized'
            stats = result_df[norm_col].describe()
            print(f"   📊 {factor} normalized: Mean={stats['mean']:.3f}, Std={stats['std']:.3f}, Min={stats['min']:.3f}, Max={stats['max']:.3f}")
        
        return result_df
    else:
        print("   ❌ No normalized factors calculated")
        return pd.DataFrame()

def detect_market_regime_fixed(benchmark_data, config):
    """Detect market regime with enhanced stability controls."""
    print("📊 Detecting market regime with enhanced stability...")
    
    lookback_days = config['regime_detection']['lookback_days']
    vol_threshold_pct = config['regime_detection']['volatility_threshold']
    return_threshold_pct = config['regime_detection']['return_threshold']
    bull_return_threshold_pct = config['regime_detection']['bull_return_threshold']
    min_regime_duration = config['regime_detection']['min_regime_duration']
    
    # Calculate rolling volatility and returns
    benchmark_data = benchmark_data.sort_values('date')
    benchmark_data['rolling_vol'] = benchmark_data['return'].rolling(lookback_days).std() * np.sqrt(252)
    benchmark_data['rolling_return'] = benchmark_data['return'].rolling(lookback_days).mean() * 252
    
    # Define regime thresholds
    vol_threshold = benchmark_data['rolling_vol'].quantile(vol_threshold_pct)
    return_threshold = benchmark_data['rolling_return'].quantile(return_threshold_pct)
    bull_return_threshold = benchmark_data['rolling_return'].quantile(bull_return_threshold_pct)
    
    # Initial regime classification
    benchmark_data['regime'] = 'normal'
    benchmark_data.loc[
        (benchmark_data['rolling_vol'] > vol_threshold) & 
        (benchmark_data['rolling_return'] < return_threshold), 'regime'
    ] = 'stress'
    benchmark_data.loc[
        (benchmark_data['rolling_vol'] < vol_threshold) & 
        (benchmark_data['rolling_return'] > bull_return_threshold), 'regime'
    ] = 'bull'
    
    # Apply minimum regime duration filter
    benchmark_data['regime_stable'] = benchmark_data['regime']
    
    for i in range(min_regime_duration, len(benchmark_data)):
        recent_regimes = benchmark_data['regime'].iloc[i-min_regime_duration+1:i+1]
        if len(recent_regimes.unique()) == 1:
            benchmark_data.iloc[i, benchmark_data.columns.get_loc('regime_stable')] = recent_regimes.iloc[0]
        else:
            if i > 0:
                benchmark_data.iloc[i, benchmark_data.columns.get_loc('regime_stable')] = benchmark_data.iloc[i-1]['regime_stable']
    
    benchmark_data['regime'] = benchmark_data['regime_stable']
    benchmark_data = benchmark_data.drop('regime_stable', axis=1)
    
    print(f"   ✅ Regime detection completed with enhanced stability")
    regime_counts = benchmark_data['regime'].value_counts()
    for regime, count in regime_counts.items():
        print(f"      {regime}: {count} days ({count/len(benchmark_data)*100:.1f}%)")
    
    return benchmark_data

def calculate_regime_adjusted_scores_fixed(factor_data, benchmark_data, config):
    """Calculate regime-adjusted composite scores with FIXED methodology."""
    print("📊 Calculating regime-adjusted scores with FIXED methodology...")
    
    # Merge factor data with regime information
    factor_data = factor_data.merge(
        benchmark_data[['date', 'regime']], 
        on='date', 
        how='left'
    )
    
    # Get regime-specific factor weights from config
    regime_weights = config['factor_weights']
    
    # Calculate regime-adjusted composite scores using NORMALIZED factors
    factor_data['composite_score'] = 0.0
    factor_data['allocation_multiplier'] = 1.0
    
    for regime, weights in regime_weights.items():
        mask = factor_data['regime'] == regime
        
        # Calculate composite score using NORMALIZED factors
        factor_data.loc[mask, 'composite_score'] = (
            factor_data.loc[mask, 'Quality_Composite_normalized'] * weights['quality'] +
            factor_data.loc[mask, 'Value_Composite_normalized'] * weights['value'] +
            factor_data.loc[mask, 'Momentum_Composite_normalized'] * weights['momentum']
        )
        
        # Apply allocation multiplier if specified
        if 'allocation' in weights:
            factor_data.loc[mask, 'allocation_multiplier'] = weights['allocation']
    
    # Fill any missing regimes with normal weights
    missing_mask = factor_data['composite_score'] == 0.0
    factor_data.loc[missing_mask, 'composite_score'] = (
        factor_data.loc[missing_mask, 'Quality_Composite_normalized'] * 0.30 +
        factor_data.loc[missing_mask, 'Value_Composite_normalized'] * 0.40 +
        factor_data.loc[missing_mask, 'Momentum_Composite_normalized'] * 0.30
    )
    
    print(f"   ✅ Regime-adjusted scores calculated with FIXED methodology")
    print(f"   📊 Score statistics:")
    print(f"      Mean: {factor_data['composite_score'].mean():.3f}")
    print(f"      Std: {factor_data['composite_score'].std():.3f}")
    print(f"      Min: {factor_data['composite_score'].min():.3f}")
    print(f"      Max: {factor_data['composite_score'].max():.3f}")
    
    return factor_data

# %%
def calculate_portfolio_returns(portfolio, price_data, start_date, end_date):
    """Calculate portfolio returns for a given period."""
    portfolio_tickers = portfolio['ticker'].tolist()
    
    # Get price data for portfolio stocks in the period
    period_prices = price_data[
        (price_data['date'] >= start_date) & 
        (price_data['date'] <= end_date) & 
        (price_data['ticker'].isin(portfolio_tickers))
    ].copy()
    
    if len(period_prices) == 0:
        return pd.Series(dtype='float64')
    
    # Calculate daily returns for each stock
    period_prices = period_prices.sort_values(['ticker', 'date'])
    period_prices['stock_return'] = period_prices.groupby('ticker')['close_price'].pct_change()
    
    # Calculate weighted portfolio returns
    daily_returns = []
    for date in period_prices['date'].unique():
        date_data = period_prices[period_prices['date'] == date].copy()
        date_data = date_data.merge(portfolio[['ticker', 'weight']], on='ticker', how='inner')
        
        if len(date_data) > 0:
            portfolio_return = (date_data['weight'] * date_data['stock_return']).sum()
            daily_returns.append({'date': date, 'return': portfolio_return})
    
    if daily_returns:
        return pd.Series([r['return'] for r in daily_returns], index=[r['date'] for r in daily_returns])
    else:
        return pd.Series(dtype='float64')

def run_long_only_backtest_fixed(config, db_engine):
    """Run long-only backtest with FIXED methodology."""
    print("🚀 Running long-only backtest with FIXED methodology...")
    
    # Load data
    price_data = load_real_price_data(db_engine, config['backtest_start_date'], config['backtest_end_date'])
    factor_data = load_real_factor_scores(db_engine, config['backtest_start_date'], config['backtest_end_date'])
    benchmark_data = load_benchmark_data(db_engine, config['backtest_start_date'], config['backtest_end_date'])
    
    if price_data.empty or factor_data.empty or benchmark_data.empty:
        print("❌ Failed to load data")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # Calculate ADTV universe with FIXED threshold
    universe_data = calculate_adtv_universe_fixed(price_data, config)
    
    if universe_data.empty:
        print("❌ Failed to calculate universe")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # Normalize factors with proper methodology
    normalized_factors = normalize_factors_properly(factor_data, universe_data, config)
    
    if normalized_factors.empty:
        print("❌ Failed to normalize factors")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # Detect market regime
    benchmark_data = detect_market_regime_fixed(benchmark_data, config)
    
    # Calculate regime-adjusted scores
    factor_data = calculate_regime_adjusted_scores_fixed(normalized_factors, benchmark_data, config)
    
    # Generate rebalancing dates
    rebalance_dates = pd.date_range(
        start=config['backtest_start_date'], 
        end=config['backtest_end_date'], 
        freq=config['rebalance_frequency']
    )
    
    # Run backtest
    backtest_results = []
    all_daily_returns = []
    
    for i, rebalance_date in enumerate(rebalance_dates[:-1]):
        next_rebalance_date = rebalance_dates[i + 1]
        
        # Get factor data for this date
        date_data = factor_data[factor_data['date'] == rebalance_date].copy()
        
        if len(date_data) > 0:
            # Sort by composite score (descending) - LONG-ONLY
            date_data = date_data.sort_values('composite_score', ascending=False)
            
            # Select top stocks for long-only portfolio (always 20 stocks)
            target_size = config['universe']['target_portfolio_size']
            actual_portfolio_size = min(len(date_data), target_size)
            
            # Get allocation multiplier for this date
            allocation_multiplier = date_data['allocation_multiplier'].iloc[0]
            
            # Equal weight portfolio with proportional allocation adjustment
            base_weight = 1.0 / actual_portfolio_size
            adjusted_weight = base_weight * allocation_multiplier
            
            weights = np.ones(actual_portfolio_size) * adjusted_weight
            
            portfolio = date_data.head(actual_portfolio_size).copy()
            portfolio['weight'] = weights
            
            # Store regime and allocation info for analysis
            current_regime = date_data['regime'].iloc[0]
            
            # Calculate ACTUAL returns using real price data
            period_returns = calculate_portfolio_returns(portfolio, price_data, rebalance_date, next_rebalance_date)
            
            if len(period_returns) > 0:
                # Apply transaction costs
                transaction_cost = config['transaction_cost_bps'] / 10000
                net_returns = period_returns - transaction_cost / len(period_returns)
                
                # Store daily returns for equity curve
                for date, ret in net_returns.items():
                    all_daily_returns.append({
                        'date': date,
                        'return': ret,
                        'rebalance_date': rebalance_date
                    })
                
                # Calculate period metrics
                period_return = (1 + net_returns).prod() - 1
                
                backtest_results.append({
                    'date': rebalance_date,
                    'next_date': next_rebalance_date,
                    'portfolio_return': period_return,
                    'portfolio_size': len(portfolio),
                    'avg_score': portfolio['composite_score'].mean(),
                    'regime': current_regime,
                    'allocation_multiplier': allocation_multiplier,
                    'actual_allocation': allocation_multiplier,
                    'avg_position_size': portfolio['weight'].mean()
                })
    
    # Convert results to DataFrames
    backtest_df = pd.DataFrame(backtest_results)
    daily_returns_df = pd.DataFrame(all_daily_returns)
    
    print(f"✅ Backtest completed with FIXED methodology")
    print(f"📊 Results summary:")
    print(f"   - Total rebalancing periods: {len(backtest_df)}")
    print(f"   - Total daily returns: {len(daily_returns_df)}")
    
    if len(backtest_df) > 0:
        total_return = (1 + backtest_df['portfolio_return']).prod() - 1
        print(f"   - Total return: {total_return:.2%}")
        print(f"   - Average monthly return: {backtest_df['portfolio_return'].mean():.2%}")
    
    return backtest_df, benchmark_data, daily_returns_df

# %%
def main():
    """Main execution function."""
    print("🚀 Starting QVM Engine v3j Long-Only Real Data Strategy - Version 17 (Fixed Factors)")
    print("="*80)
    
    try:
        # Create database connection
        db_engine = create_db_connection()
        
        # Run backtest with FIXED methodology
        backtest_results, benchmark_data, daily_returns = run_long_only_backtest_fixed(QVM_CONFIG, db_engine)
        
        if backtest_results.empty:
            print("❌ Backtest failed")
            return
        
        # Save results
        results_dir = Path("insights")
        results_dir.mkdir(exist_ok=True)
        
        backtest_results.to_csv(results_dir / "fixed_factor_backtest_results.csv", index=False)
        daily_returns.to_csv(results_dir / "fixed_factor_daily_returns.csv", index=False)
        
        print(f"\n✅ Strategy execution completed successfully!")
        print(f"📊 Results saved to {results_dir}/")
        print(f"🎯 Key improvements in Version 17:")
        print(f"   - Fixed factor normalization (ranking + z-score)")
        print(f"   - Proper ADTV threshold (10B VND)")
        print(f"   - Rebalanced factor weights (reduced quality dominance)")
        print(f"   - Enhanced regime detection stability")
        
    except Exception as e:
        print(f"❌ Strategy execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
