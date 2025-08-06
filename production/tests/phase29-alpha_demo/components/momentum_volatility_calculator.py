#!/usr/bin/env python3
"""
Momentum and Low Volatility Factor Calculator
=============================================

This component calculates momentum and low volatility factors using VNSC daily data
for maximum coverage and precision. It handles:

1. Multi-horizon momentum factors (21, 63, 126, 252 days)
2. Low volatility factors (63-day rolling volatility)
3. Volume-weighted momentum
4. Risk-adjusted momentum

The calculator uses VNSC daily data to ensure maximum historical coverage
and precise financial calculations.
"""

# %% [markdown]
# # IMPORTS AND SETUP

# %%
import sys
import os
import pandas as pd
import numpy as np
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
    print(f"✅ Successfully imported production modules.")
    print(f"   - Project Root set to: {project_root}")

except (ImportError, FileNotFoundError) as e:
    print(f"❌ ERROR: Could not import production modules. Please check your directory structure.")
    print(f"   - Final Path Searched: {project_root}")
    print(f"   - Error: {e}")
    raise

# %% [markdown]
# # MOMENTUM AND VOLATILITY CALCULATOR CLASS

# %%
class MomentumVolatilityCalculator:
    """
    Comprehensive momentum and volatility factor calculator using VNSC daily data.
    
    This calculator provides maximum coverage by using VNSC daily data
    for precise momentum and volatility calculations.
    """
    
    def __init__(self, db_engine):
        """Initialize the calculator with database connection."""
        self.db_engine = db_engine
        
        # Define momentum horizons
        self.momentum_horizons = [21, 63, 126, 252]  # 1M, 3M, 6M, 12M
        
        print("✅ Momentum and Volatility Calculator initialized")
        print("   - Using VNSC daily data for maximum coverage")
        print(f"   - Momentum horizons: {self.momentum_horizons}")
    
    def load_price_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load price data from VNSC daily data."""
        print(f"📊 Loading price data from {start_date} to {end_date}...")
        
        # Load price data with additional buffer for momentum calculations
        buffer_days = max(self.momentum_horizons) + 30  # Extra buffer
        buffer_start = (pd.to_datetime(start_date) - timedelta(days=buffer_days)).strftime('%Y-%m-%d')
        
        query = text("""
            SELECT 
                ticker,
                trading_date,
                close_price_adjusted as close,
                total_volume as volume,
                total_value as value,
                market_cap
            FROM vcsc_daily_data_complete
            WHERE trading_date BETWEEN :start_date AND :end_date
            AND close_price_adjusted > 0
            ORDER BY ticker, trading_date
        """)
        
        data = pd.read_sql(query, self.db_engine, params={
            'start_date': buffer_start,
            'end_date': end_date
        })
        
        print(f"   ✅ Loaded {len(data):,} price records")
        print(f"   📊 Coverage: {data['ticker'].nunique()} tickers")
        print(f"   📅 Period: {data['trading_date'].min()} to {data['trading_date'].max()}")
        
        return data
    
    def calculate_returns(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate daily returns for all stocks."""
        print("📊 Calculating daily returns...")
        
        # Sort by ticker and date
        price_data = price_data.sort_values(['ticker', 'trading_date'])
        
        # Calculate daily returns
        price_data['returns'] = price_data.groupby('ticker')['close'].pct_change()
        
        # Remove first day for each stock (no return)
        price_data = price_data.dropna(subset=['returns'])
        
        print(f"   ✅ Calculated returns for {len(price_data):,} records")
        return price_data
    
    def calculate_momentum_factors(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate multi-horizon momentum factors."""
        print("📊 Calculating momentum factors...")
        
        momentum_data = []
        
        for ticker in price_data['ticker'].unique():
            ticker_data = price_data[price_data['ticker'] == ticker].copy()
            ticker_data = ticker_data.sort_values('trading_date')
            
            # Calculate momentum for each horizon
            for horizon in self.momentum_horizons:
                ticker_data[f'momentum_{horizon}d'] = (
                    ticker_data['close'] / ticker_data['close'].shift(horizon) - 1
                )
            
            # Calculate volume-weighted momentum
            ticker_data['volume_weighted_momentum'] = (
                ticker_data['returns'] * ticker_data['volume']
            ).rolling(window=63).mean()
            
            # Calculate risk-adjusted momentum (Sharpe-like ratio)
            returns_std = ticker_data['returns'].rolling(window=63).std()
            ticker_data['risk_adjusted_momentum'] = (
                ticker_data['returns'].rolling(window=63).mean() / returns_std
            ).fillna(0)
            
            momentum_data.append(ticker_data)
        
        momentum_df = pd.concat(momentum_data, ignore_index=True)
        
        # Calculate composite momentum score
        momentum_columns = [f'momentum_{h}d' for h in self.momentum_horizons]
        momentum_df['composite_momentum'] = momentum_df[momentum_columns].mean(axis=1)
        
        print(f"   ✅ Calculated momentum factors for {len(momentum_df):,} records")
        return momentum_df
    
    def calculate_volatility_factors(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate low volatility factors."""
        print("📊 Calculating volatility factors...")
        
        volatility_data = []
        
        for ticker in price_data['ticker'].unique():
            ticker_data = price_data[price_data['ticker'] == ticker].copy()
            ticker_data = ticker_data.sort_values('trading_date')
            
            # Calculate rolling volatility (63-day window)
            ticker_data['volatility_63d'] = ticker_data['returns'].rolling(63).std()
            
            # Calculate low volatility score (inverse of volatility)
            ticker_data['low_vol_score'] = 1 / (1 + ticker_data['volatility_63d'])
            
            # Calculate downside volatility (semi-deviation)
            negative_returns = ticker_data['returns'].copy()
            negative_returns[negative_returns > 0] = 0
            ticker_data['downside_volatility'] = negative_returns.rolling(63).std()
            
            # Calculate upside volatility
            positive_returns = ticker_data['returns'].copy()
            positive_returns[positive_returns < 0] = 0
            ticker_data['upside_volatility'] = positive_returns.rolling(63).std()
            
            # Calculate volatility ratio (upside/downside)
            ticker_data['volatility_ratio'] = (
                ticker_data['upside_volatility'] / ticker_data['downside_volatility']
            ).fillna(1)
            
            volatility_data.append(ticker_data)
        
        volatility_df = pd.concat(volatility_data, ignore_index=True)
        
        print(f"   ✅ Calculated volatility factors for {len(volatility_df):,} records")
        return volatility_df
    
    def calculate_liquidity_factors(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate liquidity factors."""
        print("📊 Calculating liquidity factors...")
        
        liquidity_data = []
        
        for ticker in price_data['ticker'].unique():
            ticker_data = price_data[price_data['ticker'] == ticker].copy()
            ticker_data = ticker_data.sort_values('trading_date')
            
            # Calculate average daily volume (63-day)
            ticker_data['avg_volume_63d'] = ticker_data['volume'].rolling(63).mean()
            
            # Calculate average daily value (63-day)
            ticker_data['avg_value_63d'] = ticker_data['value'].rolling(63).mean()
            
            # Calculate volume turnover ratio
            ticker_data['volume_turnover'] = ticker_data['avg_volume_63d'] / ticker_data['market_cap']
            
            # Calculate Amihud illiquidity (absolute return / volume)
            ticker_data['amihud_illiquidity'] = (
                abs(ticker_data['returns']) / ticker_data['volume']
            ).rolling(63).mean()
            
            # Calculate liquidity score (inverse of Amihud)
            ticker_data['liquidity_score'] = 1 / (1 + ticker_data['amihud_illiquidity'])
            
            liquidity_data.append(ticker_data)
        
        liquidity_df = pd.concat(liquidity_data, ignore_index=True)
        
        print(f"   ✅ Calculated liquidity factors for {len(liquidity_df):,} records")
        return liquidity_df
    
    def combine_all_factors(self, momentum_data: pd.DataFrame, 
                           volatility_data: pd.DataFrame,
                           liquidity_data: pd.DataFrame) -> pd.DataFrame:
        """Combine all momentum and volatility factors."""
        print("📊 Combining all factors...")
        
        # Merge all factor data
        combined_data = momentum_data.merge(
            volatility_data[['ticker', 'trading_date', 'volatility_63d', 'low_vol_score', 
                           'downside_volatility', 'upside_volatility', 'volatility_ratio']], 
            on=['ticker', 'trading_date'], how='left'
        )
        
        combined_data = combined_data.merge(
            liquidity_data[['ticker', 'trading_date', 'avg_volume_63d', 'avg_value_63d',
                           'volume_turnover', 'amihud_illiquidity', 'liquidity_score']],
            on=['ticker', 'trading_date'], how='left'
        )
        
        # Calculate final factor scores
        combined_data['momentum_score'] = self._normalize_factor(combined_data['composite_momentum'])
        combined_data['low_vol_score_final'] = self._normalize_factor(combined_data['low_vol_score'])
        combined_data['liquidity_score_final'] = self._normalize_factor(combined_data['liquidity_score'])
        
        # Calculate combined momentum-volatility score
        combined_data['momentum_vol_score'] = (
            combined_data['momentum_score'] * 0.7 + 
            combined_data['low_vol_score_final'] * 0.3
        )
        
        print(f"   ✅ Combined factors for {len(combined_data):,} records")
        return combined_data
    
    def _normalize_factor(self, factor_series: pd.Series) -> pd.Series:
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
    
    def calculate_all_factors(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Calculate all momentum and volatility factors."""
        print("🚀 Calculating all momentum and volatility factors...")
        
        # Load price data
        price_data = self.load_price_data(start_date, end_date)
        
        # Calculate returns
        price_data = self.calculate_returns(price_data)
        
        # Calculate individual factor types
        momentum_data = self.calculate_momentum_factors(price_data)
        volatility_data = self.calculate_volatility_factors(price_data)
        liquidity_data = self.calculate_liquidity_factors(price_data)
        
        # Combine all factors
        combined_data = self.combine_all_factors(momentum_data, volatility_data, liquidity_data)
        
        # Filter to requested date range
        combined_data['trading_date'] = pd.to_datetime(combined_data['trading_date'])
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        final_data = combined_data[
            (combined_data['trading_date'] >= start_dt) & 
            (combined_data['trading_date'] <= end_dt)
        ]
        
        print(f"✅ All momentum and volatility factors calculated for {len(final_data)} records")
        print(f"   📊 Coverage: {final_data['ticker'].nunique()} tickers")
        print(f"   📅 Period: {final_data['trading_date'].min()} to {final_data['trading_date'].max()}")
        
        return final_data

# %% [markdown]
# # TESTING AND VALIDATION

# %%
def test_momentum_volatility_calculator():
    """Test the momentum and volatility factor calculator."""
    print("🧪 Testing Momentum and Volatility Factor Calculator...")
    
    try:
        # Create database connection
        db_manager = get_database_manager()
        engine = db_manager.get_engine()
        
        # Initialize calculator
        calculator = MomentumVolatilityCalculator(engine)
        
        # Test with a small period
        start_date = "2020-01-01"
        end_date = "2020-12-31"
        
        # Calculate factors
        factors = calculator.calculate_all_factors(start_date, end_date)
        
        # Display results
        print(f"\n📊 Test Results:")
        print(f"   - Records: {len(factors)}")
        print(f"   - Tickers: {factors['ticker'].nunique()}")
        print(f"   - Momentum range: {factors['composite_momentum'].min():.4f} to {factors['composite_momentum'].max():.4f}")
        print(f"   - Volatility range: {factors['volatility_63d'].min():.4f} to {factors['volatility_63d'].max():.4f}")
        print(f"   - Low Vol Score range: {factors['low_vol_score'].min():.4f} to {factors['low_vol_score'].max():.4f}")
        print(f"   - Liquidity Score range: {factors['liquidity_score'].min():.4f} to {factors['liquidity_score'].max():.4f}")
        
        # Show sample data
        print(f"\n📋 Sample Data:")
        sample_cols = ['ticker', 'trading_date', 'composite_momentum', 'volatility_63d', 
                      'low_vol_score', 'liquidity_score', 'momentum_vol_score']
        print(factors[sample_cols].head(10))
        
        print("✅ Momentum and volatility factor calculator test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_momentum_volatility_calculator()
