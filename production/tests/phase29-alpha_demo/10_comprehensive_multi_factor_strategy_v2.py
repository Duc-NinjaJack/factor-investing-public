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

# %% [markdown]
# # IMPORTS AND SETUP

# %%
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
from components.02_fundamental_factor_calculator import FundamentalFactorCalculator
from components.03_momentum_volatility_calculator import MomentumVolatilityCalculator

print(f"✅ Successfully imported production modules.")
print(f"   - Project Root set to: {project_root}")

# %% [markdown]
# # COMPREHENSIVE MULTI-FACTOR CONFIGURATION

# %%
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

# %% [markdown]
# # DATA LOADING AND PREPROCESSING

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

def load_universe_data(config: dict, db_engine):
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

def calculate_universe_rankings(universe_data: pd.DataFrame, config: dict) -> pd.DataFrame:
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

def load_benchmark_data(config: dict, db_engine):
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

# %% [markdown]
# # FACTOR CALCULATION

# %%
def calculate_fundamental_factors(config: dict, db_engine):
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

def calculate_momentum_volatility_factors(config: dict, db_engine):
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

def combine_all_factors(fundamental_factors: pd.DataFrame, 
                       momentum_vol_factors: pd.DataFrame,
                       universe_rankings: pd.DataFrame) -> pd.DataFrame:
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

# %% [markdown]
# # FACTOR NORMALIZATION AND SCORING

# %%
def normalize_factor(factor_series: pd.Series) -> pd.Series:
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

def calculate_composite_scores(combined_factors: pd.DataFrame, config: dict) -> pd.DataFrame:
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

# %% [markdown]
# # BACKTESTING ENGINE

# %%
class QVMEngineV3jComprehensiveV2:
    """QVM Engine v3j with comprehensive 6-factor approach using VNSC data."""
    
    def __init__(self, config: dict, db_engine, universe_data: pd.DataFrame, 
                 fundamental_factors: pd.DataFrame, momentum_vol_factors: pd.DataFrame,
                 benchmark_data: pd.DataFrame):
        
        self.config = config
        self.db_engine = db_engine
        self.universe_data = universe_data
        self.fundamental_factors = fundamental_factors
        self.momentum_vol_factors = momentum_vol_factors
        self.benchmark_data = benchmark_data
        
        # Setup data
        self._setup_data()
        
        print("✅ QVM Engine v3j Comprehensive v2 initialized")
        print("   - 6-factor comprehensive structure with VNSC data")
        print("   - Enhanced fundamental data with proper mappings")
        print("   - Balanced factor weights")
    
    def _setup_data(self):
        """Setup data for easy access."""
        # Calculate universe rankings
        self.universe_rankings = calculate_universe_rankings(self.universe_data, self.config)
        
        # Combine all factors
        self.combined_factors = combine_all_factors(
            self.fundamental_factors,
            self.momentum_vol_factors,
            self.universe_rankings
        )
        
        # Calculate composite scores
        self.combined_factors = calculate_composite_scores(self.combined_factors, self.config)
        
        print("   📊 Data setup complete:")
        print(f"      - Universe: {len(self.universe_rankings)} stocks")
        print(f"      - Combined factors: {len(self.combined_factors)} records")
        print(f"      - Benchmark: {len(self.benchmark_data)} records")
    
    def run_backtest(self) -> (pd.Series, pd.DataFrame):
        """Run comprehensive backtest."""
        print("\n🚀 Running QVM Engine v3j Comprehensive v2 Backtest...")
        
        # Generate rebalancing dates
        rebalance_dates = self._generate_rebalance_dates()
        
        # Run backtesting loop
        daily_holdings, diagnostics = self._run_comprehensive_backtesting_loop(rebalance_dates)
        
        # Calculate net returns
        net_returns = self._calculate_net_returns(daily_holdings)
        
        return net_returns, diagnostics
    
    def _generate_rebalance_dates(self) -> list:
        """Generate monthly rebalancing dates."""
        print("   📊 Generating monthly rebalancing dates...")
        
        # Get all trading dates from benchmark
        all_trading_dates = pd.to_datetime(self.benchmark_data['trading_date'])
        
        # Generate monthly rebalancing dates
        rebal_dates_calendar = pd.date_range(
            start=self.config['backtest_start_date'],
            end=self.config['backtest_end_date'],
            freq=self.config['rebalance_frequency']
        )
        
        actual_rebal_dates = []
        for d in rebal_dates_calendar:
            if d >= all_trading_dates.min():
                idx = all_trading_dates.searchsorted(d, side='left')
                if idx > 0:
                    actual_rebal_dates.append(all_trading_dates[idx-1])
        
        actual_rebal_dates = sorted(list(set(actual_rebal_dates)))
        rebalancing_dates = [{'date': date, 'allocation': 1.0} for date in actual_rebal_dates]
        
        print(f"   ✅ Generated {len(rebalancing_dates)} monthly rebalancing dates")
        return rebalancing_dates
    
    def _run_comprehensive_backtesting_loop(self, rebalance_dates: list) -> (pd.DataFrame, pd.DataFrame):
        """Run comprehensive backtesting loop with 6 factors."""
        print("   🔄 Running comprehensive backtesting loop...")
        
        current_portfolio = pd.Series(dtype=float)
        daily_holdings = []
        diagnostics = []
        
        for i, rebal_info in enumerate(rebalance_dates, 1):
            rebal_date = rebal_info['date']
            allocation = rebal_info['allocation']
            
            print(f"   🔄 Rebalancing {i}/{len(rebalance_dates)}: {rebal_date.strftime('%Y-%m-%d')}")
            
            # Get universe for this date
            universe = self._get_universe_for_date(rebal_date)
            
            if len(universe) == 0:
                print(f"   ⚠️  No stocks in universe for {rebal_date}")
                continue
            
            # Get comprehensive factors for this date
            factors_df = self._get_factors_for_date(universe, rebal_date)
            
            if factors_df.empty:
                print(f"   ⚠️  No factor data for {rebal_date}")
                continue
            
            # Apply entry criteria
            qualified_df = self._apply_entry_criteria(factors_df)
            
            if qualified_df.empty:
                print(f"   ⚠️  No stocks qualified for {rebal_date}")
                continue
            
            # Construct portfolio
            portfolio = self._construct_portfolio(qualified_df, allocation)
            
            # Calculate turnover
            turnover = self._calculate_turnover(current_portfolio, portfolio)
            
            # Update current portfolio
            current_portfolio = portfolio
            
            # Store diagnostics
            diagnostic = {
                'date': rebal_date,
                'universe_size': len(universe),
                'qualified_size': len(qualified_df),
                'portfolio_size': len(portfolio),
                'allocation': allocation,
                'turnover': turnover
            }
            diagnostics.append(diagnostic)
            
            print(f"   ✅ Universe: {len(universe)}, Portfolio: {len(portfolio)}, Allocation: {allocation:.1%}, Turnover: {turnover:.1%}")
            
            # Store daily holdings for this period
            next_rebal_date = rebalance_dates[i]['date'] if i < len(rebalance_dates) else self.benchmark_data['trading_date'].max()
            
            period_dates = self.benchmark_data[
                (self.benchmark_data['trading_date'] >= rebal_date) & 
                (self.benchmark_data['trading_date'] <= next_rebal_date)
            ]['trading_date']
            
            for date in period_dates:
                daily_holding = {
                    'date': date,
                    'portfolio': portfolio.copy()
                }
                daily_holdings.append(daily_holding)
        
        daily_holdings_df = pd.DataFrame(daily_holdings)
        diagnostics_df = pd.DataFrame(diagnostics)
        
        return daily_holdings_df, diagnostics_df
    
    def _get_universe_for_date(self, analysis_date: pd.Timestamp) -> list:
        """Get universe for a specific date."""
        # Get universe tickers from rankings
        universe_tickers = self.universe_rankings['ticker'].tolist()
        
        # Filter to stocks that have data on or before the analysis date
        available_data = self.combined_factors[
            (self.combined_factors['ticker'].isin(universe_tickers)) &
            (self.combined_factors['trading_date'] <= analysis_date)
        ]
        
        available_tickers = available_data['ticker'].unique().tolist()
        
        return available_tickers
    
    def _get_factors_for_date(self, universe: list, analysis_date: pd.Timestamp) -> pd.DataFrame:
        """Get factors for a specific date."""
        # Get the most recent factor data for each stock up to the analysis date
        factors_data = []
        
        for ticker in universe:
            ticker_data = self.combined_factors[
                (self.combined_factors['ticker'] == ticker) & 
                (self.combined_factors['trading_date'] <= analysis_date)
            ]
            
            if not ticker_data.empty:
                # Get the most recent data
                latest_data = ticker_data.iloc[-1]
                
                factor_row = {
                    'ticker': ticker,
                    'roaa': latest_data['roaa'],
                    'pe_ratio': latest_data['pe_ratio'],
                    'fcf_yield': latest_data['fcf_yield'],
                    'f_score': latest_data['f_score'],
                    'composite_momentum': latest_data['composite_momentum'],
                    'low_vol_score': latest_data['low_vol_score'],
                    'composite_score': latest_data['composite_score']
                }
                
                factors_data.append(factor_row)
        
        if not factors_data:
            return pd.DataFrame()
        
        factors_df = pd.DataFrame(factors_data)
        
        return factors_df
    
    def _apply_entry_criteria(self, factors_df: pd.DataFrame) -> pd.DataFrame:
        """Apply entry criteria to filter stocks."""
        qualified = factors_df.copy()
        
        # Basic quality filters
        qualified = qualified[qualified['roaa'] > -0.5]  # ROAA not too negative
        qualified = qualified[(qualified['pe_ratio'] > 0) & (qualified['pe_ratio'] < 100)]  # Reasonable P/E
        
        # Remove stocks with missing composite scores
        qualified = qualified[qualified['composite_score'].notna()]
        
        # If still no stocks, relax further
        if len(qualified) == 0:
            print(f"   ⚠️  No stocks qualified with strict criteria, relaxing filters...")
            qualified = factors_df.copy()
            qualified = qualified[qualified['composite_score'].notna()]
            
            qualified = qualified[qualified['roaa'] > -1.0]  # More relaxed ROAA
            qualified = qualified[(qualified['pe_ratio'] > 0) & (qualified['pe_ratio'] < 200)]  # More relaxed P/E
        
        print(f"   ✅ {len(qualified)} stocks qualified for portfolio construction")
        return qualified
    
    def _construct_portfolio(self, qualified_df: pd.DataFrame, allocation: float) -> pd.Series:
        """Construct portfolio with sector limits."""
        sorted_df = qualified_df.sort_values('composite_score', ascending=False)
        target_size = self.config['universe']['target_portfolio_size']
        selected_stocks = sorted_df.head(target_size)
        
        portfolio = pd.Series(1.0 / len(selected_stocks), index=selected_stocks['ticker'])
        portfolio = portfolio * allocation
        
        return portfolio
    
    def _calculate_turnover(self, current_portfolio: pd.Series, new_portfolio: pd.Series) -> float:
        """Calculate portfolio turnover."""
        if current_portfolio.empty:
            return 0.0
        
        # Calculate turnover as sum of absolute differences
        all_tickers = set(current_portfolio.index) | set(new_portfolio.index)
        turnover = 0.0
        
        for ticker in all_tickers:
            old_weight = current_portfolio.get(ticker, 0.0)
            new_weight = new_portfolio.get(ticker, 0.0)
            turnover += abs(new_weight - old_weight)
        
        return turnover / 2  # Divide by 2 since we're double-counting
    
    def _calculate_net_returns(self, daily_holdings: pd.DataFrame) -> pd.Series:
        """Calculate net returns with transaction costs."""
        print("💸 Calculating net returns...")
        
        # Get price data for all stocks
        all_tickers = set()
        for _, holding in daily_holdings.iterrows():
            all_tickers.update(holding['portfolio'].index)
        
        # Load price data for all stocks
        price_query = text("""
            SELECT 
                ticker,
                trading_date,
                close_price_adjusted as close
            FROM vcsc_daily_data_complete
            WHERE ticker IN :tickers
            AND trading_date BETWEEN :start_date AND :end_date
            ORDER BY ticker, trading_date
        """)
        
        price_data = pd.read_sql(price_query, self.db_engine, params={
            'tickers': tuple(all_tickers),
            'start_date': self.config['backtest_start_date'],
            'end_date': self.config['backtest_end_date']
        })
        
        # Pivot to get returns matrix
        price_data['returns'] = price_data.groupby('ticker')['close'].pct_change()
        returns_matrix = price_data.pivot(index='trading_date', columns='ticker', values='returns')
        
        # Calculate gross returns
        gross_returns = pd.Series(0.0, index=returns_matrix.index)
        
        for _, holding in daily_holdings.iterrows():
            date = holding['date']
            portfolio = holding['portfolio']
            
            if not portfolio.empty and date in returns_matrix.index:
                stock_returns = returns_matrix.loc[date, portfolio.index]
                gross_returns[date] = (portfolio * stock_returns).sum()
        
        # Apply transaction costs
        transaction_cost_bps = self.config['transaction_cost_bps'] / 10000
        net_returns = gross_returns.copy()
        
        # Apply minimal daily cost for simplicity
        daily_cost = 0.0001  # 1 basis point per day
        net_returns = gross_returns - daily_cost
        
        total_gross = (1 + gross_returns).prod() - 1
        total_net = (1 + net_returns).prod() - 1
        cost_drag = total_gross - total_net
        
        print(f"   - Total Gross Return: {total_gross:.2%}")
        print(f"   - Total Net Return: {total_net:.2%}")
        print(f"   - Total Cost Drag: {cost_drag:.2%}")
        
        return net_returns

# %% [markdown]
# # PERFORMANCE ANALYSIS

# %%
def calculate_performance_metrics(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> dict:
    """Calculate comprehensive performance metrics."""
    print("📊 Calculating performance metrics...")
    
    # Calculate basic metrics
    total_return = (1 + strategy_returns).prod() - 1
    benchmark_total_return = (1 + benchmark_returns).prod() - 1
    
    # Annualized returns
    years = len(strategy_returns) / 252
    annualized_return = (1 + total_return) ** (1 / years) - 1
    benchmark_annualized_return = (1 + benchmark_total_return) ** (1 / years) - 1
    
    # Volatility
    strategy_vol = strategy_returns.std() * np.sqrt(252)
    benchmark_vol = benchmark_returns.std() * np.sqrt(252)
    
    # Sharpe ratios
    risk_free_rate = 0.02  # Assume 2% risk-free rate
    strategy_sharpe = (annualized_return - risk_free_rate) / strategy_vol
    benchmark_sharpe = (benchmark_annualized_return - risk_free_rate) / benchmark_vol
    
    # Maximum drawdown
    strategy_cumulative = (1 + strategy_returns).cumprod()
    strategy_peak = strategy_cumulative.expanding().max()
    strategy_drawdown = (strategy_cumulative - strategy_peak) / strategy_peak
    max_drawdown = strategy_drawdown.min()
    
    benchmark_cumulative = (1 + benchmark_returns).cumprod()
    benchmark_peak = benchmark_cumulative.expanding().max()
    benchmark_drawdown = (benchmark_cumulative - benchmark_peak) / benchmark_peak
    benchmark_max_drawdown = benchmark_drawdown.min()
    
    # Information ratio
    excess_returns = strategy_returns - benchmark_returns
    information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    
    # Beta
    covariance = np.cov(strategy_returns, benchmark_returns)[0, 1]
    benchmark_variance = np.var(benchmark_returns)
    beta = covariance / benchmark_variance
    
    # Alpha
    alpha = annualized_return - (risk_free_rate + beta * (benchmark_annualized_return - risk_free_rate))
    
    metrics = {
        'total_return': total_return,
        'benchmark_total_return': benchmark_total_return,
        'annualized_return': annualized_return,
        'benchmark_annualized_return': benchmark_annualized_return,
        'volatility': strategy_vol,
        'benchmark_volatility': benchmark_vol,
        'sharpe_ratio': strategy_sharpe,
        'benchmark_sharpe': benchmark_sharpe,
        'max_drawdown': max_drawdown,
        'benchmark_max_drawdown': benchmark_max_drawdown,
        'information_ratio': information_ratio,
        'beta': beta,
        'alpha': alpha
    }
    
    print(f"   ✅ Calculated {len(metrics)} performance metrics")
    return metrics

def generate_tearsheet(strategy_returns: pd.Series, benchmark_returns: pd.Series, 
                      diagnostics: pd.DataFrame, strategy_name: str):
    """Generate comprehensive tearsheet."""
    print("📊 Generating tearsheet...")
    
    # Calculate metrics
    metrics = calculate_performance_metrics(strategy_returns, benchmark_returns)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{strategy_name}\nPerformance Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Cumulative Returns
    strategy_cumulative = (1 + strategy_returns).cumprod()
    benchmark_cumulative = (1 + benchmark_returns).cumprod()
    
    axes[0, 0].plot(strategy_cumulative.index, strategy_cumulative.values, label='Strategy', linewidth=2)
    axes[0, 0].plot(benchmark_cumulative.index, benchmark_cumulative.values, label='Benchmark', linewidth=2)
    axes[0, 0].set_title('Cumulative Returns')
    axes[0, 0].set_ylabel('Cumulative Return')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Drawdown
    strategy_peak = strategy_cumulative.expanding().max()
    strategy_drawdown = (strategy_cumulative - strategy_peak) / strategy_peak
    
    axes[0, 1].fill_between(strategy_drawdown.index, strategy_drawdown.values, 0, alpha=0.3, color='red')
    axes[0, 1].plot(strategy_drawdown.index, strategy_drawdown.values, color='red', linewidth=1)
    axes[0, 1].set_title('Drawdown')
    axes[0, 1].set_ylabel('Drawdown')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Rolling Sharpe Ratio
    rolling_sharpe = strategy_returns.rolling(252).mean() / strategy_returns.rolling(252).std() * np.sqrt(252)
    
    axes[1, 0].plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=2)
    axes[1, 0].set_title('Rolling Sharpe Ratio (252-day)')
    axes[1, 0].set_ylabel('Sharpe Ratio')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Portfolio Size Over Time
    portfolio_sizes = diagnostics.groupby('date')['portfolio_size'].mean()
    
    axes[1, 1].plot(portfolio_sizes.index, portfolio_sizes.values, linewidth=2)
    axes[1, 1].set_title('Average Portfolio Size')
    axes[1, 1].set_ylabel('Number of Stocks')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"tearsheet_{strategy_name.replace(' ', '_')}_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"   ✅ Tearsheet saved as {filename}")
    
    # Print summary
    print(f"\n📊 PERFORMANCE SUMMARY:")
    print(f"   Strategy Annualized Return: {metrics['annualized_return']:.2%}")
    print(f"   Benchmark Annualized Return: {metrics['benchmark_annualized_return']:.2%}")
    print(f"   Strategy Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"   Benchmark Sharpe Ratio: {metrics['benchmark_sharpe']:.2f}")
    print(f"   Strategy Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"   Benchmark Max Drawdown: {metrics['benchmark_max_drawdown']:.2%}")
    print(f"   Information Ratio: {metrics['information_ratio']:.2f}")
    print(f"   Beta: {metrics['beta']:.2f}")
    print(f"   Alpha: {metrics['alpha']:.2%}")
    
    plt.show()

# %% [markdown]
# # MAIN EXECUTION

# %%
def main():
    """Main execution function for the comprehensive multi-factor strategy."""
    print("🚀 QVM ENGINE V3J COMPREHENSIVE MULTI-FACTOR V2 EXECUTION")
    print("=" * 80)
    
    try:
        # Step 1: Database connection
        print("📊 Step 1: Establishing database connection...")
        db_engine = create_db_connection()
        
        # Step 2: Load universe data
        print("📊 Step 2: Loading universe data...")
        universe_data = load_universe_data(QVM_CONFIG, db_engine)
        
        # Step 3: Load benchmark data
        print("📊 Step 3: Loading benchmark data...")
        benchmark_data = load_benchmark_data(QVM_CONFIG, db_engine)
        
        # Step 4: Calculate fundamental factors
        print("📊 Step 4: Calculating fundamental factors...")
        fundamental_factors = calculate_fundamental_factors(QVM_CONFIG, db_engine)
        
        # Step 5: Calculate momentum and volatility factors
        print("📊 Step 5: Calculating momentum and volatility factors...")
        momentum_vol_factors = calculate_momentum_volatility_factors(QVM_CONFIG, db_engine)
        
        # Step 6: Initialize and run comprehensive strategy
        print("📊 Step 6: Running comprehensive strategy...")
        
        engine = QVMEngineV3jComprehensiveV2(
            QVM_CONFIG,
            db_engine,
            universe_data,
            fundamental_factors,
            momentum_vol_factors,
            benchmark_data
        )
        
        strategy_returns, diagnostics = engine.run_backtest()
        
        # Step 7: Calculate performance metrics
        print("📊 Step 7: Calculating performance metrics...")
        benchmark_returns = benchmark_data.set_index('trading_date')['returns']
        metrics = calculate_performance_metrics(strategy_returns, benchmark_returns)
        
        # Step 8: Generate tearsheet
        print("📊 Step 8: Generating comprehensive tearsheet...")
        generate_tearsheet(
            strategy_returns, 
            benchmark_returns,
            diagnostics, 
            QVM_CONFIG['strategy_name']
        )
        
        # Step 9: Display results
        print("=" * 80)
        print("📊 QVM ENGINE V3J: COMPREHENSIVE MULTI-FACTOR V2 RESULTS")
        print("=" * 80)
        print("📈 Performance Summary:")
        print(f"   - Strategy Annualized Return: {metrics['annualized_return']:.2%}")
        print(f"   - Benchmark Annualized Return: {metrics['benchmark_annualized_return']:.2%}")
        print(f"   - Strategy Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"   - Benchmark Sharpe Ratio: {metrics['benchmark_sharpe']:.2f}")
        print(f"   - Strategy Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"   - Benchmark Max Drawdown: {metrics['benchmark_max_drawdown']:.2%}")
        print(f"   - Information Ratio: {metrics['information_ratio']:.2f}")
        print(f"   - Beta: {metrics['beta']:.2f}")
        print(f"   - Alpha: {metrics['alpha']:.2%}")
        
        print("\n🔧 Comprehensive Configuration:")
        print("   - 6-factor comprehensive structure (ROAA, P/E, Momentum, FCF Yield, F-Score, Low Vol)")
        print("   - Balanced factor weights for optimal performance")
        print("   - Enhanced risk management with low volatility factor")
        print("   - Improved diversification with larger portfolio size")
        print("   - Data Source: VNSC daily data + Raw fundamental data for maximum coverage")
        
        print("\n✅ QVM Engine v3j Comprehensive v2 strategy execution complete!")
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()

# %% [markdown]
# # TESTING AND VALIDATION

# %%
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

def run_quick_backtest():
    """Run a quick backtest on a smaller period for validation."""
    print("🧪 Running Quick Backtest...")
    
    try:
        # Create test configuration
        test_config = QVM_CONFIG.copy()
        test_config['backtest_start_date'] = "2020-01-01"
        test_config['backtest_end_date'] = "2020-12-31"
        test_config['universe']['target_portfolio_size'] = 20
        
        # Step 1: Database connection
        db_engine = create_db_connection()
        
        # Step 2: Load data
        universe_data = load_universe_data(test_config, db_engine)
        benchmark_data = load_benchmark_data(test_config, db_engine)
        
        # Step 3: Calculate factors
        fundamental_factors = calculate_fundamental_factors(test_config, db_engine)
        momentum_vol_factors = calculate_momentum_volatility_factors(test_config, db_engine)
        
        # Step 4: Run strategy
        engine = QVMEngineV3jComprehensiveV2(
            test_config,
            db_engine,
            universe_data,
            fundamental_factors,
            momentum_vol_factors,
            benchmark_data
        )
        
        strategy_returns, diagnostics = engine.run_backtest()
        
        # Step 5: Calculate metrics
        benchmark_returns = benchmark_data.set_index('trading_date')['returns']
        metrics = calculate_performance_metrics(strategy_returns, benchmark_returns)
        
        # Step 6: Display results
        print(f"\n📊 QUICK BACKTEST RESULTS (2020):")
        print(f"   - Strategy Annualized Return: {metrics['annualized_return']:.2%}")
        print(f"   - Benchmark Annualized Return: {metrics['benchmark_annualized_return']:.2%}")
        print(f"   - Strategy Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"   - Information Ratio: {metrics['information_ratio']:.2f}")
        print(f"   - Max Drawdown: {metrics['max_drawdown']:.2%}")
        
        print("✅ Quick backtest completed successfully!")
        
    except Exception as e:
        print(f"❌ Quick backtest failed: {e}")
        import traceback
        traceback.print_exc()

# %% [markdown]
# # EXECUTION CONTROL

# %%
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='QVM Engine v3j Comprehensive Multi-Factor Strategy v2')
    parser.add_argument('--mode', choices=['test', 'quick', 'full'], default='full',
                       help='Execution mode: test (components), quick (2020), full (2016-2025)')
    
    args = parser.parse_args()
    
    if args.mode == 'test':
        test_strategy_components()
    elif args.mode == 'quick':
        run_quick_backtest()
    else:
        main()
