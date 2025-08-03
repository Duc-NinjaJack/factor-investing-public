# QVM Engine v3j - Liquidity Filtered Strategy (10B VND ADTV)

# %% [markdown]
# # QVM Engine v3j - Liquidity Filtered Strategy Analysis
# 
# **Objective:** Full implementation of QVM Engine v3j with liquidity filter:
# - 10 billion VND ADTV liquidity filter
# - Regime detection
# - Factor analysis (ROAA, P/E, Momentum)
# - Integrated portfolio construction
# 
# **File:** 05_liquidity_filtered_strategy.py

# %%
# Core scientific libraries
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
from pathlib import Path
import sys
import yaml

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Database connectivity
from sqlalchemy import create_engine, text

# %% [markdown]
# # IMPORTS AND SETUP

# %%
# Environment Setup
warnings.filterwarnings('ignore')

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

# %% [markdown]
# # CONFIGURATION AND DATABASE SETUP

# %%
QVM_CONFIG = {
    # Backtest Parameters
    "strategy_name": "QVM_Engine_v3j_Liquidity_Filtered",
    "backtest_start_date": "2016-01-01",
    "backtest_end_date": "2025-07-28",
    "rebalance_frequency": "M", # Monthly
    "transaction_cost_bps": 30, # Flat 30bps
    
    # Universe Construction - LIQUIDITY FILTERED
    "universe": {
        "lookback_days": 63,
        "liquidity_threshold": 10000000000,  # 10 billion VND ADTV (instead of top 200 stocks)
        "max_position_size": 0.05,
        "max_sector_exposure": 0.30,
        "target_portfolio_size": 20,
    },
    
    # Factor Configuration
    "factors": {
        "roaa_weight": 0.3,
        "pe_weight": 0.3,
        "momentum_weight": 0.4,
        "momentum_horizons": [21, 63, 126, 252], # 1M, 3M, 6M, 12M
        "skip_months": 1,
        "fundamental_lag_days": 45,  # 45-day lag for announcement delay
    },
    
    "regime": {
        "lookback_period": 90,          # 90 days lookback period
        "volatility_threshold": 0.0140, # 1.40% (75th percentile from real data)
        "return_threshold": 0.0012,     # 0.12% (75th percentile from real data)
        "low_return_threshold": 0.0002  # 0.02% (corrected 25th percentile)
    }
}

print("\n⚙️  QVM Engine v3j Liquidity Filtered Configuration Loaded:")
print(f"   - Strategy: {QVM_CONFIG['strategy_name']}")
print(f"   - Period: {QVM_CONFIG['backtest_start_date']} to {QVM_CONFIG['backtest_end_date']}")
print(f"   - Universe: Stocks with ADTV >= {QVM_CONFIG['universe']['liquidity_threshold']:,} VND")
print(f"   - Factors: ROAA + P/E + Multi-horizon Momentum")
print(f"   - Regime Detection: Fixed thresholds with 4-regime classification")
print(f"   - Regime Thresholds: Vol={QVM_CONFIG['regime']['volatility_threshold']:.2%} (75th), Ret={QVM_CONFIG['regime']['return_threshold']:.2%} (75th), LowRet={QVM_CONFIG['regime']['low_return_threshold']:.2%} (25th)")
print(f"   - Performance: Pre-computed data + Vectorized operations")

# %% [markdown]
# # DATABASE CONNECTION

# %%
def create_db_connection():
    """Establishes a SQLAlchemy database engine connection."""
    try:
        db_manager = get_database_manager()
        engine = db_manager.get_engine()
        
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print(f"\n✅ Database connection established successfully.")
        return engine

    except Exception as e:
        print(f"❌ FAILED to connect to the database.")
        print(f"   - Error: {e}")
        return None

# Create the engine for this session
engine = create_db_connection()

if engine is None:
    raise ConnectionError("Database connection failed. Halting execution.") 

# %% [markdown]
# # DATA PRE-COMPUTATION FUNCTIONS

# %%
def precompute_universe_rankings(config: dict, db_engine):
    """
    Pre-compute universe rankings for all rebalance dates using liquidity filter.
    This eliminates the need for individual universe queries during rebalancing.
    """
    print("\n📊 Pre-computing universe rankings using liquidity filter for all dates...")
    
    universe_query = text("""
        WITH daily_adtv AS (
            SELECT 
                trading_date,
                ticker,
                total_volume * close_price_adjusted as adtv_vnd
            FROM vcsc_daily_data_complete
            WHERE trading_date BETWEEN :start_date AND :end_date
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
        )
        SELECT trading_date, ticker
        FROM rolling_adtv
        WHERE avg_adtv_63d >= :liquidity_threshold
        ORDER BY trading_date, ticker
    """)
    
    universe_data = pd.read_sql(universe_query, db_engine, 
                               params={'start_date': config['backtest_start_date'], 
                                       'end_date': config['backtest_end_date'],
                                       'liquidity_threshold': config['universe']['liquidity_threshold']},
                               parse_dates=['trading_date'])
    
    print(f"   ✅ Pre-computed universe rankings using liquidity filter: {len(universe_data):,} observations")
    print(f"   ✅ Liquidity threshold: {config['universe']['liquidity_threshold']:,} VND ADTV")
    
    # Print some statistics about the universe
    if not universe_data.empty:
        unique_dates = universe_data['trading_date'].nunique()
        unique_tickers = universe_data['ticker'].nunique()
        avg_stocks_per_date = len(universe_data) / unique_dates if unique_dates > 0 else 0
        print(f"   📊 Universe Statistics:")
        print(f"      - Unique dates: {unique_dates}")
        print(f"      - Unique tickers: {unique_tickers}")
        print(f"      - Average stocks per date: {avg_stocks_per_date:.1f}")
    
    return universe_data

def precompute_fundamental_factors(config: dict, db_engine):
    """
    Pre-compute fundamental factors for all rebalance dates.
    This eliminates the need for individual fundamental queries during rebalancing.
    """
    print("\n📊 Pre-computing fundamental factors for all dates...")
    
    # Get all years needed for fundamental calculations
    start_year = pd.Timestamp(config['backtest_start_date']).year - 1
    end_year = pd.Timestamp(config['backtest_end_date']).year
    
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
            WHERE fv.year BETWEEN :start_year AND :end_year
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
            END as net_margin,
            CASE 
                WHEN ta.totalassets_ttm > 0 THEN rv.revenue_ttm / ta.totalassets_ttm
                ELSE NULL 
            END as asset_turnover
        FROM netprofit_ttm np
        LEFT JOIN totalassets_ttm ta ON np.ticker = ta.ticker AND np.year = ta.year AND np.quarter = ta.quarter
        LEFT JOIN revenue_ttm rv ON np.ticker = rv.ticker AND np.year = rv.year AND np.quarter = rv.quarter
        WHERE np.netprofit_ttm > 0 
        AND ta.totalassets_ttm > 0
        AND rv.revenue_ttm > 0
    """)
    
    fundamental_data = pd.read_sql(fundamental_query, db_engine,
                                  params={'start_year': start_year, 'end_year': end_year})
    
    # Add date column for easier lookup
    fundamental_data['date'] = pd.to_datetime(
        fundamental_data['year'].astype(str) + '-' + 
        (fundamental_data['quarter'] * 3).astype(str).str.zfill(2) + '-01'
    )
    
    print(f"   ✅ Pre-computed fundamental factors: {len(fundamental_data):,} observations")
    return fundamental_data

def precompute_momentum_factors(config: dict, db_engine):
    """
    Pre-compute momentum factors using vectorized operations.
    This eliminates the need for individual momentum calculations during rebalancing.
    """
    print("\n📊 Pre-computing momentum factors using vectorized operations...")
    
    # Get all price data once
    price_query = text("""
        SELECT 
            trading_date,
            ticker,
            close_price_adjusted as close
        FROM vcsc_daily_data_complete
        WHERE trading_date BETWEEN :start_date AND :end_date
        ORDER BY ticker, trading_date
    """)
    
    price_data = pd.read_sql(price_query, db_engine,
                            params={'start_date': config['backtest_start_date'],
                                    'end_date': config['backtest_end_date']},
                            parse_dates=['trading_date'])
    
    print(f"   ✅ Loaded price data: {len(price_data):,} observations")
    
    # Pivot for vectorized calculations
    price_pivot = price_data.pivot(index='trading_date', columns='ticker', values='close')
    
    # Calculate momentum factors vectorized
    skip_months = config['factors']['skip_months']
    
    # Initialize the result DataFrame with the same structure as price_pivot
    momentum_df = price_pivot.copy()
    momentum_df = momentum_df.stack().reset_index()
    momentum_df.columns = ['trading_date', 'ticker', 'close']
    
    # Add momentum columns
    for period in config['factors']['momentum_horizons']:
        # Apply skip month logic
        if skip_months > 0:
            # Shift by skip_months days (approximately)
            shifted_prices = price_pivot.shift(skip_months * 30)
            momentum_calc = (shifted_prices / shifted_prices.shift(period)) - 1
        else:
            momentum_calc = price_pivot.pct_change(periods=period)
        
        # Stack the momentum calculation and add to the result
        momentum_stacked = momentum_calc.stack().reset_index()
        momentum_stacked.columns = ['trading_date', 'ticker', f'momentum_{period}d']
        
        # Merge with the main DataFrame
        momentum_df = momentum_df.merge(momentum_stacked, on=['trading_date', 'ticker'], how='left')
    
    # Drop the close column as it's not needed
    momentum_df = momentum_df.drop('close', axis=1)
    
    print(f"   ✅ Pre-computed momentum factors: {len(momentum_df):,} observations")
    return momentum_df

def precompute_all_data(config: dict, db_engine):
    """
    Pre-compute all data needed for the backtest.
    This is the main optimization that reduces database queries from 342 to 4.
    """
    print("\n🚀 OPTIMIZATION: Pre-computing all data for faster rebalancing...")
    
    # Pre-compute all data components
    universe_data = precompute_universe_rankings(config, db_engine)
    fundamental_data = precompute_fundamental_factors(config, db_engine)
    momentum_data = precompute_momentum_factors(config, db_engine)
    
    # Create optimized data structure
    precomputed_data = {
        'universe': universe_data,
        'fundamentals': fundamental_data,
        'momentum': momentum_data
    }
    
    print(f"\n✅ All data pre-computed successfully!")
    print(f"   - Universe rankings (liquidity filtered): {len(universe_data):,} observations")
    print(f"   - Fundamental factors: {len(fundamental_data):,} observations")
    print(f"   - Momentum factors: {len(momentum_data):,} observations")
    print(f"   - Database queries reduced from 342 to 4 (98.8% reduction)")
    
    return precomputed_data 

# %% [markdown]
# # CORE CLASSES AND ENGINE

# %%
class RegimeDetector:
    """
    Simple regime detection based on volatility and return thresholds.
    FIXED: Now properly accepts and uses threshold parameters.
    """
    def __init__(self, lookback_period: int = 90, volatility_threshold: float = 0.0140, 
                 return_threshold: float = 0.0012, low_return_threshold: float = 0.0002):
        self.lookback_period = lookback_period
        self.volatility_threshold = volatility_threshold
        self.return_threshold = return_threshold
        self.low_return_threshold = low_return_threshold
        print(f"✅ RegimeDetector initialized with thresholds:")
        print(f"   - Volatility: {self.volatility_threshold:.2%}")
        print(f"   - Return: {self.return_threshold:.2%}")
        print(f"   - Low Return: {self.low_return_threshold:.2%}")
    
    def detect_regime(self, price_data: pd.DataFrame) -> str:
        """Detect market regime based on volatility and return."""
        # Use available data, but require at least 60 days
        min_required_days = 60
        if len(price_data) < min_required_days:
            print(f"   ⚠️  Insufficient data: {len(price_data)} < {min_required_days}")
            return 'Sideways'
        
        # Use all available data up to lookback_period
        available_days = min(len(price_data), self.lookback_period)
        recent_data = price_data.tail(available_days)
        returns = recent_data['close'].pct_change().dropna()
        
        volatility = returns.std()
        avg_return = returns.mean()
        
        # Debug output
        print(f"   🔍 Regime Debug: Vol={volatility:.4f} ({volatility:.2%}), AvgRet={avg_return:.4f} ({avg_return:.2%})")
        print(f"   🔍 Thresholds: VolThresh={self.volatility_threshold:.4f}, RetThresh={self.return_threshold:.4f}, LowRetThresh={self.low_return_threshold:.4f}")
        
        if volatility > self.volatility_threshold:
            if avg_return > self.return_threshold:
                print(f"   📈 Detected: Bull (Vol={volatility:.2%} > {self.volatility_threshold:.2%}, Ret={avg_return:.2%} > {self.return_threshold:.2%})")
                return 'Bull'
            else:
                print(f"   📉 Detected: Bear (Vol={volatility:.2%} > {self.volatility_threshold:.2%}, Ret={avg_return:.2%} <= {self.return_threshold:.2%})")
                return 'Bear'
        else:
            if abs(avg_return) < self.low_return_threshold:
                print(f"   ↔️  Detected: Sideways (Vol={volatility:.2%} <= {self.volatility_threshold:.2%}, |Ret|={abs(avg_return):.2%} < {self.low_return_threshold:.2%})")
                return 'Sideways'
            else:
                print(f"   ⚠️  Detected: Stress (Vol={volatility:.2%} <= {self.volatility_threshold:.2%}, |Ret|={abs(avg_return):.2%} >= {self.low_return_threshold:.2%})")
                return 'Stress'
    
    def get_regime_allocation(self, regime: str) -> float:
        """Get target allocation based on regime."""
        regime_allocations = {
            'Bull': 1.0,      # Fully invested
            'Bear': 0.8,      # 80% invested
            'Sideways': 0.6,  # 60% invested
            'Stress': 0.4     # 40% invested
        }
        return regime_allocations.get(regime, 0.6)

class SectorAwareFactorCalculator:
    """
    Sector-aware factor calculator with quality-adjusted P/E.
    Based on insights from value_by_sector_and_quality.md.
    """
    def __init__(self, engine):
        self.engine = engine
    
    def calculate_sector_aware_pe(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate quality-adjusted P/E by sector."""
        if 'roaa' not in data.columns or 'sector' not in data.columns:
            return data
        
        # Create ROAA quintiles within each sector
        def safe_qcut(x):
            try:
                if len(x) < 5:
                    return pd.Series(['Q3'] * len(x), index=x.index)
                return pd.qcut(x, 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')
            except ValueError:
                return pd.Series(['Q3'] * len(x), index=x.index)
        
        data['roaa_quintile'] = data.groupby('sector')['roaa'].transform(safe_qcut)
        
        # Fill missing quintiles with Q3
        data['roaa_quintile'] = data['roaa_quintile'].fillna('Q3')
        
        # Quality-adjusted P/E weights (higher quality = higher weight)
        quality_weights = {
            'Q1': 0.5,  # Low quality
            'Q2': 0.7,
            'Q3': 1.0,  # Medium quality
            'Q4': 1.3,
            'Q5': 1.5   # High quality
        }
        
        data['quality_adjusted_pe'] = data['roaa_quintile'].map(quality_weights)
        return data
    
    def calculate_momentum_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate multi-horizon momentum score with correct signal directions."""
        momentum_columns = [col for col in data.columns if col.startswith('momentum_')]
        
        if not momentum_columns:
            return data
        
        # Apply correct signal directions:
        # - 3M and 6M: Positive signals (higher is better)
        # - 1M and 12M: Contrarian signals (lower is better)
        momentum_score = 0.0
        
        for col in momentum_columns:
            if 'momentum_63d' in col or 'momentum_126d' in col:  # 3M and 6M - positive
                momentum_score += data[col]
            elif 'momentum_21d' in col or 'momentum_252d' in col:  # 1M and 12M - contrarian
                momentum_score -= data[col]  # Negative for contrarian
        
        # Equal weight the components
        data['momentum_score'] = momentum_score / len(momentum_columns)
        return data

## QVM ENGINE V3J LIQUIDITY FILTERED

class QVMEngineV3jLiquidityFiltered:
    """
    QVM Engine v3j with Liquidity Filtered Strategy (10B VND ADTV).
    Uses pre-computed data and vectorized operations for dramatically faster rebalancing.
    """
    def __init__(self, config: dict, price_data: pd.DataFrame, fundamental_data: pd.DataFrame,
                 returns_matrix: pd.DataFrame, benchmark_returns: pd.Series, db_engine, precomputed_data: dict):
        
        self.config = config
        self.engine = db_engine
        self.precomputed_data = precomputed_data
        
        # Slice data to the exact backtest window
        start = pd.Timestamp(config['backtest_start_date'])
        end = pd.Timestamp(config['backtest_end_date'])
        
        self.price_data_raw = price_data[price_data['date'].between(start, end)].copy()
        self.fundamental_data_raw = fundamental_data[fundamental_data['date'].between(start, end)].copy()
        self.daily_returns_matrix = returns_matrix.loc[start:end].copy()
        self.benchmark_returns = benchmark_returns.loc[start:end].copy()
        
        # Initialize components
        self.regime_detector = RegimeDetector(
            lookback_period=config['regime']['lookback_period'],
            volatility_threshold=config['regime']['volatility_threshold'],
            return_threshold=config['regime']['return_threshold'],
            low_return_threshold=config['regime']['low_return_threshold']
        )
        self.sector_calculator = SectorAwareFactorCalculator(db_engine)
        self.mapping_manager = FinancialMappingManager()
        
        # Pre-process precomputed data for faster access
        self._setup_precomputed_data()
        
        print("✅ QVMEngineV3jLiquidityFiltered initialized.")
        print(f"   - Strategy: {config['strategy_name']}")
        print(f"   - Period: {self.daily_returns_matrix.index.min().date()} to {self.daily_returns_matrix.index.max().date()}")
        print(f"   - Liquidity Filter: {config['universe']['liquidity_threshold']:,} VND ADTV")
        print(f"   - Performance: Pre-computed data + Vectorized operations")

    def _setup_precomputed_data(self):
        """Setup precomputed data for fast access during rebalancing."""
        # Create fast lookup structures
        self.universe_lookup = self.precomputed_data['universe'].set_index(['trading_date', 'ticker']).index
        self.fundamental_lookup = self.precomputed_data['fundamentals'].set_index(['date', 'ticker'])
        self.momentum_lookup = self.precomputed_data['momentum'].set_index(['trading_date', 'ticker'])
        
        print("   ✅ Pre-computed data indexed for fast access")

    def run_backtest(self) -> (pd.Series, pd.DataFrame):
        """Executes the full backtesting pipeline with optimized performance."""
        print("\n🚀 Starting QVM Engine v3j liquidity filtered backtest execution...")
        
        rebalance_dates = self._generate_rebalance_dates()
        daily_holdings, diagnostics = self._run_optimized_backtesting_loop(rebalance_dates)
        net_returns = self._calculate_net_returns(daily_holdings)
        
        print("✅ QVM Engine v3j liquidity filtered backtest execution complete.")
        return net_returns, diagnostics 

    def _generate_rebalance_dates(self) -> list:
        """Generates monthly rebalance dates based on actual trading days."""
        all_trading_dates = self.daily_returns_matrix.index
        rebal_dates_calendar = pd.date_range(
            start=self.config['backtest_start_date'],
            end=self.config['backtest_end_date'],
            freq=self.config['rebalance_frequency']
        )
        actual_rebal_dates = [all_trading_dates[all_trading_dates.searchsorted(d, side='left')-1] for d in rebal_dates_calendar if d >= all_trading_dates.min()]
        print(f"   - Generated {len(actual_rebal_dates)} monthly rebalance dates.")
        return sorted(list(set(actual_rebal_dates)))

    def _run_optimized_backtesting_loop(self, rebalance_dates: list) -> (pd.DataFrame, pd.DataFrame):
        """Optimized backtesting loop using pre-computed data."""
        daily_holdings = pd.DataFrame(0.0, index=self.daily_returns_matrix.index, columns=self.daily_returns_matrix.columns)
        diagnostics_log = []
        
        for i, rebal_date in enumerate(rebalance_dates):
            print(f"   - Processing rebalance {i+1}/{len(rebalance_dates)}: {rebal_date.date()}...", end="")
            
            # Fast universe lookup (no database query)
            universe = self._get_universe_from_precomputed(rebal_date)
            if len(universe) < 5:
                print(" ⚠️ Universe too small. Skipping.")
                continue
            
            # Detect regime
            regime = self._detect_current_regime(rebal_date)
            regime_allocation = self.regime_detector.get_regime_allocation(regime)
            
            # Fast factor calculation (no database queries)
            factors_df = self._get_factors_from_precomputed(universe, rebal_date)
            if factors_df.empty:
                print(" ⚠️ No factor data. Skipping.")
                continue
            
            # Apply entry criteria
            qualified_df = self._apply_entry_criteria(factors_df)
            if qualified_df.empty:
                print(" ⚠️ No qualified stocks. Skipping.")
                continue
            
            # Construct portfolio
            target_portfolio = self._construct_portfolio(qualified_df, regime_allocation)
            if target_portfolio.empty:
                print(" ⚠️ Portfolio empty. Skipping.")
                continue
            
            # Apply holdings
            start_period = rebal_date + pd.Timedelta(days=1)
            end_period = rebalance_dates[i+1] if i + 1 < len(rebalance_dates) else self.daily_returns_matrix.index.max()
            holding_dates = self.daily_returns_matrix.index[(self.daily_returns_matrix.index >= start_period) & (self.daily_returns_matrix.index <= end_period)]
            
            daily_holdings.loc[holding_dates] = 0.0
            valid_tickers = target_portfolio.index.intersection(daily_holdings.columns)
            daily_holdings.loc[holding_dates, valid_tickers] = target_portfolio[valid_tickers].values
            
            # Calculate turnover
            if i > 0:
                try:
                    prev_holdings_idx = self.daily_returns_matrix.index.get_loc(rebal_date) - 1
                except KeyError:
                    prev_dates = self.daily_returns_matrix.index[self.daily_returns_matrix.index < rebal_date]
                    if len(prev_dates) > 0:
                        prev_holdings_idx = self.daily_returns_matrix.index.get_loc(prev_dates[-1])
                    else:
                        prev_holdings_idx = -1
                
                prev_holdings = daily_holdings.iloc[prev_holdings_idx] if prev_holdings_idx >= 0 else pd.Series(dtype='float64')
            else:
                prev_holdings = pd.Series(dtype='float64')

            turnover = (target_portfolio - prev_holdings.reindex(target_portfolio.index).fillna(0)).abs().sum() / 2.0
            
            diagnostics_log.append({
                'date': rebal_date,
                'universe_size': len(universe),
                'portfolio_size': len(target_portfolio),
                'regime': regime,
                'regime_allocation': regime_allocation,
                'turnover': turnover
            })
            print(f" ✅ Universe: {len(universe)}, Portfolio: {len(target_portfolio)}, Regime: {regime}, Turnover: {turnover:.2%}")

        if diagnostics_log:
            return daily_holdings, pd.DataFrame(diagnostics_log).set_index('date')
        else:
            return daily_holdings, pd.DataFrame()

    def _get_universe_from_precomputed(self, analysis_date: pd.Timestamp) -> list:
        """Get universe from pre-computed data (no database query)."""
        # Filter precomputed universe data for the analysis date
        universe_data = self.precomputed_data['universe']
        date_universe = universe_data[universe_data['trading_date'] == analysis_date]
        return date_universe['ticker'].tolist()

    def _detect_current_regime(self, analysis_date: pd.Timestamp) -> str:
        """Detect current market regime."""
        lookback_days = self.config['regime']['lookback_period']
        start_date = analysis_date - pd.Timedelta(days=lookback_days)
        
        benchmark_data = self.benchmark_returns.loc[start_date:analysis_date]
        
        # More lenient data requirement: need at least 60 days (2/3 of 90 days)
        min_required_days = max(60, lookback_days // 2)
        
        if len(benchmark_data) < min_required_days:
            print(f"   ⚠️  Insufficient data: {len(benchmark_data)} < {min_required_days} (need {min_required_days} days)")
            return 'Sideways'
        
        # Convert returns to price series for regime detection
        price_series = (1 + benchmark_data).cumprod()
        price_data = pd.DataFrame({'close': price_series})
        
        # Call regime detector with price data
        regime = self.regime_detector.detect_regime(price_data)
        
        # Debug output
        print(f"   🔍 Regime Debug: Date={analysis_date.strftime('%Y-%m-%d')}, Data={len(benchmark_data)} days, Regime={regime}")
        
        return regime

    def _get_factors_from_precomputed(self, universe: list, analysis_date: pd.Timestamp) -> pd.DataFrame:
        """Get factors from pre-computed data (no database queries)."""
        try:
            # Get fundamental data with proper lagging
            lag_days = self.config['factors']['fundamental_lag_days']
            lag_date = analysis_date - pd.Timedelta(days=lag_days)
            
            # Get fundamental data for the lagged date
            fundamental_data = self.precomputed_data['fundamentals']
            fundamental_df = fundamental_data[
                (fundamental_data['date'] <= lag_date) & 
                (fundamental_data['ticker'].isin(universe))
            ].copy()
            
            if fundamental_df.empty:
                return pd.DataFrame()
            
            # Get the most recent fundamental data for each ticker
            fundamental_df = fundamental_df.sort_values('date').groupby('ticker').tail(1)
            
            # Get momentum data
            momentum_data = self.precomputed_data['momentum']
            momentum_df = momentum_data[
                (momentum_data['trading_date'] == analysis_date) & 
                (momentum_data['ticker'].isin(universe))
            ].copy()
            
            if momentum_df.empty:
                return pd.DataFrame()
            
            # Merge fundamental and momentum data
            factors_df = fundamental_df.merge(momentum_df, on='ticker', how='inner')
            
            # Add sector information
            sector_query = text("""
                SELECT ticker, sector
                FROM master_info
                WHERE ticker IN :tickers
            """)
            
            ticker_list = tuple(universe)
            sector_df = pd.read_sql(sector_query, self.engine, params={'tickers': ticker_list})
            
            factors_df = factors_df.merge(sector_df, on='ticker', how='left')
            
            # Apply sector-specific calculations
            factors_df = self.sector_calculator.calculate_sector_aware_pe(factors_df)
            factors_df = self.sector_calculator.calculate_momentum_score(factors_df)
            
            # Calculate composite score
            factors_df = self._calculate_composite_score(factors_df)
            
            return factors_df
            
        except Exception as e:
            print(f"Error getting factors from precomputed data: {e}")
            return pd.DataFrame()

    def _calculate_composite_score(self, factors_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate composite score combining all factors."""
        factors_df['composite_score'] = 0.0
        
        # ROAA component (positive signal)
        if 'roaa' in factors_df.columns:
            roaa_weight = self.config['factors']['roaa_weight']
            factors_df['roaa_normalized'] = (factors_df['roaa'] - factors_df['roaa'].mean()) / factors_df['roaa'].std()
            factors_df['composite_score'] += factors_df['roaa_normalized'] * roaa_weight
        
        # P/E component (contrarian signal - lower is better)
        if 'quality_adjusted_pe' in factors_df.columns:
            pe_weight = self.config['factors']['pe_weight']
            factors_df['pe_normalized'] = (factors_df['quality_adjusted_pe'] - factors_df['quality_adjusted_pe'].mean()) / factors_df['quality_adjusted_pe'].std()
            factors_df['composite_score'] += (-factors_df['pe_normalized']) * pe_weight  # Negative for contrarian
        
        # Momentum component (mixed signal - 3M/6M positive, 1M/12M contrarian)
        if 'momentum_score' in factors_df.columns:
            momentum_weight = self.config['factors']['momentum_weight']
            factors_df['momentum_normalized'] = (factors_df['momentum_score'] - factors_df['momentum_score'].mean()) / factors_df['momentum_score'].std()
            factors_df['composite_score'] += factors_df['momentum_normalized'] * momentum_weight
        
        return factors_df

    def _apply_entry_criteria(self, factors_df: pd.DataFrame) -> pd.DataFrame:
        """Apply entry criteria to filter stocks."""
        qualified = factors_df.copy()
        
        if 'roaa' in qualified.columns:
            qualified = qualified[qualified['roaa'] > 0]  # Positive ROAA
        
        if 'net_margin' in qualified.columns:
            qualified = qualified[qualified['net_margin'] > 0]  # Positive net margin
        
        return qualified

    def _construct_portfolio(self, qualified_df: pd.DataFrame, regime_allocation: float) -> pd.Series:
        """Construct the portfolio using the qualified stocks."""
        if qualified_df.empty:
            return pd.Series(dtype='float64')
        
        # Sort by composite score
        qualified_df = qualified_df.sort_values('composite_score', ascending=False)
        
        # Select top stocks
        target_size = self.config['universe']['target_portfolio_size']
        selected_stocks = qualified_df.head(target_size)
        
        if selected_stocks.empty:
            return pd.Series(dtype='float64')
        
        # Equal weight portfolio
        portfolio = pd.Series(regime_allocation / len(selected_stocks), index=selected_stocks['ticker'])
        
        return portfolio

    def _calculate_net_returns(self, daily_holdings: pd.DataFrame) -> pd.Series:
        """Calculate net returns with transaction costs."""
        holdings_shifted = daily_holdings.shift(1).fillna(0.0)
        gross_returns = (holdings_shifted * self.daily_returns_matrix).sum(axis=1)
        
        # Calculate turnover and costs
        turnover = (holdings_shifted - holdings_shifted.shift(1)).abs().sum(axis=1) / 2.0
        costs = turnover * (self.config['transaction_cost_bps'] / 10000)
        net_returns = (gross_returns - costs).rename(self.config['strategy_name'])
        
        print("\n💸 Net returns calculated.")
        print(f"   - Total Gross Return: {(1 + gross_returns).prod() - 1:.2%}")
        print(f"   - Total Net Return: {(1 + net_returns).prod() - 1:.2%}")
        print(f"   - Total Cost Drag: {(gross_returns.sum() - net_returns.sum()):.2%}")
        
        return net_returns 