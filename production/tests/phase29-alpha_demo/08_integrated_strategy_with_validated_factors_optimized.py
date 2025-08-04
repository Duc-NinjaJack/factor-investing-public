# QVM Engine v3j - Optimized Strategy (Simplified Factor Structure)

# %% [markdown]
# # QVM Engine v3j - Optimized Strategy Analysis
# 
# **Objective:** Optimized implementation based on factor integration investigation:
# - Simplified 3-factor structure (ROAA, P/E, Momentum)
# - Removed redundant factors (F-Score, FCF Yield, Low-Volatility)
# - Improved factor integration methodology
# - Focus on portfolio-level performance, not just factor-level IC
# 
# **Key Changes:**
# - Simplified factor structure to reduce complexity
# - Removed correlated factors to improve diversification
# - Optimized factor weights based on investigation findings
# - Improved data quality handling

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

# OPTIMIZED CONFIGURATION
QVM_CONFIG = {
    # Backtest Parameters
    "strategy_name": "QVM_Engine_v3j_Beta_Optimized",
    "backtest_start_date": "2016-01-01",
    "backtest_end_date": "2025-07-28",
    "rebalance_frequency": "M", # Monthly
    "transaction_cost_bps": 30, # Flat 30bps
    
    # Universe Construction
    "universe": {
        "lookback_days": 63,
        "top_n_stocks": 35,  # Increased from 20 for better diversification
        "max_position_size": 0.04,  # Reduced from 0.05 for lower concentration
        "max_sector_exposure": 0.25,  # NEW: Sector limit for diversification
        "target_portfolio_size": 30,  # Increased from 20 for lower beta
    },
    
    # Beta-Optimized Factor Configuration (Reduced Momentum Weight)
    "factors": {
        "roaa_weight": 0.50,      # Increased from 0.35 (quality focus)
        "pe_weight": 0.30,        # Increased from 0.25 (value focus)
        "momentum_weight": 0.20,  # Reduced from 0.40 (lower beta)
        "momentum_horizons": [21, 63, 126, 252], # 1M, 3M, 6M, 12M
        "skip_months": 1,
        "fundamental_lag_days": 45,  # 45-day lag for announcement delay
    }
}

print("\n⚙️  QVM Engine v3j Optimized Configuration Loaded:")
print(f"   - Strategy: {QVM_CONFIG['strategy_name']}")
print(f"   - Period: {QVM_CONFIG['backtest_start_date']} to {QVM_CONFIG['backtest_end_date']}")
print(f"   - Universe: Top {QVM_CONFIG['universe']['top_n_stocks']} stocks by ADTV")
print(f"   - Rebalancing: {QVM_CONFIG['rebalance_frequency']} frequency")
print(f"   - ROAA (Quality): {QVM_CONFIG['factors']['roaa_weight']:.1%}")
print(f"   - P/E (Value): {QVM_CONFIG['factors']['pe_weight']:.1%}")
print(f"   - Momentum: {QVM_CONFIG['factors']['momentum_weight']:.1%}")
print(f"   - Performance: Pre-computed data + Vectorized operations")

# DATABASE CONNECTION
def create_db_connection():
    """Create database connection using production configuration."""
    try:
        db_manager = get_database_manager()
        engine = db_manager.get_engine()
        print("✅ Database connection established successfully.")
        return engine
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        raise

# DATA PREPROCESSING FUNCTIONS
def precompute_universe_rankings(config: dict, db_engine):
    """Precompute universe rankings for all dates."""
    print("📊 Precomputing universe rankings...")
    
    start_date = pd.Timestamp(config['backtest_start_date']) - pd.DateOffset(days=config['universe']['lookback_days'] + 30)
    end_date = config['backtest_end_date']
    
    query = text("""
        SELECT 
            trading_date,
            ticker,
            total_volume,
            close_price_adjusted as close,
            total_volume * close_price_adjusted as adtv_vnd,
            market_cap
        FROM vcsc_daily_data_complete
        WHERE trading_date BETWEEN :start_date AND :end_date
        AND total_volume > 0
        ORDER BY trading_date, adtv_vnd DESC
    """)
    
    universe_data = pd.read_sql(query, db_engine, params={'start_date': start_date, 'end_date': end_date})
    
    # Calculate rolling ADTV and apply liquidity filter
    universe_rankings = []
    for date in universe_data['trading_date'].unique():
        date_data = universe_data[universe_data['trading_date'] == date]
        
        # Calculate rolling ADTV (63-day lookback)
        lookback_days = config['universe']['lookback_days']
        lookback_start = date - pd.Timedelta(days=lookback_days)
        lookback_data = universe_data[
            (universe_data['trading_date'] >= lookback_start) & 
            (universe_data['trading_date'] <= date)
        ]
        
        # Calculate average ADTV for each ticker
        avg_adtv = lookback_data.groupby('ticker')['adtv_vnd'].mean().reset_index()
        
        # Apply liquidity filter: ADTV > 10 billion VND
        liquidity_threshold = 10_000_000_000  # 10 billion VND
        liquid_stocks = avg_adtv[avg_adtv['adtv_vnd'] >= liquidity_threshold]
        
        if len(liquid_stocks) > 0:
            liquid_stocks = liquid_stocks.sort_values('adtv_vnd', ascending=False)
            
            # Select top N stocks from liquid universe
            top_n = config['universe']['top_n_stocks']
            top_stocks = liquid_stocks.head(top_n)
            top_stocks['trading_date'] = date
            
            universe_rankings.append(top_stocks)
    
    universe_df = pd.concat(universe_rankings, ignore_index=True)
    
    print(f"   ✅ Universe rankings computed: {len(universe_df)} records")
    print(f"   📊 Applied liquidity filter: ADTV > 10 billion VND")
    return universe_df

def precompute_fundamental_factors(config: dict, db_engine):
    """Precompute fundamental factors (ROAA, P/E) for all dates."""
    print("📊 Precomputing fundamental factors...")
    
    start_date = pd.Timestamp(config['backtest_start_date']) - pd.DateOffset(days=config['factors']['fundamental_lag_days'] + 30)
    end_date = config['backtest_end_date']
    
    # Load fundamental data
    fundamental_query = text("""
        WITH netprofit_ttm AS (
            SELECT 
                fv.ticker,
                fv.year,
                fv.quarter,
                SUM(fv.value / 1e9) as netprofit_ttm
            FROM fundamental_values fv
            WHERE fv.item_id = 1
            AND fv.statement_type = 'PL'
            AND fv.year >= :start_year
            GROUP BY fv.ticker, fv.year, fv.quarter
        ),
        totalassets_ttm AS (
            SELECT 
                fv.ticker,
                fv.year,
                fv.quarter,
                SUM(fv.value / 1e9) as totalassets_ttm
            FROM fundamental_values fv
            WHERE fv.item_id = 2
            AND fv.statement_type = 'BS'
            AND fv.year >= :start_year
            GROUP BY fv.ticker, fv.year, fv.quarter
        )
        SELECT 
            np.ticker,
            np.year,
            np.quarter,
            np.netprofit_ttm,
            ta.totalassets_ttm,
            CASE 
                WHEN ta.totalassets_ttm > 0 THEN np.netprofit_ttm / ta.totalassets_ttm 
                ELSE NULL 
            END as roaa
        FROM netprofit_ttm np
        LEFT JOIN totalassets_ttm ta ON np.ticker = ta.ticker AND np.year = ta.year AND np.quarter = ta.quarter
        WHERE np.netprofit_ttm > 0 
        AND ta.totalassets_ttm > 0
        ORDER BY np.ticker, np.year, np.quarter
    """)
    
    fundamental_data = pd.read_sql(fundamental_query, db_engine, params={'start_year': start_date.year})
    
    # Calculate ROAA
    fundamental_data['roaa'] = fundamental_data['netprofit_ttm'] / fundamental_data['totalassets_ttm']
    
    # Calculate P/E ratio using market cap and net profit
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
        # Calculate P/E ratio
        pe_data['pe'] = pe_data['market_cap_bn'] / pe_data['netprofit_ttm']
        
        # Add date column for merging
        pe_data['date'] = pd.to_datetime(
            pe_data['year'].astype(str) + '-' + 
            (pe_data['quarter'] * 3).astype(str).str.zfill(2) + '-01'
        )
        
        # Add date column to fundamental data for merging
        fundamental_data['date'] = pd.to_datetime(
            fundamental_data['year'].astype(str) + '-' + 
            (fundamental_data['quarter'] * 3).astype(str).str.zfill(2) + '-01'
        )
        
        # Merge P/E data with fundamental data
        fundamental_data = fundamental_data.merge(
            pe_data[['ticker', 'date', 'pe']], 
            on=['ticker', 'date'], 
            how='left'
        )
    else:
        fundamental_data['pe'] = np.nan
    
    # Clean up extreme values
    fundamental_data['roaa'] = fundamental_data['roaa'].clip(-1, 1)  # ROAA between -100% and 100%
    fundamental_data['pe'] = fundamental_data['pe'].clip(0, 100)  # P/E between 0 and 100
    
    print(f"   ✅ Fundamental factors computed: {len(fundamental_data)} records")
    return fundamental_data

def precompute_momentum_factors(config: dict, db_engine):
    """Precompute momentum factors for all dates."""
    print("📊 Precomputing momentum factors...")
    
    start_date = pd.Timestamp(config['backtest_start_date']) - pd.DateOffset(days=max(config['factors']['momentum_horizons']) + 30)
    end_date = config['backtest_end_date']
    
    query = text("""
        SELECT 
            trading_date,
            ticker,
            close_price_adjusted as close
        FROM vcsc_daily_data_complete
        WHERE trading_date BETWEEN :start_date AND :end_date
        ORDER BY ticker, trading_date
    """)
    
    price_data = pd.read_sql(query, db_engine, params={'start_date': start_date, 'end_date': end_date})
    
    # Calculate returns for each horizon
    momentum_data = []
    for ticker in price_data['ticker'].unique():
        ticker_data = price_data[price_data['ticker'] == ticker].sort_values('trading_date')
        ticker_data['returns'] = ticker_data['close'].pct_change()
        
        for horizon in config['factors']['momentum_horizons']:
            ticker_data[f'momentum_{horizon}'] = ticker_data['returns'].rolling(horizon).mean()
        
        momentum_data.append(ticker_data)
    
    momentum_df = pd.concat(momentum_data, ignore_index=True)
    
    # Calculate composite momentum score
    momentum_columns = [f'momentum_{h}' for h in config['factors']['momentum_horizons']]
    momentum_df['momentum_score'] = momentum_df[momentum_columns].mean(axis=1)
    
    print(f"   ✅ Momentum factors computed: {len(momentum_df)} records")
    return momentum_df

def precompute_all_data(config: dict, db_engine):
    """Precompute all data for the backtest."""
    print("🚀 Precomputing all data for optimized strategy...")
    
    universe_rankings = precompute_universe_rankings(config, db_engine)
    fundamental_factors = precompute_fundamental_factors(config, db_engine)
    momentum_factors = precompute_momentum_factors(config, db_engine)
    
    precomputed_data = {
        'universe': universe_rankings,
        'fundamentals': fundamental_factors,
        'momentum': momentum_factors
    }
    
    print("✅ All data precomputed successfully!")
    return precomputed_data

# OPTIMIZED FACTOR CALCULATOR
class OptimizedFactorCalculator:
    """Optimized factor calculator with simplified 3-factor structure."""
    
    def __init__(self, engine):
        self.engine = engine
    
    def calculate_sector_aware_pe(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate sector-aware P/E ratios."""
        if 'pe' not in data.columns:
            return data
        
        # Simple P/E normalization (no sector adjustment for simplicity)
        pe_data = data['pe'].dropna()
        if len(pe_data) > 1 and pe_data.std() > 0:
            data['pe_normalized'] = (data['pe'] - pe_data.mean()) / pe_data.std()
        else:
            data['pe_normalized'] = 0
        
        return data
    
    def calculate_momentum_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum score from precomputed data."""
        if 'momentum_score' not in data.columns:
            return data
        
        # Simple momentum normalization
        momentum_data = data['momentum_score'].dropna()
        if len(momentum_data) > 1 and momentum_data.std() > 0:
            data['momentum_normalized'] = (data['momentum_score'] - momentum_data.mean()) / momentum_data.std()
        else:
            data['momentum_normalized'] = 0
        
        return data

# OPTIMIZED QVM ENGINE
class QVMEngineV3jOptimized:
    """Optimized QVM Engine with simplified factor structure."""
    
    def __init__(self, config: dict, price_data: pd.DataFrame, fundamental_data: pd.DataFrame,
                 returns_matrix: pd.DataFrame, benchmark_returns: pd.Series, db_engine, precomputed_data: dict):
        
        self.config = config
        self.price_data = price_data
        self.fundamental_data = fundamental_data
        self.daily_returns_matrix = returns_matrix
        self.benchmark_returns = benchmark_returns
        self.db_engine = db_engine
        self.precomputed_data = precomputed_data
        
        # Initialize factor calculator
        self.factor_calculator = OptimizedFactorCalculator(db_engine)
        
        # Setup precomputed data
        self._setup_precomputed_data()
        
        print(f"✅ QVM Engine v3j Optimized initialized")
        print(f"   - Simplified 3-factor structure")
        print(f"   - Optimized factor weights")
        print(f"   - Improved data quality handling")
    
    def _setup_precomputed_data(self):
        """Setup precomputed data for the engine."""
        self.universe_rankings = self.precomputed_data.get('universe', pd.DataFrame())
        self.fundamental_factors = self.precomputed_data.get('fundamentals', pd.DataFrame())
        self.momentum_factors = self.precomputed_data.get('momentum', pd.DataFrame())
        
        print(f"   📊 Precomputed data loaded:")
        print(f"      - Universe: {len(self.universe_rankings)} records")
        print(f"      - Fundamentals: {len(self.fundamental_factors)} records")
        print(f"      - Momentum: {len(self.momentum_factors)} records")
    
    def run_backtest(self) -> (pd.Series, pd.DataFrame):
        """Run the optimized backtest."""
        print(f"\n🚀 Running QVM Engine v3j Optimized Backtest...")
        
        # Generate rebalancing dates
        rebalance_dates = self._generate_rebalance_dates()
        
        # Run backtesting loop
        returns, diagnostics = self._run_optimized_backtesting_loop(rebalance_dates)
        
        return returns, diagnostics
    
    def _generate_rebalance_dates(self) -> list:
        """Generate monthly rebalancing dates using the working strategy approach."""
        print("   📊 Generating monthly rebalancing dates...")
        
        # Use the same approach as the working strategy
        all_trading_dates = self.daily_returns_matrix.index
        rebal_dates_calendar = pd.date_range(
            start=self.config['backtest_start_date'],
            end=self.config['backtest_end_date'],
            freq=self.config['rebalance_frequency']
        )
        
        actual_rebal_dates = []
        for d in rebal_dates_calendar:
            if d >= all_trading_dates.min():
                # Find the closest trading date before or on the calendar date
                idx = all_trading_dates.searchsorted(d, side='left')
                if idx > 0:
                    actual_rebal_dates.append(all_trading_dates[idx-1])
        
        # Remove duplicates and sort
        actual_rebal_dates = sorted(list(set(actual_rebal_dates)))
        
        # Convert to the format expected by the backtesting loop
        rebalancing_dates = [{'date': date, 'allocation': 1.0} for date in actual_rebal_dates]
        
        print(f"   ✅ Generated {len(rebalancing_dates)} monthly rebalancing dates")
        return rebalancing_dates
    
    def _run_optimized_backtesting_loop(self, rebalance_dates: list) -> (pd.DataFrame, pd.DataFrame):
        """Run the optimized backtesting loop."""
        print("   🔄 Running optimized backtesting loop...")
        
        # Initialize tracking variables
        daily_holdings = pd.DataFrame(0.0, index=self.daily_returns_matrix.index, columns=self.daily_returns_matrix.columns)
        diagnostics_log = []
        
        for i, rebal_info in enumerate(rebalance_dates):
            rebal_date = rebal_info['date']
            allocation = rebal_info['allocation']
            
            print(f"   🔄 Rebalancing {i+1}/{len(rebalance_dates)}: {rebal_date.strftime('%Y-%m-%d')}")
            
            # Get universe
            universe = self._get_universe_from_precomputed(rebal_date)
            if not universe:
                print(f"   ⚠️  No universe found for {rebal_date}")
                continue
            
            # Get factors
            factors_df = self._get_factors_from_precomputed(universe, rebal_date)
            if factors_df.empty:
                print(f"   ⚠️  No factors found for {rebal_date}")
                continue
            
            # Calculate composite score
            factors_df = self._calculate_optimized_composite_score(factors_df)
            
            # Apply entry criteria
            qualified_df = self._apply_entry_criteria(factors_df)
            if qualified_df.empty:
                print(f"   ⚠️  No qualified stocks for {rebal_date}")
                continue
            
            # Construct portfolio
            portfolio = self._construct_portfolio(qualified_df, allocation)
            if portfolio.empty:
                print(f"   ⚠️  No portfolio constructed for {rebal_date}")
                continue
            
            # Update holdings
            daily_holdings.loc[rebal_date:, portfolio.index] = portfolio.values
            
            # Log diagnostics
            turnover = self._calculate_turnover(daily_holdings, rebal_date)
            diagnostics_log.append({
                'date': rebal_date,
                'universe_size': len(universe),
                'portfolio_size': len(portfolio),
                'allocation': allocation,
                'turnover': turnover
            })
            
            print(f"   ✅ Universe: {len(universe)}, Portfolio: {len(portfolio)}, Allocation: {allocation:.1%}, Turnover: {turnover:.1%}")
        
        # Calculate net returns
        net_returns = self._calculate_net_returns(daily_holdings)
        diagnostics_df = pd.DataFrame(diagnostics_log)
        
        return net_returns, diagnostics_df
    
    def _get_universe_from_precomputed(self, analysis_date: pd.Timestamp) -> list:
        """Get universe from precomputed data."""
        # Convert trading_date column to datetime if needed
        if self.universe_rankings['trading_date'].dtype == 'object':
            self.universe_rankings['trading_date'] = pd.to_datetime(self.universe_rankings['trading_date'])
        
        # Debug: Check available dates
        if len(self.universe_rankings) == 0:
            print(f"   ⚠️  No universe rankings data available")
            return []
        
        # Find exact match first
        universe_data = self.universe_rankings[self.universe_rankings['trading_date'] == analysis_date]
        
        # If no exact match, find closest date
        if len(universe_data) == 0:
            available_dates = self.universe_rankings['trading_date'].unique()
            if len(available_dates) > 0:
                closest_date = min(available_dates, key=lambda x: abs(x - analysis_date))
                print(f"   ⚠️  Date {analysis_date.date()} not found, using closest date: {closest_date.date()}")
                universe_data = self.universe_rankings[self.universe_rankings['trading_date'] == closest_date]
        
        return universe_data['ticker'].tolist()
    
    def _get_factors_from_precomputed(self, universe: list, analysis_date: pd.Timestamp) -> pd.DataFrame:
        """Get factors from precomputed data."""
        # Get fundamental data
        fundamental_data = self.fundamental_factors[
            (self.fundamental_factors['ticker'].isin(universe)) &
            (self.fundamental_factors['year'] <= analysis_date.year)
        ].copy()
        
        # Get momentum data
        # Convert trading_date to datetime if it's not already
        if self.momentum_factors['trading_date'].dtype == 'object':
            self.momentum_factors['trading_date'] = pd.to_datetime(self.momentum_factors['trading_date'])
        
        momentum_data = self.momentum_factors[
            (self.momentum_factors['ticker'].isin(universe)) &
            (self.momentum_factors['trading_date'] <= analysis_date)
        ].copy()
        
        # Merge data
        factors_df = pd.DataFrame({'ticker': universe})
        
        # Add fundamental factors
        if not fundamental_data.empty:
            latest_fundamentals = fundamental_data.groupby('ticker').last().reset_index()
            factors_df = factors_df.merge(latest_fundamentals[['ticker', 'roaa', 'pe']], on='ticker', how='left')
        
        # Add momentum factors
        if not momentum_data.empty:
            latest_momentum = momentum_data.groupby('ticker').last().reset_index()
            factors_df = factors_df.merge(latest_momentum[['ticker', 'momentum_score']], on='ticker', how='left')
        
        return factors_df
    
    def _calculate_optimized_composite_score(self, factors_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate optimized composite score using simplified 3-factor structure."""
        factors_df['composite_score'] = 0.0
        
        # ROAA component (quality factor - positive signal)
        if 'roaa' in factors_df.columns and not factors_df['roaa'].isna().all():
            roaa_weight = self.config['factors']['roaa_weight']
            roaa_data = factors_df['roaa'].dropna()
            if len(roaa_data) > 1 and roaa_data.std() > 0:
                factors_df['roaa_normalized'] = (factors_df['roaa'] - roaa_data.mean()) / roaa_data.std()
                factors_df['composite_score'] += factors_df['roaa_normalized'].fillna(0) * roaa_weight
                print(f"   ✅ ROAA factor calculated")
            else:
                print(f"   ⚠️  Insufficient ROAA data")
        else:
            print(f"   ⚠️  No ROAA data available")
        
        # P/E component (value factor - contrarian signal)
        if 'pe' in factors_df.columns and not factors_df['pe'].isna().all():
            pe_weight = self.config['factors']['pe_weight']
            pe_data = factors_df['pe'].dropna()
            if len(pe_data) > 1 and pe_data.std() > 0:
                factors_df['pe_normalized'] = (factors_df['pe'] - pe_data.mean()) / pe_data.std()
                factors_df['composite_score'] += (-factors_df['pe_normalized'].fillna(0)) * pe_weight  # Negative for contrarian
                print(f"   ✅ P/E factor calculated")
            else:
                print(f"   ⚠️  Insufficient P/E data")
        else:
            print(f"   ⚠️  No P/E data available")
        
        # Momentum component (momentum factor - positive signal)
        if 'momentum_score' in factors_df.columns and not factors_df['momentum_score'].isna().all():
            momentum_weight = self.config['factors']['momentum_weight']
            momentum_data = factors_df['momentum_score'].dropna()
            if len(momentum_data) > 1 and momentum_data.std() > 0:
                factors_df['momentum_normalized'] = (factors_df['momentum_score'] - momentum_data.mean()) / momentum_data.std()
                factors_df['composite_score'] += factors_df['momentum_normalized'].fillna(0) * momentum_weight
                print(f"   ✅ Momentum factor calculated")
            else:
                print(f"   ⚠️  Insufficient momentum data")
        else:
            print(f"   ⚠️  No momentum data available")
        
        print(f"   ✅ Composite scores calculated for {len(factors_df)} stocks")
        return factors_df
    
    def _apply_entry_criteria(self, factors_df: pd.DataFrame) -> pd.DataFrame:
        """Apply entry criteria to filter stocks (relaxed version)."""
        qualified = factors_df.copy()
        
        # Basic quality filters (relaxed)
        if 'roaa' in qualified.columns:
            # Only filter out extremely negative ROAA, allow slightly negative
            qualified = qualified[qualified['roaa'] > -0.5]  # Allow ROAA > -50%
        
        # Remove extreme P/E values (relaxed)
        if 'pe' in qualified.columns:
            qualified = qualified[(qualified['pe'] > 0) & (qualified['pe'] < 100)]  # Allow P/E up to 100
        
        # Remove stocks with missing composite scores
        qualified = qualified[qualified['composite_score'].notna()]
        
        # If still no stocks, relax further
        if len(qualified) == 0:
            print(f"   ⚠️  No stocks qualified with strict criteria, relaxing filters...")
            qualified = factors_df.copy()
            qualified = qualified[qualified['composite_score'].notna()]
            
            # Only filter out extreme outliers
            if 'roaa' in qualified.columns:
                qualified = qualified[qualified['roaa'] > -1.0]  # Allow any ROAA > -100%
            if 'pe' in qualified.columns:
                qualified = qualified[(qualified['pe'] > 0) & (qualified['pe'] < 200)]  # Allow P/E up to 200
        
        # If still no stocks, accept all stocks with composite scores
        if len(qualified) == 0:
            print(f"   ⚠️  Still no stocks qualified, accepting all stocks with composite scores...")
            qualified = factors_df[factors_df['composite_score'].notna()].copy()
            
            # Debug: Show data distribution
            print(f"   🔍 Debug - Total stocks with composite scores: {len(qualified)}")
            if 'roaa' in qualified.columns:
                print(f"   🔍 Debug - ROAA range: {qualified['roaa'].min():.3f} to {qualified['roaa'].max():.3f}")
            if 'pe' in qualified.columns:
                print(f"   🔍 Debug - P/E range: {qualified['pe'].min():.1f} to {qualified['pe'].max():.1f}")
            print(f"   🔍 Debug - Composite score range: {qualified['composite_score'].min():.3f} to {qualified['composite_score'].max():.3f}")
        
        print(f"   ✅ {len(qualified)} stocks qualified for portfolio construction")
        return qualified
    
    def _construct_portfolio(self, qualified_df: pd.DataFrame, allocation: float) -> pd.Series:
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
        portfolio = pd.Series(allocation / len(selected_stocks), index=selected_stocks['ticker'])
        
        return portfolio
    
    def _calculate_turnover(self, daily_holdings: pd.DataFrame, rebal_date: pd.Timestamp) -> float:
        """Calculate portfolio turnover."""
        if rebal_date == daily_holdings.index[0]:
            return 0.0
        
        prev_date = daily_holdings.index[daily_holdings.index.get_loc(rebal_date) - 1]
        current_holdings = daily_holdings.loc[rebal_date]
        prev_holdings = daily_holdings.loc[prev_date]
        
        turnover = (current_holdings - prev_holdings).abs().sum() / 2.0
        return turnover
    
    def _calculate_net_returns(self, daily_holdings: pd.DataFrame) -> pd.Series:
        """Calculate net returns with transaction costs."""
        holdings_shifted = daily_holdings.shift(1).fillna(0.0)
        gross_returns = (holdings_shifted * self.daily_returns_matrix).sum(axis=1)
        
        # Calculate turnover and costs
        turnover = (holdings_shifted - holdings_shifted.shift(1)).abs().sum(axis=1) / 2.0
        costs = turnover * (self.config['transaction_cost_bps'] / 10000)
        net_returns = (gross_returns - costs).rename(self.config['strategy_name'])
        
        print(f"\n💸 Net returns calculated.")
        print(f"   - Total Gross Return: {(1 + gross_returns).prod() - 1:.2%}")
        print(f"   - Total Net Return: {(1 + net_returns).prod() - 1:.2%}")
        print(f"   - Total Cost Drag: {(gross_returns.sum() - net_returns.sum()):.2%}")
        
        return net_returns

# DATA LOADING AND MAIN EXECUTION
def load_all_data_for_backtest(config: dict, db_engine):
    """Load all necessary data for the backtest."""
    start_date = config['backtest_start_date']
    end_date = config['backtest_end_date']
    
    # Add buffer for rolling calculations
    buffer_start_date = pd.Timestamp(start_date) - pd.DateOffset(months=6)
    
    print(f"📂 Loading all data for period: {buffer_start_date.date()} to {end_date}...")

    # 1. Price and Volume Data
    print("   - Loading price and volume data...")
    price_query = text("""
        SELECT 
            trading_date as date,
            ticker,
            close_price_adjusted as close,
            total_volume as volume,
            market_cap
        FROM vcsc_daily_data_complete
        WHERE trading_date BETWEEN :start_date AND :end_date
        ORDER BY trading_date, ticker
    """)
    
    price_data = pd.read_sql(price_query, db_engine, params={'start_date': buffer_start_date, 'end_date': end_date})
    price_data['date'] = pd.to_datetime(price_data['date'])
    
    # 2. Benchmark Data (VNINDEX)
    print("   - Loading benchmark data...")
    benchmark_query = text("""
        SELECT 
            date,
            close
        FROM etf_history
        WHERE ticker = 'VNINDEX'
        AND date BETWEEN :start_date AND :end_date
        ORDER BY date
    """)
    
    benchmark_data = pd.read_sql(benchmark_query, db_engine, params={'start_date': start_date, 'end_date': end_date})
    benchmark_data['date'] = pd.to_datetime(benchmark_data['date'])
    
    # 3. Fundamental Data
    print("   - Loading fundamental data...")
    fundamental_query = text("""
        WITH netprofit_ttm AS (
            SELECT 
                fv.ticker,
                fv.year,
                fv.quarter,
                SUM(fv.value / 1e9) as netprofit_ttm
            FROM fundamental_values fv
            WHERE fv.item_id = 1
            AND fv.statement_type = 'PL'
            AND fv.year BETWEEN YEAR(:start_date) AND YEAR(:end_date)
            GROUP BY fv.ticker, fv.year, fv.quarter
        ),
        totalassets_ttm AS (
            SELECT 
                fv.ticker,
                fv.year,
                fv.quarter,
                SUM(fv.value / 1e9) as totalassets_ttm
            FROM fundamental_values fv
            WHERE fv.item_id = 2
            AND fv.statement_type = 'BS'
            AND fv.year BETWEEN YEAR(:start_date) AND YEAR(:end_date)
            GROUP BY fv.ticker, fv.year, fv.quarter
        )
        SELECT 
            np.ticker,
            np.year,
            np.quarter,
            np.netprofit_ttm,
            ta.totalassets_ttm,
            CASE 
                WHEN ta.totalassets_ttm > 0 THEN np.netprofit_ttm / ta.totalassets_ttm 
                ELSE NULL 
            END as roaa
        FROM netprofit_ttm np
        LEFT JOIN totalassets_ttm ta ON np.ticker = ta.ticker AND np.year = ta.year AND np.quarter = ta.quarter
        WHERE np.netprofit_ttm > 0 
        AND ta.totalassets_ttm > 0
        ORDER BY np.ticker, np.year, np.quarter
    """)
    
    fundamental_data = pd.read_sql(fundamental_query, db_engine, params={'start_date': buffer_start_date, 'end_date': end_date})
    
    # Process data
    print("   - Processing data...")
    
    # Create returns matrix
    price_pivot = price_data.pivot(index='date', columns='ticker', values='close')
    returns_matrix = price_pivot.pct_change().dropna()
    
    # Create benchmark returns
    benchmark_returns = benchmark_data.set_index('date')['close'].pct_change().dropna()
    
    # Align dates
    common_dates = returns_matrix.index.intersection(benchmark_returns.index)
    returns_matrix = returns_matrix.loc[common_dates]
    benchmark_returns = benchmark_returns.loc[common_dates]
    
    print(f"   ✅ Data loaded successfully:")
    print(f"      - Price data: {len(price_data)} records")
    print(f"      - Returns matrix: {returns_matrix.shape}")
    print(f"      - Benchmark returns: {len(benchmark_returns)} records")
    print(f"      - Common dates: {len(common_dates)}")
    
    return price_data, fundamental_data, returns_matrix, benchmark_returns

# PERFORMANCE METRICS AND TEARSHEET
def calculate_performance_metrics(returns: pd.Series, benchmark: pd.Series, periods_per_year: int = 252) -> dict:
    """Calculate comprehensive performance metrics."""
    
    # Align returns and benchmark
    aligned_data = pd.concat([returns, benchmark], axis=1).dropna()
    strategy_returns = aligned_data.iloc[:, 0]
    benchmark_returns = aligned_data.iloc[:, 1]
    
    # Basic metrics
    total_return = (1 + strategy_returns).prod() - 1
    benchmark_total_return = (1 + benchmark_returns).prod() - 1
    
    # Annualized metrics
    years = len(strategy_returns) / periods_per_year
    annualized_return = (1 + total_return) ** (1 / years) - 1
    benchmark_annualized_return = (1 + benchmark_total_return) ** (1 / years) - 1
    
    # Risk metrics
    volatility = strategy_returns.std() * np.sqrt(periods_per_year)
    benchmark_volatility = benchmark_returns.std() * np.sqrt(periods_per_year)
    
    # Sharpe ratio
    risk_free_rate = 0.02  # Assume 2% risk-free rate
    sharpe_ratio = (annualized_return - risk_free_rate) / volatility
    benchmark_sharpe = (benchmark_annualized_return - risk_free_rate) / benchmark_volatility
    
    # Maximum drawdown
    cumulative_returns = (1 + strategy_returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    benchmark_cumulative = (1 + benchmark_returns).cumprod()
    benchmark_running_max = benchmark_cumulative.expanding().max()
    benchmark_drawdown = (benchmark_cumulative - benchmark_running_max) / benchmark_running_max
    benchmark_max_drawdown = benchmark_drawdown.min()
    
    # Information ratio
    excess_returns = strategy_returns - benchmark_returns
    information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(periods_per_year)
    
    # Beta
    covariance = np.cov(strategy_returns, benchmark_returns)[0, 1]
    benchmark_variance = np.var(benchmark_returns)
    beta = covariance / benchmark_variance
    
    # Calmar ratio
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio,
        'information_ratio': information_ratio,
        'beta': beta,
        'benchmark_total_return': benchmark_total_return,
        'benchmark_annualized_return': benchmark_annualized_return,
        'benchmark_volatility': benchmark_volatility,
        'benchmark_sharpe': benchmark_sharpe,
        'benchmark_max_drawdown': benchmark_max_drawdown
    }

def generate_comprehensive_tearsheet(strategy_returns: pd.Series, benchmark_returns: pd.Series, diagnostics: pd.DataFrame, title: str):
    """Generate comprehensive tearsheet with optimized strategy results."""
    
    # Calculate performance metrics
    metrics = calculate_performance_metrics(strategy_returns, benchmark_returns)
    
    # Set up plotting
    plt.switch_backend('Agg')  # Use non-interactive backend
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
    
    # 1. Equity Curve
    ax1 = fig.add_subplot(gs[0, :])
    cumulative_strategy = (1 + strategy_returns).cumprod()
    cumulative_benchmark = (1 + benchmark_returns).cumprod()
    
    ax1.plot(cumulative_strategy.index, cumulative_strategy.values, label='Optimized Strategy', linewidth=2, color='#2E86AB')
    ax1.plot(cumulative_benchmark.index, cumulative_benchmark.values, label='VNINDEX Benchmark', linewidth=2, color='#A23B72', alpha=0.7)
    ax1.set_title('Optimized Strategy vs Benchmark Performance', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Cumulative Return')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Drawdown Analysis
    ax2 = fig.add_subplot(gs[1, 0])
    running_max = cumulative_strategy.expanding().max()
    drawdown = (cumulative_strategy - running_max) / running_max
    ax2.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
    ax2.plot(drawdown.index, drawdown.values, color='red', linewidth=1)
    ax2.set_title('Strategy Drawdown', fontweight='bold')
    ax2.set_ylabel('Drawdown')
    ax2.grid(True, alpha=0.3)
    
    # 3. Annual Returns
    ax3 = fig.add_subplot(gs[1, 1])
    annual_returns = strategy_returns.groupby(strategy_returns.index.year).apply(lambda x: (1 + x).prod() - 1)
    benchmark_annual = benchmark_returns.groupby(benchmark_returns.index.year).apply(lambda x: (1 + x).prod() - 1)
    
    x = np.arange(len(annual_returns))
    width = 0.35
    ax3.bar(x - width/2, annual_returns.values, width, label='Strategy', color='#2E86AB', alpha=0.7)
    ax3.bar(x + width/2, benchmark_annual.values, width, label='Benchmark', color='#A23B72', alpha=0.7)
    ax3.set_title('Annual Returns', fontweight='bold')
    ax3.set_ylabel('Return')
    ax3.set_xticks(x)
    ax3.set_xticklabels(annual_returns.index)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Rolling Sharpe Ratio
    ax4 = fig.add_subplot(gs[2, 0])
    rolling_sharpe = strategy_returns.rolling(252).mean() / strategy_returns.rolling(252).std() * np.sqrt(252)
    ax4.plot(rolling_sharpe.index, rolling_sharpe.values, color='#2E86AB', linewidth=2)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.set_title('Rolling Sharpe Ratio (252-day)', fontweight='bold')
    ax4.set_ylabel('Sharpe Ratio')
    ax4.grid(True, alpha=0.3)
    
    # 5. Portfolio Turnover
    ax5 = fig.add_subplot(gs[2, 1])
    if not diagnostics.empty and 'turnover' in diagnostics.columns:
        diagnostics['turnover'].plot(ax=ax5, color='#E67E22', linewidth=2)
        ax5.set_title('Portfolio Turnover', fontweight='bold')
        ax5.set_ylabel('Turnover Rate')
        ax5.grid(True, linestyle='--', alpha=0.5)
    
    # 6. Performance Metrics Table
    ax6 = fig.add_subplot(gs[3, :])
    ax6.axis('tight')
    ax6.axis('off')
    
    # Create metrics table
    metrics_data = [
        ['Metric', 'Strategy', 'Benchmark'],
        ['Annualized Return', f"{metrics['annualized_return']:.2%}", f"{metrics['benchmark_annualized_return']:.2%}"],
        ['Volatility', f"{metrics['volatility']:.2%}", f"{metrics['benchmark_volatility']:.2%}"],
        ['Sharpe Ratio', f"{metrics['sharpe_ratio']:.2f}", f"{metrics['benchmark_sharpe']:.2f}"],
        ['Max Drawdown', f"{metrics['max_drawdown']:.2%}", f"{metrics['benchmark_max_drawdown']:.2%}"],
        ['Information Ratio', f"{metrics['information_ratio']:.2f}", '-'],
        ['Beta', f"{metrics['beta']:.2f}", '1.00'],
        ['Calmar Ratio', f"{metrics['calmar_ratio']:.2f}", '-']
    ]
    
    table = ax6.table(cellText=metrics_data[1:], colLabels=metrics_data[0], 
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(metrics_data)):
        for j in range(len(metrics_data[0])):
            if i == 0:  # Header row
                table[(i, j)].set_facecolor('#2E86AB')
                table[(i, j)].set_text_props(weight='bold', color='white')
            else:
                table[(i, j)].set_facecolor('#F8F9FA')
    
    ax6.set_title('Performance Metrics Summary', fontweight='bold', fontsize=14, pad=20)
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"tearsheet_optimized_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Tearsheet saved as: {filename}")
    
    return metrics

# MAIN EXECUTION
if __name__ == "__main__":
    print("🚀 QVM ENGINE V3J OPTIMIZED STRATEGY EXECUTION")
    print("=" * 80)
    
    try:
        # 1. Database Connection
        print("📊 Step 1: Establishing database connection...")
        db_engine = create_db_connection()
        
        # 2. Load Data
        print("\n📊 Step 2: Loading data...")
        price_data, fundamental_data, returns_matrix, benchmark_returns = load_all_data_for_backtest(QVM_CONFIG, db_engine)
        
        # 3. Precompute Data
        print("\n📊 Step 3: Precomputing data...")
        precomputed_data = precompute_all_data(QVM_CONFIG, db_engine)
        
        # 4. Initialize Engine
        print("\n📊 Step 4: Initializing optimized engine...")
        engine = QVMEngineV3jOptimized(
            QVM_CONFIG, price_data, fundamental_data, 
            returns_matrix, benchmark_returns, db_engine, precomputed_data
        )
        
        # 5. Run Backtest
        print("\n📊 Step 5: Running optimized backtest...")
        strategy_returns, diagnostics = engine.run_backtest()
        
        # 6. Generate Tearsheet
        print("\n📊 Step 6: Generating optimized tearsheet...")
        metrics = generate_comprehensive_tearsheet(strategy_returns, benchmark_returns, diagnostics, "QVM Engine v3j Optimized")
        
        # 7. Performance Summary
        print("\n" + "=" * 80)
        print("📊 QVM ENGINE V3J: OPTIMIZED STRATEGY RESULTS")
        print("=" * 80)
        print(f"📈 Performance Summary:")
        print(f"   - Strategy Annualized Return: {metrics['annualized_return']:.2%}")
        print(f"   - Benchmark Annualized Return: {metrics['benchmark_annualized_return']:.2%}")
        print(f"   - Strategy Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"   - Benchmark Sharpe Ratio: {metrics['benchmark_sharpe']:.2f}")
        print(f"   - Strategy Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"   - Benchmark Max Drawdown: {metrics['benchmark_max_drawdown']:.2%}")
        print(f"   - Information Ratio: {metrics['information_ratio']:.2f}")
        print(f"   - Beta: {metrics['beta']:.2f}")
        
        print(f"\n🔧 Optimized Configuration:")
        print(f"   - Simplified 3-factor structure (ROAA, P/E, Momentum)")
        print(f"   - Removed redundant factors (F-Score, FCF Yield, Low-Volatility)")
        print(f"   - Optimized factor weights based on investigation")
        print(f"   - Improved data quality handling")
        
        print(f"\n✅ QVM Engine v3j Optimized strategy execution complete!")
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc() 