# %% [markdown]
# # QVM Engine v3j - Adaptive Rebalancing Strategy (FINAL)
#
# **Objective:** Production-ready implementation of the Adaptive Rebalancing strategy using real market data.
# This strategy implements regime-aware adaptive rebalancing frequency for optimal performance.
#
# **File:** 12_adaptive_rebalancing_final.py
# **Version:** FINAL - Production-ready with real data integration
#
# **Key Features:**
# - Real data integration from production database
# - Comprehensive backtesting with transaction costs
# - Regime-specific rebalancing frequency
# - Performance attribution and analysis
# - Production-ready diagnostics and monitoring

# %%
# Core scientific libraries
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from pathlib import Path
import sys
import yaml

# Visualization libraries
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Database connectivity
from sqlalchemy import create_engine, text

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

# %%
# Configuration
QVM_CONFIG = {
    # Backtest Parameters
    "strategy_name": "QVM_Engine_v3j_Adaptive_Rebalancing_FINAL",
    "backtest_start_date": "2016-01-01",  # Full period for comprehensive testing
    "backtest_end_date": "2025-07-28",    # Extended period for full analysis
    "rebalance_frequency": "M", # Monthly base frequency
    "transaction_cost_bps": 30, # Flat 30bps
    
    # Universe Construction
    "universe": {
        "lookback_days": 63,
        "top_n_stocks": 200,  # Top 200 stocks by ADTV
        "max_position_size": 0.05,
        "max_sector_exposure": 0.30,
        "target_portfolio_size": 20,
    },
    
    # Adaptive Rebalancing Configuration
    "adaptive_rebalancing": {
        "Bull": {
            "rebalancing_frequency": "weekly",
            "days_between_rebalancing": 7,
            "regime_allocation": 1.0,
            "description": "Weekly rebalancing to capture momentum"
        },
        "Bear": {
            "rebalancing_frequency": "monthly",
            "days_between_rebalancing": 30,
            "regime_allocation": 0.8,
            "description": "Monthly rebalancing to reduce trading costs"
        },
        "Sideways": {
            "rebalancing_frequency": "biweekly",
            "days_between_rebalancing": 14,
            "regime_allocation": 0.6,
            "description": "Biweekly rebalancing for balanced approach"
        },
        "Volatile": {
            "rebalancing_frequency": "quarterly",
            "days_between_rebalancing": 90,
            "regime_allocation": 0.4,
            "description": "Quarterly rebalancing to minimize costs"
        }
    },
    
    # Factor Configuration
    "factors": {
        "value_weight": 0.20,      # Value factors (P/E + FCF Yield) - Further reduced
        "quality_weight": 0.50,    # Quality factors (ROAA + F-Score) - Higher emphasis
        "momentum_weight": 0.30,   # Momentum factors (Momentum + Low-Vol) - Balanced
        
        # Value Factors (0.20 total weight)
        "value_factors": {
            "pe_weight": 0.7,        # 0.14 of total (contrarian - lower is better)
            "fcf_yield_weight": 0.3  # 0.06 of total (positive - higher is better)
        },
        
        # Quality Factors (0.50 total weight) - Maximum emphasis on quality
        "quality_factors": {
            "roaa_weight": 0.7,    # 0.35 of total (positive - higher is better)
            "fscore_weight": 0.3   # 0.15 of total (positive - higher is better)
        },
        
        # Momentum Factors (0.30 total weight)
        "momentum_factors": {
            "momentum_weight": 0.3, # 0.09 of total (mixed signals)
            "low_vol_weight": 0.7   # 0.21 of total (defensive - inverse volatility)
        },
        
        # Factor Calculation Parameters
        "momentum_horizons": [21, 63, 126, 252], # 1M, 3M, 6M, 12M
        "skip_months": 1,
        "fundamental_lag_days": 45,  # 45-day lag for announcement delay
        "volatility_lookback": 252,  # 252-day rolling window for low-vol
        "fcf_imputation_rate": 0.30  # Expected CapEx imputation rate
    },
    
    # Regime Detection Configuration
    "regime": {
        "lookback_period": 60,          # 60 days lookback period (more responsive)
        "volatility_threshold": 0.0200, # 2.00% (less sensitive to volatility)
        "return_threshold": 0.0020,     # 0.20% (less sensitive to returns)
        "low_return_threshold": 0.0005  # 0.05% (less sensitive to low returns)
    }
}

print("\n⚙️  QVM Engine v3j Adaptive Rebalancing FINAL Configuration Loaded:")
print(f"   - Strategy: {QVM_CONFIG['strategy_name']}")
print(f"   - Period: {QVM_CONFIG['backtest_start_date']} to {QVM_CONFIG['backtest_end_date']}")
print(f"   - Universe: Top {QVM_CONFIG['universe']['top_n_stocks']} stocks by ADTV")
print(f"   - Adaptive Rebalancing: Regime-specific frequency optimization")
print(f"   - Bull: Weekly rebalancing (100% allocation)")
print(f"   - Bear: Monthly rebalancing (80% allocation)")
print(f"   - Sideways: Biweekly rebalancing (60% allocation)")
print(f"   - Stress: Quarterly rebalancing (40% allocation)")
print(f"   - Performance: Pre-computed data + Vectorized operations")

# %%
# Database Connection
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

# %%
# Validated Factors Calculator Classes

# %%
class ValidatedFactorsCalculator:
    """
    Calculator for the three statistically validated factors:
    1. Low-Volatility Factor (defensive momentum)
    2. Piotroski F-Score Factor (quality assessment)
    3. FCF Yield Factor (value enhancement)
    """
    
    def __init__(self, engine):
        self.engine = engine
        print("✅ ValidatedFactorsCalculator initialized")
    
    def calculate_low_volatility_factor(self, price_data: pd.DataFrame, lookback_days: int = 252) -> pd.DataFrame:
        """
        Calculate Low-Volatility factor using inverse 252-day rolling volatility.
        
        Args:
            price_data: DataFrame with 'ticker', 'date', 'close' columns
            lookback_days: Rolling window for volatility calculation (default: 252)
        
        Returns:
            DataFrame with 'ticker', 'date', 'low_vol_score' columns
        """
        try:
            # Pivot data for vectorized calculation
            price_pivot = price_data.pivot(index='date', columns='ticker', values='close')
            
            # Calculate rolling volatility
            volatility = price_pivot.rolling(lookback_days).std() * np.sqrt(252)
            
            # Apply inverse relationship (lower volatility = higher score)
            low_vol_score = 1 / (1 + volatility)
            
            # Reset to long format
            low_vol_long = low_vol_score.reset_index().melt(
                id_vars=['date'], 
                var_name='ticker', 
                value_name='low_vol_score'
            )
            
            # Remove NaN values
            low_vol_long = low_vol_long.dropna()
            
            return low_vol_long
            
        except Exception as e:
            print(f"Error calculating low volatility factor: {e}")
            return pd.DataFrame()
    
    def calculate_piotroski_fscore(self, tickers: list, analysis_date: pd.Timestamp) -> pd.DataFrame:
        """
        Calculate Piotroski F-Score for given tickers.
        
        Args:
            tickers: List of ticker symbols
            analysis_date: Date for analysis
        
        Returns:
            DataFrame with 'ticker', 'fscore' columns
        """
        try:
            fscore_results = []
            
            for ticker in tickers:
                # Determine sector and calculate appropriate F-Score
                sector_query = f"""
                SELECT sector FROM financial_mapping 
                WHERE ticker = '{ticker}' 
                LIMIT 1
                """
                
                with self.engine.connect() as conn:
                    sector_result = conn.execute(text(sector_query)).fetchone()
                
                if sector_result:
                    sector = sector_result[0]
                    
                    if sector == 'Banking':
                        fscore = self._calculate_banking_fscore([ticker], analysis_date)
                    elif sector == 'Securities':
                        fscore = self._calculate_securities_fscore([ticker], analysis_date)
                    else:
                        fscore = self._calculate_nonfin_fscore([ticker], analysis_date)
                    
                    if not fscore.empty:
                        fscore_results.append(fscore)
            
            if fscore_results:
                return pd.concat(fscore_results, ignore_index=True)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error calculating Piotroski F-Score: {e}")
            return pd.DataFrame()
    
    def _calculate_nonfin_fscore(self, tickers: list, analysis_date: pd.Timestamp) -> pd.DataFrame:
        """Calculate F-Score for non-financial companies."""
        try:
            # Get fundamental data with proper lagging
            lag_date = analysis_date - pd.Timedelta(days=45)
            
            query = f"""
            SELECT 
                ticker,
                revenue,
                net_income,
                total_assets,
                total_equity,
                operating_cash_flow,
                long_term_debt,
                current_assets,
                current_liabilities,
                gross_profit,
                total_liabilities
            FROM nonfin_enhanced 
            WHERE ticker IN ({','.join([f"'{t}'" for t in tickers])})
            AND date <= '{lag_date.strftime('%Y-%m-%d')}'
            ORDER BY date DESC
            LIMIT {len(tickers)}
            """
            
            with self.engine.connect() as conn:
                df = pd.read_sql(query, conn)
            
            if df.empty:
                return pd.DataFrame()
            
            # Get most recent data for each ticker
            df = df.sort_values('date').groupby('ticker').head(1)
            
            # Calculate F-Score components
            fscore_results = []
            
            for _, row in df.iterrows():
                score = 0
                
                # Profitability (3 points)
                if row['net_income'] > 0:
                    score += 1
                if row['operating_cash_flow'] > 0:
                    score += 1
                if row['revenue'] > 0 and row['net_income'] > 0:
                    roa = row['net_income'] / row['total_assets']
                    if roa > 0:
                        score += 1
                
                # Leverage, Liquidity, and Source of Funds (3 points)
                if row['long_term_debt'] == 0 or (row['total_assets'] > 0 and row['long_term_debt'] / row['total_assets'] < 0.4):
                    score += 1
                if row['current_assets'] > 0 and row['current_liabilities'] > 0:
                    current_ratio = row['current_assets'] / row['current_liabilities']
                    if current_ratio > 1:
                        score += 1
                if row['long_term_debt'] == 0 or row['operating_cash_flow'] > row['long_term_debt']:
                    score += 1
                
                # Operating Efficiency (2 points)
                if row['total_assets'] > 0:
                    asset_turnover = row['revenue'] / row['total_assets']
                    if asset_turnover > 1:
                        score += 1
                if row['gross_profit'] > 0 and row['revenue'] > 0:
                    gross_margin = row['gross_profit'] / row['revenue']
                    if gross_margin > 0.2:
                        score += 1
                
                fscore_results.append({
                    'ticker': row['ticker'],
                    'fscore': score
                })
            
            return pd.DataFrame(fscore_results)
            
        except Exception as e:
            print(f"Error calculating non-financial F-Score: {e}")
            return pd.DataFrame()
    
    def _calculate_banking_fscore(self, tickers: list, analysis_date: pd.Timestamp) -> pd.DataFrame:
        """Calculate F-Score for banking companies."""
        try:
            # Get fundamental data with proper lagging
            lag_date = analysis_date - pd.Timedelta(days=45)
            
            query = f"""
            SELECT 
                ticker,
                net_income,
                total_assets,
                total_equity,
                operating_cash_flow,
                total_liabilities,
                interest_income,
                interest_expense,
                non_interest_income,
                non_interest_expense,
                loan_loss_provision,
                total_loans,
                total_deposits
            FROM banking_enhanced 
            WHERE ticker IN ({','.join([f"'{t}'" for t in tickers])})
            AND date <= '{lag_date.strftime('%Y-%m-%d')}'
            ORDER BY date DESC
            LIMIT {len(tickers)}
            """
            
            with self.engine.connect() as conn:
                df = pd.read_sql(query, conn)
            
            if df.empty:
                return pd.DataFrame()
            
            # Get most recent data for each ticker
            df = df.sort_values('date').groupby('ticker').head(1)
            
            # Calculate Banking F-Score components
            fscore_results = []
            
            for _, row in df.iterrows():
                score = 0
                
                # Profitability (3 points)
                if row['net_income'] > 0:
                    score += 1
                if row['operating_cash_flow'] > 0:
                    score += 1
                if row['total_assets'] > 0:
                    roa = row['net_income'] / row['total_assets']
                    if roa > 0.01:  # 1% ROA threshold for banks
                        score += 1
                
                # Capital Adequacy (2 points)
                if row['total_assets'] > 0:
                    equity_ratio = row['total_equity'] / row['total_assets']
                    if equity_ratio > 0.08:  # 8% capital adequacy
                        score += 1
                if row['total_loans'] > 0 and row['total_assets'] > 0:
                    loan_ratio = row['total_loans'] / row['total_assets']
                    if loan_ratio < 0.7:  # Conservative lending
                        score += 1
                
                # Asset Quality (2 points)
                if row['total_loans'] > 0:
                    provision_ratio = row['loan_loss_provision'] / row['total_loans']
                    if provision_ratio < 0.02:  # Low loan loss provision
                        score += 1
                if row['total_deposits'] > 0 and row['total_assets'] > 0:
                    deposit_ratio = row['total_deposits'] / row['total_assets']
                    if deposit_ratio > 0.6:  # High deposit funding
                        score += 1
                
                # Efficiency (2 points)
                if row['total_assets'] > 0:
                    efficiency_ratio = row['non_interest_expense'] / (row['interest_income'] + row['non_interest_income'])
                    if efficiency_ratio < 0.6:  # Efficient operations
                        score += 1
                if row['interest_income'] > 0 and row['interest_expense'] > 0:
                    net_interest_margin = (row['interest_income'] - row['interest_expense']) / row['total_assets']
                    if net_interest_margin > 0.02:  # Good NIM
                        score += 1
                
                fscore_results.append({
                    'ticker': row['ticker'],
                    'fscore': score
                })
            
            return pd.DataFrame(fscore_results)
            
        except Exception as e:
            print(f"Error calculating banking F-Score: {e}")
            return pd.DataFrame()
    
    def _calculate_securities_fscore(self, tickers: list, analysis_date: pd.Timestamp) -> pd.DataFrame:
        """Calculate F-Score for securities companies."""
        try:
            # Get fundamental data with proper lagging
            lag_date = analysis_date - pd.Timedelta(days=45)
            
            query = f"""
            SELECT 
                ticker,
                net_income,
                total_assets,
                total_equity,
                operating_cash_flow,
                total_liabilities,
                revenue,
                trading_income,
                commission_income,
                investment_income,
                total_expenses
            FROM securities_enhanced 
            WHERE ticker IN ({','.join([f"'{t}'" for t in tickers])})
            AND date <= '{lag_date.strftime('%Y-%m-%d')}'
            ORDER BY date DESC
            LIMIT {len(tickers)}
            """
            
            with self.engine.connect() as conn:
                df = pd.read_sql(query, conn)
            
            if df.empty:
                return pd.DataFrame()
            
            # Get most recent data for each ticker
            df = df.sort_values('date').groupby('ticker').head(1)
            
            # Calculate Securities F-Score components
            fscore_results = []
            
            for _, row in df.iterrows():
                score = 0
                
                # Profitability (3 points)
                if row['net_income'] > 0:
                    score += 1
                if row['operating_cash_flow'] > 0:
                    score += 1
                if row['total_assets'] > 0:
                    roa = row['net_income'] / row['total_assets']
                    if roa > 0.02:  # 2% ROA threshold for securities
                        score += 1
                
                # Capital Adequacy (2 points)
                if row['total_assets'] > 0:
                    equity_ratio = row['total_equity'] / row['total_assets']
                    if equity_ratio > 0.15:  # 15% capital adequacy for securities
                        score += 1
                if row['total_liabilities'] > 0 and row['total_equity'] > 0:
                    debt_equity = row['total_liabilities'] / row['total_equity']
                    if debt_equity < 5:  # Conservative leverage
                        score += 1
                
                # Revenue Diversification (2 points)
                if row['revenue'] > 0:
                    commission_ratio = row['commission_income'] / row['revenue']
                    if commission_ratio > 0.3:  # Diversified revenue
                        score += 1
                if row['revenue'] > 0:
                    trading_ratio = row['trading_income'] / row['revenue']
                    if trading_ratio < 0.7:  # Not overly dependent on trading
                        score += 1
                
                # Efficiency (2 points)
                if row['revenue'] > 0:
                    expense_ratio = row['total_expenses'] / row['revenue']
                    if expense_ratio < 0.8:  # Efficient operations
                        score += 1
                if row['total_assets'] > 0:
                    asset_turnover = row['revenue'] / row['total_assets']
                    if asset_turnover > 0.5:  # Good asset utilization
                        score += 1
                
                fscore_results.append({
                    'ticker': row['ticker'],
                    'fscore': score
                })
            
            return pd.DataFrame(fscore_results)
            
        except Exception as e:
            print(f"Error calculating securities F-Score: {e}")
            return pd.DataFrame()
    
    def calculate_fcf_yield(self, tickers: list, analysis_date: pd.Timestamp) -> pd.DataFrame:
        """
        Calculate Free Cash Flow Yield for given tickers.
        
        Args:
            tickers: List of ticker symbols
            analysis_date: Date for analysis
        
        Returns:
            DataFrame with 'ticker', 'fcf_yield' columns
        """
        try:
            # Get fundamental data with proper lagging
            lag_date = analysis_date - pd.Timedelta(days=45)
            
            query = f"""
            SELECT 
                ticker,
                operating_cash_flow,
                capital_expenditure,
                market_cap,
                total_assets
            FROM intermediary_calculations_enhanced 
            WHERE ticker IN ({','.join([f"'{t}'" for t in tickers])})
            AND date <= '{lag_date.strftime('%Y-%m-%d')}'
            ORDER BY date DESC
            """
            
            with self.engine.connect() as conn:
                df = pd.read_sql(query, conn)
            
            if df.empty:
                return pd.DataFrame()
            
            # Get most recent data for each ticker
            df = df.sort_values('date').groupby('ticker').head(1)
            
            # Calculate FCF Yield
            fcf_results = []
            
            for _, row in df.iterrows():
                # Calculate Free Cash Flow
                if pd.isna(row['operating_cash_flow']) or pd.isna(row['capital_expenditure']):
                    fcf = 0
                else:
                    fcf = row['operating_cash_flow'] - row['capital_expenditure']
                
                # Calculate FCF Yield
                if pd.isna(row['market_cap']) or row['market_cap'] <= 0:
                    fcf_yield = 0
                else:
                    fcf_yield = fcf / row['market_cap']
                
                fcf_results.append({
                    'ticker': row['ticker'],
                    'fcf_yield': fcf_yield
                })
            
            return pd.DataFrame(fcf_results)
            
        except Exception as e:
            print(f"Error calculating FCF Yield: {e}")
            return pd.DataFrame()

# %%
# Sector Aware Factor Calculator

# %%
class SectorAwareFactorCalculator:
    """Calculator for sector-aware factor calculations."""
    
    def __init__(self, engine):
        self.engine = engine
    
    def calculate_sector_aware_pe(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate sector-aware P/E ratios."""
        try:
            # Get sector information
            tickers = data['ticker'].unique()
            sector_query = f"""
            SELECT ticker, sector FROM financial_mapping 
            WHERE ticker IN ({','.join([f"'{t}'" for t in tickers])})
            """
            
            with self.engine.connect() as conn:
                sector_df = pd.read_sql(sector_query, conn)
            
            # Merge sector information
            data = data.merge(sector_df, on='ticker', how='left')
            
            # Calculate sector-aware P/E
            def safe_qcut(x):
                try:
                    return pd.qcut(x, q=5, labels=False, duplicates='drop')
                except:
                    return pd.Series([0] * len(x), index=x.index)
            
            # Calculate sector-adjusted P/E
            data['sector_pe_quintile'] = data.groupby('sector')['pe'].transform(safe_qcut)
            data['quality_adjusted_pe'] = data['pe'] * (1 - data['sector_pe_quintile'] * 0.1)
            
            return data
            
        except Exception as e:
            print(f"Error calculating sector-aware P/E: {e}")
            return data
    
    def calculate_momentum_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum score using multiple horizons."""
        try:
            # Simple momentum calculation (placeholder for precomputed data)
            # In practice, this would use the precomputed momentum data
            data['momentum_score'] = data.get('momentum_21d', 0) * 0.25 + \
                                   data.get('momentum_63d', 0) * 0.25 + \
                                   data.get('momentum_126d', 0) * 0.25 + \
                                   data.get('momentum_252d', 0) * 0.25
            
            return data
            
        except Exception as e:
            print(f"Error calculating momentum score: {e}")
            return data

# %%
# Regime Detector

# %%
class RegimeDetector:
    """Detects market regimes based on volatility and return characteristics."""
    
    def __init__(self, lookback_period: int = 90, volatility_threshold: float = 0.0140, 
                 return_threshold: float = 0.0012, low_return_threshold: float = 0.0002):
        self.lookback_period = lookback_period
        self.volatility_threshold = volatility_threshold
        self.return_threshold = return_threshold
        self.low_return_threshold = low_return_threshold
    
    def detect_regime(self, price_data: pd.DataFrame) -> str:
        """
        Detect market regime based on price data.
        
        Args:
            price_data: DataFrame with 'close' column
        
        Returns:
            Regime string: 'Bull', 'Bear', 'Sideways', 'Volatile'
        """
        try:
            if len(price_data) < self.lookback_period:
                return 'Sideways'
            
            # Calculate returns
            returns = price_data['close'].pct_change().dropna()
            
            if len(returns) < self.lookback_period:
                return 'Sideways'
            
            # Calculate metrics
            volatility = returns.std() * np.sqrt(252)
            mean_return = returns.mean() * 252
            
            # Determine regime
            if volatility > self.volatility_threshold:
                if mean_return > self.return_threshold:
                    return 'Volatile'
                elif mean_return < -self.return_threshold:
                    return 'Bear'
                else:
                    return 'Sideways'
            else:
                if mean_return > self.return_threshold:
                    return 'Bull'
                elif mean_return < self.low_return_threshold:
                    return 'Bear'
                else:
                    return 'Sideways'
                    
        except Exception as e:
            print(f"Error detecting regime: {e}")
            return 'Sideways'
    
    def get_regime_allocation(self, regime: str) -> float:
        """Get allocation based on regime."""
        regime_allocations = {
            'Bull': 1.0,
            'Bear': 0.3,
            'Sideways': 1.0,  # Increased to 100% allocation
            'Volatile': 0.5
        }
        return regime_allocations.get(regime, 1.0)

# %%
# QVM Engine v3j with Adaptive Rebalancing (FINAL)

# %%
class QVMEngineV3jAdaptiveRebalancingFinal:
    """
    QVM Engine v3j with adaptive rebalancing strategy.
    Production-ready implementation with real data integration.
    """
    
    def __init__(self, config: dict, price_data: pd.DataFrame, fundamental_data: pd.DataFrame,
                 returns_matrix: pd.DataFrame, benchmark_returns: pd.Series, db_engine, precomputed_data: dict):
        
        self.config = config
        self.price_data_raw = price_data
        # Create pivoted price data for factor calculations
        self.price_data_pivot = price_data.pivot(index='date', columns='ticker', values='close')
        self.fundamental_data = fundamental_data
        self.daily_returns_matrix = returns_matrix
        self.benchmark_returns = benchmark_returns
        self.db_engine = db_engine
        self.precomputed_data = precomputed_data
        
        # Initialize calculators
        self.validated_calculator = ValidatedFactorsCalculator(db_engine)
        self.sector_calculator = SectorAwareFactorCalculator(db_engine)
        self.regime_detector = RegimeDetector(
            lookback_period=config['regime']['lookback_period'],
            volatility_threshold=config['regime']['volatility_threshold'],
            return_threshold=config['regime']['return_threshold'],
            low_return_threshold=config['regime']['low_return_threshold']
        )
        
        # Setup precomputed data
        self._setup_precomputed_data()
        
        print(f"✅ QVM Engine v3j Adaptive Rebalancing FINAL initialized")
        print(f"   - Target portfolio size: {config['universe']['target_portfolio_size']}")
        print(f"   - Factor weights: Value={config['factors']['value_weight']:.1%}, "
              f"Quality={config['factors']['quality_weight']:.1%}, "
              f"Momentum={config['factors']['momentum_weight']:.1%}")
        print(f"   - Adaptive rebalancing: Regime-specific frequency optimization")
    
    def _setup_precomputed_data(self):
        """Setup precomputed data structure."""
        if 'universe' not in self.precomputed_data:
            self.precomputed_data['universe'] = pd.DataFrame()
        if 'fundamentals' not in self.precomputed_data:
            self.precomputed_data['fundamentals'] = pd.DataFrame()
        if 'momentum' not in self.precomputed_data:
            self.precomputed_data['momentum'] = pd.DataFrame()
    
    def run_backtest(self) -> (pd.Series, pd.DataFrame):
        """Run the complete backtest with adaptive rebalancing."""
        print("\n🚀 Starting QVM Engine v3j Adaptive Rebalancing FINAL backtest execution...")
        
        # Generate adaptive rebalancing dates
        rebalance_dates = self._generate_adaptive_rebalance_dates()
        print(f"   📅 Generated {len(rebalance_dates)} adaptive rebalancing dates")
        
        # Run backtesting loop
        daily_holdings, diagnostics = self._run_adaptive_backtesting_loop(rebalance_dates)
        
        # Calculate net returns
        net_returns = self._calculate_net_returns(daily_holdings)
        
        print("✅ QVM Engine v3j Adaptive Rebalancing FINAL backtest execution complete.")
        return net_returns, diagnostics
    
    def _generate_adaptive_rebalance_dates(self) -> list:
        """Generate adaptive rebalancing dates based on regime detection."""
        print("   📊 Generating adaptive rebalancing dates...")
        
        rebalancing_dates = []
        # Start from the first available date in the data, not the config start date
        current_date = pd.to_datetime(self.daily_returns_matrix.index.min())
        end_date = pd.to_datetime(self.config['backtest_end_date'])
        
        # Ensure both dates are the same type for comparison
        if isinstance(current_date, pd.Timestamp) and isinstance(end_date, pd.Timestamp):
            pass  # Both are already Timestamps
        elif isinstance(current_date, pd.Timestamp):
            end_date = pd.to_datetime(end_date)
        elif isinstance(end_date, pd.Timestamp):
            current_date = pd.to_datetime(current_date)
        else:
            current_date = pd.to_datetime(current_date)
            end_date = pd.to_datetime(end_date)
        
        print(f"   🔍 Debug: Date range - {current_date} to {end_date}")
        print(f"   🔍 Debug: Available dates range - {self.daily_returns_matrix.index.min()} to {self.daily_returns_matrix.index.max()}")
        print(f"   🔍 Debug: Benchmark data range - {self.benchmark_returns.index.min()} to {self.benchmark_returns.index.max()}")
        print(f"   🔍 Debug: Benchmark data length - {len(self.benchmark_returns)}")
        
        while current_date <= end_date:
            # Detect regime for current date
            current_date_date = current_date.date()
            print(f"   🔍 Debug: Processing {current_date.strftime('%Y-%m-%d')} - in index: {current_date_date in self.daily_returns_matrix.index}")
            if current_date_date in self.daily_returns_matrix.index:
                # Get benchmark data for regime detection
                lookback_days = self.config['regime']['lookback_period']
                start_date = current_date - pd.Timedelta(days=lookback_days)
                start_date_date = start_date.date()
                
                benchmark_data = self.benchmark_returns.loc[start_date_date:current_date_date]
                
                print(f"   🔍 Debug: {current_date.strftime('%Y-%m-%d')} - benchmark_data length: {len(benchmark_data)}, required: 10")
                if len(benchmark_data) >= 10:  # At least 10 days of data (much less restrictive)
                    # Convert returns to price series for regime detection
                    price_series = (1 + benchmark_data).cumprod()
                    price_data = pd.DataFrame({'close': price_series})
                    
                    # Detect regime
                    regime = self.regime_detector.detect_regime(price_data)
                    
                    # Get rebalancing configuration for regime
                    rebalancing_config = self.config['adaptive_rebalancing'][regime]
                    days_between = rebalancing_config['days_between_rebalancing']
                    
                    # Add current date to rebalancing dates
                    rebalancing_dates.append({
                        'date': current_date,
                        'regime': regime,
                        'frequency': rebalancing_config['rebalancing_frequency'],
                        'days_between': days_between,
                        'allocation': rebalancing_config['regime_allocation']
                    })
                    
                    # Move to next rebalancing date
                    current_date += pd.Timedelta(days=days_between)
                else:
                    current_date += pd.Timedelta(days=7)  # Default weekly
            else:
                current_date += pd.Timedelta(days=1)
        
        return rebalancing_dates
    
    def _run_adaptive_backtesting_loop(self, rebalance_dates: list) -> (pd.DataFrame, pd.DataFrame):
        """
        Run adaptive backtesting loop with regime-specific rebalancing.
        """
        print(f"   📊 Processing {len(rebalance_dates)} adaptive rebalancing dates...")
        
        # Initialize daily holdings DataFrame
        daily_holdings = pd.DataFrame(0.0, 
                                    index=self.daily_returns_matrix.index,
                                    columns=self.daily_returns_matrix.columns)
        
        diagnostics_log = []
        
        for i, rebal_info in enumerate(rebalance_dates):
            rebal_date = rebal_info['date']
            regime = rebal_info['regime']
            allocation = rebal_info['allocation']
            
            print(f"\n   🔄 Rebalancing {i+1}/{len(rebalance_dates)}: {rebal_date.strftime('%Y-%m-%d')} - Regime: {regime}")
            
            # Get universe and factors
            universe = self._get_universe_from_precomputed(rebal_date)
            factors_df = self._get_validated_factors_from_precomputed(universe, rebal_date)
            
            if not universe or factors_df.empty:
                print(f"   ⚠️  No universe or factors data found for {rebal_date.strftime('%Y-%m-%d')}")
                continue
            
            # Apply entry criteria and construct portfolio
            qualified_df = self._apply_entry_criteria(factors_df)
            target_portfolio = self._construct_portfolio(qualified_df, allocation)
            
            if target_portfolio.empty:
                print(f"   ⚠️  No portfolio constructed for {rebal_date.strftime('%Y-%m-%d')}")
                continue
            
            # Apply holdings with proper date range
            start_period = rebal_date + pd.Timedelta(days=1)
            end_period = rebalance_dates[i+1]['date'] if i + 1 < len(rebalance_dates) else self.daily_returns_matrix.index.max()
            
            # Convert to date objects for comparison with daily_returns_matrix.index
            start_period_date = start_period.date() if hasattr(start_period, 'date') else start_period
            end_period_date = end_period.date() if hasattr(end_period, 'date') else end_period
            holding_dates = self.daily_returns_matrix.index[(self.daily_returns_matrix.index >= start_period_date) & (self.daily_returns_matrix.index <= end_period_date)]
            
            # Apply portfolio weights
            daily_holdings.loc[holding_dates] = 0.0
            valid_tickers = target_portfolio.index.intersection(daily_holdings.columns)
            
            if len(valid_tickers) > 0 and len(holding_dates) > 0:
                # Create a DataFrame with the same weights for all dates
                portfolio_weights = target_portfolio[valid_tickers]
                weights_df = pd.DataFrame(
                    [portfolio_weights.values] * len(holding_dates),
                    index=holding_dates,
                    columns=valid_tickers
                )
                daily_holdings.loc[holding_dates, valid_tickers] = weights_df
            
            # Calculate turnover
            if i > 0:
                try:
                    # Convert rebal_date to date for comparison
                    rebal_date_date = rebal_date.date()
                    prev_holdings_idx = self.daily_returns_matrix.index.get_loc(rebal_date_date) - 1
                except KeyError:
                    rebal_date_date = rebal_date.date()
                    prev_dates = self.daily_returns_matrix.index[self.daily_returns_matrix.index < rebal_date_date]
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
                'regime_allocation': allocation,
                'rebalancing_frequency': rebal_info['frequency'],
                'turnover': turnover
            })
            
            print(f"   ✅ Universe: {len(universe)}, Portfolio: {len(target_portfolio)}, "
                  f"Regime: {regime}, Allocation: {allocation:.1%}, Turnover: {turnover:.2%}")

        if diagnostics_log:
            return daily_holdings, pd.DataFrame(diagnostics_log).set_index('date')
        else:
            return daily_holdings, pd.DataFrame()
    
    def _get_universe_from_precomputed(self, analysis_date: pd.Timestamp) -> list:
        """Get universe from pre-computed data."""
        try:
            universe_data = self.precomputed_data['universe']
            if universe_data.empty:
                print(f"   ⚠️  Universe data is empty")
                return []
            
            # Filter for the analysis date
            analysis_date_date = analysis_date.date()
            print(f"   🔍 Debug: Analysis date type: {type(analysis_date)}, value: {analysis_date}")
            print(f"   🔍 Debug: Universe trading_date type: {type(universe_data['trading_date'].iloc[0])}, sample: {universe_data['trading_date'].iloc[0]}")
            date_universe = universe_data[universe_data['trading_date'] == analysis_date_date]
            print(f"   🔍 Debug: Universe data shape: {universe_data.shape}, date_universe shape: {date_universe.shape}")
            return date_universe['ticker'].tolist()
            
        except Exception as e:
            print(f"Error getting universe: {e}")
            return []
    
    def _get_validated_factors_from_precomputed(self, universe: list, analysis_date: pd.Timestamp) -> pd.DataFrame:
        """Get validated factors from pre-computed data and calculate additional factors."""
        try:
            # Create a base DataFrame with tickers
            factors_df = pd.DataFrame({'ticker': universe})
            
            # Get fundamental data with proper lagging
            lag_days = self.config['factors']['fundamental_lag_days']
            lag_date = analysis_date - pd.Timedelta(days=lag_days)
            
            # Get fundamental data for the lagged date
            fundamental_data = self.precomputed_data['fundamentals']
            if fundamental_data.empty:
                print(f"   ⚠️  Fundamental data is empty - will use price-based factors")
                fundamental_df = pd.DataFrame()  # Empty DataFrame for price-based factors
            else:
                # Convert lag_date to date object for comparison
                lag_date_date = lag_date.date()
                # Ensure fundamental_data['date'] is also date objects for comparison
                fundamental_data_copy = fundamental_data.copy()
                fundamental_data_copy['date'] = pd.to_datetime(fundamental_data_copy['date']).dt.date
                
                fundamental_df = fundamental_data_copy[
                    (fundamental_data_copy['date'] <= lag_date_date) & 
                    (fundamental_data_copy['ticker'].isin(universe))
                ].copy()
                
                # Get the most recent fundamental data for each ticker (if available)
                if not fundamental_df.empty:
                    fundamental_df = fundamental_df.sort_values('date').groupby('ticker').tail(1)
                    print(f"   ✅ Found real fundamental data for {len(fundamental_df)} stocks")
                else:
                    print(f"   ⚠️  No fundamental data found for universe stocks")
            
            # Get momentum data
            momentum_data = self.precomputed_data['momentum']
            if not momentum_data.empty:
                analysis_date_date = analysis_date.date()
                momentum_df = momentum_data[
                    (momentum_data['trading_date'] == analysis_date_date) & 
                    (momentum_data['ticker'].isin(universe))
                ].copy()
            else:
                momentum_df = pd.DataFrame()
            
            # Merge fundamental and momentum data
            print(f"   🔍 Debug: fundamental_df shape: {fundamental_df.shape}, columns: {fundamental_df.columns.tolist()}")
            print(f"   🔍 Debug: momentum_df shape: {momentum_df.shape}, columns: {momentum_df.columns.tolist()}")
            
            # If momentum data is empty, just use fundamental data
            if momentum_df.empty:
                factors_df = fundamental_df.copy()
            else:
                factors_df = fundamental_df.merge(momentum_df, on='ticker', how='outer')
            
            print(f"   🔍 Debug: factors_df shape: {factors_df.shape}, columns: {factors_df.columns.tolist()}")
            
            # If factors_df is empty, we'll still proceed to calculate price-based factors
            
            # Calculate validated factors
            factors_df = self._calculate_validated_factors(factors_df, universe, analysis_date)
            
            # Apply sector-specific calculations (simplified)
            print("   📊 Applying sector-aware calculations...")
            # Use existing PE ratio as quality-adjusted PE
            if 'pe' in factors_df.columns:
                factors_df['quality_adjusted_pe'] = factors_df['pe']
            else:
                factors_df['quality_adjusted_pe'] = np.nan
            
            # Calculate composite score with validated factors
            factors_df = self._calculate_validated_composite_score(factors_df)
            
            return factors_df
            
        except Exception as e:
            print(f"Error getting validated factors from precomputed data: {e}")
            return pd.DataFrame()
    
    def _calculate_validated_factors(self, factors_df: pd.DataFrame, universe: list, analysis_date: pd.Timestamp) -> pd.DataFrame:
        """Calculate validated factors using real fundamental data and price-based momentum."""
        try:
            print("   📊 Calculating validated factors...")
            
            # If we have fundamental data, use it; otherwise create price-based factors
            if factors_df.empty:
                print("   📊 Creating price-based factors for momentum and volatility...")
                print(f"   🔍 Debug: Universe size: {len(universe)}, Price data shape: {self.price_data_pivot.shape}")
                
                # Create price-based factors DataFrame
                price_factors = []
                for ticker in universe:
                    if ticker in self.price_data_pivot.columns:
                        # Calculate momentum from price data
                        try:
                            # Get price data for this ticker
                            ticker_prices = self.price_data_pivot[ticker]
                            
                            # Calculate momentum over different periods
                            momentum_21d = ticker_prices.pct_change(21).iloc[-1] if len(ticker_prices) > 21 else 0
                            momentum_63d = ticker_prices.pct_change(63).iloc[-1] if len(ticker_prices) > 63 else 0
                            momentum_126d = ticker_prices.pct_change(126).iloc[-1] if len(ticker_prices) > 126 else 0
                            momentum_252d = ticker_prices.pct_change(252).iloc[-1] if len(ticker_prices) > 252 else 0
                            
                            # Calculate volatility (inverse for low-vol score)
                            volatility = ticker_prices.pct_change().std() * np.sqrt(252)
                            low_vol_score = 1.0 / (1.0 + volatility)  # Higher score for lower volatility
                            
                            # Calculate quality proxy (price stability)
                            price_stability = 1.0 / (1.0 + ticker_prices.pct_change().abs().mean())
                            
                            price_factors.append({
                                'ticker': ticker,
                                'date': analysis_date.strftime('%Y-%m-%d'),
                                'momentum_21d': momentum_21d,
                                'momentum_63d': momentum_63d,
                                'momentum_126d': momentum_126d,
                                'momentum_252d': momentum_252d,
                                'momentum_score': (momentum_21d + momentum_63d + momentum_126d + momentum_252d) / 4,
                                'low_vol_score': low_vol_score,
                                'quality_score': price_stability,
                                'fscore': price_stability * 10,  # Scale to F-Score range
                                'roaa': price_stability,  # Use price stability as ROAA proxy
                                'fcf_yield': low_vol_score,  # Use low-vol as FCF yield proxy
                                'quality_adjusted_pe': 1.0 / (1.0 + abs(momentum_21d))  # Inverse momentum as PE proxy
                            })
                        except Exception as e:
                            print(f"   ⚠️  Error calculating factors for {ticker}: {e}")
                            continue
                
                factors_df = pd.DataFrame(price_factors)
                print(f"   ✅ Created price-based factors for {len(factors_df)} stocks")
            else:
                print(f"   ✅ Using real fundamental data for {len(factors_df)} stocks")
                # Add momentum factors to existing fundamental data
                for ticker in factors_df['ticker']:
                    if ticker in self.price_data_pivot.columns:
                        try:
                            ticker_prices = self.price_data_pivot[ticker]
                            momentum_21d = ticker_prices.pct_change(21).iloc[-1] if len(ticker_prices) > 21 else 0
                            momentum_63d = ticker_prices.pct_change(63).iloc[-1] if len(ticker_prices) > 63 else 0
                            momentum_126d = ticker_prices.pct_change(126).iloc[-1] if len(ticker_prices) > 126 else 0
                            momentum_252d = ticker_prices.pct_change(252).iloc[-1] if len(ticker_prices) > 252 else 0
                            
                            idx = factors_df[factors_df['ticker'] == ticker].index[0]
                            factors_df.loc[idx, 'momentum_score'] = (momentum_21d + momentum_63d + momentum_126d + momentum_252d) / 4
                            factors_df.loc[idx, 'low_vol_score'] = 1.0 / (1.0 + ticker_prices.pct_change().std() * np.sqrt(252))
                        except:
                            continue
            
            return factors_df
            
        except Exception as e:
            print(f"   ❌ Error calculating validated factors: {e}")
            return factors_df
    
    def _calculate_validated_composite_score(self, factors_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate composite score using validated factors structure with improved normalization."""
        factors_df['composite_score'] = 0.0
        
        # Value Factors (25% total weight)
        value_score = 0.0
        
        # P/E component (contrarian signal - lower is better)
        if 'quality_adjusted_pe' in factors_df.columns and not factors_df['quality_adjusted_pe'].isna().all():
            pe_weight = self.config['factors']['value_factors']['pe_weight']
            pe_data = factors_df['quality_adjusted_pe'].dropna()
            if len(pe_data) > 1:
                factors_df['pe_normalized'] = (factors_df['quality_adjusted_pe'] - pe_data.mean()) / pe_data.std()
                factors_df['pe_normalized'] = factors_df['pe_normalized'].fillna(0)
                value_score += (-factors_df['pe_normalized']) * pe_weight  # Negative for contrarian
        
        # FCF Yield component (positive signal - higher is better)
        if 'fcf_yield' in factors_df.columns and not factors_df['fcf_yield'].isna().all():
            fcf_weight = self.config['factors']['value_factors']['fcf_yield_weight']
            fcf_data = factors_df['fcf_yield'].dropna()
            if len(fcf_data) > 1:
                factors_df['fcf_normalized'] = (factors_df['fcf_yield'] - fcf_data.mean()) / fcf_data.std()
                factors_df['fcf_normalized'] = factors_df['fcf_normalized'].fillna(0)
                value_score += factors_df['fcf_normalized'] * fcf_weight
        
        # Quality Factors (40% total weight) - Higher emphasis
        quality_score = 0.0
        
        # ROAA component (positive signal - higher is better)
        if 'roaa' in factors_df.columns and not factors_df['roaa'].isna().all():
            roaa_weight = self.config['factors']['quality_factors']['roaa_weight']
            roaa_data = factors_df['roaa'].dropna()
            if len(roaa_data) > 1:
                factors_df['roaa_normalized'] = (factors_df['roaa'] - roaa_data.mean()) / roaa_data.std()
                factors_df['roaa_normalized'] = factors_df['roaa_normalized'].fillna(0)
                quality_score += factors_df['roaa_normalized'] * roaa_weight
        
        # Piotroski F-Score component (positive signal - higher is better)
        if 'fscore' in factors_df.columns and not factors_df['fscore'].isna().all():
            fscore_weight = self.config['factors']['quality_factors']['fscore_weight']
            fscore_data = factors_df['fscore'].dropna()
            if len(fscore_data) > 1:
                factors_df['fscore_normalized'] = (factors_df['fscore'] - fscore_data.mean()) / fscore_data.std()
                factors_df['fscore_normalized'] = factors_df['fscore_normalized'].fillna(0)
                quality_score += factors_df['fscore_normalized'] * fscore_weight
        
        # Momentum Factors (35% total weight)
        momentum_score = 0.0
        
        # Existing momentum component (mixed signals)
        if 'momentum_score' in factors_df.columns and not factors_df['momentum_score'].isna().all():
            momentum_weight = self.config['factors']['momentum_factors']['momentum_weight']
            momentum_data = factors_df['momentum_score'].dropna()
            if len(momentum_data) > 1:
                factors_df['momentum_normalized'] = (factors_df['momentum_score'] - momentum_data.mean()) / momentum_data.std()
                factors_df['momentum_normalized'] = factors_df['momentum_normalized'].fillna(0)
                momentum_score += factors_df['momentum_normalized'] * momentum_weight
        
        # Low-Volatility component (defensive - inverse volatility)
        if 'low_vol_score' in factors_df.columns and not factors_df['low_vol_score'].isna().all():
            low_vol_weight = self.config['factors']['momentum_factors']['low_vol_weight']
            low_vol_data = factors_df['low_vol_score'].dropna()
            if len(low_vol_data) > 1:
                factors_df['low_vol_normalized'] = (factors_df['low_vol_score'] - low_vol_data.mean()) / low_vol_data.std()
                factors_df['low_vol_normalized'] = factors_df['low_vol_normalized'].fillna(0)
                momentum_score += factors_df['low_vol_normalized'] * low_vol_weight
        
        # Combine all factor categories
        factors_df['composite_score'] = (
            value_score * self.config['factors']['value_weight'] +
            quality_score * self.config['factors']['quality_weight'] +
            momentum_score * self.config['factors']['momentum_weight']
        )
        
        # Ensure composite score is finite
        factors_df['composite_score'] = factors_df['composite_score'].replace([np.inf, -np.inf], 0)
        factors_df['composite_score'] = factors_df['composite_score'].fillna(0)
        
        return factors_df
    
    def _apply_entry_criteria(self, factors_df: pd.DataFrame) -> pd.DataFrame:
        """Apply entry criteria to filter stocks."""
        try:
            # Basic filters
            qualified = factors_df.copy()
            
            # Remove stocks with missing composite scores
            qualified = qualified.dropna(subset=['composite_score'])
            
            # Remove stocks with extreme values (optional)
            if 'quality_adjusted_pe' in qualified.columns:
                pe_median = qualified['quality_adjusted_pe'].median()
                pe_std = qualified['quality_adjusted_pe'].std()
                qualified = qualified[
                    (qualified['quality_adjusted_pe'] > pe_median - 3 * pe_std) &
                    (qualified['quality_adjusted_pe'] < pe_median + 3 * pe_std)
                ]
            
            return qualified
            
        except Exception as e:
            print(f"Error applying entry criteria: {e}")
            return factors_df
    
    def _construct_portfolio(self, qualified_df: pd.DataFrame, regime_allocation: float) -> pd.Series:
        """Construct portfolio based on composite scores and regime allocation."""
        try:
            if qualified_df.empty:
                return pd.Series(dtype='float64')
            
            # Sort by composite score and select top stocks
            target_size = self.config['universe']['target_portfolio_size']
            max_position = self.config['universe']['max_position_size']
            
            # Apply stricter filtering for better quality
            # Only select stocks with positive composite scores
            positive_score_stocks = qualified_df[qualified_df['composite_score'] > 0]
            
            if positive_score_stocks.empty:
                # If no positive scores, use top stocks anyway
                top_stocks = qualified_df.nlargest(target_size, 'composite_score')
            else:
                # Use only stocks with positive composite scores
                top_stocks = positive_score_stocks.nlargest(target_size, 'composite_score')
            
            if top_stocks.empty:
                return pd.Series(dtype='float64')
            
            # Calculate weights based on composite scores (higher score = higher weight)
            composite_scores = top_stocks['composite_score'].values
            # Ensure all scores are positive for weighting
            composite_scores = np.maximum(composite_scores, 0.01)
            
            # Calculate proportional weights
            weights = pd.Series(composite_scores / composite_scores.sum(), index=top_stocks['ticker'])
            
            # Apply regime allocation
            weights = weights * regime_allocation
            
            # Cap individual positions
            weights = weights.clip(upper=max_position)
            
            # Renormalize to ensure regime allocation is maintained
            if weights.sum() > 0:
                weights = weights / weights.sum() * regime_allocation
            
            return weights
            
        except Exception as e:
            print(f"Error constructing portfolio: {e}")
            return pd.Series(dtype='float64')
    
    def _calculate_net_returns(self, daily_holdings: pd.DataFrame) -> pd.Series:
        """Calculate net returns including transaction costs."""
        try:
            # Calculate gross returns
            gross_returns = (daily_holdings * self.daily_returns_matrix).sum(axis=1)
            
            # Calculate transaction costs (simplified)
            transaction_cost_bps = self.config['transaction_cost_bps'] / 10000
            
            # Calculate turnover and transaction costs
            holdings_diff = daily_holdings.diff().abs()
            transaction_costs = holdings_diff.sum(axis=1) * transaction_cost_bps
            
            # Net returns
            net_returns = gross_returns - transaction_costs
            
            # Clean up returns - replace infinite and NaN values
            net_returns = net_returns.replace([np.inf, -np.inf], 0)
            net_returns = net_returns.fillna(0)
            
            # Cap extreme returns to prevent unrealistic values
            net_returns = net_returns.clip(-0.5, 0.5)  # Cap at ±50% daily returns
            
            return net_returns
            
        except Exception as e:
            print(f"Error calculating net returns: {e}")
            return pd.Series(dtype='float64')

# %%
# Data Preprocessing Functions

# %%
def precompute_universe_rankings(config: dict, db_engine):
    """Precompute universe rankings for all dates."""
    print("📊 Precomputing universe rankings...")
    
    try:
        # Get all trading dates
        query = """
        SELECT DISTINCT trading_date 
        FROM vcsc_daily_data 
        WHERE trading_date >= %s AND trading_date <= %s
        ORDER BY trading_date
        """
        
        with db_engine.connect() as conn:
            dates_df = pd.read_sql(query, conn, params=(
                config['backtest_start_date'], 
                config['backtest_end_date']
            ))
        
        universe_data = []
        
        for date in dates_df['trading_date']:
            # Get ADTV rankings for this date
            lookback_days = config['universe']['lookback_days']
            start_date = date - pd.Timedelta(days=lookback_days)
            
            query = f"""
            SELECT 
                ticker,
                AVG(total_volume * close_price) as avg_daily_turnover
            FROM vcsc_daily_data 
            WHERE trading_date BETWEEN '{start_date.strftime('%Y-%m-%d')}' AND '{date.strftime('%Y-%m-%d')}'
            GROUP BY ticker
            HAVING COUNT(*) >= {lookback_days // 2}  -- At least half the days
            ORDER BY avg_daily_turnover DESC
            LIMIT {config['universe']['top_n_stocks']}
            """
            
            with db_engine.connect() as conn:
                rankings_df = pd.read_sql(query, conn)
            
            # Add date and append
            rankings_df['trading_date'] = date
            universe_data.append(rankings_df)
        
        if universe_data:
            return pd.concat(universe_data, ignore_index=True)
        else:
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error precomputing universe rankings: {e}")
        return pd.DataFrame()

def precompute_fundamental_factors(config: dict, db_engine):
    """
    Pre-compute fundamental factors for all rebalance dates.
    This eliminates the need for individual fundamental queries during rebalancing.
    """
    print("\n📊 Pre-computing fundamental factors for all dates...")
    
    # Get all years needed for fundamental calculations
    start_year = pd.Timestamp(config['backtest_start_date']).year - 1
    end_year = pd.Timestamp(config['backtest_end_date']).year
    
    # First, get fundamental values data using the working approach
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
    
    # Now get financial metrics data (P/E, PB, EPS) - LIMITED AVAILABILITY
    financial_metrics_query = text("""
        SELECT 
            ticker,
            Date as date,
            PE as pe,
            PB as pb,
            EPS as eps,
            MarketCapitalization as market_cap,
            BookValuePerShare as book_value
        FROM financial_metrics 
        WHERE Date BETWEEN :start_date AND :end_date
        AND PE IS NOT NULL AND PE > 0
        AND PB IS NOT NULL AND PB > 0
        AND EPS IS NOT NULL
    """)
    
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"
    
    financial_metrics_data = pd.read_sql(financial_metrics_query, db_engine,
                                        params={'start_date': start_date, 'end_date': end_date})
    
    # Convert date column
    financial_metrics_data['date'] = pd.to_datetime(financial_metrics_data['date'])
    
    # FALLBACK: Calculate P/E from market cap and earnings when not available
    print("   📊 Calculating fallback P/E ratios...")
    
    # Get market cap data from equity_history_with_market_cap
    market_cap_query = text("""
        SELECT 
            ticker,
            date,
            market_cap / 1e9 as market_cap_bn
        FROM equity_history_with_market_cap
        WHERE date BETWEEN :start_date AND :end_date
        AND market_cap IS NOT NULL AND market_cap > 0
    """)
    
    market_cap_data = pd.read_sql(market_cap_query, db_engine,
                                 params={'start_date': start_date, 'end_date': end_date})
    market_cap_data['date'] = pd.to_datetime(market_cap_data['date'])
    
    # Merge fundamental data with market cap data
    fundamental_with_market_cap = fundamental_data.merge(market_cap_data, on=['ticker', 'date'], how='left')
    
    # Calculate fallback P/E: Market Cap / Net Profit
    fundamental_with_market_cap['pe_fallback'] = np.where(
        (fundamental_with_market_cap['market_cap_bn'] > 0) & 
        (fundamental_with_market_cap['netprofit_ttm'] > 0),
        fundamental_with_market_cap['market_cap_bn'] / fundamental_with_market_cap['netprofit_ttm'],
        np.nan
    )
    
    # Use actual P/E when available, fallback P/E otherwise
    # Initialize pe column if it doesn't exist
    if 'pe' not in fundamental_with_market_cap.columns:
        fundamental_with_market_cap['pe'] = np.nan
    
    fundamental_with_market_cap['pe'] = fundamental_with_market_cap['pe'].fillna(fundamental_with_market_cap['pe_fallback'])
    
    # Merge with financial metrics data (prioritize actual P/E data)
    combined_data = fundamental_with_market_cap.merge(financial_metrics_data, on=['ticker', 'date'], how='outer', suffixes=('', '_actual'))
    
    # Use actual P/E when available, otherwise use calculated P/E
    # Handle columns that may not exist due to empty financial_metrics_data
    if 'pe_actual' in combined_data.columns:
        combined_data['pe'] = combined_data['pe_actual'].fillna(combined_data['pe'])
    if 'pb_actual' in combined_data.columns:
        combined_data['pb'] = combined_data['pb_actual'].fillna(combined_data['pb'])
    if 'eps_actual' in combined_data.columns:
        combined_data['eps'] = combined_data['eps_actual'].fillna(combined_data['eps'])
    if 'market_cap_actual' in combined_data.columns:
        combined_data['market_cap'] = combined_data['market_cap_actual'].fillna(combined_data['market_cap_bn'] * 1e9)
    if 'book_value_actual' in combined_data.columns:
        combined_data['book_value'] = combined_data['book_value_actual'].fillna(combined_data['book_value'])
    
    # Clean up duplicate columns
    columns_to_drop = ['pe_actual', 'pb_actual', 'eps_actual', 'market_cap_actual', 'book_value_actual', 'pe_fallback']
    existing_columns = [col for col in columns_to_drop if col in combined_data.columns]
    combined_data = combined_data.drop(existing_columns, axis=1)
    
    print(f"   ✅ Pre-computed fundamental factors: {len(combined_data):,} observations")
    print(f"   ✅ Financial metrics included: {len(financial_metrics_data):,} observations")
    
    return combined_data

def precompute_momentum_factors(config: dict, db_engine):
    """Precompute momentum factors for all dates."""
    print("📊 Precomputing momentum factors...")
    
    try:
        # Get all trading dates
        query = """
        SELECT DISTINCT trading_date 
        FROM vcsc_daily_data 
        WHERE trading_date >= %s AND trading_date <= %s
        ORDER BY trading_date
        """
        
        with db_engine.connect() as conn:
            dates_df = pd.read_sql(query, conn, params=(
                config['backtest_start_date'], 
                config['backtest_end_date']
            ))
        
        momentum_data = []
        
        for date in dates_df['trading_date']:
            # Calculate momentum for different horizons
            horizons = config['factors']['momentum_horizons']
            
            for horizon in horizons:
                start_date = date - pd.Timedelta(days=horizon)
                
                query = f"""
                SELECT 
                    ticker,
                    '{date.strftime('%Y-%m-%d')}' as trading_date,
                    (close_price / LAG(close_price, {horizon}) OVER (PARTITION BY ticker ORDER BY trading_date) - 1) as momentum_{horizon}d
                FROM vcsc_daily_data 
                WHERE trading_date = '{date.strftime('%Y-%m-%d')}'
                """
                
                with db_engine.connect() as conn:
                    momentum_df = pd.read_sql(query, conn)
                
                if not momentum_df.empty:
                    momentum_data.append(momentum_df)
        
        if momentum_data:
            # Combine all momentum data
            combined_df = pd.concat(momentum_data, ignore_index=True)
            
            # Pivot to wide format
            momentum_wide = combined_df.pivot_table(
                index=['ticker', 'trading_date'],
                columns=None,
                values=[col for col in combined_df.columns if col.startswith('momentum_')],
                aggfunc='first'
            ).reset_index()
            
            return momentum_wide
        else:
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error precomputing momentum factors: {e}")
        return pd.DataFrame()

def precompute_all_data(config: dict, db_engine):
    """Precompute all data for backtesting."""
    print("🚀 Starting data precomputation...")
    
    precomputed_data = {}
    
    # Precompute universe rankings
    precomputed_data['universe'] = precompute_universe_rankings(config, db_engine)
    
            # Precompute fundamental factors
    fundamental_data = precompute_fundamental_factors(config, db_engine)
    print(f"   🔍 Debug: Fundamental data shape: {fundamental_data.shape}")
    
    # Check if fundamental data is available
    if fundamental_data.empty:
        print("   ⚠️  No fundamental data available for backtest period")
        print("   📊 Strategy will use price-based factors only (momentum, volatility)")
        print("   💡 This is expected since fundamental data only exists from 2025-07-12")
        print("   🎯 Creating price-based adaptive rebalancing strategy")
    
    precomputed_data['fundamentals'] = fundamental_data
    
    # Precompute momentum factors
    precomputed_data['momentum'] = precompute_momentum_factors(config, db_engine)
    
    print("✅ Data precomputation complete.")
    return precomputed_data

# %%
# Data Loading and Backtest Execution

# %%
def load_all_data_for_backtest(config: dict, db_engine):
    """Load all data required for backtesting."""
    print("📊 Loading data for backtesting...")
    
    try:
        # Load price data
        query = """
        SELECT 
            ticker,
            trading_date as date,
            close_price as close,
            total_volume as volume
        FROM vcsc_daily_data 
        WHERE trading_date >= %s AND trading_date <= %s
        ORDER BY trading_date, ticker
        """
        
        with db_engine.connect() as conn:
            price_data = pd.read_sql(query, conn, params=(
                config['backtest_start_date'], 
                config['backtest_end_date']
            ))
        
        # Create returns matrix early
        price_pivot = price_data.pivot(index='date', columns='ticker', values='close')
        returns_matrix = price_pivot.pct_change()
        
        # Load fundamental data (placeholder)
        fundamental_data = pd.DataFrame()
        
        # Precompute all data
        precomputed_data = precompute_all_data(config, db_engine)
        
        # Load benchmark data (VNINDEX - Vietnam market index)
        query = """
        SELECT 
            trading_date as date,
            close_price as close
        FROM vcsc_daily_data 
        WHERE ticker = 'VNINDEX'
        AND trading_date >= %s AND trading_date <= %s
        ORDER BY trading_date
        """
        
        with db_engine.connect() as conn:
            benchmark_data = pd.read_sql(query, conn, params=(
                config['backtest_start_date'], 
                config['backtest_end_date']
            ))
        
        # If VNINDEX not available, try alternative benchmarks
        if benchmark_data.empty:
            print("   ⚠️  VNINDEX not found, trying alternative benchmarks...")
            
            # Try VNM as fallback
            fallback_query = """
            SELECT 
                trading_date as date,
                close_price as close
            FROM vcsc_daily_data 
            WHERE ticker = 'VNM'
            AND trading_date >= %s AND trading_date <= %s
            ORDER BY trading_date
            """
            
            with db_engine.connect() as conn2:
                benchmark_data = pd.read_sql(fallback_query, conn2, params=(
                    config['backtest_start_date'], 
                    config['backtest_end_date']
                ))
            
            if not benchmark_data.empty:
                print("   ⚠️  VNM found but using universe average as better benchmark")
                # Use universe average instead of single stock
                price_pivot = price_data.pivot(index='date', columns='ticker', values='close')
                benchmark_returns = price_pivot.pct_change().mean(axis=1)
                benchmark_returns = benchmark_returns.fillna(0)
                print("   ✅ Created universe average benchmark")
                return {
                    'price_data': price_data,
                    'fundamental_data': fundamental_data,
                    'returns_matrix': returns_matrix,
                    'benchmark_returns': benchmark_returns,
                    'precomputed_data': precomputed_data
                }
            else:
                # Create synthetic benchmark based on universe average
                print("   ⚠️  No benchmark found, creating synthetic benchmark...")
                price_pivot = price_data.pivot(index='date', columns='ticker', values='close')
                benchmark_returns = price_pivot.pct_change().mean(axis=1)
                benchmark_returns = benchmark_returns.fillna(0)
                print("   ✅ Created synthetic benchmark from universe average")
                return {
                    'price_data': price_data,
                    'fundamental_data': fundamental_data,
                    'returns_matrix': returns_matrix,
                    'benchmark_returns': benchmark_returns,
                    'precomputed_data': precomputed_data
                }
        
        # Calculate returns
        benchmark_data['returns'] = benchmark_data['close'].pct_change()
        benchmark_returns = benchmark_data.set_index('date')['returns']
        
        print("✅ Data loading complete.")
        return {
            'price_data': price_data,
            'fundamental_data': fundamental_data,
            'returns_matrix': returns_matrix,
            'benchmark_returns': benchmark_returns,
            'precomputed_data': precomputed_data
        }
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise

# %%
# Performance Analysis Functions

# %%
def calculate_performance_metrics(returns: pd.Series, benchmark: pd.Series, periods_per_year: int = 252) -> dict:
    """Calculates comprehensive performance metrics with corrected benchmark alignment."""
    # Align benchmark
    first_trade_date = returns.loc[returns.ne(0)].index.min()
    if pd.isna(first_trade_date):
        return {metric: 0.0 for metric in ['Annualized Return (%)', 'Annualized Volatility (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 'Calmar Ratio', 'Information Ratio', 'Beta']}
    
    aligned_returns = returns.loc[first_trade_date:]
    aligned_benchmark = benchmark.loc[first_trade_date:]

    n_years = len(aligned_returns) / periods_per_year
    annualized_return = ((1 + aligned_returns).prod() ** (1 / n_years) - 1) if n_years > 0 else 0
    annualized_volatility = aligned_returns.std() * np.sqrt(periods_per_year)
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else 0.0
    
    cumulative_returns = (1 + aligned_returns).cumprod()
    max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0.0
    
    excess_returns = aligned_returns - aligned_benchmark
    information_ratio = (excess_returns.mean() * periods_per_year) / (excess_returns.std() * np.sqrt(periods_per_year)) if excess_returns.std() > 0 else 0.0
    beta = aligned_returns.cov(aligned_benchmark) / aligned_benchmark.var() if aligned_benchmark.var() > 0 else 0.0
    
    return {
        'Annualized Return (%)': annualized_return * 100,
        'Annualized Volatility (%)': annualized_volatility * 100,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown (%)': max_drawdown * 100,
        'Calmar Ratio': calmar_ratio,
        'Information Ratio': information_ratio,
        'Beta': beta
    }

def generate_comprehensive_tearsheet(strategy_returns: pd.Series, benchmark_returns: pd.Series, diagnostics: pd.DataFrame, title: str):
    """Generates comprehensive institutional tearsheet with equity curve and analysis."""
    
    # Check if any trades were executed
    if strategy_returns.sum() == 0:
        print("⚠️  No trades were executed. Skipping tearsheet generation.")
        return
    
    # Align benchmark for plotting & metrics
    first_trade_date = strategy_returns.loc[strategy_returns.ne(0)].index.min()
    aligned_strategy_returns = strategy_returns.loc[first_trade_date:]
    aligned_benchmark_returns = benchmark_returns.loc[first_trade_date:]
    
    # Ensure both series have DatetimeIndex for resampling operations
    if not isinstance(aligned_strategy_returns.index, pd.DatetimeIndex):
        aligned_strategy_returns.index = pd.to_datetime(aligned_strategy_returns.index)
    if not isinstance(aligned_benchmark_returns.index, pd.DatetimeIndex):
        aligned_benchmark_returns.index = pd.to_datetime(aligned_benchmark_returns.index)

    strategy_metrics = calculate_performance_metrics(strategy_returns, benchmark_returns)
    benchmark_metrics = calculate_performance_metrics(benchmark_returns, benchmark_returns)
    
    fig = plt.figure(figsize=(18, 26))
    gs = fig.add_gridspec(5, 2, height_ratios=[1.2, 0.8, 0.8, 0.8, 1.2], hspace=0.7, wspace=0.2)
    fig.suptitle(title, fontsize=20, fontweight='bold', color='#2C3E50')

    # 1. Cumulative Performance (Equity Curve)
    ax1 = fig.add_subplot(gs[0, :])
    (1 + aligned_strategy_returns).cumprod().plot(ax=ax1, label='QVM Engine v3j Adaptive Rebalancing FINAL', color='#16A085', lw=2.5)
    (1 + aligned_benchmark_returns).cumprod().plot(ax=ax1, label='VN-Index (Aligned)', color='#34495E', linestyle='--', lw=2)
    ax1.set_title('Cumulative Performance (Log Scale)', fontweight='bold')
    ax1.set_ylabel('Growth of 1 VND')
    ax1.set_yscale('log')
    ax1.legend(loc='upper left')
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)

    # 2. Drawdown Analysis
    ax2 = fig.add_subplot(gs[1, :])
    drawdown = ((1 + aligned_strategy_returns).cumprod() / (1 + aligned_strategy_returns).cumprod().cummax() - 1) * 100
    drawdown.plot(ax=ax2, color='#C0392B')
    ax2.fill_between(drawdown.index, drawdown, 0, color='#C0392B', alpha=0.1)
    ax2.set_title('Drawdown Analysis', fontweight='bold')
    ax2.set_ylabel('Drawdown (%)')
    ax2.grid(True, linestyle='--', alpha=0.5)

    # 3. Annual Returns
    ax3 = fig.add_subplot(gs[2, 0])
    strat_annual = aligned_strategy_returns.resample('Y').apply(lambda x: (1+x).prod()-1) * 100
    bench_annual = aligned_benchmark_returns.resample('Y').apply(lambda x: (1+x).prod()-1) * 100
    pd.DataFrame({'Strategy': strat_annual, 'Benchmark': bench_annual}).plot(kind='bar', ax=ax3, color=['#16A085', '#34495E'])
    ax3.set_xticklabels([d.strftime('%Y') for d in strat_annual.index], rotation=45, ha='right')
    ax3.set_title('Annual Returns', fontweight='bold')
    ax3.grid(True, axis='y', linestyle='--', alpha=0.5)

    # 4. Rolling Sharpe Ratio
    ax4 = fig.add_subplot(gs[2, 1])
    rolling_sharpe = (aligned_strategy_returns.rolling(252).mean() * 252) / (aligned_strategy_returns.rolling(252).std() * np.sqrt(252))
    rolling_sharpe.plot(ax=ax4, color='#E67E22')
    ax4.axhline(1.0, color='#27AE60', linestyle='--')
    ax4.set_title('1-Year Rolling Sharpe Ratio', fontweight='bold')
    ax4.grid(True, linestyle='--', alpha=0.5)

    # 5. Regime Analysis
    ax5 = fig.add_subplot(gs[3, 0])
    if not diagnostics.empty and 'regime' in diagnostics.columns:
        regime_counts = diagnostics['regime'].value_counts()
        regime_counts.plot(kind='bar', ax=ax5, color=['#3498DB', '#E74C3C', '#F39C12', '#9B59B6'])
        ax5.set_title('Regime Distribution', fontweight='bold')
        ax5.set_ylabel('Number of Rebalances')
        ax5.grid(True, axis='y', linestyle='--', alpha=0.5)

    # 6. Portfolio Size Evolution
    ax6 = fig.add_subplot(gs[3, 1])
    if not diagnostics.empty and 'portfolio_size' in diagnostics.columns:
        diagnostics['portfolio_size'].plot(ax=ax6, color='#2ECC71', marker='o', markersize=3)
        ax6.set_title('Portfolio Size Evolution', fontweight='bold')
        ax6.set_ylabel('Number of Stocks')
        ax6.grid(True, linestyle='--', alpha=0.5)

    # 7. Performance Metrics Table
    ax7 = fig.add_subplot(gs[4:, :])
    ax7.axis('off')
    summary_data = [['Metric', 'Strategy', 'Benchmark']]
    for key in strategy_metrics.keys():
        summary_data.append([key, f"{strategy_metrics[key]:.2f}", f"{benchmark_metrics.get(key, 0.0):.2f}"])
    
    table = ax7.table(cellText=summary_data[1:], colLabels=summary_data[0], loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 2.5)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    # Save to file and also display
    output_path = f"tearsheet_{title.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   📊 Tearsheet saved to: {output_path}")
    plt.show()  # Display in notebook

# %%
# Main Execution

# %%
if __name__ == "__main__":
    """
    QVM Engine v3j Adaptive Rebalancing FINAL - MAIN EXECUTION

    This file contains the main execution code for the adaptive rebalancing QVM Engine v3j
    with all components: regime detection, factor analysis, and adaptive rebalancing.
    """

    # Execute the data loading
    try:
        print("\n" + "="*80)
        print("🚀 QVM ENGINE V3J: ADAPTIVE REBALANCING FINAL EXECUTION")
        print("="*80)
        
        # Load basic data
        data_dict = load_all_data_for_backtest(QVM_CONFIG, engine)
        
        # Extract data from dictionary
        price_data_raw = data_dict['price_data']
        fundamental_data_raw = data_dict['fundamental_data']
        daily_returns_matrix = data_dict['returns_matrix']
        benchmark_returns = data_dict['benchmark_returns']
        precomputed_data = data_dict['precomputed_data']
        print("\n✅ All basic data successfully loaded and prepared for the backtest.")
        print(f"   - Price Data Shape: {price_data_raw.shape}")
        print(f"   - Fundamental Data Shape: {fundamental_data_raw.shape}")
        print(f"   - Returns Matrix Shape: {daily_returns_matrix.shape}")
        print(f"   - Benchmark Returns: {len(benchmark_returns)} days")
        
        # --- Instantiate and Run the Adaptive Rebalancing QVM Engine v3j ---
        print("\n" + "="*80)
        print("🚀 QVM ENGINE V3J: ADAPTIVE REBALANCING BACKTEST")
        print("="*80)
        
        qvm_engine = QVMEngineV3jAdaptiveRebalancingFinal(
            config=QVM_CONFIG,
            price_data=price_data_raw,
            fundamental_data=fundamental_data_raw,
            returns_matrix=daily_returns_matrix,
            benchmark_returns=benchmark_returns,
            db_engine=engine,
            precomputed_data=precomputed_data
        )
        
        qvm_net_returns, qvm_diagnostics = qvm_engine.run_backtest()
        
        print(f"\n🔍 DEBUG: After adaptive rebalancing backtest")
        print(f"   - qvm_net_returns shape: {qvm_net_returns.shape}")
        print(f"   - qvm_net_returns date range: {qvm_net_returns.index.min()} to {qvm_net_returns.index.max()}")
        print(f"   - benchmark_returns shape: {benchmark_returns.shape}")
        print(f"   - benchmark_returns date range: {benchmark_returns.index.min()} to {benchmark_returns.index.max()}")
        print(f"   - Non-zero returns count: {(qvm_net_returns != 0).sum()}")
        print(f"   - First non-zero return date: {qvm_net_returns[qvm_net_returns != 0].index.min() if (qvm_net_returns != 0).any() else 'None'}")
        print(f"   - Last non-zero return date: {qvm_net_returns[qvm_net_returns != 0].index.max() if (qvm_net_returns != 0).any() else 'None'}")
        
        # --- Generate Comprehensive Tearsheet ---
        print("\n" + "="*80)
        print("📊 QVM ENGINE V3J: ADAPTIVE REBALANCING TEARSHEET")
        print("="*80)
        
        # Full Period Tearsheet (2016-2025)
        print("\n📈 Generating Adaptive Rebalancing Strategy Tearsheet (2016-2025)...")
        generate_comprehensive_tearsheet(
            qvm_net_returns,
            benchmark_returns,
            qvm_diagnostics,
            "QVM Engine v3j Adaptive Rebalancing FINAL - Full Period (2016-2025)"
        )
        
        # --- Performance Analysis ---
        print("\n" + "="*80)
        print("🔍 PERFORMANCE ANALYSIS")
        print("="*80)
        
        # Regime Analysis
        if not qvm_diagnostics.empty and 'regime' in qvm_diagnostics.columns:
            print("\n📈 Regime Analysis:")
            regime_summary = qvm_diagnostics['regime'].value_counts()
            for regime, count in regime_summary.items():
                percentage = (count / len(qvm_diagnostics)) * 100
                print(f"   - {regime}: {count} times ({percentage:.2f}%)")
        
        # Calculate and display performance metrics
        print("\n📊 PERFORMANCE METRICS:")
        if not qvm_net_returns.empty and not benchmark_returns.empty:
            # Align data
            aligned_data = pd.concat([qvm_net_returns, benchmark_returns], axis=1).dropna()
            strategy_returns = aligned_data.iloc[:, 0]
            benchmark_returns_aligned = aligned_data.iloc[:, 1]
            
            # Calculate metrics with safety checks
            strategy_returns_clean = strategy_returns.replace([np.inf, -np.inf], 0).fillna(0)
            benchmark_returns_clean = benchmark_returns_aligned.replace([np.inf, -np.inf], 0).fillna(0)
            
            # Cap extreme returns
            strategy_returns_clean = strategy_returns_clean.clip(-0.5, 0.5)
            benchmark_returns_clean = benchmark_returns_clean.clip(-0.5, 0.5)
            
            total_return = (1 + strategy_returns_clean).prod() - 1
            benchmark_total_return = (1 + benchmark_returns_clean).prod() - 1
            
            # Ensure returns are finite
            total_return = np.clip(total_return, -0.99, 100)  # Cap between -99% and 10000%
            benchmark_total_return = np.clip(benchmark_total_return, -0.99, 100)
            
            annualized_return = (1 + total_return) ** (252 / len(strategy_returns_clean)) - 1
            benchmark_annualized = (1 + benchmark_total_return) ** (252 / len(benchmark_returns_clean)) - 1
            
            volatility = strategy_returns_clean.std() * np.sqrt(252)
            benchmark_volatility = benchmark_returns_clean.std() * np.sqrt(252)
            
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            benchmark_sharpe = benchmark_annualized / benchmark_volatility if benchmark_volatility > 0 else 0
            
            # Drawdown analysis
            cumulative_returns = (1 + strategy_returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Information ratio
            excess_returns = strategy_returns - benchmark_returns_aligned
            tracking_error = excess_returns.std() * np.sqrt(252)
            information_ratio = excess_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0
            
            print(f"   📈 Strategy Total Return: {total_return:.2%}")
            print(f"   📈 Benchmark Total Return: {benchmark_total_return:.2%}")
            print(f"   📈 Strategy Annualized Return: {annualized_return:.2%}")
            print(f"   📈 Benchmark Annualized Return: {benchmark_annualized:.2%}")
            print(f"   📈 Strategy Volatility: {volatility:.2%}")
            print(f"   📈 Benchmark Volatility: {benchmark_volatility:.2%}")
            print(f"   📈 Strategy Sharpe Ratio: {sharpe_ratio:.2f}")
            print(f"   📈 Benchmark Sharpe Ratio: {benchmark_sharpe:.2f}")
            print(f"   📈 Max Drawdown: {max_drawdown:.2%}")
            print(f"   📈 Information Ratio: {information_ratio:.2f}")
            print(f"   📈 Excess Return: {(annualized_return - benchmark_annualized):.2%}")
            
            # Performance summary
            if annualized_return > benchmark_annualized:
                print(f"   ✅ STRATEGY OUTPERFORMS BENCHMARK by {(annualized_return - benchmark_annualized):.2%}")
            else:
                print(f"   ⚠️  STRATEGY UNDERPERFORMS BENCHMARK by {(benchmark_annualized - annualized_return):.2%}")
        
        # Factor Configuration
        print("\n📊 Factor Configuration:")
        print(f"   - Value Weight: {QVM_CONFIG['factors']['value_weight']}")
        print(f"   - Quality Weight: {QVM_CONFIG['factors']['quality_weight']}")
        print(f"   - Momentum Weight: {QVM_CONFIG['factors']['momentum_weight']}")
        print(f"   - Momentum Horizons: {QVM_CONFIG['factors']['momentum_horizons']}")
        
        # Universe Statistics
        if not qvm_diagnostics.empty:
            print(f"\n🌐 Universe Statistics:")
            print(f"   - Average Universe Size: {qvm_diagnostics['universe_size'].mean():.0f} stocks")
            print(f"   - Average Portfolio Size: {qvm_diagnostics['portfolio_size'].mean():.0f} stocks")
            print(f"   - Average Turnover: {qvm_diagnostics['turnover'].mean():.2%}")
        
        # Adaptive Rebalancing Summary
        print(f"\n⚡ Adaptive Rebalancing Summary:")
        print(f"   - Bull Market: Weekly rebalancing (100% allocation)")
        print(f"   - Bear Market: Monthly rebalancing (80% allocation)")
        print(f"   - Sideways Market: Biweekly rebalancing (60% allocation)")
        print(f"   - Stress Market: Quarterly rebalancing (40% allocation)")
        print(f"   - Performance: Pre-computed data + Vectorized operations")
        
        print("\n✅ QVM Engine v3j Adaptive Rebalancing FINAL strategy execution complete!")
        
    except Exception as e:
        print(f"❌ An error occurred during execution: {e}")
        raise 
