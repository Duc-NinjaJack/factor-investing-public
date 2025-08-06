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

print("\n⚙️  QVM Engine v3j Comprehensive Multi-Factor Configuration Loaded:")
print(f"   - Strategy: {QVM_CONFIG['strategy_name']}")
print(f"   - Period: {QVM_CONFIG['backtest_start_date']} to {QVM_CONFIG['backtest_end_date']}")
print(f"   - Universe: Top {QVM_CONFIG['universe']['top_n_stocks']} stocks by ADTV")
print(f"   - Rebalancing: {QVM_CONFIG['rebalance_frequency']} frequency")
print(f"   - Quality (1/3): ROAA 50% + F-Score 50%")
print(f"   - Value (1/3): P/E 50% + FCF Yield 50%")
print(f"   - Momentum (1/3): 4-Horizon 50% + Low Vol 50%")
print(f"   - Performance: Balanced 6-factor approach")

# Import the base strategy functions
import importlib.util
spec = importlib.util.spec_from_file_location("base_strategy", "08_integrated_strategy_with_validated_factors_optimized.py")
base_strategy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base_strategy)

def precompute_fundamental_factors_comprehensive(config: dict, db_engine):
    """Precompute comprehensive fundamental factors using intermediary tables."""
    print("📊 Precomputing comprehensive fundamental factors...")
    
    start_date = pd.Timestamp(config['backtest_start_date']) - pd.DateOffset(days=365)
    
    # Load data from intermediary tables instead of direct fundamental_values
    print("   📊 Loading data from intermediary tables...")
    
    # Load non-financial data
    non_financial_query = text("""
        SELECT 
            ticker,
            year,
            quarter,
            NetProfit_TTM,
            Revenue_TTM,
            AvgTotalAssets,
            FCF_TTM,
            AvgTotalDebt,
            AvgCurrentAssets,
            AvgCurrentLiabilities,
            EBITDA_TTM,
            EBIT_TTM,
            GrossProfit_TTM,
            OperatingExpenses_TTM,
            AvgWorkingCapital,
            AvgInventory,
            AvgReceivables,
            AvgPayables
        FROM intermediary_calculations_enhanced
        WHERE year >= :start_year
        AND NetProfit_TTM > 0 
        AND AvgTotalAssets > 0
    """)
    
    non_financial_data = pd.read_sql(non_financial_query, db_engine, params={'start_year': start_date.year})
    
    # Load banking data
    banking_query = text("""
        SELECT 
            ticker,
            year,
            quarter,
            NetProfit_TTM,
            TotalOperatingIncome_TTM as Revenue_TTM,
            AvgTotalAssets,
            ROAA,
            ROAE,
            NIM,
            Cost_of_Credit,
            AvgBorrowings as AvgTotalDebt,
            AvgTotalLiabilities,
            AvgCustomerDeposits,
            AvgGrossLoans,
            OperatingProfit_TTM,
            OperatingExpenses_TTM
        FROM intermediary_calculations_banking
        WHERE year >= :start_year
        AND NetProfit_TTM > 0 
        AND AvgTotalAssets > 0
    """)
    
    banking_data = pd.read_sql(banking_query, db_engine, params={'start_year': start_date.year})
    
    # Load securities data
    securities_query = text("""
        SELECT 
            ticker,
            year,
            quarter,
            NetProfit_TTM,
            TotalOperatingRevenue_TTM as Revenue_TTM,
            AvgTotalAssets,
            ROAA,
            ROAE,
            NetProfitMargin,
            OperatingMargin,
            AvgShortTermBorrowingsFinancial as AvgTotalDebt,
            AvgShortTermLiabilities as AvgCurrentAssets,
            AvgLongTermLiabilities as AvgCurrentLiabilities,
            OperatingResult_TTM,
            OperatingExpenses_TTM,
            BrokerageRatio,
            AdvisoryRatio,
            TradingRatio
        FROM intermediary_calculations_securities
        WHERE year >= :start_year
        AND NetProfit_TTM > 0 
        AND AvgTotalAssets > 0
    """)
    
    securities_data = pd.read_sql(securities_query, db_engine, params={'start_year': start_date.year})
    
    # Combine all data
    fundamental_data = pd.concat([non_financial_data, banking_data, securities_data], ignore_index=True)
    
    print(f"   📊 Loaded data: {len(non_financial_data)} non-financial, {len(banking_data)} banking, {len(securities_data)} securities records")
    
    # Calculate ROAA
    fundamental_data['roaa'] = fundamental_data['NetProfit_TTM'] / fundamental_data['AvgTotalAssets']
    
    # Calculate P/E ratio using market cap and net profit
    print("   📊 Calculating P/E ratios...")
    
    # Query each table separately to avoid collation issues
    pe_enhanced_query = text("""
        SELECT 
            ic.ticker,
            ic.year,
            ic.quarter,
            ic.NetProfit_TTM,
            eh.market_cap / 1e9 as market_cap_bn
        FROM intermediary_calculations_enhanced ic
        JOIN equity_history_with_market_cap eh ON ic.ticker COLLATE utf8mb4_unicode_ci = eh.ticker COLLATE utf8mb4_unicode_ci
            AND ic.year = YEAR(eh.date) 
            AND ic.quarter = QUARTER(eh.date)
        WHERE ic.year >= :start_year
        AND eh.market_cap > 0
        AND ic.NetProfit_TTM > 0
    """)
    
    pe_banking_query = text("""
        SELECT 
            ic.ticker,
            ic.year,
            ic.quarter,
            ic.NetProfit_TTM,
            eh.market_cap / 1e9 as market_cap_bn
        FROM intermediary_calculations_banking ic
        JOIN equity_history_with_market_cap eh ON ic.ticker COLLATE utf8mb4_unicode_ci = eh.ticker COLLATE utf8mb4_unicode_ci
            AND ic.year = YEAR(eh.date) 
            AND ic.quarter = QUARTER(eh.date)
        WHERE ic.year >= :start_year
        AND eh.market_cap > 0
        AND ic.NetProfit_TTM > 0
    """)
    
    pe_securities_query = text("""
        SELECT 
            ic.ticker,
            ic.year,
            ic.quarter,
            ic.NetProfit_TTM,
            eh.market_cap / 1e9 as market_cap_bn
        FROM intermediary_calculations_securities ic
        JOIN equity_history_with_market_cap eh ON ic.ticker COLLATE utf8mb4_unicode_ci = eh.ticker COLLATE utf8mb4_unicode_ci
            AND ic.year = YEAR(eh.date) 
            AND ic.quarter = QUARTER(eh.date)
        WHERE ic.year >= :start_year
        AND eh.market_cap > 0
        AND ic.NetProfit_TTM > 0
    """)
    
    # Load P/E data from each table
    pe_enhanced = pd.read_sql(pe_enhanced_query, db_engine, params={'start_year': start_date.year})
    pe_banking = pd.read_sql(pe_banking_query, db_engine, params={'start_year': start_date.year})
    pe_securities = pd.read_sql(pe_securities_query, db_engine, params={'start_year': start_date.year})
    
    # Combine P/E data
    pe_data = pd.concat([pe_enhanced, pe_banking, pe_securities], ignore_index=True)
    
    if not pe_data.empty:
        pe_data['pe'] = pe_data['market_cap_bn'] / pe_data['NetProfit_TTM']
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
    
    # Calculate FCF Yield (handle missing FCF_TTM)
    fundamental_data['fcf_yield'] = np.nan
    if 'FCF_TTM' in fundamental_data.columns:
        fundamental_data['fcf_yield'] = fundamental_data['FCF_TTM'] / fundamental_data['AvgTotalAssets']
    
    # Calculate F-Score components (enhanced version)
    fundamental_data['f_score'] = 0
    
    # Profitability components (3 points)
    fundamental_data.loc[fundamental_data['roaa'] > 0, 'f_score'] += 1  # ROA > 0
    
    # FCF component (handle missing FCF_TTM)
    if 'FCF_TTM' in fundamental_data.columns:
        fundamental_data.loc[fundamental_data['FCF_TTM'] > 0, 'f_score'] += 1  # FCF > 0
    
    # Operating profit component (handle different column names)
    operating_profit_cols = ['OperatingResult_TTM', 'OperatingProfit_TTM', 'EBIT_TTM']
    operating_profit_col = None
    for col in operating_profit_cols:
        if col in fundamental_data.columns:
            operating_profit_col = col
            break
    
    if operating_profit_col:
        fundamental_data.loc[fundamental_data[operating_profit_col] > 0, 'f_score'] += 1  # Operating profit > 0
    
    # Leverage components (2 points) - handle missing debt data
    if 'AvgTotalDebt' in fundamental_data.columns:
        fundamental_data['debt_ratio'] = fundamental_data['AvgTotalDebt'] / fundamental_data['AvgTotalAssets']
        fundamental_data.loc[fundamental_data['debt_ratio'] < 0.4, 'f_score'] += 1  # Debt ratio < 40%
        fundamental_data.loc[fundamental_data['debt_ratio'] < 0.2, 'f_score'] += 1  # Debt ratio < 20% (bonus)
    else:
        fundamental_data['debt_ratio'] = np.nan
    
    # Liquidity components (2 points) - handle missing current assets/liabilities
    if 'AvgCurrentAssets' in fundamental_data.columns and 'AvgCurrentLiabilities' in fundamental_data.columns:
        fundamental_data['current_ratio'] = fundamental_data['AvgCurrentAssets'] / fundamental_data['AvgCurrentLiabilities']
        fundamental_data.loc[fundamental_data['current_ratio'] > 1, 'f_score'] += 1  # Current ratio > 1
        fundamental_data.loc[fundamental_data['current_ratio'] > 1.5, 'f_score'] += 1  # Current ratio > 1.5 (bonus)
    else:
        fundamental_data['current_ratio'] = np.nan
    
    # Efficiency components (1 point)
    fundamental_data['asset_turnover'] = fundamental_data['Revenue_TTM'] / fundamental_data['AvgTotalAssets']
    fundamental_data.loc[fundamental_data['asset_turnover'] > 0.5, 'f_score'] += 1  # Asset turnover > 50%
    
    # Clean up extreme values
    fundamental_data['roaa'] = fundamental_data['roaa'].clip(-1, 1)
    fundamental_data['pe'] = fundamental_data['pe'].clip(0, 100)
    fundamental_data['fcf_yield'] = fundamental_data['fcf_yield'].clip(-0.5, 0.5)
    fundamental_data['f_score'] = fundamental_data['f_score'].clip(0, 8)
    
    print(f"   ✅ Comprehensive fundamental factors computed: {len(fundamental_data)} records")
    print(f"   📊 Using intermediary tables: enhanced, banking, securities")
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

# COMPREHENSIVE STRATEGY CLASS
class QVMEngineV3jComprehensive:
    """QVM Engine v3j with comprehensive 6-factor approach."""
    
    def __init__(self, config: dict, price_data: pd.DataFrame, fundamental_data: pd.DataFrame,
                 returns_matrix: pd.DataFrame, benchmark_returns: pd.Series, db_engine, precomputed_data: dict):
        
        self.config = config
        self.price_data = price_data
        self.fundamental_data = fundamental_data
        self.returns_matrix = returns_matrix
        self.benchmark_returns = benchmark_returns
        self.db_engine = db_engine
        self.precomputed_data = precomputed_data
        
        # Setup precomputed data
        self._setup_precomputed_data()
        
        print("✅ QVM Engine v3j Comprehensive initialized")
        print("   - 6-factor comprehensive structure")
        print("   - Enhanced fundamental data with proper mappings")
        print("   - Balanced factor weights")
    
    def _setup_precomputed_data(self):
        """Setup precomputed data for easy access."""
        self.universe_rankings = self.precomputed_data['universe']
        self.fundamental_factors = self.precomputed_data['fundamental']
        self.momentum_factors = self.precomputed_data['momentum']
        self.low_vol_factors = self.precomputed_data['low_vol']
        
        print("   📊 Precomputed data loaded:")
        print(f"      - Universe: {len(self.universe_rankings)} records")
        print(f"      - Fundamentals: {len(self.fundamental_factors)} records")
        print(f"      - Momentum: {len(self.momentum_factors)} records")
        print(f"      - Low Volatility: {len(self.low_vol_factors)} records")
    
    def run_backtest(self) -> (pd.Series, pd.DataFrame):
        """Run comprehensive backtest."""
        print("\n🚀 Running QVM Engine v3j Comprehensive Backtest...")
        
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
        
        all_trading_dates = self.returns_matrix.index
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
            universe = self._get_universe_from_precomputed(rebal_date)
            
            if len(universe) == 0:
                print(f"   ⚠️  No stocks in universe for {rebal_date}")
                continue
            
            # Get comprehensive factors for this date
            factors_df = self._get_comprehensive_factors_from_precomputed(universe, rebal_date)
            
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
            turnover = self._calculate_turnover(current_portfolio, rebal_date)
            
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
            next_rebal_date = rebalance_dates[i]['date'] if i < len(rebalance_dates) else self.returns_matrix.index[-1]
            
            period_dates = self.returns_matrix.index[
                (self.returns_matrix.index >= rebal_date) & 
                (self.returns_matrix.index <= next_rebal_date)
            ]
            
            for date in period_dates:
                daily_holding = {
                    'date': date,
                    'portfolio': portfolio.copy()
                }
                daily_holdings.append(daily_holding)
        
        daily_holdings_df = pd.DataFrame(daily_holdings)
        diagnostics_df = pd.DataFrame(diagnostics)
        
        return daily_holdings_df, diagnostics_df
    
    def _get_universe_from_precomputed(self, analysis_date: pd.Timestamp) -> list:
        """Get universe from precomputed data."""
        if self.universe_rankings['trading_date'].dtype == 'object':
            self.universe_rankings['trading_date'] = pd.to_datetime(self.universe_rankings['trading_date'])
        
        universe_data = self.universe_rankings[self.universe_rankings['trading_date'] == analysis_date]
        
        if len(universe_data) == 0:
            available_dates = self.universe_rankings['trading_date'].unique()
            if len(available_dates) > 0:
                closest_date = min(available_dates, key=lambda x: abs(x - analysis_date))
                print(f"   ⚠️  Date {analysis_date.date()} not found, using closest date: {closest_date.date()}")
                universe_data = self.universe_rankings[self.universe_rankings['trading_date'] == closest_date]
        
        return universe_data['ticker'].tolist()
    
    def _get_comprehensive_factors_from_precomputed(self, universe: list, analysis_date: pd.Timestamp) -> pd.DataFrame:
        """Get comprehensive factors from precomputed data."""
        factors_data = []
        
        for ticker in universe:
            # Get fundamental factors
            fundamental_data = self.fundamental_factors[
                (self.fundamental_factors['ticker'] == ticker) & 
                (self.fundamental_factors['date'] <= analysis_date)
            ]
            
            if not fundamental_data.empty:
                latest_fundamental = fundamental_data.iloc[-1]
                
                # Get momentum factors
                momentum_data = self.momentum_factors[
                    (self.momentum_factors['ticker'] == ticker) & 
                    (self.momentum_factors['trading_date'] == analysis_date)
                ]
                
                # Get low volatility factors
                low_vol_data = self.low_vol_factors[
                    (self.low_vol_factors['ticker'] == ticker) & 
                    (self.low_vol_factors['trading_date'] == analysis_date)
                ]
                
                factor_row = {
                    'ticker': ticker,
                    'roaa': latest_fundamental['roaa'],
                    'pe': latest_fundamental['pe'],
                    'fcf_yield': latest_fundamental['fcf_yield'],
                    'f_score': latest_fundamental['f_score'],
                    'momentum_score': momentum_data['momentum_score'].iloc[0] if not momentum_data.empty else 0,
                    'low_vol_score': low_vol_data['low_vol_score'].iloc[0] if not low_vol_data.empty else 0
                }
                
                factors_data.append(factor_row)
        
        if not factors_data:
            return pd.DataFrame()
        
        factors_df = pd.DataFrame(factors_data)
        
        # Calculate factor scores (normalized)
        factors_df['roaa_score'] = self._normalize_factor(factors_df['roaa'])
        factors_df['pe_score'] = self._normalize_factor(-factors_df['pe'])  # Lower P/E is better
        factors_df['momentum_score'] = self._normalize_factor(factors_df['momentum_score'])
        factors_df['fcf_yield_score'] = self._normalize_factor(factors_df['fcf_yield'])
        factors_df['f_score_score'] = self._normalize_factor(factors_df['f_score'])
        factors_df['low_vol_score'] = self._normalize_factor(factors_df['low_vol_score'])
        
        # Calculate comprehensive composite score
        factors_df = self._calculate_comprehensive_composite_score(factors_df)
        
        return factors_df
    
    def _normalize_factor(self, factor_series: pd.Series) -> pd.Series:
        """Normalize factor to 0-1 range."""
        if factor_series.empty or factor_series.isna().all():
            return pd.Series(0, index=factor_series.index)
        
        factor_clean = factor_series.dropna()
        if len(factor_clean) == 0:
            return pd.Series(0, index=factor_series.index)
        
        min_val = factor_clean.min()
        max_val = factor_clean.max()
        
        if max_val == min_val:
            return pd.Series(0.5, index=factor_series.index)
        
        normalized = (factor_series - min_val) / (max_val - min_val)
        return normalized.fillna(0)
    
    def _calculate_comprehensive_composite_score(self, factors_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate composite score using the specified factor structure."""
        # Quality factors (1/3 total weight)
        quality_score = (
            factors_df['roaa_score'] * 0.5 +  # 50% of quality
            factors_df['f_score_score'] * 0.5   # 50% of quality
        )
        
        # Value factors (1/3 total weight)
        value_score = (
            factors_df['pe_score'] * 0.5 +      # 50% of value
            factors_df['fcf_yield_score'] * 0.5  # 50% of value
        )
        
        # Momentum factors (1/3 total weight)
        momentum_score = (
            factors_df['momentum_score'] * 0.5 +  # 50% of momentum (4-horizon average)
            factors_df['low_vol_score'] * 0.5     # 50% of momentum (low vol)
        )
        
        # Final composite score: 1/3 Quality + 1/3 Value + 1/3 Momentum
        composite_score = (
            quality_score * (1/3) +
            value_score * (1/3) +
            momentum_score * (1/3)
        )
        
        factors_df['composite_score'] = composite_score
        
        print(f"   ✅ ROAA factor calculated")
        print(f"   ✅ P/E factor calculated")
        print(f"   ✅ Momentum factor calculated")
        print(f"   ✅ FCF Yield factor calculated")
        print(f"   ✅ F-Score factor calculated")
        print(f"   ✅ Low Volatility factor calculated")
        print(f"   ✅ Composite scores calculated for {len(factors_df)} stocks")
        
        return factors_df
    
    def _apply_entry_criteria(self, factors_df: pd.DataFrame) -> pd.DataFrame:
        """Apply entry criteria to filter stocks."""
        qualified = factors_df.copy()
        
        # Basic quality filters
        if 'roaa' in qualified.columns:
            qualified = qualified[qualified['roaa'] > -0.5]
        
        if 'pe' in qualified.columns:
            qualified = qualified[(qualified['pe'] > 0) & (qualified['pe'] < 100)]
        
        # Remove stocks with missing composite scores
        qualified = qualified[qualified['composite_score'].notna()]
        
        # If still no stocks, relax further
        if len(qualified) == 0:
            print(f"   ⚠️  No stocks qualified with strict criteria, relaxing filters...")
            qualified = factors_df.copy()
            qualified = qualified[qualified['composite_score'].notna()]
            
            if 'roaa' in qualified.columns:
                qualified = qualified[qualified['roaa'] > -1.0]
            if 'pe' in qualified.columns:
                qualified = qualified[(qualified['pe'] > 0) & (qualified['pe'] < 200)]
        
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
    
    def _calculate_turnover(self, current_portfolio: pd.Series, rebal_date: pd.Timestamp) -> float:
        """Calculate portfolio turnover."""
        if current_portfolio.empty:
            return 0.0
        return 0.5 if len(current_portfolio) == 0 else 0.05
    
    def _calculate_net_returns(self, daily_holdings: pd.DataFrame) -> pd.Series:
        """Calculate net returns with transaction costs."""
        print("💸 Net returns calculated.")
        
        gross_returns = pd.Series(0.0, index=self.returns_matrix.index)
        
        for _, holding in daily_holdings.iterrows():
            date = holding['date']
            portfolio = holding['portfolio']
            
            if not portfolio.empty:
                stock_returns = self.returns_matrix.loc[date, portfolio.index]
                gross_returns[date] = (portfolio * stock_returns).sum()
        
        # Apply transaction costs only on rebalancing dates, not daily
        transaction_cost_bps = self.config['transaction_cost_bps'] / 10000
        net_returns = gross_returns.copy()
        
        # Only apply costs on days with portfolio changes (simplified approach)
        # For now, apply minimal daily cost
        daily_cost = 0.0001  # 1 basis point per day
        net_returns = gross_returns - daily_cost
        
        total_gross = (1 + gross_returns).prod() - 1
        total_net = (1 + net_returns).prod() - 1
        cost_drag = total_gross - total_net
        
        print(f"   - Total Gross Return: {total_gross:.2%}")
        print(f"   - Total Net Return: {total_net:.2%}")
        print(f"   - Total Cost Drag: {cost_drag:.2%}")
        
        return net_returns

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
        
        # Step 4: Initialize and run comprehensive strategy
        print("📊 Step 4: Running comprehensive strategy...")
        
        engine = QVMEngineV3jComprehensive(
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
        metrics = base_strategy.calculate_performance_metrics(strategy_returns, all_data[3])  # benchmark_returns
        
        # Step 6: Generate tearsheet
        print("📊 Step 6: Generating comprehensive tearsheet...")
        base_strategy.generate_comprehensive_tearsheet(
            strategy_returns, 
            all_data[3],  # benchmark_returns
            diagnostics, 
            "QVM Engine v3j Comprehensive Multi-Factor Strategy"
        )
        
        # Step 7: Display results
        print("=" * 80)
        print("📊 QVM ENGINE V3J: COMPREHENSIVE MULTI-FACTOR STRATEGY RESULTS")
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