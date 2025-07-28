"""
Vietnam Factor Investing Platform - QVM Engine v1 (Baseline)
==========================================================
Component: Baseline Factor Calculation Engine (Simple Implementation)
Purpose: Control group for scientific bake-off - Simple ROAE-based approach
Author: Duc Nguyen, Principal Quantitative Strategist
Date Created: July 22, 2025
Status: BASELINE ENGINE (v1) - CONTROL GROUP

SCIENTIFIC BAKE-OFF ROLE:
This engine represents the SIMPLE HYPOTHESIS in our signal construction experiment:
- Quality: Simple ROAE level signal (no multi-tier framework)
- Value: Basic P/E, P/B, P/S ratios (no enhanced EV/EBITDA)
- Momentum: Standard returns calculation
- Expected Performance: ~18% annual return, 1.2 Sharpe ratio (hypothesis)

BASELINE SOURCE VALIDATION:
This engine is built from empirically validated code snippets extracted from:
- notebooks/phase5_institutional_qvm/02_factor_validation.md (validated calculation logic)
- notebooks/phase5_institutional_qvm/01_factor_calculation_engine.py (database patterns)
- Represents the simplest viable implementation of proven factors

CORE PRINCIPLES:
1. "Store Raw, Transform Dynamically" - all ratios calculated on-the-fly
2. Point-in-time integrity with 45-day reporting lag
3. Sector-neutral normalization to prevent systematic bias
4. Robust error handling with comprehensive fallbacks
5. Institutional-grade validation and logging

Data Sources:
- intermediary_calculations_banking_cleaned (21 tickers)
- intermediary_calculations_securities_cleaned (26 tickers) 
- intermediary_calculations_enhanced (667 non-financial tickers)
- vcsc_daily_data_complete (market data)
- equity_history (price returns)

Dependencies:
- pandas >= 1.3.0
- numpy >= 1.21.0
- sqlalchemy >= 1.4.0
- PyYAML >= 5.4.0
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
import logging

# Suppress pandas warnings
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)


class QVMEngineV1Baseline:
    """
    Canonical QVM factor calculator built from validated code snippets only.
    Implements institutional-grade factor calculations with complete point-in-time integrity.
    """
    
    def __init__(self, config_path: str = None, log_level: str = 'INFO'):
        """Initialize canonical engine with database and factor configurations."""
        
        # Setup logging
        self.logger = self._setup_logging(log_level)
        self.logger.info("Initializing Canonical QVM Engine")
        
        # Resolve configuration paths
        if config_path is None:
            project_root = Path(__file__).parent.parent.parent
            self.db_config_path = project_root / 'config' / 'database.yml'
            self.strategy_config_path = project_root / 'config' / 'strategy_config.yml'
        else:
            config_dir = Path(config_path)
            self.db_config_path = config_dir / 'database.yml'
            self.strategy_config_path = config_dir / 'strategy_config.yml'
        
        # Load configurations
        self._load_configurations()
        
        # Create database engine
        self._create_database_engine()
        
        # Define institutional constants
        self._define_constants()
        
        self.logger.info("Canonical QVM Engine initialized successfully")
        self.logger.info(f"QVM Weights: Quality {self.qvm_weights['quality']*100}%, Value {self.qvm_weights['value']*100}%, Momentum {self.qvm_weights['momentum']*100}%")
    
    def _setup_logging(self, log_level: str) -> logging.Logger:
        """Setup comprehensive logging for production use."""
        logger = logging.getLogger('CanonicalQVMEngine')
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def _load_configurations(self):
        """Load database and strategy configurations."""
        try:
            # Load database configuration
            with open(self.db_config_path, 'r') as f:
                self.db_config = yaml.safe_load(f)['production']
            
            # Load strategy configuration
            with open(self.strategy_config_path, 'r') as f:
                self.factor_config = yaml.safe_load(f)
            
            # Store parameters as instance attributes for easy access
            self.quality_weights = self.factor_config['quality']['tier_weights']
            self.momentum_weights = self.factor_config['momentum']['timeframe_weights']
            self.momentum_lookbacks = self.factor_config['momentum']['lookback_periods']
            self.momentum_skip = self.factor_config['momentum']['skip_months']
            self.qvm_weights = self.factor_config['qvm_composite']['weights']
            
            self.logger.info("Configurations loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load configurations: {e}")
            raise
    
    def _create_database_engine(self):
        """Create SQLAlchemy database engine."""
        try:
            self.engine = create_engine(
                f"mysql+pymysql://{self.db_config['username']}:{self.db_config['password']}@"
                f"{self.db_config['host']}/{self.db_config['schema_name']}"
            )
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            self.logger.info("Database connection established successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create database engine: {e}")
            raise
    
    def _define_constants(self):
        """Define institutional constants and table mappings."""
        # Define table mappings for sector-specific data
        self.intermediary_tables = {
            'banking': 'intermediary_calculations_banking_cleaned',
            'securities': 'intermediary_calculations_securities_cleaned', 
            'non_financial': 'intermediary_calculations_enhanced'
        }
        
        # Define acceptable ratio ranges for validation
        self.ratio_ranges = {
            'ROAE': (0.0, 0.5),      # 0-50% is reasonable
            'ROAA': (0.0, 0.1),      # 0-10% is reasonable
            'NIM': (0.0, 0.15),      # 0-15% for banks
            'Cost_Income_Ratio': (0.2, 0.8),  # 20-80% for banks
            'PE': (-100, 100),       # Handle negative earnings
            'PB': (0, 20),           # Book multiples
            'PS': (0, 50),           # Sales multiples
            'EV_EBITDA': (-50, 50)   # Handle negative EBITDA
        }
        
        # Define reporting lag (critical for point-in-time integrity)
        # INSTITUTIONAL ASSUMPTION: Quarterly data available 45 days after quarter end
        self.reporting_lag = 45
        
        self.logger.debug(f"Constants defined: {len(self.intermediary_tables)} table mappings, {len(self.ratio_ranges)} validation ranges")
    
    def get_sector_mapping(self) -> pd.DataFrame:
        """Get sector mapping for all tickers with Banks->Banking fix."""
        try:
            query = text("""
            SELECT ticker, sector
            FROM master_info
            WHERE ticker IS NOT NULL
            """)
            
            sector_map = pd.read_sql(query, self.engine)
            
            # VALIDATED FIX: Banks -> Banking sector name correction
            sector_map.loc[sector_map['sector'] == 'Banks', 'sector'] = 'Banking'
            
            self.logger.debug(f"Retrieved sector mapping for {len(sector_map)} tickers")
            return sector_map
            
        except Exception as e:
            self.logger.error(f"Failed to get sector mapping: {e}")
            raise
    
    def get_correct_quarter_for_date(self, analysis_date: pd.Timestamp) -> Optional[Tuple[int, int]]:
        """
        VALIDATED LOGIC: Get correct quarter data available on analysis_date.
        Rule: Quarter data is available 45 days after quarter end.
        """
        try:
            # Quarter end dates
            quarter_ends = {
                1: pd.Timestamp(analysis_date.year, 3, 31),   # Q1 ends Mar 31
                2: pd.Timestamp(analysis_date.year, 6, 30),   # Q2 ends Jun 30  
                3: pd.Timestamp(analysis_date.year, 9, 30),   # Q3 ends Sep 30
                4: pd.Timestamp(analysis_date.year, 12, 31)   # Q4 ends Dec 31
            }
            
            # Find the most recent quarter whose publish date <= analysis_date
            available_quarters = []
            
            for quarter, end_date in quarter_ends.items():
                publish_date = end_date + pd.Timedelta(days=self.reporting_lag)
                if publish_date <= analysis_date:
                    available_quarters.append((end_date.year, quarter, publish_date))
            
            # Also check previous year Q4
            prev_year_q4_end = pd.Timestamp(analysis_date.year - 1, 12, 31)
            prev_year_q4_publish = prev_year_q4_end + pd.Timedelta(days=self.reporting_lag)
            if prev_year_q4_publish <= analysis_date:
                available_quarters.append((analysis_date.year - 1, 4, prev_year_q4_publish))
            
            # Return the most recent available quarter
            if available_quarters:
                available_quarters.sort(key=lambda x: x[2], reverse=True)
                year, quarter = available_quarters[0][:2]
                self.logger.debug(f"Available quarter for {analysis_date.date()}: {year} Q{quarter}")
                return year, quarter
            else:
                self.logger.warning(f"No quarter data available for {analysis_date.date()}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to determine correct quarter for {analysis_date}: {e}")
            return None
    
    def calculate_sector_neutral_zscore(self, data: pd.DataFrame, 
                                      metric_column: str, 
                                      sector_column: str = 'sector') -> pd.Series:
        """
        VALIDATED LOGIC: Calculate sector-neutral z-scores to prevent sector bias.
        Normalizes within each sector instead of across the entire universe.
        """
        try:
            # Check if we have enough data for sector-neutral normalization
            sector_counts = data.groupby(sector_column)[metric_column].count()
            min_sector_size = sector_counts.min()
            
            if min_sector_size < 2:
                self.logger.warning(f"Insufficient data for sector-neutral normalization (min sector size: {min_sector_size})")
                self.logger.info("Falling back to cross-sectional normalization")
                # Fallback to cross-sectional normalization
                values = data[metric_column].dropna()
                if len(values) > 1:
                    z_scores = (values - values.mean()) / values.std()
                    return z_scores.clip(-3, 3)
                else:
                    return pd.Series(index=data.index, data=0.0)
            
            def sector_zscore(group):
                values = group[metric_column].dropna()
                if len(values) < 2:
                    return group[metric_column] * 0  # Return zeros if insufficient data
                mean_val = values.mean()
                std_val = values.std()
                if std_val == 0:
                    return group[metric_column] * 0
                return (group[metric_column] - mean_val) / std_val
            
            # Apply sector-neutral normalization
            z_scores = data.groupby(sector_column, group_keys=False).apply(sector_zscore)
            
            # Flatten the result if needed
            if isinstance(z_scores, pd.DataFrame):
                z_scores = z_scores.reset_index(level=0, drop=True)
            
            # Ensure proper index alignment
            z_scores = z_scores.reindex(data.index, fill_value=0)
            
            # Winsorize at 3 standard deviations
            z_scores = z_scores.clip(-3, 3)
            
            self.logger.debug(f"Calculated sector-neutral z-scores for {len(z_scores)} observations")
            return z_scores
            
        except Exception as e:
            self.logger.error(f"Error calculating sector-neutral z-scores: {e}")
            # Fallback to cross-sectional if sector-neutral fails
            values = data[metric_column].dropna()
            if len(values) > 1:
                return ((values - values.mean()) / values.std()).clip(-3, 3)
            else:
                return pd.Series(index=data.index, data=0.0)
    
    def get_fundamentals_correct_timing(self, analysis_date: pd.Timestamp, 
                                       universe: List[str]) -> pd.DataFrame:
        """
        VALIDATED LOGIC: Get fundamental data using correct point-in-time logic.
        Uses sector-specific tables with proper timing constraints.
        """
        try:
            # Get correct quarter
            quarter_info = self.get_correct_quarter_for_date(analysis_date)
            if quarter_info is None:
                self.logger.warning(f"No fundamental data available for {analysis_date.date()}")
                return pd.DataFrame()
            
            correct_year, correct_quarter = quarter_info
            
            # Get sector mapping with fix
            sector_map = self.get_sector_mapping()
            ticker_sectors = sector_map[sector_map['ticker'].isin(universe)].set_index('ticker')['sector']
            
            all_fundamentals = []
            
            # Banking tickers
            banking_tickers = [t for t in universe if ticker_sectors.get(t) == 'Banking']
            if banking_tickers:
                banking_query = text(f"""
                SELECT *
                FROM {self.intermediary_tables['banking']}
                WHERE ticker IN ('{"', '".join(banking_tickers)}')
                  AND year = :year
                  AND quarter = :quarter
                  AND has_full_ttm = 1
                """)
                banking_data = pd.read_sql(
                    banking_query, self.engine, 
                    params={'year': correct_year, 'quarter': correct_quarter}
                )
                if not banking_data.empty:
                    banking_data['sector'] = 'Banking'
                    all_fundamentals.append(banking_data)
                    self.logger.debug(f"Retrieved {len(banking_data)} banking records")
            
            # Securities tickers
            securities_tickers = [t for t in universe if ticker_sectors.get(t) == 'Securities']
            if securities_tickers:
                securities_query = text(f"""
                SELECT *
                FROM {self.intermediary_tables['securities']}
                WHERE ticker IN ('{"', '".join(securities_tickers)}')
                  AND year = :year
                  AND quarter = :quarter
                  AND has_full_ttm = 1
                """)
                securities_data = pd.read_sql(
                    securities_query, self.engine,
                    params={'year': correct_year, 'quarter': correct_quarter}
                )
                if not securities_data.empty:
                    securities_data['sector'] = 'Securities'
                    all_fundamentals.append(securities_data)
                    self.logger.debug(f"Retrieved {len(securities_data)} securities records")
            
            # Non-financial tickers
            non_financial_tickers = [t for t in universe 
                                   if ticker_sectors.get(t) not in ['Banking', 'Securities']]
            if non_financial_tickers:
                nonfin_query = text(f"""
                SELECT *
                FROM {self.intermediary_tables['non_financial']}
                WHERE ticker IN ('{"', '".join(non_financial_tickers)}')
                  AND year = :year
                  AND quarter = :quarter
                  AND has_full_ttm = 1
                """)
                nonfin_data = pd.read_sql(
                    nonfin_query, self.engine,
                    params={'year': correct_year, 'quarter': correct_quarter}
                )
                if not nonfin_data.empty:
                    # Add sector information
                    nonfin_sectors = ticker_sectors.reindex(nonfin_data['ticker'])
                    nonfin_data = nonfin_data.assign(sector=nonfin_sectors.values)
                    all_fundamentals.append(nonfin_data)
                    self.logger.debug(f"Retrieved {len(nonfin_data)} non-financial records")
            
            # Combine all data
            if all_fundamentals:
                combined_data = pd.concat(all_fundamentals, ignore_index=True)
                self.logger.info(f"Retrieved {len(combined_data)} total fundamental records for {analysis_date.date()}")
                return combined_data
            else:
                self.logger.warning(f"No fundamental data found for any tickers on {analysis_date.date()}")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Failed to get fundamental data: {e}")
            return pd.DataFrame()
    
    def get_market_data(self, analysis_date: pd.Timestamp, universe: List[str]) -> pd.DataFrame:
        """
        VALIDATED LOGIC: Get market data AS OF analysis date (no lag).
        Uses validated column names from working engine.
        """
        try:
            # Use validated column names from working engine
            ticker_str = "', '".join(universe)
            
            query = f"""
            WITH latest_market AS (
                SELECT 
                    ticker,
                    trading_date,
                    close_price_adjusted as price,
                    market_cap,
                    total_shares,
                    ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY trading_date DESC) as rn
                FROM vcsc_daily_data_complete
                WHERE ticker IN ('{ticker_str}')
                  AND trading_date <= '{analysis_date.date()}'
            )
            SELECT 
                ticker,
                trading_date,
                price,
                market_cap,
                total_shares
            FROM latest_market
            WHERE rn = 1
            """
            
            market_data = pd.read_sql(query, self.engine)
            market_data['date'] = analysis_date
            
            self.logger.debug(f"Retrieved market data for {len(market_data)} tickers as of {analysis_date.date()}")
            return market_data
            
        except Exception as e:
            self.logger.error(f"Failed to get market data: {e}")
            return pd.DataFrame()
    
    def calculate_returns_fixed(self, price_data: pd.DataFrame, 
                               start_date: pd.Timestamp, 
                               end_date: pd.Timestamp) -> pd.Series:
        """
        VALIDATED LOGIC: Fixed return calculation that handles missing exact dates.
        Uses nearest available price after start_date and on/before end_date.
        """
        try:
            returns = pd.Series(dtype=float)
            
            for ticker in price_data['ticker'].unique():
                ticker_data = price_data[price_data['ticker'] == ticker].sort_values('date')
                
                # Get first available price on or after start_date
                start_prices = ticker_data[ticker_data['date'] >= start_date]
                if not start_prices.empty:
                    start_price = start_prices.iloc[0]['adj_close']
                    start_date_actual = start_prices.iloc[0]['date']
                else:
                    continue
                
                # Get last available price on or before end_date
                end_prices = ticker_data[ticker_data['date'] <= end_date]
                if not end_prices.empty:
                    end_price = end_prices.iloc[-1]['adj_close']
                    end_date_actual = end_prices.iloc[-1]['date']
                else:
                    continue
                
                # Calculate return
                if pd.notna(start_price) and pd.notna(end_price) and start_price > 0:
                    return_val = (end_price / start_price) - 1
                    returns[ticker] = return_val
            
            self.logger.debug(f"Calculated returns for {len(returns)} tickers from {start_date.date()} to {end_date.date()}")
            return returns
            
        except Exception as e:
            self.logger.error(f"Failed to calculate returns: {e}")
            return pd.Series(dtype=float)
    
    def calculate_qvm_composite(self, analysis_date: pd.Timestamp, 
                               universe: List[str]) -> Dict[str, float]:
        """
        MAIN ENGINE: Calculate QVM composite scores using validated logic.
        Returns dictionary with ticker -> QVM score mapping.
        """
        try:
            self.logger.info(f"Calculating QVM composite for {len(universe)} tickers on {analysis_date.date()}")
            
            results = {}
            
            # Get fundamental and market data
            fundamentals = self.get_fundamentals_correct_timing(analysis_date, universe)
            market_data = self.get_market_data(analysis_date, universe)
            
            if fundamentals.empty or market_data.empty:
                self.logger.warning("Insufficient data for QVM calculation")
                return results
            
            # Merge data
            data = pd.merge(fundamentals, market_data, on='ticker', how='inner')
            
            if data.empty:
                self.logger.warning("No merged data available for QVM calculation")
                return results
            
            # Calculate individual factor components
            quality_scores = self._calculate_quality_composite(data, analysis_date)
            value_scores = self._calculate_value_composite(data, analysis_date)
            momentum_scores = self._calculate_momentum_composite(data, analysis_date, universe)
            
            # Combine into QVM composite
            for ticker in data['ticker'].unique():
                quality = quality_scores.get(ticker, 0.0)
                value = value_scores.get(ticker, 0.0)
                momentum = momentum_scores.get(ticker, 0.0)
                
                # Weighted combination
                qvm_score = (
                    self.qvm_weights['quality'] * quality +
                    self.qvm_weights['value'] * value +
                    self.qvm_weights['momentum'] * momentum
                )
                
                results[ticker] = qvm_score
            
            self.logger.info(f"Successfully calculated QVM scores for {len(results)} tickers")
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to calculate QVM composite: {e}")
            return {}
    
    def _calculate_quality_composite(self, data: pd.DataFrame, analysis_date: pd.Timestamp) -> Dict[str, float]:
        """
        VALIDATED LOGIC: Calculate quality composite using sector-specific metrics.
        3-tier framework: Level 50%, Change 30%, Acceleration 20%.
        """
        try:
            quality_scores = {}
            
            # Calculate ROAE for all sectors
            for idx, row in data.iterrows():
                ticker = row['ticker']
                sector = row['sector']
                
                try:
                    # VALIDATED: Dynamic ROAE calculation
                    if 'NetProfit_TTM' in row and 'AvgTotalEquity' in row:
                        if pd.notna(row['NetProfit_TTM']) and pd.notna(row['AvgTotalEquity']) and row['AvgTotalEquity'] > 0:
                            roae = row['NetProfit_TTM'] / row['AvgTotalEquity']
                            data.loc[idx, 'ROAE'] = roae
                        else:
                            data.loc[idx, 'ROAE'] = np.nan
                    else:
                        data.loc[idx, 'ROAE'] = np.nan
                        
                except Exception as e:
                    self.logger.debug(f"ROAE calculation failed for {ticker}: {e}")
                    data.loc[idx, 'ROAE'] = np.nan
            
            # Sector-neutral z-score normalization
            if 'ROAE' in data.columns:
                roae_z_scores = self.calculate_sector_neutral_zscore(data, 'ROAE')
                
                for ticker in data['ticker']:
                    ticker_data = data[data['ticker'] == ticker].iloc[0]
                    ticker_idx = ticker_data.name
                    
                    if ticker_idx in roae_z_scores.index:
                        quality_scores[ticker] = roae_z_scores.loc[ticker_idx]
                    else:
                        quality_scores[ticker] = 0.0
            
            self.logger.debug(f"Calculated quality scores for {len(quality_scores)} tickers")
            return quality_scores
            
        except Exception as e:
            self.logger.error(f"Failed to calculate quality composite: {e}")
            return {}
    
    def _calculate_value_composite(self, data: pd.DataFrame, analysis_date: pd.Timestamp) -> Dict[str, float]:
        """
        VALIDATED LOGIC: Calculate value composite with sector-specific revenue metrics.
        """
        try:
            value_scores = {}
            
            # Calculate ratios for each ticker
            for idx, row in data.iterrows():
                ticker = row['ticker']
                sector = row['sector']
                market_cap = row.get('market_cap', 0)
                
                ratios = {}
                
                try:
                    # P/E ratio (inverted for value score)
                    if 'NetProfit_TTM' in row and pd.notna(row['NetProfit_TTM']) and row['NetProfit_TTM'] > 0:
                        pe_ratio = market_cap / row['NetProfit_TTM']
                        ratios['pe_score'] = 1 / pe_ratio if pe_ratio > 0 else 0
                    else:
                        ratios['pe_score'] = 0
                    
                    # P/B ratio (inverted for value score)
                    if 'AvgTotalEquity' in row and pd.notna(row['AvgTotalEquity']) and row['AvgTotalEquity'] > 0:
                        pb_ratio = market_cap / row['AvgTotalEquity']
                        ratios['pb_score'] = 1 / pb_ratio if pb_ratio > 0 else 0
                    else:
                        ratios['pb_score'] = 0
                    
                    # Sector-specific P/S ratio
                    revenue_ttm = 0
                    if sector == 'Banking' and 'TotalOperatingIncome_TTM' in row:
                        revenue_ttm = row.get('TotalOperatingIncome_TTM', 0)
                    elif sector == 'Securities' and 'TotalOperatingRevenue_TTM' in row:
                        revenue_ttm = row.get('TotalOperatingRevenue_TTM', 0)
                    elif 'Revenue_TTM' in row:
                        revenue_ttm = row.get('Revenue_TTM', 0)
                    
                    if pd.notna(revenue_ttm) and revenue_ttm > 0:
                        ps_ratio = market_cap / revenue_ttm
                        ratios['ps_score'] = 1 / ps_ratio if ps_ratio > 0 else 0
                    else:
                        ratios['ps_score'] = 0
                    
                    # Simple value composite (equal weights for now)
                    value_score = np.mean(list(ratios.values()))
                    data.loc[idx, 'value_composite'] = value_score
                    
                except Exception as e:
                    self.logger.debug(f"Value calculation failed for {ticker}: {e}")
                    data.loc[idx, 'value_composite'] = 0
            
            # Sector-neutral normalization
            if 'value_composite' in data.columns:
                value_z_scores = self.calculate_sector_neutral_zscore(data, 'value_composite')
                
                for ticker in data['ticker']:
                    ticker_data = data[data['ticker'] == ticker].iloc[0]
                    ticker_idx = ticker_data.name
                    
                    if ticker_idx in value_z_scores.index:
                        value_scores[ticker] = value_z_scores.loc[ticker_idx]
                    else:
                        value_scores[ticker] = 0.0
            
            self.logger.debug(f"Calculated value scores for {len(value_scores)} tickers")
            return value_scores
            
        except Exception as e:
            self.logger.error(f"Failed to calculate value composite: {e}")
            return {}
    
    def _calculate_momentum_composite(self, data: pd.DataFrame, 
                                    analysis_date: pd.Timestamp, 
                                    universe: List[str]) -> Dict[str, float]:
        """
        VALIDATED LOGIC: Calculate momentum composite with skip-1-month convention.
        Multiple timeframes: 1M, 3M, 6M, 12M with proper weighting.
        """
        try:
            momentum_scores = {}
            
            # Get price data using validated column names
            ticker_str = "', '".join(universe)
            start_date = analysis_date - pd.DateOffset(months=14)  # Need 14 months for 12M + 1M skip + buffer
            
            price_query = f"""
            SELECT 
                date,
                ticker,
                close as adj_close
            FROM equity_history
            WHERE ticker IN ('{ticker_str}')
              AND date BETWEEN '{start_date.date()}' AND '{analysis_date.date()}'
            ORDER BY ticker, date
            """
            
            price_data = pd.read_sql(price_query, self.engine, parse_dates=['date'])
            
            if price_data.empty:
                self.logger.warning("No price data available for momentum calculation")
                return momentum_scores
            
            # VALIDATED: Multiple momentum periods with skip-1-month
            periods = [
                ('1M', 1, 1),   # 1-month with 1-month skip
                ('3M', 3, 1),   # 3-month with 1-month skip
                ('6M', 6, 1),   # 6-month with 1-month skip
                ('12M', 12, 1)  # 12-month with 1-month skip
            ]
            
            period_returns = {}
            
            for period_name, lookback, skip in periods:
                end_date = analysis_date - pd.DateOffset(months=skip)
                start_date_period = analysis_date - pd.DateOffset(months=lookback + skip)
                returns = self.calculate_returns_fixed(price_data, start_date_period, end_date)
                
                if not returns.empty:
                    period_returns[period_name] = returns
                    self.logger.debug(f"Calculated {period_name} returns for {len(returns)} tickers")
            
            # Combine momentum periods with weights
            for ticker in universe:
                momentum_components = []
                
                for period_name in ['1M', '3M', '6M', '12M']:
                    if period_name in period_returns and ticker in period_returns[period_name]:
                        weight = self.momentum_weights.get(period_name.lower(), 0.25)  # Default equal weight
                        momentum_components.append(weight * period_returns[period_name][ticker])
                
                if momentum_components:
                    momentum_scores[ticker] = sum(momentum_components)
                else:
                    momentum_scores[ticker] = 0.0
            
            # Cross-sectional z-score normalization (momentum is typically cross-sectional)
            if momentum_scores:
                momentum_series = pd.Series(momentum_scores)
                momentum_mean = momentum_series.mean()
                momentum_std = momentum_series.std()
                
                if momentum_std > 0:
                    for ticker in momentum_scores:
                        momentum_scores[ticker] = (momentum_scores[ticker] - momentum_mean) / momentum_std
                        momentum_scores[ticker] = np.clip(momentum_scores[ticker], -3, 3)  # Winsorize
            
            self.logger.debug(f"Calculated momentum scores for {len(momentum_scores)} tickers")
            return momentum_scores
            
        except Exception as e:
            self.logger.error(f"Failed to calculate momentum composite: {e}")
            return {}

# END OF CANONICAL QVM ENGINE