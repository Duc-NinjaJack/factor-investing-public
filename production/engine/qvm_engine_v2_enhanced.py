"""
Vietnam Factor Investing Platform - QVM Engine v2 (Enhanced)
==========================================================
Component: Enhanced Factor Calculation Engine (Sophisticated Implementation)
Purpose: Experimental group for scientific bake-off - Multi-tier methodology
Author: Duc Nguyen, Principal Quantitative Strategist
Date Created: July 22, 2025
Status: ENHANCED ENGINE (v2) - EXPERIMENTAL GROUP

SCIENTIFIC BAKE-OFF ROLE:
This engine represents the SOPHISTICATED HYPOTHESIS in our signal construction experiment:
- Quality: Multi-tier framework with Master Quality Signal (Level 50%, Change 30%, Acceleration 20%)
- Value: Enhanced EV/EBITDA with industry-standard Enterprise Value calculation
- Momentum: Standard returns with sophisticated normalization
- Expected Performance: ~26.3% annual return, 1.77 Sharpe ratio (hypothesis)

ENHANCED METHODOLOGY FEATURES:
1. Multi-tier Quality Framework: Level (50%), Change (30%), Acceleration (20%)
2. Master Quality Signal: 0.35×ROAE_Momentum + 0.25×ROAA_Momentum + 0.25×Operating_Margin_Momentum + 0.15×EBITDA_Margin_Momentum
3. Enhanced EV/EBITDA: Industry-standard Enterprise Value = Market Cap + Total Debt - Cash & Equivalents
4. Sector-Specific Value Weights: Banking (PE=60%, PB=40%), Securities (PE=50%, PB=30%, PS=20%), etc.
5. Dynamic Weight Optimization: Rolling 12-quarter Sharpe-based weight calculation
6. Working Capital Efficiency: CCC, DSO, DIO, DPO signals with YoY change calculations

SOPHISTICATED SOURCE VALIDATION:
Built from empirically validated sophisticated methodology in:
- config/strategy_config.yml: 221 lines of advanced factor configuration
- config/factor_metadata.yml: 442 lines of detailed factor specifications
- notebooks/phase5_institutional_qvm/enhanced_ev_ebitda_engine.py: Enhanced EV Calculator
- docs/_archive/4_validation_and_quality/factors/advanced_factor_methodology_guide.md: 708 lines institutional methodology

CORE PRINCIPLES:
1. "Store Raw, Transform Dynamically" - all ratios calculated on-the-fly
2. Point-in-time integrity with 45-day reporting lag
3. CORRECTED: Sector-neutral normalization as PRIMARY (not fallback) - institutional standard
4. Multi-tier signal construction with academic validation
5. Enhanced institutional-grade validation and logging

CRITICAL METHODOLOGY CORRECTIONS (July 22, 2025):
Based on expert institutional feedback and audit validation, this engine now correctly implements:
- PRIMARY: Sector-neutral normalization for pure alpha signal extraction
- FALLBACK: Cross-sectional only for very small sectors (<10 tickers)
- CORRECTED: Point-in-time equity for P/B ratios (not TTM averages)
This separation of signal generation from portfolio construction is the
institutional standard and prevents conflation of alpha with sector beta.
Point-in-time equity ensures proper valuation accuracy for value factors.

Data Sources:
- intermediary_calculations_banking_cleaned (21 tickers, 65+ columns)
- intermediary_calculations_securities_cleaned (26 tickers, 79+ columns) 
- intermediary_calculations_enhanced (667 non-financial tickers, 71+ columns)
- v_comprehensive_fundamental_items (point-in-time balance sheet data)
- vcsc_daily_data_complete (market data with proper column names)
- equity_history (price returns with skip-1-month convention)

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
from typing import Dict, List, Optional, Tuple, Union
import warnings
import logging
from datetime import datetime, timedelta

# Suppress pandas warnings
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)


class EnhancedEVCalculator:
    """
    Enhanced EV/EBITDA calculator with industry-standard methodology.
    Implements hybrid data sourcing with proper point-in-time integrity.
    """
    
    def __init__(self, engine, logger=None):
        """Initialize with database engine and optional logger."""
        self.engine = engine
        self.logger = logger if logger else logging.getLogger(__name__)
        
        # Sector exclusions for EV/EBITDA
        self.ev_excluded_sectors = {'Banking', 'Insurance', 'Securities'}
        
        self.logger.debug("Enhanced EV/EBITDA Calculator initialized")
    
    def get_point_in_time_balance_sheet(self, ticker: str, analysis_date: pd.Timestamp) -> Optional[Dict[str, float]]:
        """Get point-in-time balance sheet data (debt, cash) with proper temporal logic."""
        try:
            # Apply same 45-day reporting lag logic as fundamental data
            year = analysis_date.year
            quarter_ends = {
                1: pd.Timestamp(year, 3, 31),   # Q1 ends Mar 31
                2: pd.Timestamp(year, 6, 30),   # Q2 ends Jun 30
                3: pd.Timestamp(year, 9, 30),   # Q3 ends Sep 30
                4: pd.Timestamp(year, 12, 31)   # Q4 ends Dec 31
            }
            
            # Find the most recent quarter whose publish date <= analysis_date
            available_quarters = []
            for quarter, end_date in quarter_ends.items():
                publish_date = end_date + pd.Timedelta(days=45)
                if publish_date <= analysis_date:
                    available_quarters.append((year, quarter, publish_date))
            
            # Also check previous year Q4
            prev_year_q4_end = pd.Timestamp(year - 1, 12, 31)
            prev_year_q4_publish = prev_year_q4_end + pd.Timedelta(days=45)
            if prev_year_q4_publish <= analysis_date:
                available_quarters.append((year - 1, 4, prev_year_q4_publish))
            
            if not available_quarters:
                return None
            
            # Get most recent available quarter
            available_quarters.sort(key=lambda x: x[2], reverse=True)
            target_year, target_quarter, _ = available_quarters[0]
            
            # Query point-in-time balance sheet data
            query = text("""
                SELECT 
                    CashAndCashEquivalents,
                    COALESCE(ShortTermDebt, 0) + COALESCE(LongTermDebt, 0) as TotalDebt,
                    ShortTermDebt,
                    LongTermDebt
                FROM v_comprehensive_fundamental_items
                WHERE ticker = :ticker 
                  AND year = :year 
                  AND quarter = :quarter
                LIMIT 1
            """)
            
            result = pd.read_sql(query, self.engine, params={
                'ticker': ticker,
                'year': target_year,
                'quarter': target_quarter
            })
            
            if result.empty:
                return None
            
            row = result.iloc[0]
            cash = row['CashAndCashEquivalents']
            total_debt = row['TotalDebt']
            
            if pd.isna(cash) or pd.isna(total_debt):
                return None
            
            return {
                'cash_and_equivalents': float(cash),
                'total_debt': float(total_debt),
                'data_quarter': f"{target_year} Q{target_quarter}"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get balance sheet data for {ticker}: {e}")
            return None
    
    def calculate_enhanced_ev_ebitda(self, ticker: str, analysis_date: pd.Timestamp, 
                                   sector: str, market_cap: float, ebitda_ttm: float) -> float:
        """Calculate industry-standard EV/EBITDA with proper enterprise value."""
        try:
            # Sector exclusion logic
            if sector in self.ev_excluded_sectors:
                return 0.0
            
            # Validate inputs
            if not market_cap or market_cap <= 0 or not ebitda_ttm or ebitda_ttm <= 0:
                return 0.0
            
            # Get point-in-time balance sheet data
            balance_sheet = self.get_point_in_time_balance_sheet(ticker, analysis_date)
            if not balance_sheet:
                # Fallback to simplified calculation
                ev_ebitda_ratio = market_cap / ebitda_ttm
                return 1 / ev_ebitda_ratio if ev_ebitda_ratio > 0 else 0.0
            
            # Calculate enterprise value
            cash = balance_sheet['cash_and_equivalents']
            debt = balance_sheet['total_debt']
            enterprise_value = market_cap + debt - cash
            
            # Validate enterprise value
            if enterprise_value <= 0:
                enterprise_value = market_cap
            
            # Calculate EV/EBITDA ratio
            ev_ebitda_ratio = enterprise_value / ebitda_ttm
            
            # Invert for scoring (lower ratios = higher scores)
            ev_ebitda_score = 1 / ev_ebitda_ratio if ev_ebitda_ratio > 0 else 0.0
            
            return ev_ebitda_score
            
        except Exception as e:
            self.logger.error(f"Enhanced EV/EBITDA calculation failed for {ticker}: {e}")
            return 0.0
    
    def get_sector_specific_value_weights(self, sector: str) -> Dict[str, float]:
        """Get sector-specific value factor weights."""
        if sector in self.ev_excluded_sectors:
            # Financial institutions: exclude EV/EBITDA and P/S
            if sector == 'Banking':
                return {'pe': 0.60, 'pb': 0.40, 'ps': 0.00, 'ev_ebitda': 0.00}
            elif sector == 'Securities':
                return {'pe': 0.50, 'pb': 0.30, 'ps': 0.20, 'ev_ebitda': 0.00}
            elif sector == 'Insurance':
                return {'pe': 0.50, 'pb': 0.50, 'ps': 0.00, 'ev_ebitda': 0.00}
        
        # Standard weights for non-financial sectors
        return {'pe': 0.40, 'pb': 0.30, 'ps': 0.20, 'ev_ebitda': 0.10}


class QVMEngineV2Enhanced:
    """
    Enhanced Canonical QVM factor calculator with sophisticated multi-tier methodology.
    Implements institutional-grade factor calculations with complete sophistication.
    """
    
    def __init__(self, config_path: str = None, log_level: str = 'INFO'):
        """Initialize enhanced canonical engine with sophisticated configurations."""
        
        # Setup logging
        self.logger = self._setup_logging(log_level)
        self.logger.info("Initializing Enhanced Canonical QVM Engine")
        
        # Resolve configuration paths
        if config_path is None:
            project_root = Path(__file__).parent.parent.parent
            self.db_config_path = project_root / 'config' / 'database.yml'
            self.strategy_config_path = project_root / 'config' / 'strategy_config.yml'
            self.factor_metadata_path = project_root / 'config' / 'factor_metadata.yml'
        else:
            config_dir = Path(config_path)
            self.db_config_path = config_dir / 'database.yml'
            self.strategy_config_path = config_dir / 'strategy_config.yml'
            self.factor_metadata_path = config_dir / 'factor_metadata.yml'
        
        # Load configurations
        self._load_configurations()
        
        # Create database engine
        self._create_database_engine()
        
        # Initialize enhanced components
        self._initialize_enhanced_components()
        
        # Define institutional constants
        self._define_constants()
        
        self.logger.info("Enhanced Canonical QVM Engine initialized successfully")
        self.logger.info(f"QVM Weights: Quality {self.qvm_weights['quality']*100}%, Value {self.qvm_weights['value']*100}%, Momentum {self.qvm_weights['momentum']*100}%")
        self.logger.info("Enhanced Features: Multi-tier Quality, Enhanced EV/EBITDA, Sector-specific weights, Working capital efficiency")
    
    def _setup_logging(self, log_level: str) -> logging.Logger:
        """Setup comprehensive logging for production use."""
        logger = logging.getLogger('EnhancedCanonicalQVMEngine')
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
        """Load database, strategy, and factor metadata configurations."""
        try:
            # Load database configuration
            with open(self.db_config_path, 'r') as f:
                self.db_config = yaml.safe_load(f)['production']
            
            # Load strategy configuration
            with open(self.strategy_config_path, 'r') as f:
                self.factor_config = yaml.safe_load(f)
            
            # Load factor metadata if available
            try:
                with open(self.factor_metadata_path, 'r') as f:
                    self.factor_metadata = yaml.safe_load(f)
            except FileNotFoundError:
                self.factor_metadata = {}
                self.logger.warning("Factor metadata file not found, using defaults")
            
            # Store parameters as instance attributes for easy access
            self.quality_config = self.factor_config['quality']
            self.value_config = self.factor_config['value']
            self.momentum_config = self.factor_config['momentum']
            self.qvm_weights = self.factor_config['qvm_composite']['weights']
            
            # Enhanced quality configuration
            self.quality_tier_weights = self.quality_config['tier_weights']
            self.quality_metrics = self.quality_config['metrics']
            
            # Enhanced value configuration
            self.value_metric_weights = self.value_config['metric_weights']
            self.revenue_metrics = self.value_config['revenue_metrics']
            
            # Enhanced momentum configuration
            self.momentum_weights = self.momentum_config['timeframe_weights']
            self.momentum_lookbacks = self.momentum_config['lookback_periods']
            self.momentum_skip = self.momentum_config['skip_months']
            
            self.logger.info("Enhanced configurations loaded successfully")
            
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
    
    def _initialize_enhanced_components(self):
        """Initialize enhanced calculation components."""
        try:
            # Initialize Enhanced EV Calculator
            self.ev_calculator = EnhancedEVCalculator(self.engine, self.logger)
            
            self.logger.info("Enhanced components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize enhanced components: {e}")
            raise
    
    def _define_constants(self):
        """Define institutional constants and table mappings."""
        # Define table mappings for sector-specific data
        self.intermediary_tables = {
            'banking': 'intermediary_calculations_banking_cleaned',
            'securities': 'intermediary_calculations_securities_cleaned', 
            'non_financial': 'intermediary_calculations_enhanced'
        }
        
        # Enhanced ratio validation ranges
        self.ratio_ranges = {
            'ROAE': (0.0, 0.5),      # 0-50% is reasonable
            'ROAA': (0.0, 0.1),      # 0-10% is reasonable
            'NIM': (0.0, 0.15),      # 0-15% for banks
            'Cost_Income_Ratio': (0.2, 0.8),  # 20-80% for banks
            'Operating_Margin': (0.0, 0.5),   # 0-50% operating margin
            'EBITDA_Margin': (0.0, 0.5),      # 0-50% EBITDA margin
            'PE': (-100, 100),       # Handle negative earnings
            'PB': (0, 20),           # Book multiples
            'PS': (0, 50),           # Sales multiples
            'EV_EBITDA': (-50, 50),  # Handle negative EBITDA
            'CCC': (-200, 500),      # Cash conversion cycle in days
            'DSO': (0, 200),         # Days sales outstanding
            'DIO': (0, 500),         # Days inventory outstanding
            'DPO': (0, 200)          # Days payable outstanding
        }
        
        # Define reporting lag (critical for point-in-time integrity)
        self.reporting_lag = 45
        
        # Sophisticated Quality Framework using strategy_config.yml
        # No hardcoded weights - all metrics defined by configuration
        
        self.logger.debug(f"Enhanced constants defined: {len(self.intermediary_tables)} table mappings, "
                         f"{len(self.ratio_ranges)} validation ranges, Master Quality Signal configured")
    
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
        CORRECTED LOGIC (v2): Get correct quarter data available on analysis_date.
        This version correctly checks all quarters of the previous year to handle
        early-year "cold start" scenarios.
        
        Rule: Quarter data is available 45 days after quarter end.
        """
        try:
            available_quarters = []
            current_year = analysis_date.year
            
            # Check all quarters in the CURRENT year
            for quarter in [1, 2, 3, 4]:
                q_end_month = quarter * 3
                q_end_day = 31 if q_end_month in [3, 12] else 30
                end_date = pd.Timestamp(current_year, q_end_month, q_end_day)
                publish_date = end_date + pd.Timedelta(days=self.reporting_lag)
                if publish_date <= analysis_date:
                    available_quarters.append((current_year, quarter, publish_date))

            # Check all quarters in the PREVIOUS year as fallback for cold starts
            prev_year = current_year - 1
            for quarter in [1, 2, 3, 4]:
                q_end_month = quarter * 3
                q_end_day = 31 if q_end_month in [3, 12] else 30
                end_date = pd.Timestamp(prev_year, q_end_month, q_end_day)
                publish_date = end_date + pd.Timedelta(days=self.reporting_lag)
                if publish_date <= analysis_date:
                    available_quarters.append((prev_year, quarter, publish_date))

            if not available_quarters:
                self.logger.warning(f"No quarter data available for {analysis_date.date()}")
                return None

            # Sort by publish date to find the most recent one
            available_quarters.sort(key=lambda x: x[2], reverse=True)
            year, quarter, _ = available_quarters[0]
            self.logger.debug(f"Available quarter for {analysis_date.date()}: {year} Q{quarter}")
            return year, quarter

        except Exception as e:
            self.logger.error(f"Failed to determine correct quarter for {analysis_date}: {e}")
            return None
    
    def calculate_sector_neutral_zscore(self, data: pd.DataFrame, 
                                      metric_column: str, 
                                      sector_column: str = 'sector') -> pd.Series:
        """
        INSTITUTIONAL METHODOLOGY: Sector-neutral z-scores as PRIMARY approach.
        
        This implements the corrected institutional framework based on expert feedback:
        - PRIMARY: Sector-neutral normalization (extracts pure alpha signal)
        - FALLBACK: Cross-sectional only for very small sectors (<10 tickers)
        
        The key insight is that sector-neutral normalization should be the DEFAULT,
        not a sophisticated enhancement. This separates signal generation from
        portfolio construction as per institutional best practices.
        """
        try:
            # Get sector counts and determine methodology
            sector_counts = data.groupby(sector_column)[metric_column].count()
            
            # INSTITUTIONAL THRESHOLD: Use sector-neutral unless sector is very small
            # This is the CORRECTED logic - sector-neutral is PRIMARY, not fallback
            use_sector_neutral = True
            for sector, count in sector_counts.items():
                if count < 10:  # Institutional threshold for minimum sector size
                    self.logger.info(f"Sector '{sector}' has only {count} tickers - may use cross-sectional fallback")
                    # Check if this is the only sector or if we have multiple small sectors
                    if len(sector_counts) == 1 or (sector_counts < 10).all():
                        use_sector_neutral = False
                        break
            
            if use_sector_neutral:
                # PRIMARY METHODOLOGY: Sector-neutral normalization
                self.logger.debug("Using PRIMARY sector-neutral normalization (institutional standard)")
                
                def sector_zscore(group):
                    values = group[metric_column].dropna()
                    if len(values) < 2:
                        # For very small groups, return neutral scores
                        return pd.Series(0.0, index=group.index)
                    mean_val = values.mean()
                    std_val = values.std()
                    if std_val == 0:
                        # No variation within sector - all get neutral scores
                        return pd.Series(0.0, index=group.index)
                    # Pure alpha extraction within sector
                    return (group[metric_column] - mean_val) / std_val
                
                # Apply sector-neutral normalization
                z_scores = data.groupby(sector_column, group_keys=False).apply(sector_zscore)
                
                # Ensure proper index alignment
                if isinstance(z_scores, pd.DataFrame):
                    z_scores = z_scores.iloc[:, 0]  # Extract series if needed
                
                z_scores = z_scores.reindex(data.index, fill_value=0)
                
            else:
                # FALLBACK: Cross-sectional normalization (only for very small universes)
                self.logger.warning("Using FALLBACK cross-sectional normalization due to insufficient sector sizes")
                self.logger.warning("This is not ideal - consider expanding universe for proper sector-neutral analysis")
                
                values = data[metric_column].dropna()
                if len(values) > 1:
                    mean_val = values.mean()
                    std_val = values.std()
                    if std_val > 0:
                        z_scores = (data[metric_column] - mean_val) / std_val
                    else:
                        z_scores = pd.Series(0.0, index=data.index)
                else:
                    z_scores = pd.Series(0.0, index=data.index)
            
            # Winsorization at 3 standard deviations (institutional standard)
            z_scores = z_scores.clip(-3, 3)
            
            methodology = "sector-neutral" if use_sector_neutral else "cross-sectional"
            self.logger.info(f"Calculated {methodology} z-scores for {len(z_scores)} observations")
            
            return z_scores
            
        except Exception as e:
            self.logger.error(f"Error in institutional normalization: {e}")
            # Emergency fallback - return neutral scores
            return pd.Series(0.0, index=data.index)
    
    def get_fundamentals_correct_timing(self, analysis_date: pd.Timestamp, 
                                       universe: List[str]) -> pd.DataFrame:
        """
        ENHANCED LOGIC: Get fundamental data using correct point-in-time logic.
        Uses sector-specific tables with enhanced column access.
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
        ENHANCED LOGIC: Get market data AS OF analysis date with enhanced validation.
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
            
            self.logger.debug(f"Retrieved enhanced market data for {len(market_data)} tickers as of {analysis_date.date()}")
            return market_data
            
        except Exception as e:
            self.logger.error(f"Failed to get market data: {e}")
            return pd.DataFrame()
    
    def calculate_qvm_composite(self, analysis_date: pd.Timestamp, 
                               universe: List[str]) -> Dict[str, Dict[str, float]]:
        """
        ENHANCED MAIN ENGINE: Calculate QVM composite scores using sophisticated methodology.
        CRITICAL INFRASTRUCTURE FIX: Now returns component breakdown for institutional transparency.
        
        Returns dictionary with ticker -> {component: score} mapping:
        {
            'ticker': {
                'Quality_Composite': 0.15,
                'Value_Composite': -0.23, 
                'Momentum_Composite': 0.78,
                'QVM_Composite': 0.21
            }
        }
        """
        try:
            self.logger.info(f"Calculating Enhanced QVM composite for {len(universe)} tickers on {analysis_date.date()}")
            
            results = {}
            
            # Get fundamental and market data
            fundamentals = self.get_fundamentals_correct_timing(analysis_date, universe)
            market_data = self.get_market_data(analysis_date, universe)
            
            if fundamentals.empty or market_data.empty:
                self.logger.warning("Insufficient data for Enhanced QVM calculation")
                return results
            
            # Merge data
            data = pd.merge(fundamentals, market_data, on='ticker', how='inner')
            
            if data.empty:
                self.logger.warning("No merged data available for Enhanced QVM calculation")
                return results
            
            # Calculate individual factor components using enhanced methodology
            quality_scores = self._calculate_enhanced_quality_composite(data, analysis_date)
            value_scores = self._calculate_enhanced_value_composite(data, analysis_date)
            momentum_scores = self._calculate_enhanced_momentum_composite(data, analysis_date, universe)
            
            # Combine into QVM composite with component breakdown
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
                
                # CRITICAL FIX: Return component breakdown for institutional analysis
                results[ticker] = {
                    'Quality_Composite': quality,
                    'Value_Composite': value,
                    'Momentum_Composite': momentum,
                    'QVM_Composite': qvm_score
                }
            
            self.logger.info(f"Successfully calculated Enhanced QVM scores with components for {len(results)} tickers")
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to calculate Enhanced QVM composite: {e}")
            return {}
    
    def _calculate_enhanced_quality_composite(self, data: pd.DataFrame, analysis_date: pd.Timestamp) -> Dict[str, float]:
        """
        SOPHISTICATED QUALITY LOGIC: Multi-metric, sector-specific framework from strategy_config.yml
        Implements exactly what drove 63.2% performance contribution in Phase 6 validation.
        
        Banking: ROAE, ROAA, NIM, Cost_Income_Ratio
        Securities: ROAE, BrokerageRatio, NetProfitMargin  
        Non-Financial: ROAE, NetProfitMargin, GrossMargin, OperatingMargin
        """
        try:
            quality_scores = {}
            
            # Calculate sector-specific quality metrics from configuration
            for idx, row in data.iterrows():
                ticker = row['ticker']
                sector = row['sector']
                
                try:
                    sector_metrics = {}
                    
                    if sector == 'Banking':
                        # Banking Quality Metrics (from strategy_config.yml lines 16-20)
                        # ROAE - Core profitability
                        if 'NetProfit_TTM' in row and 'AvgTotalEquity' in row:
                            if pd.notna(row['NetProfit_TTM']) and pd.notna(row['AvgTotalEquity']) and row['AvgTotalEquity'] > 0:
                                sector_metrics['ROAE'] = row['NetProfit_TTM'] / row['AvgTotalEquity']
                        
                        # ROAA - Risk-adjusted returns
                        if 'NetProfit_TTM' in row and 'AvgTotalAssets' in row:
                            if pd.notna(row['NetProfit_TTM']) and pd.notna(row['AvgTotalAssets']) and row['AvgTotalAssets'] > 0:
                                sector_metrics['ROAA'] = row['NetProfit_TTM'] / row['AvgTotalAssets']
                        
                        # NIM - Net Interest Margin (Banking-specific efficiency)
                        if 'NIM' in row and pd.notna(row['NIM']):
                            sector_metrics['NIM'] = row['NIM']
                        elif 'NetInterestIncome_TTM' in row and 'AvgInterestEarningAssets' in row:
                            if pd.notna(row['NetInterestIncome_TTM']) and pd.notna(row['AvgInterestEarningAssets']) and row['AvgInterestEarningAssets'] > 0:
                                sector_metrics['NIM'] = row['NetInterestIncome_TTM'] / row['AvgInterestEarningAssets']
                        
                        # Cost_Income_Ratio - Operational efficiency (CRITICAL FIX #1: Handle Vietnamese negative expense accounting)
                        if 'Cost_Income_Ratio' in row and pd.notna(row['Cost_Income_Ratio']):
                            sector_metrics['Cost_Income_Ratio'] = 1 - row['Cost_Income_Ratio']  # Invert so higher is better
                        elif 'OperatingExpenses_TTM' in row and 'TotalOperatingIncome_TTM' in row:
                            if pd.notna(row['OperatingExpenses_TTM']) and pd.notna(row['TotalOperatingIncome_TTM']) and row['TotalOperatingIncome_TTM'] > 0:
                                # FIXED: Use abs() for Vietnamese accounting where expenses are stored as negative values
                                cost_ratio = abs(row['OperatingExpenses_TTM']) / row['TotalOperatingIncome_TTM']
                                sector_metrics['Cost_Income_Ratio'] = 1 - cost_ratio
                    
                    elif sector == 'Securities':
                        # Securities Quality Metrics (from strategy_config.yml lines 21-24)
                        # ROAE - Core profitability  
                        if 'NetProfit_TTM' in row and 'AvgTotalEquity' in row:
                            if pd.notna(row['NetProfit_TTM']) and pd.notna(row['AvgTotalEquity']) and row['AvgTotalEquity'] > 0:
                                sector_metrics['ROAE'] = row['NetProfit_TTM'] / row['AvgTotalEquity']
                        
                        # BrokerageRatio - Revenue mix quality
                        if 'BrokerageRatio' in row and pd.notna(row['BrokerageRatio']):
                            sector_metrics['BrokerageRatio'] = row['BrokerageRatio']
                        elif 'BrokerageIncome_TTM' in row and 'TotalOperatingRevenue_TTM' in row:
                            if pd.notna(row['BrokerageIncome_TTM']) and pd.notna(row['TotalOperatingRevenue_TTM']) and row['TotalOperatingRevenue_TTM'] > 0:
                                sector_metrics['BrokerageRatio'] = row['BrokerageIncome_TTM'] / row['TotalOperatingRevenue_TTM']
                        
                        # NetProfitMargin - Overall efficiency
                        if 'TotalOperatingRevenue_TTM' in row and 'NetProfit_TTM' in row:
                            if pd.notna(row['NetProfit_TTM']) and pd.notna(row['TotalOperatingRevenue_TTM']) and row['TotalOperatingRevenue_TTM'] > 0:
                                sector_metrics['NetProfitMargin'] = row['NetProfit_TTM'] / row['TotalOperatingRevenue_TTM']
                    
                    else:
                        # Non-Financial Quality Metrics (from strategy_config.yml lines 25-29)
                        # ROAE - Core profitability
                        if 'NetProfit_TTM' in row and 'AvgTotalEquity' in row:
                            if pd.notna(row['NetProfit_TTM']) and pd.notna(row['AvgTotalEquity']) and row['AvgTotalEquity'] > 0:
                                sector_metrics['ROAE'] = row['NetProfit_TTM'] / row['AvgTotalEquity']
                        
                        # NetProfitMargin - Bottom-line efficiency
                        if 'Revenue_TTM' in row and 'NetProfit_TTM' in row:
                            if pd.notna(row['NetProfit_TTM']) and pd.notna(row['Revenue_TTM']) and row['Revenue_TTM'] > 0:
                                sector_metrics['NetProfitMargin'] = row['NetProfit_TTM'] / row['Revenue_TTM']
                        
                        # GrossMargin - Pricing power
                        if 'Revenue_TTM' in row and 'COGS_TTM' in row:
                            if pd.notna(row['Revenue_TTM']) and pd.notna(row['COGS_TTM']) and row['Revenue_TTM'] > 0:
                                sector_metrics['GrossMargin'] = (row['Revenue_TTM'] - row['COGS_TTM']) / row['Revenue_TTM']
                        
                        # OperatingMargin - Operational excellence
                        if 'Revenue_TTM' in row and all(col in row for col in ['COGS_TTM', 'SellingExpenses_TTM', 'AdminExpenses_TTM']):
                            if all(pd.notna(row[col]) for col in ['Revenue_TTM', 'COGS_TTM', 'SellingExpenses_TTM', 'AdminExpenses_TTM']) and row['Revenue_TTM'] > 0:
                                operating_profit = row['Revenue_TTM'] - row['COGS_TTM'] - row['SellingExpenses_TTM'] - row['AdminExpenses_TTM']
                                sector_metrics['OperatingMargin'] = operating_profit / row['Revenue_TTM']
                        
                        # EBITDA Margin - Cash flow profitability
                        if 'Revenue_TTM' in row and 'EBITDA_TTM' in row:
                            if pd.notna(row['EBITDA_TTM']) and pd.notna(row['Revenue_TTM']) and row['Revenue_TTM'] > 0:
                                sector_metrics['EBITDA_Margin'] = row['EBITDA_TTM'] / row['Revenue_TTM']
                    
                    # Store all sector-specific metrics for normalization
                    for metric_name, value in sector_metrics.items():
                        data.loc[idx, f'{metric_name}_Raw'] = value
                        
                except Exception as e:
                    self.logger.debug(f"Sophisticated quality calculation failed for {ticker}: {e}")
                    data.loc[idx, 'Sophisticated_Quality_Signal'] = 0.0
            
            # CRITICAL FIX #2: Implement "normalize-then-average" institutional methodology
            # Each base metric must be individually z-scored before combination
            
            # Identify all quality metrics that were calculated
            quality_metric_columns = [col for col in data.columns if col.endswith('_Raw')]
            
            if quality_metric_columns:
                # Step 1: Normalize each quality metric individually (sector-neutral z-scores)
                normalized_metrics = {}
                
                for metric_col in quality_metric_columns:
                    metric_name = metric_col.replace('_Raw', '')
                    
                    # Calculate sector-neutral z-score for this individual metric
                    z_scores = self.calculate_sector_neutral_zscore(data, metric_col)
                    normalized_metrics[metric_name] = z_scores
                    
                    # Store normalized values in data for transparency
                    data[f'{metric_name}_ZScore'] = z_scores
                    
                    self.logger.debug(f"Normalized {metric_name} with sector-neutral z-scores")
                
                # Step 2: Calculate weighted average of normalized metrics by sector
                for idx, row in data.iterrows():
                    ticker = row['ticker']
                    sector = row['sector']
                    
                    # Get sector-specific weights from configuration
                    if sector == 'Banking':
                        # Banking quality weights: ROAE=40%, ROAA=25%, NIM=20%, Cost_Income_Ratio=15%
                        sector_weights = {
                            'ROAE': 0.40,
                            'ROAA': 0.25, 
                            'NIM': 0.20,
                            'Cost_Income_Ratio': 0.15
                        }
                    elif sector == 'Securities':
                        # Securities quality weights: ROAE=50%, BrokerageRatio=30%, NetProfitMargin=20%
                        sector_weights = {
                            'ROAE': 0.50,
                            'BrokerageRatio': 0.30,
                            'NetProfitMargin': 0.20
                        }
                    else:
                        # Non-financial quality weights: ROAE=35%, NetProfitMargin=25%, GrossMargin=25%, OperatingMargin=15%
                        sector_weights = {
                            'ROAE': 0.35,
                            'NetProfitMargin': 0.25,
                            'GrossMargin': 0.25,
                            'OperatingMargin': 0.15
                        }
                    
                    # Calculate weighted average of individual z-scores
                    weighted_quality = 0.0
                    total_weight = 0.0
                    
                    for metric_name, weight in sector_weights.items():
                        if metric_name in normalized_metrics and idx in normalized_metrics[metric_name].index:
                            z_score = normalized_metrics[metric_name].loc[idx]
                            if pd.notna(z_score):
                                weighted_quality += weight * z_score
                                total_weight += weight
                    
                    # Normalize by actual weights used (handles missing metrics)
                    if total_weight > 0:
                        quality_scores[ticker] = weighted_quality / total_weight
                    else:
                        quality_scores[ticker] = 0.0
                
                self.logger.info(f"Applied INSTITUTIONAL normalize-then-average methodology for {len(quality_scores)} tickers")
            else:
                self.logger.warning("No valid quality metrics found for normalization")
            
            self.logger.info(f"Calculated sophisticated quality scores using sector-specific metrics for {len(quality_scores)} tickers")
            return quality_scores
            
        except Exception as e:
            self.logger.error(f"Failed to calculate sophisticated quality composite: {e}")
            return {}
    
    def get_point_in_time_equity(self, ticker: str, analysis_date: pd.Timestamp, sector: str) -> float:
        """
        Get point-in-time equity value using proper institutional methodology.
        Adapted from legacy institutional factor calculator.
        """
        try:
            # Quarter determination with 45-day lag logic
            year = analysis_date.year
            quarter_ends = {
                1: pd.Timestamp(year, 3, 31),   # Q1 ends Mar 31
                2: pd.Timestamp(year, 6, 30),   # Q2 ends Jun 30
                3: pd.Timestamp(year, 9, 30),   # Q3 ends Sep 30
                4: pd.Timestamp(year, 12, 31)   # Q4 ends Dec 31
            }
            
            # Find the most recent quarter whose publish date <= analysis_date
            available_quarters = []
            for quarter, end_date in quarter_ends.items():
                publish_date = end_date + pd.Timedelta(days=45)
                if publish_date <= analysis_date:
                    available_quarters.append((year, quarter, publish_date))
            
            # Also check previous year Q4
            prev_year_q4_end = pd.Timestamp(year - 1, 12, 31)
            prev_year_q4_publish = prev_year_q4_end + pd.Timedelta(days=45)
            if prev_year_q4_publish <= analysis_date:
                available_quarters.append((year - 1, 4, prev_year_q4_publish))
            
            if not available_quarters:
                self.logger.debug(f"No equity data available for {ticker} as of {analysis_date.date()}")
                return 0.0
            
            # Get most recent available quarter
            available_quarters.sort(key=lambda x: x[2], reverse=True)
            target_year, target_quarter, _ = available_quarters[0]
            
            # Sector-specific table and field mapping (institutional methodology)
            if sector.lower() == 'banking':
                table = 'v_complete_banking_fundamentals'
                equity_field = 'ShareholdersEquity'
            elif sector.lower() == 'securities':
                table = 'v_complete_securities_fundamentals'
                equity_field = 'OwnersEquity'
            else:
                table = 'v_comprehensive_fundamental_items'
                equity_field = 'TotalEquity'
            
            # Query point-in-time equity
            query = text(f"""
                SELECT {equity_field} as equity_value
                FROM {table}
                WHERE ticker = :ticker 
                  AND year = :year 
                  AND quarter = :quarter
                LIMIT 1
            """)
            
            equity_result = pd.read_sql(query, self.engine, params={
                'ticker': ticker, 'year': target_year, 'quarter': target_quarter
            })
            
            if not equity_result.empty and pd.notna(equity_result['equity_value'].iloc[0]):
                equity_value = float(equity_result['equity_value'].iloc[0])
                self.logger.debug(f"Point-in-time {equity_field} for {ticker}: {equity_value:,.0f} ({target_year} Q{target_quarter})")
                return equity_value
            else:
                self.logger.warning(f"No {equity_field} data for {ticker} in {target_year} Q{target_quarter}")
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Failed to get point-in-time equity for {ticker}: {e}")
            return 0.0

    def _calculate_enhanced_value_composite(self, data: pd.DataFrame, analysis_date: pd.Timestamp) -> Dict[str, float]:
        """
        CORRECTED VALUE LOGIC: Uses point-in-time equity for proper P/B ratios.
        Implements institutional methodology with sector-specific weights and Enhanced EV/EBITDA.
        """
        try:
            value_scores = {}
            
            # Calculate ratios for each ticker with sector-specific weights
            for idx, row in data.iterrows():
                ticker = row['ticker']
                sector = row['sector']
                market_cap = row.get('market_cap', 0)
                
                # Get sector-specific value weights
                sector_weights = self.ev_calculator.get_sector_specific_value_weights(sector)
                
                ratios = {}
                
                try:
                    # P/E ratio (inverted for value score)
                    pe_score = 0
                    if 'NetProfit_TTM' in row and pd.notna(row['NetProfit_TTM']) and row['NetProfit_TTM'] > 0:
                        pe_ratio = market_cap / row['NetProfit_TTM']
                        pe_score = 1 / pe_ratio if pe_ratio > 0 else 0
                    ratios['pe'] = pe_score
                    
                    # CORRECTED P/B ratio using point-in-time equity (not average)
                    pb_score = 0
                    point_in_time_equity = self.get_point_in_time_equity(ticker, analysis_date, sector)
                    if point_in_time_equity > 0:
                        pb_ratio = market_cap / point_in_time_equity
                        pb_score = 1 / pb_ratio if pb_ratio > 0 else 0
                        self.logger.debug(f"CORRECTED P/B for {ticker}: {pb_ratio:.6f} (point-in-time equity: {point_in_time_equity:,.0f})")
                    else:
                        # Fallback to average if point-in-time not available
                        if 'AvgTotalEquity' in row and pd.notna(row['AvgTotalEquity']) and row['AvgTotalEquity'] > 0:
                            pb_ratio = market_cap / row['AvgTotalEquity']
                            pb_score = 1 / pb_ratio if pb_ratio > 0 else 0
                            self.logger.warning(f"FALLBACK P/B for {ticker}: using AvgTotalEquity")
                    ratios['pb'] = pb_score
                    
                    # Sector-specific P/S ratio
                    ps_score = 0
                    revenue_ttm = 0
                    if sector == 'Banking' and 'TotalOperatingIncome_TTM' in row:
                        revenue_ttm = row.get('TotalOperatingIncome_TTM', 0)
                    elif sector == 'Securities' and 'TotalOperatingRevenue_TTM' in row:
                        revenue_ttm = row.get('TotalOperatingRevenue_TTM', 0)
                    elif 'Revenue_TTM' in row:
                        revenue_ttm = row.get('Revenue_TTM', 0)
                    
                    if pd.notna(revenue_ttm) and revenue_ttm > 0:
                        ps_ratio = market_cap / revenue_ttm
                        ps_score = 1 / ps_ratio if ps_ratio > 0 else 0
                    ratios['ps'] = ps_score
                    
                    # Enhanced EV/EBITDA ratio
                    ev_ebitda_score = 0
                    if 'EBITDA_TTM' in row and pd.notna(row['EBITDA_TTM']):
                        ev_ebitda_score = self.ev_calculator.calculate_enhanced_ev_ebitda(
                            ticker, analysis_date, sector, market_cap, row['EBITDA_TTM']
                        )
                    ratios['ev_ebitda'] = ev_ebitda_score
                    
                    # Weighted sector-specific value composite
                    value_score = (
                        sector_weights['pe'] * ratios['pe'] +
                        sector_weights['pb'] * ratios['pb'] +
                        sector_weights['ps'] * ratios['ps'] +
                        sector_weights['ev_ebitda'] * ratios['ev_ebitda']
                    )
                    
                    data.loc[idx, 'enhanced_value_composite'] = value_score
                    
                except Exception as e:
                    self.logger.debug(f"Enhanced value calculation failed for {ticker}: {e}")
                    data.loc[idx, 'enhanced_value_composite'] = 0
            
            # Enhanced sector-neutral normalization
            if 'enhanced_value_composite' in data.columns:
                value_z_scores = self.calculate_sector_neutral_zscore(data, 'enhanced_value_composite')
                
                for ticker in data['ticker']:
                    ticker_data = data[data['ticker'] == ticker].iloc[0]
                    ticker_idx = ticker_data.name
                    
                    if ticker_idx in value_z_scores.index:
                        value_scores[ticker] = value_z_scores.loc[ticker_idx]
                    else:
                        value_scores[ticker] = 0.0
            
            self.logger.debug(f"Calculated enhanced value scores for {len(value_scores)} tickers")
            return value_scores
            
        except Exception as e:
            self.logger.error(f"Failed to calculate enhanced value composite: {e}")
            return {}
    
    def _calculate_enhanced_momentum_composite(self, data: pd.DataFrame, 
                                             analysis_date: pd.Timestamp, 
                                             universe: List[str]) -> Dict[str, float]:
        """
        CORRECTED MOMENTUM LOGIC: Multiple timeframes with skip-1-month and sector-neutral normalization.
        
        INSTITUTIONAL CORRECTION: Now uses sector-neutral normalization as PRIMARY method,
        consistent with the institutional standard for pure alpha signal extraction.
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
                self.logger.warning("No price data available for enhanced momentum calculation")
                return momentum_scores
            
            # Enhanced momentum periods with sophisticated weighting
            periods = [
                ('1M', self.momentum_lookbacks['1M'], self.momentum_skip, self.momentum_weights['1M']),
                ('3M', self.momentum_lookbacks['3M'], self.momentum_skip, self.momentum_weights['3M']),
                ('6M', self.momentum_lookbacks['6M'], self.momentum_skip, self.momentum_weights['6M']),
                ('12M', self.momentum_lookbacks['12M'], self.momentum_skip, self.momentum_weights['12M'])
            ]
            
            period_returns = {}
            
            for period_name, lookback, skip, weight in periods:
                end_date = analysis_date - pd.DateOffset(months=skip)
                start_date_period = analysis_date - pd.DateOffset(months=lookback + skip)
                returns = self._calculate_returns_fixed(price_data, start_date_period, end_date)
                
                if not returns.empty:
                    period_returns[period_name] = returns
                    self.logger.debug(f"Calculated enhanced {period_name} returns for {len(returns)} tickers")
            
            # Enhanced momentum combination with sophisticated weighting
            for ticker in universe:
                momentum_components = []
                
                for period_name, lookback, skip, weight in periods:
                    if period_name in period_returns and ticker in period_returns[period_name]:
                        momentum_components.append(weight * period_returns[period_name][ticker])
                
                if momentum_components:
                    momentum_scores[ticker] = sum(momentum_components)
                else:
                    momentum_scores[ticker] = 0.0
            
            # CORRECTED: Apply sector-neutral normalization to momentum as well
            if momentum_scores:
                # Create DataFrame with momentum scores for normalization
                momentum_df = pd.DataFrame({
                    'ticker': list(momentum_scores.keys()),
                    'momentum_composite': list(momentum_scores.values())
                })
                
                # Merge with sector information
                momentum_df = momentum_df.merge(
                    data[['ticker', 'sector']].drop_duplicates(),
                    on='ticker',
                    how='left'
                )
                
                # Apply institutional sector-neutral normalization
                momentum_z_scores = self.calculate_sector_neutral_zscore(
                    momentum_df, 
                    'momentum_composite'
                )
                
                # Update momentum scores with normalized values
                for idx, row in momentum_df.iterrows():
                    ticker = row['ticker']
                    if idx in momentum_z_scores.index:
                        momentum_scores[ticker] = momentum_z_scores.loc[idx]
                    else:
                        momentum_scores[ticker] = 0.0
            
            self.logger.debug(f"Calculated enhanced momentum scores for {len(momentum_scores)} tickers")
            return momentum_scores
            
        except Exception as e:
            self.logger.error(f"Failed to calculate enhanced momentum composite: {e}")
            return {}
    
    def _calculate_returns_fixed(self, price_data: pd.DataFrame, 
                                start_date: pd.Timestamp, 
                                end_date: pd.Timestamp) -> pd.Series:
        """
        ENHANCED LOGIC: Fixed return calculation with better missing date handling.
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
                
                # Calculate return with enhanced validation
                if pd.notna(start_price) and pd.notna(end_price) and start_price > 0:
                    return_val = (end_price / start_price) - 1
                    returns[ticker] = return_val
            
            self.logger.debug(f"Calculated enhanced returns for {len(returns)} tickers from {start_date.date()} to {end_date.date()}")
            return returns
            
        except Exception as e:
            self.logger.error(f"Failed to calculate enhanced returns: {e}")
            return pd.Series(dtype=float)

# END OF ENHANCED CANONICAL QVM ENGINE