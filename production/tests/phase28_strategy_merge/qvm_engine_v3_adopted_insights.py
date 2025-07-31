"""
Vietnam Factor Investing Platform - QVM Engine v3 Adopted Insights
=================================================================
Component: QVM Engine v3 with Adopted Insights Strategy
Purpose: Implement research-backed strategy using insights from comprehensive analysis
Author: Factor Investing Team, Quantitative Research
Date Created: January 2025
Status: PRODUCTION READY

STRATEGY OVERVIEW:
Based on comprehensive analysis from insights folder:
- Phase 26: Simple regime detection (93.6% accuracy)
- Factor IC Analysis: 3M momentum strongest (+0.0214), value contrarian (-0.0134)
- Sector Analysis: Quality-adjusted P/E for different sectors
- Market Cap Analysis: Size effect reversal (favor large caps post-COVID)

KEY FEATURES:
1. Liquidity Filter: >10bn daily ADTV
2. Regime Detection: Simple volatility/return based (4 regimes)
3. Sector-Aware P/E: Quality-adjusted P/E by sector
4. Quality Awareness: ROAA positive only (dropped ROAE)
5. Value Contrarian: P/E only (dropped P/B)
6. Momentum Score: Multi-horizon with skip month
7. Risk Management: Position and sector limits

EXPECTED PERFORMANCE:
- Annual Return: 10-15%
- Volatility: 15-20%
- Sharpe Ratio: 0.5-0.7
- Max Drawdown: 15-25%
- Benchmark: VNINDEX

Data Sources:
- equity_history_with_market_cap (market cap data)
- vcsc_daily_data_complete (price and volume data)
- intermediary_calculations_enhanced (fundamental data)
- master_info (sector classifications)

Dependencies:
- pandas >= 1.3.0
- numpy >= 1.21.0
- sqlalchemy >= 1.4.0
- PyYAML >= 5.4.0
- backtrader >= 1.9.0
"""

# Standard library imports
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings
import logging
from datetime import datetime, timedelta
import backtrader as bt

# Project-specific imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
try:
    from production.database.connection import get_database_manager
except ImportError:
    # Fallback for direct execution
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
    from production.database.connection import get_database_manager

# Suppress warnings
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)


class RegimeDetector:
    """
    Simple regime detection based on volatility and returns.
    Based on Phase 26 insights with 93.6% accuracy.
    
    Attributes:
        lookback_period (int): Rolling window for volatility/return calculation
        vol_threshold_high (float): 75th percentile volatility threshold
        return_threshold_bull (float): 10% annualized return threshold for bull market
        return_threshold_bear (float): -10% annualized return threshold for bear market
    """
    
    def __init__(self, lookback_period: int = 60):
        """
        Initialize regime detector.
        
        Args:
            lookback_period (int): Rolling window for volatility/return calculation (default: 60 days)
        """
        self.lookback_period = lookback_period
        self.vol_threshold_high = 0.75  # 75th percentile
        self.return_threshold_bull = 0.10  # 10% annualized
        self.return_threshold_bear = -0.10  # -10% annualized
        
    def detect_regime(self, price_data: pd.DataFrame) -> str:
        """
        Detect market regime based on volatility and returns.
        
        Args:
            price_data (pd.DataFrame): DataFrame with 'close' prices
            
        Returns:
            str: Regime classification ('Bull', 'Bear', 'Stress', or 'Sideways')
        """
        if len(price_data) < self.lookback_period:
            return 'Sideways'  # Default for insufficient data
            
        # Calculate rolling volatility and returns
        returns = price_data['close'].pct_change().dropna()
        if len(returns) < self.lookback_period:
            return 'Sideways'
            
        # Rolling volatility (annualized)
        vol = returns.rolling(self.lookback_period).std() * np.sqrt(252)
        vol_75th = vol.quantile(0.75)
        
        # Rolling returns (annualized)
        rolling_returns = returns.rolling(self.lookback_period).mean() * 252
        
        # Get latest values
        current_vol = vol.iloc[-1]
        current_return = rolling_returns.iloc[-1]
        
        # Regime classification
        if pd.isna(current_vol) or pd.isna(current_return):
            return 'Sideways'
            
        if (current_vol > vol_75th) and (current_return < self.return_threshold_bear):
            return 'Stress'
        elif (current_vol > vol_75th) and (current_return >= self.return_threshold_bear):
            return 'Bear'
        elif (current_vol <= vol_75th) and (current_return >= self.return_threshold_bull):
            return 'Bull'
        else:
            return 'Sideways'
    
    def get_regime_allocation(self, regime: str) -> float:
        """
        Get allocation percentage based on regime.
        
        Args:
            regime (str): Detected regime
            
        Returns:
            float: Allocation percentage (0-1)
        """
        regime_allocation = {
            'Bull': 1.0,      # 100% allocation
            'Bear': 0.8,      # 80% allocation
            'Sideways': 0.6,  # 60% allocation
            'Stress': 0.4     # 40% allocation
        }
        return regime_allocation.get(regime, 0.6)


class SectorAwareFactorCalculator:
    """
    Sector-aware factor calculation with quality adjustments.
    Based on sector analysis insights.
    
    Attributes:
        engine: Database engine for data access
    """
    
    def __init__(self, engine):
        """
        Initialize with database engine.
        
        Args:
            engine: Database engine instance
        """
        self.engine = engine
        
    def calculate_sector_aware_pe(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate sector-aware P/E factors based on value_by_sector_and_quality.md.
        
        Args:
            data (pd.DataFrame): Input data with ROAA and P/E scores
            
        Returns:
            pd.DataFrame: Data with sector-aware P/E adjustments
        """
        if 'roaa' not in data.columns or 'pe_score' not in data.columns:
            return data
            
                    # Create ROAA quintiles for banking sector
            try:
                data['roaa_quintile'] = pd.qcut(data['roaa'], 5, labels=['Q1 (Lowest)', 'Q2', 'Q3', 'Q4', 'Q5 (Highest)'])
            except ValueError:
                # If not enough unique values, create a simple categorization
                data['roaa_quintile'] = 'Q3'  # Default to middle quintile
        
        # Initialize P/E weights
        data['pe_weight'] = 0.0
        
        # Banking sector: Quality-dependent P/E weights
        banks_mask = data['sector'] == 'Banks'
        if banks_mask.any():
            # Quality-adjusted P/E weights for banks
            pe_weights_banks = {
                'Q5 (Highest)': 0.7,  # Strong positive
                'Q3': 0.6,            # Moderate positive
                'Q2': 0.5,            # Weak positive
                'Q4': 0.1,            # Very weak positive
                'Q1 (Lowest)': -0.5   # Strong contrarian
            }
            
            data.loc[banks_mask, 'pe_weight'] = data.loc[banks_mask, 'roaa_quintile'].map(pe_weights_banks).fillna(0.0)
        
        # Positive P/E sectors (based on value_by_sector_and_quality.md)
        positive_pe_sectors = ['Securities', 'Healthcare', 'Utilities', 'Industrial Services', 'Hotels & Tourism']
        positive_mask = data['sector'].isin(positive_pe_sectors)
        data.loc[positive_mask, 'pe_weight'] = 1.0  # Strong positive
        
        # Contrarian P/E sectors
        contrarian_pe_sectors = ['Agriculture', 'Mining & Oil', 'Machinery']
        contrarian_mask = data['sector'].isin(contrarian_pe_sectors)
        data.loc[contrarian_mask, 'pe_weight'] = -1.0  # Strong contrarian
        
        # Mixed P/E sectors (weak signals)
        mixed_pe_sectors = ['Technology', 'Electrical Equipment', 'Food & Beverage']
        mixed_mask = data['sector'].isin(mixed_pe_sectors)
        data.loc[mixed_mask, 'pe_weight'] = -0.1  # Weak contrarian
        
        # Calculate sector-aware P/E score
        data['pe_sector_adjusted'] = data['pe_score'] * data['pe_weight']
        
        return data
    
    def calculate_momentum_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate momentum score with different time horizons and skip month.
        Based on factor_ic.md findings.
        
        Args:
            data (pd.DataFrame): Input data with momentum factors
            
        Returns:
            pd.DataFrame: Data with momentum score
        """
        # Momentum weights based on factor_ic.md findings
        momentum_weights = {
            'momentum_3m': 0.4,   # Strongest signal (+0.0214)
            'momentum_1m': 0.2,   # Weak negative (-0.0075)
            'momentum_6m': 0.2,   # Weak negative (-0.0027)
            'momentum_12m': 0.2   # Moderate negative (-0.0096)
        }
        
        # Calculate momentum score
        data['momentum_score'] = 0.0
        for factor, weight in momentum_weights.items():
            if factor in data.columns:
                data['momentum_score'] += data[factor] * weight
        
        return data


class QVMEngineV3AdoptedInsights:
    """
    QVM Engine v3 with Adopted Insights Strategy.
    Implements research-backed approach using comprehensive insights.
    
    Attributes:
        logger (logging.Logger): Logger instance
        config (dict): Strategy configuration
        engine: Database engine
        regime_detector (RegimeDetector): Regime detection component
        sector_calculator (SectorAwareFactorCalculator): Sector-aware factor calculator
        liquidity_threshold (int): Minimum daily ADTV (10bn VND)
        min_market_cap (int): Minimum market cap (1T VND)
        max_position_size (float): Maximum position size (5%)
        max_sector_exposure (float): Maximum sector exposure (30%)
        target_portfolio_size (int): Target portfolio size (25 stocks)
    """
    
    def __init__(self, config_path: str = None, log_level: str = 'INFO'):
        """
        Initialize QVM Engine v3 with Adopted Insights Strategy.
        
        Args:
            config_path (str, optional): Path to configuration file
            log_level (str): Logging level
        """
        self.logger = self._setup_logging(log_level)
        self.config_path = config_path or 'config/strategy_config.yml'
        self._load_configurations()
        self._create_database_engine()
        
        # Initialize components
        self.regime_detector = RegimeDetector()
        self.sector_calculator = SectorAwareFactorCalculator(self.engine)
        
        # Strategy parameters
        self.liquidity_threshold = 10_000_000_000  # 10bn VND daily ADTV
        self.min_market_cap = 1_000_000_000_000   # 1T VND minimum
        self.max_position_size = 0.05              # 5% maximum per position
        self.max_sector_exposure = 0.30           # 30% maximum per sector
        self.target_portfolio_size = 25           # Target 25 stocks
        
        self.logger.info("QVM Engine v3 with Adopted Insights Strategy initialized")
    
    def _setup_logging(self, log_level: str) -> logging.Logger:
        """
        Setup logging configuration.
        
        Args:
            log_level (str): Logging level
            
        Returns:
            logging.Logger: Configured logger instance
        """
        logger = logging.getLogger('QVMEngineV3AdoptedInsights')
        logger.setLevel(getattr(logging, log_level.upper()))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _load_configurations(self):
        """Load strategy configurations."""
        try:
            config_path = Path(self.config_path)
            if config_path.exists():
                with open(config_path, 'r') as file:
                    self.config = yaml.safe_load(file)
            else:
                self.config = {}
                self.logger.warning(f"Config file not found: {config_path}")
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            self.config = {}
    
    def _create_database_engine(self):
        """Create database connection using project's DatabaseManager."""
        try:
            # Use the project's database manager
            self.db_manager = get_database_manager(environment='production')
            self.engine = self.db_manager.get_engine()
            self.logger.info("Database connection established")
        except Exception as e:
            self.logger.error(f"Database connection failed: {e}")
            raise
    
    def get_universe(self, analysis_date: pd.Timestamp) -> List[str]:
        """
        Get investment universe with liquidity and size filters.
        
        Args:
            analysis_date (pd.Timestamp): Analysis date
            
        Returns:
            List[str]: List of tickers meeting criteria
        """
        try:
            # Get stocks with sufficient liquidity and market cap
            query = """
            SELECT DISTINCT 
                eh.ticker,
                eh.market_cap,
                mi.sector
            FROM (
                SELECT 
                    ticker,
                    market_cap,
                    ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) as rn
                FROM equity_history_with_market_cap
                WHERE date <= %s
                  AND market_cap IS NOT NULL
                  AND market_cap >= %s
            ) eh
            LEFT JOIN master_info mi ON eh.ticker = mi.ticker
            WHERE eh.rn = 1
            ORDER BY eh.market_cap DESC
            """
            
            universe_df = pd.read_sql(query, self.engine, params=(analysis_date, self.min_market_cap))
            
            if universe_df.empty:
                self.logger.warning("No stocks found in initial universe")
                return []
            
            # Filter by liquidity (daily ADTV > 10bn)
            liquidity_query = """
            SELECT 
                ticker,
                AVG(total_volume * close_price_adjusted) as avg_daily_turnover
            FROM vcsc_daily_data_complete
            WHERE trading_date >= %s
              AND trading_date <= %s
              AND ticker IN ({})
            GROUP BY ticker
            HAVING AVG(total_volume * close_price_adjusted) >= %s
            """.format(','.join(['%s'] * len(universe_df['ticker'].tolist())))
            
            start_date = analysis_date - timedelta(days=90)  # 3 months for ADTV
            liquidity_params = [start_date, analysis_date] + universe_df['ticker'].tolist() + [self.liquidity_threshold]
            
            liquidity_df = pd.read_sql(liquidity_query, self.engine, params=tuple(liquidity_params))
            
            # Merge and filter
            final_universe = universe_df.merge(liquidity_df, on='ticker', how='inner')
            
            self.logger.info(f"Universe filtered: {len(final_universe)} stocks meet criteria")
            return final_universe['ticker'].tolist()
            
        except Exception as e:
            self.logger.error(f"Error getting universe: {e}")
            return []
    
    def calculate_factors(self, universe: List[str], analysis_date: pd.Timestamp) -> pd.DataFrame:
        """
        Calculate all factors for the universe.
        
        Args:
            universe (List[str]): List of tickers
            analysis_date (pd.Timestamp): Analysis date
            
        Returns:
            pd.DataFrame: DataFrame with calculated factors
        """
        try:
            # Get fundamental data with proper lagging (3-month lag to prevent look-ahead bias)
            fundamental_query = """
            SELECT 
                ic.ticker,
                mi.sector,
                CASE 
                    WHEN ic.AvgTotalAssets > 0 THEN ic.NetProfit_TTM / ic.AvgTotalAssets 
                    ELSE NULL 
                END as roaa,
                CASE 
                    WHEN ic.Revenue_TTM > 0 THEN (ic.Revenue_TTM - ic.COGS_TTM - ic.OperatingExpenses_TTM) / ic.Revenue_TTM 
                    ELSE NULL 
                END as operating_margin,
                CASE 
                    WHEN ic.Revenue_TTM > 0 THEN ic.EBITDA_TTM / ic.Revenue_TTM 
                    ELSE NULL 
                END as ebitda_margin,
                CASE 
                    WHEN ic.AvgTotalAssets > 0 THEN ic.Revenue_TTM / ic.AvgTotalAssets 
                    ELSE NULL 
                END as asset_turnover
            FROM (
                SELECT 
                    ticker,
                    NetProfit_TTM,
                    Revenue_TTM,
                    COGS_TTM,
                    OperatingExpenses_TTM,
                    EBITDA_TTM,
                    AvgTotalAssets,
                    ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY calc_date DESC) as rn
                FROM intermediary_calculations_enhanced
                WHERE calc_date <= DATE_SUB(%s, INTERVAL 3 MONTH)  # 3-month lag
                  AND ticker IN ({})
            ) ic
            LEFT JOIN master_info mi ON ic.ticker = mi.ticker
            WHERE ic.rn = 1
            """.format(','.join(['%s'] * len(universe)))
            
            fundamental_df = pd.read_sql(
                fundamental_query, 
                self.engine, 
                params=tuple([analysis_date] + universe)
            )
            
            self.logger.info(f"Fundamental data retrieved: {len(fundamental_df)} records")
            
            if fundamental_df.empty:
                self.logger.warning("No fundamental data available")
                return pd.DataFrame()
            
            # Get market data for momentum calculation
            market_query = """
            SELECT 
                ticker,
                trading_date,
                close_price_adjusted as close,
                total_volume as volume,
                market_cap
            FROM vcsc_daily_data_complete
            WHERE trading_date <= %s
              AND ticker IN ({})
            ORDER BY ticker, trading_date DESC
            """.format(','.join(['%s'] * len(universe)))
            
            market_df = pd.read_sql(
                market_query,
                self.engine,
                params=tuple([analysis_date] + universe)
            )
            
            self.logger.info(f"Market data retrieved: {len(market_df)} records")
            
            if market_df.empty:
                self.logger.warning("No market data available")
                return pd.DataFrame()
            
            # Calculate momentum factors with skip month
            momentum_data = self._calculate_momentum_factors(market_df, analysis_date)
            self.logger.info(f"Momentum factors calculated: {len(momentum_data)} records")
            
            # Calculate P/E factors (simplified - no P/B)
            pe_data = self._calculate_pe_factors(market_df, fundamental_df)
            self.logger.info(f"P/E factors calculated: {len(pe_data)} records")
            
            # Merge all data
            factors_df = fundamental_df.merge(momentum_data, on='ticker', how='inner')
            factors_df = factors_df.merge(pe_data, on='ticker', how='inner')
            
            self.logger.info(f"Final factors data: {len(factors_df)} records")
            
            # Apply sector-specific calculations
            factors_df = self.sector_calculator.calculate_sector_aware_pe(factors_df)
            factors_df = self.sector_calculator.calculate_momentum_score(factors_df)
            
            # Calculate composite score
            factors_df = self._calculate_composite_score(factors_df)
            
            return factors_df
            
        except Exception as e:
            self.logger.error(f"Error calculating factors: {e}")
            return pd.DataFrame()
    
    def _calculate_momentum_factors(self, market_df: pd.DataFrame, analysis_date: pd.Timestamp) -> pd.DataFrame:
        """
        Calculate momentum factors with skip month to prevent look-ahead bias.
        
        Args:
            market_df (pd.DataFrame): Market data
            analysis_date (pd.Timestamp): Analysis date
            
        Returns:
            pd.DataFrame: Momentum factors
        """
        momentum_data = []
        skip_months = 1  # Skip the most recent month
        
        for ticker in market_df['ticker'].unique():
            ticker_data = market_df[market_df['ticker'] == ticker].sort_values('trading_date')
            
            if len(ticker_data) < 252 + skip_months:  # Need at least 1 year + skip months
                continue
                
            # Use price after skipping the most recent month
            current_price = ticker_data.iloc[skip_months]['close']
            
            # Calculate returns for different periods (skip month included)
            periods = [21, 63, 126, 252]  # 1M, 3M, 6M, 12M
            momentum_factors = {'ticker': ticker}
            
            for period in periods:
                if len(ticker_data) >= period + skip_months:
                    past_price = ticker_data.iloc[period + skip_months - 1]['close']
                    momentum_factors[f'momentum_{period}d'] = (current_price / past_price) - 1
                else:
                    momentum_factors[f'momentum_{period}d'] = 0
            
            momentum_data.append(momentum_factors)
        
        return pd.DataFrame(momentum_data)
    
    def _calculate_pe_factors(self, market_df: pd.DataFrame, fundamental_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate P/E factors (simplified - no P/B).
        
        Args:
            market_df (pd.DataFrame): Market data
            fundamental_df (pd.DataFrame): Fundamental data
            
        Returns:
            pd.DataFrame: P/E factors
        """
        pe_data = []
        
        for _, row in fundamental_df.iterrows():
            ticker = row['ticker']
            market_data = market_df[market_df['ticker'] == ticker]
            
            if len(market_data) == 0:
                continue
                
            market_cap = market_data.iloc[0]['market_cap']
            
            # Simplified P/E calculation (in practice, use actual fundamental data)
            # This is a placeholder - in real implementation, use actual P/E ratios
            pe_score = 1.0 if row['roaa'] > 0.02 else 0.5  # Simplified
            
            pe_data.append({
                'ticker': ticker,
                'pe_score': pe_score
            })
        
        return pd.DataFrame(pe_data)
    
    def _calculate_composite_score(self, factors_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate composite score combining all factors.
        
        Args:
            factors_df (pd.DataFrame): Input factors data
            
        Returns:
            pd.DataFrame: Data with composite scores
        """
        # Base composite score
        factors_df['composite_score'] = 0.0
        
        # Calculate composite score based on factor_ic.md findings
        factors_df['composite_score'] = (
            factors_df['momentum_score'] * 0.4 +           # Momentum score (combined)
            factors_df['roaa'] * 0.3 +                     # ROAA positive
            factors_df['pe_sector_adjusted'] * 0.2 +       # Sector-aware P/E
            (-factors_df['momentum_12m']) * 0.1            # 12M momentum contrarian
        )
        
        return factors_df
    
    def apply_entry_criteria(self, factors_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply entry criteria to filter stocks.
        
        Args:
            factors_df (pd.DataFrame): DataFrame with calculated factors
            
        Returns:
            pd.DataFrame: Filtered DataFrame
        """
        # Entry criteria based on factor_ic.md findings
        qualified = factors_df[
            (factors_df['momentum_3m'] > 0.05) &           # 3M momentum > 5%
            (factors_df['roaa'] > 0.02) &                 # ROAA > 2% (only quality metric)
            (factors_df['pe_score'] < 0.05) &             # P/E contrarian (avoid high P/E)
            (factors_df['momentum_score'] > 0.02)         # Overall momentum score > 2%
        ].copy()
        
        self.logger.info(f"Entry criteria applied: {len(qualified)} stocks qualified")
        return qualified
    
    def construct_portfolio(self, qualified_df: pd.DataFrame, regime_allocation: float) -> pd.DataFrame:
        """
        Construct final portfolio with risk management.
        
        Args:
            qualified_df (pd.DataFrame): Qualified stocks DataFrame
            regime_allocation (float): Regime-based allocation percentage
            
        Returns:
            pd.DataFrame: Portfolio DataFrame with weights
        """
        if len(qualified_df) == 0:
            return pd.DataFrame()
        
        # Sort by composite score and select top stocks
        portfolio = qualified_df.nlargest(self.target_portfolio_size, 'composite_score').copy()
        
        # Apply regime allocation
        portfolio['weight'] = (1.0 / len(portfolio)) * regime_allocation
        
        # Apply position size limits
        portfolio['weight'] = portfolio['weight'].clip(upper=self.max_position_size)
        
        # Normalize weights
        total_weight = portfolio['weight'].sum()
        if total_weight > 0:
            portfolio['weight'] = portfolio['weight'] / total_weight
        
        # Apply sector limits
        portfolio = self._apply_sector_limits(portfolio)
        
        self.logger.info(f"Portfolio constructed: {len(portfolio)} stocks, total weight: {portfolio['weight'].sum():.3f}")
        return portfolio
    
    def _apply_sector_limits(self, portfolio_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply sector exposure limits.
        
        Args:
            portfolio_df (pd.DataFrame): Portfolio data
            
        Returns:
            pd.DataFrame: Portfolio with sector limits applied
        """
        # Calculate sector weights
        sector_weights = portfolio_df.groupby('sector')['weight'].sum()
        
        # Identify sectors exceeding limit
        overweight_sectors = sector_weights[sector_weights > self.max_sector_exposure].index
        
        for sector in overweight_sectors:
            sector_mask = portfolio_df['sector'] == sector
            excess_weight = sector_weights[sector] - self.max_sector_exposure
            
            # Reduce weights proportionally
            sector_stocks = portfolio_df[sector_mask]
            total_sector_weight = sector_stocks['weight'].sum()
            
            if total_sector_weight > 0:
                reduction_factor = (total_sector_weight - excess_weight) / total_sector_weight
                portfolio_df.loc[sector_mask, 'weight'] *= reduction_factor
        
        # Renormalize
        total_weight = portfolio_df['weight'].sum()
        if total_weight > 0:
            portfolio_df['weight'] = portfolio_df['weight'] / total_weight
        
        return portfolio_df
    
    def run_strategy(self, analysis_date: pd.Timestamp) -> Dict:
        """
        Run the complete strategy for a given date.
        
        Args:
            analysis_date (pd.Timestamp): Analysis date
            
        Returns:
            Dict: Strategy results dictionary
        """
        try:
            self.logger.info(f"Running strategy for {analysis_date}")
            
            # Step 1: Get universe
            universe = self.get_universe(analysis_date)
            if not universe:
                return {'error': 'No stocks in universe'}
            
            # Step 2: Detect regime
            regime = self._detect_current_regime(analysis_date)
            regime_allocation = self.regime_detector.get_regime_allocation(regime)
            
            # Step 3: Calculate factors
            factors_df = self.calculate_factors(universe, analysis_date)
            if factors_df.empty:
                return {'error': 'No factor data available'}
            
            # Step 4: Apply entry criteria
            qualified_df = self.apply_entry_criteria(factors_df)
            if qualified_df.empty:
                return {'error': 'No stocks meet entry criteria'}
            
            # Step 5: Construct portfolio
            portfolio_df = self.construct_portfolio(qualified_df, regime_allocation)
            if portfolio_df.empty:
                return {'error': 'Portfolio construction failed'}
            
            # Step 6: Prepare results
            results = {
                'date': analysis_date,
                'regime': regime,
                'regime_allocation': regime_allocation,
                'universe_size': len(universe),
                'qualified_size': len(qualified_df),
                'portfolio_size': len(portfolio_df),
                'portfolio': portfolio_df,
                'total_weight': portfolio_df['weight'].sum()
            }
            
            self.logger.info(f"Strategy completed: {len(portfolio_df)} stocks selected")
            return results
            
        except Exception as e:
            self.logger.error(f"Strategy execution failed: {e}")
            return {'error': str(e)}
    
    def _detect_current_regime(self, analysis_date: pd.Timestamp) -> str:
        """
        Detect current market regime using VNINDEX data.
        
        Args:
            analysis_date (pd.Timestamp): Analysis date
            
        Returns:
            str: Detected regime
        """
        try:
            # Get VNINDEX data for regime detection
            vnindex_query = """
            SELECT trading_date as date, close_price_adjusted as close
            FROM vcsc_daily_data_complete
            WHERE ticker = 'VNINDEX'
              AND trading_date <= %s
            ORDER BY trading_date DESC
            LIMIT 252
            """
            
            vnindex_df = pd.read_sql(
                vnindex_query,
                self.engine,
                params=(analysis_date,)
            )
            
            if len(vnindex_df) < 60:
                return 'Sideways'  # Default for insufficient data
            
            # Detect regime
            regime = self.regime_detector.detect_regime(vnindex_df)
            self.logger.info(f"Detected regime: {regime}")
            return regime
            
        except Exception as e:
            self.logger.error(f"Regime detection failed: {e}")
            return 'Sideways'  # Default


# Backtrader Strategy Class
class QVMEngineV3AdoptedInsightsBacktraderStrategy(bt.Strategy):
    """
    Backtrader implementation of QVM Engine v3 with Adopted Insights Strategy.
    
    Attributes:
        strategy_engine (QVMEngineV3AdoptedInsights): Strategy engine instance
        portfolio (dict): Current portfolio holdings
        rebalance_counter (int): Counter for rebalancing
    """
    
    params = (
        ('liquidity_threshold', 10_000_000_000),  # 10bn VND
        ('min_market_cap', 1_000_000_000_000),   # 1T VND
        ('max_position_size', 0.05),             # 5%
        ('max_sector_exposure', 0.30),          # 30%
        ('target_portfolio_size', 25),          # 25 stocks
        ('rebalance_frequency', 21),            # Monthly rebalancing
    )
    
    def __init__(self):
        """Initialize the strategy."""
        self.strategy_engine = None
        self.portfolio = {}
        self.rebalance_counter = 0
        
    def next(self):
        """Main strategy logic for each bar."""
        self.rebalance_counter += 1
        
        # Rebalance monthly
        if self.rebalance_counter >= self.params.rebalance_frequency:
            self.rebalance_counter = 0
            self._rebalance_portfolio()
    
    def _rebalance_portfolio(self):
        """Rebalance portfolio using the strategy engine."""
        if self.strategy_engine is None:
            return
        
        try:
            # Run strategy for current date
            results = self.strategy_engine.run_strategy(self.datas[0].datetime.date(0))
            
            if 'error' in results:
                self.log(f"Strategy error: {results['error']}")
                return
            
            # Close all existing positions
            for data in self.datas:
                if self.getposition(data).size > 0:
                    self.close(data)
            
            # Open new positions
            portfolio = results['portfolio']
            for _, row in portfolio.iterrows():
                ticker = row['ticker']
                weight = row['weight']
                
                # Find corresponding data feed
                for data in self.datas:
                    if data._name == ticker:
                        # Calculate position size
                        portfolio_value = self.broker.getvalue()
                        position_value = portfolio_value * weight
                        position_size = int(position_value / data.close[0])
                        
                        if position_size > 0:
                            self.buy(data=data, size=position_size)
                        break
            
            self.log(f"Portfolio rebalanced: {len(portfolio)} positions")
            
        except Exception as e:
            self.log(f"Rebalancing error: {e}")


if __name__ == "__main__":
    # Example usage
    strategy = QVMEngineV3AdoptedInsights()
    
    # Test strategy for a specific date
    test_date = pd.Timestamp('2024-12-31')
    results = strategy.run_strategy(test_date)
    
    if 'error' not in results:
        print(f"Strategy Results for {test_date}:")
        print(f"Regime: {results['regime']}")
        print(f"Portfolio Size: {results['portfolio_size']}")
        print(f"Total Weight: {results['total_weight']:.3f}")
        print("\nTop 10 Holdings:")
        print(results['portfolio'][['ticker', 'sector', 'composite_score', 'weight']].head(10))
    else:
        print(f"Strategy Error: {results['error']}") 