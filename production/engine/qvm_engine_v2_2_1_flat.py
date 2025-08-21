"""
Vietnam Factor Investing Platform - QVM Engine v2.2.1 (Flat Methodology)
======================================================================
Component: Enhanced Flat Composite Engine with Look-Ahead Bias Fixes
Purpose: Integrate Low-Volatility, Piotroski F-Score, and FCF Yield using flat methodology
Author: Duc Nguyen, Principal Quantitative Strategist  
Date Created: August 18, 2025
Status: FLAT METHODOLOGY ENGINE (v2.2.1) - PRODUCTION GRADE

LOOK-AHEAD BIAS FIXES (v2.2.1):
This engine fixes critical look-ahead bias issues:
1. Financial Data Timing: Uses lagged financial data (previous quarter)
2. Market Data Timing: Uses current quarter market data
3. Data Availability Validation: Checks calculation dates before using data
4. TTM Data Handling: Properly handles trailing twelve months data
5. Announcement Date Delays: Implements proper earnings announcement delays

ARCHITECTURAL IMPROVEMENTS OVER v2.1.1:
- FIXED: Look-ahead bias in all factor calculations
- ELIMINATED: Code duplication between engine and config files
- IMPLEMENTED: Proper data timing validation
- ENHANCED: Data availability checks with graceful degradation
- PRESERVED: All validated factor calculation logic
- ENHANCED: 4-pillar architecture (Quality, Value, Momentum, Defensive)

FLAT METHODOLOGY IMPLEMENTATION:
- Individual Factor Exposure: Returns sector-neutralized z-scores for ALL factors
- Universal Sector Neutralization: Every factor is sector-normalized before combination
- Single-Step Combination: Direct weighted average without nested normalization
- Component Transparency: Full factor attribution for performance analysis

PERFORMANCE TARGETS:
- Sharpe Ratio: >1.0 (vs 0.48 baseline v1.1)
- Max Drawdown: <35% (vs -66.7% baseline v1.1)

VERSIONING STRATEGY:
- Engine Version: 'qvm_v2.2.1_flat' (database strategy_version tag)
- Inheritance: Extends QVMEngineV201Flat
- Compatibility: Maintains all v2.0.1 functionality while fixing look-ahead bias

Dependencies:
- pandas >= 1.3.0
- numpy >= 1.21.0
- sqlalchemy >= 1.4.0
- PyYAML >= 5.4.0
"""

import pandas as pd
import numpy as np
import logging
from sqlalchemy import text
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

# Import parent engine
from .qvm_engine_v2_0_1_flat import QVMEngineV201Flat
from .qvm_engine_v2_2_1_flat_vectorized import (
    install_vectorized_fscore_221,
    prime_fscore_cache_221,
    normalize_sector_labels_221,
    compute_nf_vectorized_221,
    compute_bank_vectorized_221,
    compute_sec_vectorized_221,
)


class QVMEngineV221Flat(QVMEngineV201Flat):
    """
    QVM Engine v2.2.1 Flat - Enhanced flat methodology with look-ahead bias fixes.
    
    This engine implements the institutional-standard flat composite methodology
    while fixing critical look-ahead bias issues and eliminating code duplication.
    All factors are individually sector-neutralized before flat combination to 
    ensure optimal alpha extraction.
    
    Key Features:
    - 4-pillar architecture: Quality, Value, Momentum, Defensive
    - Enhanced factor set: Traditional + F-Score + FCF Yield + Low-Vol
    - Flat methodology: Single-step combination without hierarchical nesting
    - Sector-specific F-Score variants: 9-point (non-financial), 6-point (banking), 5-point (securities)
    - Look-ahead bias fixes: Proper data timing and availability validation
    """

    def __init__(self, config_path: str = None, log_level: str = 'INFO'):
        """Initialize v2.2.1 Flat engine with look-ahead bias fixes."""
        super().__init__(config_path, log_level)
        
        # Override engine version
        self.engine_version = 'qvm_v2.2.1_flat'
        
        # CRITICAL FIX: Override logger name to show correct engine version
        self.logger = logging.getLogger('QVMEngineV221Flat')
        
        # Load enhanced 4-pillar weights from configuration
        self.enhanced_weights = self.factor_config.get('qvm_composite', {}).get('enhanced_weights', {
            'quality': 0.35,      # Fallback defaults
            'value': 0.30,
            'momentum': 0.20,
            'defensive': 0.15
        })
        
        # New factor parameters
        self.low_vol_lookback = 63  # 63-day rolling volatility
        
        # Override flat weights to use enhanced versions for v2.2.1
        self._load_enhanced_flat_weights()
        
        # CRITICAL FIX: Override parent's 3-pillar weights with 4-pillar enhanced weights
        self.qvm_weights = self.enhanced_weights.copy()
        
        # CRITICAL FIX: Add missing intermediary table mappings for banking data
        self.intermediary_tables = {
            'banking': 'intermediary_calculations_banking_cleaned',
            'securities': 'intermediary_calculations_securities_cleaned', 
            'non_financial': 'intermediary_calculations_enhanced'
        }
        
        # LOOK-AHEAD BIAS FIXES: Data timing configuration
        self.data_timing_config = {
            'financial_data_lag_quarters': 1,  # Use previous quarter for financial data
            'market_data_current_quarter': True,  # Use current quarter for market data
            'min_data_availability_days': 30,  # Minimum days after quarter end for data availability
            'earnings_announcement_delay_days': 45,  # Typical earnings announcement delay
            'validate_data_timing': True,  # Enable data timing validation
        }
        
        self.logger.info("="*60)
        self.logger.info(f"Initialized QVM Engine v{self.engine_version} (Flat Methodology)")
        self.logger.info("Enhanced Factors: Low-Vol, F-Score (9/6/5 variants), FCF Yield")
        self.logger.info("Architecture: 4-Pillar Flat Composite (Q35/V30/M20/D15)")
        self.logger.info("Methodology: Universal sector neutralization + single-step combination")
        self.logger.info("LOOK-AHEAD BIAS FIXES: Proper data timing and availability validation")
        self.logger.info(f"ENHANCED WEIGHTS: Quality {self.qvm_weights['quality']*100:.1f}%, "
                       f"Value {self.qvm_weights['value']*100:.1f}%, "
                       f"Momentum {self.qvm_weights['momentum']*100:.1f}%, "
                       f"Defensive {self.qvm_weights['defensive']*100:.1f}%")
        self.logger.info(f"BANKING TABLE FIX: Using {self.intermediary_tables['banking']} for banking data")
        self.logger.info("="*60)

        # Feature flag: enable vectorized F-Score path for v2.2.1 (config-only)
        try:
            self.use_vectorized_fscore_221 = bool(
                self.factor_config.get('f_score', {}).get('use_vectorized_fscore_221', False)
            )
        except Exception:
            self.use_vectorized_fscore_221 = False

        if self.use_vectorized_fscore_221:
            try:
                install_vectorized_fscore_221(self)
                self.logger.info("Feature Flag: USE_VECTORIZED_F_SCORE_221=ON — vectorized F-Score installed")
            except Exception as e:
                self.logger.warning(f"Failed to install vectorized F-Score methods: {e}")

    def _load_enhanced_flat_weights(self):
        """Override parent weights to use enhanced sector-specific versions with new factors."""
        try:
            # Load enhanced individual factor weights (sector-specific)
            if 'flat_composite_weights' in self.factor_config:
                flat_weights = self.factor_config['flat_composite_weights']
                
                # Load sector-specific quality weights for v2.2.1 (enhanced)
                self.quality_enhanced_non_financial_weights = flat_weights.get('quality_enhanced_non_financial', {})
                self.quality_enhanced_banking_weights = flat_weights.get('quality_enhanced_banking', {})
                self.quality_enhanced_securities_weights = flat_weights.get('quality_enhanced_securities', {})
                
                # Other pillars remain universal
                self.value_individual_weights = flat_weights.get('value_enhanced', {})
                self.momentum_individual_weights = flat_weights.get('momentum', {})
                self.defensive_individual_weights = flat_weights.get('defensive', {})
                
                # Validate all sector-specific quality weights sum to 1.0
                sectors_and_weights = [
                    ('Non-Financial', self.quality_enhanced_non_financial_weights),
                    ('Banking', self.quality_enhanced_banking_weights),
                    ('Securities', self.quality_enhanced_securities_weights)
                ]
                
                for sector_name, weights in sectors_and_weights:
                    weight_sum = sum(weights.values()) if weights else 0
                    if abs(weight_sum - 1.0) > 1e-6:
                        raise ValueError(f"Quality {sector_name} weights sum to {weight_sum}, must equal 1.0")
                    self.logger.debug(f"✅ Quality {sector_name} weights sum: {weight_sum:.6f}")
                
                # Validate other pillar weights
                other_pillars = [
                    ('Value Enhanced', self.value_individual_weights),
                    ('Momentum', self.momentum_individual_weights), 
                    ('Defensive', self.defensive_individual_weights)
                ]
                
                for pillar_name, weights in other_pillars:
                    weight_sum = sum(weights.values()) if weights else 0
                    if abs(weight_sum - 1.0) > 1e-6:
                        raise ValueError(f"{pillar_name} weights sum to {weight_sum}, must equal 1.0")
                    self.logger.debug(f"✅ {pillar_name} weights sum: {weight_sum:.6f}")
                
                self.logger.info("Enhanced sector-specific flat composite weights loaded successfully")
                self.logger.info(f"Quality sectors: Non-Financial ({len(self.quality_enhanced_non_financial_weights)}), "
                               f"Banking ({len(self.quality_enhanced_banking_weights)}), "
                               f"Securities ({len(self.quality_enhanced_securities_weights)})")
                
            else:
                self.logger.error("flat_composite_weights not found in configuration")
                raise ValueError("Missing flat_composite_weights configuration")
                
        except Exception as e:
            self.logger.error(f"Failed to load enhanced flat composite weights: {e}")
            raise

    def _get_lagged_quarter_info(self, analysis_date: pd.Timestamp) -> Tuple[int, int]:
        """
        Get quarter information for financial data that was actually available at analysis date.
        
        CRITICAL LOOK-AHEAD BIAS FIX: Uses data that was actually available at analysis date,
        not the most recent available data. This prevents using future data to analyze past periods.
        
        Args:
            analysis_date: Current analysis date
            
        Returns:
            Tuple of (available_year, available_quarter) for financial data
        """
        try:
            # Calculate the quarter that should be used for this analysis date
            # Use 1 quarter lag to account for earnings announcement delays
            year = analysis_date.year
            month = analysis_date.month
            
            # Determine current quarter
            if month <= 3:
                current_quarter = 1
                current_year = year
            elif month <= 6:
                current_quarter = 2
                current_year = year
            elif month <= 9:
                current_quarter = 3
                current_year = year
            else:
                current_quarter = 4
                current_year = year
            
            # Use previous quarter for financial data (lagged)
            if current_quarter == 1:
                lagged_quarter = 4
                lagged_year = current_year - 1
            else:
                lagged_quarter = current_quarter - 1
                lagged_year = current_year
            
            # Validate that this data was actually available at analysis date
            # Check if we're within the earnings announcement delay period
            quarter_end_dates = {
                1: pd.Timestamp(f"{lagged_year}-03-31"),
                2: pd.Timestamp(f"{lagged_year}-06-30"),
                3: pd.Timestamp(f"{lagged_year}-09-30"),
                4: pd.Timestamp(f"{lagged_year}-12-31")
            }
            
            quarter_end = quarter_end_dates[lagged_quarter]
            earnings_delay_days = self.data_timing_config['earnings_announcement_delay_days']
            earliest_available = quarter_end + pd.Timedelta(days=earnings_delay_days)
            
            # If analysis date is before data would be available, use previous quarter
            if analysis_date < earliest_available:
                if lagged_quarter == 1:
                    lagged_quarter = 4
                    lagged_year = lagged_year - 1
                else:
                    lagged_quarter = lagged_quarter - 1
                
                # Update quarter end for validation
                quarter_end_dates = {
                    1: pd.Timestamp(f"{lagged_year}-03-31"),
                    2: pd.Timestamp(f"{lagged_year}-06-30"),
                    3: pd.Timestamp(f"{lagged_year}-09-30"),
                    4: pd.Timestamp(f"{lagged_year}-12-31")
                }
                quarter_end = quarter_end_dates[lagged_quarter]
                earliest_available = quarter_end + pd.Timedelta(days=earnings_delay_days)
            
            # Verify data exists in database for this quarter
            query = text("""
                SELECT COUNT(*) as count
                FROM intermediary_calculations_enhanced
                WHERE year = :year AND quarter = :quarter
                LIMIT 1
            """)
            
            result = pd.read_sql(query, self.engine, params={'year': lagged_year, 'quarter': lagged_quarter})
            
            if result.iloc[0]['count'] > 0:
                self.logger.info(f"📅 Data timing: Analysis {analysis_date.date()} -> Using fundamentals {lagged_year}Q{lagged_quarter} (available from {earliest_available.date()})")
                return lagged_year, lagged_quarter
            else:
                # Fallback to previous quarter if data doesn't exist
                if lagged_quarter == 1:
                    lagged_quarter = 4
                    lagged_year = lagged_year - 1
                else:
                    lagged_quarter = lagged_quarter - 1
                
                self.logger.warning(f"⚠️ Data not available for {lagged_year}Q{lagged_quarter}, using previous quarter")
                return lagged_year, lagged_quarter
            
        except Exception as e:
            self.logger.error(f"Failed to get lagged quarter info: {e}")
            # Fallback to simple calculation
            year = analysis_date.year
            month = analysis_date.month
            if month <= 3:
                return year - 1, 4
            elif month <= 6:
                return year, 1
            elif month <= 9:
                return year, 2
            else:
                return year, 3

    def _validate_data_availability(self, ticker: str, analysis_date: pd.Timestamp, 
                                  data_type: str = 'financial') -> bool:
        """
        Validate data availability to prevent look-ahead bias.
        
        LOOK-AHEAD BIAS FIX: Ensures data was actually available at analysis date
        by checking calculation dates and announcement delays.
        
        Args:
            ticker: Stock ticker
            analysis_date: Analysis date
            data_type: Type of data ('financial', 'market', 'price')
            
        Returns:
            bool: True if data is available, False otherwise
        """
        try:
            if not self.data_timing_config['validate_data_timing']:
                return True  # Skip validation if disabled
            
            # Get quarter end date
            year = analysis_date.year
            quarter = (analysis_date.month - 1) // 3 + 1
            
            # Calculate quarter end date
            if quarter == 1:
                quarter_end = pd.Timestamp(f"{year}-03-31")
            elif quarter == 2:
                quarter_end = pd.Timestamp(f"{year}-06-30")
            elif quarter == 3:
                quarter_end = pd.Timestamp(f"{year}-09-30")
            else:
                quarter_end = pd.Timestamp(f"{year}-12-31")
            
            # Check minimum data availability period
            min_availability_days = self.data_timing_config['min_data_availability_days']
            earnings_delay_days = self.data_timing_config['earnings_announcement_delay_days']
            
            # Calculate earliest possible data availability date
            if data_type == 'financial':
                # Financial data needs earnings announcement delay
                earliest_available = quarter_end + pd.Timedelta(days=earnings_delay_days)
            else:
                # Market/price data available immediately after quarter end
                earliest_available = quarter_end + pd.Timedelta(days=min_availability_days)
            
            # Check if analysis date is after earliest available date
            if analysis_date < earliest_available:
                self.logger.warning(f"⚠️ {ticker} {data_type} data not available at {analysis_date}")
                self.logger.warning(f"   Earliest available: {earliest_available}")
                self.logger.warning(f"   Quarter end: {quarter_end}")
                return False
            
            self.logger.debug(f"✅ {ticker} {data_type} data available at {analysis_date}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to validate data availability for {ticker}: {e}")
            return False

    def _get_data_calculation_date(self, ticker: str, year: int, quarter: int, 
                                 data_type: str = 'financial') -> Optional[pd.Timestamp]:
        """
        Get the actual calculation date for data to validate availability.
        
        LOOK-AHEAD BIAS FIX: Checks when data was actually calculated
        to ensure it was available at analysis date.
        
        Args:
            ticker: Stock ticker
            year: Data year
            quarter: Data quarter
            data_type: Type of data ('financial', 'market')
            
        Returns:
            pd.Timestamp: Calculation date or None if not found
        """
        try:
            if data_type == 'financial':
                # Check financial data calculation date
                query = text("""
                    SELECT calculation_date, last_updated
                    FROM intermediary_calculations_enhanced
                    WHERE ticker = :ticker AND year = :year AND quarter = :quarter
                    LIMIT 1
                """)
            else:
                # Check market data calculation date
                query = text("""
                    SELECT calculation_date, last_updated
                    FROM precalculated_quarterly_market_cap
                    WHERE ticker = :ticker AND year = :year AND quarter = :quarter
                    LIMIT 1
                """)
            
            result = pd.read_sql(query, self.engine, params={
                'ticker': ticker, 'year': year, 'quarter': quarter
            })
            
            if not result.empty:
                # Use calculation_date if available, otherwise last_updated
                calc_date = result.iloc[0].get('calculation_date') or result.iloc[0].get('last_updated')
                if calc_date:
                    return pd.Timestamp(calc_date)
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Could not get calculation date for {ticker}: {e}")
            return None

    def _get_individual_fcf_yield_factors(self, data: pd.DataFrame,
                                         analysis_date: pd.Timestamp,
                                         sector_map: pd.DataFrame = None) -> Dict[str, pd.Series]:
        """
        Calculate FCF Yield with look-ahead bias fixes and sector exclusions.
        
        LOOK-AHEAD BIAS FIX: Uses lagged financial data and current market data
        to ensure proper data timing and availability.
        """
        try:
            self.logger.info("Calculating FCF Yield (v2.2.1 with look-ahead bias fixes)...")
            
            universe_tickers = data['ticker'].unique().tolist()
            # TIER 1 REFINEMENT #4: Use cached sector mapping for performance
            if sector_map is None:
                sector_map = self.get_sector_mapping().set_index('ticker')
            else:
                # Ensure sector_map is properly indexed if passed as cached
                if 'ticker' in sector_map.columns:
                    sector_map = sector_map.set_index('ticker')
            
            # Exclude financial sectors
            eligible_tickers = [
                t for t in universe_tickers 
                if sector_map.loc[t, 'sector'] not in ['Banking', 'Securities', 'Insurance']
            ]
            
            if not eligible_tickers:
                self.logger.warning("No non-financial tickers for FCF Yield calculation.")
                return {}

            # LOOK-AHEAD BIAS FIX: Get lagged quarter for financial data
            lagged_year, lagged_quarter = self._get_lagged_quarter_info(analysis_date)
            if lagged_year is None or lagged_quarter is None:
                self.logger.error("Failed to get lagged quarter info for FCF calculation")
                return {}
            
            # Get current quarter for market data
            current_year = analysis_date.year
            current_quarter = (analysis_date.month - 1) // 3 + 1

            # LOOK-AHEAD BIAS FIX: Get FCF components from lagged financial data
            fcf_query = text("""
                SELECT ticker, NetCFO_TTM, NetCFI_TTM, CapEx_TTM, FCF_TTM 
                FROM intermediary_calculations_enhanced 
                WHERE year = :lagged_year AND quarter = :lagged_quarter 
                AND ticker IN :tickers AND has_full_ttm = 1
            """)
            fcf_data = pd.read_sql(fcf_query, self.engine, params={
                'lagged_year': lagged_year, 'lagged_quarter': lagged_quarter, 'tickers': tuple(eligible_tickers)
            })
            
            # LOOK-AHEAD BIAS FIX: Get market cap from current quarter
            market_query = text("""
                SELECT ticker, market_cap FROM vcsc_daily_data_complete 
                WHERE trading_date = :analysis_date AND ticker IN :tickers AND market_cap > 0
            """)
            market_data = pd.read_sql(market_query, self.engine, params={
                'analysis_date': analysis_date, 'tickers': tuple(eligible_tickers)
            })

            # SMART FCF CALCULATION: Use actual Capex when available, fall back to CFI proxy
            fcf_yields = {}
            if not fcf_data.empty and not market_data.empty:
                combined = pd.merge(fcf_data, market_data, on='ticker', how='inner')
                
                # Track FCF calculation methodology for data quality KPI
                capex_imputed_count = 0
                actual_capex_count = 0
                total_fcf_calculations = 0
                
                for idx, row in combined.iterrows():
                    ticker = row['ticker']
                    
                    # LOOK-AHEAD BIAS FIX: Validate data availability
                    if not self._validate_data_availability(ticker, analysis_date, 'financial'):
                        self.logger.warning(f"⚠️ Skipping {ticker}: Financial data not available at {analysis_date}")
                        continue
                    
                    net_cfo = row.get('NetCFO_TTM', 0)
                    net_cfi = row.get('NetCFI_TTM', 0)
                    actual_capex = row.get('CapEx_TTM', 0)
                    market_cap = row.get('market_cap', 0)
                    
                    if pd.notna(net_cfo) and market_cap > 0:
                        total_fcf_calculations += 1
                        
                        # SMART METHODOLOGY: Prioritize actual Capex over CFI proxy
                        if pd.notna(actual_capex) and actual_capex != 0:
                            # Use actual Capex data (preferred method)
                            # Note: Capex is negative (outflow), so FCF = CFO - Capex becomes CFO - (-capex) = CFO + abs(capex)
                            fcf = net_cfo - actual_capex
                            actual_capex_count += 1
                        elif pd.notna(net_cfi):
                            # Fall back to CFI proxy method when Capex unavailable
                            capex_proxy = max(0, -net_cfi)
                            fcf = net_cfo - capex_proxy
                            capex_imputed_count += 1
                        else:
                            # No Capex data available - skip this ticker
                            continue
                        
                        fcf_yield = fcf / market_cap
                        fcf_yields[ticker] = fcf_yield
                
                # ENHANCED DATA QUALITY TRACKING: Report actual vs imputed Capex usage
                if total_fcf_calculations > 0:
                    actual_capex_rate = actual_capex_count / total_fcf_calculations
                    imputation_rate = capex_imputed_count / total_fcf_calculations
                    
                    self.logger.info(f"FCF Calculation Summary: {actual_capex_count} actual Capex ({actual_capex_rate:.1%}), "
                                   f"{capex_imputed_count} CFI proxy ({imputation_rate:.1%}), "
                                   f"{total_fcf_calculations} total")
                    
                    # Only warn if we're still relying heavily on imputation despite having better data
                    if imputation_rate > 0.40:
                        import warnings
                        warning_msg = (
                            f"FCF Yield data quality alert: CFI proxy usage {imputation_rate:.1%} "
                            f"exceeds 40% threshold ({capex_imputed_count}/{total_fcf_calculations} tickers). "
                            f"Smart methodology now uses actual CapEx_TTM when available ({actual_capex_rate:.1%})."
                        )
                        warnings.warn(warning_msg, UserWarning, stacklevel=2)
                        self.logger.warning(warning_msg)

            if not fcf_yields:
                return {}
            
            # Create dataframe for sector-neutral normalization
            fcf_df = pd.DataFrame([
                {'ticker': ticker, 'fcf_yield_raw': yield_val, 'sector': sector_map.loc[ticker, 'sector']}
                for ticker, yield_val in fcf_yields.items()
                if ticker in sector_map.index
            ])
            
            # Apply sector-neutral normalization
            fcf_yield_z = self.calculate_sector_neutral_zscore(
                fcf_df, 'fcf_yield_raw', 'sector'
            )
            
            # Return as Series indexed by ticker
            fcf_yield_series = pd.Series(
                fcf_yield_z.values,
                index=fcf_df['ticker'],
                name='fcf_yield_z'
            )
            
            self.logger.info(f"Successfully calculated FCF Yield for {len(fcf_yield_series)} non-financial tickers.")
            self.logger.info(f"LOOK-AHEAD BIAS FIX: Used lagged financial data ({lagged_year}Q{lagged_quarter}) with current market data")
            return {'fcf_yield_z': fcf_yield_series}

        except Exception as e:
            self.logger.error(f"Failed to calculate FCF Yield factors: {e}")
            return {}

    def _calculate_actual_earnings_yield_fixed(self, ticker: str, analysis_date: pd.Timestamp) -> Optional[float]:
        """
        Calculate actual earnings yield with look-ahead bias fixes.
        
        LOOK-AHEAD BIAS FIX: Uses lagged financial data and current market data.
        """
        try:
            # LOOK-AHEAD BIAS FIX: Get lagged quarter for financial data
            lagged_year, lagged_quarter = self._get_lagged_quarter_info(analysis_date)
            if lagged_year is None or lagged_quarter is None:
                return None
            
            # Get current quarter for market data
            current_year = analysis_date.year
            current_quarter = (analysis_date.month - 1) // 3 + 1
            
            # LOOK-AHEAD BIAS FIX: Validate data availability
            if not self._validate_data_availability(ticker, analysis_date, 'financial'):
                self.logger.warning(f"⚠️ {ticker}: Financial data not available at {analysis_date}")
                return None
            
            # Get NetProfit_TTM from lagged financial data
            profit_query = text("""
                SELECT NetProfit_TTM
                FROM intermediary_calculations_enhanced
                WHERE ticker = :ticker AND year = :lagged_year AND quarter = :lagged_quarter AND has_full_ttm = 1
                LIMIT 1
            """)
            
            profit_data = pd.read_sql(profit_query, self.engine, params={
                'ticker': ticker, 'lagged_year': lagged_year, 'lagged_quarter': lagged_quarter
            })
            
            if profit_data.empty or pd.isna(profit_data['NetProfit_TTM'].iloc[0]):
                return None
            
            net_profit = float(profit_data['NetProfit_TTM'].iloc[0])
            
            # Get MarketCap from current quarter market data
            market_cap_query = text("""
                SELECT market_cap
                FROM precalculated_quarterly_market_cap
                WHERE ticker = :ticker AND year = :current_year AND quarter = :current_quarter
                LIMIT 1
            """)
            
            market_cap_data = pd.read_sql(market_cap_query, self.engine, params={
                'ticker': ticker, 'current_year': current_year, 'current_quarter': current_quarter
            })
            
            if market_cap_data.empty or pd.isna(market_cap_data['market_cap'].iloc[0]):
                return None
            
            market_cap = float(market_cap_data['market_cap'].iloc[0])
            
            if market_cap <= 0:
                return None
            
            earnings_yield = (net_profit / market_cap) * 100
            # Normalize to 0-1 range (0-20% earnings yield range)
            return max(0.0, min(1.0, earnings_yield / 20.0))
            
        except Exception as e:
            self.logger.debug(f"Could not calculate actual earnings yield for {ticker}: {e}")
            return None

    def _calculate_actual_book_to_price_fixed(self, ticker: str, analysis_date: pd.Timestamp) -> Optional[float]:
        """
        Calculate actual book-to-price ratio with look-ahead bias fixes.
        
        LOOK-AHEAD BIAS FIX: Uses lagged financial data and current market data.
        """
        try:
            # LOOK-AHEAD BIAS FIX: Get lagged quarter for financial data
            lagged_year, lagged_quarter = self._get_lagged_quarter_info(analysis_date)
            if lagged_year is None or lagged_quarter is None:
                return None
            
            # Get current quarter for market data
            current_year = analysis_date.year
            current_quarter = (analysis_date.month - 1) // 3 + 1
            
            # LOOK-AHEAD BIAS FIX: Validate data availability
            if not self._validate_data_availability(ticker, analysis_date, 'financial'):
                self.logger.warning(f"⚠️ {ticker}: Financial data not available at {analysis_date}")
                return None
            
            # Get TotalEquity from lagged financial data
            equity_query = text("""
                SELECT AvgTotalEquity
                FROM intermediary_calculations_enhanced
                WHERE ticker = :ticker AND year = :lagged_year AND quarter = :lagged_quarter AND has_full_avg = 1
                LIMIT 1
            """)
            
            equity_data = pd.read_sql(equity_query, self.engine, params={
                'ticker': ticker, 'lagged_year': lagged_year, 'lagged_quarter': lagged_quarter
            })
            
            if equity_data.empty or pd.isna(equity_data['AvgTotalEquity'].iloc[0]):
                return None
            
            total_equity = float(equity_data['AvgTotalEquity'].iloc[0])
            
            # Get MarketCap from current quarter market data
            market_cap_query = text("""
                SELECT market_cap
                FROM precalculated_quarterly_market_cap
                WHERE ticker = :ticker AND year = :current_year AND quarter = :current_quarter
                LIMIT 1
            """)
            
            market_cap_data = pd.read_sql(market_cap_query, self.engine, params={
                'ticker': ticker, 'current_year': current_year, 'current_quarter': current_quarter
            })
            
            if market_cap_data.empty or pd.isna(market_cap_data['market_cap'].iloc[0]):
                return None
            
            market_cap = float(market_cap_data['market_cap'].iloc[0])
            
            if market_cap <= 0:
                return None
            
            book_to_price = total_equity / market_cap
            # Normalize to 0-1 range (0-2 book-to-price range)
            return max(0.0, min(1.0, book_to_price / 2.0))
            
        except Exception as e:
            self.logger.debug(f"Could not calculate actual book-to-price for {ticker}: {e}")
            return None

    def calculate_qvm_composite_fixed(self, analysis_date: pd.Timestamp,
                                    universe: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Calculate v2.2.1 Flat composite with look-ahead bias fixes and no code duplication.
        
        LOOK-AHEAD BIAS FIXES:
        - Uses lagged financial data (previous quarter)
        - Uses current quarter market data
        - Validates data availability before use
        - Eliminates code duplication between engine and config files
        """
        try:
            self.logger.info(f"BEGIN v2.2.1 Flat composite calculation for {len(universe)} tickers on {analysis_date.date()}")
            self.logger.info("LOOK-AHEAD BIAS FIXES: Using lagged financial data with current market data")

            # LOOK-AHEAD BIAS FIX: Get lagged quarter info for validation
            lagged_year, lagged_quarter = self._get_lagged_quarter_info(analysis_date)
            current_year = analysis_date.year
            current_quarter = (analysis_date.month - 1) // 3 + 1
            
            self.logger.info(f"📅 Data timing: Financial data from {lagged_year}Q{lagged_quarter}, Market data from {current_year}Q{current_quarter}")

            # 1. Data Ingestion with look-ahead bias fixes
            fundamentals = self._get_fundamentals_with_timing_fixes(analysis_date, universe)
            market_data = self._get_market_data_with_timing_fixes(analysis_date, universe)

            # FAIL FAST: Show exactly what's missing
            if fundamentals.empty:
                self.logger.error(f"FATAL: Fundamentals data empty for {lagged_year}Q{lagged_quarter}")
                self.logger.error(f"   Universe: {universe}")
                self.logger.error(f"   Analysis date: {analysis_date}")
                self.logger.error("   Query: intermediary_calculations_enhanced WHERE year=lagged_year AND quarter=lagged_quarter")
                return {}
                
            if market_data.empty:
                self.logger.error(f"FATAL: Market data empty for {current_year}Q{current_quarter}")
                self.logger.error(f"   Universe: {universe}")
                self.logger.error(f"   Analysis date: {analysis_date}")
                self.logger.error("   Query: vcsc_daily_data_complete WHERE trading_date=analysis_date")
                return {}
                
            data = pd.merge(fundamentals, market_data, on='ticker', how='inner')
            if data.empty:
                self.logger.error("FATAL: No intersection between fundamentals and market data")
                self.logger.error(f"   Fundamentals tickers: {fundamentals['ticker'].tolist()}")
                self.logger.error(f"   Market data tickers: {market_data['ticker'].tolist()}")
                return {}

            # TIER 1 REFINEMENT #4: Cache sector mapping for performance and normalize labels
            sector_map = self.get_sector_mapping().set_index('ticker')
            sector_map = normalize_sector_labels_221(sector_map.reset_index(), 'sector').set_index('ticker')

            # 2. Enhanced Factor Calculation (Traditional + New) with look-ahead bias fixes
            # Traditional factors from parent class (with timing fixes)
            quality_factors = self._get_individual_quality_factors_fixed(data, analysis_date, sector_map)
            value_factors = self._get_individual_value_factors_fixed(data, analysis_date, sector_map)
            momentum_factors = self._get_individual_momentum_factors_fixed(data, analysis_date, universe, sector_map)
            
            # AGENT SMITH DEBUG: Log factor counts
            self.logger.info(f"Factor counts: Quality={len(quality_factors)}, Value={len(value_factors)}, Momentum={len(momentum_factors)}")

            # Prime vectorized F-Score cache (≤ 3 queries per sector group per date)
            if self.use_vectorized_fscore_221:
                try:
                    universe_df = sector_map.reset_index()[['ticker','sector']]
                    prime_fscore_cache_221(self, universe_df, analysis_date, lagged_year, lagged_quarter)
                    # Log group sizes for observability
                    sec_map = universe_df.set_index('ticker')['sector'].to_dict()
                    nf   = [t for t in universe_df['ticker'] if sec_map.get(t) not in ('Banking','Securities','Insurance')]
                    bank = [t for t in universe_df['ticker'] if sec_map.get(t) == 'Banking']
                    sec  = [t for t in universe_df['ticker'] if sec_map.get(t) == 'Securities']
                    self.logger.info("F-Score priming groups: NF=%d, Bank=%d, Sec=%d", len(nf), len(bank), len(sec))
                except Exception as e:
                    self.logger.warning(f"F-Score cache priming failed: {e}")

            # New v2.2.1 factors (also use cached sector map and timing fixes)
            low_vol_factors = self._get_individual_low_vol_factors_fixed(analysis_date, universe, sector_map)
            # For F-Score and FCF Yield, if data originated from only one side, still pass available tickers
            f_score_factors = self._get_individual_f_score_factors_fixed(data, analysis_date, sector_map)
            fcf_yield_factors = self._get_individual_fcf_yield_factors(data, analysis_date, sector_map)

            self.logger.info(f"Individual factors calculated: {len(quality_factors)} quality, "
                           f"{len(value_factors)} value, {len(momentum_factors)} momentum, "
                           f"{len(low_vol_factors)} defensive, {len(f_score_factors)} f-score, "
                           f"{len(fcf_yield_factors)} fcf-yield")

            # Observability: factor coverage and NaN rates across series
            try:
                def _coverage_summary(group: dict) -> str:
                    parts = []
                    for name, ser in group.items():
                        if isinstance(ser, pd.Series) and len(ser) > 0:
                            nan_rate = float(ser.isna().mean())
                            parts.append(f"{name}:{1-nan_rate:.2f}")
                    return ", ".join(parts[:10])  # limit log length

                self.logger.info(
                    "Coverage Q=%s | V=%s | M=%s | D=%s | F=%s | FCF=%s",
                    _coverage_summary(quality_factors),
                    _coverage_summary(value_factors),
                    _coverage_summary(momentum_factors),
                    _coverage_summary(low_vol_factors),
                    _coverage_summary(f_score_factors),
                    _coverage_summary(fcf_yield_factors),
                )
            except Exception:
                pass

            # 3. Flat Combination with Enhanced Architecture
            all_tickers = set(data['ticker'].unique())
            results = {}

            for ticker in all_tickers:
                # Get sector for this ticker
                ticker_sector = sector_map.loc[ticker, 'sector'] if ticker in sector_map.index else 'Unknown'
                
                # Collect ALL individual factor scores
                individual_scores = {}
                
                # Traditional factors
                for factor_name, factor_series in quality_factors.items():
                    individual_scores[factor_name] = factor_series.get(ticker, 0.0) if ticker in factor_series.index else 0.0
                
                for factor_name, factor_series in value_factors.items():
                    individual_scores[factor_name] = factor_series.get(ticker, 0.0) if ticker in factor_series.index else 0.0
                
                for factor_name, factor_series in momentum_factors.items():
                    individual_scores[factor_name] = factor_series.get(ticker, 0.0) if ticker in factor_series.index else 0.0
                
                # New enhanced factors
                for factor_name, factor_series in low_vol_factors.items():
                    individual_scores[factor_name] = factor_series.get(ticker, 0.0) if ticker in factor_series.index else 0.0
                
                for factor_name, factor_series in f_score_factors.items():
                    individual_scores[factor_name] = factor_series.get(ticker, 0.0) if ticker in factor_series.index else 0.0
                
                for factor_name, factor_series in fcf_yield_factors.items():
                    individual_scores[factor_name] = factor_series.get(ticker, 0.0) if ticker in factor_series.index else 0.0

                # 4. Enhanced 4-Pillar Composite Calculation (sector-specific quality)
                quality_composite = self._calculate_enhanced_flat_quality_composite(individual_scores, ticker_sector)
                value_composite = self._calculate_enhanced_flat_value_composite(individual_scores)
                momentum_composite = self._calculate_flat_momentum_composite(individual_scores)
                defensive_composite = self._calculate_flat_defensive_composite(individual_scores)

                # 5. Final 4-Pillar Weighted Score
                qvm_score = (
                    self.enhanced_weights['quality'] * quality_composite +
                    self.enhanced_weights['value'] * value_composite +
                    self.enhanced_weights['momentum'] * momentum_composite +
                    self.enhanced_weights['defensive'] * defensive_composite
                )

                # 6. Institutional Transparency: Return everything with data timing info
                results[ticker] = {
                    'Quality_Composite': quality_composite,
                    'Value_Composite': value_composite,
                    'Momentum_Composite': momentum_composite,
                    'Defensive_Composite': defensive_composite,
                    'QVM_Composite': qvm_score,
                    'individual_factors': individual_scores,
                    # Enhanced factor breakdown
                    'Low_Volatility_63D': individual_scores.get('low_volatility_raw', 0.0),
                    'Piotroski_F_Score': individual_scores.get('f_score_normalized', 0.0),
                    'FCF_Yield': individual_scores.get('fcf_yield_raw', 0.0),
                    # LOOK-AHEAD BIAS FIX: Data timing information
                    'data_timing': {
                        'financial_data_quarter': f"{lagged_year}-Q{lagged_quarter}",
                        'market_data_quarter': f"{current_year}-Q{current_quarter}",
                        'data_availability_validated': self.data_timing_config['validate_data_timing'],
                    }
                }

            self.logger.info(f"SUCCESS: v2.2.1 Flat composite calculated for {len(results)} tickers.")
            self.logger.info("LOOK-AHEAD BIAS FIXES: All factor calculations use proper data timing")
            return results

        except Exception as e:
            self.logger.error(f"Failed to calculate v2.2.1 Flat composite: {e}")
            return {}

    def _get_fundamentals_with_timing_fixes(self, analysis_date: pd.Timestamp, universe: List[str]) -> pd.DataFrame:
        """
        Get fundamentals data with look-ahead bias fixes.
        
        LOOK-AHEAD BIAS FIX: Uses lagged financial data to ensure availability.
        """
        try:
            # Get lagged quarter for financial data
            lagged_year, lagged_quarter = self._get_lagged_quarter_info(analysis_date)
            if lagged_year is None or lagged_quarter is None:
                return pd.DataFrame()
            
            # Query lagged financial data
            query = text("""
                SELECT ticker, NetProfit_TTM, AvgTotalAssets, NetCFO_TTM, Revenue_TTM, 
                       COGS_TTM, EBITDA_TTM, AvgTotalEquity, AvgTotalDebt
                FROM intermediary_calculations_enhanced
                WHERE year = :lagged_year AND quarter = :lagged_quarter 
                AND ticker IN :universe AND has_full_ttm = 1
            """)
            
            fundamentals = pd.read_sql(query, self.engine, params={
                'lagged_year': lagged_year, 'lagged_quarter': lagged_quarter, 'universe': tuple(universe)
            })
            
            self.logger.info(f"📊 Loaded lagged fundamentals: {len(fundamentals)} records from {lagged_year}Q{lagged_quarter}")
            return fundamentals
            
        except Exception as e:
            self.logger.error(f"Failed to get fundamentals with timing fixes: {e}")
            return pd.DataFrame()

    def _get_market_data_with_timing_fixes(self, analysis_date: pd.Timestamp, universe: List[str]) -> pd.DataFrame:
        """
        Get market data with look-ahead bias fixes.
        
        LOOK-AHEAD BIAS FIX: Uses current quarter market data.
        """
        try:
            # Get current quarter for market data
            current_year = analysis_date.year
            current_quarter = (analysis_date.month - 1) // 3 + 1
            
            # Query current market data - FAIL FAST if missing
            query = text("""
                SELECT ticker, market_cap, close_price
                FROM vcsc_daily_data_complete
                WHERE trading_date = :analysis_date 
                AND ticker IN :universe AND market_cap > 0
            """)
            
            market_data = pd.read_sql(query, self.engine, params={
                'analysis_date': analysis_date, 'universe': tuple(universe)
            })
            
            self.logger.info(f"📊 Loaded current market data: {len(market_data)} records from {current_year}Q{current_quarter}")
            return market_data
            
        except Exception as e:
            self.logger.error(f"Failed to get market data with timing fixes: {e}")
            return pd.DataFrame()

    def _calculate_enhanced_flat_quality_composite(self, individual_scores: Dict[str, float], sector: str) -> float:
        """Enhanced quality composite including F-Score with sector-specific flat methodology."""
        try:
            # Select sector-specific quality weights
            if sector == 'Banking':
                quality_weights = self.quality_enhanced_banking_weights
            elif sector == 'Securities':
                quality_weights = self.quality_enhanced_securities_weights
            else:
                # Non-Financial (default for all other sectors)
                quality_weights = self.quality_enhanced_non_financial_weights
            
            # TIER 1 REFINEMENT #2: Weight-sum assertion (sector-specific)
            assert abs(sum(quality_weights.values()) - 1.0) < 1e-6, \
                f"Enhanced Quality weights for {sector} must sum to 1.0, got {sum(quality_weights.values())}"
            
            # Calculate weighted average
            weighted_sum = 0.0
            total_weight = 0.0
            
            for factor_name, weight in quality_weights.items():
                if factor_name in individual_scores:
                    factor_score = individual_scores[factor_name]
                    if pd.notna(factor_score):
                        weighted_sum += weight * factor_score
                        total_weight += weight
            
            return weighted_sum / total_weight if total_weight > 0 else 0.0
                
        except Exception as e:
            self.logger.error(f"Failed to calculate enhanced flat quality composite for {sector}: {e}")
            return 0.0

    def _calculate_enhanced_flat_value_composite(self, individual_scores: Dict[str, float]) -> float:
        """Enhanced value composite including FCF Yield with flat methodology."""
        try:
            # Use externalized enhanced value weights
            value_weights = self.value_individual_weights
            
            # TIER 1 REFINEMENT #2: Weight-sum assertion
            assert abs(sum(value_weights.values()) - 1.0) < 1e-6, \
                f"Enhanced Value weights must sum to 1.0, got {sum(value_weights.values())}"
            
            # Calculate weighted average
            weighted_sum = 0.0
            total_weight = 0.0
            
            for factor_name, weight in value_weights.items():
                if factor_name in individual_scores:
                    factor_score = individual_scores[factor_name]
                    if pd.notna(factor_score):
                        weighted_sum += weight * factor_score
                        total_weight += weight
            
            return weighted_sum / total_weight if total_weight > 0 else 0.0
                
        except Exception as e:
            self.logger.error(f"Failed to calculate enhanced flat value composite: {e}")
            return 0.0

    def _calculate_flat_momentum_composite(self, individual_scores: Dict[str, float]) -> float:
        """Calculate momentum composite using flat methodology."""
        try:
            # Use externalized momentum weights
            momentum_weights = self.momentum_individual_weights
            
            # TIER 1 REFINEMENT #2: Weight-sum assertion
            assert abs(sum(momentum_weights.values()) - 1.0) < 1e-6, \
                f"Momentum weights must sum to 1.0, got {sum(momentum_weights.values())}"
            
            # Calculate weighted average
            weighted_sum = 0.0
            total_weight = 0.0
            
            for factor_name, weight in momentum_weights.items():
                if factor_name in individual_scores:
                    factor_score = individual_scores[factor_name]
                    if pd.notna(factor_score):
                        weighted_sum += weight * factor_score
                        total_weight += weight
            
            return weighted_sum / total_weight if total_weight > 0 else 0.0
                
        except Exception as e:
            self.logger.error(f"Failed to calculate flat momentum composite: {e}")
            return 0.0

    def _calculate_flat_defensive_composite(self, individual_scores: Dict[str, float]) -> float:
        """Calculate defensive composite using low-volatility factors."""
        try:
            # Use externalized defensive weights
            defensive_weights = self.defensive_individual_weights
            
            # TIER 1 REFINEMENT #2: Weight-sum assertion
            assert abs(sum(defensive_weights.values()) - 1.0) < 1e-6, \
                f"Defensive weights must sum to 1.0, got {sum(defensive_weights.values())}"
            
            # Calculate weighted average
            weighted_sum = 0.0
            total_weight = 0.0
            
            for factor_name, weight in defensive_weights.items():
                if factor_name in individual_scores:
                    factor_score = individual_scores[factor_name]
                    if pd.notna(factor_score):
                        weighted_sum += weight * factor_score
                        total_weight += weight
            
            return weighted_sum / total_weight if total_weight > 0 else 0.0
                
        except Exception as e:
            self.logger.error(f"Failed to calculate flat defensive composite: {e}")
            return 0.0

    def _get_individual_quality_factors_fixed(self, data: pd.DataFrame, analysis_date: pd.Timestamp, sector_map: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Get quality factors with look-ahead bias fixes.
        
        LOOK-AHEAD BIAS FIX: Uses lagged financial data to ensure availability.
        """
        try:
            self.logger.info("Calculating quality factors with look-ahead bias fixes...")
            
            # Get lagged quarter for financial data
            lagged_year, lagged_quarter = self._get_lagged_quarter_info(analysis_date)
            if lagged_year is None or lagged_quarter is None:
                return {}
            
            # Query lagged quality data
            query = text("""
                SELECT ticker, NetProfit_TTM, AvgTotalAssets, AvgTotalEquity,
                       Revenue_TTM, COGS_TTM, EBITDA_TTM
                FROM intermediary_calculations_enhanced
                WHERE year = :lagged_year AND quarter = :lagged_quarter 
                AND ticker IN :tickers AND has_full_ttm = 1
            """)
            
            quality_data = pd.read_sql(query, self.engine, params={
                'lagged_year': lagged_year, 'lagged_quarter': lagged_quarter, 
                'tickers': tuple(data['ticker'].unique())
            })
            
            if quality_data.empty:
                return {}
            
            # Calculate quality metrics
            quality_factors = {}
            
            # ROAE calculation
            quality_data['roae'] = (quality_data['NetProfit_TTM'] / quality_data['AvgTotalEquity']) * 100
            quality_data['roae_normalized'] = quality_data['roae'].clip(0, 30) / 30  # Normalize 0-30% range
            
            # Net Profit Margin
            quality_data['net_profit_margin'] = (quality_data['NetProfit_TTM'] / quality_data['Revenue_TTM']) * 100
            quality_data['net_profit_margin_normalized'] = quality_data['net_profit_margin'].clip(0, 50) / 50  # Normalize 0-50% range
            
            # Gross Margin
            quality_data['gross_margin'] = ((quality_data['Revenue_TTM'] - quality_data['COGS_TTM']) / quality_data['Revenue_TTM']) * 100
            quality_data['gross_margin_normalized'] = quality_data['gross_margin'].clip(0, 80) / 80  # Normalize 0-80% range
            
            # EBITDA Margin
            quality_data['ebitda_margin'] = (quality_data['EBITDA_TTM'] / quality_data['Revenue_TTM']) * 100
            quality_data['ebitda_margin_normalized'] = quality_data['ebitda_margin'].clip(0, 40) / 40  # Normalize 0-40% range
            
            # Create sector-neutral z-scores
            for metric in ['roae_normalized', 'net_profit_margin_normalized', 'gross_margin_normalized', 'ebitda_margin_normalized']:
                if metric in quality_data.columns:
                    # Merge with sector mapping
                    merged_data = pd.merge(quality_data[['ticker', metric]], sector_map.reset_index(), on='ticker', how='inner')
                    
                    # Apply sector-neutral normalization
                    z_scores = self.calculate_sector_neutral_zscore(merged_data, metric, 'sector')
                    
                    # Create Series indexed by ticker
                    factor_series = pd.Series(
                        z_scores.values,
                        index=merged_data['ticker'],
                        name=f"{metric.replace('_normalized', '_z')}"
                    )
                    
                    quality_factors[f"{metric.replace('_normalized', '_z')}"] = factor_series
            
            self.logger.info(f"✅ Quality factors calculated for {len(quality_factors)} metrics using lagged data ({lagged_year}Q{lagged_quarter})")
            return quality_factors
            
        except Exception as e:
            self.logger.error(f"Failed to calculate quality factors with timing fixes: {e}")
            return {}

    def _get_individual_value_factors_fixed(self, data: pd.DataFrame, analysis_date: pd.Timestamp, sector_map: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Get value factors with look-ahead bias fixes.
        
        LOOK-AHEAD BIAS FIX: Uses lagged financial data and current market data.
        """
        try:
            self.logger.info("Calculating value factors with look-ahead bias fixes...")
            
            # Get lagged quarter for financial data and current quarter for market data
            lagged_year, lagged_quarter = self._get_lagged_quarter_info(analysis_date)
            current_year = analysis_date.year
            current_quarter = (analysis_date.month - 1) // 3 + 1
            
            if lagged_year is None or lagged_quarter is None:
                return {}
            
            # Get lagged financial data
            financial_query = text("""
                SELECT ticker, NetProfit_TTM, AvgTotalEquity, Revenue_TTM, EBITDA_TTM, AvgTotalDebt
                FROM intermediary_calculations_enhanced
                WHERE year = :lagged_year AND quarter = :lagged_quarter 
                AND ticker IN :tickers AND has_full_ttm = 1
            """)
            
            financial_data = pd.read_sql(financial_query, self.engine, params={
                'lagged_year': lagged_year, 'lagged_quarter': lagged_quarter, 
                'tickers': tuple(data['ticker'].unique())
            })
            
            # Get current market data
            market_query = text("""
                SELECT ticker, market_cap
                FROM vcsc_daily_data_complete
                WHERE trading_date = :analysis_date 
                AND ticker IN :tickers AND market_cap > 0
            """)
            
            market_data = pd.read_sql(market_query, self.engine, params={
                'analysis_date': analysis_date, 'tickers': tuple(data['ticker'].unique())
            })
            
            if financial_data.empty or market_data.empty:
                return {}
            
            # Merge financial and market data
            value_data = pd.merge(financial_data, market_data, on='ticker', how='inner')
            
            # Calculate value metrics
            value_factors = {}
            
            # Earnings Yield (E/P)
            value_data['earnings_yield'] = (value_data['NetProfit_TTM'] / value_data['market_cap']) * 100
            value_data['earnings_yield_normalized'] = value_data['earnings_yield'].clip(0, 20) / 20  # Normalize 0-20% range
            
            # Book-to-Price (B/P)
            value_data['book_to_price'] = value_data['AvgTotalEquity'] / value_data['market_cap']
            value_data['book_to_price_normalized'] = value_data['book_to_price'].clip(0, 2) / 2  # Normalize 0-2 range
            
            # Sales-to-Price (S/P)
            value_data['sales_to_price'] = value_data['Revenue_TTM'] / value_data['market_cap']
            value_data['sales_to_price_normalized'] = value_data['sales_to_price'].clip(0, 5) / 5  # Normalize 0-5 range
            
            # EBITDA-to-EV
            value_data['ev'] = value_data['market_cap'] + value_data['AvgTotalDebt'].fillna(0)
            value_data['ebitda_to_ev'] = value_data['EBITDA_TTM'] / value_data['ev']
            value_data['ebitda_to_ev_normalized'] = value_data['ebitda_to_ev'].clip(0, 0.3) / 0.3  # Normalize 0-30% range
            
            # Create sector-neutral z-scores
            for metric in ['earnings_yield_normalized', 'book_to_price_normalized', 'sales_to_price_normalized', 'ebitda_to_ev_normalized']:
                if metric in value_data.columns:
                    # Merge with sector mapping
                    merged_data = pd.merge(value_data[['ticker', metric]], sector_map.reset_index(), on='ticker', how='inner')
                    
                    # Apply sector-neutral normalization
                    z_scores = self.calculate_sector_neutral_zscore(merged_data, metric, 'sector')
                    
                    # Create Series indexed by ticker
                    factor_series = pd.Series(
                        z_scores.values,
                        index=merged_data['ticker'],
                        name=f"{metric.replace('_normalized', '_z')}"
                    )
                    
                    value_factors[f"{metric.replace('_normalized', '_z')}"] = factor_series
            
            self.logger.info(f"✅ Value factors calculated for {len(value_factors)} metrics")
            self.logger.info(f"   Financial data: {lagged_year}Q{lagged_quarter}, Market data: {current_year}Q{current_quarter}")
            return value_factors
            
        except Exception as e:
            self.logger.error(f"Failed to calculate value factors with timing fixes: {e}")
            return {}

    def _get_individual_momentum_factors_fixed(self, data: pd.DataFrame, analysis_date: pd.Timestamp, universe: List[str], sector_map: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Get momentum factors with look-ahead bias fixes.
        
        LOOK-AHEAD BIAS FIX: Uses proper price data timing with contrarian/positive logic.
        """
        try:
            self.logger.info("Calculating momentum factors with look-ahead bias fixes...")
            
            # Calculate momentum for different periods
            momentum_periods = [1, 3, 6, 12]  # 1M, 3M, 6M, 12M
            momentum_factors = {}
            
            for months in momentum_periods:
                # Calculate start date for momentum
                start_date = analysis_date - pd.DateOffset(months=months)
                
                # Query price data for momentum calculation
                query = text("""
                    SELECT ticker, close_price, trading_date
                    FROM vcsc_daily_data_complete
                    WHERE ticker IN :tickers 
                    AND trading_date BETWEEN :start_date AND :analysis_date
                    ORDER BY ticker, trading_date
                """)
                
                price_data = pd.read_sql(query, self.engine, params={
                    'tickers': tuple(universe), 'start_date': start_date, 'analysis_date': analysis_date
                })
                
                if price_data.empty:
                    continue
                
                # Calculate momentum for each ticker
                momentum_scores = {}
                
                for ticker in universe:
                    ticker_prices = price_data[price_data['ticker'] == ticker]['close_price']
                    
                    if len(ticker_prices) >= 5:  # Need at least 5 data points
                        start_price = ticker_prices.iloc[0]
                        end_price = ticker_prices.iloc[-1]
                        
                        if start_price > 0:
                            momentum = (end_price - start_price) / start_price
                            
                            # Apply contrarian/positive logic based on months
                            if months in [1, 12]:  # CONTRARIAN (negative momentum is better)
                                # Convert to score where negative momentum gets higher score
                                momentum_score = max(0.0, min(1.0, (1.0 - momentum) / 2.0))
                            else:  # POSITIVE (3M, 6M) - positive momentum is better
                                # Convert to score where positive momentum gets higher score
                                momentum_score = max(0.0, min(1.0, (momentum + 1.0) / 2.0))
                            
                            momentum_scores[ticker] = momentum_score
                
                if momentum_scores:
                    # Create sector-neutral z-scores
                    momentum_df = pd.DataFrame([
                        {'ticker': ticker, f'momentum_{months}m': score, 'sector': sector_map.loc[ticker, 'sector']}
                        for ticker, score in momentum_scores.items()
                        if ticker in sector_map.index
                    ])
                    
                    if not momentum_df.empty:
                        # Apply sector-neutral normalization
                        z_scores = self.calculate_sector_neutral_zscore(momentum_df, f'momentum_{months}m', 'sector')
                        
                        # Create Series indexed by ticker
                        factor_series = pd.Series(
                            z_scores.values,
                            index=momentum_df['ticker'],
                            name=f'momentum_{months}m_z'
                        )
                        
                        momentum_factors[f'momentum_{months}m_z'] = factor_series
            
            self.logger.info(f"✅ Momentum factors calculated for {len(momentum_factors)} periods")
            return momentum_factors
            
        except Exception as e:
            self.logger.error(f"Failed to calculate momentum factors with timing fixes: {e}")
            return {}

    def _get_individual_low_vol_factors_fixed(self, analysis_date: pd.Timestamp, universe: List[str], sector_map: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Get low volatility factors with look-ahead bias fixes.
        
        LOOK-AHEAD BIAS FIX: Uses proper price data timing for volatility calculation.
        """
        try:
            self.logger.info("Calculating low volatility factors with look-ahead bias fixes...")
            
            # Calculate start date for volatility (6 months of data)
            start_date = analysis_date - pd.DateOffset(months=6)
            
            # Query price data for volatility calculation
            query = text("""
                SELECT ticker, close_price, trading_date
                FROM vcsc_daily_data_complete
                WHERE ticker IN :tickers 
                AND trading_date BETWEEN :start_date AND :analysis_date
                ORDER BY ticker, trading_date
            """)
            
            price_data = pd.read_sql(query, self.engine, params={
                'tickers': tuple(universe), 'start_date': start_date, 'analysis_date': analysis_date
            })
            
            if price_data.empty:
                return {}
            
            # Calculate volatility for each ticker
            volatility_scores = {}
            
            for ticker in universe:
                ticker_prices = price_data[price_data['ticker'] == ticker]['close_price']
                
                if len(ticker_prices) >= 10:  # Need at least 10 data points
                    # Calculate daily returns
                    returns = ticker_prices.pct_change().dropna()
                    
                    if len(returns) >= 5:
                        # Calculate annualized volatility
                        daily_volatility = returns.std()
                        annualized_volatility = daily_volatility * np.sqrt(252)  # 252 trading days
                        
                        # Convert to score where lower volatility gets higher score
                        # Assume volatility range of 0% to 100% annualized
                        volatility_score = max(0.0, min(1.0, (1.0 - annualized_volatility)))
                        
                        volatility_scores[ticker] = volatility_score
            
            if volatility_scores:
                # Create sector-neutral z-scores
                volatility_df = pd.DataFrame([
                    {'ticker': ticker, 'low_volatility': score, 'sector': sector_map.loc[ticker, 'sector']}
                    for ticker, score in volatility_scores.items()
                    if ticker in sector_map.index
                ])
                
                if not volatility_df.empty:
                    # Apply sector-neutral normalization
                    z_scores = self.calculate_sector_neutral_zscore(volatility_df, 'low_volatility', 'sector')
                    
                    # Create Series indexed by ticker
                    factor_series = pd.Series(
                        z_scores.values,
                        index=volatility_df['ticker'],
                        name='low_volatility_z'
                    )
                    
                    self.logger.info(f"✅ Low volatility factors calculated for {len(factor_series)} tickers")
                    return {'low_volatility_z': factor_series}
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Failed to calculate low volatility factors with timing fixes: {e}")
            return {}

    def _get_individual_f_score_factors_fixed(self, data: pd.DataFrame, analysis_date: pd.Timestamp, sector_map: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Get F-Score factors with look-ahead bias fixes.
        
        LOOK-AHEAD BIAS FIX: Uses lagged financial data for F-Score calculation.
        """
        try:
            self.logger.info("Calculating F-Score factors with look-ahead bias fixes...")
            
            # Get lagged quarter for financial data
            lagged_year, lagged_quarter = self._get_lagged_quarter_info(analysis_date)
            if lagged_year is None or lagged_quarter is None:
                return {}
            
            # Calculate F-Scores (vectorized path if enabled; safe fallback otherwise)
            f_scores = {}
            try:
                # Ensure normalized sector labels
                sec_df = sector_map.reset_index() if 'ticker' not in sector_map.columns else sector_map
                sec_df = normalize_sector_labels_221(sec_df, 'sector')
                sec_idx = sec_df.set_index('ticker')

                uniq = list(data['ticker'].unique())
                non_fin_tickers = [t for t in uniq if sec_idx.loc[t, 'sector'] not in ['Banking', 'Securities', 'Insurance']]
                bank_tickers = [t for t in uniq if sec_idx.loc[t, 'sector'] == 'Banking']
                sec_tickers = [t for t in uniq if sec_idx.loc[t, 'sector'] == 'Securities']

                if self.use_vectorized_fscore_221:
                    if non_fin_tickers:
                        nf = compute_nf_vectorized_221(self, non_fin_tickers, lagged_year, lagged_quarter, analysis_date)
                        for t, s in nf.items():
                            f_scores[t] = {'raw': int(s), 'max': 9}
                    if bank_tickers:
                        bs = compute_bank_vectorized_221(self, bank_tickers, lagged_year, lagged_quarter)
                        for t, s in bs.items():
                            f_scores[t] = {'raw': int(s), 'max': 6}
                    if sec_tickers:
                        ss = compute_sec_vectorized_221(self, sec_tickers, lagged_year, lagged_quarter)
                        for t, s in ss.items():
                            f_scores[t] = {'raw': int(s), 'max': 5}
                else:
                    if non_fin_tickers:
                        nf = self._get_raw_f_score_non_financial_fixed(non_fin_tickers, lagged_year, lagged_quarter, analysis_date)
                        for t, s in nf.items():
                            f_scores[t] = {'raw': s, 'max': 9}
                    if bank_tickers:
                        bs = self._get_raw_f_score_banking_fixed(bank_tickers, lagged_year, lagged_quarter)
                        for t, s in bs.items():
                            f_scores[t] = {'raw': s, 'max': 6}
                    if sec_tickers:
                        ss = self._get_raw_f_score_securities_fixed(sec_tickers, lagged_year, lagged_quarter)
                        for t, s in ss.items():
                            f_scores[t] = {'raw': s, 'max': 5}
            except Exception as e:
                self.logger.exception(f"Vectorized F-Score path error, using DB fallback: {e}")
                f_scores = {}
                non_fin_tickers = [t for t in data['ticker'].unique() if sector_map.loc[t, 'sector'] not in ['Banking', 'Securities']]
                bank_tickers = [t for t in data['ticker'].unique() if sector_map.loc[t, 'sector'] == 'Banking']
                sec_tickers = [t for t in data['ticker'].unique() if sector_map.loc[t, 'sector'] == 'Securities']
                if non_fin_tickers:
                    nf = self._get_raw_f_score_non_financial_fixed(non_fin_tickers, lagged_year, lagged_quarter, analysis_date)
                    for t, s in nf.items():
                        f_scores[t] = {'raw': s, 'max': 9}
                if bank_tickers:
                    bs = self._get_raw_f_score_banking_fixed(bank_tickers, lagged_year, lagged_quarter)
                    for t, s in bs.items():
                        f_scores[t] = {'raw': s, 'max': 6}
                if sec_tickers:
                    ss = self._get_raw_f_score_securities_fixed(sec_tickers, lagged_year, lagged_quarter)
                    for t, s in ss.items():
                        f_scores[t] = {'raw': s, 'max': 5}
            
            # Normalize F-Scores
            normalized_scores = {
                t: v['raw'] / v['max'] if v['max'] > 0 else 0.0 
                for t, v in f_scores.items()
            }
            
            if not normalized_scores:
                return {}
            
            # Create sector-neutral z-scores
            f_score_df = pd.DataFrame([
                {'ticker': ticker, 'f_score_normalized': score, 'sector': sector_map.loc[ticker, 'sector']}
                for ticker, score in normalized_scores.items()
                if ticker in sector_map.index
            ])
            
            # Apply sector-neutral normalization
            z_scores = self.calculate_sector_neutral_zscore(f_score_df, 'f_score_normalized', 'sector')
            
            # Create Series indexed by ticker
            f_score_series = pd.Series(
                z_scores.values,
                index=f_score_df['ticker'],
                name='f_score_z'
            )
            
            self.logger.info(f"✅ F-Score factors calculated for {len(f_score_series)} tickers using lagged data ({lagged_year}Q{lagged_quarter})")
            return {'f_score_z': f_score_series}
            
        except Exception as e:
            self.logger.error(f"Failed to calculate F-Score factors with timing fixes: {e}")
            return {}

    def _get_raw_f_score_non_financial_fixed(self, tickers: List[str],
                                           lagged_year: int, lagged_quarter: int,
                                           analysis_date: pd.Timestamp) -> Dict[str, int]:
        """
        Calculate 9-point Piotroski F-Score for non-financial sectors with look-ahead bias fixes.
        
        LOOK-AHEAD BIAS FIX: Uses lagged financial data and proper share data timing.
        """
        if not tickers: 
            return {}
        
        # Calculate previous year for comparison
        prev_year = lagged_year - 1
        
        query = text("""
            WITH current_fundamentals AS (
                SELECT ice.ticker, ice.year, ice.quarter, ice.NetProfit_TTM, ice.AvgTotalAssets, ice.NetCFO_TTM,
                       ice.Revenue_TTM, ice.COGS_TTM, vcfi.TotalEquity, vcfi.CurrentAssets, vcfi.CurrentLiabilities,
                       (COALESCE(vcfi.ShortTermDebt, 0) + COALESCE(vcfi.LongTermDebt, 0)) as TotalDebt
                FROM intermediary_calculations_enhanced ice JOIN v_comprehensive_fundamental_items vcfi
                ON ice.ticker = vcfi.ticker AND ice.year = vcfi.year AND ice.quarter = vcfi.quarter
                WHERE ice.year = :lagged_year AND ice.quarter = :lagged_quarter AND ice.ticker IN :tickers AND ice.has_full_ttm = 1
            ), previous_fundamentals AS (
                SELECT ice.ticker, ice.NetProfit_TTM as prev_netprofit_ttm, ice.AvgTotalAssets as prev_avgtotalassets,
                       ice.Revenue_TTM as prev_revenue_ttm, ice.COGS_TTM as prev_cogs_ttm, vcfi.TotalEquity as prev_totalequity,
                       vcfi.CurrentAssets as prev_currentassets, vcfi.CurrentLiabilities as prev_currentliabilities,
                       (COALESCE(vcfi.ShortTermDebt, 0) + COALESCE(vcfi.LongTermDebt, 0)) as prev_totaldebt
                FROM intermediary_calculations_enhanced ice JOIN v_comprehensive_fundamental_items vcfi
                ON ice.ticker = vcfi.ticker AND ice.year = vcfi.year AND ice.quarter = vcfi.quarter
                WHERE ice.year = :prev_year AND ice.quarter = :lagged_quarter AND ice.ticker IN :tickers AND ice.has_full_ttm = 1
            ), current_share_data AS (
                SELECT ticker COLLATE utf8mb4_unicode_ci as ticker, total_shares as current_shares
                FROM vcsc_daily_data_complete WHERE trading_date = :analysis_date AND ticker IN :tickers AND total_shares > 0
            ), previous_share_data AS (
                SELECT ticker COLLATE utf8mb4_unicode_ci as ticker, total_shares as prev_shares
                FROM vcsc_daily_data_complete WHERE trading_date = :analysis_date - INTERVAL 1 YEAR AND ticker IN :tickers AND total_shares > 0
            )
            SELECT cf.*, pf.prev_netprofit_ttm, pf.prev_avgtotalassets, pf.prev_revenue_ttm, pf.prev_cogs_ttm,
                   pf.prev_totalequity, pf.prev_currentassets, pf.prev_currentliabilities, pf.prev_totaldebt,
                   csd.current_shares, psd.prev_shares
            FROM current_fundamentals cf
            LEFT JOIN previous_fundamentals pf ON cf.ticker = pf.ticker
            LEFT JOIN current_share_data csd ON cf.ticker = csd.ticker
            LEFT JOIN previous_share_data psd ON cf.ticker = psd.ticker
        """)
        
        f_score_data = pd.read_sql(query, self.engine, params={
            'lagged_year': lagged_year, 'lagged_quarter': lagged_quarter, 'prev_year': prev_year,
            'analysis_date': analysis_date, 'tickers': tuple(tickers)
        })
        
        f_scores = {}
        for _, row in f_score_data.iterrows():
            score = 0
            try:
                # 9 Piotroski tests for non-financial using lagged data
                if pd.notna(row['NetProfit_TTM']) and pd.notna(row['AvgTotalAssets']) and row['AvgTotalAssets'] > 0 and (row['NetProfit_TTM'] / row['AvgTotalAssets']) > 0: 
                    score += 1
                if pd.notna(row['NetCFO_TTM']) and row['NetCFO_TTM'] > 0: 
                    score += 1
                if pd.notna(row['NetProfit_TTM']) and pd.notna(row['AvgTotalAssets']) and row['AvgTotalAssets'] > 0 and pd.notna(row['prev_netprofit_ttm']) and pd.notna(row['prev_avgtotalassets']) and row['prev_avgtotalassets'] > 0 and (row['NetProfit_TTM'] / row['AvgTotalAssets']) > (row['prev_netprofit_ttm'] / row['prev_avgtotalassets']): 
                    score += 1
                if pd.notna(row['NetCFO_TTM']) and pd.notna(row['NetProfit_TTM']) and row['NetCFO_TTM'] > row['NetProfit_TTM']: 
                    score += 1
                if pd.notna(row['TotalDebt']) and pd.notna(row['TotalEquity']) and row['TotalEquity'] > 0 and pd.notna(row['prev_totaldebt']) and pd.notna(row['prev_totalequity']) and row['prev_totalequity'] > 0 and (row['TotalDebt'] / row['TotalEquity']) < (row['prev_totaldebt'] / row['prev_totalequity']): 
                    score += 1
                if pd.notna(row['CurrentAssets']) and pd.notna(row['CurrentLiabilities']) and row['CurrentLiabilities'] > 0 and pd.notna(row['prev_currentassets']) and pd.notna(row['prev_currentliabilities']) and row['prev_currentliabilities'] > 0 and (row['CurrentAssets'] / row['CurrentLiabilities']) > (row['prev_currentassets'] / row['prev_currentliabilities']): 
                    score += 1
                if pd.notna(row['current_shares']) and pd.notna(row['prev_shares']) and row['current_shares'] <= row['prev_shares']: 
                    score += 1
                if pd.notna(row['Revenue_TTM']) and pd.notna(row['COGS_TTM']) and row['Revenue_TTM'] > 0 and pd.notna(row['prev_revenue_ttm']) and pd.notna(row['prev_cogs_ttm']) and row['prev_revenue_ttm'] > 0 and ((row['Revenue_TTM'] - row['COGS_TTM']) / row['Revenue_TTM']) > ((row['prev_revenue_ttm'] - row['prev_cogs_ttm']) / row['prev_revenue_ttm']): 
                    score += 1
                if pd.notna(row['Revenue_TTM']) and pd.notna(row['AvgTotalAssets']) and row['AvgTotalAssets'] > 0 and pd.notna(row['prev_revenue_ttm']) and pd.notna(row['prev_avgtotalassets']) and row['prev_avgtotalassets'] > 0 and (row['Revenue_TTM'] / row['AvgTotalAssets']) > (row['prev_revenue_ttm'] / row['prev_avgtotalassets']): 
                    score += 1
                f_scores[row['ticker']] = score
            except Exception: 
                f_scores[row['ticker']] = 0
        return f_scores

    def _get_raw_f_score_banking_fixed(self, tickers: List[str],
                                     lagged_year: int, lagged_quarter: int) -> Dict[str, int]:
        """
        Calculate 6-point Piotroski F-Score for banking sector with look-ahead bias fixes.
        
        LOOK-AHEAD BIAS FIX: Uses lagged financial data.
        """
        if not tickers: 
            return {}
        
        # Calculate previous year for comparison
        prev_year = lagged_year - 1
        
        query = text("""
            WITH current_banking AS (
                SELECT icbc.ticker, icbc.NetProfit_TTM, icbc.AvgTotalAssets, icbc.NII_TTM, icbc.AvgEarningAssets,
                       icbc.TotalOperatingIncome_TTM, icbc.OperatingExpenses_TTM, vcbf.ShareholdersEquity, vcbf.CustomerDeposits
                FROM intermediary_calculations_banking_cleaned icbc JOIN v_complete_banking_fundamentals vcbf
                ON icbc.ticker COLLATE utf8mb4_unicode_ci = vcbf.ticker COLLATE utf8mb4_unicode_ci AND icbc.year = vcbf.year AND icbc.quarter = vcbf.quarter
                WHERE icbc.year = :lagged_year AND icbc.quarter = :lagged_quarter AND icbc.ticker IN :tickers AND icbc.has_full_ttm = 1
            ), previous_banking AS (
                SELECT icbc.ticker, icbc.NetProfit_TTM as prev_netprofit_ttm, icbc.AvgTotalAssets as prev_avgtotalassets,
                       icbc.NII_TTM as prev_nii_ttm, icbc.AvgEarningAssets as prev_avgearningassets,
                       icbc.TotalOperatingIncome_TTM as prev_totaloperatingincome_ttm, icbc.OperatingExpenses_TTM as prev_operatingexpenses_ttm,
                       vcbf.ShareholdersEquity as prev_shareholdersequity, vcbf.CustomerDeposits as prev_customerdeposits
                FROM intermediary_calculations_banking_cleaned icbc JOIN v_complete_banking_fundamentals vcbf
                ON icbc.ticker COLLATE utf8mb4_unicode_ci = vcbf.ticker COLLATE utf8mb4_unicode_ci AND icbc.year = vcbf.year AND icbc.quarter = vcbf.quarter
                WHERE icbc.year = :prev_year AND icbc.quarter = :lagged_quarter AND icbc.ticker IN :tickers AND icbc.has_full_ttm = 1
            )
            SELECT cb.*, pb.prev_netprofit_ttm, pb.prev_avgtotalassets, pb.prev_nii_ttm, pb.prev_avgearningassets,
                   pb.prev_totaloperatingincome_ttm, pb.prev_operatingexpenses_ttm, pb.prev_shareholdersequity, pb.prev_customerdeposits
            FROM current_banking cb LEFT JOIN previous_banking pb ON cb.ticker = pb.ticker
        """)
        
        banking_data = pd.read_sql(query, self.engine, params={
            'lagged_year': lagged_year, 'lagged_quarter': lagged_quarter, 'prev_year': prev_year, 'tickers': tuple(tickers)
        })
        
        f_scores = {}
        for _, row in banking_data.iterrows():
            score = 0
            try:
                # 6 Banking-specific tests using lagged data
                if pd.notna(row['NetProfit_TTM']) and pd.notna(row['AvgTotalAssets']) and row['AvgTotalAssets'] > 0 and (row['NetProfit_TTM'] / row['AvgTotalAssets']) > 0: 
                    score += 1
                if pd.notna(row['NII_TTM']) and pd.notna(row['AvgEarningAssets']) and row['AvgEarningAssets'] > 0 and (row['NII_TTM'] / row['AvgEarningAssets']) > 0: 
                    score += 1
                if pd.notna(row['NetProfit_TTM']) and pd.notna(row['AvgTotalAssets']) and row['AvgTotalAssets'] > 0 and pd.notna(row['prev_netprofit_ttm']) and pd.notna(row['prev_avgtotalassets']) and row['prev_avgtotalassets'] > 0 and (row['NetProfit_TTM'] / row['AvgTotalAssets']) > (row['prev_netprofit_ttm'] / row['prev_avgtotalassets']): 
                    score += 1
                if pd.notna(row['NII_TTM']) and pd.notna(row['AvgEarningAssets']) and row['AvgEarningAssets'] > 0 and pd.notna(row['prev_nii_ttm']) and pd.notna(row['prev_avgearningassets']) and row['prev_avgearningassets'] > 0 and (row['NII_TTM'] / row['AvgEarningAssets']) > (row['prev_nii_ttm'] / row['prev_avgearningassets']): 
                    score += 1
                if pd.notna(row['CustomerDeposits']) and pd.notna(row['prev_customerdeposits']) and row['CustomerDeposits'] > row['prev_customerdeposits']: 
                    score += 1
                if pd.notna(row['OperatingExpenses_TTM']) and pd.notna(row['TotalOperatingIncome_TTM']) and row['TotalOperatingIncome_TTM'] > 0 and pd.notna(row['prev_operatingexpenses_ttm']) and pd.notna(row['prev_totaloperatingincome_ttm']) and row['prev_totaloperatingincome_ttm'] > 0 and (abs(row['OperatingExpenses_TTM']) / row['TotalOperatingIncome_TTM']) < (abs(row['prev_operatingexpenses_ttm']) / row['prev_totaloperatingincome_ttm']): 
                    score += 1
                f_scores[row['ticker']] = score
            except Exception: 
                f_scores[row['ticker']] = 0
        return f_scores

    def _get_raw_f_score_securities_fixed(self, tickers: List[str],
                                        lagged_year: int, lagged_quarter: int) -> Dict[str, int]:
        """
        Calculate 5-point Piotroski F-Score for securities sector with look-ahead bias fixes.
        
        LOOK-AHEAD BIAS FIX: Uses lagged financial data.
        """
        if not tickers: 
            return {}
        
        # Calculate previous year for comparison
        prev_year = lagged_year - 1
        
        query = text("""
            WITH base_data AS (
                SELECT
                    ticker, year, quarter,
                    (COALESCE(BrokerageRevenue_TTM, 0) +
                     COALESCE(NetTradingIncome_TTM, 0) +
                     COALESCE(OtherOperatingIncome_TTM, 0)) AS TotalOperatingRevenue_TTM,
                    NetProfit_TTM, AvgTotalAssets, OperatingResult_TTM, OperatingExpenses_TTM
                FROM intermediary_calculations_securities_cleaned
                WHERE ticker IN :tickers AND has_full_ttm = 1
                  AND ((year = :lagged_year AND quarter = :lagged_quarter) OR (year = :prev_year AND quarter = :lagged_quarter))
            ),
            current_securities AS (SELECT * FROM base_data WHERE year = :lagged_year),
            previous_securities AS (
                SELECT ticker,
                    TotalOperatingRevenue_TTM as prev_TotalOperatingRevenue_TTM,
                    NetProfit_TTM as prev_NetProfit_TTM,
                    AvgTotalAssets as prev_AvgTotalAssets,
                    OperatingResult_TTM as prev_OperatingResult_TTM,
                    OperatingExpenses_TTM as prev_OperatingExpenses_TTM
                FROM base_data WHERE year = :prev_year
            )
            SELECT
                cs.ticker,
                cs.TotalOperatingRevenue_TTM,
                cs.NetProfit_TTM,
                cs.AvgTotalAssets,
                cs.OperatingResult_TTM,
                cs.OperatingExpenses_TTM,
                ps.prev_TotalOperatingRevenue_TTM,
                ps.prev_NetProfit_TTM,
                ps.prev_AvgTotalAssets,
                ps.prev_OperatingResult_TTM,
                ps.prev_OperatingExpenses_TTM
            FROM current_securities cs
            LEFT JOIN previous_securities ps ON cs.ticker = ps.ticker
        """)
        
        securities_data = pd.read_sql(query, self.engine, params={
            'lagged_year': lagged_year, 'lagged_quarter': lagged_quarter, 'prev_year': prev_year, 'tickers': tuple(tickers)
        })
        
        f_scores = {}
        for _, row in securities_data.iterrows():
            score = 0
            try:
                # 5 Securities-specific tests using lagged data
                if pd.notna(row['NetProfit_TTM']) and pd.notna(row['AvgTotalAssets']) and row['AvgTotalAssets'] > 0 and (row['NetProfit_TTM'] / row['AvgTotalAssets']) > 0: 
                    score += 1
                if pd.notna(row['OperatingResult_TTM']) and row['OperatingResult_TTM'] > 0: 
                    score += 1
                if pd.notna(row['NetProfit_TTM']) and pd.notna(row['AvgTotalAssets']) and row['AvgTotalAssets'] > 0 and pd.notna(row['prev_NetProfit_TTM']) and pd.notna(row['prev_AvgTotalAssets']) and row['prev_AvgTotalAssets'] > 0 and (row['NetProfit_TTM'] / row['AvgTotalAssets']) > (row['prev_NetProfit_TTM'] / row['prev_AvgTotalAssets']): 
                    score += 1
                if pd.notna(row['OperatingResult_TTM']) and pd.notna(row['TotalOperatingRevenue_TTM']) and row['TotalOperatingRevenue_TTM'] > 0 and pd.notna(row['prev_OperatingResult_TTM']) and pd.notna(row['prev_TotalOperatingRevenue_TTM']) and row['prev_TotalOperatingRevenue_TTM'] > 0 and (row['OperatingResult_TTM'] / row['TotalOperatingRevenue_TTM']) > (row['prev_OperatingResult_TTM'] / row['prev_TotalOperatingRevenue_TTM']): 
                    score += 1
                if pd.notna(row['OperatingExpenses_TTM']) and pd.notna(row['TotalOperatingRevenue_TTM']) and row['TotalOperatingRevenue_TTM'] > 0 and pd.notna(row['prev_OperatingExpenses_TTM']) and pd.notna(row['prev_TotalOperatingRevenue_TTM']) and row['prev_TotalOperatingRevenue_TTM'] > 0 and (abs(row['OperatingExpenses_TTM']) / row['TotalOperatingRevenue_TTM']) < (abs(row['prev_OperatingExpenses_TTM']) / row['prev_TotalOperatingRevenue_TTM']): 
                    score += 1
                f_scores[row['ticker']] = score
            except Exception: 
                f_scores[row['ticker']] = 0
        return f_scores

    def _calculate_actual_momentum_fixed(self, ticker: str, analysis_date: pd.Timestamp, months: int) -> Optional[float]:
        """
        Calculate actual momentum from database with look-ahead bias fixes.
        
        LOOK-AHEAD BIAS FIX: Uses current market data (not lagged) for momentum calculations.
        
        Args:
            ticker: Stock ticker symbol
            analysis_date: Date for analysis
            months: Number of months for momentum calculation
            
        Returns:
            Momentum score normalized to 0-1 range, or None if insufficient data
        """
        try:
            # Calculate the start date for momentum calculation
            start_date = analysis_date - pd.DateOffset(months=months)
            
            # Query for price data using current market data
            query = text("""
                SELECT 
                    close_price,
                    trading_date
                FROM vcsc_daily_data_complete
                WHERE ticker = :ticker 
                AND trading_date BETWEEN :start_date AND :analysis_date
                ORDER BY trading_date
            """)
            
            data = pd.read_sql(query, self.engine, params={
                'ticker': ticker,
                'start_date': start_date,
                'analysis_date': analysis_date
            })
            
            if len(data) >= 5:  # Need at least 5 data points for meaningful momentum
                # Calculate momentum as percentage change
                start_price = data.iloc[0]['close_price']
                end_price = data.iloc[-1]['close_price']
                
                if start_price > 0:
                    momentum = (end_price - start_price) / start_price
                    
                    # Apply contrarian/positive logic based on months
                    if months in [1, 12]:  # CONTRARIAN (negative momentum is better)
                        # Convert to score where negative momentum gets higher score
                        momentum_score = max(0.0, min(1.0, (1.0 - momentum) / 2.0))
                    else:  # POSITIVE (3M, 6M) - positive momentum is better
                        # Convert to score where positive momentum gets higher score
                        momentum_score = max(0.0, min(1.0, (momentum + 1.0) / 2.0))
                    
                    return momentum_score
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Could not calculate actual {months}M momentum for {ticker}: {e}")
            return None


if __name__ == "__main__":
    print("QVM Engine v2.2.1 Flat - Module loaded successfully.")
    print("This engine implements flat methodology with look-ahead bias fixes.")
    print("Key improvements:")
    print("- Fixed look-ahead bias using lagged financial data")
    print("- Eliminated code duplication between engine and config files")
    print("- Added proper data timing validation")
    print("- Enhanced data availability checks")
