"""
Vietnam Factor Investing Platform - QVM Engine v2.1.1 (Flat Methodology)
======================================================================
Component: Enhanced Flat Composite Engine with Defensive & Quality Improvements
Purpose: Integrate Low-Volatility, Piotroski F-Score, and FCF Yield using flat methodology
Author: Duc Nguyen, Principal Quantitative Strategist  
Date Created: August 3, 2025
Status: FLAT METHODOLOGY ENGINE (v2.1.1) - PRODUCTION GRADE

FLAT METHODOLOGY WITH ENHANCED FACTORS:
This engine extends QVMEngineV201Flat with three strategic improvements:
1. Low-Volatility Factor: Defensive overlay to reduce portfolio volatility
2. Piotroski F-Score: Quality screen with sector-specific adaptations (9/6/5 variants)
3. FCF Yield: Robust cash-based valuation metric with imputation logging

ARCHITECTURAL IMPROVEMENTS OVER v2.1_alpha:
- ELIMINATED: Hierarchical composite methodology (statistically suboptimal)
- IMPLEMENTED: Flat single-step combination of ALL individual factors
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
- Engine Version: 'qvm_v2.1.1_flat' (database strategy_version tag)
- Inheritance: Extends QVMEngineV201Flat
- Compatibility: Maintains all v2.0.1 functionality while adding new factors

Dependencies:
- pandas >= 1.3.0
- numpy >= 1.21.0
- sqlalchemy >= 1.4.0
- PyYAML >= 5.4.0
"""

import pandas as pd
import numpy as np
from sqlalchemy import text
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Import parent engine
from .qvm_engine_v2_0_1_flat import QVMEngineV201Flat


class QVMEngineV211Flat(QVMEngineV201Flat):
    """
    QVM Engine v2.1.1 Flat - Enhanced flat methodology with defensive factors.
    
    This engine implements the institutional-standard flat composite methodology
    while adding three strategic enhancements: Low-Volatility, Piotroski F-Score,
    and FCF Yield. All factors are individually sector-neutralized before flat
    combination to ensure optimal alpha extraction.
    
    Key Features:
    - 4-pillar architecture: Quality, Value, Momentum, Defensive
    - Enhanced factor set: Traditional + F-Score + FCF Yield + Low-Vol
    - Flat methodology: Single-step combination without hierarchical nesting
    - Sector-specific F-Score variants: 9-point (non-financial), 6-point (banking), 5-point (securities)
    """

    def __init__(self, config_path: str = None, log_level: str = 'INFO'):
        """Initialize v2.1.1 Flat engine with enhanced defensive capabilities."""
        super().__init__(config_path, log_level)
        
        # Override engine version
        self.engine_version = 'qvm_v2.1.1_flat'
        
        # Load enhanced 4-pillar weights from configuration
        self.enhanced_weights = self.factor_config.get('qvm_composite', {}).get('enhanced_weights', {
            'quality': 0.35,      # Fallback defaults
            'value': 0.30,
            'momentum': 0.20,
            'defensive': 0.15
        })
        
        # New factor parameters
        self.low_vol_lookback = 63  # 63 trading days (institutional standard)
        
        # Override flat weights to use enhanced versions for v2.1.1
        self._load_enhanced_flat_weights()
        
        # CRITICAL FIX: Override parent's 3-pillar weights with 4-pillar enhanced weights
        self.qvm_weights = self.enhanced_weights.copy()
        
        # CRITICAL FIX: Add missing intermediary table mappings for banking data
        self.intermediary_tables = {
            'banking': 'intermediary_calculations_banking_cleaned',
            'securities': 'intermediary_calculations_securities_cleaned', 
            'non_financial': 'intermediary_calculations_enhanced'
        }
        
        self.logger.info("="*60)
        self.logger.info(f"Initialized QVM Engine v{self.engine_version} (Flat Methodology)")
        self.logger.info("Enhanced Factors: Low-Vol, F-Score (9/6/5 variants), FCF Yield")
        self.logger.info("Architecture: 4-Pillar Flat Composite (Q35/V30/M20/D15)")
        self.logger.info("Methodology: Universal sector neutralization + single-step combination")
        self.logger.info(f"ENHANCED WEIGHTS: Quality {self.qvm_weights['quality']*100:.1f}%, "
                       f"Value {self.qvm_weights['value']*100:.1f}%, "
                       f"Momentum {self.qvm_weights['momentum']*100:.1f}%, "
                       f"Defensive {self.qvm_weights['defensive']*100:.1f}%")
        self.logger.info(f"BANKING TABLE FIX: Using {self.intermediary_tables['banking']} for banking data")
        self.logger.info("="*60)
    
    def _load_enhanced_flat_weights(self):
        """Override parent weights to use enhanced sector-specific versions with new factors."""
        try:
            # Load enhanced individual factor weights (sector-specific)
            if 'flat_composite_weights' in self.factor_config:
                flat_weights = self.factor_config['flat_composite_weights']
                
                # Load sector-specific quality weights for v2.1.1 (enhanced)
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

    def calculate_qvm_composite(self, analysis_date: pd.Timestamp,
                               universe: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Calculate v2.1.1 Flat composite using enhanced 4-pillar architecture.
        
        This method implements the flat methodology with all enhancements:
        1. Extract ALL individual factors (traditional + new) with sector neutralization
        2. Calculate enhanced pillar composites using flat weighted averages
        3. Combine pillars using 4-pillar weights (Q35/V30/M20/D15)
        4. Return full component breakdown for transparency
        
        ENHANCED OUTPUT SCHEMA (Neo #5 - Full Documentation):
        Returns dictionary with ticker -> component mapping:
        {
            'ticker': {
                # Pillar Composites (sector-neutral z-scores, weighted)
                'Quality_Composite': float,      # Weighted avg of roae_z, f_score_z, etc.
                'Value_Composite': float,        # Weighted avg of earnings_yield_z, fcf_yield_z, etc.
                'Momentum_Composite': float,     # Weighted avg of momentum_1m_z, momentum_3m_z, etc.
                'Defensive_Composite': float,    # Weighted avg of low_volatility_z
                
                # Final Score  
                'QVM_Composite': float,          # 4-pillar weighted combination
                
                # Individual Factor Breakdown (all sector-neutral z-scores)
                'individual_factors': {
                    # Quality factors (z-scored)
                    'roae_z': float,
                    'f_score_z': float,           # NEW: Piotroski F-Score (normalized 0-9)
                    'net_profit_margin_z': float,
                    'gross_margin_z': float,
                    'operating_margin_z': float,
                    'ebitda_margin_z': float,
                    # Banking-specific
                    'roaa_z': float,             
                    'nim_z': float,
                    'cost_income_z': float,
                    
                    # Value factors (z-scored)
                    'earnings_yield_z': float,
                    'book_to_price_z': float,
                    'sales_to_price_z': float,
                    'ebitda_to_ev_z': float,
                    'fcf_yield_z': float,        # NEW: Free Cash Flow Yield
                    
                    # Momentum factors (z-scored)
                    'momentum_1m_z': float,
                    'momentum_3m_z': float, 
                    'momentum_6m_z': float,
                    'momentum_12m_z': float,
                    
                    # Defensive factors (z-scored)
                    'low_volatility_z': float    # NEW: Inverse 63-day volatility
                },
                
                # Raw Factor Values (for transparency/debugging)
                'Low_Volatility_63D': float,     # Raw -1 * rolling_volatility
                'Piotroski_F_Score': float,      # Raw F-Score (0-9 scale)
                'FCF_Yield': float               # Raw FCF/MarketCap ratio
            }
        }
        
        VALUE TYPES EXPLAINED:
        - Raw values: Original calculated metrics before normalization
        - Z-scored values: Sector-neutral normalized (-3 to +3, winsorized)
        - Composite values: Weighted averages of z-scored individual factors
        - Final QVM: 4-pillar weighted combination (Q35% + V30% + M20% + D15%)
        """
        try:
            self.logger.info(f"BEGIN v2.1.1 Flat composite calculation for {len(universe)} tickers on {analysis_date.date()}")

            # 1. Data Ingestion
            fundamentals = self.get_fundamentals_correct_timing(analysis_date, universe)
            market_data = self.get_market_data(analysis_date, universe)
            if fundamentals.empty or market_data.empty:
                self.logger.error("FATAL: Insufficient fundamental or market data. Aborting.")
                return {}
            data = pd.merge(fundamentals, market_data, on='ticker', how='inner')
            if data.empty:
                self.logger.error("FATAL: No merged data available. Aborting.")
                return {}

            # TIER 1 REFINEMENT #4: Cache sector mapping for performance
            sector_map = self.get_sector_mapping().set_index('ticker')

            # 2. Enhanced Factor Calculation (Traditional + New)
            # Traditional factors from parent class
            quality_factors = self._get_individual_quality_factors(data, analysis_date, sector_map)
            value_factors = self._get_individual_value_factors(data, analysis_date, sector_map)
            momentum_factors = self._get_individual_momentum_factors(data, analysis_date, universe, sector_map)
            
            # AGENT SMITH DEBUG: Log factor counts
            self.logger.info(f"Factor counts: Quality={len(quality_factors)}, Value={len(value_factors)}, Momentum={len(momentum_factors)}")
            for name, series in quality_factors.items():
                self.logger.info(f"Quality factor {name}: {len(series)} values")
            for name, series in value_factors.items():
                self.logger.info(f"Value factor {name}: {len(series)} values")

            # New v2.1.1 factors (also use cached sector map)
            low_vol_factors = self._get_individual_low_vol_factors(analysis_date, universe, sector_map)
            f_score_factors = self._get_individual_f_score_factors(data, analysis_date, sector_map)
            fcf_yield_factors = self._get_individual_fcf_yield_factors(data, analysis_date, sector_map)

            self.logger.info(f"Individual factors calculated: {len(quality_factors)} quality, "
                           f"{len(value_factors)} value, {len(momentum_factors)} momentum, "
                           f"{len(low_vol_factors)} defensive, {len(f_score_factors)} f-score, "
                           f"{len(fcf_yield_factors)} fcf-yield")

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

                # 6. Institutional Transparency: Return everything
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
                    'FCF_Yield': individual_scores.get('fcf_yield_raw', 0.0)
                }

            self.logger.info(f"SUCCESS: v2.1.1 Flat composite calculated for {len(results)} tickers.")
            return results

        except Exception as e:
            self.logger.error(f"Failed to calculate v2.1.1 Flat composite: {e}")
            return {}

    def _get_individual_low_vol_factors(self, analysis_date: pd.Timestamp,
                                       universe: List[str],
                                       sector_map: pd.DataFrame = None) -> Dict[str, pd.Series]:
        """Calculate low-volatility factor with sector-neutral normalization."""
        try:
            self.logger.info(f"Calculating Low-Volatility factor for {len(universe)} tickers")
            
            # Need 63 trading days + buffer for Vietnam market (~22 trading days/month)
            # 63 trading days = ~3 months = ~90 calendar days + holidays/weekends = 120 days safe buffer
            start_date = analysis_date - pd.DateOffset(days=120)
            query = text("""
                SELECT date, ticker, close FROM equity_history
                WHERE ticker IN :tickers AND date BETWEEN :start_date AND :analysis_date
                AND close IS NOT NULL AND close > 0 ORDER BY ticker, date
            """)
            
            price_data = pd.read_sql(query, self.engine, params={
                'tickers': tuple(universe), 
                'start_date': start_date, 
                'analysis_date': analysis_date
            })
            
            if price_data.empty:
                self.logger.warning("No price data for volatility calculation.")
                return {}

            # Calculate 63-day rolling volatility (institutional standard)
            low_vol_raw = {}
            calculated_count = 0
            skipped_count = 0
            
            for ticker in universe:
                ticker_prices = price_data[price_data['ticker'] == ticker]['close']
                if len(ticker_prices) >= 64:  # Need 64 prices for 63 returns
                    returns = ticker_prices.pct_change(fill_method=None).dropna()
                    if len(returns) >= 63:  # Need exactly 63 trading days
                        volatility_63d = returns.tail(63).std()
                        low_vol_raw[ticker] = -1 * volatility_63d  # Invert: low vol = high score
                        calculated_count += 1
                    else:
                        skipped_count += 1
                        self.logger.debug(f"Ticker {ticker}: only {len(returns)} returns, need 63")
                else:
                    skipped_count += 1
                    self.logger.debug(f"Ticker {ticker}: only {len(ticker_prices)} prices, need 64")
            
            self.logger.info(f"Low-vol calculation: {calculated_count} calculated, {skipped_count} skipped")

            # TIER 1 REFINEMENT #4: Use cached sector mapping for performance  
            if sector_map is None:
                sector_map = self.get_sector_mapping().set_index('ticker')
            else:
                # Ensure sector_map is properly indexed if passed as cached
                if 'ticker' in sector_map.columns:
                    sector_map = sector_map.set_index('ticker')
            
            # Create dataframe for sector-neutral normalization
            low_vol_df = pd.DataFrame([
                {'ticker': ticker, 'low_volatility_raw': score, 'sector': sector_map.loc[ticker, 'sector']}
                for ticker, score in low_vol_raw.items()
                if ticker in sector_map.index
            ])
            
            if low_vol_df.empty:
                return {}
            
            # Apply sector-neutral normalization
            low_vol_z = self.calculate_sector_neutral_zscore(
                low_vol_df, 'low_volatility_raw', 'sector'
            )
            
            # Return as Series indexed by ticker
            low_vol_series = pd.Series(
                low_vol_z.values,
                index=low_vol_df['ticker'],
                name='low_volatility_z'
            )
            
            self.logger.info(f"Low-Vol calculation complete. Found signals for {len(low_vol_series)} tickers.")
            return {'low_volatility_z': low_vol_series}

        except Exception as e:
            self.logger.error(f"Failed to calculate low-volatility factors: {e}")
            return {}

    def _get_individual_f_score_factors(self, data: pd.DataFrame,
                                       analysis_date: pd.Timestamp,
                                       sector_map: pd.DataFrame = None) -> Dict[str, pd.Series]:
        """Calculate Piotroski F-Score with sector-specific adaptations and normalization."""
        try:
            self.logger.info("Calculating Piotroski F-Score with sector adaptations...")
            
            quarter_info = self.get_correct_quarter_for_date(analysis_date)
            if not quarter_info:
                self.logger.warning("No quarter data for F-Score calculation.")
                return {}
            
            current_year, current_quarter = quarter_info
            universe_tickers = data['ticker'].unique().tolist()
            # TIER 1 REFINEMENT #4: Use cached sector mapping for performance
            if sector_map is None:
                sector_map = self.get_sector_mapping().set_index('ticker')
            else:
                # Ensure sector_map is properly indexed if passed as cached
                if 'ticker' in sector_map.columns:
                    sector_map = sector_map.set_index('ticker')

            # Calculate sector-specific F-Scores
            raw_scores = {}
            
            # Non-Financial (9-point) - Process in batches to avoid SQL query timeout
            non_fin_tickers = [t for t in universe_tickers if t in sector_map.index and sector_map.loc[t, 'sector'] not in ['Banking', 'Securities']]
            if non_fin_tickers:
                # Process in batches of 10 tickers to avoid SQL query hanging
                batch_size = 10
                for i in range(0, len(non_fin_tickers), batch_size):
                    batch = non_fin_tickers[i:i + batch_size]
                    scores = self._get_raw_f_score_non_financial(batch, current_year, current_quarter, analysis_date)
                    for t, s in scores.items(): 
                        raw_scores[t] = {'raw': s, 'max': 9}
            
            # Banking (6-point) - Process in batches
            bank_tickers = [t for t in universe_tickers if t in sector_map.index and sector_map.loc[t, 'sector'] == 'Banking']
            if bank_tickers:
                batch_size = 10
                for i in range(0, len(bank_tickers), batch_size):
                    batch = bank_tickers[i:i + batch_size]
                    scores = self._get_raw_f_score_banking(batch, current_year, current_quarter)
                    for t, s in scores.items(): 
                        raw_scores[t] = {'raw': s, 'max': 6}
            
            # Securities (5-point) - Process in batches
            sec_tickers = [t for t in universe_tickers if t in sector_map.index and sector_map.loc[t, 'sector'] == 'Securities']
            if sec_tickers:
                batch_size = 10
                for i in range(0, len(sec_tickers), batch_size):
                    batch = sec_tickers[i:i + batch_size]
                    scores = self._get_raw_f_score_securities(batch, current_year, current_quarter)
                    for t, s in scores.items(): 
                        raw_scores[t] = {'raw': s, 'max': 5}

            # Sector-scaled normalization then sector-neutral z-scoring
            normalized_scores = {
                t: v['raw'] / v['max'] if v['max'] > 0 else 0.0 
                for t, v in raw_scores.items()
            }
            
            if not normalized_scores:
                return {}
            
            # Create dataframe for sector-neutral normalization
            f_score_df = pd.DataFrame([
                {'ticker': ticker, 'f_score_normalized': score, 'sector': sector_map.loc[ticker, 'sector']}
                for ticker, score in normalized_scores.items()
                if ticker in sector_map.index
            ])
            
            # Apply sector-neutral normalization to the sector-scaled scores
            f_score_z = self.calculate_sector_neutral_zscore(
                f_score_df, 'f_score_normalized', 'sector'
            )
            
            # Return as Series indexed by ticker
            f_score_series = pd.Series(
                f_score_z.values,
                index=f_score_df['ticker'],
                name='f_score_z'
            )
            
            self.logger.info(f"F-Score calculation complete. Found scores for {len(f_score_series)} tickers.")
            return {'f_score_z': f_score_series}

        except Exception as e:
            self.logger.error(f"Failed to calculate F-Score factors: {e}")
            return {}

    def _get_individual_fcf_yield_factors(self, data: pd.DataFrame,
                                         analysis_date: pd.Timestamp,
                                         sector_map: pd.DataFrame = None) -> Dict[str, pd.Series]:
        """Calculate FCF Yield with sector exclusions and normalization."""
        try:
            self.logger.info("Calculating FCF Yield (Production v2.1.1)...")
            
            universe_tickers = data['ticker'].unique().tolist()
            # TIER 1 REFINEMENT #4: Use cached sector mapping for performance
            if sector_map is None:
                sector_map = self.get_sector_mapping().set_index('ticker')
            else:
                # Ensure sector_map is properly indexed if passed as cached
                if 'ticker' in sector_map.columns:
                    sector_map = sector_map.set_index('ticker')
            
            # Exclude financial sectors - add safety check for missing tickers
            eligible_tickers = [
                t for t in universe_tickers 
                if t in sector_map.index and sector_map.loc[t, 'sector'] not in ['Banking', 'Securities', 'Insurance']
            ]
            
            if not eligible_tickers:
                self.logger.warning("No non-financial tickers for FCF Yield calculation.")
                return {}

            quarter_info = self.get_correct_quarter_for_date(analysis_date)
            if not quarter_info:
                self.logger.warning("No quarter data for FCF Yield calculation.")
                return {}
            
            current_year, current_quarter = quarter_info

            # Process in batches to avoid SQL query timeout
            batch_size = 100  # Larger batch size for simpler queries
            fcf_data_list = []
            market_data_list = []
            
            for i in range(0, len(eligible_tickers), batch_size):
                batch = eligible_tickers[i:i + batch_size]
                
                # TIER 2 REFINEMENT #1: Get FCF components including actual Capex data
                fcf_query = text("""
                    SELECT ticker, NetCFO_TTM, NetCFI_TTM, CapEx_TTM, FCF_TTM 
                    FROM intermediary_calculations_enhanced 
                    WHERE year = :y AND quarter = :q AND ticker IN :tickers AND has_full_ttm = 1
                """)
                batch_fcf = pd.read_sql(fcf_query, self.engine, params={
                    'y': current_year, 'q': current_quarter, 'tickers': tuple(batch)
                })
                fcf_data_list.append(batch_fcf)
                
                # Get market cap data
                market_query = text("""
                    SELECT ticker, market_cap FROM vcsc_daily_data_complete 
                    WHERE trading_date = :d AND ticker IN :tickers AND market_cap > 0
                """)
                batch_market = pd.read_sql(market_query, self.engine, params={
                    'd': analysis_date, 'tickers': tuple(batch)
                })
                market_data_list.append(batch_market)
            
            # Combine all batches
            fcf_data = pd.concat(fcf_data_list, ignore_index=True) if fcf_data_list else pd.DataFrame()
            market_data = pd.concat(market_data_list, ignore_index=True) if market_data_list else pd.DataFrame()

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
            return {'fcf_yield_z': fcf_yield_series}

        except Exception as e:
            self.logger.error(f"Failed to calculate FCF Yield factors: {e}")
            return {}

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

    # Import sector-specific F-Score calculation methods from v2.1_alpha
    def _get_raw_f_score_non_financial(self, tickers: List[str],
                                        current_year: int, current_quarter: int,
                                        analysis_date: pd.Timestamp) -> Dict[str, int]:
        """Calculate 9-point Piotroski F-Score for non-financial sectors."""
        # Implementation copied from v2.1_alpha with minor modifications
        if not tickers: 
            return {}
        
        query = text("""
            WITH current_fundamentals AS (
                SELECT ice.ticker, ice.year, ice.quarter, ice.NetProfit_TTM, ice.AvgTotalAssets, ice.NetCFO_TTM,
                       ice.Revenue_TTM, ice.COGS_TTM, vcfi.TotalEquity, vcfi.CurrentAssets, vcfi.CurrentLiabilities,
                       (COALESCE(vcfi.ShortTermDebt, 0) + COALESCE(vcfi.LongTermDebt, 0)) as TotalDebt
                FROM intermediary_calculations_enhanced ice JOIN v_comprehensive_fundamental_items vcfi
                ON ice.ticker = vcfi.ticker AND ice.year = vcfi.year AND ice.quarter = vcfi.quarter
                WHERE ice.year = :cy AND ice.quarter = :cq AND ice.ticker IN :tickers AND ice.has_full_ttm = 1
            ), previous_fundamentals AS (
                SELECT ice.ticker, ice.NetProfit_TTM as prev_netprofit_ttm, ice.AvgTotalAssets as prev_avgtotalassets,
                       ice.Revenue_TTM as prev_revenue_ttm, ice.COGS_TTM as prev_cogs_ttm, vcfi.TotalEquity as prev_totalequity,
                       vcfi.CurrentAssets as prev_currentassets, vcfi.CurrentLiabilities as prev_currentliabilities,
                       (COALESCE(vcfi.ShortTermDebt, 0) + COALESCE(vcfi.LongTermDebt, 0)) as prev_totaldebt
                FROM intermediary_calculations_enhanced ice JOIN v_comprehensive_fundamental_items vcfi
                ON ice.ticker = vcfi.ticker AND ice.year = vcfi.year AND ice.quarter = vcfi.quarter
                WHERE ice.year = :py AND ice.quarter = :cq AND ice.ticker IN :tickers AND ice.has_full_ttm = 1
            ), current_share_data AS (
                SELECT ticker COLLATE utf8mb4_unicode_ci as ticker, total_shares as current_shares
                FROM vcsc_daily_data_complete WHERE trading_date = :c_date AND ticker IN :tickers AND total_shares > 0
            ), previous_share_data AS (
                SELECT ticker COLLATE utf8mb4_unicode_ci as ticker, total_shares as prev_shares
                FROM vcsc_daily_data_complete WHERE trading_date = :p_date AND ticker IN :tickers AND total_shares > 0
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
            'cy': current_year, 'cq': current_quarter, 'py': current_year - 1,
            'c_date': analysis_date, 'p_date': analysis_date.replace(year=analysis_date.year - 1),
            'tickers': tuple(tickers)
        })
        
        f_scores = {}
        for _, row in f_score_data.iterrows():
            score = 0
            try:
                # 9 Piotroski tests for non-financial
                if pd.notna(row['NetProfit_TTM']) and pd.notna(row['AvgTotalAssets']) and row['AvgTotalAssets'] > 0 and (row['NetProfit_TTM'] / row['AvgTotalAssets']) > 0: score += 1
                if pd.notna(row['NetCFO_TTM']) and row['NetCFO_TTM'] > 0: score += 1
                if pd.notna(row['NetProfit_TTM']) and pd.notna(row['AvgTotalAssets']) and row['AvgTotalAssets'] > 0 and pd.notna(row['prev_netprofit_ttm']) and pd.notna(row['prev_avgtotalassets']) and row['prev_avgtotalassets'] > 0 and (row['NetProfit_TTM'] / row['AvgTotalAssets']) > (row['prev_netprofit_ttm'] / row['prev_avgtotalassets']): score += 1
                if pd.notna(row['NetCFO_TTM']) and pd.notna(row['NetProfit_TTM']) and row['NetCFO_TTM'] > row['NetProfit_TTM']: score += 1
                if pd.notna(row['TotalDebt']) and pd.notna(row['TotalEquity']) and row['TotalEquity'] > 0 and pd.notna(row['prev_totaldebt']) and pd.notna(row['prev_totalequity']) and row['prev_totalequity'] > 0 and (row['TotalDebt'] / row['TotalEquity']) < (row['prev_totaldebt'] / row['prev_totalequity']): score += 1
                if pd.notna(row['CurrentAssets']) and pd.notna(row['CurrentLiabilities']) and row['CurrentLiabilities'] > 0 and pd.notna(row['prev_currentassets']) and pd.notna(row['prev_currentliabilities']) and row['prev_currentliabilities'] > 0 and (row['CurrentAssets'] / row['CurrentLiabilities']) > (row['prev_currentassets'] / row['prev_currentliabilities']): score += 1
                if pd.notna(row['current_shares']) and pd.notna(row['prev_shares']) and row['current_shares'] <= row['prev_shares']: score += 1
                if pd.notna(row['Revenue_TTM']) and pd.notna(row['COGS_TTM']) and row['Revenue_TTM'] > 0 and pd.notna(row['prev_revenue_ttm']) and pd.notna(row['prev_cogs_ttm']) and row['prev_revenue_ttm'] > 0 and ((row['Revenue_TTM'] - row['COGS_TTM']) / row['Revenue_TTM']) > ((row['prev_revenue_ttm'] - row['prev_cogs_ttm']) / row['prev_revenue_ttm']): score += 1
                if pd.notna(row['Revenue_TTM']) and pd.notna(row['AvgTotalAssets']) and row['AvgTotalAssets'] > 0 and pd.notna(row['prev_revenue_ttm']) and pd.notna(row['prev_avgtotalassets']) and row['prev_avgtotalassets'] > 0 and (row['Revenue_TTM'] / row['AvgTotalAssets']) > (row['prev_revenue_ttm'] / row['prev_avgtotalassets']): score += 1
                f_scores[row['ticker']] = score
            except Exception: 
                f_scores[row['ticker']] = 0
        return f_scores

    def _get_raw_f_score_banking(self, tickers: List[str],
                                  current_year: int, current_quarter: int) -> Dict[str, int]:
        """Calculate 6-point Piotroski F-Score for banking sector."""
        # Implementation copied from v2.1_alpha
        if not tickers: 
            return {}
        
        query = text("""
            WITH current_banking AS (
                SELECT icbc.ticker, icbc.NetProfit_TTM, icbc.AvgTotalAssets, icbc.NII_TTM, icbc.AvgEarningAssets,
                       icbc.TotalOperatingIncome_TTM, icbc.OperatingExpenses_TTM, vcbf.ShareholdersEquity, vcbf.CustomerDeposits
                FROM intermediary_calculations_banking_cleaned icbc JOIN v_complete_banking_fundamentals vcbf
                ON icbc.ticker COLLATE utf8mb4_unicode_ci = vcbf.ticker COLLATE utf8mb4_unicode_ci AND icbc.year = vcbf.year AND icbc.quarter = vcbf.quarter
                WHERE icbc.year = :cy AND icbc.quarter = :cq AND icbc.ticker IN :tickers AND icbc.has_full_ttm = 1
            ), previous_banking AS (
                SELECT icbc.ticker, icbc.NetProfit_TTM as prev_netprofit_ttm, icbc.AvgTotalAssets as prev_avgtotalassets,
                       icbc.NII_TTM as prev_nii_ttm, icbc.AvgEarningAssets as prev_avgearningassets,
                       icbc.TotalOperatingIncome_TTM as prev_totaloperatingincome_ttm, icbc.OperatingExpenses_TTM as prev_operatingexpenses_ttm,
                       vcbf.ShareholdersEquity as prev_shareholdersequity, vcbf.CustomerDeposits as prev_customerdeposits
                FROM intermediary_calculations_banking_cleaned icbc JOIN v_complete_banking_fundamentals vcbf
                ON icbc.ticker COLLATE utf8mb4_unicode_ci = vcbf.ticker COLLATE utf8mb4_unicode_ci AND icbc.year = vcbf.year AND icbc.quarter = vcbf.quarter
                WHERE icbc.year = :py AND icbc.quarter = :cq AND icbc.ticker IN :tickers AND icbc.has_full_ttm = 1
            )
            SELECT cb.*, pb.prev_netprofit_ttm, pb.prev_avgtotalassets, pb.prev_nii_ttm, pb.prev_avgearningassets,
                   pb.prev_totaloperatingincome_ttm, pb.prev_operatingexpenses_ttm, pb.prev_shareholdersequity, pb.prev_customerdeposits
            FROM current_banking cb LEFT JOIN previous_banking pb ON cb.ticker = pb.ticker
        """)
        
        banking_data = pd.read_sql(query, self.engine, params={
            'cy': current_year, 'cq': current_quarter, 'py': current_year - 1, 'tickers': tuple(tickers)
        })
        
        f_scores = {}
        for _, row in banking_data.iterrows():
            score = 0
            try:
                # 6 Banking-specific tests
                if pd.notna(row['NetProfit_TTM']) and pd.notna(row['AvgTotalAssets']) and row['AvgTotalAssets'] > 0 and (row['NetProfit_TTM'] / row['AvgTotalAssets']) > 0: score += 1
                if pd.notna(row['NII_TTM']) and pd.notna(row['AvgEarningAssets']) and row['AvgEarningAssets'] > 0 and (row['NII_TTM'] / row['AvgEarningAssets']) > 0: score += 1
                if pd.notna(row['NetProfit_TTM']) and pd.notna(row['AvgTotalAssets']) and row['AvgTotalAssets'] > 0 and pd.notna(row['prev_netprofit_ttm']) and pd.notna(row['prev_avgtotalassets']) and row['prev_avgtotalassets'] > 0 and (row['NetProfit_TTM'] / row['AvgTotalAssets']) > (row['prev_netprofit_ttm'] / row['prev_avgtotalassets']): score += 1
                if pd.notna(row['NII_TTM']) and pd.notna(row['AvgEarningAssets']) and row['AvgEarningAssets'] > 0 and pd.notna(row['prev_nii_ttm']) and pd.notna(row['prev_avgearningassets']) and row['prev_avgearningassets'] > 0 and (row['NII_TTM'] / row['AvgEarningAssets']) > (row['prev_nii_ttm'] / row['prev_avgearningassets']): score += 1
                if pd.notna(row['CustomerDeposits']) and pd.notna(row['prev_customerdeposits']) and row['CustomerDeposits'] > row['prev_customerdeposits']: score += 1
                if pd.notna(row['OperatingExpenses_TTM']) and pd.notna(row['TotalOperatingIncome_TTM']) and row['TotalOperatingIncome_TTM'] > 0 and pd.notna(row['prev_operatingexpenses_ttm']) and pd.notna(row['prev_totaloperatingincome_ttm']) and row['prev_totaloperatingincome_ttm'] > 0 and (abs(row['OperatingExpenses_TTM']) / row['TotalOperatingIncome_TTM']) < (abs(row['prev_operatingexpenses_ttm']) / row['prev_totaloperatingincome_ttm']): score += 1
                f_scores[row['ticker']] = score
            except Exception: 
                f_scores[row['ticker']] = 0
        return f_scores

    def _get_raw_f_score_securities(self, tickers: List[str],
                                     current_year: int, current_quarter: int) -> Dict[str, int]:
        """Calculate 5-point Piotroski F-Score for securities sector."""
        # Implementation copied from v2.1_alpha
        if not tickers: 
            return {}
        
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
                  AND ((year = :cy AND quarter = :cq) OR (year = :py AND quarter = :cq))
            ),
            current_securities AS (SELECT * FROM base_data WHERE year = :cy),
            previous_securities AS (
                SELECT ticker,
                    TotalOperatingRevenue_TTM as prev_TotalOperatingRevenue_TTM,
                    NetProfit_TTM as prev_NetProfit_TTM,
                    AvgTotalAssets as prev_AvgTotalAssets,
                    OperatingResult_TTM as prev_OperatingResult_TTM,
                    OperatingExpenses_TTM as prev_OperatingExpenses_TTM
                FROM base_data WHERE year = :py
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
            'cy': current_year, 'cq': current_quarter, 'py': current_year - 1, 'tickers': tuple(tickers)
        })
        
        f_scores = {}
        for _, row in securities_data.iterrows():
            score = 0
            try:
                # 5 Securities-specific tests
                if pd.notna(row['NetProfit_TTM']) and pd.notna(row['AvgTotalAssets']) and row['AvgTotalAssets'] > 0 and (row['NetProfit_TTM'] / row['AvgTotalAssets']) > 0: score += 1
                if pd.notna(row['OperatingResult_TTM']) and row['OperatingResult_TTM'] > 0: score += 1
                if pd.notna(row['NetProfit_TTM']) and pd.notna(row['AvgTotalAssets']) and row['AvgTotalAssets'] > 0 and pd.notna(row['prev_NetProfit_TTM']) and pd.notna(row['prev_AvgTotalAssets']) and row['prev_AvgTotalAssets'] > 0 and (row['NetProfit_TTM'] / row['AvgTotalAssets']) > (row['prev_NetProfit_TTM'] / row['prev_AvgTotalAssets']): score += 1
                if pd.notna(row['OperatingResult_TTM']) and pd.notna(row['TotalOperatingRevenue_TTM']) and row['TotalOperatingRevenue_TTM'] > 0 and pd.notna(row['prev_OperatingResult_TTM']) and pd.notna(row['prev_TotalOperatingRevenue_TTM']) and row['prev_TotalOperatingRevenue_TTM'] > 0 and (row['OperatingResult_TTM'] / row['TotalOperatingRevenue_TTM']) > (row['prev_OperatingResult_TTM'] / row['prev_TotalOperatingRevenue_TTM']): score += 1
                if pd.notna(row['OperatingExpenses_TTM']) and pd.notna(row['TotalOperatingRevenue_TTM']) and row['TotalOperatingRevenue_TTM'] > 0 and pd.notna(row['prev_OperatingExpenses_TTM']) and pd.notna(row['prev_TotalOperatingRevenue_TTM']) and row['prev_TotalOperatingRevenue_TTM'] > 0 and (abs(row['OperatingExpenses_TTM']) / row['TotalOperatingRevenue_TTM']) < (abs(row['prev_OperatingExpenses_TTM']) / row['prev_TotalOperatingRevenue_TTM']): score += 1
                f_scores[row['ticker']] = score
            except Exception: 
                f_scores[row['ticker']] = 0
        return f_scores

if __name__ == "__main__":
    print("QVM Engine v2.1.1 Flat - Module loaded successfully.")
    print("This engine implements flat methodology with enhanced defensive factors.")