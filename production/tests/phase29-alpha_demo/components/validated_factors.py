# Validated Factors Component
# 
# This module contains the validated factors implementation for the QVM Engine v3j.
# It can be imported and used independently for factor calculations.

import pandas as pd
import numpy as np
from sqlalchemy import text
from typing import List, Tuple, Optional

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
            DataFrame with low-volatility scores (inverse relationship)
        """
        try:
            # Pivot data for vectorized calculation
            price_pivot = price_data.pivot(index='date', columns='ticker', values='close')
            
            # Calculate rolling volatility
            volatility = price_pivot.rolling(lookback_days).std() * np.sqrt(252)
            
            # Apply inverse relationship (lower volatility = higher score)
            low_vol_score = 1 / volatility
            
            # Stack back to long format
            low_vol_stacked = low_vol_score.stack().reset_index()
            low_vol_stacked.columns = ['date', 'ticker', 'low_vol_score']
            
            # Remove infinite values and outliers
            low_vol_stacked = low_vol_stacked.replace([np.inf, -np.inf], np.nan)
            low_vol_stacked = low_vol_stacked.dropna()
            
            # Winsorize outliers (top and bottom 1%)
            q_low = low_vol_stacked['low_vol_score'].quantile(0.01)
            q_high = low_vol_stacked['low_vol_score'].quantile(0.99)
            low_vol_stacked['low_vol_score'] = low_vol_stacked['low_vol_score'].clip(q_low, q_high)
            
            print(f"   ✅ Low-Volatility factor calculated: {len(low_vol_stacked):,} observations")
            return low_vol_stacked
            
        except Exception as e:
            print(f"   ❌ Error calculating Low-Volatility factor: {e}")
            return pd.DataFrame()
    
    def calculate_piotroski_fscore(self, tickers: List[str], analysis_date: pd.Timestamp) -> pd.DataFrame:
        """
        Calculate Piotroski F-Score with sector-specific implementations.
        
        Args:
            tickers: List of tickers to analyze
            analysis_date: Date for analysis
        
        Returns:
            DataFrame with F-Scores by sector
        """
        try:
            # Get sector information
            sector_query = text("""
                SELECT ticker, sector
                FROM master_info
                WHERE ticker IN :tickers
            """)
            
            ticker_list = tuple(tickers)
            sector_df = pd.read_sql(sector_query, self.engine, params={'tickers': ticker_list})
            
            if sector_df.empty:
                print("   ⚠️  No sector data found")
                return pd.DataFrame()
            
            # Group by sector and calculate F-Scores
            fscore_results = []
            
            for sector in sector_df['sector'].unique():
                sector_tickers = sector_df[sector_df['sector'] == sector]['ticker'].tolist()
                
                if sector == 'Banking':
                    sector_fscores = self._calculate_banking_fscore(sector_tickers, analysis_date)
                elif sector == 'Securities':
                    sector_fscores = self._calculate_securities_fscore(sector_tickers, analysis_date)
                else:
                    sector_fscores = self._calculate_nonfin_fscore(sector_tickers, analysis_date)
                
                if not sector_fscores.empty:
                    sector_fscores['sector'] = sector
                    fscore_results.append(sector_fscores)
            
            if fscore_results:
                combined_fscores = pd.concat(fscore_results, ignore_index=True)
                print(f"   ✅ Piotroski F-Score calculated: {len(combined_fscores):,} observations across {len(combined_fscores['sector'].unique())} sectors")
                return combined_fscores
            else:
                print("   ⚠️  No F-Score data calculated")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"   ❌ Error calculating Piotroski F-Score: {e}")
            return pd.DataFrame()
    
    def _calculate_nonfin_fscore(self, tickers: List[str], analysis_date: pd.Timestamp) -> pd.DataFrame:
        """Calculate F-Score for non-financial companies (9 tests)."""
        try:
            query = text("""
                SELECT 
                    ticker,
                    year,
                    quarter,
                    NetProfit_TTM,
                    NetCFO_TTM,
                    AvgTotalAssets,
                    TotalDebt,
                    CurrentAssets,
                    CurrentLiabilities,
                    TotalEquity,
                    Revenue_TTM,
                    GrossProfit_TTM,
                    SharesOutstanding
                FROM intermediary_calculations_enhanced
                WHERE ticker IN :tickers
                AND year >= YEAR(:analysis_date) - 2
                ORDER BY ticker, year, quarter
            """)
            
            data = pd.read_sql(query, self.engine, 
                             params={'tickers': tuple(tickers), 'analysis_date': analysis_date})
            
            if data.empty:
                return pd.DataFrame()
            
            return self._calculate_fscore_tests(data, 'nonfin')
            
        except Exception as e:
            print(f"   ❌ Error calculating non-financial F-Score: {e}")
            return pd.DataFrame()
    
    def _calculate_banking_fscore(self, tickers: List[str], analysis_date: pd.Timestamp) -> pd.DataFrame:
        """Calculate F-Score for banking companies (9 tests)."""
        try:
            query = text("""
                SELECT 
                    ticker,
                    year,
                    quarter,
                    NetProfit_TTM,
                    AvgTotalAssets,
                    NetInterestIncome_TTM,
                    AvgInterestEarningAssets,
                    OperatingExpenses_TTM,
                    Revenue_TTM,
                    TotalEquity,
                    NonPerformingLoans,
                    TotalLoans,
                    SharesOutstanding
                FROM intermediary_calculations_banking
                WHERE ticker IN :tickers
                AND year >= YEAR(:analysis_date) - 2
                ORDER BY ticker, year, quarter
            """)
            
            data = pd.read_sql(query, self.engine, 
                             params={'tickers': tuple(tickers), 'analysis_date': analysis_date})
            
            if data.empty:
                return pd.DataFrame()
            
            return self._calculate_fscore_tests(data, 'banking')
            
        except Exception as e:
            print(f"   ❌ Error calculating banking F-Score: {e}")
            return pd.DataFrame()
    
    def _calculate_securities_fscore(self, tickers: List[str], analysis_date: pd.Timestamp) -> pd.DataFrame:
        """Calculate F-Score for securities companies (9 tests)."""
        try:
            query = text("""
                SELECT 
                    ticker,
                    year,
                    quarter,
                    NetTradingIncome_TTM,
                    BrokerageRevenue_TTM,
                    OperatingExpenses_TTM,
                    Revenue_TTM,
                    TotalEquity,
                    AvgTotalAssets,
                    SharesOutstanding
                FROM intermediary_calculations_securities
                WHERE ticker IN :tickers
                AND year >= YEAR(:analysis_date) - 2
                ORDER BY ticker, year, quarter
            """)
            
            data = pd.read_sql(query, self.engine, 
                             params={'tickers': tuple(tickers), 'analysis_date': analysis_date})
            
            if data.empty:
                return pd.DataFrame()
            
            return self._calculate_fscore_tests(data, 'securities')
            
        except Exception as e:
            print(f"   ❌ Error calculating securities F-Score: {e}")
            return pd.DataFrame()
    
    def _calculate_fscore_tests(self, data: pd.DataFrame, sector_type: str) -> pd.DataFrame:
        """Calculate F-Score tests based on sector type."""
        results = []
        
        for ticker in data['ticker'].unique():
            ticker_data = data[data['ticker'] == ticker].sort_values(['year', 'quarter'])
            
            if len(ticker_data) < 2:  # Need at least 2 periods for changes
                continue
            
            # Get current and previous period
            current = ticker_data.iloc[-1]
            previous = ticker_data.iloc[-2]
            
            if sector_type == 'nonfin':
                tests = self._calculate_nonfin_tests(current, previous)
            elif sector_type == 'banking':
                tests = self._calculate_banking_tests(current, previous)
            elif sector_type == 'securities':
                tests = self._calculate_securities_tests(current, previous)
            else:
                continue
            
            # Calculate total F-Score
            fscore = sum(tests.values())
            
            result = {'ticker': ticker, 'fscore': fscore}
            result.update(tests)
            results.append(result)
        
        return pd.DataFrame(results)
    
    def _calculate_nonfin_tests(self, current: pd.Series, previous: pd.Series) -> dict:
        """Calculate 9 tests for non-financial companies."""
        # Test 1: ROA > 0
        roa_current = current['NetProfit_TTM'] / current['AvgTotalAssets'] if current['AvgTotalAssets'] > 0 else 0
        test1 = 1 if roa_current > 0 else 0
        
        # Test 2: CFO > 0
        test2 = 1 if current['NetCFO_TTM'] > 0 else 0
        
        # Test 3: ΔROA > 0
        roa_previous = previous['NetProfit_TTM'] / previous['AvgTotalAssets'] if previous['AvgTotalAssets'] > 0 else 0
        test3 = 1 if roa_current > roa_previous else 0
        
        # Test 4: Accruals < CFO
        test4 = 1 if current['NetProfit_TTM'] < current['NetCFO_TTM'] else 0
        
        # Test 5: ΔLeverage < 0
        leverage_current = current['TotalDebt'] / current['TotalEquity'] if current['TotalEquity'] > 0 else 0
        leverage_previous = previous['TotalDebt'] / previous['TotalEquity'] if previous['TotalEquity'] > 0 else 0
        test5 = 1 if leverage_current < leverage_previous else 0
        
        # Test 6: ΔCurrent Ratio > 0
        cr_current = current['CurrentAssets'] / current['CurrentLiabilities'] if current['CurrentLiabilities'] > 0 else 0
        cr_previous = previous['CurrentAssets'] / previous['CurrentLiabilities'] if previous['CurrentLiabilities'] > 0 else 0
        test6 = 1 if cr_current > cr_previous else 0
        
        # Test 7: No new shares issued
        test7 = 1 if current['SharesOutstanding'] <= previous['SharesOutstanding'] else 0
        
        # Test 8: ΔGross Margin > 0
        gm_current = current['GrossProfit_TTM'] / current['Revenue_TTM'] if current['Revenue_TTM'] > 0 else 0
        gm_previous = previous['GrossProfit_TTM'] / previous['Revenue_TTM'] if previous['Revenue_TTM'] > 0 else 0
        test8 = 1 if gm_current > gm_previous else 0
        
        # Test 9: ΔAsset Turnover > 0
        at_current = current['Revenue_TTM'] / current['AvgTotalAssets'] if current['AvgTotalAssets'] > 0 else 0
        at_previous = previous['Revenue_TTM'] / previous['AvgTotalAssets'] if previous['AvgTotalAssets'] > 0 else 0
        test9 = 1 if at_current > at_previous else 0
        
        return {
            'test1_roa': test1, 'test2_cfo': test2, 'test3_delta_roa': test3,
            'test4_accruals': test4, 'test5_delta_leverage': test5, 'test6_delta_current_ratio': test6,
            'test7_no_new_shares': test7, 'test8_delta_gross_margin': test8, 'test9_delta_asset_turnover': test9
        }
    
    def _calculate_banking_tests(self, current: pd.Series, previous: pd.Series) -> dict:
        """Calculate 9 tests for banking companies."""
        # Test 1: NIM > 0
        nim_current = current['NetInterestIncome_TTM'] / current['AvgInterestEarningAssets'] if current['AvgInterestEarningAssets'] > 0 else 0
        test1 = 1 if nim_current > 0 else 0
        
        # Test 2: ROA > 0
        roa_current = current['NetProfit_TTM'] / current['AvgTotalAssets'] if current['AvgTotalAssets'] > 0 else 0
        test2 = 1 if roa_current > 0 else 0
        
        # Test 3: ΔROA > 0
        roa_previous = previous['NetProfit_TTM'] / previous['AvgTotalAssets'] if previous['AvgTotalAssets'] > 0 else 0
        test3 = 1 if roa_current > roa_previous else 0
        
        # Test 4: ΔNIM > 0
        nim_previous = previous['NetInterestIncome_TTM'] / previous['AvgInterestEarningAssets'] if previous['AvgInterestEarningAssets'] > 0 else 0
        test4 = 1 if nim_current > nim_previous else 0
        
        # Test 5: ΔEfficiency Ratio < 0
        eff_current = current['OperatingExpenses_TTM'] / current['Revenue_TTM'] if current['Revenue_TTM'] > 0 else 0
        eff_previous = previous['OperatingExpenses_TTM'] / previous['Revenue_TTM'] if previous['Revenue_TTM'] > 0 else 0
        test5 = 1 if eff_current < eff_previous else 0
        
        # Test 6: ΔCapital Adequacy > 0
        cap_current = current['TotalEquity'] / current['AvgTotalAssets'] if current['AvgTotalAssets'] > 0 else 0
        cap_previous = previous['TotalEquity'] / previous['AvgTotalAssets'] if previous['AvgTotalAssets'] > 0 else 0
        test6 = 1 if cap_current > cap_previous else 0
        
        # Test 7: No new shares issued
        test7 = 1 if current['SharesOutstanding'] <= previous['SharesOutstanding'] else 0
        
        # Test 8: ΔRevenue Growth > 0
        test8 = 1 if current['Revenue_TTM'] > previous['Revenue_TTM'] else 0
        
        # Test 9: ΔAsset Quality > 0
        npl_current = current['NonPerformingLoans'] / current['TotalLoans'] if current['TotalLoans'] > 0 else 0
        npl_previous = previous['NonPerformingLoans'] / previous['TotalLoans'] if previous['TotalLoans'] > 0 else 0
        test9 = 1 if npl_current < npl_previous else 0
        
        return {
            'test1_nim': test1, 'test2_roa': test2, 'test3_delta_roa': test3,
            'test4_delta_nim': test4, 'test5_delta_efficiency': test5, 'test6_delta_capital': test6,
            'test7_no_new_shares': test7, 'test8_revenue_growth': test8, 'test9_asset_quality': test9
        }
    
    def _calculate_securities_tests(self, current: pd.Series, previous: pd.Series) -> dict:
        """Calculate 9 tests for securities companies."""
        # Test 1: Trading Income > 0
        test1 = 1 if current['NetTradingIncome_TTM'] > 0 else 0
        
        # Test 2: Brokerage Revenue > 0
        test2 = 1 if current['BrokerageRevenue_TTM'] > 0 else 0
        
        # Test 3: ΔTrading Income > 0
        test3 = 1 if current['NetTradingIncome_TTM'] > previous['NetTradingIncome_TTM'] else 0
        
        # Test 4: ΔBrokerage Revenue > 0
        test4 = 1 if current['BrokerageRevenue_TTM'] > previous['BrokerageRevenue_TTM'] else 0
        
        # Test 5: ΔEfficiency Ratio < 0
        eff_current = current['OperatingExpenses_TTM'] / current['Revenue_TTM'] if current['Revenue_TTM'] > 0 else 0
        eff_previous = previous['OperatingExpenses_TTM'] / previous['Revenue_TTM'] if previous['Revenue_TTM'] > 0 else 0
        test5 = 1 if eff_current < eff_previous else 0
        
        # Test 6: ΔCapital Adequacy > 0
        cap_current = current['TotalEquity'] / current['AvgTotalAssets'] if current['AvgTotalAssets'] > 0 else 0
        cap_previous = previous['TotalEquity'] / previous['AvgTotalAssets'] if previous['AvgTotalAssets'] > 0 else 0
        test6 = 1 if cap_current > cap_previous else 0
        
        # Test 7: No new shares issued
        test7 = 1 if current['SharesOutstanding'] <= previous['SharesOutstanding'] else 0
        
        # Test 8: ΔRevenue Growth > 0
        test8 = 1 if current['Revenue_TTM'] > previous['Revenue_TTM'] else 0
        
        # Test 9: ΔAsset Quality > 0
        roa_current = current['NetTradingIncome_TTM'] / current['AvgTotalAssets'] if current['AvgTotalAssets'] > 0 else 0
        roa_previous = previous['NetTradingIncome_TTM'] / previous['AvgTotalAssets'] if previous['AvgTotalAssets'] > 0 else 0
        test9 = 1 if roa_current > roa_previous else 0
        
        return {
            'test1_trading_income': test1, 'test2_brokerage_revenue': test2, 'test3_delta_trading': test3,
            'test4_delta_brokerage': test4, 'test5_delta_efficiency': test5, 'test6_delta_capital': test6,
            'test7_no_new_shares': test7, 'test8_revenue_growth': test8, 'test9_asset_quality': test9
        }
    
    def calculate_fcf_yield(self, tickers: List[str], analysis_date: pd.Timestamp) -> pd.DataFrame:
        """
        Calculate FCF Yield factor with imputation handling.
        
        Args:
            tickers: List of tickers to analyze
            analysis_date: Date for analysis
        
        Returns:
            DataFrame with FCF Yield scores
        """
        try:
            # Get fundamental data for FCF calculation
            fundamental_query = text("""
                WITH fundamental_data AS (
                    SELECT 
                        fv.ticker,
                        fv.year,
                        fv.quarter,
                        fv.item_id,
                        fv.statement_type,
                        SUM(fv.value / 1e9) as value_bn
                    FROM fundamental_values fv
                    WHERE fv.ticker IN :tickers
                    AND fv.item_id IN (1, 2, 3)  -- NetProfit, TotalAssets, CapEx
                    AND fv.year >= YEAR(:analysis_date) - 2
                    GROUP BY fv.ticker, fv.year, fv.quarter, fv.item_id, fv.statement_type
                ),
                netprofit_ttm AS (
                    SELECT ticker, year, quarter, value_bn as netprofit_ttm
                    FROM fundamental_data
                    WHERE item_id = 1 AND statement_type = 'PL'
                ),
                totalassets_ttm AS (
                    SELECT ticker, year, quarter, value_bn as totalassets_ttm
                    FROM fundamental_data
                    WHERE item_id = 2 AND statement_type = 'BS'
                ),
                capex_ttm AS (
                    SELECT ticker, year, quarter, value_bn as capex_ttm
                    FROM fundamental_data
                    WHERE item_id = 3 AND statement_type = 'CF'
                )
                SELECT 
                    np.ticker,
                    np.year,
                    np.quarter,
                    np.netprofit_ttm,
                    ta.totalassets_ttm,
                    cx.capex_ttm
                FROM netprofit_ttm np
                LEFT JOIN totalassets_ttm ta ON np.ticker = ta.ticker AND np.year = ta.year AND np.quarter = ta.quarter
                LEFT JOIN capex_ttm cx ON np.ticker = cx.ticker AND np.year = cx.year AND np.quarter = cx.quarter
                WHERE np.netprofit_ttm > 0
                AND ta.totalassets_ttm > 0
            """)
            
            fundamental_data = pd.read_sql(fundamental_query, self.engine,
                                         params={'tickers': tuple(tickers), 'analysis_date': analysis_date})
            
            if fundamental_data.empty:
                print("   ⚠️  No fundamental data found for FCF calculation")
                return pd.DataFrame()
            
            # Get market cap data
            market_cap_query = text("""
                SELECT ticker, market_cap
                FROM vcsc_daily_data_complete
                WHERE ticker IN :tickers
                AND trading_date = :analysis_date
            """)
            
            market_cap_data = pd.read_sql(market_cap_query, self.engine,
                                        params={'tickers': tuple(tickers), 'analysis_date': analysis_date})
            
            if market_cap_data.empty:
                print("   ⚠️  No market cap data found")
                return pd.DataFrame()
            
            # Calculate FCF and FCF Yield
            fundamental_data = fundamental_data.merge(market_cap_data, on='ticker', how='inner')
            
            # Impute missing CapEx (conservative estimate: -5% of NetCFO)
            imputation_rate = 0.0
            if 'capex_ttm' in fundamental_data.columns:
                missing_capex = fundamental_data['capex_ttm'].isna().sum()
                total_obs = len(fundamental_data)
                imputation_rate = missing_capex / total_obs if total_obs > 0 else 0
                
                # Impute with conservative estimate
                fundamental_data['capex_ttm'] = fundamental_data['capex_ttm'].fillna(
                    -0.05 * fundamental_data['netprofit_ttm']
                )
            
            # Calculate FCF (simplified: NetProfit - CapEx)
            fundamental_data['fcf'] = fundamental_data['netprofit_ttm'] - fundamental_data['capex_ttm']
            
            # Calculate FCF Yield
            fundamental_data['fcf_yield'] = fundamental_data['fcf'] / fundamental_data['market_cap']
            
            # Clean and filter
            fcf_results = fundamental_data[['ticker', 'fcf', 'fcf_yield']].copy()
            fcf_results = fcf_results.dropna()
            fcf_results = fcf_results[fcf_results['fcf_yield'] > 0]  # Positive FCF Yield only
            
            # Winsorize outliers
            q_low = fcf_results['fcf_yield'].quantile(0.01)
            q_high = fcf_results['fcf_yield'].quantile(0.99)
            fcf_results['fcf_yield'] = fcf_results['fcf_yield'].clip(q_low, q_high)
            
            print(f"   ✅ FCF Yield calculated: {len(fcf_results):,} observations (imputation rate: {imputation_rate:.2%})")
            return fcf_results
            
        except Exception as e:
            print(f"   ❌ Error calculating FCF Yield: {e}")
            return pd.DataFrame()

def calculate_composite_score(factors_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Calculate composite score using validated factors structure.
    
    Args:
        factors_df: DataFrame with factor values
        config: Configuration dictionary with factor weights
    
    Returns:
        DataFrame with composite scores
    """
    factors_df['composite_score'] = 0.0
    
    # Value Factors (33% total weight)
    value_score = 0.0
    
    # P/E component (contrarian signal - lower is better)
    if 'quality_adjusted_pe' in factors_df.columns:
        pe_weight = config['factors']['value_factors']['pe_weight']
        factors_df['pe_normalized'] = (factors_df['quality_adjusted_pe'] - factors_df['quality_adjusted_pe'].mean()) / factors_df['quality_adjusted_pe'].std()
        value_score += (-factors_df['pe_normalized']) * pe_weight  # Negative for contrarian
    
    # FCF Yield component (positive signal - higher is better)
    if 'fcf_yield' in factors_df.columns:
        fcf_weight = config['factors']['value_factors']['fcf_yield_weight']
        factors_df['fcf_normalized'] = (factors_df['fcf_yield'] - factors_df['fcf_yield'].mean()) / factors_df['fcf_yield'].std()
        value_score += factors_df['fcf_normalized'] * fcf_weight
    
    # Quality Factors (33% total weight)
    quality_score = 0.0
    
    # ROAA component (positive signal - higher is better)
    if 'roaa' in factors_df.columns:
        roaa_weight = config['factors']['quality_factors']['roaa_weight']
        factors_df['roaa_normalized'] = (factors_df['roaa'] - factors_df['roaa'].mean()) / factors_df['roaa'].std()
        quality_score += factors_df['roaa_normalized'] * roaa_weight
    
    # Piotroski F-Score component (positive signal - higher is better)
    if 'fscore' in factors_df.columns:
        fscore_weight = config['factors']['quality_factors']['fscore_weight']
        factors_df['fscore_normalized'] = (factors_df['fscore'] - factors_df['fscore'].mean()) / factors_df['fscore'].std()
        quality_score += factors_df['fscore_normalized'] * fscore_weight
    
    # Momentum Factors (34% total weight)
    momentum_score = 0.0
    
    # Existing momentum component (mixed signals)
    if 'momentum_score' in factors_df.columns:
        momentum_weight = config['factors']['momentum_factors']['momentum_weight']
        factors_df['momentum_normalized'] = (factors_df['momentum_score'] - factors_df['momentum_score'].mean()) / factors_df['momentum_score'].std()
        momentum_score += factors_df['momentum_normalized'] * momentum_weight
    
    # Low-Volatility component (defensive - inverse volatility)
    if 'low_vol_score' in factors_df.columns:
        low_vol_weight = config['factors']['momentum_factors']['low_vol_weight']
        factors_df['low_vol_normalized'] = (factors_df['low_vol_score'] - factors_df['low_vol_score'].mean()) / factors_df['low_vol_score'].std()
        momentum_score += factors_df['low_vol_normalized'] * low_vol_weight
    
    # Combine all factor categories
    factors_df['composite_score'] = (
        value_score * config['factors']['value_weight'] +
        quality_score * config['factors']['quality_weight'] +
        momentum_score * config['factors']['momentum_weight']
    )
    
    return factors_df 