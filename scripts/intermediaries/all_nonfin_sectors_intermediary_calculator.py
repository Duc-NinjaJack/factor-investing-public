#!/usr/bin/env python3
"""
All Non-Financial Sectors Intermediary Calculator - Enhanced DPO Version
========================================================================
Phase 2: Calculate and store intermediary values for ALL non-financial sectors
ENHANCED with proper DPO calculation using reconstructed purchases

Key Enhancement:
- Implements Enhanced DPO calculation as specified in methodology
- Uses reconstructed purchases: Purchases_TTM = COGS_TTM + ΔInventory_YoY
- Eliminates inventory distortions in working capital efficiency metrics

This script:
1. Processes all 21 non-financial sectors (667 tickers)
2. Calculates TTM values, 5-point averages, and working capital metrics
3. Implements Enhanced DPO using AQR methodology
4. Stores results in intermediary_calculations_enhanced table
5. Provides detailed progress tracking and error reporting

Author: Duc Nguyen (Aureus Sigma Capital)
Enhanced: July 6, 2025 - Enhanced DPO Implementation
"""

import sys
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
import logging
import pymysql
import yaml
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

# Setup project path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Enhanced logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(project_root / 'logs' / 'intermediary_calculations.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

class AllNonFinSectorsIntermediaryPopulator:
    """Calculates and populates intermediary values for all non-financial sectors with Enhanced DPO"""
    
    def __init__(self):
        self.db_connection = self._get_db_connection()
        self.processed_count = 0
        self.error_count = 0
        self.total_periods = 0
        self.error_details = []
        self.enhanced_dpo_count = 0
        self.standard_dpo_fallback_count = 0
        
        # Create logs directory if it doesn't exist
        (project_root / 'logs').mkdir(exist_ok=True)
    
    def __del__(self):
        if hasattr(self, 'db_connection') and self.db_connection:
            self.db_connection.close()
    
    def _get_db_connection(self):
        try:
            config_path = project_root / 'config' / 'database.yml'
            with open(config_path, 'r') as f:
                db_config = yaml.safe_load(f)['production']
            
            connection = pymysql.connect(
                host=db_config['host'],
                user=db_config['username'],
                password=db_config['password'],
                database=db_config['schema_name'],
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor
            )
            return connection
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def get_nonfin_sectors_and_tickers(self) -> Dict[str, List[str]]:
        """Get all non-financial sectors and their tickers with enhanced filtering"""
        query = """
        SELECT sector, ticker 
        FROM master_info 
        WHERE sector NOT IN ('Banks', 'Securities', 'Insurance', 'Other Financial')
        AND ticker COLLATE utf8mb4_unicode_ci NOT IN (
            SELECT DISTINCT ticker COLLATE utf8mb4_unicode_ci FROM intermediary_calculations_banking
            UNION
            SELECT DISTINCT ticker COLLATE utf8mb4_unicode_ci FROM intermediary_calculations_securities
        )
        ORDER BY sector, ticker
        """
        
        try:
            with self.db_connection.cursor() as cursor:
                cursor.execute(query)
                results = cursor.fetchall()
                
            # Group by sector
            sectors_tickers = {}
            for row in results:
                sector = row['sector']
                ticker = row['ticker']
                if sector not in sectors_tickers:
                    sectors_tickers[sector] = []
                sectors_tickers[sector].append(ticker)
            
            total_tickers = sum(len(tickers) for tickers in sectors_tickers.values())
            logger.info(f"Found {len(sectors_tickers)} non-financial sectors with {total_tickers} total tickers")
            
            return sectors_tickers
        except Exception as e:
            logger.error(f"Error fetching sectors and tickers: {e}")
            return {}
    
    def get_fundamental_data(self, ticker: str) -> pd.DataFrame:
        """Get fundamental data for calculations - ENHANCED to include Inventory"""
        query = """
        SELECT 
            ticker, year, quarter,
            -- Income Statement (Flow items for TTM)
            NetRevenue, COGS, GrossProfit, SellingExpenses, AdminExpenses,
            FinancialIncome, FinancialExpenses, InterestExpenses,
            ProfitBeforeTax, CurrentIncomeTax, TotalIncomeTax,
            NetProfit, NetProfitAfterMI, EBIT,
            -- Balance Sheet (Stock items for 5-point averaging) - ENHANCED with Inventory
            TotalAssets, CurrentAssets, CashAndCashEquivalents,
            AccountsReceivable, Inventory, FixedAssets,
            TotalLiabilities, CurrentLiabilities, AccountsPayable,
            ShortTermDebt, LongTermDebt, TotalEquity,
            CharterCapital, RetainedEarnings,
            -- Cash Flow (Flow items for TTM) - FIXED MAPPINGS
            NetCFO, NetCFI, NetCFF, DepreciationAmortization, CapEx,
            DividendsPaid, ShareIssuanceProceeds, ShareRepurchase
        FROM v_comprehensive_fundamental_items
        WHERE ticker = %s 
        AND year >= 2010
        ORDER BY year, quarter
        """
        
        try:
            with self.db_connection.cursor() as cursor:
                cursor.execute(query, [ticker])
                results = cursor.fetchall()
                
                if not results:
                    return pd.DataFrame()
                
                result_df = pd.DataFrame(results)
                return result_df
                
        except Exception as e:
            logger.error(f"Error executing query for {ticker}: {e}")
            return pd.DataFrame()
    
    def calculate_ttm_values(self, data: pd.DataFrame, year: int, quarter: int) -> Dict[str, float]:
        """Calculate TTM intermediaries"""
        ttm_quarters = self._get_ttm_quarters(year, quarter)
        data['period_key'] = data['year'].astype(str) + 'Q' + data['quarter'].astype(str)
        ttm_data = data[data['period_key'].isin([f"{y}Q{q}" for y, q in ttm_quarters])]
        
        if len(ttm_data) < 4:
            return {}
        
        results = {}
        
        # Map from fundamental view columns to intermediary table columns
        ttm_mappings = {
            'NetRevenue': 'Revenue_TTM',
            'COGS': 'COGS_TTM',
            'GrossProfit': 'GrossProfit_TTM',
            'SellingExpenses': 'SellingExpenses_TTM',
            'AdminExpenses': 'AdminExpenses_TTM',
            'FinancialIncome': 'FinancialIncome_TTM',
            'FinancialExpenses': 'FinancialExpenses_TTM',
            'InterestExpenses': 'InterestExpense_TTM',
            'ProfitBeforeTax': 'ProfitBeforeTax_TTM',
            'CurrentIncomeTax': 'CurrentTax_TTM',
            'TotalIncomeTax': 'TotalTax_TTM',
            'NetProfit': 'NetProfit_TTM',
            'NetProfitAfterMI': 'NetProfitAfterMI_TTM',
            'NetCFO': 'NetCFO_TTM',
            'NetCFI': 'NetCFI_TTM',
            'NetCFF': 'NetCFF_TTM',
            'DepreciationAmortization': 'DepreciationAmortization_TTM',
            'CapEx': 'CapEx_TTM',
            'DividendsPaid': 'DividendsPaid_TTM',
            'ShareIssuanceProceeds': 'ShareIssuance_TTM',
            'ShareRepurchase': 'ShareRepurchase_TTM'
        }
        
        for source_col, target_col in ttm_mappings.items():
            if source_col in ttm_data.columns:
                values = pd.to_numeric(ttm_data[source_col], errors='coerce')
                results[target_col] = values.sum() if not values.isna().all() else None
        
        # Operating Expenses
        if 'SellingExpenses_TTM' in results and 'AdminExpenses_TTM' in results:
            if results['SellingExpenses_TTM'] is not None and results['AdminExpenses_TTM'] is not None:
                results['OperatingExpenses_TTM'] = results['SellingExpenses_TTM'] + results['AdminExpenses_TTM']
        
        # True EBIT (calculated, not database field)
        if all(col in results for col in ['GrossProfit_TTM', 'SellingExpenses_TTM', 'AdminExpenses_TTM']):
            if all(results[col] is not None for col in ['GrossProfit_TTM', 'SellingExpenses_TTM', 'AdminExpenses_TTM']):
                results['EBIT_TTM'] = (
                    results['GrossProfit_TTM'] - 
                    results['SellingExpenses_TTM'] - 
                    results['AdminExpenses_TTM']
                )
        
        # EBITDA
        if 'EBIT_TTM' in results and 'DepreciationAmortization_TTM' in results:
            if results['EBIT_TTM'] is not None and results['DepreciationAmortization_TTM'] is not None:
                results['EBITDA_TTM'] = results['EBIT_TTM'] + results['DepreciationAmortization_TTM']
        
        # FCF
        if 'NetCFO_TTM' in results and 'CapEx_TTM' in results:
            if results['NetCFO_TTM'] is not None and results['CapEx_TTM'] is not None:
                results['FCF_TTM'] = results['NetCFO_TTM'] + results['CapEx_TTM']
        
        return results
    
    def calculate_5point_averages(self, data: pd.DataFrame, year: int, quarter: int) -> Dict[str, float]:
        """Calculate 5-point balance sheet averages"""
        avg_quarters = self._get_5point_quarters(year, quarter)
        data['period_key'] = data['year'].astype(str) + 'Q' + data['quarter'].astype(str)
        avg_data = data[data['period_key'].isin([f"{y}Q{q}" for y, q in avg_quarters])]
        
        if len(avg_data) < 5:
            return {}
        
        results = {}
        
        # Map from fundamental view columns to intermediary table columns
        avg_mappings = {
            'TotalAssets': 'AvgTotalAssets',
            'TotalEquity': 'AvgTotalEquity',
            'TotalLiabilities': 'AvgTotalLiabilities',
            'CurrentAssets': 'AvgCurrentAssets',
            'CurrentLiabilities': 'AvgCurrentLiabilities',
            'CashAndCashEquivalents': 'AvgCash',
            'Inventory': 'AvgInventory',
            'AccountsReceivable': 'AvgReceivables',
            'AccountsPayable': 'AvgPayables',
            'FixedAssets': 'AvgFixedAssets',
            'ShortTermDebt': 'AvgShortTermDebt',
            'LongTermDebt': 'AvgLongTermDebt',
            'RetainedEarnings': 'AvgRetainedEarnings'
        }
        
        for source_col, target_col in avg_mappings.items():
            if source_col in avg_data.columns:
                values = pd.to_numeric(avg_data[source_col], errors='coerce')
                if not values.isna().all():
                    results[target_col] = values.mean()
        
        # Derived values
        if 'AvgShortTermDebt' in results and 'AvgLongTermDebt' in results:
            results['AvgTotalDebt'] = results['AvgShortTermDebt'] + results['AvgLongTermDebt']
        
        if 'AvgTotalDebt' in results and 'AvgCash' in results:
            results['AvgNetDebt'] = results['AvgTotalDebt'] - results['AvgCash']
        
        if 'AvgCurrentAssets' in results and 'AvgCurrentLiabilities' in results:
            results['AvgWorkingCapital'] = results['AvgCurrentAssets'] - results['AvgCurrentLiabilities']
        
        # Invested Capital
        if 'AvgTotalEquity' in results and 'AvgTotalDebt' in results and 'AvgCash' in results:
            results['AvgInvestedCapital'] = results['AvgTotalEquity'] + results['AvgTotalDebt'] - results['AvgCash']
        
        return results
    
    def calculate_working_capital_metrics(self, ttm_values: Dict, avg_values: Dict, data: pd.DataFrame, year: int, quarter: int) -> Dict[str, float]:
        """Calculate working capital metrics with ENHANCED DPO calculation"""
        results = {}
        
        # DSO (Days Sales Outstanding)
        if 'AvgReceivables' in avg_values and 'Revenue_TTM' in ttm_values:
            revenue_ttm = ttm_values.get('Revenue_TTM')
            if revenue_ttm is not None and revenue_ttm != 0:
                results['DSO'] = (avg_values['AvgReceivables'] * 365) / revenue_ttm
        
        # DIO (Days Inventory Outstanding)
        if 'AvgInventory' in avg_values and 'COGS_TTM' in ttm_values:
            cogs_ttm = ttm_values.get('COGS_TTM')
            if cogs_ttm is not None and cogs_ttm != 0:
                results['DIO'] = (avg_values['AvgInventory'] * 365) / abs(cogs_ttm)
        
        # DPO (Days Payables Outstanding) - ENHANCED VERSION
        if 'AvgPayables' in avg_values and 'COGS_TTM' in ttm_values:
            cogs_ttm = ttm_values.get('COGS_TTM')
            
            if cogs_ttm is not None and cogs_ttm != 0:
                # Attempt Enhanced DPO calculation
                enhanced_dpo = self._calculate_enhanced_dpo(
                    data, year, quarter, avg_values['AvgPayables'], cogs_ttm
                )
                
                if enhanced_dpo is not None:
                    results['DPO'] = enhanced_dpo
                    self.enhanced_dpo_count += 1
                else:
                    # Fallback to standard DPO
                    results['DPO'] = (avg_values['AvgPayables'] * 365) / abs(cogs_ttm)
                    self.standard_dpo_fallback_count += 1
                    logger.debug(f"Enhanced DPO failed for period {year}Q{quarter}, using standard DPO")
        
        # CCC (Cash Conversion Cycle)
        if all(metric in results for metric in ['DSO', 'DIO', 'DPO']):
            results['CCC'] = results['DSO'] + results['DIO'] - results['DPO']
        
        return results
    
    def _calculate_enhanced_dpo(self, data: pd.DataFrame, year: int, quarter: int, avg_payables: float, cogs_ttm: float) -> Optional[float]:
        """
        Calculate Enhanced DPO using reconstructed purchases
        
        Formula: DPO_Enhanced = (AvgPayables × 365) / Purchases_TTM
        Where: Purchases_TTM = COGS_TTM + ΔInventory_YoY
        """
        try:
            # Get current period inventory
            data['period_key'] = data['year'].astype(str) + 'Q' + data['quarter'].astype(str)
            current_period_data = data[data['period_key'] == f"{year}Q{quarter}"]
            
            if current_period_data.empty or 'Inventory' not in current_period_data.columns:
                return None
            
            current_inventory = pd.to_numeric(current_period_data['Inventory'].iloc[0], errors='coerce')
            
            if pd.isna(current_inventory):
                return None
            
            # Get inventory from 4 quarters ago (YoY)
            yoy_quarters = self._get_ttm_quarters(year, quarter)
            if len(yoy_quarters) < 4:
                return None
                
            # Get the earliest quarter (4 quarters ago)
            yoy_year, yoy_quarter = yoy_quarters[0]
            yoy_period_data = data[data['period_key'] == f"{yoy_year}Q{yoy_quarter}"]
            
            if yoy_period_data.empty or 'Inventory' not in yoy_period_data.columns:
                return None
            
            yoy_inventory = pd.to_numeric(yoy_period_data['Inventory'].iloc[0], errors='coerce')
            
            if pd.isna(yoy_inventory):
                return None
            
            # Calculate inventory change (ΔInventory_YoY)
            delta_inventory = current_inventory - yoy_inventory
            
            # Calculate reconstructed purchases
            purchases_ttm = cogs_ttm + delta_inventory
            
            # Avoid division by zero or negative purchases
            if purchases_ttm == 0:
                return None
            
            # Calculate Enhanced DPO
            enhanced_dpo = (avg_payables * 365) / abs(purchases_ttm)
            
            # Log successful calculation for debugging
            logger.debug(f"Enhanced DPO calculated: COGS_TTM={cogs_ttm:,.0f}, "
                        f"ΔInv={delta_inventory:,.0f}, Purchases={purchases_ttm:,.0f}, "
                        f"DPO={enhanced_dpo:.1f}")
            
            return enhanced_dpo
            
        except Exception as e:
            logger.debug(f"Enhanced DPO calculation failed: {e}")
            return None
    
    def validate_sector_assignment(self, ticker: str) -> bool:
        """Validate that ticker belongs to non-financial sector"""
        query = "SELECT sector FROM master_info WHERE ticker = %s"
        try:
            with self.db_connection.cursor() as cursor:
                cursor.execute(query, [ticker])
                result = cursor.fetchone()
                if result:
                    sector = result['sector']
                    is_valid = sector not in ['Banks', 'Securities', 'Insurance', 'Other Financial']
                    if not is_valid:
                        logger.warning(f"Sector validation failed: {ticker} is {sector}, expected non-financial sector")
                    return is_valid
                else:
                    logger.warning(f"Ticker {ticker} not found in master_info")
                    return False
        except Exception as e:
            logger.error(f"Error validating sector for {ticker}: {e}")
            return False
    
    def insert_intermediary_values(self, ticker: str, year: int, quarter: int, values: Dict) -> bool:
        """Insert intermediary values into database with sector validation"""
        try:
            # Validate sector assignment before insertion
            if not self.validate_sector_assignment(ticker):
                logger.error(f"Skipping insertion for {ticker} due to sector validation failure")
                return False
            
            # Build column lists and values
            columns = ['ticker', 'year', 'quarter', 'calc_date']
            vals = [ticker, year, quarter, datetime.now().strftime('%Y-%m-%d')]
            
            # Add all calculated values
            for key, value in values.items():
                if value is not None and not pd.isna(value):
                    columns.append(key)
                    vals.append(value)
            
            # Build insert query
            placeholders = ', '.join(['%s'] * len(columns))
            columns_str = ', '.join(columns)
            
            # Insert with ON DUPLICATE KEY UPDATE
            update_parts = [f"{col} = VALUES({col})" for col in columns if col not in ['ticker', 'year', 'quarter']]
            update_str = ', '.join(update_parts)
            
            query = f"""
            INSERT INTO intermediary_calculations_enhanced ({columns_str})
            VALUES ({placeholders})
            ON DUPLICATE KEY UPDATE {update_str}
            """
            
            with self.db_connection.cursor() as cursor:
                cursor.execute(query, vals)
                self.db_connection.commit()
                return True
                
        except Exception as e:
            logger.error(f"Error inserting data for {ticker} {year}Q{quarter}: {e}")
            self.db_connection.rollback()
            return False
    
    def process_ticker(self, ticker: str, sector: str, ticker_progress: tqdm = None) -> int:
        """Process all periods for a single ticker"""
        try:
            data = self.get_fundamental_data(ticker)
            
            if data.empty:
                if ticker_progress:
                    ticker_progress.set_description(f"No data: {ticker}")
                return 0
            
            # Get unique year-quarter combinations
            periods = data[['year', 'quarter']].drop_duplicates().sort_values(['year', 'quarter'])
            count = 0
            
            for _, period in periods.iterrows():
                year, quarter = int(period['year']), int(period['quarter'])
                
                # Calculate all intermediaries
                ttm_values = self.calculate_ttm_values(data, year, quarter)
                avg_values = self.calculate_5point_averages(data, year, quarter)
                wc_values = self.calculate_working_capital_metrics(ttm_values, avg_values, data, year, quarter)
                
                # Add metadata
                metadata = {}
                # Count quarters available for TTM
                ttm_quarters = self._get_ttm_quarters(year, quarter)
                data['period_key'] = data['year'].astype(str) + 'Q' + data['quarter'].astype(str)
                ttm_data = data[data['period_key'].isin([f"{y}Q{q}" for y, q in ttm_quarters])]
                metadata['quarters_available'] = len(ttm_data)
                metadata['has_full_ttm'] = len(ttm_data) == 4
                
                # Count points for averaging
                avg_quarters = self._get_5point_quarters(year, quarter)
                avg_data = data[data['period_key'].isin([f"{y}Q{q}" for y, q in avg_quarters])]
                metadata['avg_points_used'] = len(avg_data)
                metadata['has_full_avg'] = len(avg_data) == 5
                
                # Combine all values
                all_values = {**ttm_values, **avg_values, **wc_values, **metadata}
                
                # Only insert if we have meaningful data
                if len(ttm_values) > 5 or len(avg_values) > 5:  # Must have some TTM or avg data
                    if self.insert_intermediary_values(ticker, year, quarter, all_values):
                        count += 1
            
            if ticker_progress:
                ticker_progress.set_description(f"✅ {ticker}: {count} periods")
            
            return count
            
        except Exception as e:
            error_msg = f"Error processing {ticker} in {sector}: {str(e)}"
            logger.error(error_msg)
            self.error_details.append(error_msg)
            if ticker_progress:
                ticker_progress.set_description(f"❌ {ticker}: ERROR")
            return 0
    
    def _get_ttm_quarters(self, year: int, quarter: int) -> List[Tuple[int, int]]:
        """Get 4 quarters for TTM calculation"""
        quarters = []
        for i in range(4):
            q = quarter - i
            y = year
            if q <= 0:
                q += 4
                y -= 1
            quarters.append((y, q))
        return quarters[::-1]
    
    def _get_5point_quarters(self, year: int, quarter: int) -> List[Tuple[int, int]]:
        """Get 5 quarters for 5-point averaging"""
        quarters = []
        for i in range(5):
            q = quarter - i
            y = year
            if q <= 0:
                q += 4
                y -= 1
            quarters.append((y, q))
        return quarters[::-1]
    
    def run(self, specific_sectors: Optional[List[str]] = None):
        """Run the intermediary calculation process for all non-financial sectors"""
        start_time = datetime.now()
        logger.info("=" * 80)
        logger.info("STARTING ENHANCED ALL NON-FINANCIAL SECTORS INTERMEDIARY CALCULATIONS")
        logger.info(f"Start time: {start_time}")
        logger.info("🔧 Enhanced with AQR-standard DPO calculation using reconstructed purchases")
        logger.info("=" * 80)
        
        # Get all sectors and tickers
        sectors_tickers = self.get_nonfin_sectors_and_tickers()
        
        if specific_sectors:
            sectors_tickers = {k: v for k, v in sectors_tickers.items() if k in specific_sectors}
        
        total_sectors = len(sectors_tickers)
        total_tickers = sum(len(tickers) for tickers in sectors_tickers.values())
        
        logger.info(f"Processing {total_sectors} sectors with {total_tickers} total tickers")
        
        # Sector-level progress bar
        sector_progress = tqdm(
            sectors_tickers.items(), 
            desc="Sectors", 
            unit="sector",
            position=0,
            leave=True
        )
        
        for sector, tickers in sector_progress:
            sector_progress.set_description(f"📂 {sector} ({len(tickers)} tickers)")
            
            # Ticker-level progress bar
            ticker_progress = tqdm(
                tickers,
                desc="Tickers",
                unit="ticker", 
                position=1,
                leave=False
            )
            
            sector_processed = 0
            sector_errors = 0
            sector_periods = 0
            
            for ticker in ticker_progress:
                try:
                    periods = self.process_ticker(ticker, sector, ticker_progress)
                    sector_periods += periods
                    if periods > 0:
                        sector_processed += 1
                    else:
                        sector_errors += 1
                        
                except Exception as e:
                    logger.error(f"Failed to process {ticker} in {sector}: {e}")
                    sector_errors += 1
                    self.error_details.append(f"{sector}.{ticker}: {str(e)}")
            
            # Update totals
            self.processed_count += sector_processed
            self.error_count += sector_errors
            self.total_periods += sector_periods
            
            # Close ticker progress bar
            ticker_progress.close()
            
            logger.info(f"✅ {sector}: {sector_processed}/{len(tickers)} tickers, {sector_periods} periods")
            if sector_errors > 0:
                logger.warning(f"⚠️ {sector}: {sector_errors} errors")
        
        # Close sector progress bar
        sector_progress.close()
        
        # Final summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("=" * 80)
        logger.info("ENHANCED NON-FINANCIAL SECTORS INTERMEDIARY CALCULATION COMPLETE")
        logger.info(f"End time: {end_time}")
        logger.info(f"Duration: {duration}")
        logger.info("=" * 80)
        logger.info(f"✅ Sectors processed: {total_sectors}")
        logger.info(f"✅ Tickers processed: {self.processed_count}")
        logger.info(f"❌ Errors: {self.error_count}")
        logger.info(f"📊 Total periods: {self.total_periods}")
        logger.info("=" * 80)
        logger.info("🔧 ENHANCED DPO CALCULATION SUMMARY:")
        logger.info(f"  ✅ Enhanced DPO calculations: {self.enhanced_dpo_count}")
        logger.info(f"  ⚠️ Standard DPO fallbacks: {self.standard_dpo_fallback_count}")
        total_dpo = self.enhanced_dpo_count + self.standard_dpo_fallback_count
        if total_dpo > 0:
            enhanced_pct = (self.enhanced_dpo_count / total_dpo) * 100
            logger.info(f"  📈 Enhanced DPO success rate: {enhanced_pct:.1f}%")
        logger.info("=" * 80)
        
        if self.error_details:
            logger.warning("=" * 80)
            logger.warning("ERROR DETAILS:")
            for error in self.error_details[:10]:  # Show first 10 errors
                logger.warning(f"  - {error}")
            if len(self.error_details) > 10:
                logger.warning(f"  ... and {len(self.error_details) - 10} more errors")
        
        logger.info("=" * 80)

def main():
    parser = argparse.ArgumentParser(description='Enhanced All Non-Financial Sectors Intermediary Calculator')
    parser.add_argument('--sectors', nargs='+', help='Process specific sectors only')
    parser.add_argument('--test', action='store_true', help='Test mode - process Technology and Real Estate only')
    args = parser.parse_args()
    
    populator = AllNonFinSectorsIntermediaryPopulator()
    
    if args.test:
        logger.info("Running in test mode - processing Technology and Real Estate only")
        populator.run(['Technology', 'Real Estate'])
    else:
        populator.run(args.sectors)

if __name__ == "__main__":
    main()