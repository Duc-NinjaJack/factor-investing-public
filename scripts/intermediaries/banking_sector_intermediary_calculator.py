#!/usr/bin/env python3
"""
Banking Sector Intermediary Calculator - CLEANED VERSION
========================================================
Phase 5: Calculate and store ONLY raw building blocks for banking tickers.
REMOVED: All pre-calculated ratios (ROAE, ROAA, NIM, etc.) to eliminate percentage storage issues.
KEEPS: Raw TTM flows, 5-point averages, and complex aggregations only.

This script:
1. Processes all 21 banking tickers
2. Calculates banking TTM values and 5-point averages ONLY
3. Stores ONLY raw building blocks in intermediary_calculations_banking table
4. ELIMINATES percentage storage issues by removing ratio calculations

Author: Duc Nguyen (Aureus Sigma Capital)
Date: July 16, 2025 - Phase 5 Cleanup
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

# Setup project path - Find project root by looking for config directory
def find_project_root():
    current = Path(__file__).parent
    while current != current.parent:
        if (current / 'config' / 'database.yml').exists():
            return current
        current = current.parent
    raise FileNotFoundError("Could not find project root with config/database.yml")

project_root = find_project_root()
sys.path.append(str(project_root))

# Enhanced logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(project_root / 'logs' / 'banking_intermediary_calculations_cleaned.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

class BankingIntermediaryPopulatorCleaned:
    """Calculates and populates ONLY raw building blocks for banking sector"""
    
    def __init__(self):
        self.db_connection = self._get_db_connection()
        self.processed_count = 0
        self.error_count = 0
        self.total_periods = 0
        self.error_details = []
        
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
    
    def get_banking_tickers(self) -> List[str]:
        """Get all banking sector tickers with validation"""
        query = """
        SELECT ticker 
        FROM master_info 
        WHERE sector = 'Banks'
        AND ticker COLLATE utf8mb4_unicode_ci NOT IN (
            SELECT DISTINCT ticker COLLATE utf8mb4_unicode_ci FROM intermediary_calculations_enhanced
        )
        ORDER BY ticker
        """
        
        try:
            with self.db_connection.cursor() as cursor:
                cursor.execute(query)
                results = cursor.fetchall()
                
            tickers = [row['ticker'] for row in results]
            logger.info(f"Found {len(tickers)} banking tickers: {tickers}")
            return tickers
            
        except Exception as e:
            logger.error(f"Error fetching banking tickers: {e}")
            return []
    
    def get_banking_fundamental_data(self, ticker: str) -> pd.DataFrame:
        """Get banking fundamental data for calculations"""
        query = """
        SELECT 
            ticker, year, quarter,
            -- Core Banking Income Statement
            NetInterestIncome, InterestAndSimilarIncome, InterestAndSimilarExpenses,
            NetFeeCommissionIncome, NetForeignExchangeIncome, NetTradingSecuritiesIncome, 
            NetInvestmentSecuritiesIncome, NetOtherIncome,
            IncomeFromEquityInvestments, OperatingExpenses,
            OperatingProfitBeforeProvisions, CreditLossProvisions,
            ProfitBeforeTax, IncomeTaxExpense, ProfitAfterTax,
            NetProfitAfterMinorityInterest,
            
            -- Core Banking Balance Sheet
            TotalAssets, GrossCustomerLoans, LoanLossProvisions,
            CustomerLoans, TradingSecurities, InvestmentSecurities,
            CashValuablePapersPreciousMetals, DepositsAtCentralBank,
            CustomerDeposits, ShareholdersEquity,
            FixedAssets, OtherAssets, CharterCapital
            
        FROM v_complete_banking_fundamentals
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
    
    def calculate_banking_ttm_values(self, data: pd.DataFrame, year: int, quarter: int) -> Dict[str, float]:
        """Calculate banking TTM intermediaries - RAW VALUES ONLY"""
        ttm_quarters = self._get_ttm_quarters(year, quarter)
        data['period_key'] = data['year'].astype(str) + 'Q' + data['quarter'].astype(str)
        ttm_data = data[data['period_key'].isin([f"{y}Q{q}" for y, q in ttm_quarters])]
        
        if len(ttm_data) < 4:
            return {}
        
        results = {}
        
        # Banking-specific TTM mappings - RAW VALUES ONLY
        banking_ttm_mappings = {
            'NetInterestIncome': 'NII_TTM',
            'InterestAndSimilarIncome': 'InterestIncome_TTM',
            'InterestAndSimilarExpenses': 'InterestExpense_TTM',
            'NetFeeCommissionIncome': 'NetFeeIncome_TTM',
            'NetForeignExchangeIncome': 'ForexIncome_TTM',
            'NetTradingSecuritiesIncome': 'TradingIncome_TTM',
            'NetInvestmentSecuritiesIncome': 'InvestmentIncome_TTM',
            'NetOtherIncome': 'OtherIncome_TTM',
            'IncomeFromEquityInvestments': 'EquityInvestmentIncome_TTM',
            'OperatingExpenses': 'OperatingExpenses_TTM',
            'OperatingProfitBeforeProvisions': 'OperatingProfit_TTM',
            'CreditLossProvisions': 'CreditProvisions_TTM',
            'ProfitBeforeTax': 'ProfitBeforeTax_TTM',
            'IncomeTaxExpense': 'TaxExpense_TTM',
            'ProfitAfterTax': 'NetProfit_TTM',
            'NetProfitAfterMinorityInterest': 'NetProfitAfterMI_TTM'
        }
        
        for source_col, target_col in banking_ttm_mappings.items():
            if source_col in ttm_data.columns:
                values = pd.to_numeric(ttm_data[source_col], errors='coerce')
                results[target_col] = values.sum() if not values.isna().all() else None
        
        return results
    
    def calculate_banking_5point_averages(self, data: pd.DataFrame, year: int, quarter: int) -> Dict[str, float]:
        """Calculate banking 5-point balance sheet averages - RAW VALUES ONLY"""
        avg_quarters = self._get_5point_quarters(year, quarter)
        data['period_key'] = data['year'].astype(str) + 'Q' + data['quarter'].astype(str)
        avg_data = data[data['period_key'].isin([f"{y}Q{q}" for y, q in avg_quarters])]
        
        if len(avg_data) < 5:
            return {}
        
        results = {}
        
        # Banking-specific averaging mappings - RAW VALUES ONLY
        banking_avg_mappings = {
            'TotalAssets': 'AvgTotalAssets',
            'GrossCustomerLoans': 'AvgGrossLoans',
            'LoanLossProvisions': 'AvgLoanLossReserves',
            'CustomerLoans': 'AvgNetLoans',
            'TradingSecurities': 'AvgTradingSecurities',
            'InvestmentSecurities': 'AvgInvestmentSecurities',
            'CashValuablePapersPreciousMetals': 'AvgCash',
            'CustomerDeposits': 'AvgCustomerDeposits',
            'ShareholdersEquity': 'AvgTotalEquity',
            'CharterCapital': 'AvgPaidInCapital'
        }
        
        for source_col, target_col in banking_avg_mappings.items():
            if source_col in avg_data.columns:
                values = pd.to_numeric(avg_data[source_col], errors='coerce')
                if not values.isna().all():
                    results[target_col] = values.mean()
        
        # Calculate derived averages - COMPLEX AGGREGATIONS ONLY
        if 'AvgCash' in results and 'AvgInvestmentSecurities' in results and 'AvgNetLoans' in results:
            # Earning Assets = Cash + Investment Securities + Net Loans (simplified)
            results['AvgEarningAssets'] = (
                results.get('AvgCash', 0) + 
                results.get('AvgInvestmentSecurities', 0) + 
                results.get('AvgNetLoans', 0)
            )
        
        # Total Deposits (Customer + Interbank)
        if 'AvgCustomerDeposits' in results:
            results['AvgTotalDeposits'] = results['AvgCustomerDeposits']  # Simplified for now
        
        # Borrowings (can be enhanced later)
        results['AvgBorrowings'] = 0  # Placeholder - would need specific borrowing data
        
        return results
    
    def calculate_banking_complex_aggregations(self, ttm_values: Dict, avg_values: Dict) -> Dict[str, float]:
        """Calculate ONLY complex aggregations that are difficult to compute dynamically"""
        results = {}
        
        # ✅ KEEP: Total Operating Income (Complex Formula) - Hard to calculate dynamically
        total_operating_income = (
            ttm_values.get('NII_TTM', 0) + 
            ttm_values.get('NetFeeIncome_TTM', 0) + 
            ttm_values.get('ForexIncome_TTM', 0) +
            ttm_values.get('TradingIncome_TTM', 0) + 
            ttm_values.get('InvestmentIncome_TTM', 0) +
            ttm_values.get('OtherIncome_TTM', 0) +
            ttm_values.get('EquityInvestmentIncome_TTM', 0)
        )
        results['TotalOperatingIncome_TTM'] = total_operating_income
        
        # ✅ KEEP: Non-Interest Income (Complex aggregation) - Note: matches existing column name  
        non_interest_income = (
            ttm_values.get('NetFeeIncome_TTM', 0) + 
            ttm_values.get('ForexIncome_TTM', 0) + 
            ttm_values.get('TradingIncome_TTM', 0) + 
            ttm_values.get('InvestmentIncome_TTM', 0) +
            ttm_values.get('OtherIncome_TTM', 0) +
            ttm_values.get('EquityInvestmentIncome_TTM', 0)
        )
        # Store as building block only (no ratio calculation)
        results['NonInterestIncome_TTM_Raw'] = non_interest_income
        
        # ❌ REMOVED: All ratio calculations (ROAE, ROAA, NIM, Cost_Income_Ratio, etc.)
        # These will be calculated dynamically in the backtesting layer
        
        return results
    
    def validate_banking_sector_assignment(self, ticker: str) -> bool:
        """Validate that ticker belongs to banking sector"""
        query = "SELECT sector FROM master_info WHERE ticker = %s"
        try:
            with self.db_connection.cursor() as cursor:
                cursor.execute(query, [ticker])
                result = cursor.fetchone()
                if result:
                    sector = result['sector']
                    is_valid = sector == 'Banks'
                    if not is_valid:
                        logger.warning(f"Banking sector validation failed: {ticker} is {sector}, expected Banks")
                    return is_valid
                else:
                    logger.warning(f"Ticker {ticker} not found in master_info")
                    return False
        except Exception as e:
            logger.error(f"Error validating banking sector for {ticker}: {e}")
            return False
    
    def insert_banking_intermediary_values(self, ticker: str, year: int, quarter: int, values: Dict) -> bool:
        """Insert banking intermediary values into database with sector validation"""
        try:
            # Validate sector assignment before insertion
            if not self.validate_banking_sector_assignment(ticker):
                logger.error(f"Skipping insertion for {ticker} due to banking sector validation failure")
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
            INSERT INTO intermediary_calculations_banking_cleaned ({columns_str})
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
    
    def process_banking_ticker(self, ticker: str, ticker_progress: tqdm = None) -> int:
        """Process all periods for a single banking ticker"""
        try:
            data = self.get_banking_fundamental_data(ticker)
            
            if data.empty:
                if ticker_progress:
                    ticker_progress.set_description(f"No data: {ticker}")
                return 0
            
            # Get unique year-quarter combinations
            periods = data[['year', 'quarter']].drop_duplicates().sort_values(['year', 'quarter'])
            count = 0
            
            for _, period in periods.iterrows():
                year, quarter = int(period['year']), int(period['quarter'])
                
                # Calculate ONLY raw building blocks
                ttm_values = self.calculate_banking_ttm_values(data, year, quarter)
                avg_values = self.calculate_banking_5point_averages(data, year, quarter)
                complex_aggs = self.calculate_banking_complex_aggregations(ttm_values, avg_values)
                
                # Add metadata
                metadata = {}
                # Count quarters available for TTM
                ttm_quarters = self._get_ttm_quarters(year, quarter)
                data['period_key'] = data['year'].astype(str) + 'Q' + data['quarter'].astype(str)
                ttm_data = data[data['period_key'].isin([f"{y}Q{q}" for y, q in ttm_quarters])]
                metadata['quarters_available_ttm'] = len(ttm_data)
                metadata['has_full_ttm'] = len(ttm_data) == 4
                
                # Count points for averaging
                avg_quarters = self._get_5point_quarters(year, quarter)
                avg_data = data[data['period_key'].isin([f"{y}Q{q}" for y, q in avg_quarters])]
                metadata['avg_points_used'] = len(avg_data)
                metadata['has_full_avg'] = len(avg_data) == 5
                
                # Data quality score (simplified)
                quality_score = 0
                if ttm_values: quality_score += 40
                if avg_values: quality_score += 40  
                if complex_aggs: quality_score += 20
                metadata['data_quality_score'] = quality_score
                
                # Combine all values - ONLY RAW BUILDING BLOCKS
                all_values = {**ttm_values, **avg_values, **complex_aggs, **metadata}
                
                # Only insert if we have meaningful data
                if len(ttm_values) > 3 or len(avg_values) > 3:  # Must have some TTM or avg data
                    if self.insert_banking_intermediary_values(ticker, year, quarter, all_values):
                        count += 1
            
            if ticker_progress:
                ticker_progress.set_description(f"✅ {ticker}: {count} periods")
            
            return count
            
        except Exception as e:
            error_msg = f"Error processing {ticker}: {str(e)}"
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
    
    def run(self, specific_tickers: Optional[List[str]] = None):
        """Run the cleaned banking intermediary calculation process"""
        start_time = datetime.now()
        logger.info("=" * 80)
        logger.info("STARTING CLEANED BANKING SECTOR INTERMEDIARY CALCULATIONS")
        logger.info("PHASE 5: Raw building blocks only - NO pre-calculated ratios")
        logger.info(f"Start time: {start_time}")
        logger.info("=" * 80)
        
        # Get banking tickers
        all_tickers = self.get_banking_tickers()
        
        if specific_tickers:
            tickers = [t for t in specific_tickers if t in all_tickers]
        else:
            tickers = all_tickers
        
        total_tickers = len(tickers)
        logger.info(f"Processing {total_tickers} banking tickers: {tickers}")
        
        # Ticker-level progress bar
        ticker_progress = tqdm(
            tickers,
            desc="Banking Tickers",
            unit="ticker", 
            position=0,
            leave=True
        )
        
        total_processed = 0
        total_errors = 0
        total_periods = 0
        
        for ticker in ticker_progress:
            try:
                periods = self.process_banking_ticker(ticker, ticker_progress)
                total_periods += periods
                if periods > 0:
                    total_processed += 1
                else:
                    total_errors += 1
                    
            except Exception as e:
                logger.error(f"Failed to process {ticker}: {e}")
                total_errors += 1
                self.error_details.append(f"{ticker}: {str(e)}")
        
        # Close progress bar
        ticker_progress.close()
        
        # Final summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("=" * 80)
        logger.info("CLEANED BANKING SECTOR INTERMEDIARY CALCULATION COMPLETE")
        logger.info("✅ NO MORE PERCENTAGE STORAGE ISSUES!")
        logger.info(f"End time: {end_time}")
        logger.info(f"Duration: {duration}")
        logger.info("=" * 80)
        logger.info(f"✅ Tickers processed: {total_processed}")
        logger.info(f"❌ Errors: {total_errors}")
        logger.info(f"📊 Total periods: {total_periods}")
        
        if self.error_details:
            logger.warning("=" * 80)
            logger.warning("ERROR DETAILS:")
            for error in self.error_details:
                logger.warning(f"  - {error}")
        
        logger.info("=" * 80)

def main():
    parser = argparse.ArgumentParser(description='Cleaned Banking Sector Intermediary Calculator')
    parser.add_argument('--tickers', nargs='+', help='Process specific tickers only')
    parser.add_argument('--test', action='store_true', help='Test mode - process VCB and TCB only')
    args = parser.parse_args()
    
    populator = BankingIntermediaryPopulatorCleaned()
    
    if args.test:
        logger.info("Running in test mode - processing VCB and TCB only")
        populator.run(['VCB', 'TCB'])
    else:
        populator.run(args.tickers)

if __name__ == "__main__":
    main()