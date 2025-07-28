#!/usr/bin/env python3
"""
Securities Sector Intermediary Calculator - CLEANED VERSION
===========================================================
Phase 5: Calculate and store ONLY raw building blocks for securities tickers.
REMOVED: All pre-calculated ratios (ROAE, ROAA, margins, etc.) to eliminate percentage storage issues.
KEEPS: Raw TTM flows, 5-point averages, and complex aggregations only.

This script:
1. Processes all 26 securities tickers
2. Calculates securities TTM values and 5-point averages ONLY
3. Stores ONLY raw building blocks in intermediary_calculations_securities table
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
        logging.FileHandler(project_root / 'logs' / 'securities_intermediary_calculations_cleaned.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

class SecuritiesIntermediaryPopulatorCleaned:
    """Calculates and populates ONLY raw building blocks for securities sector"""
    
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
    
    def get_securities_tickers(self) -> List[str]:
        """Get all securities sector tickers with validation"""
        query = """
        SELECT ticker 
        FROM master_info 
        WHERE sector = 'Securities'
        AND ticker COLLATE utf8mb4_unicode_ci NOT IN (
            SELECT DISTINCT ticker COLLATE utf8mb4_unicode_ci FROM intermediary_calculations_enhanced
            UNION
            SELECT DISTINCT ticker COLLATE utf8mb4_unicode_ci FROM intermediary_calculations_banking
        )
        ORDER BY ticker
        """
        
        try:
            with self.db_connection.cursor() as cursor:
                cursor.execute(query)
                results = cursor.fetchall()
                
            tickers = [row['ticker'] for row in results]
            logger.info(f"Found {len(tickers)} securities tickers: {tickers}")
            return tickers
            
        except Exception as e:
            logger.error(f"Error fetching securities tickers: {e}")
            return []
    
    def get_securities_fundamental_data(self, ticker: str) -> pd.DataFrame:
        """Get securities fundamental data for calculations"""
        query = """
        SELECT 
            ticker, year, quarter,
            -- Securities service revenues
            BrokerageRevenue, UnderwritingRevenue, AdvisoryRevenue,
            CustodyServiceRevenue, EntrustedAuctionRevenue, OtherOperatingIncome,
            
            -- Trading revenues
            TradingGainFVTPL, TradingGainHTM, TradingGainLoans,
            TradingGainAFS, TradingGainDerivatives,
            
            -- Operating expenses and results
            ManagementExpenses, OperatingResult,
            
            -- Core P&L items
            ProfitBeforeTax, IncomeTaxExpense, ProfitAfterTax,
            
            -- Balance Sheet items
            TotalAssets, FinancialAssets, CashAndCashEquivalents,
            FinancialAssetsFVTPL, LoanReceivables, OwnersEquity,
            CharterCapital, RetainedEarnings
            
        FROM v_complete_securities_fundamentals
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
    
    def calculate_securities_ttm_values(self, data: pd.DataFrame, year: int, quarter: int) -> Dict[str, float]:
        """Calculate securities TTM intermediaries - RAW VALUES ONLY"""
        ttm_quarters = self._get_ttm_quarters(year, quarter)
        data['period_key'] = data['year'].astype(str) + 'Q' + data['quarter'].astype(str)
        ttm_data = data[data['period_key'].isin([f"{y}Q{q}" for y, q in ttm_quarters])]
        
        if len(ttm_data) < 4:
            return {}
        
        results = {}
        
        # Securities-specific TTM mappings - RAW VALUES ONLY
        securities_ttm_mappings = {
            'BrokerageRevenue': 'BrokerageRevenue_TTM',
            'UnderwritingRevenue': 'UnderwritingRevenue_TTM',
            'AdvisoryRevenue': 'AdvisoryRevenue_TTM',
            'CustodyServiceRevenue': 'CustodyServiceRevenue_TTM',
            'EntrustedAuctionRevenue': 'EntrustedAuctionRevenue_TTM',
            'OtherOperatingIncome': 'OtherOperatingIncome_TTM',
            'TradingGainFVTPL': 'TradingGainFVTPL_TTM',
            'TradingGainHTM': 'TradingGainHTM_TTM',
            'TradingGainLoans': 'TradingGainLoans_TTM',
            'TradingGainAFS': 'TradingGainAFS_TTM',
            'TradingGainDerivatives': 'TradingGainDerivatives_TTM',
            'ManagementExpenses': 'ManagementExpenses_TTM',
            'BrokerageExpenses': 'BrokerageExpenses_TTM',
            'AdvisoryExpenses': 'AdvisoryExpenses_TTM',
            'CustodyExpenses': 'CustodyServiceExpenses_TTM',
            'OtherExpenses': 'OtherOperatingExpenses_TTM',
            'OperatingResult': 'OperatingResult_TTM',
            'ProfitBeforeTax': 'ProfitBeforeTax_TTM',
            'IncomeTaxExpense': 'IncomeTaxExpense_TTM',
            'ProfitAfterTax': 'NetProfit_TTM'
        }
        
        for source_col, target_col in securities_ttm_mappings.items():
            if source_col in ttm_data.columns:
                values = pd.to_numeric(ttm_data[source_col], errors='coerce')
                results[target_col] = values.sum() if not values.isna().all() else None
        
        # Calculate OperatingExpenses_TTM as sum of expense components
        expense_components = [
            'BrokerageExpenses_TTM',
            'AdvisoryExpenses_TTM', 
            'CustodyServiceExpenses_TTM',
            'ManagementExpenses_TTM',
            'OtherOperatingExpenses_TTM'
        ]
        
        total_expenses = 0
        has_expenses = False
        for expense_field in expense_components:
            if expense_field in results and results[expense_field] is not None:
                total_expenses += results[expense_field]
                has_expenses = True
        
        results['OperatingExpenses_TTM'] = total_expenses if has_expenses else None
        
        # Calculate TotalOperatingRevenue_TTM (bonus fix mentioned in handoff)
        revenue_components = [
            'BrokerageRevenue_TTM',
            'UnderwritingRevenue_TTM',
            'AdvisoryRevenue_TTM',
            'CustodyServiceRevenue_TTM',
            'TradingGainFVTPL_TTM',
            'TradingGainHTM_TTM', 
            'TradingGainLoans_TTM',
            'TradingGainAFS_TTM',
            'OtherOperatingIncome_TTM'
        ]
        
        total_revenue = 0
        has_revenue = False
        for revenue_field in revenue_components:
            if revenue_field in results and results[revenue_field] is not None:
                total_revenue += results[revenue_field]
                has_revenue = True
        
        results['TotalOperatingRevenue_TTM'] = total_revenue if has_revenue else None
        
        return results
    
    def calculate_securities_5point_averages(self, data: pd.DataFrame, year: int, quarter: int) -> Dict[str, float]:
        """Calculate securities 5-point balance sheet averages - RAW VALUES ONLY"""
        avg_quarters = self._get_5point_quarters(year, quarter)
        data['period_key'] = data['year'].astype(str) + 'Q' + data['quarter'].astype(str)
        avg_data = data[data['period_key'].isin([f"{y}Q{q}" for y, q in avg_quarters])]
        
        if len(avg_data) < 5:
            return {}
        
        results = {}
        
        # Securities-specific averaging mappings - RAW VALUES ONLY
        securities_avg_mappings = {
            'TotalAssets': 'AvgTotalAssets',
            'FinancialAssets': 'AvgFinancialAssets',
            'CashAndCashEquivalents': 'AvgCashAndCashEquivalents',
            'FinancialAssetsFVTPL': 'AvgFinancialAssetsFVTPL',
            'LoanReceivables': 'AvgLoanReceivables',
            'OwnersEquity': 'AvgTotalEquity',
            'CharterCapital': 'AvgCharterCapital',
            'RetainedEarnings': 'AvgRetainedEarnings'
        }
        
        for source_col, target_col in securities_avg_mappings.items():
            if source_col in avg_data.columns:
                values = pd.to_numeric(avg_data[source_col], errors='coerce')
                if not values.isna().all():
                    results[target_col] = values.mean()
        
        return results
    
    def calculate_securities_complex_aggregations(self, ttm_values: Dict, avg_values: Dict) -> Dict[str, float]:
        """Calculate ONLY complex aggregations that are difficult to compute dynamically"""
        results = {}
        
        # ✅ KEEP: Total securities service revenue (Complex aggregation)
        service_revenue = (
            ttm_values.get('BrokerageRevenue_TTM', 0) +
            ttm_values.get('AdvisoryRevenue_TTM', 0) +
            ttm_values.get('CustodyServiceRevenue_TTM', 0) +
            ttm_values.get('UnderwritingRevenue_TTM', 0) +
            ttm_values.get('EntrustedAuctionRevenue_TTM', 0)
        )
        results['TotalSecuritiesServices_TTM'] = service_revenue if service_revenue > 0 else None
        
        # ✅ KEEP: Net trading income (Complex aggregation)
        trading_income = (
            ttm_values.get('TradingGainFVTPL_TTM', 0) +
            ttm_values.get('TradingGainHTM_TTM', 0) +
            ttm_values.get('TradingGainLoans_TTM', 0) +
            ttm_values.get('TradingGainAFS_TTM', 0) +
            ttm_values.get('TradingGainDerivatives_TTM', 0)
        )
        results['NetTradingIncome_TTM'] = trading_income if trading_income > 0 else None
        
        # ✅ KEEP: Total operating revenue (Complex aggregation)
        total_revenue = service_revenue + trading_income + ttm_values.get('OtherOperatingIncome_TTM', 0)
        results['TotalOperatingRevenue_TTM'] = total_revenue if total_revenue > 0 else None
        
        # ❌ REMOVED: All ratio calculations with * 100
        # These will be calculated dynamically in the backtesting layer:
        # - ROAE, ROAA, BrokerageRatio, AdvisoryRatio, CustodyRatio
        # - TradingRatio, ServiceRatio, NetProfitMargin, MarginLendingYield
        # - EquityRatio, LeverageRatio, AssetTurnover
        
        return results
    
    def validate_securities_sector_assignment(self, ticker: str) -> bool:
        """Validate that ticker belongs to securities sector"""
        query = "SELECT sector FROM master_info WHERE ticker = %s"
        try:
            with self.db_connection.cursor() as cursor:
                cursor.execute(query, [ticker])
                result = cursor.fetchone()
                if result:
                    sector = result['sector']
                    is_valid = sector == 'Securities'
                    if not is_valid:
                        logger.warning(f"Securities sector validation failed: {ticker} is {sector}, expected Securities")
                    return is_valid
                else:
                    logger.warning(f"Ticker {ticker} not found in master_info")
                    return False
        except Exception as e:
            logger.error(f"Error validating securities sector for {ticker}: {e}")
            return False
    
    def insert_securities_intermediary_values(self, ticker: str, year: int, quarter: int, values: Dict) -> bool:
        """Insert securities intermediary values into database with sector validation"""
        try:
            # Validate sector assignment before insertion
            if not self.validate_securities_sector_assignment(ticker):
                logger.error(f"Skipping insertion for {ticker} due to securities sector validation failure")
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
            INSERT INTO intermediary_calculations_securities_cleaned ({columns_str})
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
    
    def process_securities_ticker(self, ticker: str, ticker_progress: tqdm = None) -> int:
        """Process all periods for a single securities ticker"""
        try:
            data = self.get_securities_fundamental_data(ticker)
            
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
                ttm_values = self.calculate_securities_ttm_values(data, year, quarter)
                avg_values = self.calculate_securities_5point_averages(data, year, quarter)
                complex_aggs = self.calculate_securities_complex_aggregations(ttm_values, avg_values)
                
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
                    if self.insert_securities_intermediary_values(ticker, year, quarter, all_values):
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
        """Run the cleaned securities intermediary calculation process"""
        start_time = datetime.now()
        logger.info("=" * 80)
        logger.info("STARTING CLEANED SECURITIES SECTOR INTERMEDIARY CALCULATIONS")
        logger.info("PHASE 5: Raw building blocks only - NO pre-calculated ratios")
        logger.info(f"Start time: {start_time}")
        logger.info("=" * 80)
        
        # Get securities tickers
        all_tickers = self.get_securities_tickers()
        
        if specific_tickers:
            tickers = [t for t in specific_tickers if t in all_tickers]
        else:
            tickers = all_tickers
        
        total_tickers = len(tickers)
        logger.info(f"Processing {total_tickers} securities tickers: {tickers}")
        
        # Ticker-level progress bar
        ticker_progress = tqdm(
            tickers,
            desc="Securities Tickers",
            unit="ticker", 
            position=0,
            leave=True
        )
        
        total_processed = 0
        total_errors = 0
        total_periods = 0
        
        for ticker in ticker_progress:
            try:
                periods = self.process_securities_ticker(ticker, ticker_progress)
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
        logger.info("CLEANED SECURITIES SECTOR INTERMEDIARY CALCULATION COMPLETE")
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
    parser = argparse.ArgumentParser(description='Cleaned Securities Sector Intermediary Calculator')
    parser.add_argument('--tickers', nargs='+', help='Process specific tickers only')
    parser.add_argument('--test', action='store_true', help='Test mode - process VCI and SSI only')
    args = parser.parse_args()
    
    populator = SecuritiesIntermediaryPopulatorCleaned()
    
    if args.test:
        logger.info("Running in test mode - processing VCI and SSI only")
        populator.run(['VCI', 'SSI'])
    else:
        populator.run(args.tickers)

if __name__ == "__main__":
    main()