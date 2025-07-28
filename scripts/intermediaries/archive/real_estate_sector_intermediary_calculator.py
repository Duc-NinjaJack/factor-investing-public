#!/usr/bin/env python3
"""
Real Estate Sector Intermediary Calculator - Database Population
===============================================================
Phase 2: Calculate and store intermediary values for Real Estate sector
Based on successful tech sector implementation

This script:
1. Calculates TTM values, 5-point averages, and working capital metrics
2. Stores results in intermediary_calculations_enhanced table
3. Processes all real estate sector tickers (82 tickers)
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

# Setup project path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealEstateSectorIntermediaryPopulator:
    """Calculates and populates intermediary values for real estate sector"""
    
    def __init__(self):
        self.db_connection = self._get_db_connection()
        self.processed_count = 0
        self.error_count = 0
    
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
    
    def get_real_estate_tickers(self) -> List[str]:
        """Get all real estate sector tickers"""
        query = """
        SELECT DISTINCT ticker 
        FROM master_info 
        WHERE sector = 'Real Estate'
        ORDER BY ticker
        """
        
        try:
            with self.db_connection.cursor() as cursor:
                cursor.execute(query)
                results = cursor.fetchall()
                tickers = [row['ticker'] for row in results]
                logger.info(f"Found {len(tickers)} real estate sector tickers")
                return tickers
        except Exception as e:
            logger.error(f"Error fetching tickers: {e}")
            return []
    
    def get_fundamental_data(self, ticker: str) -> pd.DataFrame:
        """Get fundamental data for calculations"""
        query = """
        SELECT 
            ticker, year, quarter,
            -- Income Statement (Flow items for TTM)
            NetRevenue, COGS, GrossProfit, SellingExpenses, AdminExpenses,
            FinancialIncome, FinancialExpenses, InterestExpenses,
            ProfitBeforeTax, CurrentIncomeTax, TotalIncomeTax,
            NetProfit, NetProfitAfterMI, EBIT,
            -- Balance Sheet (Stock items for 5-point averaging)
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
                    logger.warning(f"No data found for {ticker}")
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
    
    def calculate_working_capital_metrics(self, ttm_values: Dict, avg_values: Dict) -> Dict[str, float]:
        """Calculate working capital metrics"""
        results = {}
        
        # DSO (Days Sales Outstanding)
        if 'AvgReceivables' in avg_values and 'Revenue_TTM' in ttm_values:
            if ttm_values.get('Revenue_TTM', 0) != 0:
                results['DSO'] = (avg_values['AvgReceivables'] * 365) / ttm_values['Revenue_TTM']
        
        # DIO (Days Inventory Outstanding)
        if 'AvgInventory' in avg_values and 'COGS_TTM' in ttm_values:
            if ttm_values.get('COGS_TTM', 0) != 0:
                results['DIO'] = (avg_values['AvgInventory'] * 365) / abs(ttm_values['COGS_TTM'])
        
        # DPO (Days Payables Outstanding)
        if 'AvgPayables' in avg_values and 'COGS_TTM' in ttm_values:
            if ttm_values.get('COGS_TTM', 0) != 0:
                results['DPO'] = (avg_values['AvgPayables'] * 365) / abs(ttm_values['COGS_TTM'])
        
        # CCC (Cash Conversion Cycle)
        if all(metric in results for metric in ['DSO', 'DIO', 'DPO']):
            results['CCC'] = results['DSO'] + results['DIO'] - results['DPO']
        
        return results
    
    def insert_intermediary_values(self, ticker: str, year: int, quarter: int, values: Dict) -> bool:
        """Insert intermediary values into database"""
        try:
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
    
    def process_ticker(self, ticker: str) -> int:
        """Process all periods for a single ticker"""
        logger.info(f"Processing {ticker}...")
        data = self.get_fundamental_data(ticker)
        
        if data.empty:
            return 0
        
        # Get unique year-quarter combinations
        periods = data[['year', 'quarter']].drop_duplicates().sort_values(['year', 'quarter'])
        count = 0
        
        for _, period in periods.iterrows():
            year, quarter = int(period['year']), int(period['quarter'])
            
            # Calculate all intermediaries
            ttm_values = self.calculate_ttm_values(data, year, quarter)
            avg_values = self.calculate_5point_averages(data, year, quarter)
            wc_values = self.calculate_working_capital_metrics(ttm_values, avg_values)
            
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
        
        logger.info(f"Completed {ticker}: {count} periods processed")
        return count
    
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
    
    def run(self, specific_ticker: Optional[str] = None):
        """Run the intermediary calculation process"""
        if specific_ticker:
            tickers = [specific_ticker]
        else:
            tickers = self.get_real_estate_tickers()
        
        logger.info(f"Starting intermediary calculations for {len(tickers)} real estate tickers")
        
        total_periods = 0
        for ticker in tickers:
            try:
                periods = self.process_ticker(ticker)
                total_periods += periods
                self.processed_count += 1
            except Exception as e:
                logger.error(f"Failed to process {ticker}: {e}")
                self.error_count += 1
        
        logger.info(f"""
Real Estate Intermediary Calculation Complete:
- Tickers processed: {self.processed_count}
- Errors: {self.error_count}
- Total periods: {total_periods}
""")

def main():
    parser = argparse.ArgumentParser(description='Real Estate Sector Intermediary Calculator')
    parser.add_argument('--ticker', help='Process specific ticker only')
    parser.add_argument('--test', action='store_true', help='Test mode - process NLG only')
    args = parser.parse_args()
    
    populator = RealEstateSectorIntermediaryPopulator()
    
    if args.test:
        logger.info("Running in test mode - processing NLG only")
        populator.run('NLG')
    else:
        populator.run(args.ticker)

if __name__ == "__main__":
    main()