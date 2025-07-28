#!/usr/bin/env python3
"""
Securities Sector Intermediary Calculator (Phase 2)
===================================================
Production-ready calculator for securities intermediary values.

Features:
- Proper 5-point rolling averages for balance sheet items
- Complete P&L revenue component breakdowns
- Accurate TTM calculations for all metrics

Author: Duc Nguyen (Aureus Sigma Capital)
Date: July 3, 2025
"""

import sys
import pandas as pd
import pymysql
import yaml
from pathlib import Path
from datetime import datetime
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Setup project path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SecuritiesIntermediaryCalculator:
    """Securities sector intermediary calculations with fixed rolling windows"""
    
    def __init__(self):
        self.db_connection = self._get_db_connection()
        self.processed_count = 0
        self.error_count = 0
        
    def __del__(self):
        if hasattr(self, 'db_connection') and self.db_connection:
            self.db_connection.close()
    
    def _get_db_connection(self):
        """Get database connection"""
        try:
            config_path = project_root / 'config' / 'database.yml'
            with open(config_path, 'r') as f:
                db_config = yaml.safe_load(f)['production']
            
            connection = pymysql.connect(
                host=db_config['host'],
                user=db_config['username'],
                password=db_config['password'],
                database=db_config['schema_name'],
                charset='utf8mb4'
            )
            return connection
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def get_securities_tickers(self):
        """Get all securities tickers from master_info"""
        cursor = self.db_connection.cursor()
        cursor.execute("""
            SELECT ticker FROM master_info 
            WHERE sector = 'Securities' 
            ORDER BY ticker
        """)
        results = cursor.fetchall()
        tickers = [row[0] for row in results]
        logger.info(f"Found {len(tickers)} securities tickers: {', '.join(tickers)}")
        return tickers
    
    def process_ticker(self, ticker):
        """Process a single ticker"""
        cursor = self.db_connection.cursor()
        
        # Get ALL historical data for proper rolling calculations
        cursor.execute("""
            SELECT * FROM v_complete_securities_fundamentals 
            WHERE ticker = %s 
            AND year > 2000 
            AND quarter BETWEEN 1 AND 4
            ORDER BY year DESC, quarter DESC 
            LIMIT 100
        """, [ticker])
        
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        
        if not rows:
            logger.warning(f"No data found for {ticker}")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(rows, columns=columns)
        
        # Convert numeric columns to float
        for col in df.columns:
            if col not in ['ticker', 'year', 'quarter']:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    pass
        
        # Create a period identifier for easier sorting
        df['period'] = df['year'] * 10 + df['quarter']
        df = df.sort_values('period', ascending=False).reset_index(drop=True)
        
        # Process each quarter with at least 2 quarters of history
        for i in range(len(df) - 1):
            try:
                current_row = df.iloc[i]
                year = int(current_row['year'])
                quarter = int(current_row['quarter'])
                
                # Get TTM data (current + 3 previous quarters)
                ttm_data = df.iloc[i:i+4]
                
                # Get 5-point average data (current + 4 previous quarters)
                avg_data = df.iloc[i:i+5]
                
                if len(ttm_data) < 2:
                    continue
                
                # Calculate intermediaries
                intermediary_values = self.calculate_intermediaries(
                    ticker, year, quarter, ttm_data, avg_data
                )
                
                # Upsert to database
                self.upsert_intermediary_values(intermediary_values)
                self.processed_count += 1
                
            except Exception as e:
                logger.error(f"Error processing {ticker} {year}Q{quarter}: {e}")
                self.error_count += 1
    
    def calculate_intermediaries(self, ticker, year, quarter, ttm_data, avg_data):
        """Calculate all intermediary values with proper rolling windows"""
        
        intermediaries = {
            'ticker': ticker,
            'year': year,
            'quarter': quarter,
            'calc_date': datetime.now().date()
        }
        
        # TTM Revenue Streams (sum of 4 quarters)
        ttm_columns = {
            # Securities service revenues
            'BrokerageRevenue': 'BrokerageRevenue_TTM',
            'UnderwritingRevenue': 'UnderwritingRevenue_TTM',
            'AdvisoryRevenue': 'AdvisoryRevenue_TTM',
            'CustodyServiceRevenue': 'CustodyServiceRevenue_TTM',
            'EntrustedAuctionRevenue': 'EntrustedAuctionRevenue_TTM',
            'OtherOperatingIncome': 'OtherOperatingIncome_TTM',
            
            # Trading revenues
            'TradingGainFVTPL': 'TradingGainFVTPL_TTM',
            'TradingGainHTM': 'TradingGainHTM_TTM',
            'TradingGainLoans': 'TradingGainLoans_TTM',  # Margin lending income
            'TradingGainAFS': 'TradingGainAFS_TTM',
            'TradingGainDerivatives': 'TradingGainDerivatives_TTM',
            
            # Operating expenses and results
            'ManagementExpenses': 'ManagementExpenses_TTM',
            'OperatingResult': 'OperatingResult_TTM',
            
            # Core P&L items
            'ProfitBeforeTax': 'ProfitBeforeTax_TTM',
            'IncomeTaxExpense': 'IncomeTaxExpense_TTM',
            'ProfitAfterTax': 'NetProfit_TTM',
            
            # Legacy operating items (if available)
            'OperatingExpenses': 'OperatingExpenses_TTM'
        }
        
        quarters_available = len(ttm_data)
        for col, target in ttm_columns.items():
            if col in ttm_data.columns:
                value = ttm_data[col].sum()
                intermediaries[target] = float(value) if pd.notna(value) else None
        
        # Calculate total securities service revenue
        service_revenue = 0
        for col in ['BrokerageRevenue_TTM', 'AdvisoryRevenue_TTM', 'CustodyServiceRevenue_TTM']:
            if col in intermediaries and intermediaries[col]:
                service_revenue += intermediaries[col]
        intermediaries['TotalSecuritiesServices_TTM'] = service_revenue if service_revenue > 0 else None
        
        # Calculate net trading income
        trading_income = 0
        for col in ['TradingGainFVTPL_TTM', 'TradingGainHTM_TTM', 'TradingGainLoans_TTM', 
                    'TradingGainAFS_TTM', 'TradingGainDerivatives_TTM']:
            if col in intermediaries and intermediaries[col]:
                trading_income += intermediaries[col]
        intermediaries['NetTradingIncome_TTM'] = trading_income if trading_income > 0 else None
        
        # Calculate total operating revenue (services + trading)
        total_revenue = (service_revenue + trading_income) if (service_revenue + trading_income) > 0 else None
        intermediaries['TotalOperatingRevenue_TTM'] = total_revenue
        
        # Balance Sheet 5-Point Rolling Averages
        avg_columns = {
            'TotalAssets': 'AvgTotalAssets',
            'FinancialAssets': 'AvgFinancialAssets',
            'CashAndCashEquivalents': 'AvgCashAndCashEquivalents',
            'FinancialAssetsFVTPL': 'AvgFinancialAssetsFVTPL',
            'LoanReceivables': 'AvgLoanReceivables',  # Margin lending book
            'OwnersEquity': 'AvgTotalEquity',
            'CharterCapital': 'AvgCharterCapital',
            'RetainedEarnings': 'AvgRetainedEarnings'
        }
        
        avg_points = len(avg_data)
        for col, target in avg_columns.items():
            if col in avg_data.columns:
                value = avg_data[col].mean()
                intermediaries[target] = float(value) if pd.notna(value) else None
        
        # Derived Metrics
        self.calculate_derived_metrics(intermediaries)
        
        # Metadata
        intermediaries['quarters_available_ttm'] = quarters_available
        intermediaries['has_full_ttm'] = quarters_available >= 4
        intermediaries['avg_points_used'] = avg_points
        intermediaries['has_full_avg'] = avg_points >= 5
        intermediaries['data_quality_score'] = min(100, (quarters_available / 4 * 50) + (avg_points / 5 * 50))
        
        return intermediaries
    
    def calculate_derived_metrics(self, values):
        """Calculate derived metrics"""
        
        # Revenue mix ratios
        total_revenue = values.get('TotalOperatingRevenue_TTM', 0)
        if total_revenue and total_revenue > 0:
            # Service revenue ratios
            values['BrokerageRatio'] = (values.get('BrokerageRevenue_TTM', 0) / total_revenue * 100) if values.get('BrokerageRevenue_TTM') else None
            values['AdvisoryRatio'] = (values.get('AdvisoryRevenue_TTM', 0) / total_revenue * 100) if values.get('AdvisoryRevenue_TTM') else None
            values['CustodyRatio'] = (values.get('CustodyServiceRevenue_TTM', 0) / total_revenue * 100) if values.get('CustodyServiceRevenue_TTM') else None
            
            # Trading vs service split
            values['TradingRatio'] = (values.get('NetTradingIncome_TTM', 0) / total_revenue * 100) if values.get('NetTradingIncome_TTM') else None
            values['ServiceRatio'] = (values.get('TotalSecuritiesServices_TTM', 0) / total_revenue * 100) if values.get('TotalSecuritiesServices_TTM') else None
        
        # Profitability metrics
        avg_assets = values.get('AvgTotalAssets', 0)
        if avg_assets and avg_assets > 0:
            values['ROAA'] = (values.get('NetProfit_TTM', 0) / avg_assets * 100) if values.get('NetProfit_TTM') else None
            values['AssetTurnover'] = total_revenue / avg_assets if total_revenue else None
        
        avg_equity = values.get('AvgTotalEquity', 0)
        if avg_equity and avg_equity > 0:
            values['ROAE'] = (values.get('NetProfit_TTM', 0) / avg_equity * 100) if values.get('NetProfit_TTM') else None
        
        # Capital and leverage metrics
        if avg_assets and avg_assets > 0 and avg_equity:
            values['EquityRatio'] = avg_equity / avg_assets * 100
            values['LeverageRatio'] = avg_assets / avg_equity if avg_equity > 0 else None
        
        # Margin lending efficiency
        avg_loans = values.get('AvgLoanReceivables', 0)
        if avg_loans and avg_loans > 0:
            margin_income = values.get('TradingGainLoans_TTM', 0)
            values['MarginLendingYield'] = (margin_income / avg_loans * 100) if margin_income else None
    
    def upsert_intermediary_values(self, values):
        """Upsert values to database"""
        
        # Filter out None values
        clean_values = {k: v for k, v in values.items() if v is not None}
        
        columns = list(clean_values.keys())
        placeholders = ', '.join(['%s'] * len(columns))
        update_clause = ', '.join([f'{col} = VALUES({col})' for col in columns if col not in ['ticker', 'year', 'quarter']])
        
        query = f"""
        INSERT INTO intermediary_calculations_securities 
        ({', '.join(columns)})
        VALUES ({placeholders})
        ON DUPLICATE KEY UPDATE {update_clause}
        """
        
        cursor = self.db_connection.cursor()
        cursor.execute(query, list(clean_values.values()))
        self.db_connection.commit()
    
    def run(self):
        """Run calculations for all securities tickers"""
        
        logger.info("=" * 80)
        logger.info("SECURITIES INTERMEDIARY CALCULATIONS (Production)")
        logger.info(f"Start time: {datetime.now()}")
        logger.info("=" * 80)
        
        tickers = self.get_securities_tickers()
        
        for ticker in tqdm(tickers, desc="Processing securities"):
            try:
                self.process_ticker(ticker)
            except Exception as e:
                logger.error(f"Failed to process {ticker}: {e}")
                self.error_count += 1
        
        logger.info("=" * 80)
        logger.info(f"Completed: {self.processed_count} records")
        logger.info(f"Errors: {self.error_count}")
        logger.info(f"End time: {datetime.now()}")
        logger.info("=" * 80)

def main():
    calculator = SecuritiesIntermediaryCalculator()
    calculator.run()

if __name__ == "__main__":
    main()