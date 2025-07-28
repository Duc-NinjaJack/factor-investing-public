#!/usr/bin/env python3
"""
Banking Intermediary Table Creator
================================
Creates specialized intermediary_calculations_banking table for banking sector.

This table stores pre-calculated banking-specific intermediaries:
- Banking TTM values (NII, Fee Income, Provisions, etc.)
- Banking 5-point averages (Loans, Deposits, Assets, etc.)
- Banking derived metrics (NIM, LDR, Cost of Credit, etc.)

Author: Duc Nguyen (Aureus Sigma Capital)
Date: July 3, 2025
"""

import sys
import pymysql
import yaml
from pathlib import Path
from datetime import datetime
import logging

# Setup project path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BankingIntermediaryTableCreator:
    """Creates the banking-specific intermediary calculations table"""
    
    def __init__(self):
        self.db_connection = self._get_db_connection()
    
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
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor
            )
            return connection
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def create_banking_intermediary_table(self):
        """Create the banking intermediary calculations table"""
        
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS intermediary_calculations_banking (
            -- Primary Keys & Metadata
            ticker CHAR(10) NOT NULL,
            year INT NOT NULL,
            quarter INT NOT NULL,
            calc_date DATE NOT NULL DEFAULT (CURRENT_DATE),
            
            -- Banking TTM Flow Items (Income Statement)
            NII_TTM DECIMAL(30,2) COMMENT 'Net Interest Income TTM',
            InterestIncome_TTM DECIMAL(30,2) COMMENT 'Interest Income TTM',
            InterestExpense_TTM DECIMAL(30,2) COMMENT 'Interest Expense TTM',
            NetFeeIncome_TTM DECIMAL(30,2) COMMENT 'Net Fee and Commission Income TTM',
            FeeIncome_TTM DECIMAL(30,2) COMMENT 'Fee and Commission Income TTM',
            FeeExpense_TTM DECIMAL(30,2) COMMENT 'Fee and Commission Expense TTM',
            ForexIncome_TTM DECIMAL(30,2) COMMENT 'Net Foreign Exchange Income TTM',
            TradingIncome_TTM DECIMAL(30,2) COMMENT 'Net Trading Securities Income TTM',
            InvestmentIncome_TTM DECIMAL(30,2) COMMENT 'Net Investment Securities Income TTM',
            OtherIncome_TTM DECIMAL(30,2) COMMENT 'Net Other Income TTM',
            EquityInvestmentIncome_TTM DECIMAL(30,2) COMMENT 'Income from Equity Investments TTM',
            TotalOperatingIncome_TTM DECIMAL(30,2) COMMENT 'Total Operating Income (NII + All Non-Interest Income) TTM',
            OperatingExpenses_TTM DECIMAL(30,2) COMMENT 'Operating Expenses TTM',
            OperatingProfit_TTM DECIMAL(30,2) COMMENT 'Operating Profit Before Provisions TTM',
            CreditProvisions_TTM DECIMAL(30,2) COMMENT 'Credit Loss Provisions TTM',
            ProfitBeforeTax_TTM DECIMAL(30,2) COMMENT 'Profit Before Tax TTM',
            TaxExpense_TTM DECIMAL(30,2) COMMENT 'Income Tax Expense TTM',
            NetProfit_TTM DECIMAL(30,2) COMMENT 'Net Profit After Tax TTM',
            NetProfitAfterMI_TTM DECIMAL(30,2) COMMENT 'Net Profit After Minority Interest TTM',
            
            -- Banking Balance Sheet 5-Point Averages
            AvgTotalAssets DECIMAL(30,2) COMMENT 'Average Total Assets (5-point)',
            AvgGrossLoans DECIMAL(30,2) COMMENT 'Average Gross Customer Loans (5-point)',
            AvgLoanLossReserves DECIMAL(30,2) COMMENT 'Average Loan Loss Reserves (5-point)',
            AvgNetLoans DECIMAL(30,2) COMMENT 'Average Net Customer Loans (5-point)',
            AvgEarningAssets DECIMAL(30,2) COMMENT 'Average Earning Assets (5-point)',
            AvgTradingSecurities DECIMAL(30,2) COMMENT 'Average Trading Securities (5-point)',
            AvgInvestmentSecurities DECIMAL(30,2) COMMENT 'Average Investment Securities (5-point)',
            AvgCash DECIMAL(30,2) COMMENT 'Average Cash and Central Bank Deposits (5-point)',
            AvgCustomerDeposits DECIMAL(30,2) COMMENT 'Average Customer Deposits (5-point)',
            AvgTotalDeposits DECIMAL(30,2) COMMENT 'Average Total Deposits (5-point)',
            AvgBorrowings DECIMAL(30,2) COMMENT 'Average Borrowings (5-point)',
            AvgTotalLiabilities DECIMAL(30,2) COMMENT 'Average Total Liabilities (5-point)',
            AvgTotalEquity DECIMAL(30,2) COMMENT 'Average Total Equity (5-point)',
            AvgShareholderEquity DECIMAL(30,2) COMMENT 'Average Shareholder Equity (5-point)',
            AvgPaidInCapital DECIMAL(30,2) COMMENT 'Average Paid-In Capital (5-point)',
            AvgRetainedEarnings DECIMAL(30,2) COMMENT 'Average Retained Earnings (5-point)',
            
            -- Banking-Specific Derived Metrics
            NIM DECIMAL(10,4) COMMENT 'Net Interest Margin (%)',
            NIM_Gross DECIMAL(10,4) COMMENT 'Gross Interest Margin (%)',
            Cost_of_Funds DECIMAL(10,4) COMMENT 'Cost of Funds (%)',
            LDR DECIMAL(10,4) COMMENT 'Loan-to-Deposit Ratio (%)',
            LDR_Net DECIMAL(10,4) COMMENT 'Net Loan-to-Deposit Ratio (%)',
            Cost_of_Credit DECIMAL(10,4) COMMENT 'Cost of Credit (Provisions/Avg Loans) (%)',
            CAR_Proxy DECIMAL(10,4) COMMENT 'Capital Adequacy Ratio Proxy (%)',
            Leverage_Ratio DECIMAL(10,4) COMMENT 'Leverage Ratio (Equity/Assets) (%)',
            
            -- Banking Efficiency Metrics
            Cost_Income_Ratio DECIMAL(10,4) COMMENT 'Cost-to-Income Ratio (%)',
            Operating_Leverage DECIMAL(10,4) COMMENT 'Operating Leverage Ratio',
            Fee_Income_Ratio DECIMAL(10,4) COMMENT 'Fee Income / Total Income (%)',
            NonInterest_Income_Ratio DECIMAL(10,4) COMMENT 'Non-Interest Income Ratio (%)',
            
            -- Banking Asset Quality Metrics
            Provision_Coverage DECIMAL(10,4) COMMENT 'Provision Coverage Ratio (%)',
            Asset_Quality_Proxy DECIMAL(10,4) COMMENT 'Asset Quality Proxy (1 - Provisions/Loans)',
            Loan_Growth_QoQ DECIMAL(10,4) COMMENT 'Loan Growth Quarter-over-Quarter (%)',
            Deposit_Growth_QoQ DECIMAL(10,4) COMMENT 'Deposit Growth Quarter-over-Quarter (%)',
            
            -- Banking Profitability Metrics
            ROAA DECIMAL(10,4) COMMENT 'Return on Average Assets (%)',
            ROAE DECIMAL(10,4) COMMENT 'Return on Average Equity (%)',
            ROAA_PreProvision DECIMAL(10,4) COMMENT 'Pre-Provision ROAA (%)',
            ROAE_PreProvision DECIMAL(10,4) COMMENT 'Pre-Provision ROAE (%)',
            
            -- Metadata & Quality Indicators
            quarters_available_ttm INT COMMENT 'Number of quarters available for TTM calculation',
            has_full_ttm BOOLEAN COMMENT 'Whether full 4-quarter TTM is available',
            avg_points_used INT COMMENT 'Number of points used for 5-point averaging',
            has_full_avg BOOLEAN COMMENT 'Whether full 5-point averaging is available',
            data_quality_score DECIMAL(5,2) COMMENT 'Data quality score (0-100)',
            calculation_notes TEXT COMMENT 'Calculation notes and warnings',
            
            -- Indexes and Constraints
            PRIMARY KEY (ticker, year, quarter),
            INDEX idx_ticker (ticker),
            INDEX idx_year_quarter (year, quarter),
            INDEX idx_calc_date (calc_date),
            INDEX idx_nim (NIM),
            INDEX idx_roaa (ROAA),
            INDEX idx_roae (ROAE),
            INDEX idx_ldr (LDR)
        ) ENGINE=InnoDB 
        COMMENT='Banking sector intermediary calculations - TTM, averages, and derived metrics'
        DEFAULT CHARSET=utf8mb4 
        COLLATE=utf8mb4_unicode_ci;
        """
        
        try:
            with self.db_connection.cursor() as cursor:
                logger.info("Creating intermediary_calculations_banking table...")
                cursor.execute(create_table_sql)
                self.db_connection.commit()
                logger.info("✅ Banking intermediary table created successfully!")
                
                # Verify table creation
                cursor.execute("DESCRIBE intermediary_calculations_banking")
                columns = cursor.fetchall()
                logger.info(f"✅ Table created with {len(columns)} columns")
                
                # Show sample columns
                logger.info("Key banking-specific columns:")
                key_columns = ['NII_TTM', 'TotalOperatingIncome_TTM', 'AvgGrossLoans', 'NIM', 'LDR', 'ROAA', 'ROAE']
                for col in columns:
                    if col['Field'] in key_columns:
                        comment = col.get('Comment', 'No comment')
                        logger.info(f"  • {col['Field']}: {col['Type']} - {comment}")
                
        except Exception as e:
            logger.error(f"Failed to create banking intermediary table: {e}")
            self.db_connection.rollback()
            raise
    
    def check_existing_table(self):
        """Check if table already exists and show its structure"""
        try:
            with self.db_connection.cursor() as cursor:
                cursor.execute("SHOW TABLES LIKE 'intermediary_calculations_banking'")
                result = cursor.fetchone()
                
                if result:
                    logger.info("✅ Table already exists. Current structure:")
                    cursor.execute("DESCRIBE intermediary_calculations_banking")
                    columns = cursor.fetchall()
                    logger.info(f"Current table has {len(columns)} columns")
                    return True
                else:
                    logger.info("Table does not exist. Will create new table.")
                    return False
                    
        except Exception as e:
            logger.error(f"Error checking existing table: {e}")
            return False

def main():
    """Main execution function"""
    logger.info("=" * 80)
    logger.info("BANKING INTERMEDIARY TABLE CREATOR")
    logger.info(f"Start time: {datetime.now()}")
    logger.info("=" * 80)
    
    creator = BankingIntermediaryTableCreator()
    
    # Check if table exists
    exists = creator.check_existing_table()
    
    if exists:
        response = input("\nTable exists. Recreate? (y/n): ").strip().lower()
        if response != 'y':
            logger.info("Keeping existing table. Exiting.")
            return
        else:
            # Drop existing table
            with creator.db_connection.cursor() as cursor:
                cursor.execute("DROP TABLE IF EXISTS intermediary_calculations_banking")
                creator.db_connection.commit()
                logger.info("Dropped existing table.")
    
    # Create the table
    creator.create_banking_intermediary_table()
    
    logger.info("=" * 80)
    logger.info("BANKING INTERMEDIARY TABLE CREATION COMPLETE")
    logger.info(f"End time: {datetime.now()}")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()