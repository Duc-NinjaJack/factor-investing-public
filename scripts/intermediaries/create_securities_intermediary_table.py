#!/usr/bin/env python3
"""
Securities Intermediary Table Creator (Phase 2)
==============================================
Creates specialized intermediary_calculations_securities table for securities sector.

This table stores pre-calculated securities-specific intermediaries:
- Securities TTM values (Trading Income, Brokerage Revenue, Advisory Revenue, etc.)
- Securities 5-point averages (Financial Assets, Client Receivables, Trading Portfolio, etc.)
- Securities derived metrics (Brokerage Ratio, Advisory Ratio, Asset Efficiency, etc.)

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

class SecuritiesIntermediaryTableCreator:
    """Creates the securities-specific intermediary calculations table"""
    
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
    
    def create_securities_intermediary_table(self):
        """Create the securities intermediary calculations table"""
        
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS intermediary_calculations_securities_cleaned (
            -- Primary Keys & Metadata
            ticker CHAR(10) NOT NULL,
            year INT NOT NULL,
            quarter INT NOT NULL,
            calc_date DATE NOT NULL DEFAULT (CURRENT_DATE),
            
            -- Securities TTM Revenue Streams (Income Statement)
            BrokerageRevenue_TTM DECIMAL(30,2) COMMENT 'Brokerage Revenue TTM',
            UnderwritingRevenue_TTM DECIMAL(30,2) COMMENT 'Underwriting Revenue TTM',
            AdvisoryRevenue_TTM DECIMAL(30,2) COMMENT 'Advisory Revenue TTM',
            CustodyServiceRevenue_TTM DECIMAL(30,2) COMMENT 'Custody Service Revenue TTM',
            EntrustedAuctionRevenue_TTM DECIMAL(30,2) COMMENT 'Entrusted Auction Revenue TTM',
            OtherOperatingIncome_TTM DECIMAL(30,2) COMMENT 'Other Operating Income TTM',
            TotalOperatingRevenue_TTM DECIMAL(30,2) COMMENT 'Total Operating Revenue TTM',
            TotalSecuritiesServices_TTM DECIMAL(30,2) COMMENT 'Total Securities Services Revenue TTM',
            
            -- Securities TTM Trading Income (Core P&L)
            TradingGainFVTPL_TTM DECIMAL(30,2) COMMENT 'Trading Gain FVTPL TTM',
            TradingGainHTM_TTM DECIMAL(30,2) COMMENT 'Trading Gain HTM TTM',
            TradingGainLoans_TTM DECIMAL(30,2) COMMENT 'Trading Gain Loans TTM',
            TradingGainAFS_TTM DECIMAL(30,2) COMMENT 'Trading Gain AFS TTM',
            TradingGainDerivatives_TTM DECIMAL(30,2) COMMENT 'Trading Gain Derivatives TTM',
            NetTradingIncome_TTM DECIMAL(30,2) COMMENT 'Net Trading Income TTM',
            
            -- Securities TTM Operating Expenses (Components)
            BrokerageExpenses_TTM DECIMAL(30,2) COMMENT 'Brokerage Expenses TTM',
            AdvisoryExpenses_TTM DECIMAL(30,2) COMMENT 'Advisory Expenses TTM',
            CustodyServiceExpenses_TTM DECIMAL(30,2) COMMENT 'Custody Service Expenses TTM',
            ManagementExpenses_TTM DECIMAL(30,2) COMMENT 'Management Expenses TTM',
            OtherOperatingExpenses_TTM DECIMAL(30,2) COMMENT 'Other Operating Expenses TTM',
            
            -- Securities TTM Core P&L
            OperatingRevenue_TTM DECIMAL(30,2) COMMENT 'Operating Revenue TTM',
            OperatingExpenses_TTM DECIMAL(30,2) COMMENT 'Operating Expenses TTM (Total)',
            OperatingResult_TTM DECIMAL(30,2) COMMENT 'Operating Result TTM',
            FinancialIncome_TTM DECIMAL(30,2) COMMENT 'Financial Income TTM',
            FinancialExpenses_TTM DECIMAL(30,2) COMMENT 'Financial Expenses TTM',
            ProfitBeforeTax_TTM DECIMAL(30,2) COMMENT 'Profit Before Tax TTM',
            IncomeTaxExpense_TTM DECIMAL(30,2) COMMENT 'Income Tax Expense TTM',
            NetProfit_TTM DECIMAL(30,2) COMMENT 'Net Profit After Tax TTM',
            
            -- Securities TTM Cash Flow
            NetCashFlowFromOperatingActivities_TTM DECIMAL(30,2) COMMENT 'Operating Cash Flow TTM',
            NetCashFlowFromInvestingActivities_TTM DECIMAL(30,2) COMMENT 'Investing Cash Flow TTM',
            NetCashFlowFromFinancingActivities_TTM DECIMAL(30,2) COMMENT 'Financing Cash Flow TTM',
            CapitalExpenditures_TTM DECIMAL(30,2) COMMENT 'Capital Expenditures TTM',
            DividendsPaidToOwners_TTM DECIMAL(30,2) COMMENT 'Dividends Paid TTM',
            
            -- Securities Balance Sheet 5-Point Averages
            AvgTotalAssets DECIMAL(30,2) COMMENT 'Average Total Assets (5-point)',
            AvgFinancialAssets DECIMAL(30,2) COMMENT 'Average Financial Assets (5-point)',
            AvgCashAndCashEquivalents DECIMAL(30,2) COMMENT 'Average Cash and Cash Equivalents (5-point)',
            AvgFinancialAssetsFVTPL DECIMAL(30,2) COMMENT 'Average Financial Assets FVTPL (5-point)',
            AvgHeldToMaturityInvestments DECIMAL(30,2) COMMENT 'Average HTM Investments (5-point)',
            AvgAvailableForSaleFinancial DECIMAL(30,2) COMMENT 'Average AFS Financial Assets (5-point)',
            AvgLoanReceivables DECIMAL(30,2) COMMENT 'Average Loan Receivables (5-point)',
            AvgReceivables DECIMAL(30,2) COMMENT 'Average Receivables (5-point)',
            AvgSecuritiesServicesReceivables DECIMAL(30,2) COMMENT 'Average Securities Services Receivables (5-point)',
            AvgFixedAssets DECIMAL(30,2) COMMENT 'Average Fixed Assets (5-point)',
            AvgLongTermFinancialAssets DECIMAL(30,2) COMMENT 'Average Long-Term Financial Assets (5-point)',
            
            -- Securities Liabilities 5-Point Averages
            AvgShortTermLiabilities DECIMAL(30,2) COMMENT 'Average Short-Term Liabilities (5-point)',
            AvgLongTermLiabilities DECIMAL(30,2) COMMENT 'Average Long-Term Liabilities (5-point)',
            AvgShortTermBorrowingsFinancial DECIMAL(30,2) COMMENT 'Average Short-Term Borrowings Financial (5-point)',
            AvgPayablesSecuritiesTrading DECIMAL(30,2) COMMENT 'Average Payables Securities Trading (5-point)',
            AvgTotalLiabilities DECIMAL(30,2) COMMENT 'Average Total Liabilities (5-point)',
            
            -- Securities Equity 5-Point Averages
            AvgTotalEquity DECIMAL(30,2) COMMENT 'Average Total Equity (5-point)',
            AvgOwnerCapital DECIMAL(30,2) COMMENT 'Average Owner Capital (5-point)',
            AvgCharterCapital DECIMAL(30,2) COMMENT 'Average Charter Capital (5-point)',
            AvgSharePremium DECIMAL(30,2) COMMENT 'Average Share Premium (5-point)',
            AvgRetainedEarnings DECIMAL(30,2) COMMENT 'Average Retained Earnings (5-point)',
            
            -- Securities-Specific Derived Metrics
            BrokerageRatio DECIMAL(10,4) COMMENT 'Brokerage Revenue / Total Operating Revenue (%)',
            AdvisoryRatio DECIMAL(10,4) COMMENT 'Advisory Revenue / Total Operating Revenue (%)',
            CustodyRatio DECIMAL(10,4) COMMENT 'Custody Revenue / Total Operating Revenue (%)',
            TradingRatio DECIMAL(10,4) COMMENT 'Trading Income / Total Operating Revenue (%)',
            
            -- Securities Asset Efficiency Metrics
            AssetTurnover DECIMAL(10,4) COMMENT 'Operating Revenue / Average Total Assets',
            FinancialAssetTurnover DECIMAL(10,4) COMMENT 'Trading Income / Average Financial Assets',
            RevenuePerAsset DECIMAL(10,4) COMMENT 'Total Revenue / Average Total Assets',
            
            -- Securities Profitability Metrics
            ROAA DECIMAL(10,4) COMMENT 'Return on Average Assets (%)',
            ROAE DECIMAL(10,4) COMMENT 'Return on Average Equity (%)',
            NetProfitMargin DECIMAL(10,4) COMMENT 'Net Profit Margin (%)',
            OperatingMargin DECIMAL(10,4) COMMENT 'Operating Margin (%)',
            
            -- Securities Capital & Leverage Metrics
            EquityRatio DECIMAL(10,4) COMMENT 'Equity Ratio (Equity/Assets) (%)',
            LeverageRatio DECIMAL(10,4) COMMENT 'Leverage Ratio (Assets/Equity)',
            CapitalAdequacyProxy DECIMAL(10,4) COMMENT 'Capital Adequacy Proxy (%)',
            
            -- Securities Efficiency Metrics
            CostToIncomeRatio DECIMAL(10,4) COMMENT 'Cost-to-Income Ratio (%)',
            OperatingExpenseRatio DECIMAL(10,4) COMMENT 'Operating Expenses / Operating Revenue (%)',
            FinancialExpenseRatio DECIMAL(10,4) COMMENT 'Financial Expenses / Operating Revenue (%)',
            
            -- Securities Growth Metrics
            RevenueGrowthQoQ DECIMAL(10,4) COMMENT 'Revenue Growth Quarter-over-Quarter (%)',
            BrokerageGrowthQoQ DECIMAL(10,4) COMMENT 'Brokerage Growth Quarter-over-Quarter (%)',
            AdvisoryGrowthQoQ DECIMAL(10,4) COMMENT 'Advisory Growth Quarter-over-Quarter (%)',
            AssetGrowthQoQ DECIMAL(10,4) COMMENT 'Asset Growth Quarter-over-Quarter (%)',
            
            -- Securities Cash Flow Metrics
            OperatingCashFlowMargin DECIMAL(10,4) COMMENT 'Operating Cash Flow Margin (%)',
            FCFMargin DECIMAL(10,4) COMMENT 'Free Cash Flow Margin (%)',
            DividendPayoutRatio DECIMAL(10,4) COMMENT 'Dividend Payout Ratio (%)',
            
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
            INDEX idx_brokerage_ratio (BrokerageRatio),
            INDEX idx_roaa (ROAA),
            INDEX idx_roae (ROAE),
            INDEX idx_asset_turnover (AssetTurnover)
        ) ENGINE=InnoDB 
        COMMENT='Securities sector intermediary calculations - TTM, averages, and derived metrics'
        DEFAULT CHARSET=utf8mb4 
        COLLATE=utf8mb4_unicode_ci;
        """
        
        try:
            with self.db_connection.cursor() as cursor:
                logger.info("Creating intermediary_calculations_securities table...")
                cursor.execute(create_table_sql)
                self.db_connection.commit()
                logger.info("✅ Securities intermediary table created successfully!")
                
                # Verify table creation
                cursor.execute("DESCRIBE intermediary_calculations_securities_cleaned")
                columns = cursor.fetchall()
                logger.info(f"✅ Table created with {len(columns)} columns")
                
                # Show sample columns
                logger.info("Key securities-specific columns:")
                key_columns = ['BrokerageRevenue_TTM', 'AvgFinancialAssets', 'BrokerageRatio', 'AssetTurnover', 'ROAA', 'ROAE']
                for col in columns:
                    if col['Field'] in key_columns:
                        comment = col.get('Comment', 'No comment')
                        logger.info(f"  • {col['Field']}: {col['Type']} - {comment}")
                
        except Exception as e:
            logger.error(f"Failed to create securities intermediary table: {e}")
            self.db_connection.rollback()
            raise
    
    def check_existing_table(self):
        """Check if table already exists and show its structure"""
        try:
            with self.db_connection.cursor() as cursor:
                cursor.execute("SHOW TABLES LIKE 'intermediary_calculations_securities_cleaned'")
                result = cursor.fetchone()
                
                if result:
                    logger.info("✅ Table already exists. Current structure:")
                    cursor.execute("DESCRIBE intermediary_calculations_securities_cleaned")
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
    logger.info("SECURITIES INTERMEDIARY TABLE CREATOR")
    logger.info(f"Start time: {datetime.now()}")
    logger.info("=" * 80)
    
    creator = SecuritiesIntermediaryTableCreator()
    
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
                cursor.execute("DROP TABLE IF EXISTS intermediary_calculations_securities_cleaned")
                creator.db_connection.commit()
                logger.info("Dropped existing table.")
    
    # Create the table
    creator.create_securities_intermediary_table()
    
    logger.info("=" * 80)
    logger.info("SECURITIES INTERMEDIARY TABLE CREATION COMPLETE")
    logger.info(f"End time: {datetime.now()}")
    logger.info("=" * 80)
    
    print("\n📊 **SECURITIES INTERMEDIARY TABLE CREATED**")
    print("🎯 Next steps:")
    print("   1. Create securities intermediary calculator")
    print("   2. Populate with historical data (2010-2025)")
    print("   3. Test with sample securities tickers (SSI, VCI, HCM)")
    print("   4. Update factor menu integration")

if __name__ == "__main__":
    main()