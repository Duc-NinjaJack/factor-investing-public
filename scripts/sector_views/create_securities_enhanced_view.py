#!/usr/bin/env python3
"""
Create Complete Securities Enhanced View - All Securities Items
==============================================================
Creates v_complete_securities_fundamentals view with ALL securities fundamental items
discovered in the comprehensive analysis. This includes:

- Core P&L items (including securities-specific revenue breakdowns)
- Balance Sheet items with securities-specific assets/liabilities
- Cash Flow items (including dividend payments for strategies)
- Securities business metrics and client asset tracking

Based on comprehensive securities database analysis using validated SSI/VCI data.
"""

import pymysql
import yaml
from pathlib import Path

def connect_to_database():
    """Create database connection"""
    try:
        config_path = Path(__file__).parent.parent.parent / 'config' / 'database.yml'
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
        print(f"❌ Database connection failed: {e}")
        return None

def create_complete_securities_view():
    """Create comprehensive securities view with all fundamental items"""
    
    # Complete securities view SQL based on validated SSI/VCI investigation
    securities_view_sql = """
    CREATE OR REPLACE VIEW v_complete_securities_fundamentals AS
    SELECT 
        fv.ticker,
        fv.year,
        fv.quarter,
        
        -- === PROFIT & LOSS STATEMENT (Securities-specific) ===
        
        -- Trading Income (Major Revenue Source - Items 101-105)
        MAX(CASE WHEN fv.item_id = 101 AND fv.statement_type = 'PL' THEN fv.value END) AS TradingGainFVTPL,
        MAX(CASE WHEN fv.item_id = 102 AND fv.statement_type = 'PL' THEN fv.value END) AS TradingGainHTM,
        MAX(CASE WHEN fv.item_id = 103 AND fv.statement_type = 'PL' THEN fv.value END) AS TradingGainLoans,
        MAX(CASE WHEN fv.item_id = 104 AND fv.statement_type = 'PL' THEN fv.value END) AS TradingGainAFS,
        MAX(CASE WHEN fv.item_id = 105 AND fv.statement_type = 'PL' THEN fv.value END) AS TradingGainDerivatives,
        
        -- Core P&L Structure (Items 1-14) 
        MAX(CASE WHEN fv.item_id = 1 AND fv.statement_type = 'PL' THEN fv.value END) AS OperatingRevenue,
        MAX(CASE WHEN fv.item_id = 2 AND fv.statement_type = 'PL' THEN fv.value END) AS OperatingExpenses,
        MAX(CASE WHEN fv.item_id = 3 AND fv.statement_type = 'PL' THEN fv.value END) AS FinancialIncome,
        MAX(CASE WHEN fv.item_id = 4 AND fv.statement_type = 'PL' THEN fv.value END) AS FinancialExpenses,
        MAX(CASE WHEN fv.item_id = 5 AND fv.statement_type = 'PL' THEN fv.value END) AS SellingExpenses,
        MAX(CASE WHEN fv.item_id = 6 AND fv.statement_type = 'PL' THEN fv.value END) AS ManagementExpenses,
        MAX(CASE WHEN fv.item_id = 7 AND fv.statement_type = 'PL' THEN fv.value END) AS OperatingResult,
        MAX(CASE WHEN fv.item_id = 8 AND fv.statement_type = 'PL' THEN fv.value END) AS OtherIncomeExpenses,
        MAX(CASE WHEN fv.item_id = 9 AND fv.statement_type = 'PL' THEN fv.value END) AS ProfitBeforeTax,
        MAX(CASE WHEN fv.item_id = 10 AND fv.statement_type = 'PL' THEN fv.value END) AS IncomeTaxExpense,
        MAX(CASE WHEN fv.item_id = 11 AND fv.statement_type = 'PL' THEN fv.value END) AS ProfitAfterTax,
        MAX(CASE WHEN fv.item_id = 12 AND fv.statement_type = 'PL' THEN fv.value END) AS ComprehensiveIncome,
        MAX(CASE WHEN fv.item_id = 13 AND fv.statement_type = 'PL' THEN fv.value END) AS TotalComprehensiveIncome,
        MAX(CASE WHEN fv.item_id = 14 AND fv.statement_type = 'PL' THEN fv.value END) AS EarningsPerShare,
        
        -- Securities-Specific Revenue Detail (Level 2) - VALIDATED FROM SSI/VCI
        MAX(CASE WHEN fv.item_id = 106 AND fv.statement_type = 'PL' THEN fv.value END) AS BrokerageRevenue,
        MAX(CASE WHEN fv.item_id = 107 AND fv.statement_type = 'PL' THEN fv.value END) AS UnderwritingRevenue,
        MAX(CASE WHEN fv.item_id = 108 AND fv.statement_type = 'PL' THEN fv.value END) AS AdvisoryRevenue,
        MAX(CASE WHEN fv.item_id = 109 AND fv.statement_type = 'PL' THEN fv.value END) AS EntrustedAuctionRevenue,
        MAX(CASE WHEN fv.item_id = 110 AND fv.statement_type = 'PL' THEN fv.value END) AS CustodyServiceRevenue,
        MAX(CASE WHEN fv.item_id = 111 AND fv.statement_type = 'PL' THEN fv.value END) AS OtherOperatingIncome,
        MAX(CASE WHEN fv.item_id = 112 AND fv.statement_type = 'PL' THEN fv.value END) AS TotalOperatingRevenue,
        
        -- Revenue Detail (Level 3)
        MAX(CASE WHEN fv.item_id = 101 AND fv.statement_type = 'PL' THEN fv.value END) AS GainFinancialAssetsFVTPL,
        MAX(CASE WHEN fv.item_id = 102 AND fv.statement_type = 'PL' THEN fv.value END) AS GainHeldToMaturityInvestments,
        MAX(CASE WHEN fv.item_id = 103 AND fv.statement_type = 'PL' THEN fv.value END) AS GainLoansReceivables,
        MAX(CASE WHEN fv.item_id = 104 AND fv.statement_type = 'PL' THEN fv.value END) AS GainAvailableForSaleFinancial,
        MAX(CASE WHEN fv.item_id = 105 AND fv.statement_type = 'PL' THEN fv.value END) AS GainHedgingDerivatives,
        
        -- Expense Detail (Level 3)
        MAX(CASE WHEN fv.item_id = 201 AND fv.statement_type = 'PL' THEN fv.value END) AS LossFinancialAssetsFVTPL,
        MAX(CASE WHEN fv.item_id = 202 AND fv.statement_type = 'PL' THEN fv.value END) AS LossHeldToMaturityInvestments,
        MAX(CASE WHEN fv.item_id = 203 AND fv.statement_type = 'PL' THEN fv.value END) AS InterestExpenseLossLoans,
        MAX(CASE WHEN fv.item_id = 207 AND fv.statement_type = 'PL' THEN fv.value END) AS BrokerageExpenses,
        MAX(CASE WHEN fv.item_id = 209 AND fv.statement_type = 'PL' THEN fv.value END) AS AdvisoryExpenses,
        MAX(CASE WHEN fv.item_id = 211 AND fv.statement_type = 'PL' THEN fv.value END) AS CustodyExpenses,
        
        -- Other Income/Expenses Detail
        MAX(CASE WHEN fv.item_id = 301 AND fv.statement_type = 'PL' THEN fv.value END) AS RealizedUnrealizedForexGains,
        MAX(CASE WHEN fv.item_id = 302 AND fv.statement_type = 'PL' THEN fv.value END) AS DividendInterestIncome,
        MAX(CASE WHEN fv.item_id = 305 AND fv.statement_type = 'PL' THEN fv.value END) AS TotalFinancialIncome,
        MAX(CASE WHEN fv.item_id = 402 AND fv.statement_type = 'PL' THEN fv.value END) AS InterestExpense,
        MAX(CASE WHEN fv.item_id = 405 AND fv.statement_type = 'PL' THEN fv.value END) AS TotalFinancialExpenses,
        MAX(CASE WHEN fv.item_id = 802 AND fv.statement_type = 'PL' THEN fv.value END) AS OtherExpenses,
        
        -- === BALANCE SHEET (Securities-specific) ===
        
        -- Assets Overview (Level 1)
        MAX(CASE WHEN fv.item_id = 2 AND fv.statement_type = 'BS' THEN fv.value END) AS TotalAssets,
        MAX(CASE WHEN fv.item_id = 4 AND fv.statement_type = 'BS' THEN fv.value END) AS OwnersEquity,
        MAX(CASE WHEN fv.item_id = 5 AND fv.statement_type = 'BS' THEN fv.value END) AS TotalLiabilitiesAndEquity,
        
        -- Asset Categories (Level 2)
        MAX(CASE WHEN fv.item_id = 101 AND fv.statement_type = 'BS' THEN fv.value END) AS CurrentAssets,
        MAX(CASE WHEN fv.item_id = 102 AND fv.statement_type = 'BS' THEN fv.value END) AS NonCurrentAssets,
        
        -- Financial Assets Detail (Level 3) - Critical for securities firms
        MAX(CASE WHEN fv.item_id = 10101 AND fv.statement_type = 'BS' THEN fv.value END) AS FinancialAssets,
        MAX(CASE WHEN fv.item_id = 10102 AND fv.statement_type = 'BS' THEN fv.value END) AS OtherCurrentAssets,
        MAX(CASE WHEN fv.item_id = 1010101 AND fv.statement_type = 'BS' THEN fv.value END) AS CashAndCashEquivalents,
        MAX(CASE WHEN fv.item_id = 1010102 AND fv.statement_type = 'BS' THEN fv.value END) AS FinancialAssetsFVTPL,
        MAX(CASE WHEN fv.item_id = 1010103 AND fv.statement_type = 'BS' THEN fv.value END) AS HeldToMaturityInvestments,
        MAX(CASE WHEN fv.item_id = 1010104 AND fv.statement_type = 'BS' THEN fv.value END) AS LoanReceivables,
        MAX(CASE WHEN fv.item_id = 1010105 AND fv.statement_type = 'BS' THEN fv.value END) AS AvailableForSaleFinancial,
        MAX(CASE WHEN fv.item_id = 1010106 AND fv.statement_type = 'BS' THEN fv.value END) AS ProvisionImpairmentFinancial,
        MAX(CASE WHEN fv.item_id = 1010107 AND fv.statement_type = 'BS' THEN fv.value END) AS Receivables,
        
        -- Securities Business Receivables (Level 4) - Client business tracking
        MAX(CASE WHEN fv.item_id = 1010109 AND fv.statement_type = 'BS' THEN fv.value END) AS SecuritiesServicesReceivables,
        MAX(CASE WHEN fv.item_id = 1010110 AND fv.statement_type = 'BS' THEN fv.value END) AS IntraCompanyReceivables,
        MAX(CASE WHEN fv.item_id = 1010112 AND fv.statement_type = 'BS' THEN fv.value END) AS OtherReceivables,
        MAX(CASE WHEN fv.item_id = 1010113 AND fv.statement_type = 'BS' THEN fv.value END) AS ProvisionDoubtfulReceivables,
        
        -- Non-Current Assets Detail
        MAX(CASE WHEN fv.item_id = 10201 AND fv.statement_type = 'BS' THEN fv.value END) AS LongTermFinancialAssets,
        MAX(CASE WHEN fv.item_id = 10202 AND fv.statement_type = 'BS' THEN fv.value END) AS FixedAssets,
        MAX(CASE WHEN fv.item_id = 10205 AND fv.statement_type = 'BS' THEN fv.value END) AS OtherNonCurrentAssets,
        
        -- Liability Categories (Level 2)
        MAX(CASE WHEN fv.item_id = 301 AND fv.statement_type = 'BS' THEN fv.value END) AS ShortTermLiabilities,
        MAX(CASE WHEN fv.item_id = 302 AND fv.statement_type = 'BS' THEN fv.value END) AS LongTermLiabilities,
        
        -- Securities-Specific Liabilities (Level 3)
        MAX(CASE WHEN fv.item_id = 30101 AND fv.statement_type = 'BS' THEN fv.value END) AS ShortTermBorrowingsFinancial,
        MAX(CASE WHEN fv.item_id = 30106 AND fv.statement_type = 'BS' THEN fv.value END) AS PayablesSecuritiesTrading,
        MAX(CASE WHEN fv.item_id = 30108 AND fv.statement_type = 'BS' THEN fv.value END) AS ShortTermTradePayables,
        MAX(CASE WHEN fv.item_id = 30110 AND fv.statement_type = 'BS' THEN fv.value END) AS TaxesDuesGovernment,
        MAX(CASE WHEN fv.item_id = 30111 AND fv.statement_type = 'BS' THEN fv.value END) AS PayablesToEmployees,
        MAX(CASE WHEN fv.item_id = 30114 AND fv.statement_type = 'BS' THEN fv.value END) AS IntraCompanyPayables,
        MAX(CASE WHEN fv.item_id = 30119 AND fv.statement_type = 'BS' THEN fv.value END) AS BonusWelfareFund,
        
        -- Equity Detail (Level 3) - Critical for capital adequacy
        MAX(CASE WHEN fv.item_id = 401 AND fv.statement_type = 'BS' THEN fv.value END) AS OwnersEquityDetail,
        MAX(CASE WHEN fv.item_id = 40101 AND fv.statement_type = 'BS' THEN fv.value END) AS OwnerCapital,
        MAX(CASE WHEN fv.item_id = 40102 AND fv.statement_type = 'BS' THEN fv.value END) AS FairValueRevaluationSurplus,
        MAX(CASE WHEN fv.item_id = 40107 AND fv.statement_type = 'BS' THEN fv.value END) AS RetainedEarnings,
        
        -- Capital Detail (Level 4) - INCLUDING CHARTER CAPITAL
        MAX(CASE WHEN fv.item_id = 4010101 AND fv.statement_type = 'BS' THEN fv.value END) AS CharterCapital,
        MAX(CASE WHEN fv.item_id = 4010102 AND fv.statement_type = 'BS' THEN fv.value END) AS SharePremium,
        MAX(CASE WHEN fv.item_id = 4010105 AND fv.statement_type = 'BS' THEN fv.value END) AS TreasuryShares,
        MAX(CASE WHEN fv.item_id = 4010701 AND fv.statement_type = 'BS' THEN fv.value END) AS RealizedRetainedEarnings,
        MAX(CASE WHEN fv.item_id = 4010702 AND fv.statement_type = 'BS' THEN fv.value END) AS UnrealizedRetainedEarnings,
        
        -- === CASH FLOW STATEMENT (focus on key items) ===
        
        -- Operating Activities (Major)
        MAX(CASE WHEN fv.item_id = 1 AND fv.statement_type = 'CF' THEN fv.value END) AS NetCashFlowFromOperatingActivities,
        MAX(CASE WHEN fv.item_id = 2 AND fv.statement_type = 'CF' THEN fv.value END) AS NetCashFlowFromInvestingActivities,
        MAX(CASE WHEN fv.item_id = 3 AND fv.statement_type = 'CF' THEN fv.value END) AS NetCashFlowFromFinancingActivities,
        MAX(CASE WHEN fv.item_id = 4 AND fv.statement_type = 'CF' THEN fv.value END) AS NetIncreaseDecreaseInCash,
        MAX(CASE WHEN fv.item_id = 5 AND fv.statement_type = 'CF' THEN fv.value END) AS CashAtBeginningOfPeriod,
        MAX(CASE WHEN fv.item_id = 6 AND fv.statement_type = 'CF' THEN fv.value END) AS CashAtEndOfPeriod,
        
        -- Operating Details
        MAX(CASE WHEN fv.item_id = 101 AND fv.statement_type = 'CF' THEN fv.value END) AS ProfitBeforeTaxCF,
        MAX(CASE WHEN fv.item_id = 102 AND fv.statement_type = 'CF' THEN fv.value END) AS Adjustments,
        MAX(CASE WHEN fv.item_id = 107 AND fv.statement_type = 'CF' THEN fv.value END) AS NetCashFlowOperating,
        
        -- Investment Activities
        MAX(CASE WHEN fv.item_id = 201 AND fv.statement_type = 'CF' THEN fv.value END) AS CapitalExpenditures,
        MAX(CASE WHEN fv.item_id = 206 AND fv.statement_type = 'CF' THEN fv.value END) AS NetCashFlowInvesting,
        
        -- Financing Activities (Critical for dividend strategies)
        MAX(CASE WHEN fv.item_id = 301 AND fv.statement_type = 'CF' THEN fv.value END) AS ProceedsFromShareIssuance,
        MAX(CASE WHEN fv.item_id = 303 AND fv.statement_type = 'CF' THEN fv.value END) AS PrincipalBorrowings,
        MAX(CASE WHEN fv.item_id = 304 AND fv.statement_type = 'CF' THEN fv.value END) AS RepaymentBorrowings,
        MAX(CASE WHEN fv.item_id = 306 AND fv.statement_type = 'CF' THEN fv.value END) AS DividendsPaidToOwners,
        MAX(CASE WHEN fv.item_id = 307 AND fv.statement_type = 'CF' THEN fv.value END) AS NetCashFlowFinancing

    FROM fundamental_values fv
    INNER JOIN master_info mi ON fv.ticker = mi.ticker
    WHERE mi.sector = 'Securities'
    GROUP BY fv.ticker, fv.year, fv.quarter
    ORDER BY fv.ticker, fv.year DESC, fv.quarter DESC;
    """
    
    conn = connect_to_database()
    if not conn:
        print("❌ Failed to connect to database")
        return False
    
    try:
        with conn.cursor() as cursor:
            cursor.execute(securities_view_sql)
            conn.commit()
            print("✅ Complete securities enhanced view created successfully!")
            print("📊 View: v_complete_securities_fundamentals")
            print("🏛️ Scope: ALL securities fundamental items")
            print("💡 Coverage: Securities-specific revenue streams and client asset tracking")
            
        # Test the view with SSI - show key ratios
        test_query = """
        SELECT ticker, year, quarter, 
               BrokerageRevenue/1e9 as Brokerage_B,
               AdvisoryRevenue/1e9 as Advisory_B,
               CustodyServiceRevenue/1e9 as Custody_B,
               TotalAssets/1e9 as Assets_B,
               OwnersEquity/1e9 as Equity_B,
               FinancialAssetsFVTPL/1e9 as Trading_B,
               -- Key securities ratios
               ROUND(BrokerageRevenue*100.0/TotalOperatingRevenue, 1) as BrokerageRatio_Pct,
               ROUND(OwnersEquity*100.0/TotalAssets, 1) as EquityRatio_Pct
        FROM v_complete_securities_fundamentals 
        WHERE ticker = 'SSI'
        ORDER BY year DESC, quarter DESC 
        LIMIT 5
        """
        
        with conn.cursor() as cursor:
            cursor.execute(test_query)
            results = cursor.fetchall()
            if results:
                print("\n🧪 Complete Securities View Test Results (SSI):")
                print("Yr Q  | Brok | Adv  | Cust | Assets| Equity| Trad | Brok%| Eq% ")
                print("-" * 70)
                for row in results:
                    brok = row[3] if row[3] else 0
                    adv = row[4] if row[4] else 0  
                    cust = row[5] if row[5] else 0
                    assets = row[6] if row[6] else 0
                    equity = row[7] if row[7] else 0
                    trading = row[8] if row[8] else 0
                    brok_pct = row[9] if row[9] else 0
                    eq_pct = row[10] if row[10] else 0
                    print(f"{row[1]} Q{row[2]} | {brok:4.0f} | {adv:4.0f} | {cust:4.0f} | "
                          f"{assets:6.0f}| {equity:6.0f}| {trading:4.0f} | {brok_pct:5.1f}| {eq_pct:3.0f}")
            else:
                print("\n⚠️  No SSI data found")
                
        return True
        
    except Exception as e:
        print(f"❌ Error creating complete securities view: {e}")
        return False
    finally:
        conn.close()

def main():
    """Main execution"""
    print("🏛️ Creating Complete Securities Enhanced View...")
    print("📋 Based on comprehensive securities analysis with SSI/VCI validation")
    print("🎯 Securities-specific revenue streams and client asset tracking")
    print("💰 Includes dividend items for factor strategies")
    print()
    
    success = create_complete_securities_view()
    
    if success:
        print("\n✅ Complete securities infrastructure ready!")
        print("\n📈 **Available Securities Factor Calculations**:")
        print("   **Revenue**: Brokerage ratios, Advisory revenue, Custody fees")
        print("   **Assets**: Trading portfolio, Client receivables, Financial assets")
        print("   **Profitability**: ROA, ROE, Revenue per asset")  
        print("   **Capital**: Equity ratio, Capital adequacy")
        print("   **Dividend**: Dividend yield, payout ratios")
        print("\n📝 Next steps:")
        print("   1. Create complete securities extraction script")
        print("   2. Validate against multiple securities firms (SSI, VCI, HCM)")
        print("   3. Implement securities factor calculations")
        print("   4. Build securities-specific analysis reports")
    else:
        print("\n❌ Complete securities view creation failed")

if __name__ == "__main__":
    main()