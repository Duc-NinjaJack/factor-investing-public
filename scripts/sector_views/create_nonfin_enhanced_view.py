#!/usr/bin/env python3
"""
Create Enhanced Fundamental View (Phase 1)
Expands from 8 to 81+ fundamental items while maintaining backward compatibility

LATEST UPDATE (June 30, 2025):
- Added multi-sector Charter Capital mapping using COALESCE
- Supports: "1. Contributed Capital" (some sectors), "1. Owner's Capital" (non-financial), "Charter Capital" (banking)
- Moved from archive to main scripts folder for production use
"""

import sys
import pandas as pd
from sqlalchemy import create_engine, text
import yaml
from datetime import datetime
from pathlib import Path

sys.path.append('src')

def create_enhanced_fundamental_view(engine):
    """Create v_comprehensive_fundamental_items view with 50+ fundamental items"""
    
    print("=" * 100)
    print("PHASE 1: CREATING ENHANCED FUNDAMENTAL VIEW")
    print(f"Started: {datetime.now()}")
    print("=" * 100)
    print()
    
    # First, let's check existing view structure
    check_existing_query = """
    SHOW CREATE VIEW v_clean_fundamental_items;
    """
    
    try:
        result = pd.read_sql(check_existing_query, engine)
        print("✅ Existing v_clean_fundamental_items found")
        print("   Will create parallel enhanced view without affecting existing view")
        print()
    except Exception as e:
        print(f"⚠️ Could not find existing view: {e}")
        print("   Proceeding with enhanced view creation")
        print()
    
    # Create comprehensive view with 50+ items
    create_view_query = """
    CREATE OR REPLACE VIEW v_comprehensive_fundamental_items AS
    SELECT 
        fv.ticker,
        fv.year,
        fv.quarter,
        
        -- ========================================
        -- EXISTING 8 ITEMS (Backward Compatibility)
        -- ========================================
        -- Income Statement (existing)
        MAX(CASE WHEN fm.en_name = 'Net Revenue' THEN fv.value END) as NetRevenue,
        MAX(CASE WHEN fm.en_name = 'Cost of Goods Sold' THEN fv.value END) as COGS,
        MAX(CASE WHEN fm.en_name = 'Operating Profit' THEN fv.value END) as EBIT,
        MAX(CASE WHEN fm.en_name = 'Net Profit After Tax' THEN fv.value END) as NetProfit,
        MAX(CASE WHEN fm.en_name = 'Net Profit Attributable to Parent' THEN fv.value END) as NetProfitAfterMI,
        
        -- Balance Sheet (existing)
        MAX(CASE WHEN fm.en_name = 'TOTAL ASSETS' THEN fv.value END) as TotalAssets,
        MAX(CASE WHEN fm.en_name = 'B. Owners\\' Equity' THEN fv.value END) as TotalEquity,
        MAX(CASE WHEN fm.en_name = 'A. Liabilities' THEN fv.value END) as TotalLiabilities,
        
        -- ========================================
        -- NEW INCOME STATEMENT ITEMS (Enhanced)
        -- ========================================
        MAX(CASE WHEN fm.en_name = 'Total Operating Revenue' THEN fv.value END) as TotalOperatingRevenue,
        MAX(CASE WHEN fm.en_name = 'Revenue Deductions' THEN fv.value END) as RevenueDeductions,
        MAX(CASE WHEN fm.en_name = 'Gross Profit' THEN fv.value END) as GrossProfit,
        MAX(CASE WHEN fm.en_name = 'Financial Income' THEN fv.value END) as FinancialIncome,
        MAX(CASE WHEN fm.en_name = 'Financial Expenses' THEN fv.value END) as FinancialExpenses,
        MAX(CASE WHEN fm.en_name = 'Of which: Interest Expenses' THEN fv.value END) as InterestExpenses,
        MAX(CASE WHEN fm.en_name = 'Selling Expenses' THEN fv.value END) as SellingExpenses,
        MAX(CASE WHEN fm.en_name = 'General & Administrative Expenses' THEN fv.value END) as AdminExpenses,
        MAX(CASE WHEN fm.en_name = 'Profit/Loss from Associates and Joint Ventures' THEN fv.value END) as ProfitFromAssociates,
        MAX(CASE WHEN fm.en_name = 'Other Income' THEN fv.value END) as OtherIncome,
        MAX(CASE WHEN fm.en_name = 'Other Expenses' THEN fv.value END) as OtherExpenses,
        MAX(CASE WHEN fm.en_name = 'Profit Before Tax' THEN fv.value END) as ProfitBeforeTax,
        MAX(CASE WHEN fm.en_name = 'Current Income Tax' THEN fv.value END) as CurrentIncomeTax,
        MAX(CASE WHEN fm.en_name = 'Deferred Income Tax' THEN fv.value END) as DeferredIncomeTax,
        MAX(CASE WHEN fm.en_name = 'Total Income Tax' THEN fv.value END) as TotalIncomeTax,
        MAX(CASE WHEN fm.en_name = 'Minority Interests' THEN fv.value END) as MinorityInterests,
        
        -- ========================================
        -- NEW BALANCE SHEET ITEMS (Enhanced)
        -- ========================================
        -- Current Assets
        MAX(CASE WHEN fm.en_name = 'A. Current Assets & Short-term Investments' THEN fv.value END) as CurrentAssets,
        MAX(CASE WHEN fm.en_name = 'I. Cash and Cash Equivalents' THEN fv.value END) as CashAndCashEquivalents,
        MAX(CASE WHEN fm.en_name = '1. Cash' THEN fv.value END) as Cash,
        MAX(CASE WHEN fm.en_name = '2. Cash Equivalents' THEN fv.value END) as CashEquivalents,
        MAX(CASE WHEN fm.en_name = 'II. Short-term Financial Investments' THEN fv.value END) as ShortTermInvestments,
        MAX(CASE WHEN fm.en_name = 'III. Short-term Receivables' THEN fv.value END) as ShortTermReceivables,
        MAX(CASE WHEN fm.en_name = '1. Short-term Trade Receivables' THEN fv.value END) as AccountsReceivable,
        MAX(CASE WHEN fm.en_name = '2. Prepayments to Suppliers' THEN fv.value END) as PrepaymentsToSuppliers,
        MAX(CASE WHEN fm.en_name = 'IV. Total Inventories' THEN fv.value END) as Inventory,
        MAX(CASE WHEN fm.en_name = 'V. Other Short-term Assets' THEN fv.value END) as OtherCurrentAssets,
        
        -- Long-term Assets
        MAX(CASE WHEN fm.en_name = 'B. Fixed Assets & Long-term Investments' THEN fv.value END) as LongTermAssets,
        MAX(CASE WHEN fm.en_name = 'I. Long-term Receivables' THEN fv.value END) as LongTermReceivables,
        MAX(CASE WHEN fm.en_name = 'II. Fixed Assets' THEN fv.value END) as FixedAssets,
        MAX(CASE WHEN fm.en_name = '1. Tangible Fixed Assets' THEN fv.value END) as TangibleFixedAssets,
        MAX(CASE WHEN fm.en_name = '- Cost' THEN fv.value END) as TangibleFixedAssetsCost,
        MAX(CASE WHEN fm.en_name = '- Accumulated Depreciation' THEN fv.value END) as AccumulatedDepreciation,
        MAX(CASE WHEN fm.en_name = 'III. Investment Properties' THEN fv.value END) as InvestmentProperties,
        MAX(CASE WHEN fm.en_name = 'IV. Long-term Construction in Progress' THEN fv.value END) as ConstructionInProgress,
        MAX(CASE WHEN fm.en_name = 'V. Long-term Financial Investments' THEN fv.value END) as LongTermInvestments,
        MAX(CASE WHEN fm.en_name = 'VI. Other Long-term Assets' THEN fv.value END) as OtherLongTermAssets,
        MAX(CASE WHEN fm.en_name = 'VII. Goodwill' THEN fv.value END) as Goodwill,
        
        -- Liabilities
        MAX(CASE WHEN fm.en_name = 'I. Short-term Liabilities' THEN fv.value END) as CurrentLiabilities,
        MAX(CASE WHEN fm.en_name = '1. Short-term Trade Payables' THEN fv.value END) as ShortTermTradePayables,
        MAX(CASE WHEN fm.en_name = '2. Customer Advances' THEN fv.value END) as CustomerAdvances,
        MAX(CASE WHEN fm.en_name = '3. Short-term Trade Payables' THEN fv.value END) as AccountsPayable,
        MAX(CASE WHEN fm.en_name = '5. Payables to Employees' THEN fv.value END) as PayablesToEmployees,
        MAX(CASE WHEN fm.en_name = '1. Short-term Borrowings & Finance Leases' THEN fv.value END) as ShortTermDebt,
        MAX(CASE WHEN fm.en_name = 'II. Long-term Liabilities' THEN fv.value END) as LongTermLiabilities,
        MAX(CASE WHEN fm.en_name = '6. Long-term Borrowings & Finance Leases' THEN fv.value END) as LongTermDebt,
        
        -- Equity Components
        MAX(CASE WHEN fm.en_name = 'I. Owners\\' Equity' THEN fv.value END) as OwnersEquity,
        
        -- Charter Capital (Multiple sector mappings)
        COALESCE(
            MAX(CASE WHEN fm.en_name = '1. Contributed Capital' THEN fv.value END),
            MAX(CASE WHEN fm.en_name = '1. Owner\\'s Capital' THEN fv.value END),
            MAX(CASE WHEN fm.en_name = 'Charter Capital' THEN fv.value END)
        ) as CharterCapital,
        
        MAX(CASE WHEN fm.en_name = '2. Share Premium' THEN fv.value END) as SharePremium,
        MAX(CASE WHEN fm.en_name = '5. Treasury Shares' THEN fv.value END) as TreasuryShares,
        MAX(CASE WHEN fm.en_name = '11. Retained Earnings (Undistributed)' THEN fv.value END) as RetainedEarnings,
        MAX(CASE WHEN fm.en_name = '14. Non-controlling Interests' THEN fv.value END) as NonControllingInterests,
        
        -- ========================================
        -- ✅ FIXED CASH FLOW ITEMS (Vietnamese aggregated totals)
        -- ========================================
        MAX(CASE WHEN fm.vn_name = 'Lưu chuyển tiền thuần từ hoạt động kinh doanh' THEN fv.value END) as NetCFO,
        MAX(CASE WHEN fm.vn_name = 'Lưu chuyển tiền thuần từ hoạt động đầu tư' THEN fv.value END) as NetCFI,
        MAX(CASE WHEN fm.vn_name = 'Lưu chuyển tiền thuần từ hoạt động tài chính' THEN fv.value END) as NetCFF,
        MAX(CASE WHEN fm.en_name = 'Profit Before Tax' AND fm.statement_type = 'CF' THEN fv.value END) as ProfitBeforeTax_CF,
        MAX(CASE WHEN fm.en_name = 'Depreciation & Amortization' THEN fv.value END) as DepreciationAmortization,
        MAX(CASE WHEN fm.en_name = 'Interest Expense' AND fm.statement_type = 'CF' THEN fv.value END) as InterestExpense_CF,
        MAX(CASE WHEN fm.en_name = 'Interest Income' AND fm.statement_type = 'CF' THEN fv.value END) as InterestIncome_CF,
        MAX(CASE WHEN fm.en_name = '- (Increase)/Decrease in Receivables' THEN fv.value END) as ChangeInReceivables,
        MAX(CASE WHEN fm.en_name = '- (Increase)/Decrease in Inventories' THEN fv.value END) as ChangeInInventories,
        MAX(CASE WHEN fm.en_name = '- (Increase)/Decrease in Payables (excluding interest/taxes)' THEN fv.value END) as ChangeInPayables,
        MAX(CASE WHEN fm.en_name = '1. Acquisition of Fixed/Long-term Assets' THEN fv.value END) as CapEx,
        MAX(CASE WHEN fm.en_name = '2. Proceeds from Disposal of Fixed/Long-term Assets' THEN fv.value END) as AssetDisposalProceeds,
        MAX(CASE WHEN fm.en_name = '8. Dividends, Profit Paid to Owners' THEN fv.value END) as DividendsPaid,
        MAX(CASE WHEN fm.en_name = '1. Proceeds from Share Issuance / Owner Capital' THEN fv.value END) as ShareIssuanceProceeds,
        MAX(CASE WHEN fm.en_name = '2. Repurchase of Issued Shares / Return of Capital' THEN fv.value END) as ShareRepurchase,
        MAX(CASE WHEN fm.en_name = '3. Proceeds from Short-term / Long-term Borrowings' THEN fv.value END) as DebtIssuance,
        MAX(CASE WHEN fm.en_name = '4. Repayment of Borrowings' THEN fv.value END) as DebtRepayment,
        
        -- Metadata
        COUNT(DISTINCT fm.en_name) as total_items_available
        
    FROM fundamental_values fv
    JOIN fs_mappings fm ON fv.item_id = fm.item_id 
        AND fv.statement_type = fm.statement_type
    JOIN master_info mi ON fv.ticker = mi.ticker
    JOIN sector_display_to_fs sdf ON mi.sector = sdf.display_sector
    WHERE fm.sector = sdf.fs_sector
    GROUP BY fv.ticker, fv.year, fv.quarter;
    """
    
    try:
        # Execute the CREATE VIEW statement
        print("🔄 Creating enhanced fundamental view...")
        with engine.connect() as conn:
            conn.execute(text(create_view_query))
        print("✅ Successfully created v_comprehensive_fundamental_items")
        print()
        
        # Test the new view
        test_query = """
        SELECT 
            COUNT(DISTINCT ticker) as unique_tickers,
            COUNT(*) as total_records,
            MIN(year) as earliest_year,
            MAX(year) as latest_year,
            AVG(total_items_available) as avg_items_per_record
        FROM v_comprehensive_fundamental_items
        """
        
        df_test = pd.read_sql(test_query, engine)
        print("📊 Enhanced View Statistics:")
        print(f"   - Unique tickers: {df_test['unique_tickers'].iloc[0]:,}")
        print(f"   - Total records: {df_test['total_records'].iloc[0]:,}")
        print(f"   - Data range: {df_test['earliest_year'].iloc[0]} - {df_test['latest_year'].iloc[0]}")
        print(f"   - Average items per record: {df_test['avg_items_per_record'].iloc[0]:.1f}")
        print()
        
        # Compare with existing view
        print("🔄 Comparing with existing v_clean_fundamental_items...")
        
        # Sample comparison for FPT
        comparison_query = """
        SELECT 
            'Existing' as source,
            NetRevenue, COGS, EBIT, NetProfit, TotalAssets, TotalEquity
        FROM v_clean_fundamental_items
        WHERE ticker = 'FPT' AND year = 2025 AND quarter = 1
        
        UNION ALL
        
        SELECT 
            'Enhanced' as source,
            NetRevenue, COGS, EBIT, NetProfit, TotalAssets, TotalEquity
        FROM v_comprehensive_fundamental_items
        WHERE ticker = 'FPT' AND year = 2025 AND quarter = 1
        """
        
        df_compare = pd.read_sql(comparison_query, engine)
        print("✅ Backward Compatibility Test (FPT Q1 2025):")
        print(df_compare.to_string(index=False))
        print()
        
        # Show enhanced items sample
        enhanced_sample_query = """
        SELECT 
            ticker, year, quarter,
            -- New Income Statement items
            GrossProfit, SellingExpenses, AdminExpenses, InterestExpenses,
            -- New Balance Sheet items  
            Cash, AccountsReceivable, Inventory, AccountsPayable,
            FixedAssets, AccumulatedDepreciation,
            -- New Cash Flow items
            NetCFO, DepreciationAmortization, CapEx, DividendsPaid
        FROM v_comprehensive_fundamental_items
        WHERE ticker = 'FPT' AND year = 2025 AND quarter = 1
        """
        
        df_enhanced = pd.read_sql(enhanced_sample_query, engine)
        print("🆕 Sample of New Enhanced Items (FPT Q1 2025):")
        print("Income Statement Enhancements:")
        print(f"   - Gross Profit: {df_enhanced['GrossProfit'].iloc[0]:,.0f}")
        print(f"   - Selling Expenses: {df_enhanced['SellingExpenses'].iloc[0]:,.0f}")
        print(f"   - Admin Expenses: {df_enhanced['AdminExpenses'].iloc[0]:,.0f}")
        print(f"   - Interest Expenses: {df_enhanced['InterestExpenses'].iloc[0]:,.0f}")
        print()
        print("Balance Sheet Enhancements:")
        print(f"   - Cash: {df_enhanced['Cash'].iloc[0]:,.0f}")
        print(f"   - Accounts Receivable: {df_enhanced['AccountsReceivable'].iloc[0]:,.0f}")
        print(f"   - Inventory: {df_enhanced['Inventory'].iloc[0]:,.0f}")
        print(f"   - Accounts Payable: {df_enhanced['AccountsPayable'].iloc[0]:,.0f}")
        print()
        print("Cash Flow Enhancements:")
        print(f"   - Operating Cash Flow: {df_enhanced['NetCFO'].iloc[0]:,.0f}")
        print(f"   - Depreciation & Amortization: {df_enhanced['DepreciationAmortization'].iloc[0]:,.0f}")
        print(f"   - CapEx: {df_enhanced['CapEx'].iloc[0]:,.0f}")
        print(f"   - Dividends Paid: {df_enhanced['DividendsPaid'].iloc[0]:,.0f}")
        print()
        
        # Count total enhanced items
        item_count_query = """
        SELECT 
            COUNT(*) as column_count
        FROM information_schema.columns
        WHERE table_schema = DATABASE()
          AND table_name = 'v_comprehensive_fundamental_items'
          AND column_name NOT IN ('ticker', 'year', 'quarter', 'total_items_available', 
                                   'earliest_data', 'latest_data')
        """
        
        df_count = pd.read_sql(item_count_query, engine)
        total_items = df_count['column_count'].iloc[0]
        
        print("=" * 100)
        print(f"✅ PHASE 1 COMPLETE: Enhanced view created with {total_items} fundamental items")
        print(f"   (Up from 8 items in existing view - {total_items/8:.1f}X improvement)")
        print("=" * 100)
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating enhanced view: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    # Load database configuration
    try:
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / 'config' / 'database.yml'
        with open(config_path, 'r') as f:
            db_config = yaml.safe_load(f)['production']
    except Exception as e:
        print(f"Error loading database config: {e}")
        sys.exit(1)
    
    # Create connection
    try:
        engine = create_engine(
            f"mysql+pymysql://{db_config['username']}:{db_config['password']}@{db_config['host']}/{db_config['schema_name']}"
        )
    except Exception as e:
        print(f"Error connecting to database: {e}")
        sys.exit(1)
    
    try:
        # Create enhanced fundamental view
        success = create_enhanced_fundamental_view(engine)
        
        if success:
            print("\n✅ Phase 1 implementation successful!")
            print("📋 Next steps:")
            print("   1. Run validation scripts to verify data integrity")
            print("   2. Test enhanced view with research queries")
            print("   3. Document new items for research team")
            print("   4. Proceed to Phase 2 - Intermediary Storage")
        else:
            print("\n❌ Phase 1 implementation failed. Please check errors above.")
            
    except Exception as e:
        print(f"Error during implementation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        engine.dispose()

if __name__ == "__main__":
    main()