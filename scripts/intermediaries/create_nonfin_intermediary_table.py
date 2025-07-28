#!/usr/bin/env python3
"""
Non-Financial Intermediary Table Creator (Phase 2)
=================================================
Creates intermediary_calculations_enhanced table for non-financial sectors.
Store pre-calculated TTM values, 5-point balance sheet averages, and working capital metrics.

This table serves all 21 non-financial sectors (667 tickers):
Technology, Real Estate, Construction, Food & Beverage, etc.

Author: Duc Nguyen (Aureus Sigma Capital)
Date: June 29, 2025 (moved to proper location July 3, 2025)
"""

import sys
import pandas as pd
from sqlalchemy import create_engine, text
import yaml
from datetime import datetime

sys.path.append('src')

def create_nonfin_intermediary_table(engine):
    """Create intermediary_calculations_enhanced table for non-financial sectors"""
    
    print("=" * 100)
    print("PHASE 2: CREATING NON-FINANCIAL INTERMEDIARY CALCULATIONS TABLE")
    print(f"Started: {datetime.now()}")
    print("=" * 100)
    print()
    
    # First check if table already exists
    check_table_query = """
    SELECT COUNT(*) as table_exists
    FROM information_schema.tables
    WHERE table_schema = DATABASE()
      AND table_name = 'intermediary_calculations_enhanced'
    """
    
    df_check = pd.read_sql(check_table_query, engine)
    if df_check['table_exists'].iloc[0] > 0:
        print("⚠️ Table intermediary_calculations_enhanced already exists")
        print("   Dropping existing table...")
        
        with engine.connect() as conn:
            conn.execute(text("DROP TABLE intermediary_calculations_enhanced"))
            conn.commit()
        print("✅ Existing table dropped")
    
    # Create comprehensive intermediary table
    create_table_query = """
    CREATE TABLE intermediary_calculations_enhanced (
        -- Primary Keys
        ticker CHAR(10) NOT NULL,
        year INT NOT NULL,
        quarter INT NOT NULL,
        calc_date DATE NOT NULL,
        
        -- ========================================
        -- TTM FLOW METRICS (Income Statement)
        -- ========================================
        Revenue_TTM DECIMAL(30,2),
        COGS_TTM DECIMAL(30,2),
        GrossProfit_TTM DECIMAL(30,2),
        SellingExpenses_TTM DECIMAL(30,2),
        AdminExpenses_TTM DECIMAL(30,2),
        OperatingExpenses_TTM DECIMAL(30,2),  -- Selling + Admin
        FinancialIncome_TTM DECIMAL(30,2),
        FinancialExpenses_TTM DECIMAL(30,2),
        InterestExpense_TTM DECIMAL(30,2),
        EBITDA_TTM DECIMAL(30,2),
        EBIT_TTM DECIMAL(30,2),
        ProfitBeforeTax_TTM DECIMAL(30,2),
        CurrentTax_TTM DECIMAL(30,2),
        DeferredTax_TTM DECIMAL(30,2),
        TotalTax_TTM DECIMAL(30,2),
        NetProfit_TTM DECIMAL(30,2),
        NetProfitAfterMI_TTM DECIMAL(30,2),
        
        -- ========================================
        -- TTM FLOW METRICS (Cash Flow)
        -- ========================================
        NetCFO_TTM DECIMAL(30,2),
        NetCFI_TTM DECIMAL(30,2),
        NetCFF_TTM DECIMAL(30,2),
        DepreciationAmortization_TTM DECIMAL(30,2),
        CapEx_TTM DECIMAL(30,2),
        FCF_TTM DECIMAL(30,2),  -- NetCFO_TTM - CapEx_TTM
        DividendsPaid_TTM DECIMAL(30,2),
        ShareIssuance_TTM DECIMAL(30,2),
        ShareRepurchase_TTM DECIMAL(30,2),
        DebtIssuance_TTM DECIMAL(30,2),
        DebtRepayment_TTM DECIMAL(30,2),
        
        -- ========================================
        -- BALANCE SHEET AVERAGES (5-point for non-financial)
        -- ========================================
        AvgTotalAssets DECIMAL(30,2),
        AvgTotalEquity DECIMAL(30,2),
        AvgTotalLiabilities DECIMAL(30,2),
        AvgCurrentAssets DECIMAL(30,2),
        AvgCurrentLiabilities DECIMAL(30,2),
        AvgWorkingCapital DECIMAL(30,2),
        AvgCash DECIMAL(30,2),
        AvgCashEquivalents DECIMAL(30,2),
        AvgShortTermInvestments DECIMAL(30,2),
        AvgInventory DECIMAL(30,2),
        AvgReceivables DECIMAL(30,2),
        AvgPayables DECIMAL(30,2),
        AvgFixedAssets DECIMAL(30,2),
        AvgTangibleAssets DECIMAL(30,2),
        AvgIntangibleAssets DECIMAL(30,2),
        AvgGoodwill DECIMAL(30,2),
        AvgShortTermDebt DECIMAL(30,2),
        AvgLongTermDebt DECIMAL(30,2),
        AvgTotalDebt DECIMAL(30,2),
        AvgNetDebt DECIMAL(30,2),  -- Total Debt - Cash
        AvgRetainedEarnings DECIMAL(30,2),
        AvgInvestedCapital DECIMAL(30,2),  -- Equity + Debt - Cash
        
        -- ========================================
        -- WORKING CAPITAL METRICS
        -- ========================================
        DSO DECIMAL(10,2),  -- Days Sales Outstanding
        DIO DECIMAL(10,2),  -- Days Inventory Outstanding
        DPO DECIMAL(10,2),  -- Days Payables Outstanding
        CCC DECIMAL(10,2),  -- Cash Conversion Cycle (DSO + DIO - DPO)
        
        -- ========================================
        -- PER SHARE METRICS (when available)
        -- ========================================
        SharesOutstanding DECIMAL(20,0),
        EPS_TTM DECIMAL(10,4),
        BookValuePerShare DECIMAL(10,4),
        TangibleBookValuePerShare DECIMAL(10,4),
        SalesPerShare_TTM DECIMAL(10,4),
        CFOPerShare_TTM DECIMAL(10,4),
        FCFPerShare_TTM DECIMAL(10,4),
        DividendPerShare_TTM DECIMAL(10,4),
        
        -- ========================================
        -- CALCULATION METADATA
        -- ========================================
        quarters_available INT,  -- Number of quarters used in TTM calc
        avg_points_used INT,     -- Number of points used in averaging (3 or 4)
        has_full_ttm BOOLEAN,    -- True if 4 quarters available
        has_full_avg BOOLEAN,    -- True if 4 points available for averaging
        calculation_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        
        -- Indexes
        PRIMARY KEY (ticker, year, quarter),
        INDEX idx_ticker_date (ticker, year, quarter),
        INDEX idx_calc_date (calc_date),
        INDEX idx_ticker (ticker),
        INDEX idx_period (year, quarter)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """
    
    try:
        print("🔄 Creating intermediary calculations table...")
        with engine.connect() as conn:
            conn.execute(text(create_table_query))
            conn.commit()
        print("✅ Successfully created intermediary_calculations_enhanced table")
        print()
        
        # Verify table structure
        verify_query = """
        SELECT 
            COUNT(*) as column_count,
            COUNT(CASE WHEN column_name LIKE '%%_TTM' THEN 1 END) as ttm_columns,
            COUNT(CASE WHEN column_name LIKE 'Avg%%' THEN 1 END) as avg_columns,
            COUNT(CASE WHEN column_name IN ('DSO', 'DIO', 'DPO', 'CCC') THEN 1 END) as wc_columns,
            COUNT(CASE WHEN column_name LIKE '%%PerShare%%' THEN 1 END) as per_share_columns
        FROM information_schema.columns
        WHERE table_schema = DATABASE()
          AND table_name = 'intermediary_calculations_enhanced'
        """
        
        df_verify = pd.read_sql(verify_query, engine)
        
        print("📊 Table Structure Summary:")
        print(f"   - Total columns: {df_verify['column_count'].iloc[0]}")
        print(f"   - TTM metrics: {df_verify['ttm_columns'].iloc[0]}")
        print(f"   - Balance sheet averages: {df_verify['avg_columns'].iloc[0]}")
        print(f"   - Working capital metrics: {df_verify['wc_columns'].iloc[0]}")
        print(f"   - Per-share metrics: {df_verify['per_share_columns'].iloc[0]}")
        print()
        
        # Add table comment
        comment_query = """
        ALTER TABLE intermediary_calculations_enhanced 
        COMMENT = 'Pre-calculated intermediary values for factor calculations. 
                   Stores TTM aggregations, balance sheet averages (5-point for non-financial), 
                   working capital metrics, and per-share calculations. 
                   Created: 2025-06-29 as part of Major Surgery Phase 2.';
        """
        
        with engine.connect() as conn:
            conn.execute(text(comment_query))
            conn.commit()
        print("✅ Table comment added for documentation")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating intermediary table: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_calculation_procedures(engine):
    """Create stored procedures for efficient calculation"""
    
    print()
    print("🔄 Creating calculation helper procedures...")
    
    # Create TTM calculation function
    ttm_function_query = """
    CREATE OR REPLACE FUNCTION calculate_ttm_sum(
        p_ticker VARCHAR(10),
        p_year INT,
        p_quarter INT,
        p_metric_name VARCHAR(100)
    )
    RETURNS DECIMAL(30,2)
    DETERMINISTIC
    READS SQL DATA
    BEGIN
        DECLARE ttm_value DECIMAL(30,2);
        
        SELECT SUM(value) INTO ttm_value
        FROM (
            SELECT value
            FROM v_comprehensive_fundamental_items v
            WHERE v.ticker = p_ticker
              AND (
                  (v.year = p_year AND v.quarter <= p_quarter)
                  OR (v.year = p_year - 1 AND v.quarter > p_quarter)
              )
            ORDER BY v.year DESC, v.quarter DESC
            LIMIT 4
        ) last_4_quarters;
        
        RETURN ttm_value;
    END;
    """
    
    try:
        # Note: MySQL functions require specific privileges
        # For now, we'll handle calculations in Python
        print("📝 Note: TTM calculations will be handled in Python")
        print("   (MySQL function creation requires SUPER privileges)")
        return True
        
    except Exception as e:
        print(f"⚠️ Could not create MySQL functions: {e}")
        print("   Will use Python for all calculations")
        return True

def main():
    # Load database configuration
    try:
        with open('config/database.yml', 'r') as f:
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
        # Create intermediary table
        success = create_nonfin_intermediary_table(engine)
        
        if success:
            # Create helper procedures
            create_calculation_procedures(engine)
            
            print()
            print("=" * 100)
            print("✅ PHASE 2 TABLE CREATION COMPLETE")
            print("   Intermediary calculations table ready for data population")
            print("=" * 100)
            print()
            print("📋 Next steps:")
            print("   1. Develop population scripts for TTM calculations")
            print("   2. Implement 5-point balance sheet averaging")
            print("   3. Calculate working capital metrics")
            print("   4. Begin historical data population (2010-2025)")
            
        else:
            print("\n❌ Phase 2 table creation failed")
            
    except Exception as e:
        print(f"Error during implementation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        engine.dispose()

if __name__ == "__main__":
    main()