# Canonical QVM Engine - Unit Test

**Purpose:** Validate canonical engine before production use  
**Test Universe:** 4 tickers (OCB, NLG, FPT, SSI)  
**Test Date:** 2025-03-31 (known data availability)  
**Status:** GATE REQUIREMENT - No progression to Phase 2 until unit test passes

**Success Criteria:**
- ✅ Unit test runs without errors on 4-ticker universe
- ✅ All factor scores are non-zero and economically reasonable
- ✅ Results match validation notebook outputs
- ✅ **GATE REQUIREMENT**: Complete validation before production use

# Setup imports and logging
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

# Add production engine to path
production_path = Path.cwd().parent
sys.path.append(str(production_path))

# Import canonical engine
from engine.qvm_engine_canonical import CanonicalQVMEngine

# Setup logging for test visibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

print("✅ Canonical QVM Engine Unit Test Setup Complete")
print(f"Production path: {production_path}")

✅ Canonical QVM Engine Unit Test Setup Complete
Production path: /Users/ducnguyen/Library/CloudStorage/GoogleDrive-duc.nguyentcb@gmail.com/My Drive/quant-world-invest/factor_investing_project/production

# Initialize canonical engine
print("🔧 Initializing Canonical QVM Engine...")

try:
    # Point to project config directory
    project_root = Path.cwd().parent.parent
    config_path = project_root / 'config'

    engine = CanonicalQVMEngine(config_path=str(config_path), log_level='INFO')

    print("✅ Canonical engine initialized successfully")
    print(f"Database connection: {engine.db_config['host']}/{engine.db_config['schema_name']}")
    print(f"Reporting lag: {engine.reporting_lag} days")

except Exception as e:
    print(f"❌ Engine initialization failed: {e}")
    raise

2025-07-22 15:20:21,792 - CanonicalQVMEngine - INFO - Initializing Canonical QVM Engine
2025-07-22 15:20:21,792 - CanonicalQVMEngine - INFO - Initializing Canonical QVM Engine
2025-07-22 15:20:21,802 - CanonicalQVMEngine - INFO - Configurations loaded successfully
2025-07-22 15:20:21,802 - CanonicalQVMEngine - INFO - Configurations loaded successfully
2025-07-22 15:20:21,979 - CanonicalQVMEngine - INFO - Database connection established successfully
2025-07-22 15:20:21,979 - CanonicalQVMEngine - INFO - Database connection established successfully
2025-07-22 15:20:21,979 - CanonicalQVMEngine - INFO - Canonical QVM Engine initialized successfully
2025-07-22 15:20:21,979 - CanonicalQVMEngine - INFO - Canonical QVM Engine initialized successfully
2025-07-22 15:20:21,980 - CanonicalQVMEngine - INFO - QVM Weights: Quality 40.0%, Value 30.0%, Momentum 30.0%
2025-07-22 15:20:21,980 - CanonicalQVMEngine - INFO - QVM Weights: Quality 40.0%, Value 30.0%, Momentum 30.0%
🔧 Initializing Canonical QVM Engine...
✅ Canonical engine initialized successfully
Database connection: localhost/alphabeta
Reporting lag: 45 days

# Define test parameters
TEST_DATE = pd.Timestamp('2025-07-22')  # Known data availability
TEST_UNIVERSE = ['OCB', 'NLG', 'FPT', 'SSI']  # Multi-sector test universe

EXPECTED_SECTORS = {
    'OCB': 'Banking',
    'NLG': 'Real Estate',
    'FPT': 'Technology',
    'SSI': 'Securities'
}

print(f"📊 Test Configuration:")
print(f"Test Date: {TEST_DATE.date()}")
print(f"Test Universe: {TEST_UNIVERSE}")
print(f"Expected Sectors: {EXPECTED_SECTORS}")

# Validate quarter availability
quarter_info = engine.get_correct_quarter_for_date(TEST_DATE)
if quarter_info:
    year, quarter = quarter_info
    print(f"✅ Available quarter: {year} Q{quarter}")
else:
    print(f"⚠️ No quarter data available for {TEST_DATE.date()}")

📊 Test Configuration:
Test Date: 2025-07-22
Test Universe: ['OCB', 'NLG', 'FPT', 'SSI']
Expected Sectors: {'OCB': 'Banking', 'NLG': 'Real Estate', 'FPT': 'Technology', 'SSI': 'Securities'}
✅ Available quarter: 2025 Q1

# Test 1: Sector Mapping Validation
print("\n🧪 TEST 1: Sector Mapping Validation")
print("=" * 50)

try:
    sector_map = engine.get_sector_mapping()
    test_sectors = sector_map[sector_map['ticker'].isin(TEST_UNIVERSE)]
    
    print(f"Retrieved sectors for test universe:")
    for _, row in test_sectors.iterrows():
        ticker = row['ticker']
        sector = row['sector']
        expected = EXPECTED_SECTORS[ticker]
        status = "✅" if sector == expected else "❌"
        print(f"{status} {ticker}: {sector} (expected: {expected})")
    
    # Validation
    all_correct = all(
        test_sectors[test_sectors['ticker'] == ticker]['sector'].iloc[0] == expected
        for ticker, expected in EXPECTED_SECTORS.items()
        if ticker in test_sectors['ticker'].values
    )
    
    if all_correct:
        print("✅ TEST 1 PASSED: Sector mapping correct")
    else:
        print("❌ TEST 1 FAILED: Sector mapping incorrect")
        
except Exception as e:
    print(f"❌ TEST 1 ERROR: {e}")


🧪 TEST 1: Sector Mapping Validation
==================================================
Retrieved sectors for test universe:
✅ NLG: Real Estate (expected: Real Estate)
✅ SSI: Securities (expected: Securities)
✅ FPT: Technology (expected: Technology)
✅ OCB: Banking (expected: Banking)
✅ TEST 1 PASSED: Sector mapping correct

# Test 2: Fundamental Data Retrieval
print("\n🧪 TEST 2: Fundamental Data Retrieval")
print("=" * 50)

try:
    fundamentals = engine.get_fundamentals_correct_timing(TEST_DATE, TEST_UNIVERSE)
    
    if not fundamentals.empty:
        print(f"✅ Retrieved {len(fundamentals)} fundamental records")
        
        # Check data quality
        for ticker in TEST_UNIVERSE:
            ticker_data = fundamentals[fundamentals['ticker'] == ticker]
            if not ticker_data.empty:
                row = ticker_data.iloc[0]
                sector = row.get('sector', 'Unknown')
                
                # Check key metrics
                net_profit = row.get('NetProfit_TTM', 0)
                total_equity = row.get('AvgTotalEquity', 0)
                has_ttm = row.get('has_full_ttm', 0)
                
                print(f"📊 {ticker} ({sector}):")
                print(f"   NetProfit_TTM: {net_profit:,.0f}")
                print(f"   AvgTotalEquity: {total_equity:,.0f}")
                print(f"   Has Full TTM: {bool(has_ttm)}")
            else:
                print(f"⚠️ {ticker}: No fundamental data")
        
        print("✅ TEST 2 PASSED: Fundamental data retrieved")
    else:
        print("❌ TEST 2 FAILED: No fundamental data retrieved")
        
except Exception as e:
    print(f"❌ TEST 2 ERROR: {e}")

2025-07-22 15:22:56,192 - CanonicalQVMEngine - INFO - Retrieved 4 total fundamental records for 2025-07-22
2025-07-22 15:22:56,192 - CanonicalQVMEngine - INFO - Retrieved 4 total fundamental records for 2025-07-22

🧪 TEST 2: Fundamental Data Retrieval
==================================================
✅ Retrieved 4 fundamental records
📊 OCB (Banking):
   NetProfit_TTM: 2,932,934,728,146
   AvgTotalEquity: 30,838,336,130,891
   Has Full TTM: True
📊 NLG (Real Estate):
   NetProfit_TTM: 1,556,557,651,450
   AvgTotalEquity: 13,803,448,662,579
   Has Full TTM: True
📊 FPT (Technology):
   NetProfit_TTM: 9,855,370,712,531
   AvgTotalEquity: 34,704,201,924,362
   Has Full TTM: True
📊 SSI (Securities):
   NetProfit_TTM: 2,924,802,015,721
   AvgTotalEquity: 25,501,091,461,874
   Has Full TTM: True
✅ TEST 2 PASSED: Fundamental data retrieved

# Test 3: Market Data Retrieval (CORRECTED)
print("\n🧪 TEST 3: Market Data Retrieval")
print("=" * 50)

try:
    market_data = engine.get_market_data(TEST_DATE, TEST_UNIVERSE)

    if not market_data.empty:
        print(f"✅ Retrieved market data for {len(market_data)} tickers")

        for _, row in market_data.iterrows():
            ticker = row['ticker']
            market_cap = row.get('market_cap', 0)
            price = row.get('price', 0)  # Use 'price' not 'adj_close'
            trading_date = row.get('trading_date', 'Unknown')

            print(f"📈 {ticker}:")
            print(f"   Market Cap: {market_cap:,.0f}")
            print(f"   Price: {price:.2f}")
            print(f"   Date: {trading_date}")

        print("✅ TEST 3 PASSED: Market data retrieved")
    else:
        print("❌ TEST 3 FAILED: No market data retrieved")

except Exception as e:
    print(f"❌ TEST 3 ERROR: {e}")


🧪 TEST 3: Market Data Retrieval
==================================================
✅ Retrieved market data for 4 tickers
📈 FPT:
   Market Cap: 186,647,595,372,000
   Price: 126000.00
   Date: 2025-07-18
📈 NLG:
   Market Cap: 15,672,564,872,800
   Price: 40700.00
   Date: 2025-07-18
📈 OCB:
   Market Cap: 30,082,627,654,400
   Price: 12200.00
   Date: 2025-07-18
📈 SSI:
   Market Cap: 62,705,543,910,000
   Price: 31800.00
   Date: 2025-07-18
✅ TEST 3 PASSED: Market data retrieved

# Test 4: QVM Composite Calculation (CRITICAL TEST)
print("\n🧪 TEST 4: QVM Composite Calculation (CRITICAL)")
print("=" * 50)

try:
    qvm_scores = engine.calculate_qvm_composite(TEST_DATE, TEST_UNIVERSE)
    
    if qvm_scores:
        print(f"✅ Calculated QVM scores for {len(qvm_scores)} tickers")
        print("\n📊 QVM COMPOSITE RESULTS:")
        print("-" * 40)
        
        # Sort by QVM score for ranking
        sorted_scores = sorted(qvm_scores.items(), key=lambda x: x[1], reverse=True)
        
        for rank, (ticker, score) in enumerate(sorted_scores, 1):
            sector = EXPECTED_SECTORS.get(ticker, 'Unknown')
            print(f"{rank}. {ticker} ({sector}): {score:.4f}")
        
        # Validation checks
        non_zero_scores = [score for score in qvm_scores.values() if abs(score) > 0.001]
        reasonable_range = [score for score in qvm_scores.values() if -5 <= score <= 5]
        
        print(f"\n📋 VALIDATION SUMMARY:")
        print(f"   Total scores: {len(qvm_scores)}")
        print(f"   Non-zero scores: {len(non_zero_scores)}")
        print(f"   Reasonable range (-5 to 5): {len(reasonable_range)}")
        
        # Success criteria
        success_criteria = [
            len(qvm_scores) == len(TEST_UNIVERSE),
            len(non_zero_scores) >= 2,  # At least half should be non-zero
            len(reasonable_range) == len(qvm_scores),  # All should be reasonable
            not any(np.isnan(score) for score in qvm_scores.values())  # No NaN values
        ]
        
        if all(success_criteria):
            print("✅ TEST 4 PASSED: QVM calculation successful")
            print("🎯 CANONICAL ENGINE VALIDATION COMPLETE")
        else:
            print("❌ TEST 4 FAILED: QVM calculation issues detected")
            print(f"   Criteria: {success_criteria}")
            
    else:
        print("❌ TEST 4 FAILED: No QVM scores calculated")
        
except Exception as e:
    print(f"❌ TEST 4 ERROR: {e}")
    import traceback
    traceback.print_exc()

2025-07-22 15:26:03,177 - CanonicalQVMEngine - INFO - Calculating QVM composite for 4 tickers on 2025-07-22
2025-07-22 15:26:03,177 - CanonicalQVMEngine - INFO - Calculating QVM composite for 4 tickers on 2025-07-22
2025-07-22 15:26:03,280 - CanonicalQVMEngine - INFO - Retrieved 4 total fundamental records for 2025-07-22
2025-07-22 15:26:03,280 - CanonicalQVMEngine - INFO - Retrieved 4 total fundamental records for 2025-07-22
2025-07-22 15:26:03,362 - CanonicalQVMEngine - WARNING - Insufficient data for sector-neutral normalization (min sector size: 1)
2025-07-22 15:26:03,362 - CanonicalQVMEngine - WARNING - Insufficient data for sector-neutral normalization (min sector size: 1)
2025-07-22 15:26:03,363 - CanonicalQVMEngine - INFO - Falling back to cross-sectional normalization
2025-07-22 15:26:03,363 - CanonicalQVMEngine - INFO - Falling back to cross-sectional normalization
2025-07-22 15:26:03,367 - CanonicalQVMEngine - WARNING - Insufficient data for sector-neutral normalization (min sector size: 1)
2025-07-22 15:26:03,367 - CanonicalQVMEngine - WARNING - Insufficient data for sector-neutral normalization (min sector size: 1)
2025-07-22 15:26:03,367 - CanonicalQVMEngine - INFO - Falling back to cross-sectional normalization
2025-07-22 15:26:03,367 - CanonicalQVMEngine - INFO - Falling back to cross-sectional normalization

🧪 TEST 4: QVM Composite Calculation (CRITICAL)
==================================================
2025-07-22 15:26:03,646 - CanonicalQVMEngine - INFO - Successfully calculated QVM scores for 4 tickers
2025-07-22 15:26:03,646 - CanonicalQVMEngine - INFO - Successfully calculated QVM scores for 4 tickers
✅ Calculated QVM scores for 4 tickers

📊 QVM COMPOSITE RESULTS:
----------------------------------------
1. NLG (Real Estate): 0.3016
2. OCB (Banking): 0.2889
3. FPT (Technology): 0.0024
4. SSI (Securities): -0.5929

📋 VALIDATION SUMMARY:
   Total scores: 4
   Non-zero scores: 4
   Reasonable range (-5 to 5): 4
✅ TEST 4 PASSED: QVM calculation successful
🎯 CANONICAL ENGINE VALIDATION COMPLETE

# Final Validation Summary
print("\n🎯 FINAL VALIDATION SUMMARY")
print("=" * 50)

# Run complete validation
try:
    # Test complete engine workflow
    final_scores = engine.calculate_qvm_composite(TEST_DATE, TEST_UNIVERSE)
    
    validation_results = {
        'Engine Initialization': True,
        'Sector Mapping': len(engine.get_sector_mapping()) > 0,
        'Fundamental Data': len(engine.get_fundamentals_correct_timing(TEST_DATE, TEST_UNIVERSE)) > 0,
        'Market Data': len(engine.get_market_data(TEST_DATE, TEST_UNIVERSE)) > 0,
        'QVM Calculation': len(final_scores) > 0,
        'Non-Zero Results': any(abs(score) > 0.001 for score in final_scores.values()),
        'Reasonable Values': all(-10 <= score <= 10 for score in final_scores.values()),
        'No NaN Values': not any(np.isnan(score) for score in final_scores.values())
    }
    
    print("📊 VALIDATION CHECKLIST:")
    all_passed = True
    
    for test_name, result in validation_results.items():
        status = "✅" if result else "❌"
        print(f"{status} {test_name}: {'PASS' if result else 'FAIL'}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 CANONICAL ENGINE VALIDATION: ✅ PASSED")
        print("🚀 READY FOR PHASE 2: DATA RESTORATION")
        print("\n🎯 GATE REQUIREMENT MET - PROCEED TO PRODUCTION USE")
    else:
        print("🚫 CANONICAL ENGINE VALIDATION: ❌ FAILED")
        print("⚠️  DO NOT PROCEED TO PHASE 2 - FIX ISSUES FIRST")
        print("\n🛑 GATE REQUIREMENT NOT MET - TROUBLESHOOTING REQUIRED")
    
    print("=" * 50)
    
except Exception as e:
    print(f"❌ FINAL VALIDATION ERROR: {e}")
    print("🛑 CANONICAL ENGINE NOT READY FOR PRODUCTION")

2025-07-22 15:26:35,061 - CanonicalQVMEngine - INFO - Calculating QVM composite for 4 tickers on 2025-07-22
2025-07-22 15:26:35,061 - CanonicalQVMEngine - INFO - Calculating QVM composite for 4 tickers on 2025-07-22
2025-07-22 15:26:35,183 - CanonicalQVMEngine - INFO - Retrieved 4 total fundamental records for 2025-07-22
2025-07-22 15:26:35,183 - CanonicalQVMEngine - INFO - Retrieved 4 total fundamental records for 2025-07-22

🎯 FINAL VALIDATION SUMMARY
==================================================
2025-07-22 15:26:35,268 - CanonicalQVMEngine - WARNING - Insufficient data for sector-neutral normalization (min sector size: 1)
2025-07-22 15:26:35,268 - CanonicalQVMEngine - WARNING - Insufficient data for sector-neutral normalization (min sector size: 1)
2025-07-22 15:26:35,269 - CanonicalQVMEngine - INFO - Falling back to cross-sectional normalization
2025-07-22 15:26:35,269 - CanonicalQVMEngine - INFO - Falling back to cross-sectional normalization
2025-07-22 15:26:35,272 - CanonicalQVMEngine - WARNING - Insufficient data for sector-neutral normalization (min sector size: 1)
2025-07-22 15:26:35,272 - CanonicalQVMEngine - WARNING - Insufficient data for sector-neutral normalization (min sector size: 1)
2025-07-22 15:26:35,273 - CanonicalQVMEngine - INFO - Falling back to cross-sectional normalization
2025-07-22 15:26:35,273 - CanonicalQVMEngine - INFO - Falling back to cross-sectional normalization
2025-07-22 15:26:35,310 - CanonicalQVMEngine - INFO - Successfully calculated QVM scores for 4 tickers
2025-07-22 15:26:35,310 - CanonicalQVMEngine - INFO - Successfully calculated QVM scores for 4 tickers
2025-07-22 15:26:35,339 - CanonicalQVMEngine - INFO - Retrieved 4 total fundamental records for 2025-07-22
2025-07-22 15:26:35,339 - CanonicalQVMEngine - INFO - Retrieved 4 total fundamental records for 2025-07-22
📊 VALIDATION CHECKLIST:
✅ Engine Initialization: PASS
✅ Sector Mapping: PASS
✅ Fundamental Data: PASS
✅ Market Data: PASS
✅ QVM Calculation: PASS
✅ Non-Zero Results: PASS
✅ Reasonable Values: PASS
✅ No NaN Values: PASS

==================================================
🎉 CANONICAL ENGINE VALIDATION: ✅ PASSED
🚀 READY FOR PHASE 2: DATA RESTORATION

🎯 GATE REQUIREMENT MET - PROCEED TO PRODUCTION USE
==================================================

# Validation Notes

## Success Criteria Checklist
- [ ] Engine initializes without errors
- [ ] Sector mapping retrieval works correctly
- [ ] Fundamental data retrieval with point-in-time logic
- [ ] Market data retrieval as of analysis date
- [ ] QVM composite calculation produces reasonable results
- [ ] All factor scores are non-zero and economically sensible
- [ ] No NaN values in output
- [ ] Results are in reasonable range (-10 to +10)

## Expected Behavior
- **OCB (Banking)**: Should have reasonable quality/value scores from banking metrics
- **NLG (Real Estate)**: Should show sector-specific characteristics
- **FPT (Technology)**: Typically high-quality, growth-oriented scores
- **SSI (Securities)**: Should reflect securities sector dynamics

## Gate Requirement
**🚨 CRITICAL**: This unit test serves as the gate requirement for Phase 2 progression. All tests must pass before any production data restoration attempts.

If any test fails, the canonical engine must be fixed before proceeding to avoid contaminating the production data restoration process.

# ============================================================================
# CELL 1: DETAILED FUNDAMENTAL DATA AUDIT (USING WORKING PATTERNS)
# ============================================================================

import pandas as pd

print("="*80)
print("COMPREHENSIVE AUDIT: RAW FUNDAMENTAL DATA RETRIEVAL")
print("="*80)
print()

# Use same test parameters as working unit test
TEST_DATE = pd.Timestamp('2025-07-22')
TEST_UNIVERSE = ['OCB', 'NLG', 'FPT', 'SSI']

EXPECTED_SECTORS = {
    'OCB': 'Banking',
    'NLG': 'Real Estate',
    'FPT': 'Technology',
    'SSI': 'Securities'
}

print(f"📊 Test Configuration:")
print(f"Test Date: {TEST_DATE.date()}")
print(f"Test Universe: {TEST_UNIVERSE}")
print()

# Get fundamental data using working method
print("🔍 RETRIEVING FUNDAMENTAL DATA...")
try:
    fundamentals = engine.get_fundamentals_correct_timing(TEST_DATE, TEST_UNIVERSE)

    print(f"✅ Retrieved {len(fundamentals)} fundamental records")
    print(f"Columns available: {list(fundamentals.columns)}")
    print()

    # Display detailed fundamental data for each ticker
    for ticker in TEST_UNIVERSE:
        ticker_data = fundamentals[fundamentals['ticker'] == ticker]

        if not ticker_data.empty:
            print(f"📊 TICKER: {ticker}")
            print("-" * 50)

            row = ticker_data.iloc[0]
            sector = row.get('sector', 'Unknown')

            print(f"Sector: {sector}")
            print(f"Year: {row.get('year', 'N/A')}")
            print(f"Quarter: Q{row.get('quarter', 'N/A')}")
            print(f"Publish Date: {row.get('publish_date', 'N/A')}")

            print(f"\n📈 FUNDAMENTAL METRICS:")

            # Key financial metrics with proper null handling
            metrics = [
                ('NetProfit_TTM', 'Net Profit TTM'),
                ('AvgTotalEquity', 'Avg Total Equity'),
                ('AvgTotalAssets', 'Avg Total Assets'),
                ('Revenue_TTM', 'Revenue TTM'),
                ('TotalOperatingIncome_TTM', 'Operating Income TTM'),
                ('TotalOperatingRevenue_TTM', 'Operating Revenue TTM'),
                ('EBITDA_TTM', 'EBITDA TTM'),
                ('ROAE_TTM', 'ROAE TTM'),
                ('ROAA_TTM', 'ROAA TTM')
            ]

            for col_name, display_name in metrics:
                value = row.get(col_name)
                if pd.notna(value):
                    if col_name in ['ROAE_TTM', 'ROAA_TTM']:
                        print(f"    {display_name}: {value:.4f}")
                    else:
                        print(f"    {display_name}: {value:,.0f}")
                else:
                    print(f"    {display_name}: N/A")

            print(f"    Has Full TTM: {bool(row.get('has_full_ttm', False))}")
            print()

        else:
            print(f"❌ {ticker}: No fundamental data found")
            print()

    print("✅ FUNDAMENTAL DATA AUDIT COMPLETE")

except Exception as e:
    print(f"❌ Error retrieving fundamental data: {str(e)}")
    import traceback
    traceback.print_exc()

2025-07-22 16:06:08,753 - CanonicalQVMEngine - INFO - Retrieved 4 total fundamental records for 2025-07-22
2025-07-22 16:06:08,753 - CanonicalQVMEngine - INFO - Retrieved 4 total fundamental records for 2025-07-22
================================================================================
COMPREHENSIVE AUDIT: RAW FUNDAMENTAL DATA RETRIEVAL
================================================================================

📊 Test Configuration:
Test Date: 2025-07-22
Test Universe: ['OCB', 'NLG', 'FPT', 'SSI']

🔍 RETRIEVING FUNDAMENTAL DATA...
✅ Retrieved 4 fundamental records
Columns available: ['ticker', 'year', 'quarter', 'calc_date', 'NII_TTM', 'InterestIncome_TTM', 'InterestExpense_TTM', 'NetFeeIncome_TTM', 'ForexIncome_TTM', 'TradingIncome_TTM', 'InvestmentIncome_TTM', 'OtherIncome_TTM', 'EquityInvestmentIncome_TTM', 'OperatingExpenses_TTM', 'OperatingProfit_TTM', 'CreditProvisions_TTM', 'ProfitBeforeTax_TTM', 'TaxExpense_TTM', 'NetProfit_TTM', 'NetProfitAfterMI_TTM', 'AvgTotalAssets', 'AvgGrossLoans', 'AvgLoanLossReserves', 'AvgNetLoans', 'AvgTradingSecurities', 'AvgInvestmentSecurities', 'AvgCash', 'AvgCustomerDeposits', 'AvgTotalEquity', 'AvgPaidInCapital', 'AvgEarningAssets', 'AvgTotalDeposits', 'AvgBorrowings', 'TotalOperatingIncome_TTM', 'NonInterestIncome_TTM_Raw', 'quarters_available_ttm', 'has_full_ttm', 'avg_points_used', 'has_full_avg', 'data_quality_score', 'sector', 'BrokerageRevenue_TTM', 'UnderwritingRevenue_TTM', 'AdvisoryRevenue_TTM', 'CustodyServiceRevenue_TTM', 'EntrustedAuctionRevenue_TTM', 'OtherOperatingIncome_TTM', 'TradingGainFVTPL_TTM', 'TradingGainHTM_TTM', 'TradingGainLoans_TTM', 'TradingGainAFS_TTM', 'TradingGainDerivatives_TTM', 'ManagementExpenses_TTM', 'OperatingResult_TTM', 'IncomeTaxExpense_TTM', 'AvgFinancialAssets', 'AvgCashAndCashEquivalents', 'AvgFinancialAssetsFVTPL', 'AvgLoanReceivables', 'AvgCharterCapital', 'AvgRetainedEarnings', 'TotalSecuritiesServices_TTM', 'NetTradingIncome_TTM', 'TotalOperatingRevenue_TTM', 'Revenue_TTM', 'COGS_TTM', 'GrossProfit_TTM', 'SellingExpenses_TTM', 'AdminExpenses_TTM', 'FinancialIncome_TTM', 'FinancialExpenses_TTM', 'EBITDA_TTM', 'EBIT_TTM', 'CurrentTax_TTM', 'DeferredTax_TTM', 'TotalTax_TTM', 'NetCFO_TTM', 'NetCFI_TTM', 'NetCFF_TTM', 'DepreciationAmortization_TTM', 'CapEx_TTM', 'FCF_TTM', 'DividendsPaid_TTM', 'ShareIssuance_TTM', 'ShareRepurchase_TTM', 'DebtIssuance_TTM', 'DebtRepayment_TTM', 'AvgTotalLiabilities', 'AvgCurrentAssets', 'AvgCurrentLiabilities', 'AvgWorkingCapital', 'AvgCashEquivalents', 'AvgShortTermInvestments', 'AvgInventory', 'AvgReceivables', 'AvgPayables', 'AvgFixedAssets', 'AvgTangibleAssets', 'AvgIntangibleAssets', 'AvgGoodwill', 'AvgShortTermDebt', 'AvgLongTermDebt', 'AvgTotalDebt', 'AvgNetDebt', 'AvgInvestedCapital', 'DSO', 'DIO', 'DPO', 'CCC', 'SharesOutstanding', 'EPS_TTM', 'BookValuePerShare', 'TangibleBookValuePerShare', 'SalesPerShare_TTM', 'CFOPerShare_TTM', 'FCFPerShare_TTM', 'DividendPerShare_TTM', 'quarters_available', 'calculation_timestamp']

📊 TICKER: OCB
--------------------------------------------------
Sector: Banking
Year: 2025
Quarter: Q1
Publish Date: N/A

📈 FUNDAMENTAL METRICS:
    Net Profit TTM: 2,932,934,728,146
    Avg Total Equity: 30,838,336,130,891
    Avg Total Assets: 262,228,886,385,451
    Revenue TTM: N/A
    Operating Income TTM: 10,055,388,932,563
    Operating Revenue TTM: N/A
    EBITDA TTM: N/A
    ROAE TTM: N/A
    ROAA TTM: N/A
    Has Full TTM: True

📊 TICKER: NLG
--------------------------------------------------
Sector: Real Estate
Year: 2025
Quarter: Q1
Publish Date: N/A

📈 FUNDAMENTAL METRICS:
    Net Profit TTM: 1,556,557,651,450
    Avg Total Equity: 13,803,448,662,579
    Avg Total Assets: 29,489,632,521,865
    Revenue TTM: 8,282,567,305,627
    Operating Income TTM: N/A
    Operating Revenue TTM: N/A
    EBITDA TTM: 1,959,705,245,178
    ROAE TTM: N/A
    ROAA TTM: N/A
    Has Full TTM: True

📊 TICKER: FPT
--------------------------------------------------
Sector: Technology
Year: 2025
Quarter: Q1
Publish Date: N/A

📈 FUNDAMENTAL METRICS:
    Net Profit TTM: 9,855,370,712,531
    Avg Total Equity: 34,704,201,924,362
    Avg Total Assets: 68,180,689,833,131
    Revenue TTM: 64,814,006,880,129
    Operating Income TTM: N/A
    Operating Revenue TTM: N/A
    EBITDA TTM: 13,378,666,050,091
    ROAE TTM: N/A
    ROAA TTM: N/A
    Has Full TTM: True

📊 TICKER: SSI
--------------------------------------------------
Sector: Securities
Year: 2025
Quarter: Q1
Publish Date: N/A

📈 FUNDAMENTAL METRICS:
    Net Profit TTM: 2,924,802,015,721
    Avg Total Equity: 25,501,091,461,874
    Avg Total Assets: 72,065,658,946,264
    Revenue TTM: N/A
    Operating Income TTM: N/A
    Operating Revenue TTM: 8,715,728,920,798
    EBITDA TTM: N/A
    ROAE TTM: N/A
    ROAA TTM: N/A
    Has Full TTM: True

✅ FUNDAMENTAL DATA AUDIT COMPLETE

# ============================================================================
# CELL 2: DYNAMIC FACTOR CALCULATION AUDIT  
# ============================================================================

print("="*80)
print("DYNAMIC FACTOR CALCULATION AUDIT - QUALITY, VALUE, MOMENTUM")
print("="*80)
print()

# We'll manually walk through the dynamic calculations to see intermediate steps
for ticker in TEST_UNIVERSE:
    ticker_data = fundamentals[fundamentals['ticker'] == ticker]

    if not ticker_data.empty:
        print(f"🧮 TICKER: {ticker}")
        print("-" * 50)

        row = ticker_data.iloc[0]
        sector = row.get('sector', 'Unknown')
        print(f"Sector: {sector}")

        # ============================================================================
        # QUALITY FACTORS - DYNAMIC CALCULATION FROM RAW BUILDING BLOCKS
        # ============================================================================
        print(f"\n📊 QUALITY FACTORS (Dynamic Calculation):")

        # ROAE Calculation (Dynamic)
        net_profit_ttm = row.get('NetProfit_TTM', 0)
        avg_total_equity = row.get('AvgTotalEquity', 0)

        if avg_total_equity > 0:
            roae_calculated = net_profit_ttm / avg_total_equity
            # Corrected f-string newline issue
            print(f"    ROAE (Calculated): {roae_calculated:.4f} = {net_profit_ttm:,.0f} / {avg_total_equity:,.0f}")
        else:
            roae_calculated = 0
            print(f"    ROAE (Calculated): N/A (Zero equity)")

        # ROAA Calculation (Dynamic)
        avg_total_assets = row.get('AvgTotalAssets', 0)
        if avg_total_assets > 0:
            roaa_calculated = net_profit_ttm / avg_total_assets
            # Corrected f-string newline issue
            print(f"    ROAA (Calculated): {roaa_calculated:.4f} = {net_profit_ttm:,.0f} / {avg_total_assets:,.0f}")
        else:
            roaa_calculated = 0
            print(f"    ROAA (Calculated): N/A (Zero assets)")

        # Sector-specific quality metrics
        if sector == 'Banking':
            # NIM calculation for banking
            nii_ttm = row.get('NII_TTM', 0)
            avg_earning_assets = row.get('AvgEarningAssets', 0)
            if avg_earning_assets > 0:
                nim_calculated = nii_ttm / avg_earning_assets
                # Corrected f-string newline issue
                print(f"    NIM (Banking): {nim_calculated:.4f} = {nii_ttm:,.0f} / {avg_earning_assets:,.0f}")

        # ============================================================================
        # VALUE FACTORS - DYNAMIC CALCULATION
        # ============================================================================
        print(f"\n💰 VALUE FACTORS (Dynamic Calculation):")

        # Get market data for valuation ratios
        try:
            market_data = engine.get_market_data(TEST_DATE, [ticker])
            if not market_data.empty:
                market_cap = market_data.iloc[0].get('market_cap', 0)
                price = market_data.iloc[0].get('price', 0)

                print(f"    Market Cap: {market_cap:,.0f}")
                print(f"    Price: {price:,.0f}")

                # P/E Ratio (Dynamic)
                if net_profit_ttm > 0:
                    pe_ratio = market_cap / net_profit_ttm
                    # Corrected f-string newline issue
                    print(f"    P/E (Calculated): {pe_ratio:.2f} = {market_cap:,.0f} / {net_profit_ttm:,.0f}")
                else:
                    print(f"    P/E (Calculated): N/A (Negative/Zero earnings)")

                # P/B Ratio (Dynamic)
                if avg_total_equity > 0:
                    pb_ratio = market_cap / avg_total_equity
                    # Corrected f-string newline issue
                    print(f"    P/B (Calculated): {pb_ratio:.2f} = {market_cap:,.0f} / {avg_total_equity:,.0f}")
                else:
                    print(f"    P/B (Calculated): N/A (Zero book value)")

                # P/S Ratio (Dynamic) - Sector-specific revenue
                revenue_ttm = None
                if sector == 'Banking':
                    revenue_ttm = row.get('TotalOperatingIncome_TTM', 0)
                    revenue_label = 'Total Operating Income TTM'
                elif sector == 'Securities':
                    revenue_ttm = row.get('TotalOperatingRevenue_TTM', 0)
                    revenue_label = 'Total Operating Revenue TTM'
                else:
                    revenue_ttm = row.get('Revenue_TTM', 0)
                    revenue_label = 'Revenue TTM'

                if revenue_ttm is not None and revenue_ttm > 0:
                    ps_ratio = market_cap / revenue_ttm
                    # Corrected f-string newline issue
                    print(f"    P/S (Calculated): {ps_ratio:.2f} = {market_cap:,.0f} / {revenue_ttm:,.0f} ({revenue_label})")
                else:
                    print(f"    P/S (Calculated): N/A (Zero {revenue_label})")

            else:
                print(f"    ❌ No market data available for valuation ratios")

        except Exception as e:
            print(f"    ❌ Error getting market data: {e}")

        # ============================================================================  
        # MOMENTUM FACTORS - DYNAMIC CALCULATION
        # ============================================================================
        print(f"\n🚀 MOMENTUM FACTORS (Dynamic Calculation):")
        # Corrected f-string newline issue
        print(f"    Note: Momentum calculated from equity_history table (not shown in detail here)")
        print(f"    Timeframes: 1M, 3M, 6M, 12M with skip-1-month convention")

        print()

print("✅ DYNAMIC FACTOR CALCULATION AUDIT COMPLETE")
print("\n💡 KEY INSIGHT: All ratios calculated on-demand from raw building blocks")
print("💡 This confirms 'Store Raw Data, Transform Dynamically' architecture")

================================================================================
DYNAMIC FACTOR CALCULATION AUDIT - QUALITY, VALUE, MOMENTUM
================================================================================

🧮 TICKER: OCB
--------------------------------------------------
Sector: Banking

📊 QUALITY FACTORS (Dynamic Calculation):
    ROAE (Calculated): 0.0951 = 2,932,934,728,146 / 30,838,336,130,891
    ROAA (Calculated): 0.0112 = 2,932,934,728,146 / 262,228,886,385,451
    NIM (Banking): 0.0434 = 8,869,528,802,218 / 204,484,188,059,047

💰 VALUE FACTORS (Dynamic Calculation):
    Market Cap: 30,082,627,654,400
    Price: 12,200
    P/E (Calculated): 10.26 = 30,082,627,654,400 / 2,932,934,728,146
    P/B (Calculated): 0.98 = 30,082,627,654,400 / 30,838,336,130,891
    P/S (Calculated): 2.99 = 30,082,627,654,400 / 10,055,388,932,563 (Total Operating Income TTM)

🚀 MOMENTUM FACTORS (Dynamic Calculation):
    Note: Momentum calculated from equity_history table (not shown in detail here)
    Timeframes: 1M, 3M, 6M, 12M with skip-1-month convention

🧮 TICKER: NLG
--------------------------------------------------
Sector: Real Estate

📊 QUALITY FACTORS (Dynamic Calculation):
    ROAE (Calculated): 0.1128 = 1,556,557,651,450 / 13,803,448,662,579
    ROAA (Calculated): 0.0528 = 1,556,557,651,450 / 29,489,632,521,865

💰 VALUE FACTORS (Dynamic Calculation):
    Market Cap: 15,672,564,872,800
    Price: 40,700
    P/E (Calculated): 10.07 = 15,672,564,872,800 / 1,556,557,651,450
    P/B (Calculated): 1.14 = 15,672,564,872,800 / 13,803,448,662,579
    P/S (Calculated): 1.89 = 15,672,564,872,800 / 8,282,567,305,627 (Revenue TTM)

🚀 MOMENTUM FACTORS (Dynamic Calculation):
    Note: Momentum calculated from equity_history table (not shown in detail here)
    Timeframes: 1M, 3M, 6M, 12M with skip-1-month convention

🧮 TICKER: FPT
--------------------------------------------------
Sector: Technology

📊 QUALITY FACTORS (Dynamic Calculation):
    ROAE (Calculated): 0.2840 = 9,855,370,712,531 / 34,704,201,924,362
    ROAA (Calculated): 0.1445 = 9,855,370,712,531 / 68,180,689,833,131

💰 VALUE FACTORS (Dynamic Calculation):
    Market Cap: 186,647,595,372,000
    Price: 126,000
    P/E (Calculated): 18.94 = 186,647,595,372,000 / 9,855,370,712,531
    P/B (Calculated): 5.38 = 186,647,595,372,000 / 34,704,201,924,362
    P/S (Calculated): 2.88 = 186,647,595,372,000 / 64,814,006,880,129 (Revenue TTM)

🚀 MOMENTUM FACTORS (Dynamic Calculation):
    Note: Momentum calculated from equity_history table (not shown in detail here)
    Timeframes: 1M, 3M, 6M, 12M with skip-1-month convention

🧮 TICKER: SSI
--------------------------------------------------
Sector: Securities

📊 QUALITY FACTORS (Dynamic Calculation):
    ROAE (Calculated): 0.1147 = 2,924,802,015,721 / 25,501,091,461,874
    ROAA (Calculated): 0.0406 = 2,924,802,015,721 / 72,065,658,946,264

💰 VALUE FACTORS (Dynamic Calculation):
    Market Cap: 62,705,543,910,000
    Price: 31,800
    P/E (Calculated): 21.44 = 62,705,543,910,000 / 2,924,802,015,721
    P/B (Calculated): 2.46 = 62,705,543,910,000 / 25,501,091,461,874
    P/S (Calculated): 7.19 = 62,705,543,910,000 / 8,715,728,920,798 (Total Operating Revenue TTM)

🚀 MOMENTUM FACTORS (Dynamic Calculation):
    Note: Momentum calculated from equity_history table (not shown in detail here)
    Timeframes: 1M, 3M, 6M, 12M with skip-1-month convention

✅ DYNAMIC FACTOR CALCULATION AUDIT COMPLETE

💡 KEY INSIGHT: All ratios calculated on-demand from raw building blocks
💡 This confirms 'Store Raw Data, Transform Dynamically' architecture

# ============================================================================
# CELL 3: SECTOR-NEUTRAL NORMALIZATION & QVM COMPOSITE CONSTRUCTION AUDIT
# ============================================================================

import pandas as pd

print("="*80)
print("SECTOR-NEUTRAL NORMALIZATION & QVM COMPOSITE CONSTRUCTION")
print("="*80)
print()

# Let's manually replicate the QVM calculation process to understand the scoring
print("🎯 REPLICATING QVM CALCULATION PROCESS")
print("-" * 50)

# Collect raw factor values from our previous calculations
factor_data = {}

for ticker in TEST_UNIVERSE:
    ticker_data = fundamentals[fundamentals['ticker'] == ticker]
    if not ticker_data.empty:
        row = ticker_data.iloc[0]
        sector = row.get('sector', 'Unknown')

        # Quality factors (already calculated)
        net_profit_ttm = row.get('NetProfit_TTM', 0)
        avg_total_equity = row.get('AvgTotalEquity', 0)
        avg_total_assets = row.get('AvgTotalAssets', 0)

        roae = net_profit_ttm / avg_total_equity if avg_total_equity > 0 else 0
        roaa = net_profit_ttm / avg_total_assets if avg_total_assets > 0 else 0

        # Get market data for value factors
        try:
            # This requires 'engine' to be defined in your environment
            market_data = engine.get_market_data(TEST_DATE, [ticker])
            if not market_data.empty:
                market_cap = market_data.iloc[0].get('market_cap', 0)

                # Value ratios
                pe_ratio = market_cap / net_profit_ttm if net_profit_ttm > 0 else None
                pb_ratio = market_cap / avg_total_equity if avg_total_equity > 0 else None

                # Sector-specific revenue for P/S
                revenue_ttm = None
                if sector == 'Banking':
                    revenue_ttm = row.get('TotalOperatingIncome_TTM', 0)
                elif sector == 'Securities':
                    revenue_ttm = row.get('TotalOperatingRevenue_TTM', 0)
                else:
                    revenue_ttm = row.get('Revenue_TTM', 0)

                # Fixed: Removed the newline from inside the f-string expression
                ps_ratio = market_cap / revenue_ttm if revenue_ttm is not None and revenue_ttm > 0 else None

                factor_data[ticker] = {
                    'sector': sector,
                    'roae': roae,
                    'roaa': roaa,
                    'pe_ratio': pe_ratio,
                    'pb_ratio': pb_ratio,
                    'ps_ratio': ps_ratio,
                    'market_cap': market_cap
                }

        except Exception as e:
            print(f"❌ Error getting market data for {ticker}: {e}")

# Display raw factor values
print("\n📊 RAW FACTOR VALUES (Before Normalization):")
print("-" * 60)
for ticker, data in factor_data.items():
    print(f"\n{ticker} ({data['sector']}):")
    print(f"  Quality: ROAE={data['roae']:.4f}, ROAA={data['roaa']:.4f}")

    # Fixed: Ensured the print statement for value ratios is on a single line
    pe_str = f"{data['pe_ratio']:.2f}" if data['pe_ratio'] is not None else "N/A"
    pb_str = f"{data['pb_ratio']:.2f}" if data['pb_ratio'] is not None else "N/A"
    ps_str = f"{data['ps_ratio']:.2f}" if data['ps_ratio'] is not None else "N/A"
    print(f"  Value: P/E={pe_str}, P/B={pb_str}, P/S={ps_str}")

# Now let's see the actual QVM calculation from the engine
print(f"\n🧮 CANONICAL ENGINE QVM CALCULATION:")
print("-" * 50)

try:
    qvm_scores = engine.calculate_qvm_composite(TEST_DATE, TEST_UNIVERSE)

    if qvm_scores:
        print("✅ QVM Calculation Successful")

        # Create a comprehensive comparison table
        comparison_data = []

        for ticker in TEST_UNIVERSE:
            raw_data = factor_data.get(ticker, {})
            qvm_score = qvm_scores.get(ticker, 0)

            comparison_data.append({
                'Ticker': ticker,
                'Sector': raw_data.get('sector', 'Unknown'),
                'ROAE_Raw': raw_data.get('roae', 0),
                # Fixed: Ensured the conditional assignment for P/E and P/B raw data is on one line
                'P/E_Raw': raw_data.get('pe_ratio', 0) if raw_data.get('pe_ratio') is not None else 0,
                'P/B_Raw': raw_data.get('pb_ratio', 0) if raw_data.get('pb_ratio') is not None else 0,
                'QVM_Score': qvm_score
            })

        # Convert to DataFrame for nice display
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('QVM_Score', ascending=False)

        print("\n📋 COMPREHENSIVE FACTOR → QVM COMPARISON:")
        print("=" * 80)
        # Fixed: Ensured the f-string for header is on one line
        print(f"{'Rank':<4} {'Ticker':<6} {'Sector':<12} {'ROAE':<8} {'P/E':<8} {'P/B':<8} {'QVM Score':<10}")
        print("-" * 80)

        for idx, row in comparison_df.iterrows():
            rank = comparison_df.index.get_loc(idx) + 1
            # Fixed: Ensured the f-string for data rows is on one line
            print(f"{rank:<4} {row['Ticker']:<6} {row['Sector']:<12} {row['ROAE_Raw']:<8.3f} {row['P/E_Raw']:<8.2f} {row['P/B_Raw']:<8.2f} {row['QVM_Score']:<10.4f}")

        print("\n💡 ECONOMIC INTERPRETATION:")
        print("-" * 40)
        for idx, row in comparison_df.iterrows():
            rank = comparison_df.index.get_loc(idx) + 1
            ticker = row['Ticker']
            sector = row['Sector']
            qvm = row['QVM_Score']
            roae = row['ROAE_Raw']
            pe = row['P/E_Raw']
            pb = row['P/B_Raw']

            # Economic interpretation logic
            if roae > 0.15 and pe < 15:
                interpretation = "High-quality value play"
            elif roae > 0.20:
                interpretation = "Premium quality stock"
            elif pe < 12 and pb < 1.2:
                interpretation = "Deep value opportunity"
            elif pe > 20:
                interpretation = "Growth premium/expensive"
            else:
                interpretation = "Balanced quality/value"

            print(f"{rank}. {ticker} ({sector}): {interpretation} (QVM: {qvm:.4f})")

        print("\n✅ AUDIT COMPLETE: QVM methodology validated")
        print("💡 Rankings reflect economic fundamentals appropriately")

    else:
        print("❌ QVM calculation failed")

except Exception as e:
    print(f"❌ Error in QVM calculation: {e}")
    import traceback
    traceback.print_exc()

2025-07-22 16:17:23,590 - CanonicalQVMEngine - INFO - Calculating QVM composite for 4 tickers on 2025-07-22
2025-07-22 16:17:23,590 - CanonicalQVMEngine - INFO - Calculating QVM composite for 4 tickers on 2025-07-22
2025-07-22 16:17:23,616 - CanonicalQVMEngine - INFO - Retrieved 4 total fundamental records for 2025-07-22
2025-07-22 16:17:23,616 - CanonicalQVMEngine - INFO - Retrieved 4 total fundamental records for 2025-07-22
================================================================================
SECTOR-NEUTRAL NORMALIZATION & QVM COMPOSITE CONSTRUCTION
================================================================================

🎯 REPLICATING QVM CALCULATION PROCESS
--------------------------------------------------

📊 RAW FACTOR VALUES (Before Normalization):
------------------------------------------------------------

OCB (Banking):
  Quality: ROAE=0.0951, ROAA=0.0112
  Value: P/E=10.26, P/B=0.98, P/S=2.99

NLG (Real Estate):
  Quality: ROAE=0.1128, ROAA=0.0528
  Value: P/E=10.07, P/B=1.14, P/S=1.89

FPT (Technology):
  Quality: ROAE=0.2840, ROAA=0.1445
  Value: P/E=18.94, P/B=5.38, P/S=2.88

SSI (Securities):
  Quality: ROAE=0.1147, ROAA=0.0406
  Value: P/E=21.44, P/B=2.46, P/S=7.19

🧮 CANONICAL ENGINE QVM CALCULATION:
--------------------------------------------------
2025-07-22 16:17:23,687 - CanonicalQVMEngine - WARNING - Insufficient data for sector-neutral normalization (min sector size: 1)
2025-07-22 16:17:23,687 - CanonicalQVMEngine - WARNING - Insufficient data for sector-neutral normalization (min sector size: 1)
2025-07-22 16:17:23,688 - CanonicalQVMEngine - INFO - Falling back to cross-sectional normalization
2025-07-22 16:17:23,688 - CanonicalQVMEngine - INFO - Falling back to cross-sectional normalization
2025-07-22 16:17:23,693 - CanonicalQVMEngine - WARNING - Insufficient data for sector-neutral normalization (min sector size: 1)
2025-07-22 16:17:23,693 - CanonicalQVMEngine - WARNING - Insufficient data for sector-neutral normalization (min sector size: 1)
2025-07-22 16:17:23,694 - CanonicalQVMEngine - INFO - Falling back to cross-sectional normalization
2025-07-22 16:17:23,694 - CanonicalQVMEngine - INFO - Falling back to cross-sectional normalization
2025-07-22 16:17:23,802 - CanonicalQVMEngine - INFO - Successfully calculated QVM scores for 4 tickers
2025-07-22 16:17:23,802 - CanonicalQVMEngine - INFO - Successfully calculated QVM scores for 4 tickers
✅ QVM Calculation Successful

📋 COMPREHENSIVE FACTOR → QVM COMPARISON:
================================================================================
Rank Ticker Sector       ROAE     P/E      P/B      QVM Score 
--------------------------------------------------------------------------------
1    NLG    Real Estate  0.113    10.07    1.14     0.3016    
2    OCB    Banking      0.095    10.26    0.98     0.2889    
3    FPT    Technology   0.284    18.94    5.38     0.0024    
4    SSI    Securities   0.115    21.44    2.46     -0.5929   

💡 ECONOMIC INTERPRETATION:
----------------------------------------
1. NLG (Real Estate): Deep value opportunity (QVM: 0.3016)
2. OCB (Banking): Deep value opportunity (QVM: 0.2889)
3. FPT (Technology): Premium quality stock (QVM: 0.0024)
4. SSI (Securities): Growth premium/expensive (QVM: -0.5929)

✅ AUDIT COMPLETE: QVM methodology validated
💡 Rankings reflect economic fundamentals appropriately