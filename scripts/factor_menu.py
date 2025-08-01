
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive menu interface for Vietnam Factor Investing Platform workflows
---------------------------------------------------------------------------
Author: Duc Nguyen
Date: May 12, 2025

This script provides a simple, interactive menu-driven interface
to run various Vietnam Factor Investing Platform workflows.
"""

import sys
import subprocess
import os
from pathlib import Path
from datetime import datetime

# Ensure we're running from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if Path.cwd() != PROJECT_ROOT:
    print(f"Error: Please run this script from the project root: {PROJECT_ROOT}")
    sys.exit(1)

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def run_workflow_command(command, args=None):
    """Run a specific workflow command with optional arguments"""
    cmd = ["python", "scripts/run_workflow.py", command]
    if args:
        cmd.extend(args)
    
    print(f"\nRunning: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, text=True)
        success = result.returncode == 0
        if success:
            print("\n✅ Command completed successfully!")
        else:
            print("\n❌ Command failed. Check logs for details.")
        return success
    except Exception as e:
        print(f"\n❌ Error executing command: {e}")
        return False

def run_script(script_path, args=None):
    """Run a specific Python script with optional arguments"""
    cmd = ["python", script_path]
    if args:
        cmd.extend(args)
    
    print(f"\nRunning: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, text=True)
        success = result.returncode == 0
        if success:
            print("\n✅ Script completed successfully!")
        else:
            print("\n❌ Script failed. Check logs for details.")
        return success
    except Exception as e:
        print(f"\n❌ Error executing script: {e}")
        return False

def run_fundamental_extract(ticker):
    """Run the appropriate fundamental extract script based on ticker's sector"""
    ticker = ticker.upper().strip()
    
    # Define sector tickers
    banking_tickers = ['VCB', 'TCB', 'BID', 'CTG', 'VPB', 'TPB', 'MBB', 'STB', 'HDB', 'ACB', 
                      'SHB', 'EIB', 'MSB', 'OCB', 'LPB', 'KLB', 'NVB', 'PGB', 'VIB', 'NAB', 'BAB']
    
    securities_tickers = ['SSI', 'VCI', 'VND', 'HCM', 'BSI', 'SHS', 'MBS', 'FTS', 'VIG', 'TVS',
                         'AGR', 'VDS', 'PSI', 'APS', 'IVS', 'BVS', 'CTS', 'DSC', 'EVS', 'ORS',
                         'TCI', 'VFS', 'WSS', 'ASP', 'VIX', 'CSI']
    
    insurance_tickers = ['BIC', 'BMI', 'MIG', 'PRE', 'PTI', 'VNR', 'PVI', 'PGI', 'VIG', 'BVH', 'PLC']
    
    if ticker in banking_tickers:
        print(f"🏦 Detected banking sector for {ticker}")
        return run_script("scripts/sector_extracts/banking_enhanced_extract.py", [ticker])
    elif ticker in securities_tickers:
        print(f"📈 Detected securities sector for {ticker}")
        return run_script("scripts/sector_extracts/securities_enhanced_extract.py", [ticker])
    elif ticker in insurance_tickers:
        print(f"🛡️ Detected insurance sector for {ticker}")
        return run_script("scripts/sector_extracts/insurance_enhanced_extract_final_corrected.py", [ticker])
    else:
        print(f"🏭 Detected non-financial sector for {ticker} (auto-detecting specific sector)")
        return run_script("scripts/sector_extracts/nonfin_enhanced_extract.py", [ticker])

def run_intermediary_display(ticker):
    """Run the appropriate intermediary display script based on ticker's sector"""
    ticker = ticker.upper().strip()
    
    # Check if banking
    banking_tickers = ['VCB', 'TCB', 'BID', 'CTG', 'VPB', 'TPB', 'MBB', 'STB', 'HDB', 'ACB', 
                      'SHB', 'EIB', 'MSB', 'OCB', 'LPB', 'KLB', 'NVB', 'PGB', 'VIB', 'NAB', 'BAB']
    
    if ticker in banking_tickers:
        print(f"🏦 Detected banking sector for {ticker}")
        return run_script("scripts/intermediaries/banking_sector_intermediary_display.py", [ticker, "--save"])
    else:
        print(f"🔍 Auto-detecting non-financial sector for {ticker}")
        return run_script("scripts/intermediaries/non_financial_intermediary_display.py", [ticker, "--save"])

def show_main_menu():
    """Display the main menu options"""
    clear_screen()
    print('\n' + '='*70)
    print('🚀 VIETNAM FACTOR INVESTING PLATFORM')
    print(f'📅 Current Date: {datetime.now().strftime("%Y-%m-%d")}')
    print('='*70)
    
    print('\n📊 DAILY UPDATES:')
    print('1  -  Market Data Update (OHLCV, ETFs/Indices)')
    print('2  -  Daily Financial Information (Shares Outstanding, etc.)')
    print('2a -  VCSC Complete Data Update (Adjusted/Unadjusted Prices & Microstructure)')
    print('3  -  Foreign Flow Data Update (Foreign Buy/Sell Volumes)')
    print('4  -  Full Daily Update (Market + Financial + VCSC + Foreign Flows) [Includes all]')
    
    print('\n📈 QUARTERLY UPDATES:')
    print('5a - Banking: Fundamental Data Update (Fetch + Import)')
    print('5b - Other Sectors: Fundamental Data Update (Fetch + Import)')
    print('5c - Dividend Data Extraction & Update (Extract from Fundamentals)')
    print('6  -  Factor Calculation (All Sectors) [PLACEHOLDER - Phase 3]')
    print('7  -  Full Quarterly Update (Fundamentals + Dividends + Factors) [Combines 5a, 5b, 5c & 6]')
    
    print('\n🚀 PHASE 2 - INTERMEDIARY INFRASTRUCTURE:')
    print('8  -  Create/Update Fundamental Views (PREREQUISITE)')
    print('8a -    Create Enhanced Fundamental View (81 columns)')
    print('8b -    Refresh Research Tables (incremental)')
    print('9  -  Intermediary Calculations (requires step 8 first)')
    print('9a -    All Non-Financial Sectors (667 tickers)')
    print('9b -    Banking Sector (21 tickers) ✅ READY')
    print('9c -    Securities Sector (26 tickers) ✅ READY')
    print('9d -    Single Sector (choose specific sector)')
    print('9e -    Single Ticker (test specific ticker)')
    
    print('\n📊 DATA VALIDATION & REVIEW:')
    print('10 -  Data Review & Validation')
    print('10a -    Fundamental Data Extract (ANY ticker - auto-detects sector)')
    print('10b -    Intermediary Values Display (ANY ticker - auto-detects sector)')
    print('10c -    Factor Values Display (ticker + periods) [PLACEHOLDER]')
    print('10d -    Cross-validation Reports [PLACEHOLDER]')
    
    print('\n🔧 MAINTENANCE:')
    print('11 -  FS Mappings Update (All Sectors)')
    print('12 -  OHLCV Full Reload (From Specific Date)')
    print('13 -  Update Sector Mappings (Web Scraping from VietStock)')
    
    print('\n📚 HELP & INFO:')
    print('h  -  Show Workflow Documentation')
    print('i  -  Show System Information')
    
    print('\n0  -  Exit')

def show_workflow_documentation():
    """Display summary of workflow documentation"""
    clear_screen()
    print('\n' + '='*70)
    print('📚 VIETNAM FACTOR INVESTING PLATFORM: WORKFLOW DOCUMENTATION')
    print('='*70)
    
    print('''
DAILY WORKFLOW:
--------------
1. Run Market Data Update (OHLCV, ETFs/Indices)
   - Updates daily price and volume data for all stocks
   - Updates ETF and index values

2. Run Daily Financial Information Update
   - Updates shares outstanding and other daily financial metrics
   - Critical for accurate market cap calculations

2a. Run VCSC Complete Data Update (ENHANCED)
   - Fetches BOTH adjusted and unadjusted prices
   - Adjusted prices enable accurate backtesting with corporate actions
   - Market microstructure: match vs deal volumes, VWAP
   - Order book analytics: buy/sell imbalance, unmatched volumes  
   - Enhanced foreign flow data with ownership percentages
   - Daily shares outstanding (more frequent than quarterly)
   - Stores in vcsc_daily_data_complete table with 60+ fields

3. Run Foreign Flow Data Update
   - Fetches REAL-TIME foreign buy/sell volumes from API
   - Calculates net foreign flows and participation rates
   - Detects flow anomalies and generates trading signals
   - CRITICAL for Vietnam markets due to foreign ownership limits
   - NOTE: Must be run daily - no historical data available

4. Run Full Daily Update
   - Combines all three daily updates above
   - Recommended for production use
   
QUARTERLY WORKFLOW:
-----------------
1. Run Fundamental Data Update
   - Fetches and imports latest financial statements
   - Updates `fundamental_items` and `fundamental_values` tables

2. Run Factor Calculation
   - Computes sector-specific metrics (TTM, YoY, ratios)
   - Updates the `factor_values` table

MAINTENANCE:
-----------
- FS Mappings Update: Run when financial statement structures change
- OHLCV Full Reload: Use for correcting historical data issues
- Market Cap Table Update: Updates historical_daily_market_cap table with latest Wong API data
- Charter Capital Market Cap Update: Updates market cap based on charter capital data from financial statements

FOREIGN FLOW TRACKING:
---------------------
Foreign investor flows are included in the daily updates (option 3).
The system tracks:
- Foreign buy/sell volumes
- Net foreign flows
- Z-scores for anomaly detection
- Accumulation/distribution patterns
- Flow reversals and extreme movements

Note: Foreign flow data collection started April 30, 2025.
API only provides real-time data (no historical backfill).

For more details, refer to:
- docs/detailed_workflow.md
- docs/run_workflow_summary.md
- docs/market_cap_update_setup.md
- docs/charter_capital_market_cap_update.md
''')

def show_system_information():
    """Display system information and database status"""
    clear_screen()
    print('\n' + '='*70)
    print('ℹ️ VIETNAM FACTOR INVESTING PLATFORM: SYSTEM INFORMATION')
    print('='*70)
    
    print("\nEnvironment:")
    print(f"• Python Version: {sys.version.split()[0]}")
    print(f"• Working Directory: {os.getcwd()}")
    
    # Check if required scripts exist
    workflow_script = Path("scripts/run_workflow.py")
    print("\nCore Components:")
    print(f"• Workflow Script: {'✅ Present' if workflow_script.exists() else '❌ Missing'}")
    
    # List available pipeline scripts
    pipeline_dir = Path("src/pipelines/data_pipeline")
    metrics_dir = Path("src/pipelines/metrics_pipeline")
    
    if pipeline_dir.exists():
        data_pipelines = len(list(pipeline_dir.glob("*.py")))
        print(f"• Data Pipelines: {data_pipelines} scripts")
    
    if metrics_dir.exists():
        metric_pipelines = len(list(metrics_dir.glob("*.py")))
        print(f"• Metric Pipelines: {metric_pipelines} scripts")
    
    print("\nDatabase Configuration:")
    config_path = Path("config/database.yml")
    if config_path.exists():
        print("• Config file: ✅ Present")
        print("• Database: alphabeta")
        print("• Host: localhost")
    else:
        print("• Config file: ❌ Missing")
    
    print("\nFor detailed database status, use MySQL client:")
    print("  mysql -u root -p12345678 alphabeta")

def main():
    while True:
        show_main_menu()
        
        choice = input('\nSelect an option: ').strip().lower()
        
        if choice == '1':
            run_workflow_command("daily")
        
        elif choice == '2':
            run_workflow_command("daily-financial")
        
        elif choice == '2a':
            print("\n📊 VCSC COMPLETE Data Update includes:")
            print("  • ✨ ADJUSTED PRICES - Corporate action adjusted prices for accurate backtesting")
            print("  • Unadjusted daily prices (raw prices as traded)")
            print("  • Daily shares outstanding (vs quarterly from financials)")
            print("  • Market microstructure data:")
            print("    - Match vs Deal volume separation")
            print("    - VWAP (average price)")
            print("    - Buy/Sell trade counts and volumes")
            print("    - Order book imbalance metrics")
            print("  • Enhanced foreign trading analytics:")
            print("    - Separated by match/deal types")
            print("    - Foreign ownership percentages")
            print("    - Room utilization metrics")
            print("  • Price limits (ceiling/floor/reference prices)")
            print("\n🎯 This provides COMPLETE data for institutional-grade analysis!")
            confirm = input("\nProceed with VCSC Complete data update? (y/n): ").strip().lower()
            if confirm == 'y':
                run_workflow_command("vcsc-update")
        
        elif choice == '3':
            run_workflow_command("foreign-flows")
            
        elif choice == '4':
            run_workflow_command("full-daily")
        
        elif choice == '5a':
            run_workflow_command("banking-fundamentals")
        
        elif choice == '5b':
            run_workflow_command("fundamentals")
        
        elif choice == '5c':
            run_workflow_command("dividend-extraction")
        
        elif choice == '6':
            print("\n🚧 PLACEHOLDER - Phase 3 Factor Calculations")
            print("This will calculate final factors using pre-computed intermediaries:")
            print("  • ROAE_TTM, ROAA_TTM, Gross_Margin_TTM")
            print("  • Current_Ratio, Quick_Ratio, Asset_Turnover_TTM")
            print("  • DSO, DIO, DPO, CCC working capital metrics")
            print("  • And many more...")
            print("\n⏳ Coming soon in Phase 3!")
        
        elif choice == '7':
            print("\n📊 Running Full Quarterly Update...")
            print("This will:")
            print("  1. Fetch and import banking fundamentals")
            print("  2. Fetch and import non-banking fundamentals")
            print("  3. Extract and update dividend data")
            print("  4. Calculate factors for all sectors")
            print("\nThis process may take 30-60 minutes.")
            confirm = input("Continue? (y/n): ").strip().lower()
            
            if confirm == 'y':
                # Use the improved quarterly command that handles both banking and non-banking
                success = run_workflow_command("quarterly")
                
                if not success:
                    print("\n⚠️  Some parts of the quarterly update failed.")
                    print("You can run individual components:")
                    print("  - Option 5a: Banking fundamentals only")
                    print("  - Option 5b: Non-banking fundamentals only")
                    print("  - Option 5c: Dividend extraction only")
                    print("  - Option 6: Factor calculations only")
        
        elif choice == '8':
            print("\n🏗️ PHASE 2 - CREATE/UPDATE FUNDAMENTAL VIEWS (PREREQUISITE)")
            print("Choose view management option:")
            print("  8a - Create Enhanced Fundamental View (81 columns)")
            print("  8b - Refresh Research Tables (incremental)")
            sub_choice = input("\nSelect sub-option: ").strip().lower()
            
            if sub_choice == '8a':
                print("\n🏗️ Create Enhanced Fundamental View")
                print("This will create v_comprehensive_fundamental_items with 81 columns:")
                print("  • Enhanced income statement items (GrossProfit, SellingExpenses, etc.)")
                print("  • Enhanced balance sheet items (Cash, AccountsReceivable, etc.)")
                print("  • ✅ FIXED cash flow items (NetCFO, NetCFI, NetCFF)")
                print("  • Multi-sector charter capital mapping")
                print("\n⚠️ CRITICAL: This is REQUIRED before running intermediary calculations")
                confirm = input("\nProceed? (y/n): ").strip().lower()
                if confirm == 'y':
                    run_script("scripts/sector_views/create_nonfin_enhanced_view.py")
            
            elif sub_choice == '8b':
                print("\n🔄 Refresh Research Tables")
                print("This will refresh factor_values_research table:")
                print("  • Denormalized table optimized for fast research queries")
                print("  • Incremental refresh of latest data")
                print("  • Maintains year/quarter columns for easier analysis")
                confirm = input("\nProceed? (y/n): ").strip().lower()
                if confirm == 'y':
                    run_script("src/utils/refresh_research_tables.py", ["--incremental"])
            
        
        elif choice == '9':
            print("\n🚀 PHASE 2 - INTERMEDIARY CALCULATIONS (requires step 8 first)")
            print("Choose intermediary calculation option:")
            print("  9a - All Non-Financial Sectors (667 tickers)")
            print("  9b - Banking Sector (21 tickers) [PLACEHOLDER]")
            print("  9c - Securities Sector (26 tickers) ✅ READY")
            print("  9d - Single Sector")
            print("  9e - Single Ticker")
            sub_choice = input("\nSelect sub-option: ").strip().lower()
            
            if sub_choice == '9a':
                print("\n🏭 All Non-Financial Sectors Intermediary Calculations")
                print("This will process all 21 non-financial sectors (667 tickers):")
                print("  • Calculate TTM values for income statement and cash flow")
                print("  • Calculate 5-point averages for balance sheet items")
                print("  • Calculate working capital metrics (DSO, DIO, DPO, CCC)")
                print("  • Store in intermediary_calculations_enhanced table")
                print("\n⚠️ PREREQUISITE: Run option 8a first to create fundamental view")
                print("\n⏱️ Estimated time: 10-15 minutes")
                confirm = input("\nProceed? (y/n): ").strip().lower()
                if confirm == 'y':
                    run_script("scripts/intermediaries/all_nonfin_sectors_intermediary_calculator.py")
            
            elif sub_choice == '9b':
                print("\n🏦 Banking Sector Intermediary Calculations")
                print("This will process banking-specific intermediaries for 21 banking tickers:")
                print("  • NII_TTM, InterestIncome_TTM, InterestExpense_TTM")
                print("  • NetFeeIncome_TTM, TradingIncome_TTM, OperatingExpenses_TTM")
                print("  • NIM, LDR, Cost_of_Credit, Cost_Income_Ratio")
                print("  • CAR_Proxy, ROAA, ROAE, Fee_Income_Ratio")
                print("  • Stores in intermediary_calculations_banking table")
                print("\n⏱️ Estimated time: 3-5 minutes")
                confirm = input("\nProceed? (y/n): ").strip().lower()
                if confirm == 'y':
                    run_script("scripts/intermediaries/banking_sector_intermediary_calculator.py")
            
            elif sub_choice == '9c':
                print("\n📈 Securities Sector Intermediary Calculations")
                print("This will process securities-specific intermediaries:")
                print("  • Brokerage_Revenue_TTM, Trading_Income_TTM")
                print("  • Commission_Margins, Advisory_Revenue_Ratio") 
                print("  • Securities-specific metrics")
                
                confirm = input("\nProceed with securities intermediary calculations? (y/n): ").strip().lower()
                if confirm in ['y', 'yes']:
                    try:
                        print("\n🚀 Running securities intermediary calculations...")
                        result = subprocess.run(['python', 'scripts/intermediaries/securities_sector_intermediary_calculator.py'], 
                                              capture_output=True, text=True, cwd=os.getcwd())
                        
                        if result.returncode == 0:
                            print("✅ Securities intermediary calculations completed successfully!")
                            print(f"Output:\n{result.stdout}")
                        else:
                            print(f"❌ Error running securities intermediary calculations: {result.stderr}")
                    except Exception as e:
                        print(f"❌ Error: {e}")
                else:
                    print("❌ Securities intermediary calculations cancelled.")
            
            elif sub_choice == '9d':
                available_sectors = ['Technology', 'Real Estate', 'Construction', 'Agriculture', 
                                   'Ancillary Production', 'Construction Materials', 'Electrical Equipment',
                                   'Food & Beverage', 'Healthcare', 'Hotels & Tourism', 'Household Goods',
                                   'Industrial Services', 'Logistics', 'Machinery', 'Mining & Oil',
                                   'Plastics', 'Real Estate', 'Retail', 'Rubber Products', 'Seafood',
                                   'Utilities', 'Wholesale']
                print(f"\n📂 Available non-financial sectors:")
                for i, sector in enumerate(available_sectors, 1):
                    print(f"  {i:2d}. {sector}")
                
                try:
                    sector_idx = int(input("\nSelect sector number: ")) - 1
                    if 0 <= sector_idx < len(available_sectors):
                        selected_sector = available_sectors[sector_idx]
                        print(f"\n🎯 Processing {selected_sector} sector intermediaries...")
                        run_script("scripts/intermediaries/all_nonfin_sectors_intermediary_calculator.py", 
                                 ["--sectors", selected_sector])
                    else:
                        print("❌ Invalid sector number")
                except ValueError:
                    print("❌ Invalid input")
            
            elif sub_choice == '9e':
                ticker = input("\nEnter ticker symbol (e.g., FPT, VIC): ").strip().upper()
                if ticker:
                    print(f"\n🎯 Processing {ticker} intermediary calculations...")
                    run_script("scripts/intermediaries/all_nonfin_sectors_intermediary_calculator.py", 
                             ["--ticker", ticker])
                else:
                    print("❌ Invalid ticker")
        
        # Direct handlers for 9a-9e (user convenience)
        elif choice == '9a':
            print("\n🏭 All Non-Financial Sectors Intermediary Calculations")
            print("This will process all 21 non-financial sectors (667 tickers):")
            print("  • Calculate TTM values for income statement and cash flow")
            print("  • Calculate 5-point averages for balance sheet items")
            print("  • Calculate working capital metrics (DSO, DIO, DPO, CCC)")
            print("  • Store in intermediary_calculations_enhanced table")
            print("\n⚠️ PREREQUISITE: Run option 8a first to create fundamental view")
            print("\n⏱️ Estimated time: 10-15 minutes")
            confirm = input("\nProceed? (y/n): ").strip().lower()
            if confirm == 'y':
                run_script("scripts/intermediaries/all_nonfin_sectors_intermediary_calculator.py")
        
        elif choice == '9b':
            print("\n🏦 Banking Sector Intermediary Calculations")
            print("This will process banking-specific intermediaries for 21 banking tickers:")
            print("  • NII_TTM, InterestIncome_TTM, InterestExpense_TTM")
            print("  • NetFeeIncome_TTM, TradingIncome_TTM, OperatingExpenses_TTM")
            print("  • NIM, LDR, Cost_of_Credit, Cost_Income_Ratio")
            print("  • CAR_Proxy, ROAA, ROAE, Fee_Income_Ratio")
            print("  • Stores in intermediary_calculations_banking table")
            print("\n⏱️ Estimated time: 3-5 minutes")
            confirm = input("\nProceed? (y/n): ").strip().lower()
            if confirm == 'y':
                run_script("scripts/intermediaries/banking_sector_intermediary_calculator.py")
        
        elif choice == '9c':
            print("\n📈 Securities Sector Intermediary Calculations [PLACEHOLDER]")
            print("This will process securities-specific intermediaries:")
            print("  • Brokerage_Revenue_TTM, Trading_Income_TTM")
            print("  • Commission_Margins, Advisory_Revenue_Ratio")
            print("  • Securities-specific metrics")
            print("\n⏳ Coming soon!")
        
        elif choice == '9d':
            available_sectors = ['Technology', 'Real Estate', 'Construction', 'Agriculture', 
                               'Ancillary Production', 'Construction Materials', 'Electrical Equipment',
                               'Food & Beverage', 'Healthcare', 'Hotels & Tourism', 'Household Goods',
                               'Industrial Services', 'Logistics', 'Machinery', 'Mining & Oil',
                               'Plastics', 'Real Estate', 'Retail', 'Rubber Products', 'Seafood',
                               'Utilities', 'Wholesale']
            print(f"\n📂 Available non-financial sectors:")
            for i, sector in enumerate(available_sectors, 1):
                print(f"  {i:2d}. {sector}")
            
            try:
                sector_idx = int(input("\nSelect sector number: ")) - 1
                if 0 <= sector_idx < len(available_sectors):
                    selected_sector = available_sectors[sector_idx]
                    print(f"\n🎯 Processing {selected_sector} sector intermediaries...")
                    run_script("scripts/intermediaries/all_nonfin_sectors_intermediary_calculator.py", 
                             ["--sectors", selected_sector])
                else:
                    print("❌ Invalid sector number")
            except ValueError:
                print("❌ Invalid input")
        
        elif choice == '9e':
            ticker = input("\nEnter ticker symbol (e.g., FPT, VIC): ").strip().upper()
            if ticker:
                print(f"\n🎯 Processing {ticker} intermediary calculations...")
                run_script("scripts/intermediaries/all_nonfin_sectors_intermediary_calculator.py", 
                         ["--ticker", ticker])
            else:
                print("❌ Invalid ticker")
        
        elif choice == '10a':
            ticker = input("\n🎯 Enter ANY ticker (e.g., FPT, VCB, SSI, BMI): ").strip().upper()
            if ticker:
                print(f"\n📋 Extracting fundamental data for {ticker}...")
                run_fundamental_extract(ticker)
            else:
                print("❌ Invalid ticker")
        
        elif choice == '10b':
            ticker = input("\n🎯 Enter ANY ticker (e.g., FPT, VCB, NLG, OCB): ").strip().upper()
            if ticker:
                print(f"\n📊 Displaying intermediary values for {ticker}...")
                run_intermediary_display(ticker)
            else:
                print("❌ Invalid ticker")
        
        elif choice == '11':
            run_workflow_command("update-fs-mappings")
        
        elif choice == '12':
            date = input("\nEnter start date for reload (YYYY-MM-DD): ")
            run_workflow_command("full-ohlcv-reload", [date])
        
        elif choice == '13':
            print("\n🌐 Sector Mapping Update from VietStock")
            print("This will:")
            print("  • Use Selenium to scrape sector data from finance.vietstock.vn")
            print("  • Map raw sector names to standardized names")
            print("  • Update the master_info table with current mappings")
            print("\nNote: This requires Chrome browser and will auto-install ChromeDriver")
            confirm = input("\nProceed with sector mapping update? (y/n): ").strip().lower()
            if confirm == 'y':
                run_workflow_command("update-sectors")
        
        elif choice == 'h':
            show_workflow_documentation()
        
        elif choice == 'i':
            show_system_information()
        
        elif choice == '0':
            print("\nExiting Vietnam Factor Investing Platform. Goodbye!")
            break
        
        else:
            print("\n❌ Invalid option. Please try again.")
        
        input("\nPress Enter to continue...")

if __name__ == '__main__':
    main()
