#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Master script to run various parts of the Vietnam Factor Investing data workflow.
Updated to run both non-financial and real estate scripts separately.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional, List
import os # Added for PIPELINE_DIR

# Third-party libraries for better terminal output
try:
    from tqdm import tqdm
    from termcolor import colored, cprint
except ImportError:
    print("Error: Required packages 'tqdm' and 'termcolor' are not installed.", file=sys.stderr)
    print("Please install them using: pip install tqdm termcolor", file=sys.stderr)
    sys.exit(1)

# Ensure the script is run from the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if Path.cwd() != PROJECT_ROOT:
    print(f"Error: Please run this script from the project root: {PROJECT_ROOT}", file=sys.stderr)
    sys.exit(1)

# Add project root to Python path to allow for `src.` imports
sys.path.insert(0, str(PROJECT_ROOT))

# Import refactored pipelines
from src.pipelines.market_cap_pipeline import run_market_cap_pipeline, run_market_cap_update

# Define script locations relative to project root
PIPELINE_DIR = PROJECT_ROOT / 'src' / 'pipelines' / 'data_pipeline'
METRICS_DIR = PROJECT_ROOT / 'src' / 'pipelines' / 'metrics_pipeline'
SCRIPT_DIR = PROJECT_ROOT / "scripts" # If you have other scripts here

# Ensure PIPELINE_DIR is in PYTHONPATH if scripts there import relative to src
sys.path.insert(0, str(PROJECT_ROOT))

# -------------------- Helper Functions --------------------

def file_path_to_module_path(path: Path) -> str:
    """Converts a Path object (e.g., src/pipelines/foo.py) to a module string (e.g., src.pipelines.foo)."""
    # Make path relative to project root
    relative_path = path.relative_to(PROJECT_ROOT)
    # Remove .py extension and replace slashes with dots
    return str(relative_path).replace('.py', '').replace(os.path.sep, '.')

def run_script(script_path: Path, args: Optional[List[str]] = None, description: Optional[str] = None, 
               show_live_output: bool = False):
    """
    Runs a Python script as a module using `python -m`, capturing its output.
    This ensures robust resolution of internal project imports.
    
    Parameters:
    -----------
    show_live_output : bool
        If True, displays live output instead of capturing it.
        Use for long operations like OHLCV full reload.
    """
    if description:
        tqdm.write(f"  -> Executing: {description}...")
        
    module_path = file_path_to_module_path(script_path)
    command = [sys.executable, "-m", module_path]
    if args:
        command.extend(args)
        
    # The environment setup is still good practice as a fallback.
    env = os.environ.copy()
    env['PYTHONPATH'] = str(PROJECT_ROOT) + os.pathsep + env.get('PYTHONPATH', '')
        
    try:
        if show_live_output:
            # For long operations, show output in real-time
            result = subprocess.run(
                command, 
                check=True,
                env=env,
                text=True
            )
            return True
        else:
            # For quick operations, capture output
            result = subprocess.run(
                command, 
                check=True, 
                text=True, 
                capture_output=True,
                encoding='utf-8',
                env=env  # Pass the modified environment
            )
            return True
    except subprocess.CalledProcessError as e:
        tqdm.write(colored(f"!!! Error in module: {module_path} !!!", 'red'))
        tqdm.write(colored(f"Return code: {e.returncode}", 'red'))
        # Print captured stdout/stderr only on error
        if not show_live_output and e.stdout:
            tqdm.write(colored("--- STDOUT ---", "yellow"))
            tqdm.write(e.stdout)
        if not show_live_output and e.stderr:
            tqdm.write(colored("--- STDERR ---", "yellow"))
            tqdm.write(e.stderr)
        return False
    except FileNotFoundError:
        tqdm.write(colored(f"!!! Script not found: {script_path} !!!", 'red'))
        return False

# -------------------- Workflow Functions --------------------

def run_daily_market_data():
    """Runs the daily OHLCV and ETF/Index data pipelines."""
    tqdm.write(colored("\n[1/3] Starting Daily Market Data Update...", 'cyan', attrs=['bold']))
    
    ohlcv_success = run_script(
        PIPELINE_DIR / 'ohlcv_data_pipeline.py', 
        ['--run-daily'],
        description="Daily OHLCV price/volume update"
    )
    
    etf_success = run_script(
        PIPELINE_DIR / 'etfs_indices_data_pipeline.py',
        description="Daily ETFs & Indices update"
    )
    
    overall_success = ohlcv_success and etf_success
    
    if overall_success:
        tqdm.write(colored("✅ Daily Market Data Update... SUCCESS", 'green'))
    else:
        tqdm.write(colored("❌ Daily Market Data Update... FAILED", 'red'))
        
    return overall_success

def run_daily_financial_data():
    """Runs the daily financial metrics fetcher and pipeline."""
    tqdm.write(colored("\n[2/3] Starting Daily Financial Information Ingestion...", 'cyan', attrs=['bold']))
    
    fetcher_success = run_script(
        PIPELINE_DIR / 'financial_metrics_fetcher.py',
        description="Fetch daily financial metrics (e.g., shares outstanding)"
    )
    
    pipeline_success = False
    if fetcher_success:
        pipeline_success = run_script(
            PIPELINE_DIR / 'financial_metrics_pipeline.py',
            description="Process and import daily financial metrics"
        )
    else:
        tqdm.write(colored("  -> Skipping pipeline due to fetcher failure.", 'yellow'))
        
    overall_success = fetcher_success and pipeline_success

    if overall_success:
        tqdm.write(colored("✅ Daily Financial Information Ingestion... SUCCESS", 'green'))
    else:
        tqdm.write(colored("❌ Daily Financial Information Ingestion... FAILED", 'red'))
        
    return overall_success

def run_banking_fundamental_updates():
    """
    Runs the dedicated fundamental data fetcher for the banking sector, then the main importer.
    """
    tqdm.write(colored("\n[1/1] Starting Banking Fundamental Data Update...", 'cyan', attrs=['bold']))
    
    # --- Step 1: Fetch Banking Sector Data ---
    tqdm.write(colored("  -> Fetching data for BANKING sector...", 'blue'))
    banking_fetch_success = run_script(
        PIPELINE_DIR / 'run_banking_fetcher.py',
        description="Run dedicated fetcher for all banking companies"
    )

    # --- Step 2: Import All Fetched Data ---
    import_success = False
    if banking_fetch_success:
        tqdm.write(colored("  -> Importing new banking fundamental data into database...", 'blue'))
        import_success = run_script(
            PIPELINE_DIR / 'fundamental_data_importer.py',
            description="Upsert all new JSON data into fundamental_values table"
        )
    else:
        tqdm.write(colored("!!! Banking fetcher failed, skipping import. !!!", 'red'))

    if import_success:
        tqdm.write(colored("✅ Banking Fundamental Data Update completed successfully.", 'green'))
    else:
        tqdm.write(colored("❌ Banking Fundamental Data Update FAILED. Check logs.", 'red'))
        
    return import_success

def run_fundamental_updates():
    """
    Runs the fundamental data fetcher and importer for NON-BANKING sectors.
    """
    tqdm.write(colored("\n[1/1] Starting Non-Banking Fundamental Data Update...", 'cyan', attrs=['bold']))
    
    # --- Step 1: Fetch Other Sectors Data ---
    tqdm.write(colored("  -> Fetching data for NON-FINANCIAL/OTHER sectors...", 'blue'))
    fetch_success = run_script(
        PIPELINE_DIR / 'fundamental_data_fetcher.py',
        description="Run generic fetcher for all other companies"
    )
    
    # --- Step 2: Import All Fetched Data ---
    import_success = False
    if fetch_success:
        tqdm.write(colored("  -> Importing all new fundamental data into database...", 'blue'))
        import_success = run_script(
            PIPELINE_DIR / 'fundamental_data_importer.py',
            description="Upsert all new JSON data into fundamental_values table"
        )
    else:
        tqdm.write(colored("!!! Generic fetcher failed, skipping import. !!!", 'red'))

    if import_success:
        tqdm.write(colored("✅ Non-Banking Fundamental Data Update completed successfully.", 'green'))
    else:
        tqdm.write(colored("❌ Non-Banking Fundamental Data Update FAILED. Check logs.", 'red'))
        
    return import_success

def run_factor_calculation():
    """Runs all factor calculation scripts."""
    tqdm.write(colored("\n[2/2] Starting Factor Calculation...", 'cyan', attrs=['bold']))
    success = True
    
    # List of calculators to run
    calculators = [
        ('Banking', 'banking_metrics_dynamic_upsert.py'),
        ('Insurance', 'insurance_metrics_dynamic_upsert.py'),
        ('Securities', 'securities_metrics_dynamic_upsert.py'),
        ('Non-Financial', 'non_fin_all_sectors_metrics_dynamic_upsert.py'),
        ('Real Estate', 'realestate_metrics_dynamic_upsert.py')
    ]
    
    # Run each calculator with progress
    for idx, (name, script) in enumerate(calculators, 1):
        tqdm.write(colored(f"\n[{idx}/5] Running {name} calculator...", 'yellow'))
        success &= run_script(METRICS_DIR / script)
    
    tqdm.write(colored("\nFactor Calculation completed.", 'green', attrs=['bold']))
    return success

def run_full_quarterly_update():
    """Runs the full fundamental update (banking + non-banking), dividend extraction, and factor calculation sequence."""
    tqdm.write(colored("\n=== FULL QUARTERLY UPDATE CYCLE ===", 'cyan', attrs=['bold']))
    
    # Step 1: Banking fundamentals
    tqdm.write(colored("\n[1/4] Banking Fundamental Update...", 'cyan', attrs=['bold']))
    banking_success = run_banking_fundamental_updates()
    
    # Step 2: Non-banking fundamentals
    tqdm.write(colored("\n[2/4] Non-Banking Fundamental Update...", 'cyan', attrs=['bold']))
    nonbanking_success = run_fundamental_updates()
    
    # Step 3: Dividend extraction (only if at least one fundamental update succeeded)
    dividend_success = False
    if banking_success or nonbanking_success:
        tqdm.write(colored("\n[3/4] Dividend Data Extraction...", 'cyan', attrs=['bold']))
        dividend_success = run_dividend_extraction()
    else:
        tqdm.write(colored("\n❌ Both fundamental updates failed. Skipping dividend extraction.", 'red'))
    
    # Step 4: Factor calculations (only if at least one fundamental update succeeded)
    factor_success = False
    if banking_success or nonbanking_success:
        tqdm.write(colored("\n[4/4] Factor Calculation for All Sectors...", 'cyan', attrs=['bold']))
        factor_success = run_factor_calculation()
    else:
        tqdm.write(colored("\n❌ No successful fundamental updates. Skipping factor calculation.", 'red'))
    
    # Summary report
    tqdm.write(colored("\n=== QUARTERLY UPDATE SUMMARY ===", 'yellow', attrs=['bold']))
    tqdm.write(f"Banking Fundamentals:     {'✅ Success' if banking_success else '❌ Failed'}")
    tqdm.write(f"Non-Banking Fundamentals: {'✅ Success' if nonbanking_success else '❌ Failed'}")
    tqdm.write(f"Dividend Extraction:      {'✅ Success' if dividend_success else '❌ Failed'}")
    tqdm.write(f"Factor Calculations:      {'✅ Success' if factor_success else '❌ Failed'}")
    
    overall_success = banking_success and nonbanking_success and dividend_success and factor_success
    if overall_success:
        tqdm.write(colored("\n✅ Full Quarterly Update completed successfully!", 'green', attrs=['bold']))
    else:
        tqdm.write(colored("\n⚠️  Quarterly Update completed with some failures. Check logs.", 'yellow'))
        
    return overall_success

def run_full_daily_update():
    """Runs daily market data, financial data, VCSC data, and foreign flow updates."""
    cprint("\n" + "="*70, 'yellow')
    cprint("🚀 STARTING FULL DAILY UPDATE CYCLE 🚀", 'yellow', attrs=['bold'])
    cprint("="*70, 'yellow')

    market_success = run_daily_market_data()
    financial_success = run_daily_financial_data()
    vcsc_success = run_vcsc_data_update()  # NEW: VCSC unadjusted data
    foreign_flow_success = run_foreign_flow_update()
    
    cprint("\n" + "="*70, 'yellow')
    cprint("✨ DAILY UPDATE CYCLE COMPLETE ✨", 'yellow', attrs=['bold'])
    
    if market_success and financial_success and vcsc_success and foreign_flow_success:
        cprint("\n✅ Overall Status: ALL TASKS COMPLETED SUCCESSFULLY!", 'green', attrs=['bold'])
    else:
        cprint("\n❌ Overall Status: ONE OR MORE TASKS FAILED.", 'red', attrs=['bold'])
        cprint("   Please review the error messages above for details.", 'red')
    
    cprint("="*70, 'yellow')
    
    return market_success and financial_success and vcsc_success and foreign_flow_success

def run_ohlcv_full_reload(start_date: str):
    """Runs the OHLCV pipeline with full reload, showing live progress."""
    tqdm.write(colored(f"\n[1/1] Starting OHLCV Full Reload from {start_date}...", 'cyan', attrs=['bold']))
    tqdm.write(colored("=" * 70, 'cyan'))
    tqdm.write(colored("📊 QUARTERLY FULL RELOAD - CORPORATE ACTION ADJUSTMENT", 'yellow', attrs=['bold']))
    tqdm.write(colored("=" * 70, 'cyan'))
    tqdm.write("")
    tqdm.write("This process will:")
    tqdm.write("  1. Download complete historical data for ALL stocks")
    tqdm.write("  2. Replace existing data with adjusted prices (for splits/dividends)")
    tqdm.write("  3. Import all data into the database")
    tqdm.write("")
    tqdm.write(colored("⚠️  This is the CORRECT approach for handling corporate actions!", 'green'))
    tqdm.write(colored("⏳ This may take 10-30 minutes. Progress will be shown below:", 'yellow'))
    tqdm.write("")
    
    # Show live output for this long operation
    success = run_script(
        PIPELINE_DIR / 'ohlcv_data_pipeline.py', 
        ['--full-reload-start-date', start_date],
        description="Full historical OHLCV reload with corporate action adjustments",
        show_live_output=True  # This enables live progress display
    )
    
    if success:
        tqdm.write("")
        tqdm.write(colored("✅ OHLCV Full Reload completed successfully!", 'green', attrs=['bold']))
        tqdm.write("All historical prices have been refreshed with latest adjustments.")
    else:
        tqdm.write("")
        tqdm.write(colored("❌ OHLCV Full Reload failed. Check logs for details.", 'red'))
    
    return success

def run_fs_mappings_update():
    """Runs the script to update all FS mappings."""
    tqdm.write(colored("\n[2/2] Starting FS Mappings Update...", 'cyan', attrs=['bold']))
    success = run_script(PIPELINE_DIR / 'run_all_fs_mappings.py', ['--sync-db'])
    tqdm.write("FS Mappings Update completed.")
    return success

def run_historical_market_cap_update():
    """Runs the refactored historical market cap update pipeline."""
    tqdm.write(colored("\n[1/1] Starting Historical Market Cap Table Update...", 'cyan', attrs=['bold']))
    # Call the new, correct function
    success = run_market_cap_update()
    tqdm.write("Historical Market Cap Table Update completed.")
    return success

def run_charter_capital_market_cap_update():
    """Runs the script to update market cap based on charter capital data from financial statements."""
    tqdm.write(colored("\n[1/1] Starting Charter Capital-Based Market Cap Update...", 'cyan', attrs=['bold']))
    script_path = PROJECT_ROOT / 'scripts' / 'update_market_cap_from_charter_capital.py'
    success = run_script(script_path)
    tqdm.write("Charter Capital-Based Market Cap Update completed.")
    return success

def run_foreign_flow_update():
    """Runs the foreign flow data pipeline for daily foreign investor trading data."""
    tqdm.write(colored("\n[3/3] Starting Foreign Flow Data Update...", 'cyan', attrs=['bold']))
    success = run_script(
        PIPELINE_DIR / 'foreign_flow_data_pipeline.py',
        description="Fetch, process, and store foreign investor flows"
    )
    
    if success:
        tqdm.write(colored("✅ Foreign Flow Data Update... SUCCESS", 'green'))
    else:
        tqdm.write(colored("❌ Foreign Flow Data Update... FAILED", 'red'))

    return success

def run_foreign_flow_setup():
    """Create foreign flow database tables (one-time setup)."""
    tqdm.write(colored("\n[1/1] Starting Foreign Flow Database Setup...", 'cyan', attrs=['bold']))
    success = run_script(PIPELINE_DIR / 'foreign_flow_data_pipeline.py', ['--create-tables'])
    tqdm.write("Foreign Flow Database Setup completed.")
    return success

def run_dividend_extraction():
    """Run dividend data extraction from fundamental_values and update dividend_payments table."""
    tqdm.write(colored("\n[1/1] Starting Dividend Data Extraction...", 'cyan', attrs=['bold']))
    tqdm.write(colored("This will extract dividend data from fundamental_values table", 'blue'))
    tqdm.write(colored("and update the dividend_payments table.", 'blue'))
    
    # Path to dividend pipeline script
    dividend_script = PROJECT_ROOT / 'src' / 'dividend_research' / 'dividend_pipeline.py'
    
    if not dividend_script.exists():
        tqdm.write(colored(f"❌ Dividend pipeline script not found at: {dividend_script}", 'red'))
        return False
    
    success = run_script(
        dividend_script,
        description="Extract dividends from fundamentals and update dividend_payments table"
    )
    
    if success:
        tqdm.write(colored("✅ Dividend Data Extraction... SUCCESS", 'green'))
        tqdm.write(colored("Dividend payments table has been updated with latest data.", 'green'))
    else:
        tqdm.write(colored("❌ Dividend Data Extraction... FAILED", 'red'))
        
    return success

def run_vcsc_data_update():
    """Run VCSC data update with COMPLETE data including adjusted prices and microstructure."""
    tqdm.write(colored("\n[1/1] Starting VCSC Complete Data Update...", 'cyan', attrs=['bold']))
    tqdm.write(colored("Fetching complete price data (adjusted & unadjusted) from VCSC API", 'blue'))
    tqdm.write(colored("This includes microstructure data, foreign ownership, and order book metrics", 'blue'))
    tqdm.write(colored("This may take several minutes for all tickers...", 'yellow'))
    
    # Show live output for VCSC fetcher
    success = run_script(
        PIPELINE_DIR / 'vcsc_data_fetcher_complete.py',
        description="Fetch complete market data from VCSC",
        show_live_output=True  # Enable live output
    )
    
    if success:
        tqdm.write(colored("✅ VCSC Data Update... SUCCESS", 'green'))
    else:
        tqdm.write(colored("❌ VCSC Data Update... FAILED", 'red'))
        
    return success

def run_sector_mapping_update():
    """Run sector mapping update from VietStock web scraping."""
    tqdm.write(colored("\n[1/1] Starting Sector Mapping Update...", 'cyan', attrs=['bold']))
    tqdm.write(colored("This will scrape sector information from VietStock.vn", 'blue'))
    tqdm.write(colored("and update the master_info table with current mappings.", 'blue'))
    tqdm.write("")
    tqdm.write(colored("Options:", 'yellow'))
    tqdm.write("  --dry-run     : Preview changes without updating database")
    tqdm.write("  --save-json   : Save scraped data to JSON file")
    tqdm.write("  --report-only : Generate comparison report only")
    tqdm.write("")
    
    # Ask user for options
    options = []
    
    dry_run = input("Run in dry-run mode? (y/N): ").strip().lower() == 'y'
    if dry_run:
        options.append('--dry-run')
        tqdm.write(colored("✓ Dry-run mode enabled - no database changes will be made", 'yellow'))
    
    save_json = input("Save scraped data to JSON? (y/N): ").strip().lower() == 'y'
    if save_json:
        options.append('--save-json')
    
    report_only = input("Generate report only? (y/N): ").strip().lower() == 'y'
    if report_only:
        options.append('--report-only')
    
    # Run the scraper
    success = run_script(
        PIPELINE_DIR / 'sector_mapping_scraper.py',
        args=options,
        description="Scrape and update sector mappings",
        show_live_output=True
    )
    
    if success:
        tqdm.write(colored("✅ Sector Mapping Update... SUCCESS", 'green'))
    else:
        tqdm.write(colored("❌ Sector Mapping Update... FAILED", 'red'))
        
    return success

# -------------------- Main Argparse Setup --------------------

def main():
    parser = argparse.ArgumentParser(description="Run parts of the Vietnam Factor Investing data workflow.")
    subparsers = parser.add_subparsers(dest='command', help='Workflow command to execute', required=True)

    # Subparser for daily tasks
    parser_daily = subparsers.add_parser('daily', help='Run daily market data updates (OHLCV, ETFs/Indices).')
    parser_daily.set_defaults(func=run_daily_market_data)
    
    # Subparser for daily financial data
    parser_daily_financial = subparsers.add_parser('daily-financial', 
                                                 help='Run daily financial metrics fetch and import (shares outstanding, etc.)')
    parser_daily_financial.set_defaults(func=run_daily_financial_data)
    
    # Subparser for complete daily update
    parser_full_daily = subparsers.add_parser('full-daily', 
                                            help='Run both market data and financial data updates (complete daily process)')
    parser_full_daily.set_defaults(func=run_full_daily_update)

    # Subparser for banking fundamental fetch + import
    parser_banking_fundamentals = subparsers.add_parser('banking-fundamentals', help='Run banking fundamental data fetch and import.')
    parser_banking_fundamentals.set_defaults(func=run_banking_fundamental_updates)
    
    # Subparser for fundamental fetch + import
    parser_fundamentals = subparsers.add_parser('fundamentals', help='Run fundamental data fetch and import for non-banking sectors.')
    parser_fundamentals.set_defaults(func=run_fundamental_updates)

    # Subparser for dividend extraction
    parser_dividends = subparsers.add_parser('dividend-extraction', 
                                            help='Extract dividend data from fundamentals and update dividend_payments table.')
    parser_dividends.set_defaults(func=run_dividend_extraction)

    # Subparser for factor calculation
    parser_factors = subparsers.add_parser('factors', help='Run factor calculation (run AFTER fundamentals).')
    parser_factors.set_defaults(func=run_factor_calculation)

    # Subparser for the full quarterly cycle
    parser_quarterly = subparsers.add_parser('quarterly', help='Run full fundamentals -> factors update cycle.')
    parser_quarterly.set_defaults(func=run_full_quarterly_update)

    # Subparser for OHLCV full reload
    parser_reload = subparsers.add_parser('full-ohlcv-reload', help='Run OHLCV full reload from a specific date.')
    parser_reload.add_argument('start_date', type=str, help='Start date for full reload (YYYY-MM-DD).')
    parser_reload.set_defaults(func=lambda args: run_ohlcv_full_reload(args.start_date)) # Use lambda to pass args

    # Subparser for FS mappings update
    parser_fsmap = subparsers.add_parser('update-fs-mappings', help='Run the FS mapping update script.')
    parser_fsmap.set_defaults(func=run_fs_mappings_update)

    # Subparser for historical market cap update
    parser_mktcap = subparsers.add_parser('update-market-cap', help='Run the historical market cap update pipeline.')
    parser_mktcap.set_defaults(func=run_historical_market_cap_update)
    
    # Subparser for charter capital market cap update
    parser_charter_mktcap = subparsers.add_parser('update-charter-cap', help='Run the charter capital based market cap update.')
    parser_charter_mktcap.set_defaults(func=run_charter_capital_market_cap_update)
    
    # Subparser for foreign flow data
    parser_foreign_flow = subparsers.add_parser('foreign-flows',
                                               help='Run daily foreign investor flow data update (real-time data only).')
    parser_foreign_flow.set_defaults(func=run_foreign_flow_update)
    
    # Subparser for foreign flow setup
    parser_ff_setup = subparsers.add_parser('foreign-flows-setup',
                                           help='Create foreign flow database tables (one-time setup).')
    parser_ff_setup.set_defaults(func=run_foreign_flow_setup)
    
    # Subparser for VCSC data update
    parser_vcsc = subparsers.add_parser('vcsc-update',
                                        help='Run VCSC data update for unadjusted prices and daily shares outstanding.')
    parser_vcsc.set_defaults(func=run_vcsc_data_update)
    
    # Subparser for sector mapping update
    parser_sector = subparsers.add_parser('update-sectors',
                                         help='Update sector mappings from VietStock web scraping.')
    parser_sector.set_defaults(func=run_sector_mapping_update)

    args = parser.parse_args()

    # Execute the function associated with the chosen subcommand
    success = False
    if args.command in ['full-ohlcv-reload']:
        success = args.func(args) # Pass args only to the function that needs it
    else:
        success = args.func() # Call other functions without arguments

    if not success:
        sys.exit(1) # Exit with error code if function returned False

if __name__ == "__main__":
    main()