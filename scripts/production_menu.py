#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Production Menu Interface for Vietnam Factor Investing Platform
==============================================================
Author: Duc Nguyen
Date: July 27, 2025
Status: PRODUCTION READY

This production menu provides a structured interface for the complete
quantitative workflow from raw data updates to factor generation.

CRITICAL WORKFLOW:
1. Daily Updates (Raw Data) - Market data, financial info, VCSC, foreign flows
2. Quarterly Updates (Raw Data) - Financial statements for all sectors
3. Data Processing - Views creation and intermediary calculations  
4. Factor Generation - QVM factor scores using production engine
5. Backtesting & Execution - Strategy validation and order generation

IMPORTANT: Q2 2025 financial statements are due soon. Ensure all quarterly
update processes are thoroughly tested and validated.
"""

import sys
import subprocess
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List
import time
import pandas as pd

# Ensure we're running from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if Path.cwd() != PROJECT_ROOT:
    os.chdir(PROJECT_ROOT)
    print(f"Changed working directory to: {PROJECT_ROOT}")

class Colors:
    """Terminal color codes for better visibility"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header(text: str, symbol: str = "="):
    """Print a formatted header"""
    print(f"\n{Colors.CYAN}{symbol * 70}{Colors.ENDC}")
    print(f"{Colors.CYAN}{Colors.BOLD}{text}{Colors.ENDC}")
    print(f"{Colors.CYAN}{symbol * 70}{Colors.ENDC}")

def print_success(text: str):
    """Print success message in green"""
    print(f"{Colors.GREEN}✅ {text}{Colors.ENDC}")

def print_error(text: str):
    """Print error message in red"""
    print(f"{Colors.RED}❌ {text}{Colors.ENDC}")

def print_warning(text: str):
    """Print warning message in yellow"""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.ENDC}")

def print_info(text: str):
    """Print info message in blue"""
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.ENDC}")

def get_quarterly_reporting_status():
    """
    Calculate dynamic quarterly reporting status based on current date.
    Returns information about current quarter, next expected quarter, and reporting status.
    """
    current_date = datetime.now()
    current_year = current_date.year
    
    # Define quarter end dates (same logic as QVM engine)
    quarter_ends = {
        1: datetime(current_year, 3, 31),   # Q1 ends Mar 31
        2: datetime(current_year, 6, 30),   # Q2 ends Jun 30  
        3: datetime(current_year, 9, 30),   # Q3 ends Sep 30
        4: datetime(current_year, 12, 31)   # Q4 ends Dec 31
    }
    
    # Reporting lag: 45 days after quarter end (same as QVM engine)
    reporting_lag = 45
    
    # Find the most recent quarter that should have data available
    available_quarters = []
    for quarter, end_date in quarter_ends.items():
        publish_date = end_date + timedelta(days=reporting_lag)
        if publish_date <= current_date:
            available_quarters.append((current_year, quarter, end_date, publish_date))
    
    # Also check previous year Q4
    prev_year_q4_end = datetime(current_year - 1, 12, 31)
    prev_year_q4_publish = prev_year_q4_end + timedelta(days=reporting_lag)
    if prev_year_q4_publish <= current_date:
        available_quarters.append((current_year - 1, 4, prev_year_q4_end, prev_year_q4_publish))
    
    # Sort by publish date to get most recent
    if available_quarters:
        available_quarters.sort(key=lambda x: x[3], reverse=True)
        latest_year, latest_quarter, latest_end, latest_publish = available_quarters[0]
    else:
        # Fallback to previous year Q4
        latest_year = current_year - 1
        latest_quarter = 4
        latest_end = datetime(latest_year, 12, 31)
        latest_publish = latest_end + timedelta(days=reporting_lag)
    
    # Find next expected quarter
    next_quarter_info = None
    for quarter, end_date in quarter_ends.items():
        publish_date = end_date + timedelta(days=reporting_lag)
        if publish_date > current_date:
            next_quarter_info = (current_year, quarter, end_date, publish_date)
            break
    
    # If no quarter found in current year, use Q1 of next year
    if next_quarter_info is None:
        next_year = current_year + 1
        next_end = datetime(next_year, 3, 31)
        next_publish = next_end + timedelta(days=reporting_lag)
        next_quarter_info = (next_year, 1, next_end, next_publish)
    
    next_year, next_quarter, next_end, next_publish = next_quarter_info
    
    # Calculate days until next quarter data is due
    days_until_next = (next_publish - current_date).days
    
    # Determine urgency level
    if days_until_next <= 7:
        urgency = "URGENT"
        urgency_color = Colors.RED
    elif days_until_next <= 30:
        urgency = "HIGH"
        urgency_color = Colors.YELLOW
    else:
        urgency = "NORMAL"
        urgency_color = Colors.GREEN
    
    return {
        'current_date': current_date,
        'latest_available_quarter': f"Q{latest_quarter} {latest_year}",
        'latest_quarter_year': latest_year,
        'latest_quarter_num': latest_quarter,
        'next_expected_quarter': f"Q{next_quarter} {next_year}",
        'next_quarter_year': next_year,
        'next_quarter_num': next_quarter,
        'next_quarter_end': next_end,
        'next_quarter_publish': next_publish,
        'days_until_next': days_until_next,
        'urgency': urgency,
        'urgency_color': urgency_color,
        'is_quarter_due_soon': days_until_next <= 30
    }

def run_workflow_command(command: str, args: Optional[List[str]] = None) -> bool:
    """Run a workflow command via run_workflow.py"""
    cmd = ["python", "scripts/run_workflow.py", command]
    if args:
        cmd.extend(args)
    
    print(f"\n{Colors.BLUE}Running: {' '.join(cmd)}{Colors.ENDC}")
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, text=True, capture_output=False)
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print_success(f"Command completed successfully in {elapsed:.1f}s")
            return True
        else:
            print_error(f"Command failed after {elapsed:.1f}s")
            return False
    except Exception as e:
        print_error(f"Error executing command: {e}")
        return False

def run_script(script_path: str, args: Optional[List[str]] = None) -> bool:
    """Run a Python script directly"""
    cmd = ["python", script_path]
    if args:
        cmd.extend(args)
    
    print(f"\n{Colors.BLUE}Running: {' '.join(cmd)}{Colors.ENDC}")
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, text=True, capture_output=False)
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print_success(f"Script completed successfully in {elapsed:.1f}s")
            return True
        else:
            print_error(f"Script failed after {elapsed:.1f}s")
            return False
    except Exception as e:
        print_error(f"Error executing script: {e}")
        return False

def confirm_action(prompt: str) -> bool:
    """Get user confirmation for an action"""
    response = input(f"\n{Colors.YELLOW}{prompt} (y/n): {Colors.ENDC}").strip().lower()
    return response in ['y', 'yes']

def show_main_menu():
    """Display the main production menu"""
    clear_screen()
    print_header("🚀 VIETNAM FACTOR INVESTING PLATFORM - PRODUCTION MENU", "=")
    print(f"{Colors.BOLD}📅 Current Date: {datetime.now().strftime('%Y-%m-%d %H:%M')} ICT{Colors.ENDC}")
    
    # Get dynamic quarterly status
    quarterly_status = get_quarterly_reporting_status()
    
    # Show current system status with dynamic quarterly information
    print(f"\n{Colors.CYAN}SYSTEM STATUS:{Colors.ENDC}")
    print("• Database: alphabeta (MySQL)")
    print(f"• Latest Available Quarter: {quarterly_status['latest_available_quarter']}")
    print(f"• Next Expected Quarter: {quarterly_status['next_expected_quarter']}")
    print("• Factor Engine: qvm_engine_v2_enhanced")
    
    # Dynamic quarterly warning based on reporting schedule
    if quarterly_status['is_quarter_due_soon']:
        urgency_color = quarterly_status['urgency_color']
        urgency_level = quarterly_status['urgency']
        days_until = quarterly_status['days_until_next']
        next_quarter = quarterly_status['next_expected_quarter']
        next_publish = quarterly_status['next_quarter_publish'].strftime('%Y-%m-%d')
        
        print(f"{urgency_color}⚠️  {urgency_level} PRIORITY: {next_quarter} Financial Statements Due in {days_until} days ({next_publish})!{Colors.ENDC}")
    else:
        days_until = quarterly_status['days_until_next']
        next_quarter = quarterly_status['next_expected_quarter']
        print(f"{Colors.GREEN}✅ Next quarterly update: {next_quarter} in {days_until} days{Colors.ENDC}")
    
    print(f"\n{Colors.BOLD}═══ 0. MARKET INTELLIGENCE ═══{Colors.ENDC}")
    print("0.1 - 📊 Daily Alpha Pulse (Terminal)")
    print("0.2 - 📈 Advanced Market Intelligence Dashboard (Future)")
    
    # Side-by-side layout for main sections
    print(f"\n{Colors.BOLD}CORE WORKFLOW                     │ EXECUTION & MONITORING{Colors.ENDC}")
    print("──────────────────────────────────┼──────────────────────────────────")
    
    print(f"{Colors.BOLD}1. DAILY DATA UPDATES (CRITICAL)  │ 4. BACKTESTING & EXECUTION{Colors.ENDC}")
    print("1.1 - Market Data (OHLCV, ETFs)  │ 4.1 - Run Canonical Backtest")
    print("1.2 - Financial Info (Shares)     │ 4.2 - Generate Target Portfolio")
    print("1.3 - VCSC Complete Data          │ 4.3 - Pre-Trade Compliance")
    print("1.4 - Foreign Flow Data           │ 4.4 - Export Trade List")
    print("1.5 - 🔄 FULL DAILY UPDATE        │ 4.5 - Post-Trade Reconciliation")
    print("1.6 - 📊 Daily Data Status        │")
    print("                                  │")
    
    # Dynamic quarterly section header based on urgency
    if quarterly_status['is_quarter_due_soon']:
        urgency_indicator = f"({quarterly_status['next_expected_quarter']} {quarterly_status['urgency']} PRIORITY)"
    else:
        urgency_indicator = f"(Next: {quarterly_status['next_expected_quarter']})"
    
    print(f"{Colors.BOLD}2. QUARTERLY UPDATES {urgency_indicator[:15]:<15} │ 5. MONITORING & VALIDATION{Colors.ENDC}")
    print("2.1 - Banking Fundamentals        │ 5.1 - Daily System Health")
    print("2.2 - Non-Financial Fundamentals  │ 5.2 - Performance Attribution")
    print("2.3 - Dividend Data Extraction    │ 5.3 - Portfolio Risk Analytics")
    print("2.4 - 🔄 FULL QUARTERLY UPDATE    │ 5.4 - Factor Generation Status")
    print("2.5 - 📊 Data Quality Report       │ 5.5 - Processing Pipeline Check")
    print("                                  │")
    
    print(f"{Colors.BOLD}3. DATA PROCESSING & VIEWS        │ 6. UTILITIES & MAINTENANCE{Colors.ENDC}")
    print("3.1 - Enhanced Fundamental View   │ 6.1 - Update Sector Mappings")
    print("3.2 - Banking Intermediaries      │ 6.2 - OHLCV Full Reload")
    print("3.3 - Securities Intermediaries   │ 6.3 - Database Backup")
    print("3.4 - Non-Financial Intermediaries│ 6.4 - Clear Cache/Temp Files")
    print("3.5 - 🔄 FULL INTERMEDIARY CALC   │")
    print("3.6 - 📊 Pipeline Status Check    │")
    
    print(f"\n{Colors.BOLD}FACTOR GENERATION (PRODUCTION ENGINE){Colors.ENDC}")
    print("──────────────────────────────────────────────────────────────────────")
    print("7.0 - 📚 Factor Generation Guide & Best Practices (CRITICAL)")
    print("7.1 - Generate QVM Factors (Date Range)    │ 7.4 - 📊 Factor Status & Validation")
    print("7.2 - Generate QVM Factors (Single Date)   │ 7.5 - Factor Generation Diagnostics")
    print("7.3 - Incremental Update (Auto Gap Detect) │ 7.7 - 🔬 Analytics Factor Extraction (RAW)")
    print("                                           │ 7.8 - 📊 Analytics Status & Validation")
    
    print(f"\n{Colors.BOLD}HELP & DOCUMENTATION{Colors.ENDC}")
    print("──────────────────────────────────────────────────────────────────────")
    print("h  - Show Detailed Help           │ s  - System Information")
    print("d  - Display Workflow Diagram     │ x  - Exit Platform")

def handle_daily_updates(choice: str):
    """Handle daily raw data update options"""
    if choice == '1.1':
        print_header("MARKET DATA UPDATE", "-")
        print_info("Updating OHLCV prices and ETF/Index data...")
        return run_workflow_command("daily")
    
    elif choice == '1.2':
        print_header("DAILY FINANCIAL INFORMATION", "-")
        print_info("Updating shares outstanding and financial metrics...")
        return run_workflow_command("daily-financial")
    
    elif choice == '1.3':
        print_header("VCSC COMPLETE DATA UPDATE", "-")
        print_info("This includes:")
        print("  • Adjusted & unadjusted prices")
        print("  • Market microstructure data")
        print("  • Foreign ownership metrics")
        print("  • Order book analytics")
        if confirm_action("Proceed with VCSC data update?"):
            return run_workflow_command("vcsc-update")
    
    elif choice == '1.4':
        print_header("FOREIGN FLOW DATA UPDATE", "-")
        print_info("Updating foreign investor trading data...")
        return run_workflow_command("foreign-flows")
    
    elif choice == '1.5':
        print_header("FULL DAILY UPDATE", "-")
        print_warning("This will run all daily updates sequentially")
        print("Order: Market Data → Financial Info → VCSC → Foreign Flows")
        if confirm_action("Run full daily update?"):
            return run_workflow_command("full-daily")
    
    elif choice == '1.6':
        print_header("DAILY DATA STATUS CHECK", "-")
        print_info("Comprehensive status check for all daily data sources...")
        print_info("This includes:")
        print("  • Market data freshness (equity_history)")
        print("  • VCSC data completeness and foreign flows")
        print("  • Data quality metrics and anomaly detection")
        print("  • Pipeline health monitoring")
        return run_script("scripts/monitoring/check_daily_data_status.py")

def handle_quarterly_updates(choice: str):
    """Handle quarterly fundamental data updates"""
    if choice == '2.1':
        print_header("BANKING SECTOR FUNDAMENTALS", "-")
        print_info("Fetching and importing banking financial statements...")
        return run_workflow_command("banking-fundamentals")
    
    elif choice == '2.2':
        print_header("NON-FINANCIAL SECTORS FUNDAMENTALS", "-")
        print_info("Fetching and importing non-financial statements...")
        return run_workflow_command("fundamentals")
    
    elif choice == '2.3':
        print_header("DIVIDEND DATA EXTRACTION", "-")
        print_info("Extracting dividend data from fundamental_values...")
        return run_workflow_command("dividend-extraction")
    
    elif choice == '2.4':
        print_header("FULL QUARTERLY UPDATE", "-")
        print_warning("This process will take 30-60 minutes")
        print("Steps:")
        print("  1. Banking fundamentals")
        print("  2. Non-financial fundamentals")
        print("  3. Dividend extraction")
        if confirm_action("Run full quarterly update?"):
            return run_workflow_command("quarterly")
    
    elif choice == '2.5':
        print_header("COMPREHENSIVE DATA QUALITY REPORT", "-")
        print_info("Checking quarterly FS coverage + processing pipeline status...")
        print_info("This combines quarterly data status and processing pipeline monitoring")
        
        # Run both the quarterly FS status and processing pipeline monitor
        print("\n🔍 PART 1: QUARTERLY FINANCIAL STATEMENT COVERAGE")
        print("-" * 60)
        result1 = run_script("scripts/check_quarterly_fs_status.py")
        
        print("\n🔍 PART 2: PROCESSING PIPELINE STATUS")
        print("-" * 60)
        result2 = run_script("scripts/monitoring/check_processing_pipeline_status_dynamic.py")
        
        return result1 and result2

def handle_data_processing(choice: str):
    """Handle data processing and view creation"""
    if choice == '3.1':
        print_header("CREATE/UPDATE ENHANCED FUNDAMENTAL VIEW", "-")
        print_warning("This is a PREREQUISITE for intermediary calculations")
        print_info("Creating v_comprehensive_fundamental_items with 81 columns...")
        if confirm_action("Create/update fundamental view?"):
            return run_script("scripts/sector_views/create_nonfin_enhanced_view.py")
    
    elif choice == '3.2':
        print_header("BANKING INTERMEDIARY CALCULATIONS", "-")
        print_info("Calculating banking-specific metrics (NIM, CAR, etc.)...")
        return run_script("scripts/intermediaries/banking_sector_intermediary_calculator.py")
    
    elif choice == '3.3':
        print_header("SECURITIES INTERMEDIARY CALCULATIONS", "-")
        print_info("Calculating securities-specific metrics...")
        return run_script("scripts/intermediaries/securities_sector_intermediary_calculator.py")
    
    elif choice == '3.4':
        print_header("NON-FINANCIAL INTERMEDIARY CALCULATIONS", "-")
        print_info("This will process 667 tickers across 21 sectors")
        print_warning("Estimated time: 10-15 minutes")
        if confirm_action("Calculate all non-financial intermediaries?"):
            return run_script("scripts/intermediaries/all_nonfin_sectors_intermediary_calculator.py")
    
    elif choice == '3.5':
        print_header("FULL INTERMEDIARY CALCULATION", "-")
        print_warning("This will calculate intermediaries for ALL sectors")
        if confirm_action("Run full intermediary calculation?"):
            success = True
            # Run in sequence
            print("\n[1/3] Banking sector...")
            success &= run_script("scripts/intermediaries/banking_sector_intermediary_calculator.py")
            
            print("\n[2/3] Securities sector...")
            success &= run_script("scripts/intermediaries/securities_sector_intermediary_calculator.py")
            
            print("\n[3/3] Non-financial sectors...")
            success &= run_script("scripts/intermediaries/all_nonfin_sectors_intermediary_calculator.py")
            
            return success
    
    elif choice == '3.6':
        print_header("CHECK PROCESSING PIPELINE STATUS", "-")
        print_info("Comprehensive status check for views and intermediary calculations...")
        print_info("This will show:")
        print("  • Raw data coverage by quarter")
        print("  • Enhanced view status and current quarter coverage")
        print("  • Intermediary calculation status by sector")
        print("  • Processing gaps and recommendations")
        if confirm_action("Run processing pipeline status check?"):
            return run_script("scripts/monitoring/check_processing_pipeline_status_dynamic.py")

def handle_factor_generation(choice: str):
    """Handle QVM factor generation using production engine"""
    if choice == '7.0':
        show_factor_generation_guide()
        return True
    
    elif choice == '7.1':
        print_header("GENERATE QVM FACTORS (DATE RANGE)", "-")
        start_date = input("Enter start date (YYYY-MM-DD): ").strip()
        end_date = input("Enter end date (YYYY-MM-DD): ").strip()
        mode = input("Enter mode (incremental/refresh) [incremental]: ").strip() or "incremental"
        version = input("Enter version tag [v2_enhanced]: ").strip() or "v2_enhanced"
        
        args = [
            "--start-date", start_date,
            "--end-date", end_date,
            "--mode", mode,
            "--version", version
        ]
        return run_script("production/scripts/run_factor_generation.py", args)
    
    elif choice == '7.2':
        print_header("GENERATE QVM FACTORS (SINGLE DATE)", "-")
        date = input("Enter date (YYYY-MM-DD): ").strip()
        mode = input("Enter mode (incremental/refresh) [incremental]: ").strip() or "incremental"
        version = input("Enter version tag [v2_enhanced]: ").strip() or "v2_enhanced"
        
        args = [
            "--start-date", date,
            "--end-date", date,
            "--mode", mode,
            "--version", version
        ]
        return run_script("production/scripts/run_factor_generation.py", args)
    
    elif choice == '7.3':
        print_header("INCREMENTAL FACTOR UPDATE (AUTO GAP DETECTION)", "-")
        print_info("Detecting gaps in factor_scores_qvm table...")
        
        # Get version preference
        version = input("Enter version tag [qvm_v2.0_enhanced]: ").strip() or "qvm_v2.0_enhanced"
        
        # Use intelligent gap detection by letting the script find missing dates
        from datetime import timedelta
        # Set a reasonable range to search for gaps (e.g., last 30 days)
        end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')  # Yesterday
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')  # 30 days ago
        
        print_info(f"Searching for missing dates between {start_date} and {end_date}")
        print_info(f"Version: {version}")
        print_info("Note: Only missing dates will be processed (incremental mode)")
        
        args = [
            "--start-date", start_date,
            "--end-date", end_date,
            "--mode", "incremental",
            "--version", version
        ]
        return run_script("production/scripts/run_factor_generation.py", args)
    
    elif choice == '7.4':
        print_header("COMPREHENSIVE FACTOR STATUS & VALIDATION", "-")
        print_info("Comprehensive factor generation monitoring and validation...")
        print_info("This combines factor status monitoring with validation checks")
        
        # Run the comprehensive factor status monitor
        print("\n🔍 FACTOR GENERATION STATUS & VALIDATION")
        print("-" * 60)
        return run_script("scripts/monitoring/check_factor_generation_status.py")
    
    elif choice == '7.5':
        print_header("FACTOR GENERATION DIAGNOSTICS", "-")
        print_info("Advanced diagnostics for factor generation issues...")
        # TODO: Implement advanced factor diagnostics
        print_warning("Factor generation diagnostics to be implemented")
        print_info("This will include:")
        print("  • Engine performance profiling")
        print("  • Calculation accuracy testing")
        print("  • Cross-validation with alternative methods")
        print("  • Historical consistency checks")
        return False

    elif choice == '7.7':
        print_header("ANALYTICS FACTOR EXTRACTION (Sidecar: RAW individual factors)", "-")
        print_info("Wrapper-only capture at sector-neutralization callsite; composites untouched.")
        print("Options:")
        print("  a) Daily update (latest composite date, all canonical factors)")
        print("  b) Incremental from last processed date (auto-continue)")
        print("  c) Range update (specify dates and optional factor list)")
        sub = input("Select option (a/b/c): ").strip().lower()

        base_args = [
            '--version', input("Analytics version [analytics_v1_neo_fixed]: ").strip() or 'analytics_v1_neo_fixed',
            '--audit-tier', 'tier0', '--write-sidecar', '--resume',
            '--batch-size', input("Batch size [30]: ").strip() or '30'
        ]

        if sub == 'a':
            args = ['--factors', 'all', '--mode', 'daily'] + base_args
            return run_script('production/scripts/run_factor_analytics_wrapper_neo_fix.py', args)
        elif sub == 'b':
            args = ['--factors', 'all', '--mode', 'from-last'] + base_args
            return run_script('production/scripts/run_factor_analytics_wrapper_neo_fix.py', args)
        elif sub == 'c':
            start_date = input("Start date (YYYY-MM-DD): ").strip()
            end_date = input("End date (YYYY-MM-DD): ").strip()
            factors_in = input("Factors (comma-separated or 'all') [all]: ").strip()
            factors_list = ['all'] if factors_in == '' or factors_in.lower() == 'all' else [f.strip() for f in factors_in.split(',') if f.strip()]
            args = ['--mode', 'range', '--start-date', start_date, '--end-date', end_date, '--factors'] + factors_list + base_args
            return run_script('production/scripts/run_factor_analytics_wrapper_neo_fix.py', args)
        else:
            print_error('Invalid option')
            return False

    elif choice == '7.8':
        print_header("ANALYTICS STATUS & VALIDATION", "-")
        print_info("Sidecar monitoring uses MySQL queries over factor_signals_raw and factor_norm_stats.")
        print_info("Running quick sidecar status...")
        return run_script('scripts/monitoring/check_sidecar_status.py')

def handle_backtesting_execution(choice: str):
    """Handle backtesting and portfolio execution"""
    if choice == '4.1':
        print_header("RUN CANONICAL STRATEGY BACKTEST", "-")
        print_info("This will run the official QVM strategy backtest")
        print_warning("Ensure factor_scores_qvm table is up to date")
        if confirm_action("Launch Jupyter notebook for backtesting?"):
            os.system("cd notebooks/phase6_backtesting && jupyter notebook 06_canonical_qvm_backtest_final.ipynb")
        return True
    
    elif choice == '4.2':
        print_header("GENERATE TARGET PORTFOLIO", "-")
        dry_run = confirm_action("Run in dry-run mode?")
        args = ["--dry-run"] if dry_run else []
        return run_script("production/scripts/generate_target_portfolio.py", args)
    
    elif choice == '4.3':
        print_header("PRE-TRADE COMPLIANCE CHECK", "-")
        print_info("Checking portfolio against risk limits...")
        # TODO: Implement compliance check
        print_warning("Compliance check to be implemented")
        return False
    
    elif choice == '4.4':
        print_header("EXPORT TRADE LIST", "-")
        print_info("Exporting trade list for execution...")
        # TODO: Implement trade list export
        print_warning("Trade list export to be implemented")
        return False
    
    elif choice == '4.5':
        print_header("POST-TRADE RECONCILIATION", "-")
        print_info("Reconciling executed trades...")
        # TODO: Implement reconciliation
        print_warning("Post-trade reconciliation to be implemented")
        return False

def show_factor_generation_guide():
    """Display comprehensive factor generation guide and best practices"""
    clear_screen()
    print_header("📚 FACTOR GENERATION GUIDE & BEST PRACTICES", "=")
    
    # Get current temporal logic status (DYNAMIC - calculates based on reporting rules)
    current_date = datetime.now()
    current_year = current_date.year
    
    # Calculate Q2 availability date dynamically (Q2 ends June 30 + 45 days reporting lag)
    q2_end_date = datetime(current_year, 6, 30)
    q2_available_date = q2_end_date + timedelta(days=45)  # Dynamic Q2 regulatory deadline
    days_until_q2 = (q2_available_date - current_date).days
    
    print(f"📅 Current Date: {current_date.strftime('%Y-%m-%d %H:%M ICT')}")
    if days_until_q2 > 0:
        print(f"⏰ Q2 {current_year} Regulatory Deadline: {q2_available_date.strftime('%B %d, %Y')} ({days_until_q2} days)")
        print_warning(f"TEMPORAL LOGIC: Currently locked to Q1 {current_year} data (45-day lag rule)")
        print_info(f"Q2 {current_year} data available in intermediaries but not used until {q2_available_date.strftime('%b %d')}")
    else:
        print_success(f"Q2 {current_year} data is now available for factor generation")
    
    print(f"\n{Colors.BOLD}🎯 CRITICAL PREREQUISITES (CHECK FIRST!):{Colors.ENDC}")
    print("1. ✅ Run Option 3.6 - Check Processing Pipeline Status")
    print("   • Verify 99%+ processing efficiency")
    print("   • Confirm enhanced views are updated")
    print("   • Check intermediary calculations are complete")
    print()
    print("2. ✅ Ensure Data Prerequisites:")
    print("   • Enhanced fundamental views created (Option 3.1)")
    print("   • All intermediary calculations complete (Option 3.5)")
    print("   • VCSC daily data is current")
    print("   • Equity history adjusted prices are current")
    print()
    
    print(f"\n{Colors.BOLD}⚙️ FACTOR GENERATION OPTIONS EXPLAINED:{Colors.ENDC}")
    print(f"{Colors.BLUE}Option 4.1 - Generate QVM Factors (Date Range):{Colors.ENDC}")
    print("   • Use for: Historical factor generation, backtesting")
    print("   • Example: 2024-01-01 to 2024-12-31")
    print("   • Parameters:")
    print("     - start-date: YYYY-MM-DD format")
    print("     - end-date: YYYY-MM-DD format")
    print("     - mode: incremental (default) or refresh")
    print("     - version: qvm_v2.0_enhanced (recommended)")
    print("   • Runtime: ~1-2 minutes per month of data")
    print()
    
    print(f"{Colors.BLUE}Option 4.2 - Generate QVM Factors (Single Date):{Colors.ENDC}")
    print("   • Use for: Daily factor updates, specific date testing")
    print("   • Example: 2025-07-30 (latest available)")
    print("   • Faster than date range for single dates")
    print("   • Ideal for production daily updates")
    print()
    
    print(f"{Colors.BLUE}Option 4.3 - Incremental Factor Update (Auto Gap Detection):{Colors.ENDC}")
    print("   • Use for: Automatic gap detection and filling")
    print("   • Scans factor_scores_qvm for missing dates (last 30 days)")
    print("   • Generates factors for missing dates only")
    print("   • Perfect for daily production workflow")
    print("   • Version-aware: Only processes specified strategy version")
    print()
    
    print(f"{Colors.BLUE}Option 4.4 - Factor Generation Status Monitor:{Colors.ENDC}")
    print("   • Use for: Checking factor generation coverage")
    print("   • Shows latest factors by version and date")
    print("   • Identifies processing gaps and issues")
    print("   • Provides recommendations for next steps")
    print()
    
    print(f"\n{Colors.BOLD}🛡️ CRITICAL SAFETY GUIDELINES:{Colors.ENDC}")
    print_warning("ALWAYS backup factor_scores_qvm table before large regenerations!")
    print()
    print("1. 🔍 Pre-Generation Checklist:")
    print("   • Run Option 3.6 to verify data pipeline status")
    print("   • Check database disk space (factor table can be large)")
    print("   • Ensure no other processes are using the database")
    print("   • Verify version tag matches your requirements")
    print()
    
    print("2. 📊 Version Tag Guidelines:")
    print("   • 'qvm_v2.0_enhanced' - Latest production engine (RECOMMENDED)")
    print("   • 'v2_enhanced' - Legacy tag (compatibility only)")
    print("   • NEVER mix version tags in same backtest")
    print("   • Use consistent tags across all dates")
    print()
    
    print("3. ⚡ Performance Optimization:")
    print("   • Use 'incremental' mode for daily updates")
    print("   • Use 'refresh' mode only when recalculation needed")
    print("   • Process in smaller date ranges for large historical periods")
    print("   • Monitor database performance during generation")
    print()
    
    print(f"\n{Colors.BOLD}🎯 RECOMMENDED WORKFLOWS:{Colors.ENDC}")
    print(f"{Colors.GREEN}Daily Production Workflow:{Colors.ENDC}")
    print("1. Run Option 3.6 (Check pipeline status)")
    print("2. If 99%+ efficiency → Run Option 4.3 (Incremental update)")
    print("3. Run Option 4.4 (Verify results)")
    print("4. Proceed to backtesting/execution")
    print()
    
    print(f"{Colors.GREEN}Historical Backtest Workflow:{Colors.ENDC}")
    print("1. Backup factor_scores_qvm table")
    print("2. Run Option 4.1 with desired date range")
    print("3. Use consistent version tag (qvm_v2.0_enhanced)")
    print("4. Verify factor scores before backtesting")
    print()
    
    print(f"{Colors.GREEN}Q2 2025 Transition Workflow (Current Priority):{Colors.ENDC}")
    if days_until_q2 > 0:
        print_warning("CURRENT SITUATION: Mixed-quarter data available but not used")
        print("• Q2 2025 data: 99.4% processed in intermediaries")
        print("• Temporal logic: Locked to Q1 2025 until Aug 14")
        print("• Options:")
        print("  A) Wait until Aug 14 for automatic Q2 usage")
        print("  B) Implement progressive temporal logic (advanced)")
        print("  C) Generate factors now for Q1 consistency")
    else:
        print_success("Q2 2025 data is now automatically used in factor generation")
        print("• Run Option 4.3 to generate latest factors with Q2 data")
        print("• Expect improved factor scores due to current earnings")
    
    print(f"\n{Colors.BOLD}🚨 COMMON PITFALLS TO AVOID:{Colors.ENDC}")
    print_error("1. Running factor generation without updated intermediaries")
    print_error("2. Mixing different version tags in same analysis")
    print_error("3. Not backing up before large historical regenerations")
    print_error("4. Ignoring processing pipeline status (Option 3.6)")
    print_error("5. Using refresh mode when incremental would suffice")
    print()
    
    print(f"\n{Colors.BOLD}📈 FACTOR GENERATION MONITORING:{Colors.ENDC}")
    print("• Monitor factor_scores_qvm table size during generation")
    print("• Check for consistent ticker coverage across dates")
    print("• Verify factor scores are reasonable (not all zeros/nulls)")
    print("• Use Option 4.4 to validate calculation results")
    print("• Track generation performance for optimization")
    print()
    
    print(f"\n{Colors.BOLD}🔧 TROUBLESHOOTING QUICK REFERENCE:{Colors.ENDC}")
    print("Issue: 'No data available for date X'")
    print("→ Run Option 3.6 to check pipeline status")
    print("→ Verify intermediaries are calculated for that date")
    print()
    print("Issue: Factor generation taking too long")
    print("→ Use smaller date ranges")
    print("→ Check database performance")
    print("→ Consider incremental mode")
    print()
    print("Issue: Inconsistent factor scores")
    print("→ Verify version tag consistency")
    print("→ Check for data quality issues")
    print("→ Run Option 4.4 for validation")
    print()
    
    print(f"\n{Colors.BOLD}📞 SUPPORT & DOCUMENTATION:{Colors.ENDC}")
    print("• QVM Engine Specification: docs/2_technical_implementation/02a_qvm_engine_v2_enhanced_specification.md")
    print("• Temporal Logic Rules: docs/2_technical_implementation/02b_temporal_logic_and_data_availability.md")
    print("• Production Workflow: docs/2_technical_implementation/02g_quant_workflow_v1.1_production_ready.md")
    print("• Current Status: CLAUDE.md (Phase 16 - Weighted Composite Models)")
    
    print(f"\n{Colors.CYAN}{'='*80}{Colors.ENDC}")
    print(f"💡 NEXT STEPS: Run Option 3.6 first, then choose appropriate factor generation option")
    print(f"🚀 PRODUCTION TIP: Option 4.3 (Incremental Update) is safest for daily workflow")
    input(f"\n{Colors.CYAN}Press Enter to return to main menu...{Colors.ENDC}")

def show_help():
    """Display detailed help information"""
    clear_screen()
    print_header("📚 PRODUCTION MENU HELP", "=")
    
    # Get current quarterly status for dynamic help
    quarterly_status = get_quarterly_reporting_status()
    next_quarter = quarterly_status['next_expected_quarter']
    urgency = quarterly_status['urgency']
    days_until = quarterly_status['days_until_next']
    
    print(f"\n{Colors.BOLD}CRITICAL WORKFLOW SEQUENCE:{Colors.ENDC}")
    print("1. Daily Updates → 2. Quarterly Updates → 3. Data Processing")
    print("→ 4. Factor Generation → 5. Backtesting & Execution")
    
    # Dynamic quarterly checklist
    print(f"\n{Colors.BOLD}{next_quarter} QUARTERLY UPDATE CHECKLIST ({urgency} - {days_until} days):{Colors.ENDC}")
    print(f"[ ] 1. Run option 2.1 (Banking fundamentals)")
    print(f"[ ] 2. Run option 2.2 (Non-financial fundamentals)")
    print(f"[ ] 3. Run option 2.3 (Dividend extraction)")
    print(f"[ ] 4. Run option 3.1 (Create enhanced view)")
    print(f"[ ] 5. Run option 3.5 (Calculate all intermediaries)")
    print(f"[ ] 6. Run option 4.1 (Generate factors for {next_quarter})")
    print(f"[ ] 7. Run option 2.5 (Verify data completeness)")
    
    print(f"\n{Colors.BOLD}IMPORTANT NOTES:{Colors.ENDC}")
    print("• Always backup database before quarterly updates")
    print("• Enhanced fundamental view (3.1) must be created before intermediaries")
    print("• Factor generation requires completed intermediary calculations")
    print("• Use incremental mode for daily factor updates")
    
    input(f"\n{Colors.CYAN}Press Enter to return to main menu...{Colors.ENDC}")

def show_workflow_diagram():
    """Display the workflow diagram"""
    clear_screen()
    print_header("📊 PRODUCTION WORKFLOW DIAGRAM", "=")
    
    print("""
    ┌─────────────────────┐
    │   RAW DATA INPUTS   │
    ├─────────────────────┤
    │ • Daily Updates     │
    │ • Quarterly Updates │
    └──────────┬──────────┘
               │
               ▼
    ┌─────────────────────┐
    │  DATA PROCESSING    │
    ├─────────────────────┤
    │ • Enhanced Views    │
    │ • Intermediaries    │
    └──────────┬──────────┘
               │
               ▼
    ┌─────────────────────┐
    │  FACTOR GENERATION  │
    ├─────────────────────┤
    │ • QVM Engine v2     │
    │ • factor_scores_qvm │
    └──────────┬──────────┘
               │
               ▼
    ┌─────────────────────┐
    │    BACKTESTING      │
    ├─────────────────────┤
    │ • Strategy Validation│
    │ • Performance Analysis│
    └──────────┬──────────┘
               │
               ▼
    ┌─────────────────────┐
    │  PORTFOLIO EXECUTION │
    ├─────────────────────┤
    │ • Target Generation │
    │ • Trade Execution   │
    └─────────────────────┘
    """)
    
    input(f"\n{Colors.CYAN}Press Enter to return to main menu...{Colors.ENDC}")

def show_system_info():
    """Display system information"""
    clear_screen()
    print_header("ℹ️ SYSTEM INFORMATION", "=")
    
    print(f"\n{Colors.BOLD}Environment:{Colors.ENDC}")
    print(f"• Python Version: {sys.version.split()[0]}")
    print(f"• Working Directory: {os.getcwd()}")
    print(f"• Current User: {os.environ.get('USER', 'Unknown')}")
    
    print(f"\n{Colors.BOLD}Database:{Colors.ENDC}")
    print("• Host: localhost")
    print("• Database: alphabeta")
    print("• Engine: MySQL 8.0+")
    
    print(f"\n{Colors.BOLD}Key Tables:{Colors.ENDC}")
    print("• equity_history: 16+ years of adjusted OHLCV")
    print("• vcsc_daily_data_complete: Market microstructure")
    print("• fundamental_values: Quarterly financials")
    print("• intermediary_calculations_*: Pre-computed metrics")
    print("• factor_scores_qvm: Final QVM scores")
    
    print(f"\n{Colors.BOLD}Production Scripts:{Colors.ENDC}")
    print("• Factor Engine: production/engine/qvm_engine_v2_enhanced.py")
    print("• Orchestrator: production/scripts/run_factor_generation.py")
    print("• Workflow Runner: scripts/run_workflow.py")
    
    input(f"\n{Colors.CYAN}Press Enter to return to main menu...{Colors.ENDC}")

def main():
    """Main menu loop"""
    while True:
        show_main_menu()
        
        choice = input(f"\n{Colors.BOLD}Select option: {Colors.ENDC}").strip().lower()
        
        # Market Intelligence (NEW OPTION 0)
        if choice == '0.1':
            print_header("DAILY ALPHA PULSE (TERMINAL)", "-")
            print_info("Launching terminal-based daily market intelligence...")
            run_script("production/market_intelligence/terminal_daily_pulse.py")
        
        elif choice == '0.2':
            print_header("ADVANCED MARKET INTELLIGENCE DASHBOARD", "-")
            print_info("Future: Advanced market intelligence dashboard")
            print_warning("Advanced dashboard to be implemented")
        
        # Daily updates
        elif choice.startswith('1.'):
            handle_daily_updates(choice)
        
        # Quarterly updates
        elif choice.startswith('2.'):
            handle_quarterly_updates(choice)
        
        # Data processing
        elif choice.startswith('3.'):
            handle_data_processing(choice)
        
        # Backtesting & execution (moved to section 4)
        elif choice.startswith('4.'):
            handle_backtesting_execution(choice)
        
        # Monitoring & validation (moved to section 5)
        elif choice == '5.1':
            print_header("DAILY SYSTEM HEALTH CHECK", "-")
            run_script("production/scripts/daily_system_check.py")
        
        elif choice == '5.2':
            print_header("PERFORMANCE ATTRIBUTION", "-")
            print_warning("Performance attribution to be implemented")
        
        elif choice == '5.3':
            print_header("PORTFOLIO RISK ANALYTICS", "-")
            print_warning("Portfolio risk analytics to be implemented")
        
        elif choice == '5.4':
            print_header("COMPREHENSIVE FACTOR STATUS & VALIDATION", "-")
            print_info("Comprehensive factor generation monitoring and validation...")
            run_script("scripts/monitoring/check_factor_generation_status.py")
            
        elif choice == '5.5':
            print_header("CHECK PROCESSING PIPELINE STATUS", "-")
            print_info("Comprehensive status check for views and intermediary calculations...")
            if confirm_action("Run processing pipeline status check?"):
                run_script("scripts/monitoring/check_processing_pipeline_status_dynamic.py")
        
        # Utilities (moved to section 6)
        elif choice == '6.1':
            print_header("UPDATE SECTOR MAPPINGS", "-")
            run_workflow_command("update-sectors")
        
        elif choice == '6.2':
            print_header("OHLCV FULL RELOAD", "-")
            print_warning("This will reload ALL historical data")
            start_date = input("Enter start date (YYYY-MM-DD): ").strip()
            if confirm_action(f"Reload OHLCV from {start_date}?"):
                run_workflow_command("full-ohlcv-reload", [start_date])
        
        elif choice == '6.3':
            print_header("DATABASE BACKUP", "-")
            print_warning("Database backup to be implemented")
        
        elif choice == '6.4':
            print_header("CLEAR CACHE/TEMP FILES", "-")
            print_warning("Cache clearing to be implemented")
        
        # Factor generation (moved to section 7)
        elif choice.startswith('7.'):
            handle_factor_generation(choice)
        
        # Help & documentation
        elif choice == 'h':
            show_help()
        
        elif choice == 'd':
            show_workflow_diagram()
        
        elif choice == 's':
            show_system_info()
        
        # Exit
        elif choice == 'x':
            print_success("\nExiting Production Menu. Goodbye!")
            break
        
        else:
            print_error("Invalid option. Please try again.")
        
        if choice != 'x':
            input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.ENDC}")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print_warning("\n\nOperation cancelled by user.")
        print_success("Exiting Production Menu.")
        sys.exit(0)
    except Exception as e:
        print_error(f"\nUnexpected error: {e}")
        sys.exit(1)