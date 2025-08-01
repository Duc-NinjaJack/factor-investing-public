#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Market Intelligence Menu Interface
==================================
Standalone menu system for quantitative market intelligence products.
This module does not modify any existing codebase.

Author: Duc Nguyen
Date: July 31, 2025
Status: Production Ready
"""

import sys
import subprocess
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List
import time

# Ensure we're running from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if Path.cwd() != PROJECT_ROOT:
    os.chdir(PROJECT_ROOT)
    print(f"Changed working directory to: {PROJECT_ROOT}")

# Add project root to Python path
sys.path.insert(0, str(PROJECT_ROOT))

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
            print_success(f"Command completed successfully in {elapsed:.1f}s")
            return True
        else:
            print_error(f"Command failed after {elapsed:.1f}s")
            return False
    except Exception as e:
        print_error(f"Error executing command: {e}")
        return False

def confirm_action(prompt: str) -> bool:
    """Get user confirmation for an action"""
    response = input(f"\n{Colors.YELLOW}{prompt} (y/n): {Colors.ENDC}").strip().lower()
    return response in ['y', 'yes']

def show_main_menu():
    """Display the main market intelligence menu"""
    clear_screen()
    print_header("📊 VIETNAM MARKET INTELLIGENCE PLATFORM", "=")
    print(f"{Colors.BOLD}📅 Current Date: {datetime.now().strftime('%Y-%m-%d %H:%M')} ICT{Colors.ENDC}")
    
    print(f"\n{Colors.CYAN}PLATFORM STATUS:{Colors.ENDC}")
    print("• Market Intelligence: Ready")
    print("• Data Source: MySQL alphabeta database")
    print("• Report Output: HTML dashboards + PDF exports")
    print("• Real-time Data: Wong Trading API integration")
    
    print(f"\n{Colors.BOLD}═══ 📈 DAILY PRODUCTS ═══{Colors.ENDC}")
    print("1.1 - Generate Daily Alpha Pulse Dashboard")
    print("     • Pre-market quantitative insights")
    print("     • Factor performance monitoring")
    print("     • Risk metrics and market regime analysis")
    print("     • Trading signals and opportunities")
    print("     • Runtime: ~10 seconds | Output: Beautiful terminal display")
    print()
    
    print("1.2 - View Latest Daily Alpha Pulse")
    print("     • Open most recent dashboard in browser")
    print("     • Quick access to today's insights")
    print()
    
    print("1.3 - Daily Alpha Pulse Historical Analysis")
    print("     • Generate dashboard for specific past date")
    print("     • Compare historical market conditions")
    print("     • Backtest signal performance")
    print()

    print(f"\n{Colors.BOLD}═══ 📊 WEEKLY PRODUCTS ═══{Colors.ENDC}")
    print("2.1 - Generate Weekly Strategic Alpha Review")
    print("     • Comprehensive portfolio construction insights")
    print("     • Market regime deep dive analysis")
    print("     • Factor attribution and correlation analysis")
    print("     • Risk-adjusted optimization recommendations")
    print("     • Runtime: ~10-15 minutes | Output: HTML + PDF")
    print()
    
    print("2.2 - Weekly Performance Attribution")
    print("     • Factor contribution analysis")
    print("     • Sector allocation breakdown")
    print("     • Risk decomposition (systematic vs idiosyncratic)")
    print()

    print(f"\n{Colors.BOLD}═══ ⚙️ CONFIGURATION & UTILITIES ═══{Colors.ENDC}")
    print("3.1 - Configure Risk Thresholds")
    print("     • Set volatility spike alerts")
    print("     • Correlation breakdown warnings")
    print("     • Liquidity condition thresholds")
    print()
    
    print("3.2 - Signal Generation Settings")
    print("     • Mean reversion parameters")
    print("     • Momentum breakout criteria")
    print("     • Factor rotation thresholds")
    print()
    
    print("3.3 - Report Templates & Styling")
    print("     • Customize HTML dashboard appearance")
    print("     • PDF report formatting options")
    print("     • Email alert configurations")
    print()

    print(f"\n{Colors.BOLD}═══ 🔍 ANALYSIS & RESEARCH ═══{Colors.ENDC}")
    print("4.1 - Factor Research Toolkit")
    print("     • Custom factor creation and testing")
    print("     • Cross-sectional analysis tools")
    print("     • Factor decay and persistence analysis")
    print()
    
    print("4.2 - Market Regime Detection")
    print("     • Hidden Markov Model regime classification")
    print("     • Volatility clustering analysis (GARCH)")
    print("     • Correlation regime transitions")
    print()
    
    print("4.3 - Signal Backtesting Framework")
    print("     • Historical signal performance analysis")
    print("     • Transaction cost estimation")
    print("     • Risk-adjusted return calculation")
    print()

    print(f"\n{Colors.BOLD}═══ 📋 MONITORING & VALIDATION ═══{Colors.ENDC}")
    print("5.1 - Data Quality Dashboard")
    print("     • Real-time data freshness monitoring")
    print("     • Missing data gap detection")
    print("     • Data consistency validation")
    print()
    
    print("5.2 - System Health Check")
    print("     • Database connectivity status")
    print("     • API endpoint availability")
    print("     • Report generation performance")
    print()
    
    print("5.3 - Usage Analytics")
    print("     • Report generation frequency")
    print("     • Feature utilization statistics")
    print("     • Performance benchmarks")
    print()

    print(f"\n{Colors.BOLD}═══ 🚀 ADVANCED FEATURES (FUTURE) ═══{Colors.ENDC}")
    print("6.1 - Real-time Alert System")
    print("     • SMS/Email notifications for critical signals")
    print("     • Slack/Teams integration")
    print("     • Custom alert rule engine")
    print()
    
    print("6.2 - API Endpoints")
    print("     • RESTful API for systematic strategies")  
    print("     • Real-time data streaming")
    print("     • Portfolio optimization API")
    print()
    
    print("6.3 - Multi-Asset Expansion")
    print("     • Bond market analysis")
    print("     • Commodity price monitoring")
    print("     • Cross-market correlation analysis")
    print()

    print(f"\n{Colors.BOLD}═══ 📚 HELP & DOCUMENTATION ═══{Colors.ENDC}")
    print("h  - Market Intelligence User Guide")
    print("d  - Data Sources Documentation")
    print("s  - Signal Methodology Reference")
    print("r  - Risk Metrics Explanation")
    
    print("\n0  - Exit Market Intelligence Platform")
    print(f"\n{Colors.CYAN}{'='*70}{Colors.ENDC}")
    print(f"💡 TIP: Start with option 1.1 to generate your first Daily Alpha Pulse")
    print(f"💡 NEW: All reports are saved in production/market_intelligence/reports/")
    print(f"💡 HELP: Type option number + ENTER, or 'h' for detailed documentation")
    print(f"{Colors.CYAN}{'='*70}{Colors.ENDC}")

def handle_daily_products(choice: str):
    """Handle daily product options"""
    if choice == '1.1':
        print_header("GENERATE DAILY ALPHA PULSE DASHBOARD", "-")
        print_info("Generating comprehensive daily market intelligence...")
        print_info("This will analyze:")
        print("  • Market breadth and sector performance")
        print("  • Factor performance (Quality, Value, Momentum)")
        print("  • Foreign flow analysis")
        print("  • Risk metrics and volatility regime")
        print("  • Top trading signals and opportunities")
        
        if confirm_action("Generate Daily Alpha Pulse dashboard?"):
            return run_script("production/market_intelligence/terminal_daily_pulse.py")
    
    elif choice == '1.2':
        print_header("VIEW LATEST DAILY ALPHA PULSE", "-")
        reports_dir = PROJECT_ROOT / "production" / "market_intelligence" / "reports"
        
        # Find latest daily report
        daily_reports = list(reports_dir.glob("daily_alpha_pulse_*.html"))
        
        if daily_reports:
            latest_report = max(daily_reports, key=lambda x: x.stat().st_mtime)
            print_info(f"Opening latest report: {latest_report.name}")
            
            # Open in default browser
            import webbrowser
            webbrowser.open(f"file://{latest_report}")
            print_success("Report opened in browser")
            return True
        else:
            print_warning("No daily reports found. Generate one first using option 1.1")
            return False
    
    elif choice == '1.3':
        print_header("DAILY ALPHA PULSE HISTORICAL ANALYSIS", "-")
        date_str = input("Enter date for analysis (YYYY-MM-DD): ").strip()
        
        try:
            # Validate date format
            datetime.strptime(date_str, '%Y-%m-%d')
            print_info(f"Generating Daily Alpha Pulse for {date_str}...")
            return run_script("production/market_intelligence/daily_alpha_pulse.py", ["--date", date_str])
        except ValueError:
            print_error("Invalid date format. Please use YYYY-MM-DD")
            return False

def handle_weekly_products(choice: str):
    """Handle weekly product options"""
    if choice == '2.1':
        print_header("GENERATE WEEKLY STRATEGIC ALPHA REVIEW", "-")
        print_warning("Weekly Strategic Alpha Review is under development")
        print_info("Coming soon: Comprehensive portfolio construction insights")
        return False
    
    elif choice == '2.2':
        print_header("WEEKLY PERFORMANCE ATTRIBUTION", "-")
        print_warning("Weekly Performance Attribution is under development")
        print_info("Coming soon: Factor contribution and risk decomposition analysis")
        return False

def handle_configuration(choice: str):
    """Handle configuration options"""
    if choice == '3.1':
        print_header("CONFIGURE RISK THRESHOLDS", "-")
        print_warning("Risk threshold configuration is under development")
        return False
    
    elif choice == '3.2':
        print_header("SIGNAL GENERATION SETTINGS", "-")
        print_warning("Signal generation settings are under development")
        return False
        
    elif choice == '3.3':
        print_header("REPORT TEMPLATES & STYLING", "-")
        print_warning("Template customization is under development")
        return False

def show_help():
    """Display help information"""
    clear_screen()
    print_header("📚 MARKET INTELLIGENCE USER GUIDE", "=")
    
    print(f"\n{Colors.BOLD}GETTING STARTED:{Colors.ENDC}")
    print("1. Generate your first Daily Alpha Pulse dashboard (option 1.1)")
    print("2. Review the insights and trading signals")
    print("3. Set up automated generation via cron (optional)")
    print("4. Explore weekly strategic analysis (option 2.1)")
    
    print(f"\n{Colors.BOLD}DATA SOURCES:{Colors.ENDC}")
    print("• Historical Prices: equity_history (16+ years, 728 tickers)")
    print("• Market Data: vcsc_daily_data_complete (microstructure)")
    print("• Fundamentals: fundamental_values (quarterly, 721 tickers)")
    print("• Factor Scores: factor_scores_qvm (QVM engine v2_enhanced)")
    print("• Foreign Flows: vcsc_foreign_flow_summary")
    
    print(f"\n{Colors.BOLD}REPORT FORMATS:{Colors.ENDC}")
    print("• HTML Dashboard: Interactive charts, responsive design")
    print("• PDF Report: Print-ready, executive summary format")
    print("• JSON Data: Raw data for API integration")
    
    print(f"\n{Colors.BOLD}AUTOMATION:{Colors.ENDC}")
    print("• Daily Reports: Recommended generation at 7:30 AM ICT")
    print("• Weekly Reports: Recommended generation Friday 4:00 PM ICT")
    print("• Cron Integration: Add to daily workflow after factor generation")
    
    input(f"\n{Colors.CYAN}Press Enter to return to main menu...{Colors.ENDC}")

def show_data_documentation():
    """Show data sources documentation"""
    clear_screen()
    print_header("📊 DATA SOURCES DOCUMENTATION", "=")
    
    print(f"\n{Colors.BOLD}PRIMARY DATA TABLES:{Colors.ENDC}")
    print("• equity_history: Adjusted OHLCV prices (Wong Trading API)")
    print("• vcsc_daily_data_complete: Market microstructure (VCSC API)")
    print("• fundamental_values: Quarterly financial statements")
    print("• factor_scores_qvm: QVM factor scores (Quality, Value, Momentum)")
    print("• master_info: Company information and sector mappings")
    
    print(f"\n{Colors.BOLD}DATA FRESHNESS:{Colors.ENDC}")
    print("• Price Data: Updated daily after market close (18:30 ICT)")
    print("• Factor Scores: Updated daily when QVM engine runs")
    print("• Fundamentals: Updated quarterly (45-day lag from quarter end)")
    print("• Foreign Flows: Updated daily in real-time")
    
    print(f"\n{Colors.BOLD}DATA QUALITY INDICATORS:{Colors.ENDC}")
    print("• Coverage: 728 tickers, 16+ years historical depth")
    print("• Completeness: 98%+ daily price data coverage")
    print("• Accuracy: Cross-validated against multiple sources")
    print("• Timeliness: T+1 data availability for most metrics")
    
    input(f"\n{Colors.CYAN}Press Enter to return to main menu...{Colors.ENDC}")

def main():
    """Main menu loop"""
    while True:
        show_main_menu()
        
        choice = input(f"\n{Colors.BOLD}Select option: {Colors.ENDC}").strip().lower()
        
        # Daily products
        if choice.startswith('1.'):
            handle_daily_products(choice)
        
        # Weekly products
        elif choice.startswith('2.'):
            handle_weekly_products(choice)
        
        # Configuration
        elif choice.startswith('3.'):
            handle_configuration(choice)
        
        # Analysis & Research
        elif choice.startswith('4.'):
            print_header("ANALYSIS & RESEARCH", "-")
            print_warning("Advanced analysis features are under development")
        
        # Monitoring & Validation
        elif choice.startswith('5.'):
            print_header("MONITORING & VALIDATION", "-")
            print_warning("Monitoring features are under development")
        
        # Advanced Features
        elif choice.startswith('6.'):
            print_header("ADVANCED FEATURES", "-")
            print_warning("Advanced features are planned for future releases")
        
        # Help & documentation
        elif choice == 'h':
            show_help()
        
        elif choice == 'd':
            show_data_documentation()
        
        elif choice == 's':
            print_header("SIGNAL METHODOLOGY REFERENCE", "-")
            print_warning("Signal methodology documentation is under development")
        
        elif choice == 'r':
            print_header("RISK METRICS EXPLANATION", "-")
            print_warning("Risk metrics documentation is under development")
        
        # Exit
        elif choice == '0':
            print_success("\nExiting Market Intelligence Platform. Goodbye!")
            break
        
        else:
            print_error("Invalid option. Please try again.")
        
        if choice != '0':
            input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.ENDC}")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print_warning("\n\nOperation cancelled by user.")
        print_success("Exiting Market Intelligence Platform.")
        sys.exit(0)
    except Exception as e:
        print_error(f"\nUnexpected error: {e}")
        sys.exit(1)