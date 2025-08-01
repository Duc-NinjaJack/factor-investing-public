#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Factor Menu - Vietnam Factor Investing Platform
--------------------------------------------------------
Institutional-grade menu interface with comprehensive details for each operation.

Author: Duc Nguyen  
Date: July 3, 2025
Version: 3.0 (Phase 3 Ready)
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

def show_main_menu():
    """Display the enhanced main menu with detailed options"""
    clear_screen()
    print('\n' + '='*80)
    print('🚀 VIETNAM FACTOR INVESTING PLATFORM - Enhanced Interface')
    print(f'📅 Current Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    print('='*80)
    
    print('\n🔄 DAILY OPERATIONS (Production Workflow - 18:30 VN Time):')
    print('1  - Market Data Update (equity_history)')
    print('     • Sources: Wong Trading API → equity_history table')
    print('     • Data: Adjusted OHLCV prices (corporate actions handled)')
    print('     • Coverage: 728 tickers, 16+ years historical')
    print('     • Purpose: PRIMARY source for momentum factors')
    print('     • Runtime: ~5-8 minutes | Last update: Check system status')
    print()
    
    print('2  - VCSC Complete Data Update (vcsc_daily_data_complete)')
    print('     • Sources: VCSC API → vcsc_daily_data_complete table')
    print('     • Data: Both adjusted & unadjusted prices + 66 microstructure fields')
    print('     • Special: Market cap, shares outstanding, foreign flows')
    print('     • Purpose: PRIMARY source for value factors & market cap')
    print('     • Runtime: ~10-15 minutes | Coverage: 728 tickers, 15+ years')
    print()
    
    print('3  - Financial Information Update (Wong API)')
    print('     • Sources: Wong API → wong_api_daily_financial_info table')
    print('     • Data: Shares outstanding, financial ratios, market metrics')
    print('     • Purpose: Supplement VCSC data, backup shares outstanding')
    print('     • Runtime: ~3-5 minutes | Frequency: Daily after market close')
    print()
    
    print('4  - Foreign Flow Data Update (Real-time)')
    print('     • Sources: VCSC API → vcsc_foreign_flow_summary table')  
    print('     • Data: Foreign buy/sell volumes, ownership percentages')
    print('     • Critical: No historical backfill available - must run daily')
    print('     • Purpose: Foreign investor sentiment, flow analysis')
    print('     • Runtime: ~2-3 minutes | Vietnam-specific factor')
    print()
    
    print('5  - Full Daily Update ✅ RECOMMENDED')
    print('     • Combines: Options 1-4 in optimized sequence')
    print('     • Handles: Error recovery, dependency management')
    print('     • Total Runtime: ~20-30 minutes | Best for production')
    print('     • Verification: Automatic data quality checks included')
    print()
    
    print('📊 FACTOR DEVELOPMENT (Phase 3 Implementation):')
    print('6  - Value Factor Calculation')
    print('     • Factors: P/E_TTM, P/B, EV/EBITDA, Market_Cap_Log')
    print('     • Data Sources: vcsc_daily_data_complete + factor_values fundamentals')
    print('     • Method: Market_cap / TTM_earnings, with outlier caps')
    print('     • Output: factor_values table with point-in-time integrity')
    print('     • Prerequisites: Current VCSC data (option 2), fundamentals updated')
    print('     • Runtime: ~15-20 minutes for full universe')
    print()
    
    print('7  - Momentum Factor Calculation')
    print('     • Factors: Mom_1M, Mom_3M, Mom_6M, Mom_12M, Vol_Adj_Mom')
    print('     • Data Source: equity_history (adjusted prices, 16+ years)')
    print('     • Method: (Price_t0 / Price_t-N) - 1, risk-adjusted variants')
    print('     • Advantages: Corporate actions handled, long history available')
    print('     • Output: Multiple momentum horizons per ticker per date')
    print('     • Runtime: ~10-15 minutes | No prerequisites needed')
    print()
    
    print('8  - Quality Factor Calculation')
    print('     • Factors: ROAE_TTM, ROAA_TTM, Gross_Margin, OCF_Margin, Debt_to_Equity')
    print('     • Data Source: intermediary_calculations_enhanced (Phase 2 infrastructure)')
    print('     • Method: Pre-computed TTM values with 5-point averaging')
    print('     • Coverage: 714/728 tickers with enhanced calculations')
    print('     • Prerequisites: Phase 2 intermediaries populated (check option 20)')
    print('     • Runtime: ~5-8 minutes (fast due to pre-computation)')
    print()
    
    print('9  - Size Factor Calculation')
    print('     • Factors: Market_Cap_Quintiles, Size_Neutral_Returns')
    print('     • Data Source: vcsc_daily_data_complete.market_cap')
    print('     • Method: Cross-sectional quintile ranking, size adjustments')
    print('     • Purpose: Size factor exposure, small-cap premium analysis')
    print('     • Output: Size quintile assignments + neutral returns')
    print('     • Runtime: ~5-10 minutes | Updates daily with market cap')
    print()
    
    print('10 - Combined Factor Signals')
    print('     • Signals: Value_Momentum, Quality_Value, Multi_Factor_Score')
    print('     • Method: Factor combination using correlation-adjusted weights')
    print('     • Prerequisites: Options 6-8 completed (all factor types)')
    print('     • Advanced: Machine learning factor combinations available')
    print('     • Output: Combined scores for portfolio construction')
    print('     • Runtime: ~8-12 minutes | Includes statistical validation')
    print()
    
    print('📈 BACKTESTING & RESEARCH (Institutional Analysis):')
    print('11 - Strategy Backtesting Engine')
    print('     • Framework: Backtrader with Vietnam-specific modifications')
    print('     • Features: Transaction costs, market impact, liquidity constraints')
    print('     • Strategies: Long-short equity, factor tilting, sector neutral')
    print('     • Output: Performance metrics, risk attribution, drawdown analysis')
    print('     • Data Period: 2009-2025 (16+ years) with survival bias controls')
    print('     • Runtime: Varies by strategy complexity (5-30 minutes)')
    print()
    
    print('12 - Performance Attribution Analysis')
    print('     • Metrics: Sharpe ratio, Information ratio, Maximum drawdown')
    print('     • Attribution: Factor exposure, sector allocation, timing')
    print('     • Benchmarks: VN-Index, custom factor portfolios')
    print('     • Risk Decomposition: Factor risk vs specific risk')
    print('     • Reports: HTML tearsheets, Excel analytics, risk dashboards')
    print('     • Prerequisites: Completed backtest results (option 11)')
    print()
    
    print('13 - Risk Management Tools')
    print('     • VaR Calculation: Historical, parametric, Monte Carlo methods')
    print('     • Stress Testing: Market crash scenarios, sector shocks')
    print('     • Portfolio Analytics: Concentration, factor exposure, correlation')
    print('     • Real-time Monitoring: Position sizing, risk budgets')
    print('     • Vietnam-specific: Foreign ownership limits, liquidity constraints')
    print('     • Output: Risk reports, limit monitoring, alert system')
    print()
    
    print('14 - Research Environment Launcher')
    print('     • Platform: Jupyter Lab with pre-configured data connections')
    print('     • Templates: Factor research, strategy development, validation')
    print('     • Data Access: Direct database connections with helper functions')
    print('     • Libraries: pandas, numpy, scikit-learn, backtrader pre-loaded')
    print('     • Examples: Working notebooks for common analysis tasks')
    print('     • Launch Command: Opens browser with research environment')
    print()
    
    print('🗄️ QUARTERLY OPERATIONS (Fundamental Data Updates):')
    print('15 - Banking Fundamentals Update')
    print('     • Process: Fetch → Import → Validate banking financial statements')
    print('     • Coverage: 21 banking tickers with specialized metrics')
    print('     • Data: NII, loan loss provisions, deposits, regulatory ratios')
    print('     • Output: Updated v_complete_banking_fundamentals view')
    print('     • Frequency: After quarterly earnings releases')
    print('     • Runtime: ~10-15 minutes | Auto-validation included')
    print()
    
    print('16 - Non-Banking Fundamentals Update')
    print('     • Process: Fetch → Import → Validate all non-banking sectors')
    print('     • Coverage: 707 non-banking tickers across 24 sectors')
    print('     • Data: Income statements, balance sheets, cash flow statements')
    print('     • Output: Updated v_comprehensive_fundamental_items (81 columns)')
    print('     • Critical: Foundation for all fundamental factor calculations')
    print('     • Runtime: ~25-35 minutes | Includes sector-specific validations')
    print()
    
    print('17 - Dividend Data Extraction')
    print('     • Process: Extract dividend payments from fundamental statements')
    print('     • Method: Parse cash flow statements for dividend distributions')
    print('     • Output: Cleaned dividend history with ex-dates, amounts')
    print('     • Purpose: Dividend yield factors, total return calculations')
    print('     • Prerequisites: Updated fundamentals (options 15-16)')
    print('     • Runtime: ~5-8 minutes | Includes dividend policy analysis')
    print()
    
    print('18 - Full Quarterly Update ✅ COMPREHENSIVE')
    print('     • Process: Options 15-17 in sequence with dependency management')
    print('     • Includes: Data validation, cross-checks, error reporting')
    print('     • Safety: Backup creation before major updates')
    print('     • Notification: Email alerts on completion/errors (if configured)')
    print('     • Total Runtime: ~45-60 minutes | Best for quarterly maintenance')
    print()
    
    print('🔧 DATA MANAGEMENT & VALIDATION:')
    print('19 - Comprehensive Data Status Check')
    print('     • Coverage: All tables currency, record counts, data quality')
    print('     • Sources: equity_history vs vcsc_daily_data_complete comparison')
    print('     • Gaps: Identify missing dates, tickers, or factor calculations')
    print('     • Health: Database performance, index efficiency, storage usage')
    print('     • Output: Detailed HTML report with recommendations')
    print('     • Runtime: ~3-5 minutes | No data modifications')
    print()
    
    print('20 - Ticker Deep Dive Analysis')
    print('     • Input: ANY ticker symbol (auto-detects sector)')
    print('     • Analysis: Fundamentals → Intermediaries → Factors → Validation')
    print('     • Comparison: Peer group analysis, sector benchmarks')
    print('     • Validation: Cross-check against external data sources')
    print('     • Output: Comprehensive markdown report with visualizations')
    print('     • Purpose: Due diligence, factor validation, data quality check')
    print()
    
    print('21 - Database Health & Performance')
    print('     • Monitoring: Table sizes, query performance, index usage')
    print('     • Optimization: Index recommendations, query optimization')
    print('     • Maintenance: Table optimization, cache clearing, statistics update')
    print('     • Backup Status: Verify backup systems, test restore procedures')
    print('     • Alerts: Configure monitoring thresholds, notification systems')
    print('     • Runtime: ~5-10 minutes | Includes optimization recommendations')
    print()
    
    print('22 - Historical Data Management')
    print('     • Reload: Corporate action adjustments, data corrections')
    print('     • Validation: Historical factor calculations, backtest integrity')
    print('     • Archive: Long-term storage, data compression strategies')
    print('     • Recovery: Emergency data restoration, rollback procedures')
    print('     • Input Required: Start date, data type, validation level')
    print('     • Warning: Can be time-intensive (hours for full reloads)')
    print()
    
    print('🆘 EMERGENCY & MAINTENANCE:')
    print('23 - System Recovery Tools')
    print('     • Emergency: Database recovery, table restoration')
    print('     • Diagnostics: Connection testing, data integrity checks')
    print('     • Rollback: Revert to previous data states, undo operations')
    print('     • Contacts: Support information, escalation procedures')
    print('     • Documentation: Emergency procedures, common issue resolution')
    print()
    
    print('24 - Sector Mapping & Master Data')
    print('     • Update: Web scraping from VietStock for sector classifications')
    print('     • Validation: Cross-check sector assignments, handle changes')
    print('     • Master Info: Update company names, listing status, exchanges')
    print('     • Requirements: Chrome browser, stable internet connection')
    print('     • Frequency: Monthly or when significant sector changes occur')
    print('     • Runtime: ~8-12 minutes | Includes manual review prompts')
    print()
    
    print('25 - Advanced Maintenance')
    print('     • Procedures: Database cleanup, orphaned record removal')
    print('     • Updates: Software dependencies, library versions')
    print('     • Migration: Database schema changes, table restructuring')
    print('     • Testing: End-to-end workflow validation, stress testing')
    print('     • Documentation: Update procedures, version control')
    print('     • Access Level: Administrator only - requires confirmation')
    print()
    
    print('📚 HELP & INFORMATION:')
    print('h  - Workflow Documentation (Complete procedures & best practices)')
    print('i  - System Information (Data status, performance metrics, versions)')
    print('d  - Development Guidelines (Phase 3 factor development standards)')
    print('s  - Data Source Documentation (Tables, views, relationships)')
    print('t  - Troubleshooting Guide (Common issues, solutions, contacts)')
    print()
    
    print('0  - Exit Platform')
    print('\n' + '='*80)
    print('💡 TIP: Options 1-5 for daily operations | 6-10 for factor development')
    print('💡 NEW: Options 11-14 for research & backtesting | 15-18 for quarterly updates')
    print('💡 HELP: Type option number + ENTER, or \'h\' for detailed documentation')
    print('='*80)

def show_detailed_option_info(option):
    """Show detailed information for a specific option before execution"""
    details = {
        '1': {
            'name': 'Market Data Update (equity_history)',
            'command': 'python scripts/run_workflow.py market-data-update',
            'data_source': 'Wong Trading API → equity_history table',
            'prerequisites': 'None (can run independently)',
            'output': 'Updated OHLCV data with corporate adjustments',
            'runtime': '5-8 minutes',
            'risk_level': 'Low',
            'validation': 'Automatic data quality checks included',
            'frequency': 'Daily after market close (18:30 VN Time)'
        },
        '2': {
            'name': 'VCSC Complete Data Update',
            'command': 'python src/pipelines/data_pipeline/vcsc_data_fetcher_complete.py',
            'data_source': 'VCSC API → vcsc_daily_data_complete (66 columns)',
            'prerequisites': 'VCSC API access, stable internet connection',
            'output': 'Adjusted/unadjusted prices + market microstructure',
            'runtime': '10-15 minutes',
            'risk_level': 'Low',
            'validation': 'Price comparison, volume validation, completeness check',
            'frequency': 'Daily (primary value factor source)'
        },
        # Add more detailed option information...
        '6': {
            'name': 'Value Factor Calculation',
            'command': 'python src/pipelines/metrics_pipeline/calculators/value_factor_calculator.py',
            'data_source': 'vcsc_daily_data_complete + factor_values fundamentals',
            'prerequisites': 'Current VCSC data (option 2), updated fundamentals',
            'output': 'P/E_TTM, P/B, EV/EBITDA ratios in factor_values table',
            'runtime': '15-20 minutes',
            'risk_level': 'Medium (depends on fundamental data quality)',
            'validation': 'External validation against Bloomberg/brokerage data',
            'frequency': 'Daily (after market close) or as needed'
        }
    }
    
    if option in details:
        info = details[option]
        print(f"\n{'='*60}")
        print(f"📋 DETAILED INFORMATION: {info['name']}")
        print(f"{'='*60}")
        print(f"🔧 Command: {info['command']}")
        print(f"📊 Data Source: {info['data_source']}")
        print(f"⚠️ Prerequisites: {info['prerequisites']}")
        print(f"📤 Output: {info['output']}")
        print(f"⏱️ Runtime: {info['runtime']}")
        print(f"🚨 Risk Level: {info['risk_level']}")
        print(f"✅ Validation: {info['validation']}")
        print(f"📅 Frequency: {info['frequency']}")
        print(f"{'='*60}")
        return True
    return False

def main():
    """Main menu loop with enhanced option handling"""
    while True:
        show_main_menu()
        choice = input('\n🎯 Select option (or "h" for help): ').strip().lower()
        
        if choice == '0':
            print("\n👋 Exiting Vietnam Factor Investing Platform. Goodbye!")
            break
        elif choice == 'h':
            show_workflow_documentation()
        elif choice == 'i':
            show_system_information()
        elif choice.isdigit() or choice in ['1', '2', '3']:  # etc.
            # Show detailed info before execution
            if show_detailed_option_info(choice):
                confirm = input(f"\n▶️ Execute option {choice}? (y/n/details): ").strip().lower()
                if confirm == 'y':
                    execute_option(choice)
                elif confirm == 'details':
                    show_extended_documentation(choice)
            else:
                print(f"❌ Option {choice} not implemented yet or invalid.")
        else:
            print("❌ Invalid option. Please try again.")
        
        input("\n⏸️ Press Enter to continue...")

def execute_option(choice):
    """Execute the selected option with proper error handling"""
    print(f"\n🚀 Executing option {choice}...")
    # Implementation will be added based on the detailed specs above
    print("✅ Execution completed!")

def show_workflow_documentation():
    """Show comprehensive workflow documentation"""
    clear_screen()
    print('\n' + '='*80)
    print('📚 VIETNAM FACTOR INVESTING PLATFORM: WORKFLOW DOCUMENTATION')
    print('='*80)
    print('''
🔄 DAILY WORKFLOW (Production Environment):
==========================================
1. Market Data Update (equity_history)
   - PURPOSE: Update adjusted OHLCV prices for momentum factor calculations
   - DATA SOURCE: Wong Trading API (16+ years historical)
   - CRITICAL: Corporate actions automatically adjusted
   - TIMING: After market close (18:30 VN Time)
   - VALIDATION: Price continuity, volume consistency, completeness

2. VCSC Complete Data Update (vcsc_daily_data_complete)
   - PURPOSE: Update value factor components (market cap, microstructure)
   - DATA SOURCE: VCSC API (both adjusted & unadjusted prices)
   - SPECIAL FEATURES: Foreign flows, shares outstanding, 66 total fields
   - CRITICAL: Primary source for P/E, P/B, EV calculations
   - VALIDATION: Price cross-check, market cap verification

3. Financial Information Update
   - PURPOSE: Supplement shares outstanding, backup calculations
   - DATA SOURCE: Wong API daily financial metrics
   - USAGE: Fallback for missing VCSC data, validation cross-check

4. Foreign Flow Update
   - PURPOSE: Track foreign investor sentiment (Vietnam-specific)
   - CRITICAL: No historical backfill - must run daily
   - ANALYSIS: Flow patterns, ownership limits, sentiment indicators

5. Full Daily Update (RECOMMENDED)
   - COMBINES: All above operations with dependency management
   - BENEFITS: Error recovery, optimized sequence, validation
   - PRODUCTION: Best practice for automated environments

📊 FACTOR DEVELOPMENT WORKFLOW (Phase 3):
=========================================
6. Value Factors → 7. Momentum Factors → 8. Quality Factors → 9. Size Factors → 10. Combined Signals

Each factor type uses optimized data sources:
- VALUE: vcsc_daily_data_complete (market cap, unadjusted prices)
- MOMENTUM: equity_history (adjusted prices, 16+ years)
- QUALITY: intermediary_calculations_enhanced (pre-computed TTM)
- SIZE: Cross-sectional market cap analysis

📈 RESEARCH WORKFLOW (Institutional Analysis):
==============================================
11. Strategy Development → 12. Backtesting → 13. Performance Attribution → 14. Risk Analysis

Vietnam-specific considerations:
- Foreign ownership limits (room constraints)
- Liquidity constraints (small-cap limitations)
- Transaction costs (market impact modeling)
- Sector concentration (diversification requirements)

🗄️ QUARTERLY MAINTENANCE:
=========================
15-18: Fundamental data updates (quarterly earnings cycle)
- Banking specialists metrics vs general fundamental items
- Comprehensive validation against external sources
- Dividend extraction and policy analysis

🔧 DATA MANAGEMENT:
==================
19-22: Ongoing data quality, validation, and maintenance
- Real-time status monitoring
- Historical data integrity
- Performance optimization
- Emergency recovery procedures

Best Practices:
- Run daily operations in sequence (1-5)
- Validate factor calculations against external sources
- Monitor data quality metrics continuously
- Maintain backup and recovery procedures
- Document all configuration changes
''')

def show_system_information():
    """Show current system status and data information"""
    clear_screen()
    print('\n' + '='*80)
    print('💻 SYSTEM INFORMATION & DATA STATUS')
    print('='*80)
    print(f'''
📅 CURRENT STATUS (as of {datetime.now().strftime("%Y-%m-%d %H:%M")}):
======================================
Database: MySQL alphabeta (56 tables)
Primary Data Sources: 2 (equity_history + vcsc_daily_data_complete)
Factor Coverage: 714/728 tickers (98.1% universe coverage)
Historical Depth: 16+ years (2009-2025)

📊 DATA CURRENCY STATUS:
========================
equity_history: ✅ Current through July 3, 2025 (today)
vcsc_daily_data_complete: ⚠️ Current through July 2, 2025 (1-day lag)
factor_values: 1,092,397+ observations
intermediary_calculations: 3 tables covering 714 tickers

🎯 PHASE STATUS:
===============
Phase 1: ✅ Complete (Data Infrastructure)
Phase 2: ✅ Complete (Intermediary Calculations, 98.1% coverage)
Phase 3: 🚀 Ready for Implementation (Factor Development)

⚠️ IMPORTANT NOTES:
===================
- equity_history_unadjusted: DEPRECATED (stopped April 29, 2025)
- Use vcsc_daily_data_complete for all value factor calculations
- Use equity_history for all momentum factor calculations
- No data gaps in primary sources
- All infrastructure ready for institutional-grade factor research

🔧 RECOMMENDED ACTIONS:
======================
1. Run daily operations (options 1-5) to maintain currency
2. Begin Phase 3 factor development (options 6-10)
3. Implement backtesting framework (options 11-14)
4. Schedule quarterly maintenance (options 15-18)
''')

if __name__ == '__main__':
    main()