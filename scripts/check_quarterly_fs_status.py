#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quarterly Financial Statement Data Status Checker
================================================
Author: Duc Nguyen
Date: July 30, 2025

This script dynamically checks and displays the current status of quarterly
financial statement data availability by sector.
"""

import mysql.connector
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load database configuration
import yaml
import configparser

def get_db_config():
    """Load database configuration from YAML files"""
    config_ini_path = PROJECT_ROOT / 'config' / 'config.ini'
    config = configparser.ConfigParser()
    config.read(config_ini_path)
    
    db_yaml_path = PROJECT_ROOT / 'config' / 'database.yml'
    with open(db_yaml_path, 'r') as f:
        db_yaml = yaml.safe_load(f)
    
    # Use production config
    return db_yaml.get('production', db_yaml.get('development'))

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

def get_latest_quarter():
    """
    Get the most recent completed quarter based on current date.
    Companies can report anytime after quarter end, so we check what's actually ended.
    """
    current_date = datetime.now()
    current_year = current_date.year
    current_month = current_date.month
    
    # Determine the last completed quarter
    if current_month >= 10:  # Oct, Nov, Dec
        latest_quarter = 3
        latest_year = current_year
    elif current_month >= 7:  # Jul, Aug, Sep
        latest_quarter = 2
        latest_year = current_year
    elif current_month >= 4:  # Apr, May, Jun
        latest_quarter = 1
        latest_year = current_year
    else:  # Jan, Feb, Mar
        latest_quarter = 4
        latest_year = current_year - 1
        
    return latest_year, latest_quarter

def check_quarterly_fs_status():
    """Main function to check quarterly FS data status"""
    
    # Get database config
    db_config = get_db_config()
    
    # Connect to database
    try:
        conn = mysql.connector.connect(
            host=db_config['host'],
            database=db_config['schema_name'],
            user=db_config['username'],
            password=db_config['password']
        )
        cursor = conn.cursor()
        
        # Get latest completed quarter
        latest_year, latest_quarter = get_latest_quarter()
        
        # Clear screen
        print('\033[2J\033[H')  # Clear screen and move cursor to top
        
        # Header
        print(f"{Colors.CYAN}{'='*80}{Colors.ENDC}")
        print(f"{Colors.CYAN}{Colors.BOLD}📊 QUARTERLY FINANCIAL STATEMENT DATA STATUS{Colors.ENDC}")
        print(f"{Colors.CYAN}{'='*80}{Colors.ENDC}")
        print(f"{Colors.BOLD}📅 Current Date:{Colors.ENDC} {datetime.now().strftime('%Y-%m-%d %H:%M')} ICT")
        print(f"{Colors.BOLD}📈 Latest Completed Quarter:{Colors.ENDC} Q{latest_quarter} {latest_year}")
        print(f"{Colors.CYAN}{'='*80}{Colors.ENDC}\n")
        
        # Query for sector-wise coverage
        query = """
        SELECT 
            mi.sector,
            COUNT(DISTINCT fv.ticker) as total_tickers,
            COUNT(DISTINCT CASE 
                WHEN fv.year = %s AND fv.quarter = %s 
                THEN fv.ticker 
            END) as latest_quarter_tickers,
            ROUND(
                COUNT(DISTINCT CASE 
                    WHEN fv.year = %s AND fv.quarter = %s 
                    THEN fv.ticker 
                END) * 100.0 / COUNT(DISTINCT fv.ticker), 
                2
            ) as coverage_pct
        FROM fundamental_values fv
        JOIN master_info mi ON fv.ticker = mi.ticker
        WHERE fv.year >= %s
        GROUP BY mi.sector
        ORDER BY coverage_pct DESC, mi.sector
        """
        
        cursor.execute(query, (latest_year, latest_quarter, latest_year, latest_quarter, latest_year - 1))
        results = cursor.fetchall()
        
        # Display results by coverage level
        print(f"{Colors.GREEN}{Colors.BOLD}✅ HIGH COVERAGE SECTORS (>70%):{Colors.ENDC}")
        print(f"{'Sector':<30} {'Total':<8} {'Q%d %d':<10} {'Coverage':<10}" % (latest_quarter, latest_year))
        print("-" * 60)
        
        high_coverage = [r for r in results if r[3] > 70]
        for sector, total, latest, pct in high_coverage:
            print(f"{sector:<30} {total:<8} {latest:<10} {Colors.GREEN}{pct:>6.1f}%{Colors.ENDC}")
        
        print(f"\n{Colors.YELLOW}{Colors.BOLD}⚠️  MEDIUM COVERAGE SECTORS (30-70%):{Colors.ENDC}")
        print(f"{'Sector':<30} {'Total':<8} {'Q%d %d':<10} {'Coverage':<10}" % (latest_quarter, latest_year))
        print("-" * 60)
        
        medium_coverage = [r for r in results if 30 < r[3] <= 70]
        for sector, total, latest, pct in medium_coverage:
            print(f"{sector:<30} {total:<8} {latest:<10} {Colors.YELLOW}{pct:>6.1f}%{Colors.ENDC}")
        
        print(f"\n{Colors.RED}{Colors.BOLD}❌ LOW COVERAGE SECTORS (<30%):{Colors.ENDC}")
        print(f"{'Sector':<30} {'Total':<8} {'Q%d %d':<10} {'Coverage':<10}" % (latest_quarter, latest_year))
        print("-" * 60)
        
        low_coverage = [r for r in results if r[3] <= 30]
        for sector, total, latest, pct in low_coverage:
            print(f"{sector:<30} {total:<8} {latest:<10} {Colors.RED}{pct:>6.1f}%{Colors.ENDC}")
        
        # Overall summary
        cursor.execute("""
        SELECT 
            COUNT(DISTINCT ticker) as total,
            COUNT(DISTINCT CASE WHEN year = %s AND quarter = %s THEN ticker END) as latest
        FROM fundamental_values
        WHERE year >= %s
        """, (latest_year, latest_quarter, latest_year - 1))
        
        total_tickers, latest_tickers = cursor.fetchone()
        overall_pct = (latest_tickers / total_tickers * 100) if total_tickers > 0 else 0
        
        print(f"\n{Colors.CYAN}{'='*80}{Colors.ENDC}")
        print(f"{Colors.BOLD}📊 OVERALL SUMMARY:{Colors.ENDC}")
        print(f"Total Tickers with Recent Data: {total_tickers}")
        print(f"Tickers with Q{latest_quarter} {latest_year} Data: {latest_tickers}")
        
        if overall_pct >= 70:
            color = Colors.GREEN
            status = "GOOD"
        elif overall_pct >= 50:
            color = Colors.YELLOW
            status = "MODERATE"
        else:
            color = Colors.RED
            status = "NEEDS UPDATE"
            
        print(f"Overall Coverage: {color}{overall_pct:.1f}% - {status}{Colors.ENDC}")
        
        # Check for anomalies (future quarters)
        cursor.execute("""
        SELECT DISTINCT year, quarter, COUNT(DISTINCT ticker) as ticker_count
        FROM fundamental_values
        WHERE (year > %s) OR (year = %s AND quarter > %s)
        GROUP BY year, quarter
        ORDER BY year DESC, quarter DESC
        """, (latest_year, latest_year, latest_quarter))
        
        future_data = cursor.fetchall()
        
        if future_data:
            print(f"\n{Colors.RED}{Colors.BOLD}⚠️  DATA ANOMALIES DETECTED:{Colors.ENDC}")
            print("The following future quarters have data (should be investigated):")
            for year, quarter, count in future_data:
                print(f"  - Q{quarter} {year}: {count} ticker(s)")
        
        # Recommendations
        print(f"\n{Colors.CYAN}{'='*80}{Colors.ENDC}")
        print(f"{Colors.BOLD}💡 RECOMMENDATIONS:{Colors.ENDC}")
        
        if overall_pct < 70:
            print(f"{Colors.YELLOW}• Run quarterly update process to fetch missing Q{latest_quarter} {latest_year} data{Colors.ENDC}")
            if low_coverage:
                print(f"{Colors.YELLOW}• Priority sectors needing updates:{Colors.ENDC}")
                for sector, _, _, pct in low_coverage[:5]:  # Top 5 sectors needing updates
                    print(f"  - {sector} ({pct:.1f}% coverage)")
        else:
            print(f"{Colors.GREEN}✅ Data coverage is good. Continue with regular daily updates.{Colors.ENDC}")
            
        # Next quarter info
        next_quarter = latest_quarter + 1 if latest_quarter < 4 else 1
        next_year = latest_year if latest_quarter < 4 else latest_year + 1
        next_quarter_end = {
            1: f"March 31, {next_year}",
            2: f"June 30, {next_year}",
            3: f"September 30, {next_year}",
            4: f"December 31, {next_year}"
        }[next_quarter]
        
        print(f"\n{Colors.BLUE}📅 Next Quarter: Q{next_quarter} {next_year} (ends {next_quarter_end}){Colors.ENDC}")
        print(f"{Colors.CYAN}{'='*80}{Colors.ENDC}")
        
        cursor.close()
        conn.close()
        
    except mysql.connector.Error as e:
        print(f"{Colors.RED}❌ Database Error: {e}{Colors.ENDC}")
        return False
    except Exception as e:
        print(f"{Colors.RED}❌ Error: {e}{Colors.ENDC}")
        return False
    
    return True

if __name__ == "__main__":
    check_quarterly_fs_status()