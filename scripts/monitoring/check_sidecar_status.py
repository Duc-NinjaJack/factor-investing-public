#!/usr/bin/env python3
"""
Analytics Sidecar Status Monitor
================================
Monitor the factor_signals_raw and factor_norm_stats tables for coverage and completeness.

This script provides visibility into the analytics sidecar data that captures individual
factor values (not composites) at the point of sector neutralization.

Author: Vietnam Factor Investing Team
Date: August 2025
"""

import pymysql
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import sys


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


def print_header(text: str):
    """Print a formatted header"""
    print(f"\n{Colors.CYAN}{'='*80}{Colors.ENDC}")
    print(f"{Colors.CYAN}{Colors.BOLD}{text}{Colors.ENDC}")
    print(f"{Colors.CYAN}{'='*80}{Colors.ENDC}")


def print_section(text: str):
    """Print a section header"""
    print(f"\n{Colors.BLUE}{Colors.BOLD}{text}{Colors.ENDC}")
    print(f"{Colors.BLUE}{'-'*40}{Colors.ENDC}")


def load_db_config() -> Dict:
    """Load database configuration from database.yml"""
    config_path = Path(__file__).parent.parent.parent / 'production' / 'config' / 'database.yml'
    
    if not config_path.exists():
        print(f"{Colors.RED}Database config not found at {config_path}{Colors.ENDC}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    return config.get('production', config)


def get_db_connection():
    """Create database connection"""
    config = load_db_config()
    
    return pymysql.connect(
        host=config['host'],
        user=config['username'],
        password=config['password'],
        database=config['schema_name'],
        charset='utf8mb4'
    )


def check_sidecar_overview(conn) -> None:
    """Check overall sidecar data status"""
    print_section("📊 SIDECAR OVERVIEW")
    
    with conn.cursor() as cursor:
        # Check factor_signals_raw
        cursor.execute("""
            SELECT 
                COUNT(*) as total_records,
                COUNT(DISTINCT date) as trading_days,
                COUNT(DISTINCT ticker) as unique_tickers,
                COUNT(DISTINCT factor_id) as unique_factors,
                MIN(date) as earliest_date,
                MAX(date) as latest_date,
                COUNT(DISTINCT strategy_version) as versions
            FROM factor_signals_raw
            WHERE strategy_version = 'analytics_v1_neo_fixed'
        """)
        
        result = cursor.fetchone()
        if result and result[0] > 0:
            print(f"{Colors.GREEN}✅ factor_signals_raw:{Colors.ENDC}")
            print(f"   • Records: {result[0]:,}")
            print(f"   • Trading days: {result[1]:,}")
            print(f"   • Unique tickers: {result[2]:,}")
            print(f"   • Unique factors: {result[3]}")
            print(f"   • Date range: {result[4]} to {result[5]}")
            print(f"   • Strategy versions: {result[6]}")
        else:
            print(f"{Colors.YELLOW}⚠️  No data in factor_signals_raw{Colors.ENDC}")
        
        # Check factor_norm_stats
        cursor.execute("""
            SELECT 
                COUNT(*) as total_records,
                COUNT(DISTINCT date) as trading_days,
                COUNT(DISTINCT sector) as unique_sectors,
                COUNT(DISTINCT factor_id) as unique_factors,
                MIN(date) as earliest_date,
                MAX(date) as latest_date
            FROM factor_norm_stats
            WHERE strategy_version = 'analytics_v1_neo_fixed'
        """)
        
        result = cursor.fetchone()
        if result and result[0] > 0:
            print(f"\n{Colors.GREEN}✅ factor_norm_stats:{Colors.ENDC}")
            print(f"   • Records: {result[0]:,}")
            print(f"   • Trading days: {result[1]:,}")
            print(f"   • Unique sectors: {result[2]}")
            print(f"   • Unique factors: {result[3]}")
            print(f"   • Date range: {result[4]} to {result[5]}")
        else:
            print(f"{Colors.YELLOW}⚠️  No data in factor_norm_stats{Colors.ENDC}")


def check_factor_coverage(conn) -> None:
    """Check per-factor coverage"""
    print_section("🎯 PER-FACTOR COVERAGE")
    
    with conn.cursor() as cursor:
        cursor.execute("""
            SELECT 
                df.factor_code,
                COUNT(DISTINCT fsr.date) as days_covered,
                COUNT(DISTINCT fsr.ticker) as tickers_covered,
                COUNT(*) as total_records,
                MIN(fsr.date) as earliest,
                MAX(fsr.date) as latest
            FROM factor_signals_raw fsr
            JOIN dim_factor df ON fsr.factor_id = df.factor_id
            WHERE fsr.strategy_version = 'analytics_v1_neo_fixed'
            GROUP BY df.factor_code
            ORDER BY df.factor_code
        """)
        
        results = cursor.fetchall()
        
        if results:
            print(f"{'Factor':<20} {'Days':<8} {'Tickers':<10} {'Records':<12} {'Date Range':<25}")
            print("-" * 75)
            
            for row in results:
                factor_code, days, tickers, records, earliest, latest = row
                date_range = f"{earliest} to {latest}"
                print(f"{factor_code:<20} {days:<8} {tickers:<10} {records:<12,} {date_range:<25}")
        else:
            print(f"{Colors.YELLOW}No factor coverage data found{Colors.ENDC}")


def check_recent_activity(conn) -> None:
    """Check recent sidecar activity"""
    print_section("⏰ RECENT ACTIVITY")
    
    with conn.cursor() as cursor:
        # Last 5 dates with data
        cursor.execute("""
            SELECT 
                date,
                COUNT(DISTINCT ticker) as tickers,
                COUNT(DISTINCT factor_id) as factors,
                COUNT(*) as records
            FROM factor_signals_raw
            WHERE strategy_version = 'analytics_v1_neo_fixed'
            GROUP BY date
            ORDER BY date DESC
            LIMIT 5
        """)
        
        results = cursor.fetchall()
        
        if results:
            print(f"{'Date':<15} {'Tickers':<10} {'Factors':<10} {'Records':<10}")
            print("-" * 45)
            
            for row in results:
                date, tickers, factors, records = row
                print(f"{str(date):<15} {tickers:<10} {factors:<10} {records:<10,}")
        else:
            print(f"{Colors.YELLOW}No recent activity found{Colors.ENDC}")


def check_data_gaps(conn) -> None:
    """Check for potential data gaps"""
    print_section("🔍 DATA GAP ANALYSIS")
    
    with conn.cursor() as cursor:
        # Compare with composite dates
        cursor.execute("""
            WITH composite_dates AS (
                SELECT DISTINCT date 
                FROM factor_scores_qvm 
                WHERE strategy_version = 'qvm_v2.1.1_flat_corrected'
                AND date >= '2024-01-01'
            ),
            sidecar_dates AS (
                SELECT DISTINCT date 
                FROM factor_signals_raw 
                WHERE strategy_version = 'analytics_v1_neo_fixed'
                AND date >= '2024-01-01'
            )
            SELECT 
                (SELECT COUNT(*) FROM composite_dates) as composite_days,
                (SELECT COUNT(*) FROM sidecar_dates) as sidecar_days,
                (SELECT COUNT(*) FROM composite_dates WHERE date NOT IN (SELECT date FROM sidecar_dates)) as missing_days
        """)
        
        result = cursor.fetchone()
        if result:
            composite_days, sidecar_days, missing_days = result
            
            if missing_days == 0:
                print(f"{Colors.GREEN}✅ No gaps detected!{Colors.ENDC}")
                print(f"   Composite dates: {composite_days}")
                print(f"   Sidecar dates: {sidecar_days}")
            else:
                print(f"{Colors.YELLOW}⚠️  Gaps detected:{Colors.ENDC}")
                print(f"   Composite dates: {composite_days}")
                print(f"   Sidecar dates: {sidecar_days}")
                print(f"   Missing in sidecar: {missing_days}")
                
                # Show first 5 missing dates
                cursor.execute("""
                    WITH composite_dates AS (
                        SELECT DISTINCT date 
                        FROM factor_scores_qvm 
                        WHERE strategy_version = 'qvm_v2.1.1_flat_corrected'
                    ),
                    sidecar_dates AS (
                        SELECT DISTINCT date 
                        FROM factor_signals_raw 
                        WHERE strategy_version = 'analytics_v1_neo_fixed'
                    )
                    SELECT date FROM composite_dates 
                    WHERE date NOT IN (SELECT date FROM sidecar_dates)
                    ORDER BY date DESC
                    LIMIT 5
                """)
                
                missing = cursor.fetchall()
                if missing:
                    print(f"\n   Recent missing dates:")
                    for (date,) in missing:
                        print(f"     • {date}")


def main():
    """Main execution function"""
    print_header("📊 ANALYTICS SIDECAR STATUS MONITOR")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        conn = get_db_connection()
        
        # Run all checks
        check_sidecar_overview(conn)
        check_factor_coverage(conn)
        check_recent_activity(conn)
        check_data_gaps(conn)
        
        print(f"\n{Colors.GREEN}✅ Status check complete{Colors.ENDC}")
        
    except Exception as e:
        print(f"\n{Colors.RED}❌ Error: {e}{Colors.ENDC}")
        sys.exit(1)
    
    finally:
        if 'conn' in locals():
            conn.close()


if __name__ == "__main__":
    main()