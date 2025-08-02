#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Daily Data Status Monitor
=========================
Comprehensive status check for all daily data sources:
- Market data (equity_history) 
- VCSC complete data
- Foreign flow data
- Data quality metrics and pipeline health

Author: Duc Nguyen
Date: August 2, 2025
Status: Production Ready
"""

import sys
import pandas as pd
import numpy as np
import mysql.connector
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

class Colors:
    """Terminal color codes"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    BLUE = '\033[94m'  
    CYAN = '\033[96m'
    YELLOW = '\033[93m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'

def get_db_config():
    """Load database configuration"""
    db_yaml_path = PROJECT_ROOT / 'config' / 'database.yml'
    with open(db_yaml_path, 'r') as f:
        db_yaml = yaml.safe_load(f)
    return db_yaml.get('production', db_yaml.get('development'))

def format_status(is_good: bool, text: str) -> str:
    """Format status with appropriate color"""
    if is_good:
        return f"{Colors.GREEN}✅ {text}{Colors.ENDC}"
    else:
        return f"{Colors.RED}❌ {text}{Colors.ENDC}"

def format_warning(text: str) -> str:
    """Format warning message"""
    return f"{Colors.YELLOW}⚠️  {text}{Colors.ENDC}"

def format_info(text: str) -> str:
    """Format info message"""
    return f"{Colors.BLUE}ℹ️  {text}{Colors.ENDC}"

def get_business_days_lag(target_date, current_date: datetime) -> int:
    """Calculate business days lag between dates"""
    if target_date is None:
        return 999
    
    # Convert to date objects if needed
    if hasattr(target_date, 'date'):
        target_date = target_date.date()
    if hasattr(current_date, 'date'):
        current_date = current_date.date()
        
    # If target date is today or future, no lag
    if target_date >= current_date:
        return 0
    
    # Calculate business days between dates
    total_days = (current_date - target_date).days
    if total_days == 0:
        return 0
    
    business_days = 0
    check_date = target_date
    
    while check_date < current_date:
        check_date += timedelta(days=1)
        # Only count weekdays (Monday=0, Sunday=6)
        if check_date.weekday() < 5:  # Monday to Friday
            business_days += 1
    
    # For market data, if it's the current trading day and before market close (say 3:30 PM ICT)
    # and we have yesterday's data, that's considered current
    if current_date.weekday() < 5:  # Today is a weekday
        current_time = current_date if isinstance(current_date, datetime) else datetime.combine(current_date, datetime.min.time())
        if hasattr(current_date, 'hour'):
            current_time = current_date
        else:
            current_time = datetime.now()
            
        # If it's before market close (15:30 ICT) and we have previous trading day data
        if current_time.hour < 15 or (current_time.hour == 15 and current_time.minute < 30):
            # Check if target_date is the previous trading day
            prev_trading_day = current_date
            while prev_trading_day.weekday() >= 5:  # Go back to find last trading day
                prev_trading_day -= timedelta(days=1)
            prev_trading_day -= timedelta(days=1)  # One day before
            while prev_trading_day.weekday() >= 5:  # Make sure it's a trading day
                prev_trading_day -= timedelta(days=1)
                
            if target_date == prev_trading_day:
                return 0  # Previous trading day data is considered current before market close
    
    return business_days

def check_market_data(cursor) -> Dict[str, Any]:
    """Check equity_history market data status"""
    
    # Get latest date and stock count
    cursor.execute("""
        SELECT 
            MAX(date) as latest_date,
            COUNT(DISTINCT ticker) as total_stocks
        FROM equity_history 
        WHERE ticker NOT LIKE '%INDEX%'
    """)
    
    result = cursor.fetchone()
    latest_date = result[0] if result[0] else None
    total_stocks = result[1] if result[1] else 0
    
    # Get stock count for latest date
    latest_date_stocks = 0
    if latest_date:
        cursor.execute("""
            SELECT COUNT(DISTINCT ticker) 
            FROM equity_history 
            WHERE date = %s AND ticker NOT LIKE '%INDEX%'
        """, (latest_date,))
        latest_date_stocks = cursor.fetchone()[0]
    
    # Check ETF/Index data
    cursor.execute("""
        SELECT MAX(date) as latest_date
        FROM etf_history 
        WHERE ticker = 'VNINDEX'
    """)
    etf_result = cursor.fetchone()
    etf_latest_date = etf_result[0] if etf_result[0] else None
    
    # Calculate lag
    current_date = datetime.now()
    market_lag = get_business_days_lag(latest_date, current_date) if latest_date else 999
    etf_lag = get_business_days_lag(etf_latest_date, current_date) if etf_latest_date else 999
    
    return {
        'latest_date': latest_date,
        'total_stocks': total_stocks,
        'latest_date_stocks': latest_date_stocks,
        'market_lag': market_lag,
        'etf_latest_date': etf_latest_date,
        'etf_lag': etf_lag,
        'is_healthy': market_lag <= 2 and latest_date_stocks > 600
    }

def check_vcsc_data(cursor) -> Dict[str, Any]:
    """Check vcsc_daily_data_complete status"""
    
    # Get latest date and record count
    cursor.execute("""
        SELECT 
            MAX(trading_date) as latest_date,
            COUNT(*) as total_records
        FROM vcsc_daily_data_complete
    """)
    
    result = cursor.fetchone()
    latest_date = result[0] if result[0] else None
    total_records = result[1] if result[1] else 0
    
    # Get record count for latest date
    latest_date_records = 0
    if latest_date:
        cursor.execute("""
            SELECT COUNT(*) 
            FROM vcsc_daily_data_complete 
            WHERE trading_date = %s
        """, (latest_date,))
        latest_date_records = cursor.fetchone()[0]
    
    # Check foreign flow availability
    foreign_flow_available = False
    foreign_flow_total = None
    if latest_date:
        cursor.execute("""
            SELECT 
                COUNT(*) as records_with_ff,
                SUM(foreign_net_value_total) as total_ff
            FROM vcsc_daily_data_complete 
            WHERE trading_date = %s 
                AND foreign_net_value_total IS NOT NULL
        """, (latest_date,))
        
        ff_result = cursor.fetchone()
        if ff_result and ff_result[0] > 0:
            foreign_flow_available = True
            foreign_flow_total = ff_result[1]
    
    # Calculate lag
    current_date = datetime.now()
    vcsc_lag = get_business_days_lag(latest_date, current_date) if latest_date else 999
    
    return {
        'latest_date': latest_date,
        'total_records': total_records,
        'latest_date_records': latest_date_records,
        'vcsc_lag': vcsc_lag,
        'foreign_flow_available': foreign_flow_available,
        'foreign_flow_total': foreign_flow_total,
        'is_healthy': vcsc_lag <= 3 and latest_date_records > 700
    }

def check_data_quality(cursor) -> Dict[str, Any]:
    """Perform data quality checks"""
    
    issues = []
    
    # Check for missing prices in latest equity data
    cursor.execute("""
        SELECT COUNT(*) as missing_prices
        FROM equity_history 
        WHERE date = (SELECT MAX(date) FROM equity_history WHERE ticker NOT LIKE '%INDEX%')
            AND (close IS NULL OR close <= 0)
            AND ticker NOT LIKE '%INDEX%'
    """)
    
    missing_prices = cursor.fetchone()[0] or 0
    if missing_prices > 0:
        issues.append(f"{missing_prices} stocks missing/invalid prices")
    
    # Check for volume anomalies (>10x average)
    cursor.execute("""
        WITH recent_avg AS (
            SELECT 
                ticker,
                AVG(volume) as avg_volume,
                MAX(CASE WHEN date = (SELECT MAX(date) FROM equity_history WHERE ticker NOT LIKE '%INDEX%') 
                         THEN volume END) as latest_volume
            FROM equity_history 
            WHERE date >= DATE_SUB((SELECT MAX(date) FROM equity_history WHERE ticker NOT LIKE '%INDEX%'), INTERVAL 30 DAY)
                AND ticker NOT LIKE '%INDEX%'
                AND volume > 0
            GROUP BY ticker
            HAVING avg_volume > 0 AND latest_volume IS NOT NULL
        )
        SELECT COUNT(*) as volume_spikes
        FROM recent_avg
        WHERE latest_volume > avg_volume * 10
    """)
    
    volume_spikes = cursor.fetchone()[0] or 0
    if volume_spikes > 5:
        issues.append(f"{volume_spikes} stocks with unusual volume spikes")
    
    # Check VCSC vs equity_history price reconciliation for latest common date
    cursor.execute("""
        SELECT COUNT(*) as price_mismatches
        FROM (
            SELECT 
                e.ticker,
                e.close as equity_close,
                v.close_price as vcsc_close,
                ABS(e.close - v.close_price) / e.close as price_diff
            FROM equity_history e
            JOIN vcsc_daily_data_complete v ON CONVERT(e.ticker USING utf8mb4) COLLATE utf8mb4_unicode_ci = CONVERT(v.ticker USING utf8mb4) COLLATE utf8mb4_unicode_ci
            WHERE e.date = (SELECT MAX(e2.date) FROM equity_history e2 WHERE e2.ticker NOT LIKE '%INDEX%')
                AND v.trading_date = (SELECT MAX(v2.trading_date) FROM vcsc_daily_data_complete v2)
                AND e.close > 0 
                AND v.close_price > 0
        ) price_comparison
        WHERE price_diff > 0.05  -- More than 5% difference
    """)
    
    price_mismatches = cursor.fetchone()[0] or 0
    if price_mismatches > 10:
        issues.append(f"{price_mismatches} stocks with VCSC/equity price mismatches >5%")
    
    return {
        'missing_prices': missing_prices,
        'volume_spikes': volume_spikes,
        'price_mismatches': price_mismatches,
        'issues': issues,
        'is_healthy': len(issues) <= 1  # Allow up to 1 minor issue
    }

def check_pipeline_health(cursor) -> Dict[str, Any]:
    """Check data pipeline health and freshness"""
    
    # Get last update timestamps from various tables
    pipeline_status = {}
    
    tables_to_check = [
        ('equity_history', 'date'),
        ('vcsc_daily_data_complete', 'trading_date'),
        ('etf_history', 'date')
    ]
    
    for table, date_col in tables_to_check:
        try:
            cursor.execute(f"""
                SELECT 
                    MAX({date_col}) as latest_date,
                    COUNT(*) as total_records
                FROM {table}
            """)
            
            result = cursor.fetchone()
            pipeline_status[table] = {
                'latest_date': result[0],
                'total_records': result[1],
                'is_current': get_business_days_lag(result[0], datetime.now()) <= 2 if result[0] else False
            }
        except Exception as e:
            pipeline_status[table] = {
                'latest_date': None,
                'total_records': 0,
                'is_current': False,
                'error': str(e)
            }
    
    # Overall pipeline health
    all_current = all(status.get('is_current', False) for status in pipeline_status.values())
    
    return {
        'pipeline_status': pipeline_status,
        'is_healthy': all_current
    }

def generate_daily_data_status_report():
    """Generate comprehensive daily data status report"""
    
    db_config = get_db_config()
    
    try:
        conn = mysql.connector.connect(
            host=db_config['host'],
            database=db_config['schema_name'],
            user=db_config['username'],
            password=db_config['password']
        )
        
        cursor = conn.cursor()
        current_time = datetime.now()
        
        # Header
        print(f"\n{Colors.CYAN}{'='*70}{Colors.ENDC}")
        print(f"{Colors.CYAN}{Colors.BOLD}🔍 DAILY DATA STATUS REPORT{Colors.ENDC}")
        print(f"{Colors.CYAN}{Colors.BOLD}{current_time.strftime('%B %d, %Y %H:%M ICT')}{Colors.ENDC}")
        print(f"{Colors.CYAN}{'━'*70}{Colors.ENDC}")
        
        # 1. Market Data Check
        print(f"\n{Colors.BOLD}📈 MARKET DATA (equity_history):{Colors.ENDC}")
        market_data = check_market_data(cursor)
        
        if market_data['latest_date']:
            if market_data['market_lag'] == 0:
                lag_text = "current"
            elif market_data['market_lag'] == 1:
                lag_text = "1 trading day lag"
            else:
                lag_text = f"{market_data['market_lag']} trading days lag"
            
            stock_text = f"({market_data['latest_date_stocks']} stocks)"
            print(format_status(market_data['market_lag'] <= 1, 
                              f"Latest Date: {market_data['latest_date'].strftime('%Y-%m-%d')} {stock_text} - {lag_text}"))
        else:
            print(format_status(False, "No market data found"))
        
        if market_data['etf_latest_date']:
            if market_data['etf_lag'] == 0:
                etf_lag_text = "current"
            elif market_data['etf_lag'] == 1:
                etf_lag_text = "1 trading day lag"
            else:
                etf_lag_text = f"{market_data['etf_lag']} trading days lag"
                
            print(format_status(market_data['etf_lag'] <= 1,
                              f"ETF Data: {market_data['etf_latest_date'].strftime('%Y-%m-%d')} (VNINDEX available) - {etf_lag_text}"))
        else:
            print(format_status(False, "ETF Data: Not available"))
        
        # 2. VCSC Data Check
        print(f"\n{Colors.BOLD}💰 VCSC DATA (vcsc_daily_data_complete):{Colors.ENDC}")
        vcsc_data = check_vcsc_data(cursor)
        
        if vcsc_data['latest_date']:
            if vcsc_data['vcsc_lag'] == 0:
                lag_text = "current"
            elif vcsc_data['vcsc_lag'] == 1:
                lag_text = "1 trading day lag"
            else:
                lag_text = f"{vcsc_data['vcsc_lag']} trading days lag"
                
            record_text = f"({vcsc_data['latest_date_records']} stocks)"
            print(format_status(vcsc_data['vcsc_lag'] <= 2,
                              f"Latest Date: {vcsc_data['latest_date'].strftime('%Y-%m-%d')} {record_text} - {lag_text}"))
        else:
            print(format_status(False, "No VCSC data found"))
        
        if vcsc_data['foreign_flow_available'] and vcsc_data['foreign_flow_total'] is not None:
            ff_value = float(vcsc_data['foreign_flow_total']) / 1e12  # Convert to trillions
            ff_direction = "inflow" if ff_value > 0 else "outflow" 
            print(format_status(True, f"Foreign Flow: Available ({ff_value:+.1f}T VND net {ff_direction})"))
        else:
            print(format_status(False, "Foreign Flow: Not available"))
        
        # 3. Data Quality Check
        print(f"\n{Colors.BOLD}🔄 DATA QUALITY:{Colors.ENDC}")
        quality_data = check_data_quality(cursor)
        
        if quality_data['missing_prices'] == 0:
            print(format_status(True, "Price Coverage: Complete (all stocks have valid prices)"))
        else:
            print(format_warning(f"Price Coverage: {quality_data['missing_prices']} stocks missing prices"))
        
        if quality_data['volume_spikes'] <= 3:
            print(format_status(True, f"Volume Patterns: Normal ({quality_data['volume_spikes']} minor spikes)"))
        else:
            print(format_warning(f"Volume Patterns: {quality_data['volume_spikes']} stocks with unusual volume"))
        
        if quality_data['price_mismatches'] <= 5:
            print(format_status(True, f"Price Reconciliation: Good ({quality_data['price_mismatches']} minor mismatches)"))
        else:
            print(format_warning(f"Price Reconciliation: {quality_data['price_mismatches']} significant mismatches"))
        
        # 4. Pipeline Health
        print(f"\n{Colors.BOLD}📊 PIPELINE SUMMARY:{Colors.ENDC}")
        pipeline_data = check_pipeline_health(cursor)
        
        overall_health = (market_data['is_healthy'] and 
                         vcsc_data['is_healthy'] and 
                         quality_data['is_healthy'] and 
                         pipeline_data['is_healthy'])
        
        if overall_health:
            print(format_status(True, "Overall Status: HEALTHY - All systems operational"))
        else:
            print(format_status(False, "Overall Status: ATTENTION NEEDED - Check issues above"))
        
        # Last updated info
        print(f"\n{Colors.BLUE}ℹ️  Report generated: {current_time.strftime('%Y-%m-%d %H:%M:%S ICT')}{Colors.ENDC}")
        
        # Summary of critical issues
        if quality_data['issues']:
            print(f"\n{Colors.YELLOW}⚠️  ATTENTION REQUIRED:{Colors.ENDC}")
            for issue in quality_data['issues']:
                print(f"  • {issue}")
        
        cursor.close()
        conn.close()
        
        return overall_health
        
    except Exception as e:
        print(f"{Colors.RED}❌ Error generating Daily Data Status Report: {e}{Colors.ENDC}")
        return False

if __name__ == "__main__":
    generate_daily_data_status_report()