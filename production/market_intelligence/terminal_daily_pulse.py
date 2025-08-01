#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Terminal Daily Alpha Pulse - FACTUAL DATA ONLY
==============================================
Displays only actual data from database with no fallbacks or fabrications.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import sys
from pathlib import Path
import mysql.connector
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set up logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

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

def format_change(value, suffix='%'):
    """Format a change value with appropriate color"""
    if pd.isna(value) or value is None:
        return "N/A"
    if value > 0:
        return f"{Colors.GREEN}+{value:.1f}{suffix}{Colors.ENDC}"
    elif value < 0:
        return f"{Colors.RED}{value:.1f}{suffix}{Colors.ENDC}"
    else:
        return f"{value:.1f}{suffix}"

def format_ratio(value, suffix='x'):
    """Format a ratio with appropriate color"""
    if pd.isna(value) or value is None:
        return "N/A"
    if value > 1.0:
        return f"{Colors.GREEN}{value:.1f}{suffix}{Colors.ENDC}"
    elif value < 1.0:
        return f"{Colors.RED}{value:.1f}{suffix}{Colors.ENDC}"
    else:
        return f"{value:.1f}{suffix}"

def create_progress_bar(value, max_val=1.0, width=10):
    """Create a simple progress bar"""
    if pd.isna(value) or value is None or max_val == 0:
        return "░" * width
    
    filled = int((abs(value) / max_val) * width)
    bar = "█" * filled + "░" * (width - filled)
    
    if value > 0:
        return f"{Colors.GREEN}{bar}{Colors.ENDC}"
    elif value < 0:
        return f"{Colors.RED}{bar}{Colors.ENDC}"
    else:
        return bar

def generate_terminal_daily_pulse():
    """Generate factual daily alpha pulse - no fallbacks or fabricated data"""
    
    db_config = get_db_config()
    
    try:
        conn = mysql.connector.connect(
            host=db_config['host'],
            database=db_config['schema_name'],
            user=db_config['username'],
            password=db_config['password']
        )
        
        cursor = conn.cursor()
        
        # Get latest trading date from ETF table
        cursor.execute("SELECT MAX(date) FROM etf_history WHERE ticker = 'VNINDEX'")
        latest_etf_date = cursor.fetchone()[0]
        
        if not latest_etf_date:
            print(f"{Colors.RED}❌ No VNINDEX data found in etf_history table{Colors.ENDC}")
            return False
        
        # Get previous trading date
        cursor.execute("SELECT MAX(date) FROM etf_history WHERE ticker = 'VNINDEX' AND date < %s", (latest_etf_date,))
        prev_etf_date = cursor.fetchone()[0]
        
        # Get VN-Index data from ETF table
        if prev_etf_date:
            cursor.execute("""
                SELECT 
                    current.close as current_level,
                    (current.close - prev.close) / prev.close * 100 as daily_change,
                    current.volume as index_volume
                FROM etf_history current
                LEFT JOIN etf_history prev ON prev.ticker = 'VNINDEX' AND prev.date = %s
                WHERE current.ticker = 'VNINDEX' AND current.date = %s
            """, (prev_etf_date, latest_etf_date))
            
            vn_index_result = cursor.fetchone()
            if vn_index_result:
                vn_index_level = float(vn_index_result[0])
                vn_index_change = float(vn_index_result[1]) if vn_index_result[1] else None
                vn_index_volume = float(vn_index_result[2]) / 1e6 if vn_index_result[2] else None  # Millions
            else:
                vn_index_level = None
                vn_index_change = None
                vn_index_volume = None
        else:
            # Only current day data
            cursor.execute("SELECT close, volume FROM etf_history WHERE ticker = 'VNINDEX' AND date = %s", (latest_etf_date,))
            current_only = cursor.fetchone()
            if current_only:
                vn_index_level = float(current_only[0])
                vn_index_volume = float(current_only[1]) / 1e6 if current_only[1] else None
            else:
                vn_index_level = None
                vn_index_volume = None
            vn_index_change = None
        
        # Check if we can use current date data from VCSC when equity_history is incomplete
        cursor.execute("SELECT MAX(date) FROM equity_history")
        max_equity_date = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT ticker) FROM equity_history WHERE date = %s AND ticker NOT LIKE '%INDEX%'", (max_equity_date,))
        max_equity_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT MAX(trading_date) FROM vcsc_daily_data_complete")
        max_vcsc_date = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT ticker) FROM vcsc_daily_data_complete WHERE trading_date = %s", (max_vcsc_date,))
        max_vcsc_count = cursor.fetchone()[0]
        
        # Use VCSC data if it's more recent and complete, otherwise use equity_history
        if max_vcsc_date >= max_equity_date and max_vcsc_count > max_equity_count:
            latest_equity_date = max_vcsc_date
            total_stocks_available = max_vcsc_count
            use_vcsc_data = True
            print(f"{Colors.YELLOW}ℹ️  Using VCSC data from {latest_equity_date} ({total_stocks_available} stocks) - more complete than equity_history ({max_equity_count} stocks){Colors.ENDC}")
        else:
            # Get latest trading date with substantial data (>100 stocks) from equity_history
            cursor.execute("""
                SELECT date, COUNT(DISTINCT ticker) as stock_count
                FROM equity_history 
                WHERE ticker NOT LIKE '%INDEX%'
                GROUP BY date 
                HAVING stock_count > 100
                ORDER BY date DESC 
                LIMIT 1
            """)
            
            equity_date_result = cursor.fetchone()
            if not equity_date_result:
                print(f"{Colors.RED}❌ No equity data with sufficient stocks found{Colors.ENDC}")
                return False
                
            latest_equity_date = equity_date_result[0]
            total_stocks_available = equity_date_result[1]
            use_vcsc_data = False
            print(f"{Colors.YELLOW}ℹ️  Using equity_history data from {latest_equity_date} ({total_stocks_available} stocks){Colors.ENDC}")
            
        # Get previous equity trading date
        cursor.execute("SELECT MAX(date) FROM equity_history WHERE date < %s", (latest_equity_date,))
        prev_equity_date = cursor.fetchone()[0]
        
        # Get market breadth - adapt query based on data source
        advances, declines, total = None, None, None
        if prev_equity_date:
            if use_vcsc_data:
                # Use VCSC for current day, equity_history for previous day
                cursor.execute("""
                    SELECT 
                        SUM(CASE WHEN (current.close_price - prev.close) / prev.close > 0 THEN 1 ELSE 0 END) as advances,
                        SUM(CASE WHEN (current.close_price - prev.close) / prev.close < 0 THEN 1 ELSE 0 END) as declines,
                        COUNT(*) as total
                    FROM vcsc_daily_data_complete current
                    LEFT JOIN equity_history prev ON CONVERT(current.ticker USING utf8mb4) COLLATE utf8mb4_0900_ai_ci = CONVERT(prev.ticker USING utf8mb4) COLLATE utf8mb4_0900_ai_ci AND prev.date = %s
                    WHERE current.trading_date = %s 
                        AND current.close_price > 0 
                        AND prev.close > 0
                """, (prev_equity_date, latest_equity_date))
            else:
                # Use equity_history for both dates
                cursor.execute("""
                    SELECT 
                        SUM(CASE WHEN (current.close - prev.close) / prev.close > 0 THEN 1 ELSE 0 END) as advances,
                        SUM(CASE WHEN (current.close - prev.close) / prev.close < 0 THEN 1 ELSE 0 END) as declines,
                        COUNT(*) as total
                    FROM equity_history current
                    LEFT JOIN equity_history prev ON current.ticker = prev.ticker AND prev.date = %s
                    WHERE current.date = %s 
                        AND current.ticker NOT LIKE '%INDEX%' 
                        AND current.close > 0 
                        AND prev.close > 0
                """, (prev_equity_date, latest_equity_date))
            
            breadth_result = cursor.fetchone()
            if breadth_result:
                advances, declines, total = breadth_result
        
        # Get total market volume and turnover - adapt based on data source
        if use_vcsc_data:
            cursor.execute("""
                SELECT 
                    SUM(total_volume) as total_volume,
                    COUNT(DISTINCT ticker) as active_stocks,
                    SUM(total_volume * close_price) as total_turnover
                FROM vcsc_daily_data_complete 
                WHERE trading_date = %s 
                    AND total_volume > 0 
                    AND close_price > 0
            """, (latest_equity_date,))
        else:
            cursor.execute("""
                SELECT 
                    SUM(volume) as total_volume,
                    COUNT(DISTINCT ticker) as active_stocks,
                    SUM(volume * close) as total_turnover
                FROM equity_history 
                WHERE date = %s 
                    AND ticker NOT LIKE '%INDEX%' 
                    AND volume > 0 
                    AND close > 0
            """, (latest_equity_date,))
        
        market_stats = cursor.fetchone()
        if market_stats:
            total_volume = float(market_stats[0]) / 1e6 if market_stats[0] else None  # Millions
            active_stocks = int(market_stats[1]) if market_stats[1] else None
            total_turnover = float(market_stats[2]) / 1e9 if market_stats[2] else None  # Billions
        else:
            total_volume = None
            active_stocks = None
            total_turnover = None
        
        # Get volume ratio calculation
        volume_ratio = None
        cursor.execute("""
            WITH volume_comparison AS (
                SELECT 
                    current.ticker,
                    current.volume as current_volume,
                    AVG(hist.volume) as avg_20d_volume
                FROM equity_history current
                LEFT JOIN equity_history hist ON current.ticker = hist.ticker
                    AND hist.date BETWEEN DATE_SUB(current.date, INTERVAL 21 DAY) 
                                      AND DATE_SUB(current.date, INTERVAL 1 DAY)
                WHERE current.date = %s 
                    AND current.ticker NOT LIKE '%INDEX%' 
                    AND current.volume > 0
                GROUP BY current.ticker, current.volume
                HAVING avg_20d_volume > 0
            )
            SELECT AVG(current_volume / avg_20d_volume) as volume_ratio
            FROM volume_comparison
        """, (latest_equity_date,))
        
        volume_result = cursor.fetchone()
        if volume_result and volume_result[0]:
            volume_ratio = float(volume_result[0])
        
        # Get top traded stocks - adapt based on data source
        if use_vcsc_data:
            # Get top traded stocks by volume from VCSC
            cursor.execute("""
                SELECT 
                    ticker,
                    total_volume / 1e6 as volume_millions,
                    (close_price - open_price) / open_price * 100 as change_pct,
                    close_price,
                    total_volume * close_price / 1e9 as turnover_billions
                FROM vcsc_daily_data_complete 
                WHERE trading_date = %s 
                    AND total_volume > 0 
                    AND close_price > 0
                    AND open_price > 0
                ORDER BY total_volume DESC
                LIMIT 5
            """, (latest_equity_date,))
            
            top_volume_stocks = cursor.fetchall()
            
            # Get top traded stocks by turnover from VCSC
            cursor.execute("""
                SELECT 
                    ticker,
                    total_volume * close_price / 1e9 as turnover_billions,
                    (close_price - open_price) / open_price * 100 as change_pct,
                    close_price,
                    total_volume / 1e6 as volume_millions
                FROM vcsc_daily_data_complete 
                WHERE trading_date = %s 
                    AND total_volume > 0 
                    AND close_price > 0
                    AND open_price > 0
                ORDER BY total_volume * close_price DESC
                LIMIT 5
            """, (latest_equity_date,))
        else:
            # Get top traded stocks by volume from equity_history
            cursor.execute("""
                SELECT 
                    ticker,
                    volume / 1e6 as volume_millions,
                    (close - open) / open * 100 as change_pct,
                    close,
                    volume * close / 1e9 as turnover_billions
                FROM equity_history 
                WHERE date = %s 
                    AND ticker NOT LIKE '%INDEX%' 
                    AND volume > 0 
                    AND close > 0
                    AND open > 0
                ORDER BY volume DESC
                LIMIT 5
            """, (latest_equity_date,))
            
            top_volume_stocks = cursor.fetchall()
            
            # Get top traded stocks by turnover from equity_history
            cursor.execute("""
                SELECT 
                    ticker,
                    volume * close / 1e9 as turnover_billions,
                    (close - open) / open * 100 as change_pct,
                    close,
                    volume / 1e6 as volume_millions
                FROM equity_history 
                WHERE date = %s 
                    AND ticker NOT LIKE '%INDEX%' 
                    AND volume > 0 
                    AND close > 0
                    AND open > 0
                ORDER BY volume * close DESC
                LIMIT 5
            """, (latest_equity_date,))
        
        top_turnover_stocks = cursor.fetchall()
        
        # Get foreign flow data from vcsc_daily_data_complete
        # First try the current date, then fall back to most recent available date
        cursor.execute("""
            SELECT 
                SUM(foreign_net_value_total) as net_value,
                %s as trading_date
            FROM vcsc_daily_data_complete
            WHERE trading_date = %s
                AND foreign_net_value_total IS NOT NULL
        """, (latest_equity_date, latest_equity_date))
        
        foreign_result = cursor.fetchone()
        
        if not foreign_result or foreign_result[0] is None:
            # Fall back to most recent available date
            cursor.execute("""
                SELECT 
                    SUM(foreign_net_value_total) as net_value,
                    MAX(trading_date) as trading_date
                FROM vcsc_daily_data_complete
                WHERE trading_date = (
                    SELECT MAX(trading_date) 
                    FROM vcsc_daily_data_complete 
                    WHERE foreign_net_value_total IS NOT NULL
                )
                AND foreign_net_value_total IS NOT NULL
            """)
            foreign_result = cursor.fetchone()
        
        foreign_net = float(foreign_result[0]) / 1e9 if foreign_result and foreign_result[0] else None  # Billions
        foreign_date = foreign_result[1] if foreign_result else None
        
        # Get factor performance - use most recent available factor data
        factors = {}
        latest_factor_date = None
        
        try:
            # Find the most recent factor data date
            cursor.execute("""
                SELECT MAX(date) 
                FROM factor_scores_qvm 
                WHERE strategy_version = 'qvm_v2.0_enhanced'
            """)
            
            latest_factor_date = cursor.fetchone()[0]
            
            if latest_factor_date:
                cursor.execute("""
                    WITH factor_returns AS (
                        SELECT 
                            f1.Quality_Composite, f1.Value_Composite, f1.Momentum_Composite,
                            f1.ticker,
                            (r.close - r.open) / r.open as daily_return
                        FROM factor_scores_qvm f1
                        JOIN equity_history r ON f1.ticker = r.ticker AND r.date = %s
                        WHERE f1.date = %s
                            AND f1.strategy_version = 'qvm_v2.0_enhanced'
                            AND r.close > 0 AND r.open > 0
                            AND f1.Quality_Composite IS NOT NULL 
                            AND f1.Value_Composite IS NOT NULL 
                            AND f1.Momentum_Composite IS NOT NULL
                    ),
                    quintiles AS (
                        SELECT 
                            ticker,
                            daily_return,
                            NTILE(5) OVER (ORDER BY Quality_Composite DESC) as q_quintile,
                            NTILE(5) OVER (ORDER BY Value_Composite DESC) as v_quintile,
                            NTILE(5) OVER (ORDER BY Momentum_Composite DESC) as m_quintile
                        FROM factor_returns
                    )
                    SELECT 
                        AVG(CASE WHEN q_quintile = 1 THEN daily_return END) - 
                        AVG(CASE WHEN q_quintile = 5 THEN daily_return END) as quality_ls,
                        AVG(CASE WHEN v_quintile = 1 THEN daily_return END) - 
                        AVG(CASE WHEN v_quintile = 5 THEN daily_return END) as value_ls,
                        AVG(CASE WHEN m_quintile = 1 THEN daily_return END) - 
                        AVG(CASE WHEN m_quintile = 5 THEN daily_return END) as momentum_ls,
                        COUNT(*) as stock_count
                    FROM quintiles
                """, (latest_equity_date, latest_factor_date))
            
            factor_result = cursor.fetchone()
            
            if factor_result and factor_result[3] and factor_result[3] > 0:  # Check if we have stocks
                quality_ls = float(factor_result[0] * 100) if factor_result[0] is not None else None
                value_ls = float(factor_result[1] * 100) if factor_result[1] is not None else None
                momentum_ls = float(factor_result[2] * 100) if factor_result[2] is not None else None
                
                factors = {
                    'Quality': quality_ls,
                    'Value': value_ls,
                    'Momentum': momentum_ls
                }
        except Exception as e:
            logger.warning(f"Could not fetch factor performance: {e}")
            factors = {}
        
        cursor.close()
        conn.close()
        
        # Calculate derived metrics only if we have the data
        advance_decline_ratio = None
        if advances and declines and declines > 0:
            advance_decline_ratio = advances / declines
        
        market_participation = None
        if active_stocks and total and total > 0:
            market_participation = active_stocks / total * 100
        
        # Generate the display
        current_time = datetime.now()
        
        print()
        print("┌────────────────────────────────────────────┐")
        print("│          DAILY ALPHA PULSE                 │")
        print(f"│         {current_time.strftime('%B %d, %Y %H:%M')} ICT           │")
        print("├────────────────────────────────────────────┤")
        print("│ MARKET OVERVIEW            │ TRADING DATA  │")
        
        # VN-Index display
        vn_index_display = "N/A"
        if vn_index_level is not None:
            vn_change_str = format_change(vn_index_change) if vn_index_change is not None else "N/A"
            vn_index_display = f"{vn_index_level:>7.1f} {vn_change_str:>7}"
        
        # Volume display
        vol_display = f"{total_volume:>6.0f}M" if total_volume is not None else "N/A"
        
        # Turnover display  
        to_display = f"{total_turnover:>6.1f}B" if total_turnover is not None else "N/A"
        
        # Breadth display
        if advances is not None and declines is not None:
            if advance_decline_ratio is not None:
                breadth_display = f"{advances:>3}/{declines:<3} ({advance_decline_ratio:.1f})"
            else:
                breadth_display = f"{advances:>3}/{declines:<3}"
        else:
            breadth_display = "N/A"
        
        # Active stocks display
        active_display = f"{active_stocks:>4} stocks" if active_stocks is not None else "N/A"
        
        # Foreign flow display
        if foreign_net is not None:
            ff_display = format_change(foreign_net, 'B')
            if foreign_date and foreign_date != latest_equity_date:
                ff_display = format_change(foreign_net, 'B*')  # Add asterisk for different date
        else:
            ff_display = "N/A"
        
        print(f"│ • VN-Index: {vn_index_display:>15} │ • Vol: {vol_display:>8} │")
        print(f"│ • Breadth: {breadth_display:>16} │ • T/O: {to_display:>8} │")
        print(f"│ • Active: {active_display:>17} │ • FF: {ff_display:>9} │")
        
        # Factor performance
        print("├────────────────────────────────────────────┤")
        print("│ FACTOR PERFORMANCE                         │")
        
        if factors:
            max_abs_factor = max(abs(v) for v in factors.values() if v is not None) if any(v is not None for v in factors.values()) else 1
            
            for factor, value in factors.items():
                if value is not None:
                    bar = create_progress_bar(value, max_abs_factor)
                    strength = "Strong" if abs(value) > 2.0 else "Weak" if abs(value) < 0.5 else "Neutral"
                    print(f"│ • {factor:<8}: {format_change(value):>8} {bar} {strength:<8} │")
                else:
                    print(f"│ • {factor:<8}: {'N/A':>8} {'░'*10} N/A      │")
        else:
            print("│ • No factor data available                 │")
        
        # Top traded stocks
        print("├────────────────────────────────────────────┤")
        print("│ TOP TRADED BY VOLUME       │ BY TURNOVER   │")
        
        max_rows = min(3, len(top_volume_stocks) if top_volume_stocks else 0, len(top_turnover_stocks) if top_turnover_stocks else 0)
        
        if max_rows > 0:
            for i in range(max_rows):
                vol_stock = top_volume_stocks[i] if i < len(top_volume_stocks) else None
                to_stock = top_turnover_stocks[i] if i < len(top_turnover_stocks) else None
                
                if vol_stock:
                    vol_ticker = vol_stock[0][:4]
                    vol_change = format_change(float(vol_stock[2]), '')
                    vol_millions = vol_stock[1]
                    vol_display = f"{i+1}. {vol_ticker} {vol_change:>7} ({vol_millions:>4.0f}M)"
                else:
                    vol_display = "N/A"
                
                if to_stock:
                    to_ticker = to_stock[0][:4]
                    to_change = format_change(float(to_stock[2]), '')
                    to_billions = to_stock[1]
                    to_display = f"{to_ticker} {to_change:>7} ({to_billions:.1f}B)"
                else:
                    to_display = "N/A"
                
                print(f"│ {vol_display:<22} │ {to_display:<13} │")
        else:
            print("│ No trading data available  │               │")
        
        print("└────────────────────────────────────────────┘")
        
        # Summary
        print()
        print(f"{Colors.BOLD}Market Summary:{Colors.ENDC}")
        
        # Sentiment
        if advances is not None and declines is not None:
            sentiment = 'Bullish' if advances > declines else 'Bearish' if declines > advances else 'Neutral'
            print(f"• Sentiment: {sentiment} ({advances} advances vs {declines} declines)")
        else:
            print("• Sentiment: N/A (no breadth data)")
        
        # VN-Index summary
        if vn_index_level is not None:
            vol_str = f" (Vol: {vn_index_volume:.0f}M)" if vn_index_volume is not None else ""
            change_str = format_change(vn_index_change) if vn_index_change is not None else "N/A"
            print(f"• VN-Index: {vn_index_level:.1f} {change_str}{vol_str}")
        else:
            print("• VN-Index: N/A")
        
        # Market volume summary
        if total_volume is not None and total_turnover is not None:
            print(f"• Market Volume: {total_volume:.0f}M shares, Turnover: {total_turnover:.1f}B VND")
        else:
            print("• Market Volume: N/A")
        
        # Participation
        if market_participation is not None and active_stocks is not None and total is not None:
            print(f"• Market Participation: {market_participation:.1f}% ({active_stocks}/{total} stocks active)")
        else:
            print("• Market Participation: N/A")
        
        # Foreign flow and A/D ratio
        if foreign_net is not None:
            ff_str = format_change(foreign_net, 'B VND')
            if foreign_date and foreign_date != latest_equity_date:
                ff_str += f" ({foreign_date.strftime('%m-%d')})"
        else:
            ff_str = "N/A"
        ad_str = f", A/D Ratio: {advance_decline_ratio:.2f}" if advance_decline_ratio is not None else ""
        print(f"• Foreign Flow: {ff_str} (net){ad_str}")
        
        # Data dates with stock count
        factor_date_str = f", Factors({latest_factor_date.strftime('%Y-%m-%d')})" if latest_factor_date else ", Factors(N/A)"
        print(f"• Data as of: Equity({latest_equity_date.strftime('%Y-%m-%d')}, {total_stocks_available} stocks), ETF({latest_etf_date.strftime('%Y-%m-%d')}){factor_date_str}")
        
        print(f"\n{Colors.CYAN}Generated at {current_time.strftime('%Y-%m-%d %H:%M:%S')} ICT{Colors.ENDC}")
        
        return True
        
    except Exception as e:
        print(f"{Colors.RED}❌ Error generating Daily Alpha Pulse: {e}{Colors.ENDC}")
        return False

if __name__ == "__main__":
    generate_terminal_daily_pulse()