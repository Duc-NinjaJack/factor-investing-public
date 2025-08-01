#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Daily Alpha Pulse - MVP Version
=====================================
A simplified version of the Daily Alpha Pulse that focuses on core functionality
and handles edge cases gracefully.
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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_db_config():
    """Load database configuration"""
    db_yaml_path = PROJECT_ROOT / 'config' / 'database.yml'
    with open(db_yaml_path, 'r') as f:
        db_yaml = yaml.safe_load(f)
    return db_yaml.get('production', db_yaml.get('development'))

def generate_simple_daily_pulse():
    """Generate a simple daily alpha pulse report"""
    
    # Get database connection
    db_config = get_db_config()
    
    try:
        conn = mysql.connector.connect(
            host=db_config['host'],
            database=db_config['schema_name'],
            user=db_config['username'],
            password=db_config['password']
        )
        
        logger.info("Database connection established")
        
        # Get latest trading date
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(date) FROM equity_history")
        latest_date = cursor.fetchone()[0]
        logger.info(f"Latest trading date: {latest_date}")
        
        # Get basic market data
        market_query = """
        SELECT 
            COUNT(*) as total_stocks,
            SUM(CASE WHEN close > open THEN 1 ELSE 0 END) as gainers,
            SUM(CASE WHEN close < open THEN 1 ELSE 0 END) as losers,
            AVG((close - open) / open * 100) as avg_change
        FROM equity_history 
        WHERE date = %s 
            AND ticker NOT LIKE '%INDEX%'
            AND close > 0 AND open > 0
        """
        
        cursor.execute(market_query, (latest_date,))
        market_data = cursor.fetchone()
        
        total_stocks, gainers, losers, avg_change = market_data
        gainers = gainers or 0
        losers = losers or 0
        avg_change = avg_change or 0.0
        unchanged = total_stocks - gainers - losers
        
        # Get sector performance (simplified)
        sector_query = """
        SELECT 
            mi.sector,
            COUNT(*) as stock_count,
            AVG((eh.close - eh.open) / eh.open * 100) as avg_return
        FROM equity_history eh
        JOIN master_info mi ON eh.ticker = mi.ticker
        WHERE eh.date = %s
            AND eh.close > 0 AND eh.open > 0
        GROUP BY mi.sector
        ORDER BY avg_return DESC
        """
        
        cursor.execute(sector_query, (latest_date,))
        sector_data = cursor.fetchall()
        
        # Get foreign flow summary (if available)
        try:
            foreign_query = """
            SELECT 
                SUM(net_value) as total_net_value,
                SUM(buy_value) as total_buy_value,
                SUM(sell_value) as total_sell_value
            FROM vcsc_foreign_flow_summary
            WHERE date = %s
            """
            cursor.execute(foreign_query, (latest_date,))
            foreign_result = cursor.fetchone()
            foreign_data = {
                'net_value': foreign_result[0] or 0,
                'buy_value': foreign_result[1] or 0,
                'sell_value': foreign_result[2] or 0
            } if foreign_result else {'net_value': 0, 'buy_value': 0, 'sell_value': 0}
        except Exception as e:
            logger.warning(f"Could not fetch foreign flow data: {e}")
            foreign_data = {'net_value': 0, 'buy_value': 0, 'sell_value': 0}
        
        cursor.close()
        conn.close()
        
        # Generate HTML report
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Daily Alpha Pulse - {datetime.now().strftime('%Y-%m-%d')}</title>
    <style>
        body {{ 
            font-family: 'Segoe UI', Arial, sans-serif; 
            margin: 20px; 
            background-color: #f5f5f5; 
        }}
        .header {{ 
            background: linear-gradient(135deg, #1976D2, #42A5F5);
            color: white; 
            padding: 30px; 
            border-radius: 10px; 
            text-align: center;
            margin-bottom: 20px;
        }}
        .header h1 {{ margin: 0; font-size: 2.5em; }}
        .header p {{ margin: 10px 0 0 0; font-size: 1.2em; opacity: 0.9; }}
        .section {{ 
            background: white; 
            padding: 20px; 
            margin: 20px 0; 
            border-radius: 10px; 
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .section h2 {{ 
            color: #1976D2; 
            border-bottom: 2px solid #E3F2FD; 
            padding-bottom: 10px; 
        }}
        .metric-grid {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); 
            gap: 15px; 
            margin: 20px 0; 
        }}
        .metric-card {{ 
            background: #f8f9fa; 
            padding: 15px; 
            border-radius: 8px; 
            text-align: center;
            border-left: 4px solid #1976D2;
        }}
        .metric-value {{ 
            font-size: 2em; 
            font-weight: bold; 
            margin: 5px 0; 
        }}
        .positive {{ color: #2E7D32; }}
        .negative {{ color: #C62828; }}
        .neutral {{ color: #757575; }}
        .sector-table {{ 
            width: 100%; 
            border-collapse: collapse; 
            margin: 15px 0;
        }}
        .sector-table th, .sector-table td {{ 
            padding: 10px; 
            text-align: left; 
            border-bottom: 1px solid #ddd; 
        }}
        .sector-table th {{ 
            background-color: #f8f9fa; 
            font-weight: bold; 
        }}
        .timestamp {{ 
            text-align: center; 
            color: #666; 
            font-size: 0.9em; 
            margin-top: 30px; 
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Daily Alpha Pulse</h1>
        <p>Vietnam Market Intelligence • {datetime.now().strftime('%A, %B %d, %Y')}</p>
        <p>Data as of: {latest_date.strftime('%Y-%m-%d')}</p>
    </div>
    
    <div class="section">
        <h2>🏢 Market Overview</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value positive">{gainers}</div>
                <div>Gainers</div>
            </div>
            <div class="metric-card">
                <div class="metric-value negative">{losers}</div>
                <div>Losers</div>
            </div>
            <div class="metric-card">
                <div class="metric-value neutral">{unchanged}</div>
                <div>Unchanged</div>
            </div>
            <div class="metric-card">
                <div class="metric-value {'positive' if avg_change > 0 else 'negative' if avg_change < 0 else 'neutral'}">{avg_change:.2f}%</div>
                <div>Avg Change</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{total_stocks}</div>
                <div>Total Stocks</div>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>🏭 Sector Performance</h2>
        <table class="sector-table">
            <thead>
                <tr>
                    <th>Sector</th>
                    <th>Stock Count</th>
                    <th>Avg Return (%)</th>
                </tr>
            </thead>
            <tbody>
"""
        
        # Add sector data to HTML
        for sector, count, avg_ret in sector_data[:15]:  # Top 15 sectors
            avg_ret = avg_ret or 0.0
            color_class = 'positive' if avg_ret > 0 else 'negative' if avg_ret < 0 else 'neutral'
            html_content += f"""
                <tr>
                    <td>{sector}</td>
                    <td>{count}</td>
                    <td class="{color_class}">{avg_ret:.2f}%</td>
                </tr>
            """
        
        html_content += f"""
            </tbody>
        </table>
    </div>
    
    <div class="section">
        <h2>🌍 Foreign Flows</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value {'positive' if foreign_data['net_value'] > 0 else 'negative' if foreign_data['net_value'] < 0 else 'neutral'}">{foreign_data['net_value']/1e9:.1f}B</div>
                <div>Net Flow (VND)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value positive">{foreign_data['buy_value']/1e9:.1f}B</div>
                <div>Buy Value (VND)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value negative">{foreign_data['sell_value']/1e9:.1f}B</div>
                <div>Sell Value (VND)</div>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>🚀 Key Insights</h2>
        <ul>
            <li><strong>Market Sentiment:</strong> {'Bullish' if gainers > losers else 'Bearish' if losers > gainers else 'Neutral'} ({gainers} gainers vs {losers} losers)</li>
            <li><strong>Average Market Move:</strong> {avg_change:.2f}% ({'up' if avg_change > 0 else 'down' if avg_change < 0 else 'flat'})</li>
            <li><strong>Foreign Flow:</strong> {'Net buying' if foreign_data['net_value'] > 0 else 'Net selling' if foreign_data['net_value'] < 0 else 'Balanced'} of {abs(foreign_data['net_value'])/1e9:.1f}B VND</li>
            <li><strong>Best Performing Sector:</strong> {sector_data[0][0] if sector_data else 'N/A'} ({sector_data[0][2]:.2f}% avg return)</li>
        </ul>
    </div>
    
    <div class="timestamp">
        Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ICT
    </div>
</body>
</html>
        """
        
        # Save report
        reports_dir = PROJECT_ROOT / "production" / "market_intelligence" / "reports"
        reports_dir.mkdir(exist_ok=True)
        
        report_filename = f"daily_alpha_pulse_{datetime.now().strftime('%Y%m%d')}.html"
        report_path = reports_dir / report_filename
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"✅ Daily Alpha Pulse generated successfully!")
        logger.info(f"📄 Report saved to: {report_path}")
        
        print(f"✅ Daily Alpha Pulse generated successfully!")
        print(f"📄 Report saved to: {report_path}")
        print(f"📊 Market Summary: {gainers} gainers, {losers} losers, {avg_change:.2f}% avg change")
        
        return True
        
    except Exception as e:
        logger.error(f"Error generating dashboard: {e}")
        print(f"❌ Error generating dashboard: {e}")
        return False

if __name__ == "__main__":
    generate_simple_daily_pulse()