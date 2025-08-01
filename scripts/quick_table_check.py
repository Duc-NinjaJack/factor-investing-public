#!/usr/bin/env python3
"""
Quick table check for equity_history vs vcsc_daily_data_complete
"""

import pandas as pd
import mysql.connector
import yaml
from pathlib import Path

# Database configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
config_path = PROJECT_ROOT / 'config' / 'database.yml'

def get_db_config():
    with open(config_path, 'r') as f:
        db_yaml = yaml.safe_load(f)
    return db_yaml.get('production', db_yaml.get('development'))

def create_connection():
    db_config = get_db_config()
    return mysql.connector.connect(
        host=db_config['host'],
        database=db_config['schema_name'],
        user=db_config['username'],
        password=db_config['password']
    )

def main():
    print("🔍 QUICK TABLE ANALYSIS")
    print("=" * 50)
    
    conn = create_connection()
    
    # Equity History basic stats
    print("📈 EQUITY HISTORY:")
    equity_query = """
    SELECT 
        MIN(date) as earliest,
        MAX(date) as latest,
        COUNT(DISTINCT ticker) as tickers,
        COUNT(*) as records
    FROM equity_history
    """
    
    equity_stats = pd.read_sql(equity_query, conn)
    print(f"• Date Range: {equity_stats['earliest'].iloc[0]} to {equity_stats['latest'].iloc[0]}")
    print(f"• Tickers: {equity_stats['tickers'].iloc[0]:,}")
    print(f"• Records: {equity_stats['records'].iloc[0]:,}")
    
    # VCSC basic stats
    print("\n📊 VCSC DAILY DATA COMPLETE:")
    vcsc_query = """
    SELECT 
        MIN(trading_date) as earliest,
        MAX(trading_date) as latest,
        COUNT(DISTINCT ticker) as tickers,
        COUNT(*) as records
    FROM vcsc_daily_data_complete
    """
    
    vcsc_stats = pd.read_sql(vcsc_query, conn)
    print(f"• Date Range: {vcsc_stats['earliest'].iloc[0]} to {vcsc_stats['latest'].iloc[0]}")
    print(f"• Tickers: {vcsc_stats['tickers'].iloc[0]:,}")
    print(f"• Records: {vcsc_stats['records'].iloc[0]:,}")
    
    # Sample price comparison for one recent date
    print("\n🔍 SAMPLE PRICE COMPARISON (Recent Date):")
    sample_query = """
    SELECT 
        e.ticker,
        e.close as equity_adjusted_close,
        v.close_price_adjusted as vcsc_adjusted_close,
        ABS(e.close - v.close_price_adjusted) as diff,
        ABS(e.close - v.close_price_adjusted) / e.close * 100 as diff_pct
    FROM equity_history e
    INNER JOIN vcsc_daily_data_complete v 
        ON e.ticker COLLATE utf8mb4_unicode_ci = v.ticker COLLATE utf8mb4_unicode_ci 
        AND e.date = v.trading_date
    WHERE e.date = '2025-07-30'
        AND e.close > 0 
        AND v.close_price_adjusted > 0
    ORDER BY diff_pct DESC
    LIMIT 10
    """
    
    sample_comparison = pd.read_sql(sample_query, conn)
    if len(sample_comparison) > 0:
        print(f"Found {len(sample_comparison)} matching records on 2025-07-30")
        for _, row in sample_comparison.head(5).iterrows():
            print(f"• {row['ticker']}: {row['diff_pct']:.4f}% difference")
        
        avg_diff = sample_comparison['diff_pct'].mean()
        max_diff = sample_comparison['diff_pct'].max()
        print(f"\n• Average difference: {avg_diff:.4f}%")
        print(f"• Maximum difference: {max_diff:.4f}%")
        
        if avg_diff < 0.01:
            print("✅ GOOD: Prices reconcile well")
        else:
            print("⚠️  WARNING: Significant price differences detected")
    else:
        print("❌ No matching records found for 2025-07-30")
    
    conn.close()
    print("\n✅ Quick analysis complete")

if __name__ == "__main__":
    main()