#!/usr/bin/env python3
"""
Check benchmark data availability and performance
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

# Database connection
engine = create_engine('mysql+pymysql://root:password@localhost/alphabeta')

# Check available benchmark-like tickers
print("🔍 Checking available benchmark tickers...")
with engine.connect() as conn:
    result = conn.execute(text("""
        SELECT DISTINCT ticker 
        FROM vcsc_daily_data 
        WHERE ticker LIKE '%VN%' OR ticker LIKE '%INDEX%' OR ticker LIKE '%VNINDEX%'
        ORDER BY ticker
    """))
    benchmark_tickers = [row[0] for row in result]

print(f"Available benchmark-like tickers: {benchmark_tickers}")

# Check VNM specifically
print("\n🔍 Checking VNM benchmark data...")
with engine.connect() as conn:
    result = conn.execute(text("""
        SELECT 
            trading_date as date,
            close_price as close
        FROM vcsc_daily_data 
        WHERE ticker = 'VNM'
        AND trading_date >= '2016-01-01' AND trading_date <= '2025-07-28'
        ORDER BY trading_date
    """))
    vnm_data = pd.read_sql(result, conn)

print(f"VNM data shape: {vnm_data.shape}")
print(f"VNM date range: {vnm_data['date'].min()} to {vnm_data['date'].max()}")
print(f"VNM price range: {vnm_data['close'].min():.2f} to {vnm_data['close'].max():.2f}")

# Calculate VNM returns
vnm_data['returns'] = vnm_data['close'].pct_change()
vnm_returns = vnm_data.set_index('date')['returns'].dropna()

total_return = (1 + vnm_returns).prod() - 1
annualized_return = (1 + total_return) ** (252 / len(vnm_returns)) - 1
volatility = vnm_returns.std() * np.sqrt(252)

print(f"\n📊 VNM Performance (2016-2025):")
print(f"   Total Return: {total_return:.2%}")
print(f"   Annualized Return: {annualized_return:.2%}")
print(f"   Volatility: {volatility:.2%}")

# Check if there are other better benchmarks
print("\n🔍 Checking other potential benchmarks...")
for ticker in ['VNINDEX', 'VNM', 'VIC', 'HPG']:
    try:
        with engine.connect() as conn:
            result = conn.execute(text(f"""
                SELECT 
                    trading_date as date,
                    close_price as close
                FROM vcsc_daily_data 
                WHERE ticker = '{ticker}'
                AND trading_date >= '2016-01-01' AND trading_date <= '2025-07-28'
                ORDER BY trading_date
            """))
            data = pd.read_sql(result, conn)
            
            if not data.empty:
                data['returns'] = data['close'].pct_change()
                returns = data.set_index('date')['returns'].dropna()
                
                total_ret = (1 + returns).prod() - 1
                ann_ret = (1 + total_ret) ** (252 / len(returns)) - 1
                vol = returns.std() * np.sqrt(252)
                
                print(f"   {ticker}: Total={total_ret:.2%}, Annual={ann_ret:.2%}, Vol={vol:.2%}")
            else:
                print(f"   {ticker}: No data")
    except Exception as e:
        print(f"   {ticker}: Error - {e}") 