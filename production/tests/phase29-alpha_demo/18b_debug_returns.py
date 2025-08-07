#!/usr/bin/env python3

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add Project Root to Python Path
current_path = Path.cwd()
while not (current_path / 'production').is_dir():
    current_path = current_path.parent
project_root = current_path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from production.database.connection import get_database_manager
from sqlalchemy import text

print("✅ QVM Engine v3j Debug Returns (v18b)")

def main():
    print("🔍 Debugging returns calculation...")
    print("=" * 60)
    
    try:
        # Database connection
        db_manager = get_database_manager()
        engine = db_manager.get_engine()
        print("✅ Database connected")
        
        # Load holdings data
        holdings_file = Path("insights/18b_complete_holdings.csv")
        if holdings_file.exists():
            holdings_df = pd.read_csv(holdings_file)
            print(f"✅ Loaded holdings: {len(holdings_df)} records")
        else:
            print("❌ Holdings file not found")
            return
        
        # Check holdings data
        print(f"\n📊 Holdings Analysis:")
        print(f"   Date range: {holdings_df['date'].min()} to {holdings_df['date'].max()}")
        print(f"   Unique dates: {holdings_df['date'].nunique()}")
        print(f"   Sample dates: {holdings_df['date'].head(5).tolist()}")
        
        # Load price data for a sample date
        sample_date = holdings_df['date'].iloc[0]
        sample_tickers = holdings_df[holdings_df['date'] == sample_date]['ticker'].unique()
        ticker_list = "', '".join(sample_tickers[:5])  # First 5 tickers
        
        print(f"\n🔍 Checking price data for {sample_date}:")
        print(f"   Sample tickers: {sample_tickers[:5].tolist()}")
        
        price_query = f"""
        SELECT 
            trading_date as date,
            ticker,
            close_price
        FROM vcsc_daily_data_complete
        WHERE ticker IN ('{ticker_list}')
        AND trading_date = '{sample_date}'
        ORDER BY ticker
        """
        
        price_data = pd.read_sql(price_query, engine)
        print(f"   Price data found: {len(price_data)} records")
        
        if not price_data.empty:
            print(f"   Sample price data:")
            print(price_data.head())
        else:
            print(f"   ❌ No price data found for {sample_date}")
            
            # Check what dates are available for these tickers
            check_query = f"""
            SELECT 
                MIN(trading_date) as min_date,
                MAX(trading_date) as max_date,
                COUNT(DISTINCT trading_date) as unique_dates
            FROM vcsc_daily_data_complete
            WHERE ticker IN ('{ticker_list}')
            """
            
            check_data = pd.read_sql(check_query, engine)
            print(f"   Available date range: {check_data['min_date'].iloc[0]} to {check_data['max_date'].iloc[0]}")
            print(f"   Available dates: {check_data['unique_dates'].iloc[0]}")
        
        # Check benchmark data
        print(f"\n🔍 Checking benchmark data:")
        benchmark_query = f"""
        SELECT 
            MIN(date) as min_date,
            MAX(date) as max_date,
            COUNT(*) as records
        FROM etf_history
        WHERE ticker = 'VNINDEX'
        """
        
        benchmark_data = pd.read_sql(benchmark_query, engine)
        print(f"   Benchmark date range: {benchmark_data['min_date'].iloc[0]} to {benchmark_data['max_date'].iloc[0]}")
        print(f"   Benchmark records: {benchmark_data['records'].iloc[0]}")
        
        # Check if sample date exists in benchmark
        benchmark_check_query = f"""
        SELECT COUNT(*) as count
        FROM etf_history
        WHERE ticker = 'VNINDEX'
        AND date = '{sample_date}'
        """
        
        benchmark_check = pd.read_sql(benchmark_check_query, engine)
        print(f"   Benchmark data for {sample_date}: {benchmark_check['count'].iloc[0]} records")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
