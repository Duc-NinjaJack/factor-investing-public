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

print("✅ QVM Engine v3j Step-by-Step Debug (v18b)")

def main():
    print("🔍 Step-by-step debugging...")
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
        
        # Step 1: Check holdings data
        print(f"\n📊 Step 1: Holdings Analysis")
        print(f"   Date range: {holdings_df['date'].min()} to {holdings_df['date'].max()}")
        print(f"   Unique dates: {holdings_df['date'].nunique()}")
        print(f"   Unique tickers: {holdings_df['ticker'].nunique()}")
        
        # Step 2: Check a specific date
        sample_date = holdings_df['date'].iloc[0]
        print(f"\n📊 Step 2: Sample Date Analysis - {sample_date}")
        
        date_holdings = holdings_df[holdings_df['date'] == sample_date]
        print(f"   Holdings for {sample_date}: {len(date_holdings)} records")
        print(f"   Sample tickers: {date_holdings['ticker'].head(5).tolist()}")
        
        # Step 3: Check price data for this date
        print(f"\n📊 Step 3: Price Data Check")
        sample_tickers = date_holdings['ticker'].head(5).tolist()
        ticker_list = "', '".join(sample_tickers)
        
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
        print(f"   Price data for {sample_date}: {len(price_data)} records")
        
        if not price_data.empty:
            print(f"   Sample price data:")
            print(price_data.head())
            
            # Step 4: Calculate portfolio value for this date
            print(f"\n📊 Step 4: Portfolio Value Calculation")
            current_capital = 1000000
            portfolio_value = 0
            valid_holdings = 0
            
            for _, holding in date_holdings.iterrows():
                ticker = holding['ticker']
                ticker_price = price_data[price_data['ticker'] == ticker]
                
                if not ticker_price.empty:
                    price = ticker_price['close_price'].iloc[0]
                    position_size = current_capital / len(date_holdings)
                    shares = position_size / price
                    portfolio_value += shares * price
                    valid_holdings += 1
                    print(f"     {ticker}: Price={price:,.0f}, Shares={shares:.2f}, Value={shares*price:,.0f}")
                else:
                    print(f"     {ticker}: No price data")
            
            print(f"   Portfolio value: {portfolio_value:,.0f} VND")
            print(f"   Valid holdings: {valid_holdings}/{len(date_holdings)}")
            
            if portfolio_value > 0:
                print(f"   ✅ Portfolio value calculated successfully")
            else:
                print(f"   ❌ Portfolio value is zero")
        else:
            print(f"   ❌ No price data found for {sample_date}")
            
            # Check what dates are available
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
        
        # Step 5: Check if the issue is with date format
        print(f"\n📊 Step 5: Date Format Check")
        print(f"   Holdings date type: {type(holdings_df['date'].iloc[0])}")
        print(f"   Holdings date sample: '{holdings_df['date'].iloc[0]}'")
        
        # Check price data date format
        price_check_query = f"""
        SELECT 
            trading_date,
            ticker,
            close_price
        FROM vcsc_daily_data_complete
        WHERE ticker IN ('{sample_tickers[0]}')
        ORDER BY trading_date
        LIMIT 5
        """
        
        price_check = pd.read_sql(price_check_query, engine)
        print(f"   Price date type: {type(price_check['trading_date'].iloc[0])}")
        print(f"   Price date sample: '{price_check['trading_date'].iloc[0]}'")
        
        # Step 6: Try with date conversion
        print(f"\n📊 Step 6: Date Conversion Test")
        holdings_df['date'] = pd.to_datetime(holdings_df['date'])
        price_check['trading_date'] = pd.to_datetime(price_check['trading_date'])
        
        print(f"   Holdings date after conversion: {holdings_df['date'].iloc[0]}")
        print(f"   Price date after conversion: {price_check['trading_date'].iloc[0]}")
        
        # Test matching
        test_date = holdings_df['date'].iloc[0]
        test_price = price_check[price_check['trading_date'] == test_date]
        print(f"   Matching test: {len(test_price)} records found")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
