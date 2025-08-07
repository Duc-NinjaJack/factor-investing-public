#!/usr/bin/env python3

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add Project Root to Python Path
current_path = Path.cwd()
while not (current_path / 'production').is_dir():
    current_path = current_path.parent
project_root = current_path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from production.database.connection import DatabaseManager

print("✅ Holdings File Generator for Tearsheet Demonstrations")

# Configuration
CONFIG = {
    'strategy_name': 'QVM_Engine_v3j_Holdings_Generator',
    'backtest_start_date': '2016-01-01',
    'backtest_end_date': '2025-12-31',
    'strategy_version': 'qvm_v2.0_enhanced'  # Use the same version as 18b
}

def generate_holdings_file():
    """Generate holdings file compatible with tearsheet demonstrations."""
    print("🚀 Generating holdings file...")
    print("=" * 80)
    
    try:
        # Database connection
        db_manager = DatabaseManager()
        engine = db_manager.get_engine()
        print("✅ Database connected")
        
        # Get available dates
        dates_query = f"""
        SELECT DISTINCT date
        FROM factor_scores_qvm
        WHERE date >= '{CONFIG['backtest_start_date']}'
        AND date <= '{CONFIG['backtest_end_date']}'
        AND strategy_version = '{CONFIG['strategy_version']}'
        ORDER BY date
        """
        
        dates_df = pd.read_sql(dates_query, engine)
        print(f"📅 Available dates: {len(dates_df)}")
        
        # Monthly rebalancing
        rebalance_dates = []
        current_month = None
        
        for date in dates_df['date']:
            date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
            month = date_str[:7]
            
            if month != current_month:
                rebalance_dates.append(date_str)
                current_month = month
        
        print(f"📊 Rebalancing dates: {len(rebalance_dates)}")
        
        # Process all dates to generate holdings
        portfolio_holdings = []
        
        print(f"🔄 Processing {len(rebalance_dates)} rebalancing periods...")
        
        for i, date in enumerate(rebalance_dates):
            if i % 20 == 0:
                print(f"   Progress: {i}/{len(rebalance_dates)} periods")
            
            # Load factor scores for this date
            factor_query = f"""
            SELECT 
                ticker,
                Quality_Composite,
                Value_Composite,
                Momentum_Composite,
                QVM_Composite
            FROM factor_scores_qvm
            WHERE date = '{date}'
            AND strategy_version = '{CONFIG['strategy_version']}'
            ORDER BY QVM_Composite DESC
            """
            
            factors_df = pd.read_sql(factor_query, engine)
            
            if not factors_df.empty:
                # Record all holdings for this date (not just top 20, let the tearsheet filter)
                for _, stock in factors_df.iterrows():
                    portfolio_holdings.append({
                        'date': date,
                        'ticker': stock['ticker'],
                        'quality_score': stock['Quality_Composite'],
                        'value_score': stock['Value_Composite'],
                        'momentum_score': stock['Momentum_Composite'],
                        'composite_score': stock['QVM_Composite']
                    })
        
        # Convert to DataFrame
        holdings_df = pd.DataFrame(portfolio_holdings)
        print(f"✅ Portfolio holdings: {len(holdings_df)} records")
        
        # Create docs directory if it doesn't exist
        docs_dir = Path("docs")
        docs_dir.mkdir(exist_ok=True)
        
        # Save holdings file
        holdings_file = docs_dir / "18b_complete_holdings.csv"
        holdings_df.to_csv(holdings_file, index=False)
        
        print(f"✅ Holdings file saved: {holdings_file}")
        print(f"📊 File size: {holdings_file.stat().st_size / 1024 / 1024:.2f} MB")
        
        # Show sample data
        print("\n📋 Sample holdings data:")
        print(holdings_df.head(10))
        
        # Show statistics
        print(f"\n📊 Holdings statistics:")
        print(f"   Total records: {len(holdings_df)}")
        print(f"   Unique dates: {holdings_df['date'].nunique()}")
        print(f"   Unique tickers: {holdings_df['ticker'].nunique()}")
        print(f"   Date range: {holdings_df['date'].min()} to {holdings_df['date'].max()}")
        
        # Show factor score statistics
        print(f"\n📊 Factor score statistics:")
        for col in ['quality_score', 'value_score', 'momentum_score', 'composite_score']:
            print(f"   {col}: mean={holdings_df[col].mean():.3f}, std={holdings_df[col].std():.3f}")
        
        return holdings_df
        
    except Exception as e:
        print(f"❌ Error generating holdings file: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    holdings_df = generate_holdings_file()
    if holdings_df is not None:
        print("\n🎉 Holdings file generation completed successfully!")
    else:
        print("\n💥 Holdings file generation failed!")

