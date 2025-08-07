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

print("✅ QVM Engine v3j Simple Backtest (v18b)")

# Configuration
CONFIG = {
    'strategy_name': 'QVM_Engine_v3j_Simple_v18b',
    'top_n_stocks': 20,
    'backtest_start_date': '2016-01-01',
    'backtest_end_date': '2025-12-31'
}

def main():
    print("🚀 Starting Simple Backtest")
    print("=" * 60)
    
    try:
        # Database connection
        db_manager = get_database_manager()
        engine = db_manager.get_engine()
        print("✅ Database connected")
        
        # Get available dates
        dates_query = f"""
        SELECT DISTINCT date
        FROM factor_scores_qvm
        WHERE date >= '{CONFIG['backtest_start_date']}'
        AND date <= '{CONFIG['backtest_end_date']}'
        AND strategy_version = 'qvm_v2.0_enhanced'
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
        
        # Process first 20 dates for testing
        test_dates = rebalance_dates[:20]
        portfolio_holdings = []
        
        for i, date in enumerate(test_dates):
            print(f"\n📅 Processing {date} ({i+1}/{len(test_dates)})")
            
            # Load factor scores
            factor_query = f"""
            SELECT 
                ticker,
                Quality_Composite,
                Value_Composite,
                Momentum_Composite,
                QVM_Composite
            FROM factor_scores_qvm
            WHERE date = '{date}'
            AND strategy_version = 'qvm_v2.0_enhanced'
            ORDER BY QVM_Composite DESC
            LIMIT {CONFIG['top_n_stocks']}
            """
            
            factors_df = pd.read_sql(factor_query, engine)
            
            if not factors_df.empty:
                # Apply ranking normalization
                for col in ['Quality_Composite', 'Value_Composite', 'Momentum_Composite']:
                    factors_df[f'{col}_Rank'] = factors_df[col].rank(ascending=True, method='min')
                    factors_df[f'{col}_Normalized'] = (factors_df[f'{col}_Rank'] - 1) / (len(factors_df) - 1)
                
                # Create fixed QVM composite
                factors_df['QVM_Composite_Fixed'] = (
                    factors_df['Quality_Composite_Normalized'] * 0.3 +
                    factors_df['Value_Composite_Normalized'] * 0.4 +
                    factors_df['Momentum_Composite_Normalized'] * 0.3
                )
                
                # Record holdings
                for _, stock in factors_df.iterrows():
                    portfolio_holdings.append({
                        'date': date,
                        'ticker': stock['ticker'],
                        'quality_score': stock['Quality_Composite_Normalized'],
                        'value_score': stock['Value_Composite_Normalized'],
                        'momentum_score': stock['Momentum_Composite_Normalized'],
                        'qvm_score': stock['QVM_Composite_Fixed']
                    })
                
                print(f"   ✅ Selected {len(factors_df)} stocks")
        
        # Convert to DataFrame
        holdings_df = pd.DataFrame(portfolio_holdings)
        
        print(f"\n✅ Backtest completed")
        print(f"📊 Total holdings: {len(holdings_df)}")
        
        # Generate tearsheet
        generate_tearsheet(holdings_df, CONFIG)
        
        # Save results
        results_dir = Path("insights")
        results_dir.mkdir(exist_ok=True)
        holdings_df.to_csv(results_dir / "18b_simple_holdings.csv", index=False)
        print(f"📁 Results saved to insights/18b_simple_holdings.csv")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def generate_tearsheet(holdings_df, config):
    """Generate comprehensive tearsheet analysis."""
    print("\n📊 COMPREHENSIVE TEARSHEET ANALYSIS")
    print("=" * 80)
    
    # Strategy Overview
    print("\n🎯 STRATEGY OVERVIEW")
    print("-" * 40)
    print(f"Strategy Name: {config['strategy_name']}")
    print(f"Backtest Period: {config['backtest_start_date']} to {config['backtest_end_date']}")
    print(f"Portfolio Size: {config['top_n_stocks']} stocks")
    print(f"Factor Weights: Quality 30%, Value 40%, Momentum 30%")
    
    # Portfolio Statistics
    print("\n📊 PORTFOLIO STATISTICS")
    print("-" * 40)
    print(f"Total Holdings: {len(holdings_df)}")
    print(f"Unique Dates: {holdings_df['date'].nunique()}")
    print(f"Unique Tickers: {holdings_df['ticker'].nunique()}")
    print(f"Average Holdings per Date: {len(holdings_df) / holdings_df['date'].nunique():.1f}")
    
    # Factor Score Analysis
    print("\n📈 FACTOR SCORE ANALYSIS")
    print("-" * 40)
    print(f"Quality Score Range: {holdings_df['quality_score'].min():.3f} to {holdings_df['quality_score'].max():.3f}")
    print(f"Value Score Range: {holdings_df['value_score'].min():.3f} to {holdings_df['value_score'].max():.3f}")
    print(f"Momentum Score Range: {holdings_df['momentum_score'].min():.3f} to {holdings_df['momentum_score'].max():.3f}")
    print(f"QVM Score Range: {holdings_df['qvm_score'].min():.3f} to {holdings_df['qvm_score'].max():.3f}")
    
    # Top Holdings
    print("\n🏆 TOP HOLDINGS (Most Frequently Selected)")
    print("-" * 40)
    top_stocks = holdings_df['ticker'].value_counts().head(15)
    for ticker, count in top_stocks.items():
        print(f"{ticker}: {count} periods")
    
    # Factor Score Statistics
    print("\n📊 FACTOR SCORE STATISTICS")
    print("-" * 40)
    print(f"Quality Score - Mean: {holdings_df['quality_score'].mean():.3f}, Std: {holdings_df['quality_score'].std():.3f}")
    print(f"Value Score - Mean: {holdings_df['value_score'].mean():.3f}, Std: {holdings_df['value_score'].std():.3f}")
    print(f"Momentum Score - Mean: {holdings_df['momentum_score'].mean():.3f}, Std: {holdings_df['momentum_score'].std():.3f}")
    print(f"QVM Score - Mean: {holdings_df['qvm_score'].mean():.3f}, Std: {holdings_df['qvm_score'].std():.3f}")
    
    # Date Range Analysis
    print("\n📅 DATE RANGE ANALYSIS")
    print("-" * 40)
    print(f"First Date: {holdings_df['date'].min()}")
    print(f"Last Date: {holdings_df['date'].max()}")
    print(f"Date Range: {(pd.to_datetime(holdings_df['date'].max()) - pd.to_datetime(holdings_df['date'].min())).days} days")
    
    # Sector Analysis (if available)
    print("\n🏢 SECTOR ANALYSIS")
    print("-" * 40)
    print("Note: Sector information not available in this simple version")
    print("Full sector analysis would require additional data sources")
    
    # Performance Metrics
    print("\n📈 PERFORMANCE METRICS")
    print("-" * 40)
    print("Note: This is a holdings analysis only")
    print("Full performance metrics would require price data and return calculations")
    print("Next step: Integrate with price data for complete performance analysis")
    
    print("\n✅ Tearsheet analysis completed")

if __name__ == "__main__":
    main()
