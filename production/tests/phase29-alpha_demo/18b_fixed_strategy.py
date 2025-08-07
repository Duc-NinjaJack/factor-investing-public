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
    if current_path.parent == current_path:
        raise FileNotFoundError("Could not find the 'production' directory.")
    current_path = current_path.parent

project_root = current_path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from production.database.connection import get_database_manager
from sqlalchemy import text

print("✅ QVM Engine v3j Fixed Strategy (v18b) initialized")

# Configuration
QVM_CONFIG = {
    'strategy_name': 'QVM_Engine_v3j_Fixed_v18b',
    'universe': {
        'lookback_days': 252,
        'top_n_stocks': 20,
        'target_portfolio_size': 20,
        'backtest_start_date': '2016-01-01',
        'backtest_end_date': '2025-12-31',
        'rebalance_frequency': 'monthly',
        'transaction_cost_bps': 10,
        'adtv_threshold_bn': 10.0
    },
    'factors': {
        'quality_weight': 0.3,
        'value_weight': 0.4,
        'momentum_weight': 0.3
    },
    'regime_weights': {
        'normal': {'quality_weight': 0.3, 'value_weight': 0.4, 'momentum_weight': 0.3, 'allocation_multiplier': 1.0},
        'stress': {'quality_weight': 0.5, 'value_weight': 0.4, 'momentum_weight': 0.1, 'allocation_multiplier': 0.6},
        'bull': {'quality_weight': 0.1, 'value_weight': 0.3, 'momentum_weight': 0.6, 'allocation_multiplier': 1.0}
    }
}

def create_db_connection():
    """Create database connection."""
    try:
        db_manager = get_database_manager()
        engine = db_manager.get_engine()
        print("✅ Database connection established")
        return engine
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        raise

def load_existing_factor_scores(db_engine, date):
    """Load existing factor scores from factor_scores_qvm table."""
    print(f"📊 Loading factor scores for {date}")
    
    try:
        # Load existing factor scores
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
        ORDER BY ticker
        """
        
        factor_data = pd.read_sql(factor_query, db_engine)
        print(f"   ✅ Loaded {len(factor_data)} factor scores")
        
        if factor_data.empty:
            return pd.DataFrame()
        
        # Apply ranking-based normalization
        for col in ['Quality_Composite', 'Value_Composite', 'Momentum_Composite']:
            if col in factor_data.columns:
                factor_data[f'{col}_Rank'] = factor_data[col].rank(ascending=True, method='min')
                factor_data[f'{col}_Normalized'] = (factor_data[f'{col}_Rank'] - 1) / (len(factor_data) - 1)
        
        # Create new QVM composite
        factor_data['QVM_Composite_Fixed'] = (
            factor_data['Quality_Composite_Normalized'] * QVM_CONFIG['factors']['quality_weight'] +
            factor_data['Value_Composite_Normalized'] * QVM_CONFIG['factors']['value_weight'] +
            factor_data['Momentum_Composite_Normalized'] * QVM_CONFIG['factors']['momentum_weight']
        )
        
        return factor_data
        
    except Exception as e:
        print(f"   ❌ Error loading factor scores: {e}")
        return pd.DataFrame()

def detect_market_regime(db_engine, date):
    """Detect market regime based on volatility and returns."""
    print(f"🌍 Detecting market regime for {date}")
    
    try:
        # Get market data for regime detection
        regime_query = f"""
        SELECT 
            trading_date,
            close_price
        FROM vcsc_daily_data_complete
        WHERE ticker = 'VNINDEX'
        AND trading_date >= DATE_SUB('{date}', INTERVAL 30 DAY)
        AND trading_date <= '{date}'
        ORDER BY trading_date
        """
        
        market_data = pd.read_sql(regime_query, db_engine)
        
        if len(market_data) < 20:
            return 'normal'
        
        # Calculate daily returns
        market_data['daily_return'] = market_data['close_price'].pct_change()
        market_data = market_data.dropna()
        
        if len(market_data) < 10:
            return 'normal'
        
        # Calculate volatility and return
        volatility = market_data['daily_return'].std() * np.sqrt(252)
        avg_return = market_data['daily_return'].mean() * 252
        
        # Simple regime detection
        if volatility > 0.25:  # High volatility
            regime = 'stress'
        elif avg_return > 0.20:  # High returns
            regime = 'bull'
        else:
            regime = 'normal'
        
        print(f"   📊 Regime: {regime.upper()}")
        return regime
        
    except Exception as e:
        print(f"   ❌ Error detecting regime: {e}")
        return 'normal'

def run_fixed_backtest(config, db_engine):
    """Run fixed backtest using existing factor scores."""
    print("🚀 Starting Fixed Strategy Backtest")
    print("=" * 80)
    
    try:
        # Get available dates from factor_scores_qvm
        dates_query = f"""
        SELECT DISTINCT date
        FROM factor_scores_qvm
        WHERE date >= '{config['universe']['backtest_start_date']}'
        AND date <= '{config['universe']['backtest_end_date']}'
        AND strategy_version = 'qvm_v2.0_enhanced'
        ORDER BY date
        """
        
        available_dates = pd.read_sql(dates_query, db_engine)['date'].tolist()
        print(f"📅 Available dates: {len(available_dates)}")
        
        # Monthly rebalancing
        rebalance_dates = []
        current_month = None
        
        for date in available_dates:
            date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
            month = date_str[:7]  # YYYY-MM
            
            if month != current_month:
                rebalance_dates.append(date_str)
                current_month = month
        
        print(f"📊 Rebalancing dates: {len(rebalance_dates)}")
        
        # Initialize results
        portfolio_holdings = []
        regime_history = []
        
        # Process each rebalancing date
        for i, date in enumerate(rebalance_dates[:50]):  # Limit to first 50 for testing
            print(f"\n📅 Processing {date} ({i+1}/{min(50, len(rebalance_dates))})")
            
            try:
                # Load factor scores
                factors_data = load_existing_factor_scores(db_engine, date)
                if factors_data.empty:
                    continue
                
                # Detect market regime
                regime = detect_market_regime(db_engine, date)
                regime_history.append({'date': date, 'regime': regime})
                
                # Select top stocks
                top_stocks = factors_data.nlargest(config['universe']['top_n_stocks'], 'QVM_Composite_Fixed')
                
                # Record holdings
                for _, stock in top_stocks.iterrows():
                    portfolio_holdings.append({
                        'date': date,
                        'ticker': stock['ticker'],
                        'quality_score': stock['Quality_Composite_Normalized'],
                        'value_score': stock['Value_Composite_Normalized'],
                        'momentum_score': stock['Momentum_Composite_Normalized'],
                        'qvm_score': stock['QVM_Composite_Fixed'],
                        'regime': regime
                    })
                
                print(f"   ✅ Selected {len(top_stocks)} stocks for {regime} regime")
                
            except Exception as e:
                print(f"   ❌ Error processing {date}: {e}")
                continue
        
        # Convert to DataFrame
        holdings_df = pd.DataFrame(portfolio_holdings)
        regime_df = pd.DataFrame(regime_history)
        
        print(f"\n✅ Backtest completed")
        print(f"📊 Total holdings: {len(holdings_df)}")
        
        return holdings_df, regime_df
        
    except Exception as e:
        print(f"❌ Error running backtest: {e}")
        return pd.DataFrame(), pd.DataFrame()

def generate_tearsheet(holdings_df, regime_df, config):
    """Generate comprehensive tearsheet analysis."""
    print("\n📊 GENERATING COMPREHENSIVE TEARSHEET")
    print("=" * 80)
    
    # Strategy Overview
    print("\n🎯 STRATEGY OVERVIEW")
    print("-" * 40)
    print(f"Strategy Name: {config['strategy_name']}")
    print(f"Backtest Period: {config['universe']['backtest_start_date']} to {config['universe']['backtest_end_date']}")
    print(f"Portfolio Size: {config['universe']['top_n_stocks']} stocks")
    print(f"Rebalancing: {config['universe']['rebalance_frequency']}")
    
    # Factor Configuration
    print("\n🔧 FACTOR CONFIGURATION")
    print("-" * 40)
    print(f"Quality Weight: {config['factors']['quality_weight']:.1%}")
    print(f"Value Weight: {config['factors']['value_weight']:.1%}")
    print(f"Momentum Weight: {config['factors']['momentum_weight']:.1%}")
    
    # Regime Analysis
    print("\n🌍 REGIME ANALYSIS")
    print("-" * 40)
    regime_counts = regime_df['regime'].value_counts()
    total_regimes = len(regime_df)
    for regime, count in regime_counts.items():
        percentage = count / total_regimes * 100
        print(f"{regime.upper()}: {count} periods ({percentage:.1f}%)")
    
    # Portfolio Statistics
    print("\n📊 PORTFOLIO STATISTICS")
    print("-" * 40)
    print(f"Total Rebalancing Periods: {len(regime_df)}")
    print(f"Total Holdings: {len(holdings_df)}")
    print(f"Average Holdings per Period: {len(holdings_df) / len(regime_df):.1f}")
    
    # Factor Score Analysis
    if not holdings_df.empty:
        print("\n📈 FACTOR SCORE ANALYSIS")
        print("-" * 40)
        print(f"Quality Score Range: {holdings_df['quality_score'].min():.3f} to {holdings_df['quality_score'].max():.3f}")
        print(f"Value Score Range: {holdings_df['value_score'].min():.3f} to {holdings_df['value_score'].max():.3f}")
        print(f"Momentum Score Range: {holdings_df['momentum_score'].min():.3f} to {holdings_df['momentum_score'].max():.3f}")
        print(f"QVM Score Range: {holdings_df['qvm_score'].min():.3f} to {holdings_df['qvm_score'].max():.3f}")
    
    # Top Holdings
    if not holdings_df.empty:
        print("\n🏆 TOP HOLDINGS")
        print("-" * 40)
        top_stocks = holdings_df['ticker'].value_counts().head(10)
        for ticker, count in top_stocks.items():
            print(f"{ticker}: {count} periods")
    
    print("\n✅ Tearsheet analysis completed")

def main():
    """Main execution function."""
    print("🎯 QVM Engine v3j Fixed Strategy (v18b)")
    print("=" * 80)
    
    try:
        # Create database connection
        db_engine = create_db_connection()
        
        # Run fixed backtest
        holdings_df, regime_df = run_fixed_backtest(QVM_CONFIG, db_engine)
        
        if not holdings_df.empty:
            # Save results
            results_dir = Path("insights")
            results_dir.mkdir(exist_ok=True)
            
            holdings_df.to_csv(results_dir / "18b_fixed_holdings.csv", index=False)
            regime_df.to_csv(results_dir / "18b_fixed_regime_history.csv", index=False)
            
            print(f"\n✅ Results saved to insights/")
            
            # Generate tearsheet
            generate_tearsheet(holdings_df, regime_df, QVM_CONFIG)
        
    except Exception as e:
        print(f"❌ Strategy execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
