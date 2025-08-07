#!/usr/bin/env python3

# %% [markdown]
# # QVM Engine v3j - Available Metrics Strategy (Version 18b) - FIXED
# 
# This version uses existing factor scores from the database instead of calculating from scratch.
# 
# **Key Fix**: Use existing `factor_scores_qvm` table with `qvm_v2.0_enhanced` strategy version
# **Data Coverage**: 2016-2025 with 728 tickers, 7.3M+ fundamental records
# **Performance**: Much faster execution using pre-calculated factors

# %%
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
from scipy.stats import norm

print("✅ QVM Engine v3j Available Metrics Strategy (v18b) - FIXED initialized")

# %%
# Configuration
QVM_CONFIG = {
    'strategy_name': 'QVM_Engine_v3j_Available_Metrics_v18b_Fixed',
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
        'quality_weight': 0.3,  # Reduced since no F-Score
        'value_weight': 0.4,    # Increased focus on value
        'momentum_weight': 0.3,
        'quality_factors': {
            'roaa_weight': 1.0   # Only ROAA available
        },
        'value_factors': {
            'pe_weight': 0.5,
            'fcf_yield_weight': 0.5
        },
        'momentum_factors': {
            'momentum_weight': 0.5,
            'low_vol_weight': 0.5
        }
    },
    'regime_detection': {
        'lookback_days': 30,
        'vol_threshold_pct': 75,
        'return_threshold_pct': 25,
        'bull_return_threshold_pct': 75,
        'min_regime_duration': 5
    },
    'regime_weights': {
        'normal': {
            'quality_weight': 0.3,
            'value_weight': 0.4,
            'momentum_weight': 0.3,
            'allocation_multiplier': 1.0
        },
        'stress': {
            'quality_weight': 0.5,
            'value_weight': 0.4,
            'momentum_weight': 0.1,
            'allocation_multiplier': 0.6
        },
        'bull': {
            'quality_weight': 0.1,
            'value_weight': 0.3,
            'momentum_weight': 0.6,
            'allocation_multiplier': 1.0
        }
    }
}

# %%
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

def calculate_universe_rankings(db_engine, date, config):
    """Calculate universe rankings with ADTV filter."""
    print(f"📊 Calculating universe rankings for {date}")
    
    try:
        # Get ADTV data for liquidity filtering
        adtv_query = f"""
        SELECT 
            ticker,
            AVG(total_volume * close_price) / 1e9 as avg_daily_trading_value_bn
        FROM vcsc_daily_data_complete
        WHERE trading_date >= DATE_SUB('{date}', INTERVAL {config['universe']['lookback_days']} DAY)
        AND trading_date <= '{date}'
        AND total_volume > 0
        AND close_price > 0
        GROUP BY ticker
        HAVING avg_daily_trading_value_bn >= {config['universe']['adtv_threshold_bn']}
        ORDER BY avg_daily_trading_value_bn DESC
        """
        
        adtv_data = pd.read_sql(adtv_query, db_engine)
        
        if adtv_data.empty:
            print(f"   ⚠️ No stocks meet ADTV threshold")
            return []
        
        print(f"   ✅ Found {len(adtv_data)} stocks meeting ADTV threshold")
        return adtv_data['ticker'].tolist()
        
    except Exception as e:
        print(f"   ❌ Error calculating universe rankings: {e}")
        return []

def load_existing_factor_scores(db_engine, date, tickers):
    """Load existing factor scores from factor_scores_qvm table."""
    print(f"📊 Loading existing factor scores for {date}")
    
    ticker_list = "', '".join(tickers)
    
    try:
        # Load existing factor scores from qvm_v2.0_enhanced
        factor_query = f"""
        SELECT 
            ticker,
            Quality_Composite,
            Value_Composite,
            Momentum_Composite,
            QVM_Composite,
            calculation_timestamp
        FROM factor_scores_qvm
        WHERE date = '{date}'
        AND strategy_version = 'qvm_v2.0_enhanced'
        AND ticker IN ('{ticker_list}')
        ORDER BY ticker
        """
        
        factor_data = pd.read_sql(factor_query, db_engine)
        print(f"   ✅ Loaded factor scores for {len(factor_data)} stocks")
        
        if factor_data.empty:
            print(f"   ⚠️ No factor scores available for {date}")
            return pd.DataFrame()
        
        # Apply ranking-based normalization to fix the issues we identified
        print(f"   🔧 Applying ranking-based normalization...")
        
        # Quality factor normalization (0-1 scale)
        if 'Quality_Composite' in factor_data.columns:
            factor_data['Quality_Composite_Rank'] = factor_data['Quality_Composite'].rank(ascending=True, method='min')
            factor_data['Quality_Composite_Normalized'] = (factor_data['Quality_Composite_Rank'] - 1) / (len(factor_data) - 1)
        else:
            factor_data['Quality_Composite_Normalized'] = 0.5
        
        # Value factor normalization (0-1 scale)
        if 'Value_Composite' in factor_data.columns:
            factor_data['Value_Composite_Rank'] = factor_data['Value_Composite'].rank(ascending=True, method='min')
            factor_data['Value_Composite_Normalized'] = (factor_data['Value_Composite_Rank'] - 1) / (len(factor_data) - 1)
        else:
            factor_data['Value_Composite_Normalized'] = 0.5
        
        # Momentum factor normalization (0-1 scale)
        if 'Momentum_Composite' in factor_data.columns:
            factor_data['Momentum_Composite_Rank'] = factor_data['Momentum_Composite'].rank(ascending=True, method='min')
            factor_data['Momentum_Composite_Normalized'] = (factor_data['Momentum_Composite_Rank'] - 1) / (len(factor_data) - 1)
        else:
            factor_data['Momentum_Composite_Normalized'] = 0.5
        
        # Create new QVM composite with proper weights
        factor_data['QVM_Composite_Fixed'] = (
            factor_data['Quality_Composite_Normalized'] * QVM_CONFIG['factors']['quality_weight'] +
            factor_data['Value_Composite_Normalized'] * QVM_CONFIG['factors']['value_weight'] +
            factor_data['Momentum_Composite_Normalized'] * QVM_CONFIG['factors']['momentum_weight']
        )
        
        print(f"   ✅ Normalized factor scores created")
        print(f"   📊 Factor ranges:")
        print(f"      Quality: {factor_data['Quality_Composite_Normalized'].min():.3f} to {factor_data['Quality_Composite_Normalized'].max():.3f}")
        print(f"      Value: {factor_data['Value_Composite_Normalized'].min():.3f} to {factor_data['Value_Composite_Normalized'].max():.3f}")
        print(f"      Momentum: {factor_data['Momentum_Composite_Normalized'].min():.3f} to {factor_data['Momentum_Composite_Normalized'].max():.3f}")
        print(f"      QVM Fixed: {factor_data['QVM_Composite_Fixed'].min():.3f} to {factor_data['QVM_Composite_Fixed'].max():.3f}")
        
        return factor_data
        
    except Exception as e:
        print(f"   ❌ Error loading factor scores: {e}")
        return pd.DataFrame()

def detect_market_regime(db_engine, date, config):
    """Detect market regime based on volatility and returns."""
    print(f"🌍 Detecting market regime for {date}")
    
    try:
        # Get market data for regime detection
        regime_query = f"""
        SELECT 
            trading_date,
            close_price,
            total_volume
        FROM vcsc_daily_data_complete
        WHERE ticker = 'VNINDEX'
        AND trading_date >= DATE_SUB('{date}', INTERVAL {config['regime_detection']['lookback_days']} DAY)
        AND trading_date <= '{date}'
        ORDER BY trading_date
        """
        
        market_data = pd.read_sql(regime_query, db_engine)
        
        if len(market_data) < config['regime_detection']['lookback_days'] * 0.8:
            print(f"   ⚠️ Insufficient market data for regime detection")
            return 'normal'
        
        # Calculate daily returns
        market_data['daily_return'] = market_data['close_price'].pct_change()
        market_data = market_data.dropna()
        
        if len(market_data) < 10:
            print(f"   ⚠️ Insufficient return data for regime detection")
            return 'normal'
        
        # Calculate volatility and return percentiles
        volatility = market_data['daily_return'].std() * np.sqrt(252)  # Annualized
        avg_return = market_data['daily_return'].mean() * 252  # Annualized
        
        # Get historical percentiles for comparison
        hist_query = f"""
        SELECT 
            STDDEV(daily_return) * SQRT(252) as vol,
            AVG(daily_return) * 252 as ret
        FROM (
            SELECT 
                trading_date,
                (close_price - LAG(close_price) OVER (ORDER BY trading_date)) / LAG(close_price) OVER (ORDER BY trading_date) as daily_return
            FROM vcsc_daily_data_complete
            WHERE ticker = 'VNINDEX'
            AND trading_date >= DATE_SUB('{date}', INTERVAL 252 DAY)
            AND trading_date <= '{date}'
        ) returns
        WHERE daily_return IS NOT NULL
        """
        
        hist_data = pd.read_sql(hist_query, db_engine)
        
        if hist_data.empty:
            print(f"   ⚠️ No historical data for regime comparison")
            return 'normal'
        
        hist_vol = hist_data['vol'].iloc[0]
        hist_ret = hist_data['ret'].iloc[0]
        
        # Determine regime
        if volatility > hist_vol * (config['regime_detection']['vol_threshold_pct'] / 100):
            regime = 'stress'
        elif avg_return > hist_ret * (config['regime_detection']['bull_return_threshold_pct'] / 100):
            regime = 'bull'
        else:
            regime = 'normal'
        
        print(f"   📊 Regime: {regime.upper()}")
        print(f"      Volatility: {volatility:.2%} (vs historical {hist_vol:.2%})")
        print(f"      Return: {avg_return:.2%} (vs historical {hist_ret:.2%})")
        
        return regime
        
    except Exception as e:
        print(f"   ❌ Error detecting market regime: {e}")
        return 'normal'

def calculate_regime_adjusted_scores(data_df, regime, config):
    """Calculate regime-adjusted factor scores."""
    print(f"🎯 Applying regime adjustments for {regime} regime")
    
    try:
        regime_config = config['regime_weights'][regime]
        
        # Apply regime-specific weights
        data_df['QVM_Composite_Regime'] = (
            data_df['Quality_Composite_Normalized'] * regime_config['quality_weight'] +
            data_df['Value_Composite_Normalized'] * regime_config['value_weight'] +
            data_df['Momentum_Composite_Normalized'] * regime_config['momentum_weight']
        )
        
        # Apply allocation multiplier
        data_df['allocation_multiplier'] = regime_config['allocation_multiplier']
        
        print(f"   ✅ Regime adjustments applied")
        print(f"      Quality weight: {regime_config['quality_weight']:.1%}")
        print(f"      Value weight: {regime_config['value_weight']:.1%}")
        print(f"      Momentum weight: {regime_config['momentum_weight']:.1%}")
        print(f"      Allocation: {regime_config['allocation_multiplier']:.1%}")
        
        return data_df
        
    except Exception as e:
        print(f"   ❌ Error applying regime adjustments: {e}")
        return data_df

def run_fixed_backtest(config, db_engine):
    """Run fixed backtest using existing factor scores."""
    print("🚀 Starting Fixed Available Metrics Strategy Backtest")
    print("=" * 80)
    
    try:
        # Get trading dates
        dates_query = f"""
        SELECT DISTINCT trading_date
        FROM vcsc_daily_data_complete
        WHERE trading_date >= '{config['universe']['backtest_start_date']}'
        AND trading_date <= '{config['universe']['backtest_end_date']}'
        ORDER BY trading_date
        """
        
        trading_dates = pd.read_sql(dates_query, db_engine)['trading_date'].tolist()
        
        print(f"📅 Backtest period: {len(trading_dates)} trading days")
        
        # Initialize results
        portfolio_holdings = []
        regime_history = []
        
        # Monthly rebalancing
        rebalance_dates = []
        current_month = None
        
        for date in trading_dates:
            date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
            month = date_str[:7]  # YYYY-MM
            
            if month != current_month:
                rebalance_dates.append(date_str)
                current_month = month
        
        print(f"📊 Rebalancing dates: {len(rebalance_dates)}")
        
        # Process each rebalancing date
        for i, date in enumerate(rebalance_dates):
            print(f"\n📅 Processing {date} ({i+1}/{len(rebalance_dates)})")
            
            try:
                # 1. Calculate universe
                universe = calculate_universe_rankings(db_engine, date, config)
                if not universe:
                    continue
                
                # 2. Load existing factor scores
                factors_data = load_existing_factor_scores(db_engine, date, universe)
                if factors_data.empty:
                    continue
                
                # 3. Detect market regime
                regime = detect_market_regime(db_engine, date, config)
                regime_history.append({'date': date, 'regime': regime})
                
                # 4. Apply regime adjustments
                adjusted_data = calculate_regime_adjusted_scores(factors_data, regime, config)
                
                # 5. Select top stocks
                top_stocks = adjusted_data.nlargest(config['universe']['top_n_stocks'], 'QVM_Composite_Regime')
                
                # 6. Calculate position sizes
                allocation = config['universe']['target_portfolio_size'] * adjusted_data['allocation_multiplier'].iloc[0]
                position_size = allocation / len(top_stocks)
                
                # 7. Record holdings
                for _, stock in top_stocks.iterrows():
                    portfolio_holdings.append({
                        'date': date,
                        'ticker': stock['ticker'],
                        'quality_score': stock['Quality_Composite_Normalized'],
                        'value_score': stock['Value_Composite_Normalized'],
                        'momentum_score': stock['Momentum_Composite_Normalized'],
                        'qvm_score': stock['QVM_Composite_Regime'],
                        'regime': regime,
                        'position_size': position_size
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
        print(f"📊 Regime distribution:")
        regime_counts = regime_df['regime'].value_counts()
        for regime, count in regime_counts.items():
            print(f"   {regime}: {count} ({count/len(regime_df)*100:.1f}%)")
        
        return holdings_df, regime_df
        
    except Exception as e:
        print(f"❌ Error running fixed backtest: {e}")
        return pd.DataFrame(), pd.DataFrame()

# %%
def main():
    """Main execution function."""
    print("🎯 QVM Engine v3j Available Metrics Strategy (v18b) - FIXED")
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
            print(f"📊 Holdings: {len(holdings_df)} records")
            print(f"📊 Regime history: {len(regime_df)} records")
            
            # Generate tearsheet
            generate_tearsheet(holdings_df, regime_df, QVM_CONFIG)
        
    except Exception as e:
        print(f"❌ Strategy execution failed: {e}")
        import traceback
        traceback.print_exc()

def generate_tearsheet(holdings_df, regime_df, config):
    """Generate comprehensive tearsheet analysis."""
    print("\n📊 Generating Comprehensive Tearsheet Analysis")
    print("=" * 80)
    
    try:
        # 1. Strategy Overview
        print("\n🎯 STRATEGY OVERVIEW")
        print("-" * 40)
        print(f"Strategy Name: {config['strategy_name']}")
        print(f"Backtest Period: {config['universe']['backtest_start_date']} to {config['universe']['backtest_end_date']}")
        print(f"Rebalancing: {config['universe']['rebalance_frequency']}")
        print(f"Portfolio Size: {config['universe']['top_n_stocks']} stocks")
        print(f"Transaction Cost: {config['universe']['transaction_cost_bps']} bps")
        
        # 2. Factor Configuration
        print("\n🔧 FACTOR CONFIGURATION")
        print("-" * 40)
        print(f"Quality Weight: {config['factors']['quality_weight']:.1%}")
        print(f"Value Weight: {config['factors']['value_weight']:.1%}")
        print(f"Momentum Weight: {config['factors']['momentum_weight']:.1%}")
        
        # 3. Regime Analysis
        print("\n🌍 REGIME ANALYSIS")
        print("-" * 40)
        regime_counts = regime_df['regime'].value_counts()
        total_regimes = len(regime_df)
        for regime, count in regime_counts.items():
            percentage = count / total_regimes * 100
            print(f"{regime.upper()}: {count} periods ({percentage:.1f}%)")
        
        # 4. Portfolio Statistics
        print("\n📊 PORTFOLIO STATISTICS")
        print("-" * 40)
        print(f"Total Rebalancing Periods: {len(regime_df)}")
        print(f"Total Holdings: {len(holdings_df)}")
        print(f"Average Holdings per Period: {len(holdings_df) / len(regime_df):.1f}")
        
        # 5. Factor Score Analysis
        print("\n📈 FACTOR SCORE ANALYSIS")
        print("-" * 40)
        if not holdings_df.empty:
            print(f"Quality Score Range: {holdings_df['quality_score'].min():.3f} to {holdings_df['quality_score'].max():.3f}")
            print(f"Value Score Range: {holdings_df['value_score'].min():.3f} to {holdings_df['value_score'].max():.3f}")
            print(f"Momentum Score Range: {holdings_df['momentum_score'].min():.3f} to {holdings_df['momentum_score'].max():.3f}")
            print(f"QVM Score Range: {holdings_df['qvm_score'].min():.3f} to {holdings_df['qvm_score'].max():.3f}")
        
        # 6. Top Holdings Analysis
        print("\n🏆 TOP HOLDINGS ANALYSIS")
        print("-" * 40)
        if not holdings_df.empty:
            # Most frequently held stocks
            top_stocks = holdings_df['ticker'].value_counts().head(10)
            print("Most Frequently Held Stocks:")
            for ticker, count in top_stocks.items():
                print(f"  {ticker}: {count} periods")
        
        # 7. Regime Performance
        print("\n📊 REGIME PERFORMANCE ANALYSIS")
        print("-" * 40)
        for regime in ['normal', 'stress', 'bull']:
            regime_holdings = holdings_df[holdings_df['regime'] == regime]
            if not regime_holdings.empty:
                avg_qvm = regime_holdings['qvm_score'].mean()
                print(f"{regime.upper()} Regime: {len(regime_holdings)} holdings, Avg QVM Score: {avg_qvm:.3f}")
        
        print("\n✅ Tearsheet analysis completed")
        
    except Exception as e:
        print(f"❌ Error generating tearsheet: {e}")

if __name__ == "__main__":
    main()
