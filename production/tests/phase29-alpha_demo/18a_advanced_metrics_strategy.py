#!/usr/bin/env python3

# %% [markdown]
# # QVM Engine v3j - Advanced Metrics Strategy (Version 18a)
# 
# This version implements:
# 1. **Proper Factor Refresh Cycles**: 
#    - Quality factors (fundamental-based): Updated quarterly
#    - Value factors (price-based): Updated daily
#    - Momentum factors (price-based): Updated daily
# 2. **Full Advanced Metrics Integration**:
#    - Quality: ROAA + F-Score (50/50)
#    - Value: P/E + FCF Yield (50/50)
#    - Momentum: Momentum + Low Vol (50/50)
# 3. **Proper Normalization**: Ranking-based 0-1 scale
# 4. **Regime Detection**: Dynamic factor weighting

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

print("✅ QVM Engine v3j Advanced Metrics Strategy (v18a) initialized")

# %%
# Configuration
QVM_CONFIG = {
    'strategy_name': 'QVM_Engine_v3j_Advanced_Metrics_v18a',
    'universe': {
        'lookback_days': 252,
        'top_n_stocks': 20,
        'target_portfolio_size': 20,
        'backtest_start_date': '2022-01-01',
        'backtest_end_date': '2025-12-31',
        'rebalance_frequency': 'monthly',
        'transaction_cost_bps': 10,
        'adtv_threshold_bn': 10.0
    },
    'factors': {
        'quality_weight': 0.4,
        'value_weight': 0.3,
        'momentum_weight': 0.3,
        'quality_factors': {
            'roaa_weight': 0.5,
            'fscore_weight': 0.5
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
            'quality_weight': 0.4,
            'value_weight': 0.3,
            'momentum_weight': 0.3,
            'allocation_multiplier': 1.0
        },
        'stress': {
            'quality_weight': 0.6,
            'value_weight': 0.3,
            'momentum_weight': 0.1,
            'allocation_multiplier': 0.6
        },
        'bull': {
            'quality_weight': 0.2,
            'value_weight': 0.3,
            'momentum_weight': 0.5,
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

def load_advanced_metrics(db_engine, date, tickers):
    """Load advanced metrics including F-Score, FCF Yield, and Low Vol."""
    print(f"📊 Loading advanced metrics for {date}")
    
    ticker_list = "', '".join(tickers)
    
    try:
        # 1. Load fundamental data (quarterly refresh)
        fundamental_query = f"""
        SELECT 
            ticker,
            roa,
            roe,
            pe_ratio,
            pb_ratio,
            ps_ratio,
            gross_margin,
            operating_margin,
            net_profit_margin,
            market_capitalization,
            data_date
        FROM wong_api_daily_financial_info
        WHERE data_date = '{date}'
        AND ticker IN ('{ticker_list}')
        ORDER BY ticker
        """
        
        fundamental_data = pd.read_sql(fundamental_query, db_engine)
        print(f"   ✅ Loaded fundamental data for {len(fundamental_data)} stocks")
        
        # 2. Load F-Score data (quarterly refresh)
        fscore_query = f"""
        SELECT 
            ticker,
            value as f_score,
            period_end_date
        FROM precalculated_metrics
        WHERE metric_name LIKE '%f_score%'
        AND period_end_date <= '{date}'
        AND ticker IN ('{ticker_list}')
        ORDER BY ticker, period_end_date DESC
        """
        
        fscore_data = pd.read_sql(fscore_query, db_engine)
        if not fscore_data.empty:
            # Get latest F-Score for each ticker
            fscore_data = fscore_data.groupby('ticker').first().reset_index()
            print(f"   ✅ Loaded F-Score data for {len(fscore_data)} stocks")
        else:
            print(f"   ⚠️ No F-Score data available")
        
        # 3. Load FCF Yield data (quarterly refresh)
        fcf_query = f"""
        SELECT 
            ticker,
            value as fcf_yield,
            period_end_date
        FROM precalculated_metrics
        WHERE metric_name LIKE '%fcf%' OR metric_name LIKE '%free_cash_flow%'
        AND period_end_date <= '{date}'
        AND ticker IN ('{ticker_list}')
        ORDER BY ticker, period_end_date DESC
        """
        
        fcf_data = pd.read_sql(fcf_query, db_engine)
        if not fcf_data.empty:
            # Get latest FCF Yield for each ticker
            fcf_data = fcf_data.groupby('ticker').first().reset_index()
            print(f"   ✅ Loaded FCF Yield data for {len(fcf_data)} stocks")
        else:
            print(f"   ⚠️ No FCF Yield data available")
        
        # 4. Load price data for momentum and low vol (daily refresh)
        price_query = f"""
        SELECT 
            ticker,
            trading_date,
            close_price,
            total_volume
        FROM vcsc_daily_data_complete
        WHERE trading_date >= DATE_SUB('{date}', INTERVAL 252 DAY)
        AND trading_date <= '{date}'
        AND ticker IN ('{ticker_list}')
        ORDER BY ticker, trading_date
        """
        
        price_data = pd.read_sql(price_query, db_engine)
        print(f"   ✅ Loaded price data for {len(price_data['ticker'].unique())} stocks")
        
        # Merge all data
        merged_data = fundamental_data.copy()
        
        if not fscore_data.empty:
            merged_data = merged_data.merge(fscore_data[['ticker', 'f_score']], on='ticker', how='left')
        
        if not fcf_data.empty:
            merged_data = merged_data.merge(fcf_data[['ticker', 'fcf_yield']], on='ticker', how='left')
        
        # Calculate momentum and low vol from price data
        momentum_data = calculate_momentum_factors(price_data, date)
        low_vol_data = calculate_low_volatility_factors(price_data, date)
        
        if not momentum_data.empty:
            merged_data = merged_data.merge(momentum_data, on='ticker', how='left')
        
        if not low_vol_data.empty:
            merged_data = merged_data.merge(low_vol_data, on='ticker', how='left')
        
        print(f"   ✅ Final merged data: {len(merged_data)} stocks")
        return merged_data
        
    except Exception as e:
        print(f"   ❌ Error loading advanced metrics: {e}")
        return pd.DataFrame()

def calculate_momentum_factors(price_data, date):
    """Calculate momentum factors (daily refresh)."""
    print(f"   📈 Calculating momentum factors")
    
    try:
        momentum_results = []
        
        for ticker in price_data['ticker'].unique():
            ticker_data = price_data[price_data['ticker'] == ticker].sort_values('trading_date')
            
            if len(ticker_data) < 60:  # Need at least 60 days
                continue
            
            # Calculate returns for different horizons (skip-1-month convention)
            returns = {}
            for horizon in [21, 63, 126, 252]:  # 1M, 3M, 6M, 12M
                if len(ticker_data) >= horizon + 21:  # +21 for skip-1-month
                    start_price = ticker_data.iloc[-(horizon + 21)]['close_price']
                    end_price = ticker_data.iloc[-21]['close_price']  # Skip last month
                    returns[f'{horizon}d_return'] = (end_price - start_price) / start_price
            
            if returns:
                # Composite momentum (equal weighted)
                momentum_score = np.mean(list(returns.values()))
                momentum_results.append({
                    'ticker': ticker,
                    'momentum_score': momentum_score
                })
        
        return pd.DataFrame(momentum_results)
        
    except Exception as e:
        print(f"   ❌ Error calculating momentum factors: {e}")
        return pd.DataFrame()

def calculate_low_volatility_factors(price_data, date):
    """Calculate low volatility factors (daily refresh)."""
    print(f"   📉 Calculating low volatility factors")
    
    try:
        low_vol_results = []
        
        for ticker in price_data['ticker'].unique():
            ticker_data = price_data[price_data['ticker'] == ticker].sort_values('trading_date')
            
            if len(ticker_data) < 60:  # Need at least 60 days
                continue
            
            # Calculate daily returns
            ticker_data['daily_return'] = ticker_data['close_price'].pct_change()
            ticker_data = ticker_data.dropna()
            
            if len(ticker_data) < 30:  # Need at least 30 days of returns
                continue
            
            # Calculate rolling volatility (30-day)
            ticker_data['volatility'] = ticker_data['daily_return'].rolling(30).std()
            
            # Low vol score (inverse of volatility)
            latest_vol = ticker_data['volatility'].iloc[-1]
            if pd.notna(latest_vol) and latest_vol > 0:
                low_vol_score = 1 / latest_vol  # Higher score = lower volatility
                low_vol_results.append({
                    'ticker': ticker,
                    'low_vol_score': low_vol_score
                })
        
        return pd.DataFrame(low_vol_results)
        
    except Exception as e:
        print(f"   ❌ Error calculating low volatility factors: {e}")
        return pd.DataFrame()

def calculate_advanced_factors(data_df, config):
    """Calculate advanced factors with proper normalization."""
    print(f"📊 Calculating advanced factors")
    
    try:
        # 1. Quality Factor (ROAA + F-Score) - Quarterly refresh
        print(f"   🏆 Calculating Quality Factor...")
        
        # ROAA component (0-1 scale)
        if 'roa' in data_df.columns:
            data_df['roaa_rank'] = data_df['roa'].rank(ascending=True, method='min')
            data_df['roaa_normalized'] = (data_df['roaa_rank'] - 1) / (len(data_df) - 1)
        else:
            data_df['roaa_normalized'] = 0.5
        
        # F-Score component (0-1 scale)
        if 'f_score' in data_df.columns:
            data_df['fscore_rank'] = data_df['f_score'].rank(ascending=True, method='min')
            data_df['fscore_normalized'] = (data_df['fscore_rank'] - 1) / (len(data_df) - 1)
        else:
            data_df['fscore_normalized'] = 0.5
        
        # Quality composite (50% ROAA + 50% F-Score)
        data_df['Quality_Composite'] = (
            data_df['roaa_normalized'] * config['factors']['quality_factors']['roaa_weight'] +
            data_df['fscore_normalized'] * config['factors']['quality_factors']['fscore_weight']
        )
        
        # 2. Value Factor (P/E + FCF Yield) - Daily refresh
        print(f"   💰 Calculating Value Factor...")
        
        # P/E component (0-1 scale, lower P/E = higher value)
        if 'pe_ratio' in data_df.columns:
            valid_pe = data_df[data_df['pe_ratio'] > 0].copy()
            if len(valid_pe) > 0:
                valid_pe['pe_rank'] = valid_pe['pe_ratio'].rank(ascending=False, method='min')  # Lower P/E = higher rank
                valid_pe['pe_normalized'] = (valid_pe['pe_rank'] - 1) / (len(valid_pe) - 1)
                data_df = data_df.merge(valid_pe[['ticker', 'pe_normalized']], on='ticker', how='left')
                data_df['pe_normalized'] = data_df['pe_normalized'].fillna(0.5)
            else:
                data_df['pe_normalized'] = 0.5
        else:
            data_df['pe_normalized'] = 0.5
        
        # FCF Yield component (0-1 scale)
        if 'fcf_yield' in data_df.columns:
            data_df['fcf_rank'] = data_df['fcf_yield'].rank(ascending=True, method='min')
            data_df['fcf_normalized'] = (data_df['fcf_rank'] - 1) / (len(data_df) - 1)
        else:
            data_df['fcf_normalized'] = 0.5
        
        # Value composite (50% P/E + 50% FCF Yield)
        data_df['Value_Composite'] = (
            data_df['pe_normalized'] * config['factors']['value_factors']['pe_weight'] +
            data_df['fcf_normalized'] * config['factors']['value_factors']['fcf_yield_weight']
        )
        
        # 3. Momentum Factor (Momentum + Low Vol) - Daily refresh
        print(f"   📈 Calculating Momentum Factor...")
        
        # Momentum component (0-1 scale)
        if 'momentum_score' in data_df.columns:
            data_df['momentum_rank'] = data_df['momentum_score'].rank(ascending=True, method='min')
            data_df['momentum_normalized'] = (data_df['momentum_rank'] - 1) / (len(data_df) - 1)
        else:
            data_df['momentum_normalized'] = 0.5
        
        # Low Vol component (0-1 scale)
        if 'low_vol_score' in data_df.columns:
            data_df['lowvol_rank'] = data_df['low_vol_score'].rank(ascending=True, method='min')
            data_df['lowvol_normalized'] = (data_df['lowvol_rank'] - 1) / (len(data_df) - 1)
        else:
            data_df['lowvol_normalized'] = 0.5
        
        # Momentum composite (50% Momentum + 50% Low Vol)
        data_df['Momentum_Composite'] = (
            data_df['momentum_normalized'] * config['factors']['momentum_factors']['momentum_weight'] +
            data_df['lowvol_normalized'] * config['factors']['momentum_factors']['low_vol_weight']
        )
        
        # 4. Final QVM Composite
        print(f"   🎯 Calculating QVM Composite...")
        data_df['QVM_Composite'] = (
            data_df['Quality_Composite'] * config['factors']['quality_weight'] +
            data_df['Value_Composite'] * config['factors']['value_weight'] +
            data_df['Momentum_Composite'] * config['factors']['momentum_weight']
        )
        
        print(f"   ✅ Advanced factors calculated for {len(data_df)} stocks")
        print(f"   📊 Factor ranges:")
        print(f"      Quality: {data_df['Quality_Composite'].min():.3f} to {data_df['Quality_Composite'].max():.3f}")
        print(f"      Value: {data_df['Value_Composite'].min():.3f} to {data_df['Value_Composite'].max():.3f}")
        print(f"      Momentum: {data_df['Momentum_Composite'].min():.3f} to {data_df['Momentum_Composite'].max():.3f}")
        print(f"      QVM: {data_df['QVM_Composite'].min():.3f} to {data_df['QVM_Composite'].max():.3f}")
        
        return data_df
        
    except Exception as e:
        print(f"   ❌ Error calculating advanced factors: {e}")
        return data_df

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
            data_df['Quality_Composite'] * regime_config['quality_weight'] +
            data_df['Value_Composite'] * regime_config['value_weight'] +
            data_df['Momentum_Composite'] * regime_config['momentum_weight']
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

def run_advanced_backtest(config, db_engine):
    """Run advanced backtest with proper factor refresh cycles."""
    print("🚀 Starting Advanced Metrics Strategy Backtest")
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
                
                # 2. Load advanced metrics
                metrics_data = load_advanced_metrics(db_engine, date, universe)
                if metrics_data.empty:
                    continue
                
                # 3. Calculate advanced factors
                factors_data = calculate_advanced_factors(metrics_data, config)
                if factors_data.empty:
                    continue
                
                # 4. Detect market regime
                regime = detect_market_regime(db_engine, date, config)
                regime_history.append({'date': date, 'regime': regime})
                
                # 5. Apply regime adjustments
                adjusted_data = calculate_regime_adjusted_scores(factors_data, regime, config)
                
                # 6. Select top stocks
                top_stocks = adjusted_data.nlargest(config['universe']['top_n_stocks'], 'QVM_Composite_Regime')
                
                # 7. Calculate position sizes
                allocation = config['universe']['target_portfolio_size'] * adjusted_data['allocation_multiplier'].iloc[0]
                position_size = allocation / len(top_stocks)
                
                # 8. Record holdings
                for _, stock in top_stocks.iterrows():
                    portfolio_holdings.append({
                        'date': date,
                        'ticker': stock['ticker'],
                        'quality_score': stock['Quality_Composite'],
                        'value_score': stock['Value_Composite'],
                        'momentum_score': stock['Momentum_Composite'],
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
        print(f"❌ Error running advanced backtest: {e}")
        return pd.DataFrame(), pd.DataFrame()

# %%
def main():
    """Main execution function."""
    print("🎯 QVM Engine v3j Advanced Metrics Strategy (v18a)")
    print("=" * 80)
    
    try:
        # Create database connection
        db_engine = create_db_connection()
        
        # Run advanced backtest
        holdings_df, regime_df = run_advanced_backtest(QVM_CONFIG, db_engine)
        
        if not holdings_df.empty:
            # Save results
            results_dir = Path("insights")
            results_dir.mkdir(exist_ok=True)
            
            holdings_df.to_csv(results_dir / "18a_advanced_holdings.csv", index=False)
            regime_df.to_csv(results_dir / "18a_regime_history.csv", index=False)
            
            print(f"\n✅ Results saved to insights/")
            print(f"📊 Holdings: {len(holdings_df)} records")
            print(f"📊 Regime history: {len(regime_df)} records")
        
    except Exception as e:
        print(f"❌ Strategy execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
