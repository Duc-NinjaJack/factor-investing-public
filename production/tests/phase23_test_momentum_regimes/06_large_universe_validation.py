#!/usr/bin/env python3
"""
Large Universe Momentum Validation (200 Stocks)

Comprehensive validation of momentum factor with larger universe and
data quality checks to detect simulation/synthetic data issues.

Author: Factor Investing Team
Date: 2025-07-30
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add production path
sys.path.append('../../../production')

from engine.qvm_engine_v2_enhanced import QVMEngineV2Enhanced

def setup_logging():
    """Setup logging for the validation."""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def get_large_universe(engine, analysis_date, limit=200):
    """Get larger universe of 200 stocks by market cap."""
    try:
        # Get the latest market cap data for each ticker
        query = f"""
        SELECT 
            eh.ticker,
            eh.market_cap,
            mi.sector
        FROM (
            SELECT 
                ticker,
                market_cap,
                ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) as rn
            FROM equity_history_with_market_cap
            WHERE date <= '{analysis_date.date()}'
              AND market_cap IS NOT NULL
              AND market_cap >= 5000000000  -- 5B VND minimum
        ) eh
        LEFT JOIN master_info mi ON eh.ticker = mi.ticker
        WHERE eh.rn = 1
        ORDER BY eh.market_cap DESC
        LIMIT {limit}
        """
        
        universe_data = pd.read_sql(query, engine.engine)
        
        if universe_data.empty:
            print(f"❌ No universe data found for {analysis_date}")
            return []
        
        print(f"✅ Universe: {len(universe_data)} stocks")
        print(f"📊 Market Cap Range: {universe_data['market_cap'].min():,.0f} - {universe_data['market_cap'].max():,.0f} VND")
        print(f"🏢 Sectors: {universe_data['sector'].nunique()} unique sectors")
        
        return universe_data['ticker'].tolist()
        
    except Exception as e:
        print(f"❌ Error getting universe: {e}")
        return []

def check_data_quality(engine, analysis_date, universe):
    """Comprehensive data quality checks to detect simulation issues."""
    print(f"\n🔍 DATA QUALITY ANALYSIS")
    print("=" * 50)
    
    quality_issues = []
    
    # 1. Check price data completeness
    print(f"\n📈 Price Data Quality Check:")
    ticker_str = "', '".join(universe[:50])  # Check first 50 stocks
    
    price_query = f"""
    SELECT ticker, date, close, volume
    FROM equity_history
    WHERE ticker IN ('{ticker_str}')
      AND date BETWEEN '{analysis_date - pd.DateOffset(months=6)}' AND '{analysis_date}'
    ORDER BY ticker, date
    """
    
    try:
        price_data = pd.read_sql(price_query, engine.engine)
        
        if price_data.empty:
            quality_issues.append("No price data available")
            print("❌ No price data found")
        else:
            # Check for suspicious patterns
            print(f"✅ Price data: {len(price_data)} records")
            
            # Check for zero prices
            zero_prices = price_data[price_data['close'] == 0]
            if not zero_prices.empty:
                quality_issues.append(f"Found {len(zero_prices)} zero price records")
                print(f"⚠️ Zero prices: {len(zero_prices)} records")
            
            # Check for missing values
            missing_prices = price_data['close'].isnull().sum()
            if missing_prices > 0:
                quality_issues.append(f"Found {missing_prices} missing price values")
                print(f"⚠️ Missing prices: {missing_prices} records")
            
            # Check for suspicious price patterns (simulation indicators)
            price_stats = price_data.groupby('ticker')['close'].agg(['count', 'std', 'min', 'max'])
            
            # Check for too much regularity (simulation indicator)
            low_volatility = price_stats[price_stats['std'] < 1000]  # Very low volatility
            if len(low_volatility) > len(universe) * 0.1:  # More than 10% have very low volatility
                quality_issues.append("Suspicious: Many stocks with very low price volatility")
                print(f"⚠️ Low volatility stocks: {len(low_volatility)}")
            
            # Check for perfect data (simulation indicator)
            perfect_data = price_stats[price_stats['count'] == price_stats['count'].max()]
            if len(perfect_data) > len(universe) * 0.8:  # More than 80% have perfect data
                quality_issues.append("Suspicious: Too many stocks with perfect data completeness")
                print(f"⚠️ Perfect data stocks: {len(perfect_data)}")
    
    except Exception as e:
        quality_issues.append(f"Price data query error: {e}")
        print(f"❌ Price data error: {e}")
    
    # 2. Check fundamental data quality
    print(f"\n📊 Fundamental Data Quality Check:")
    try:
        fundamental_data = engine.get_fundamentals_correct_timing(analysis_date, universe[:50])
        
        if fundamental_data.empty:
            quality_issues.append("No fundamental data available")
            print("❌ No fundamental data found")
        else:
            print(f"✅ Fundamental data: {len(fundamental_data)} records")
            
            # Check for suspicious patterns in fundamental data
            numeric_cols = fundamental_data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols[:5]:  # Check first 5 numeric columns
                col_data = fundamental_data[col].dropna()
                
                if len(col_data) > 0:
                    # Check for too much regularity
                    if col_data.std() < col_data.mean() * 0.01:  # Very low variation
                        quality_issues.append(f"Suspicious: {col} has very low variation")
                        print(f"⚠️ Low variation in {col}: std={col_data.std():.2f}")
                    
                    # Check for perfect distribution (simulation indicator)
                    unique_vals = col_data.nunique()
                    if unique_vals < len(col_data) * 0.1:  # Less than 10% unique values
                        quality_issues.append(f"Suspicious: {col} has too few unique values")
                        print(f"⚠️ Few unique values in {col}: {unique_vals}/{len(col_data)}")
    
    except Exception as e:
        quality_issues.append(f"Fundamental data error: {e}")
        print(f"❌ Fundamental data error: {e}")
    
    # 3. Check for simulation indicators
    print(f"\n🤖 Simulation Detection Check:")
    
    # Check for perfect correlations (simulation indicator)
    try:
        sample_tickers = universe[:20]
        correlation_data = []
        
        for ticker in sample_tickers:
            ticker_prices = price_data[price_data['ticker'] == ticker]['close']
            if len(ticker_prices) > 10:
                correlation_data.append(ticker_prices)
        
        if len(correlation_data) > 5:
            # Calculate correlations between stocks
            price_matrix = pd.DataFrame(correlation_data).T
            correlations = price_matrix.corr()
            
            # Check for suspiciously high correlations
            high_corr = (correlations > 0.95).sum().sum() - len(correlations)  # Exclude diagonal
            if high_corr > len(correlations) * 0.1:  # More than 10% high correlations
                quality_issues.append("Suspicious: Too many highly correlated stocks")
                print(f"⚠️ High correlations: {high_corr} pairs")
    
    except Exception as e:
        print(f"⚠️ Correlation check error: {e}")
    
    # 4. Check data freshness
    print(f"\n📅 Data Freshness Check:")
    try:
        latest_date_query = """
        SELECT MAX(date) as latest_date
        FROM equity_history
        """
        latest_date = pd.read_sql(latest_date_query, engine.engine).iloc[0]['latest_date']
        
        days_old = (pd.Timestamp.now() - pd.Timestamp(latest_date)).days
        
        if days_old > 30:
            quality_issues.append(f"Data is {days_old} days old")
            print(f"⚠️ Data age: {days_old} days")
        else:
            print(f"✅ Data freshness: {days_old} days old")
    
    except Exception as e:
        print(f"⚠️ Data freshness check error: {e}")
    
    return quality_issues

def calculate_momentum_ic_large_universe(engine, analysis_date, universe, forward_months=1):
    """Calculate momentum IC for large universe with enhanced error handling."""
    try:
        # Get fundamental data for sector mapping
        fundamental_data = engine.get_fundamentals_correct_timing(analysis_date, universe)
        if fundamental_data.empty:
            return None
        
        # Calculate momentum factors
        momentum_scores = engine._calculate_enhanced_momentum_composite(
            fundamental_data, analysis_date, universe
        )
        if not momentum_scores:
            return None
        
        # Calculate forward returns
        end_date = analysis_date + pd.DateOffset(months=forward_months)
        ticker_str = "', '".join(universe)
        
        query = f"""
        SELECT ticker, date, close as adj_close
        FROM equity_history
        WHERE ticker IN ('{ticker_str}')
          AND date BETWEEN '{analysis_date.date()}' AND '{end_date.date()}'
        ORDER BY ticker, date
        """
        
        with engine.engine.connect() as conn:
            price_data = pd.read_sql(query, conn, parse_dates=['date'])
        
        if price_data.empty:
            return None
        
        forward_returns = {}
        for ticker in universe:
            ticker_data = price_data[price_data['ticker'] == ticker].sort_values('date')
            if len(ticker_data) >= 2:
                start_price = ticker_data.iloc[0]['adj_close']
                end_price = ticker_data.iloc[-1]['adj_close']
                if start_price > 0:
                    forward_returns[ticker] = (end_price / start_price) - 1
        
        common_tickers = set(momentum_scores.keys()) & set(forward_returns.keys())
        if len(common_tickers) < 20:  # Minimum 20 stocks for large universe
            return None
        
        factor_series = pd.Series([momentum_scores[t] for t in common_tickers], index=list(common_tickers))
        return_series = pd.Series([forward_returns[t] for t in common_tickers], index=list(common_tickers))
        
        # Calculate Spearman correlation manually
        ic = _calculate_spearman_correlation(factor_series, return_series)
        
        return {
            'date': analysis_date,
            'ic': ic,
            'n_stocks': len(common_tickers),
            'momentum_mean': factor_series.mean(),
            'momentum_std': factor_series.std(),
            'return_mean': return_series.mean(),
            'return_std': return_series.std(),
            'universe_size': len(universe)
        }
        
    except Exception as e:
        print(f"❌ Error calculating IC: {e}")
        return None

def _calculate_spearman_correlation(x, y):
    """Calculate Spearman correlation manually to avoid scipy dependency."""
    try:
        # Get ranks
        x_ranks = x.rank()
        y_ranks = y.rank()

        # Calculate correlation using Pearson formula on ranks
        n = len(x)
        if n < 2:
            return 0.0

        x_mean = x_ranks.mean()
        y_mean = y_ranks.mean()

        numerator = ((x_ranks - x_mean) * (y_ranks - y_mean)).sum()
        x_var = ((x_ranks - x_mean) ** 2).sum()
        y_var = ((y_ranks - y_mean) ** 2).sum()

        if x_var == 0 or y_var == 0:
            return 0.0

        correlation = numerator / (x_var * y_var) ** 0.5
        return correlation
    except:
        return 0.0

def test_large_universe_regime(engine, regime_name, start_date, end_date, forward_months=1, limit=200):
    """Test momentum factor with large universe across different regimes."""
    print(f"\n{'='*60}")
    print(f"🧪 TESTING LARGE UNIVERSE REGIME: {regime_name}")
    print(f"📅 Period: {start_date} to {end_date}")
    print(f"📈 Forward Horizon: {forward_months} month(s)")
    print(f"📊 Universe Size: {limit} stocks")
    print(f"{'='*60}")
    
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    test_dates = pd.date_range(
        start=start_dt + pd.DateOffset(months=12),
        end=end_dt - pd.DateOffset(months=forward_months),
        freq='Q'
    )
    
    print(f"📊 Generated {len(test_dates)} test dates")
    
    results = []
    quality_reports = []
    
    for i, test_date in enumerate(test_dates):
        print(f"\n📅 Processing {test_date.date()} ({i+1}/{len(test_dates)})")
        
        # Get large universe
        universe = get_large_universe(engine, test_date, limit=limit)
        if len(universe) < 50:  # Minimum 50 stocks for large universe
            print(f"⚠️ Universe too small: {len(universe)} stocks")
            continue
        
        # Check data quality (only for first few dates to avoid performance issues)
        if i < 3:
            quality_issues = check_data_quality(engine, test_date, universe)
            quality_reports.append({
                'date': test_date,
                'issues': quality_issues,
                'universe_size': len(universe)
            })
        
        # Calculate IC
        ic_result = calculate_momentum_ic_large_universe(engine, test_date, universe, forward_months)
        if ic_result:
            results.append(ic_result)
            print(f"✅ IC: {ic_result['ic']:.4f} (n={ic_result['n_stocks']})")
        else:
            print(f"❌ Failed to calculate IC")
    
    return results, quality_reports

def analyze_large_universe_results(results, quality_reports, regime_name):
    """Analyze results for large universe with quality assessment."""
    if not results:
        print(f"❌ No results for {regime_name}")
        return
    
    ic_values = [r['ic'] for r in results if not np.isnan(r['ic'])]
    if not ic_values:
        print(f"❌ No valid IC values for {regime_name}")
        return
    
    ic_series = pd.Series(ic_values)
    
    print(f"\n📊 LARGE UNIVERSE IC ANALYSIS FOR {regime_name}")
    print(f"{'='*60}")
    print(f"Number of observations: {len(ic_series)}")
    print(f"Mean IC: {ic_series.mean():.4f}")
    print(f"Std IC: {ic_series.std():.4f}")
    print(f"Min IC: {ic_series.min():.4f}")
    print(f"Max IC: {ic_series.max():.4f}")
    
    # Calculate t-statistic
    t_stat = ic_series.mean() / (ic_series.std() / np.sqrt(len(ic_series)))
    print(f"T-statistic: {t_stat:.4f}")
    
    # Calculate hit rate
    hit_rate = (ic_series > 0).mean()
    print(f"Hit Rate: {hit_rate:.1%}")
    
    # Quality assessment
    print(f"\n🔍 QUALITY ASSESSMENT:")
    print(f"{'='*30}")
    
    # Check quality gates
    mean_ic = ic_series.mean()
    quality_gates = {
        'Mean IC > 0.02': mean_ic > 0.02,
        'T-stat > 2.0': t_stat > 2.0,
        'Hit Rate > 55%': hit_rate > 0.55
    }
    
    for gate, passed in quality_gates.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{gate}: {status}")
    
    # Data quality summary
    if quality_reports:
        total_issues = sum(len(report['issues']) for report in quality_reports)
        print(f"\n📋 Data Quality Issues Found: {total_issues}")
        
        if total_issues > 0:
            print("⚠️ Potential data quality concerns detected")
        else:
            print("✅ No significant data quality issues detected")
    
    return {
        'regime': regime_name,
        'mean_ic': mean_ic,
        't_stat': t_stat,
        'hit_rate': hit_rate,
        'n_obs': len(ic_series),
        'quality_issues': total_issues if quality_reports else 0,
        'quality_gates_passed': sum(quality_gates.values())
    }

def main():
    """Main execution function for large universe validation."""
    print("🚀 LARGE UNIVERSE MOMENTUM VALIDATION")
    print("=" * 60)
    print("Testing momentum factor with 200 stocks universe")
    print("Including comprehensive data quality checks")
    print("=" * 60)
    
    # Initialize engine
    try:
        engine = QVMEngineV2Enhanced()
        print("✅ Engine initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize engine: {e}")
        return
    
    # Test regimes
    regimes = [
        ('2016-2020', '2016-01-01', '2020-12-31'),
        ('2021-2025', '2021-01-01', '2025-07-30')
    ]
    
    all_results = {}
    
    for regime_name, start_date, end_date in regimes:
        print(f"\n{'='*60}")
        print(f"🧪 TESTING REGIME: {regime_name}")
        print(f"{'='*60}")
        
        # Test different forward horizons
        for forward_months in [1, 3, 6, 12]:
            print(f"\n📈 Testing {forward_months}M forward horizon...")
            
            results, quality_reports = test_large_universe_regime(
                engine, regime_name, start_date, end_date, forward_months, limit=200
            )
            
            analysis_result = analyze_large_universe_results(results, quality_reports, f"{regime_name} ({forward_months}M)")
            
            if analysis_result:
                key = f"{regime_name}_{forward_months}M"
                all_results[key] = analysis_result
    
    # Summary report
    print(f"\n{'='*60}")
    print(f"📊 LARGE UNIVERSE VALIDATION SUMMARY")
    print(f"{'='*60}")
    
    for key, result in all_results.items():
        print(f"\n{key}:")
        print(f"  Mean IC: {result['mean_ic']:.4f}")
        print(f"  T-stat: {result['t_stat']:.4f}")
        print(f"  Hit Rate: {result['hit_rate']:.1%}")
        print(f"  Quality Issues: {result['quality_issues']}")
        print(f"  Gates Passed: {result['quality_gates_passed']}/3")
    
    print(f"\n✅ Large universe validation completed!")

if __name__ == "__main__":
    main() 