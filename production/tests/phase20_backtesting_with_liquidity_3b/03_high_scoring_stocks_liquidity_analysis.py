#!/usr/bin/env python3
"""
High-Scoring Stocks Liquidity Bucket Analysis
============================================

This script analyzes the distribution of high-scoring stocks across liquidity buckets
and their performance when included in factor portfolios.

Key Questions:
1. Are high-scoring stocks concentrated in certain liquidity buckets?
2. How do high-scoring stocks perform when included in portfolios?
3. What is the performance impact of including/excluding these stocks?
"""

import sys
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Add production engine to path
project_root = Path.cwd()
while not (project_root / 'production').exists() and not (project_root / 'config').exists():
    if project_root.parent == project_root:
        raise FileNotFoundError("Could not find project root")
    project_root = project_root.parent

production_path = project_root / 'production'
if str(production_path) not in sys.path:
    sys.path.insert(0, str(production_path))

print("🔍 HIGH-SCORING STOCKS LIQUIDITY ANALYSIS")
print("=" * 50)
print(f"Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def load_data():
    """Load the unrestricted universe data."""
    data_path = Path(__file__).parent / "data" / "unrestricted_universe_data.pkl"
    
    print("📊 Loading unrestricted universe data...")
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    
    return data


def create_liquidity_buckets(adtv, date):
    """Create liquidity buckets for a specific date."""
    # Get ADTV for the specific date
    if date in adtv.index:
        date_adtv = adtv.loc[date].dropna()
    else:
        available_dates = adtv.index[adtv.index <= date]
        if len(available_dates) == 0:
            raise ValueError(f"No ADTV data available for {date}")
        date_adtv = adtv.loc[available_dates[-1]].dropna()
    
    # Define liquidity buckets
    buckets = {
        'below_1b': (0, 1_000_000_000),
        '1b_to_3b': (1_000_000_000, 3_000_000_000),
        '3b_to_5b': (3_000_000_000, 5_000_000_000),
        '5b_to_10b': (5_000_000_000, 10_000_000_000),
        'above_10b': (10_000_000_000, float('inf'))
    }
    
    bucket_assignments = {}
    for bucket_name, (min_adtv, max_adtv) in buckets.items():
        if max_adtv == float('inf'):
            mask = date_adtv >= min_adtv
        else:
            mask = (date_adtv >= min_adtv) & (date_adtv < max_adtv)
        
        bucket_assignments[bucket_name] = date_adtv[mask].index.tolist()
    
    return bucket_assignments, date_adtv


def analyze_high_scoring_distribution(factor_data, adtv, analysis_dates):
    """Analyze distribution of high-scoring stocks across liquidity buckets."""
    print("\n📊 Analyzing high-scoring stocks distribution...")
    
    # Debug: Check date format
    print(f"    Factor data calculation_date dtype: {factor_data['calculation_date'].dtype}")
    print(f"    Sample factor dates: {factor_data['calculation_date'].head().tolist()}")
    print(f"    Analysis dates dtype: {analysis_dates.dtype}")
    print(f"    Sample analysis dates: {analysis_dates[:5].tolist()}")
    
    results = []
    
    for date in analysis_dates:
        print(f"    Analyzing {date.date()}...")
        
        # Convert date to same format as factor data
        date_str = date.strftime('%Y-%m-%d')
        print(f"      Looking for date: {date_str}")
        
        # Get factor scores for this date
        date_factors = factor_data[factor_data['calculation_date'] == date]
        print(f"      Found {len(date_factors)} factor records for this date")
        
        if date_factors.empty:
            print(f"      ⚠️ No factor data for {date.date()}")
            continue
        
        # Get liquidity buckets
        try:
            bucket_assignments, date_adtv = create_liquidity_buckets(adtv, date)
            print(f"      Created {len(bucket_assignments)} liquidity buckets")
        except Exception as e:
            print(f"      ❌ Error creating liquidity buckets: {e}")
            continue
        
        # Analyze each factor type
        for factor_type in ['quality_score', 'value_score', 'momentum_score', 'qvm_composite_score']:
            print(f"      Analyzing {factor_type}...")
            
            # Get top 25 stocks by this factor
            top_25 = date_factors.nlargest(25, factor_type)
            top_25_tickers = top_25['ticker'].tolist()
            print(f"        Top 25 tickers: {len(top_25_tickers)}")
            
            # Analyze distribution across buckets
            bucket_counts = {}
            bucket_scores = {}
            
            for bucket_name, bucket_stocks in bucket_assignments.items():
                # Count how many top 25 stocks are in this bucket
                bucket_top_stocks = [t for t in top_25_tickers if t in bucket_stocks]
                bucket_counts[bucket_name] = len(bucket_top_stocks)
                
                # Get average score for stocks in this bucket
                bucket_factor_scores = date_factors[
                    date_factors['ticker'].isin(bucket_stocks)
                ][factor_type].mean()
                bucket_scores[bucket_name] = bucket_factor_scores
            
            print(f"        Bucket counts: {bucket_counts}")
            
            # Store results
            result = {
                'date': date,
                'factor_type': factor_type,
                'below_1b_count': bucket_counts.get('below_1b', 0),
                '1b_to_3b_count': bucket_counts.get('1b_to_3b', 0),
                '3b_to_5b_count': bucket_counts.get('3b_to_5b', 0),
                '5b_to_10b_count': bucket_counts.get('5b_to_10b', 0),
                'above_10b_count': bucket_counts.get('above_10b', 0),
                'below_1b_score': bucket_scores.get('below_1b', np.nan),
                '1b_to_3b_score': bucket_scores.get('1b_to_3b', np.nan),
                '3b_to_5b_score': bucket_scores.get('3b_to_5b', np.nan),
                '5b_to_10b_score': bucket_scores.get('5b_to_10b', np.nan),
                'above_10b_score': bucket_scores.get('above_10b', np.nan),
                'total_top_25': len(top_25_tickers)
            }
            
            results.append(result)
            print(f"        ✅ Added result for {factor_type}")
    
    print(f"    Total results generated: {len(results)}")
    return pd.DataFrame(results)


def calculate_returns_performance(factor_data, volume_data, adtv, analysis_dates):
    """Calculate performance of high-scoring stocks by liquidity bucket."""
    print("\n📈 Calculating performance by liquidity bucket...")
    
    # Calculate daily returns
    price_pivot = volume_data.pivot(index='date', columns='ticker', values='close_price_adjusted')
    daily_returns = price_pivot.pct_change().iloc[1:]
    
    performance_results = []
    
    for date in analysis_dates:
        print(f"    Analyzing performance for {date.date()}...")
        
        # Get factor scores for this date
        date_factors = factor_data[factor_data['calculation_date'] == date]
        if date_factors.empty:
            continue
        
        # Get liquidity buckets
        bucket_assignments, date_adtv = create_liquidity_buckets(adtv, date)
        
        # Analyze each factor type
        for factor_type in ['qvm_composite_score']:
            # Get top 25 stocks by this factor
            top_25 = date_factors.nlargest(25, factor_type)
            top_25_tickers = top_25['ticker'].tolist()
            
            # Calculate forward returns (next 63 days)
            forward_start = date + pd.Timedelta(days=1)
            forward_end = date + pd.Timedelta(days=63)
            
            # Get forward returns for top 25 stocks
            forward_returns = daily_returns.loc[forward_start:forward_end, top_25_tickers]
            
            if forward_returns.empty:
                continue
            
            # Calculate performance by bucket
            for bucket_name, bucket_stocks in bucket_assignments.items():
                bucket_top_stocks = [t for t in top_25_tickers if t in bucket_stocks]
                
                if len(bucket_top_stocks) > 0:
                    bucket_returns = forward_returns[bucket_top_stocks].mean(axis=1)
                    
                    # Calculate performance metrics
                    total_return = (1 + bucket_returns).prod() - 1
                    annualized_return = (1 + total_return) ** (252/63) - 1
                    volatility = bucket_returns.std() * np.sqrt(252)
                    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
                    
                    performance_results.append({
                        'date': date,
                        'bucket': bucket_name,
                        'stock_count': len(bucket_top_stocks),
                        'total_return': total_return,
                        'annualized_return': annualized_return,
                        'volatility': volatility,
                        'sharpe_ratio': sharpe_ratio
                    })
    
    return pd.DataFrame(performance_results)


def create_visualizations(distribution_df, performance_df):
    """Create visualizations of the analysis results."""
    print("\n📊 Creating visualizations...")
    
    # Check if we have data
    if distribution_df.empty:
        print("❌ No distribution data available for visualization")
        return
    
    # Set up plotting
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('High-Scoring Stocks Liquidity Bucket Analysis', fontsize=16, fontweight='bold')
    
    # Check available columns
    print(f"Available columns: {distribution_df.columns.tolist()}")
    
    # 1. Distribution of top 25 stocks across buckets
    bucket_cols = ['below_1b_count', '1b_to_3b_count', '3b_to_5b_count', '5b_to_10b_count', 'above_10b_count']
    
    # Filter to only include columns that exist
    available_bucket_cols = [col for col in bucket_cols if col in distribution_df.columns]
    
    if available_bucket_cols:
        avg_distribution = distribution_df[available_bucket_cols].mean()
        
        bucket_names = ['Below 1B', '1B-3B', '3B-5B', '5B-10B', 'Above 10B'][:len(available_bucket_cols)]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'][:len(available_bucket_cols)]
        
        bars = axes[0, 0].bar(bucket_names, avg_distribution.values, color=colors, alpha=0.7)
        axes[0, 0].set_title('Average Distribution of Top 25 Stocks by Liquidity Bucket')
        axes[0, 0].set_ylabel('Average Count in Top 25')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, avg_distribution.values):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                           f'{value:.1f}', ha='center', va='bottom')
    else:
        axes[0, 0].text(0.5, 0.5, 'No distribution data available', ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 0].set_title('Distribution Data Not Available')
    
    # 2. Average factor scores by bucket
    score_cols = ['below_1b_score', '1b_to_3b_score', '3b_to_5b_score', '5b_to_10b_score', 'above_10b_score']
    available_score_cols = [col for col in score_cols if col in distribution_df.columns]
    
    if available_score_cols:
        avg_scores = distribution_df[available_score_cols].mean()
        
        bucket_names = ['Below 1B', '1B-3B', '3B-5B', '5B-10B', 'Above 10B'][:len(available_score_cols)]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'][:len(available_score_cols)]
        
        bars = axes[0, 1].bar(bucket_names, avg_scores.values, color=colors, alpha=0.7)
        axes[0, 1].set_title('Average QVM Composite Score by Liquidity Bucket')
        axes[0, 1].set_ylabel('Average Factor Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, avg_scores.values):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                           f'{value:.3f}', ha='center', va='bottom')
    else:
        axes[0, 1].text(0.5, 0.5, 'No score data available', ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Score Data Not Available')
    
    # 3. Performance comparison by bucket
    if not performance_df.empty:
        bucket_performance = performance_df.groupby('bucket').agg({
            'annualized_return': 'mean',
            'sharpe_ratio': 'mean',
            'stock_count': 'mean'
        })
        
        # Reindex to match expected order
        expected_buckets = ['below_1b', '1b_to_3b', '3b_to_5b', '5b_to_10b', 'above_10b']
        available_buckets = [b for b in expected_buckets if b in bucket_performance.index]
        
        if available_buckets:
            bucket_performance = bucket_performance.reindex(available_buckets)
            
            bucket_names = ['Below 1B', '1B-3B', '3B-5B', '5B-10B', 'Above 10B'][:len(available_buckets)]
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'][:len(available_buckets)]
            
            bars = axes[1, 0].bar(bucket_names, bucket_performance['annualized_return'].values, 
                                 color=colors, alpha=0.7)
            axes[1, 0].set_title('Average Annualized Return by Liquidity Bucket')
            axes[1, 0].set_ylabel('Annualized Return')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, bucket_performance['annualized_return'].values):
                axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                               f'{value:.1%}', ha='center', va='bottom')
            
            # 4. Sharpe ratio comparison
            bars = axes[1, 1].bar(bucket_names, bucket_performance['sharpe_ratio'].values, 
                                 color=colors, alpha=0.7)
            axes[1, 1].set_title('Average Sharpe Ratio by Liquidity Bucket')
            axes[1, 1].set_ylabel('Sharpe Ratio')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, bucket_performance['sharpe_ratio'].values):
                axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                               f'{value:.2f}', ha='center', va='bottom')
        else:
            axes[1, 0].text(0.5, 0.5, 'No performance data available', ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Performance Data Not Available')
            axes[1, 1].text(0.5, 0.5, 'No performance data available', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Performance Data Not Available')
    else:
        axes[1, 0].text(0.5, 0.5, 'No performance data available', ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Performance Data Not Available')
        axes[1, 1].text(0.5, 0.5, 'No performance data available', ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Performance Data Not Available')
    
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / 'img' / 'high_scoring_stocks_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Visualizations saved to img/high_scoring_stocks_analysis.png")


def generate_insights(distribution_df, performance_df):
    """Generate insights from the analysis."""
    print("\n🎯 GENERATING INSIGHTS")
    print("=" * 50)
    
    # Debug: Check what data we have
    print(f"Distribution DataFrame shape: {distribution_df.shape}")
    print(f"Distribution DataFrame columns: {distribution_df.columns.tolist()}")
    print(f"Performance DataFrame shape: {performance_df.shape}")
    if not performance_df.empty:
        print(f"Performance DataFrame columns: {performance_df.columns.tolist()}")
    
    # Check if we have any data
    if distribution_df.empty:
        print("❌ No distribution data available for insights")
        return
    
    # Distribution insights
    bucket_cols = ['below_1b_count', '1b_to_3b_count', '3b_to_5b_count', '5b_to_10b_count', 'above_10b_count']
    
    # Filter to only include columns that exist
    available_bucket_cols = [col for col in bucket_cols if col in distribution_df.columns]
    
    if available_bucket_cols:
        avg_distribution = distribution_df[available_bucket_cols].mean()
        
        print("\n📊 DISTRIBUTION INSIGHTS:")
        print(f"    Average distribution of top 25 stocks across liquidity buckets:")
        for i, (col, value) in enumerate(avg_distribution.items()):
            bucket_name = ['Below 1B', '1B-3B', '3B-5B', '5B-10B', 'Above 10B'][i]
            percentage = (value / 25) * 100
            print(f"    {bucket_name}: {value:.1f} stocks ({percentage:.1f}%)")
        
        # Find most concentrated bucket
        most_concentrated = avg_distribution.idxmax()
        most_concentrated_value = avg_distribution.max()
        most_concentrated_pct = (most_concentrated_value / 25) * 100
        
        print(f"\n🎯 CONCENTRATION PATTERN:")
        print(f"    Most concentrated bucket: {most_concentrated} ({most_concentrated_pct:.1f}% of top 25)")
        
        # 3B VND threshold implications
        if 'below_1b_count' in available_bucket_cols and '1b_to_3b_count' in available_bucket_cols:
            below_3b_total = avg_distribution['below_1b_count'] + avg_distribution['1b_to_3b_count']
            below_3b_pct = (below_3b_total / 25) * 100
            
            print(f"\n💡 3B VND THRESHOLD IMPLICATIONS:")
            print(f"    Stocks below 3B VND in top 25: {below_3b_total:.1f} ({below_3b_pct:.1f}%)")
            print(f"    Stocks above 3B VND in top 25: {25 - below_3b_total:.1f} ({100 - below_3b_pct:.1f}%)")
            
            if below_3b_pct > 50:
                print(f"    🚨 CRITICAL: Majority of high-scoring stocks are below 3B VND!")
                print(f"    Current 10B VND filter is excluding significant alpha opportunities")
            else:
                print(f"    ✅ Most high-scoring stocks are above 3B VND")
    else:
        print("❌ No bucket distribution data available")
    
    # Performance insights
    if not performance_df.empty:
        print(f"\n📈 PERFORMANCE INSIGHTS:")
        bucket_performance = performance_df.groupby('bucket').agg({
            'annualized_return': 'mean',
            'sharpe_ratio': 'mean',
            'stock_count': 'mean'
        })
        
        print(f"    Performance by bucket:")
        for bucket, row in bucket_performance.iterrows():
            print(f"    {bucket}: Return={row['annualized_return']:.1%}, Sharpe={row['sharpe_ratio']:.2f}, Count={row['stock_count']:.1f}")
        
        best_return_bucket = bucket_performance['annualized_return'].idxmax()
        best_sharpe_bucket = bucket_performance['sharpe_ratio'].idxmax()
        
        print(f"    Best performing bucket (return): {best_return_bucket}")
        print(f"    Best performing bucket (Sharpe): {best_sharpe_bucket}")
        
        # Compare with concentration if we have distribution data
        if available_bucket_cols:
            most_concentrated = avg_distribution.idxmax()
            if best_return_bucket == most_concentrated or best_sharpe_bucket == most_concentrated:
                print(f"    ✅ High-scoring stocks are concentrated in high-performing buckets!")
            else:
                print(f"    ⚠️ High-scoring stocks are NOT in the best-performing buckets")
    else:
        print("❌ No performance data available")


def main():
    """Main analysis function."""
    try:
        # Load data
        data = load_data()
        factor_data = data['factor_data']
        volume_data = data['volume_data']
        adtv = data['adtv']
        
        # Debug: Check data structure
        print(f"\n🔍 DATA STRUCTURE DEBUG:")
        print(f"    Factor data shape: {factor_data.shape}")
        print(f"    Factor data date range: {factor_data['calculation_date'].min()} to {factor_data['calculation_date'].max()}")
        print(f"    Volume data shape: {volume_data.shape}")
        print(f"    ADTV shape: {adtv.shape}")
        print(f"    ADTV date range: {adtv.index.min()} to {adtv.index.max()}")
        
        # Select analysis dates (quarterly rebalancing)
        analysis_dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='Q')
        
        print(f"    Analysis period: {len(analysis_dates)} quarterly dates")
        print(f"    Analysis dates: {analysis_dates.tolist()}")
        
        # Check if we have data for these dates
        factor_dates = pd.to_datetime(factor_data['calculation_date']).dt.date
        adtv_dates = pd.Series(adtv.index.date)
        
        print(f"    Factor data has dates: {len(factor_dates.unique())} unique dates")
        print(f"    ADTV data has dates: {len(adtv_dates.unique())} unique dates")
        
        # Check overlap
        analysis_dates_set = set(analysis_dates.date)
        factor_dates_set = set(factor_dates)
        adtv_dates_set = set(adtv_dates)
        
        factor_overlap = analysis_dates_set.intersection(factor_dates_set)
        adtv_overlap = analysis_dates_set.intersection(adtv_dates_set)
        
        print(f"    Analysis dates in factor data: {len(factor_overlap)}")
        print(f"    Analysis dates in ADTV data: {len(adtv_overlap)}")
        
        # Debug: Check some actual dates
        print(f"    Sample factor dates: {list(factor_dates.unique())[:5]}")
        print(f"    Sample analysis dates: {list(analysis_dates_set)[:5]}")
        
        # Use actual dates from factor data (monthly frequency for manageable analysis)
        print("    ⚠️ Using actual dates from factor data for analysis...")
        factor_dates_series = pd.to_datetime(factor_data['calculation_date'])
        
        # Get unique dates and filter to recent years
        available_dates = factor_dates_series.unique()
        available_dates = pd.to_datetime(available_dates)
        available_dates = available_dates[available_dates >= pd.to_datetime('2020-01-01')]
        available_dates = available_dates[available_dates <= pd.to_datetime('2023-12-31')]
        
        # Sample every 30 days to get manageable number of dates
        analysis_dates = available_dates[::30]  # Every 30th date
        
        print(f"    Using {len(analysis_dates)} sampled dates from 2020-2023")
        print(f"    Date range: {analysis_dates.min().date()} to {analysis_dates.max().date()}")
        print(f"    Sample dates: {analysis_dates[:5].date.tolist()}")
        
        # Convert factor data dates to datetime for proper comparison
        factor_data['calculation_date'] = pd.to_datetime(factor_data['calculation_date'])
        
        # Run analysis
        distribution_df = analyze_high_scoring_distribution(factor_data, adtv, analysis_dates)
        performance_df = calculate_returns_performance(factor_data, volume_data, adtv, analysis_dates)
        
        # Create visualizations
        create_visualizations(distribution_df, performance_df)
        
        # Generate insights
        generate_insights(distribution_df, performance_df)
        
        # Save results
        results = {
            'distribution_analysis': distribution_df,
            'performance_analysis': performance_df,
            'metadata': {
                'analysis_date': datetime.now(),
                'analysis_period': f"{analysis_dates.min().date()} to {analysis_dates.max().date()}",
                'total_dates': len(analysis_dates)
            }
        }
        
        results_path = Path(__file__).parent / "data" / "high_scoring_stocks_analysis_results.pkl"
        with open(results_path, "wb") as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"\n💾 Results saved to: {results_path}")
        print(f"✅ High-scoring stocks analysis complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()