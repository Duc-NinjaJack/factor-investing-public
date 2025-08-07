#!/usr/bin/env python3

# %% [markdown]
# # Raw Metrics Analysis for Recent Dates (v3)
# 
# This script analyzes raw metrics (ROA, P/E) for stocks on recent dates since fundamental data is only available from May 2025.

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

print("✅ Raw metrics analysis script v3 initialized")

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

def get_recent_dates():
    """Get recent dates for analysis (fundamental data available from May 2025)."""
    return [
        '2025-05-15',  # Mid-May 2025
        '2025-05-20',  # Late May 2025
        '2025-05-25',  # End May 2025
        '2025-05-30',  # Early June 2025
        '2025-06-05',  # Mid June 2025
        '2025-06-10'   # Recent June 2025
    ]

def get_sample_stocks():
    """Get sample stocks for analysis."""
    return [
        'HPG', 'VNM', 'VCB', 'TCB', 'FPT', 'MWG', 'MSN', 'VIC', 'VHM', 'GAS',
        'CTG', 'BID', 'ACB', 'MBB', 'STB', 'TPB', 'SHB', 'EIB', 'LPB', 'HDB'
    ]

def load_raw_metrics(db_engine, date, tickers):
    """Load raw metrics for given date and tickers."""
    print(f"📊 Loading raw metrics for {date}...")
    
    ticker_list = "', '".join(tickers)
    
    # Query for fundamental metrics from wong_api_daily_financial_info
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
        book_value_per_share,
        sales_per_share,
        eps,
        data_date
    FROM wong_api_daily_financial_info
    WHERE data_date = '{date}'
    AND ticker IN ('{ticker_list}')
    ORDER BY ticker
    """
    
    # Query for factor scores (check if available for recent dates)
    factor_query = f"""
    SELECT 
        ticker,
        Quality_Composite,
        Value_Composite,
        Momentum_Composite,
        QVM_Composite,
        strategy_version
    FROM factor_scores_qvm
    WHERE date = '{date}'
    AND ticker IN ('{ticker_list}')
    ORDER BY ticker
    """
    
    try:
        # Load fundamental data
        fundamental_data = pd.read_sql(fundamental_query, db_engine)
        
        # Load factor scores
        factor_data = pd.read_sql(factor_query, db_engine)
        
        print(f"   ✅ Loaded fundamental data for {len(fundamental_data)} stocks")
        print(f"   ✅ Loaded factor scores for {len(factor_data)} stocks")
        
        # Merge data
        if not fundamental_data.empty and not factor_data.empty:
            merged_data = fundamental_data.merge(factor_data, on='ticker', how='inner')
            return merged_data
        elif not fundamental_data.empty:
            print(f"   ⚠️ No factor scores available for {date}")
            return fundamental_data
        else:
            print(f"   ⚠️ No fundamental data available for {date}")
            return pd.DataFrame()
        
    except Exception as e:
        print(f"   ❌ Failed to load data: {e}")
        return pd.DataFrame()

def analyze_metrics_by_date(merged_data, date):
    """Analyze metrics for a specific date."""
    print(f"\n📅 Analysis for {date}")
    print("=" * 60)
    
    if merged_data.empty:
        print("   ❌ No data available")
        return
    
    print(f"   📊 Data summary: {len(merged_data)} stocks")
    
    # Basic statistics for key metrics
    key_metrics = ['roa', 'roe', 'pe_ratio', 'pb_ratio', 'ps_ratio', 'gross_margin', 'operating_margin', 'net_profit_margin']
    
    print(f"\n   📈 Key Metrics Statistics:")
    for metric in key_metrics:
        if metric in merged_data.columns:
            stats = merged_data[metric].describe()
            print(f"      {metric.upper()}:")
            print(f"        Mean: {stats['mean']:.4f}")
            print(f"        Std: {stats['std']:.4f}")
            print(f"        Min: {stats['min']:.4f}")
            print(f"        Max: {stats['max']:.4f}")
    
    # Show top and bottom stocks for key metrics
    print(f"\n   🏆 Top/Bottom Stocks Analysis:")
    
    # ROA analysis
    if 'roa' in merged_data.columns:
        print(f"\n      📊 ROA (Return on Assets):")
        top_roa = merged_data.nlargest(5, 'roa')[['ticker', 'roa']]
        bottom_roa = merged_data.nsmallest(5, 'roa')[['ticker', 'roa']]
        
        print(f"        Top 5: {top_roa.to_dict('records')}")
        print(f"        Bottom 5: {bottom_roa.to_dict('records')}")
    
    # P/E analysis
    if 'pe_ratio' in merged_data.columns:
        print(f"\n      📊 P/E Ratio:")
        # Filter out negative or extreme P/E values
        valid_pe = merged_data[merged_data['pe_ratio'] > 0].copy()
        if len(valid_pe) > 0:
            top_pe = valid_pe.nlargest(5, 'pe_ratio')[['ticker', 'pe_ratio']]
            bottom_pe = valid_pe.nsmallest(5, 'pe_ratio')[['ticker', 'pe_ratio']]
            
            print(f"        Highest P/E: {top_pe.to_dict('records')}")
            print(f"        Lowest P/E: {bottom_pe.to_dict('records')}")
    
    # Factor score analysis (if available)
    if 'Quality_Composite' in merged_data.columns:
        print(f"\n      📊 Factor Scores:")
        top_quality = merged_data.nlargest(5, 'Quality_Composite')[['ticker', 'Quality_Composite', 'roa', 'roe']]
        top_value = merged_data.nlargest(5, 'Value_Composite')[['ticker', 'Value_Composite', 'pe_ratio', 'pb_ratio']]
        top_momentum = merged_data.nlargest(5, 'Momentum_Composite')[['ticker', 'Momentum_Composite']]
        
        print(f"        Top Quality: {top_quality.to_dict('records')}")
        print(f"        Top Value: {top_value.to_dict('records')}")
        print(f"        Top Momentum: {top_momentum.to_dict('records')}")

def validate_factor_calculations(merged_data):
    """Validate factor calculations against raw metrics."""
    print(f"\n🔍 Factor Calculation Validation")
    print("=" * 60)
    
    if merged_data.empty:
        print("   ❌ No data available for validation")
        return
    
    # Quality factor validation
    print(f"\n   🔍 Quality Factor Validation:")
    if 'Quality_Composite' in merged_data.columns and 'roa' in merged_data.columns:
        # Check correlation between Quality factor and ROA
        quality_roa_corr = merged_data[['Quality_Composite', 'roa']].corr().iloc[0, 1]
        print(f"      Quality vs ROA correlation: {quality_roa_corr:.3f}")
        
        # Check if high quality stocks have high ROA
        high_quality = merged_data.nlargest(3, 'Quality_Composite')[['ticker', 'Quality_Composite', 'roa', 'roe']]
        low_quality = merged_data.nsmallest(3, 'Quality_Composite')[['ticker', 'Quality_Composite', 'roa', 'roe']]
        
        print(f"      High Quality stocks: {high_quality.to_dict('records')}")
        print(f"      Low Quality stocks: {low_quality.to_dict('records')}")
    
    # Value factor validation
    print(f"\n   🔍 Value Factor Validation:")
    if 'Value_Composite' in merged_data.columns and 'pe_ratio' in merged_data.columns:
        # Check correlation between Value factor and P/E (should be negative)
        valid_pe = merged_data[merged_data['pe_ratio'] > 0].copy()
        if len(valid_pe) > 0:
            value_pe_corr = valid_pe[['Value_Composite', 'pe_ratio']].corr().iloc[0, 1]
            print(f"      Value vs P/E correlation: {value_pe_corr:.3f} (should be negative)")
            
            # Check if high value stocks have low P/E
            high_value = merged_data.nlargest(3, 'Value_Composite')[['ticker', 'Value_Composite', 'pe_ratio', 'pb_ratio']]
            low_value = merged_data.nsmallest(3, 'Value_Composite')[['ticker', 'Value_Composite', 'pe_ratio', 'pb_ratio']]
            
            print(f"      High Value stocks: {high_value.to_dict('records')}")
            print(f"      Low Value stocks: {low_value.to_dict('records')}")

def create_summary_report(all_data):
    """Create a summary report across all dates."""
    print(f"\n📋 Summary Report Across All Dates")
    print("=" * 60)
    
    if not all_data:
        print("   ❌ No data available")
        return
    
    # Collect summary statistics
    summary_stats = []
    
    for date, data in all_data.items():
        if not data.empty:
            stats = {
                'date': date,
                'num_stocks': len(data),
                'avg_roa': data['roa'].mean() if 'roa' in data.columns else None,
                'avg_roe': data['roe'].mean() if 'roe' in data.columns else None,
                'avg_pe': data['pe_ratio'].mean() if 'pe_ratio' in data.columns else None,
                'avg_pb': data['pb_ratio'].mean() if 'pb_ratio' in data.columns else None,
                'avg_quality': data['Quality_Composite'].mean() if 'Quality_Composite' in data.columns else None,
                'avg_value': data['Value_Composite'].mean() if 'Value_Composite' in data.columns else None,
                'avg_momentum': data['Momentum_Composite'].mean() if 'Momentum_Composite' in data.columns else None
            }
            summary_stats.append(stats)
    
    if summary_stats:
        summary_df = pd.DataFrame(summary_stats)
        print(f"   📊 Summary Statistics:")
        print(summary_df.round(4))
        
        # Save summary to file
        results_dir = Path("insights")
        results_dir.mkdir(exist_ok=True)
        summary_df.to_csv(results_dir / "raw_metrics_summary_v3.csv", index=False)
        print(f"\n   ✅ Summary saved to {results_dir}/raw_metrics_summary_v3.csv")

# %%
def main():
    """Main analysis function."""
    print("🔍 Starting Raw Metrics Analysis v3 (Recent Dates)")
    print("=" * 60)
    
    try:
        # Create database connection
        db_engine = create_db_connection()
        
        # Get analysis parameters
        dates = get_recent_dates()
        tickers = get_sample_stocks()
        
        print(f"📊 Analysis parameters:")
        print(f"   Dates: {dates}")
        print(f"   Stocks: {len(tickers)} stocks")
        print(f"   Note: Using recent dates (May-June 2025) due to fundamental data availability")
        
        # Load and analyze data for each date
        all_data = {}
        
        for date in dates:
            print(f"\n{'='*60}")
            data = load_raw_metrics(db_engine, date, tickers)
            
            if not data.empty:
                all_data[date] = data
                analyze_metrics_by_date(data, date)
                validate_factor_calculations(data)
            else:
                print(f"   ⚠️ No data available for {date}")
        
        # Create summary report
        create_summary_report(all_data)
        
        print(f"\n✅ Analysis completed")
        print(f"📊 Key findings:")
        print(f"   - Analyzed {len(all_data)} dates")
        print(f"   - Total data points: {sum(len(data) for data in all_data.values())}")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
