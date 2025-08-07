#!/usr/bin/env python3

# %% [markdown]
# # Comprehensive Factor Analysis and Fixes
# 
# This script addresses the key issues identified:
# 1. Quality composite identical across dates → Check if stale
# 2. Value composite negative → Check calculation and rebase to 0-1 scale
# 3. Other metrics (F-Score, FCF Yield, Low Vol) → Check how they're factored in

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

print("✅ Comprehensive factor analysis script initialized")

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

def analyze_factor_staleness(db_engine):
    """Analyze if factors are stale (identical across dates)."""
    print("🔍 Analyzing Factor Staleness")
    print("=" * 60)
    
    try:
        with db_engine.connect() as conn:
            # Check if quality factors are identical across dates
            result = conn.execute(text("""
                SELECT 
                    ticker,
                    Quality_Composite,
                    Value_Composite,
                    Momentum_Composite,
                    date,
                    calculation_timestamp
                FROM factor_scores_qvm
                WHERE strategy_version = 'qvm_v2.0_enhanced'
                AND ticker IN ('HPG', 'VNM', 'VCB', 'TCB', 'FPT')
                AND date IN ('2025-05-30', '2025-06-05', '2025-06-10')
                ORDER BY ticker, date
            """))
            
            # Group by ticker to check consistency
            ticker_data = {}
            for row in result:
                ticker = row[0]
                if ticker not in ticker_data:
                    ticker_data[ticker] = []
                ticker_data[ticker].append({
                    'date': row[4],
                    'quality': row[1],
                    'value': row[2],
                    'momentum': row[3],
                    'timestamp': row[5]
                })
            
            print("📊 Factor Consistency Analysis:")
            for ticker, data in ticker_data.items():
                print(f"\n{ticker}:")
                
                # Check quality factor consistency
                quality_values = [record['quality'] for record in data]
                quality_identical = len(set(quality_values)) == 1
                
                # Check value factor consistency
                value_values = [record['value'] for record in data]
                value_identical = len(set(value_values)) == 1
                
                # Check momentum factor consistency
                momentum_values = [record['momentum'] for record in data]
                momentum_identical = len(set(momentum_values)) == 1
                
                print(f"  Quality Factor: {'❌ IDENTICAL (STALE)' if quality_identical else '✅ VARYING (FRESH)'}")
                print(f"  Value Factor: {'❌ IDENTICAL (STALE)' if value_identical else '✅ VARYING (FRESH)'}")
                print(f"  Momentum Factor: {'❌ IDENTICAL (STALE)' if momentum_identical else '✅ VARYING (FRESH)'}")
                
                # Show values
                for record in data:
                    print(f"    {record['date']}: Q={record['quality']:.6f}, V={record['value']:.6f}, M={record['momentum']:.6f}")
            
            # Check generation timestamps
            result = conn.execute(text("""
                SELECT 
                    MAX(calculation_timestamp) as last_generated,
                    COUNT(DISTINCT calculation_timestamp) as unique_timestamps,
                    COUNT(DISTINCT date) as unique_dates
                FROM factor_scores_qvm
                WHERE strategy_version = 'qvm_v2.0_enhanced'
            """))
            
            row = result.fetchone()
            print(f"\n🕒 Factor Generation Status:")
            print(f"  Last Generated: {row[0]}")
            print(f"  Unique Timestamps: {row[1]}")
            print(f"  Unique Dates: {row[2]}")
            
            if row[1] == 1:
                print(f"  ⚠️ WARNING: All factors generated at same time - may be stale!")
            elif row[1] < row[2]:
                print(f"  ⚠️ WARNING: Fewer timestamps than dates - some factors may be stale!")
            else:
                print(f"  ✅ Factors appear to be generated properly")
                
    except Exception as e:
        print(f"❌ Error analyzing factor staleness: {e}")

def analyze_value_factor_issues(db_engine):
    """Analyze value factor calculation issues and propose fixes."""
    print("\n🔍 Analyzing Value Factor Issues")
    print("=" * 60)
    
    try:
        with db_engine.connect() as conn:
            # Get value factor statistics
            result = conn.execute(text("""
                SELECT 
                    AVG(Value_Composite) as avg_value,
                    MIN(Value_Composite) as min_value,
                    MAX(Value_Composite) as max_value,
                    STDDEV(Value_Composite) as std_value,
                    COUNT(*) as total_records
                FROM factor_scores_qvm
                WHERE strategy_version = 'qvm_v2.0_enhanced'
                AND date = '2025-06-10'
            """))
            
            row = result.fetchone()
            print(f"📊 Value Factor Statistics (2025-06-10):")
            print(f"  Average: {row[0]:.6f}")
            print(f"  Minimum: {row[1]:.6f}")
            print(f"  Maximum: {row[2]:.6f}")
            print(f"  Std Dev: {row[3]:.6f}")
            print(f"  Total Records: {row[4]}")
            
            # Check if average is negative
            if row[0] < 0:
                print(f"  ❌ PROBLEM: Average value factor is negative ({row[0]:.6f})")
                print(f"  🔧 SOLUTION: Value factors need to be rebased to 0-1 scale")
            else:
                print(f"  ✅ Value factor average is positive")
            
            # Get sample of value factors with P/E data
            result = conn.execute(text("""
                SELECT 
                    f.ticker,
                    f.Value_Composite,
                    w.pe_ratio,
                    w.pb_ratio,
                    w.ps_ratio
                FROM factor_scores_qvm f
                JOIN wong_api_daily_financial_info w ON f.ticker = w.ticker AND f.date = w.data_date
                WHERE f.strategy_version = 'qvm_v2.0_enhanced'
                AND f.date = '2025-06-10'
                AND w.pe_ratio > 0
                ORDER BY f.Value_Composite DESC
                LIMIT 10
            """))
            
            print(f"\n📊 Top Value Stocks (by Value_Composite):")
            for row in result:
                print(f"  {row[0]}: Value={row[1]:.6f}, P/E={row[2]:.2f}, P/B={row[3]:.2f}, P/S={row[4]:.2f}")
            
            # Check correlation
            result = conn.execute(text("""
                SELECT 
                    CORR(f.Value_Composite, w.pe_ratio) as value_pe_corr,
                    CORR(f.Value_Composite, w.pb_ratio) as value_pb_corr,
                    CORR(f.Value_Composite, w.ps_ratio) as value_ps_corr
                FROM factor_scores_qvm f
                JOIN wong_api_daily_financial_info w ON f.ticker = w.ticker AND f.date = w.data_date
                WHERE f.strategy_version = 'qvm_v2.0_enhanced'
                AND f.date = '2025-06-10'
                AND w.pe_ratio > 0
            """))
            
            row = result.fetchone()
            print(f"\n📊 Value Factor Correlations:")
            print(f"  Value vs P/E: {row[0]:.3f} (should be negative)")
            print(f"  Value vs P/B: {row[1]:.3f} (should be negative)")
            print(f"  Value vs P/S: {row[2]:.3f} (should be negative)")
            
            if row[0] > 0:
                print(f"  ❌ PROBLEM: Value factor positively correlated with P/E!")
            else:
                print(f"  ✅ Value factor correctly negatively correlated with P/E")
                
    except Exception as e:
        print(f"❌ Error analyzing value factor issues: {e}")

def check_other_metrics(db_engine):
    """Check if other metrics (F-Score, FCF Yield, Low Vol) are being used."""
    print("\n🔍 Checking Other Metrics Usage")
    print("=" * 60)
    
    try:
        with db_engine.connect() as conn:
            # Check if F-Score data exists
            result = conn.execute(text("""
                SELECT 
                    COUNT(*) as total_records,
                    COUNT(DISTINCT ticker) as unique_tickers,
                    COUNT(DISTINCT period_end_date) as unique_dates
                FROM precalculated_metrics
                WHERE metric_name LIKE '%f_score%'
                AND period_end_date >= '2025-01-01'
            """))
            
            row = result.fetchone()
            print(f"📊 F-Score Data Availability:")
            print(f"  Total Records: {row[0]}")
            print(f"  Unique Tickers: {row[1]}")
            print(f"  Unique Dates: {row[2]}")
            
            if row[0] > 0:
                print(f"  ✅ F-Score data is available")
                
                # Check sample F-Score values
                result = conn.execute(text("""
                    SELECT 
                        ticker,
                        metric_name,
                        value,
                        period_end_date
                    FROM precalculated_metrics
                    WHERE metric_name LIKE '%f_score%'
                    AND period_end_date >= '2025-01-01'
                    ORDER BY period_end_date DESC, ticker
                    LIMIT 10
                """))
                
                print(f"\n📊 Sample F-Score Data:")
                for row in result:
                    print(f"  {row[0]}: {row[1]} = {row[2]} ({row[3]})")
            else:
                print(f"  ❌ No F-Score data available")
            
            # Check if FCF Yield data exists
            result = conn.execute(text("""
                SELECT 
                    COUNT(*) as total_records,
                    COUNT(DISTINCT ticker) as unique_tickers,
                    COUNT(DISTINCT period_end_date) as unique_dates
                FROM precalculated_metrics
                WHERE metric_name LIKE '%fcf%' OR metric_name LIKE '%free_cash_flow%'
                AND period_end_date >= '2025-01-01'
            """))
            
            row = result.fetchone()
            print(f"\n📊 FCF Yield Data Availability:")
            print(f"  Total Records: {row[0]}")
            print(f"  Unique Tickers: {row[1]}")
            print(f"  Unique Dates: {row[2]}")
            
            if row[0] > 0:
                print(f"  ✅ FCF Yield data is available")
            else:
                print(f"  ❌ No FCF Yield data available")
            
            # Check if Low Vol data exists
            result = conn.execute(text("""
                SELECT 
                    COUNT(*) as total_records,
                    COUNT(DISTINCT ticker) as unique_tickers,
                    COUNT(DISTINCT trading_date) as unique_dates
                FROM vcsc_daily_data_complete
                WHERE trading_date >= '2025-01-01'
                AND total_volume > 0
            """))
            
            row = result.fetchone()
            print(f"\n📊 Low Volatility Data Availability:")
            print(f"  Total Records: {row[0]}")
            print(f"  Unique Tickers: {row[1]}")
            print(f"  Unique Dates: {row[2]}")
            
            if row[0] > 0:
                print(f"  ✅ Price/volume data available for low vol calculation")
            else:
                print(f"  ❌ No price/volume data available")
                
    except Exception as e:
        print(f"❌ Error checking other metrics: {e}")

def propose_fixes():
    """Propose fixes for the identified issues."""
    print("\n🔧 Proposed Fixes")
    print("=" * 60)
    
    print("1. 🕒 Factor Staleness Issue:")
    print("   PROBLEM: Quality factors identical across dates")
    print("   SOLUTION:")
    print("   - Regenerate factors for recent dates")
    print("   - Implement daily factor generation pipeline")
    print("   - Add timestamp validation checks")
    
    print("\n2. 📊 Value Factor Negative Issue:")
    print("   PROBLEM: Value factors consistently negative")
    print("   SOLUTION:")
    print("   - Rebase value factors to 0-1 scale using min-max normalization")
    print("   - Use ranking-based approach: rank stocks by P/E, P/B, P/S")
    print("   - Convert to percentile scores (0-100) then normalize to 0-1")
    print("   - Formula: normalized_value = (rank - 1) / (total_stocks - 1)")
    
    print("\n3. 🎯 Other Metrics Integration:")
    print("   PROBLEM: F-Score, FCF Yield, Low Vol not being used")
    print("   SOLUTION:")
    print("   - Integrate F-Score into Quality factor (50% ROAA + 50% F-Score)")
    print("   - Integrate FCF Yield into Value factor (50% P/E + 50% FCF Yield)")
    print("   - Integrate Low Vol into Momentum factor (50% Momentum + 50% Low Vol)")
    print("   - Use equal weighting within each factor category")
    
    print("\n4. 🔄 Factor Generation Pipeline:")
    print("   - Implement daily factor generation")
    print("   - Add data quality checks")
    print("   - Implement factor validation tests")
    print("   - Add monitoring and alerts")

def create_fixed_factor_calculation():
    """Create a fixed factor calculation function."""
    print("\n📝 Fixed Factor Calculation Code")
    print("=" * 60)
    
    code = '''
def calculate_fixed_factors(data_df, config):
    """
    Calculate fixed factors with proper normalization and additional metrics.
    
    Args:
        data_df: DataFrame with fundamental and market data
        config: Configuration dictionary
    
    Returns:
        DataFrame with fixed factor scores
    """
    import pandas as pd
    import numpy as np
    
    # 1. Quality Factor (ROAA + F-Score)
    print("📊 Calculating Quality Factor...")
    
    # ROAA component (0-1 scale)
    if 'roaa' in data_df.columns:
        data_df['roaa_rank'] = data_df['roaa'].rank(ascending=True, method='min')
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
        data_df['roaa_normalized'] * 0.5 +
        data_df['fscore_normalized'] * 0.5
    )
    
    # 2. Value Factor (P/E + FCF Yield)
    print("📊 Calculating Value Factor...")
    
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
        data_df['pe_normalized'] * 0.5 +
        data_df['fcf_normalized'] * 0.5
    )
    
    # 3. Momentum Factor (Momentum + Low Vol)
    print("📊 Calculating Momentum Factor...")
    
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
        data_df['momentum_normalized'] * 0.5 +
        data_df['lowvol_normalized'] * 0.5
    )
    
    # 4. Final QVM Composite
    print("📊 Calculating QVM Composite...")
    data_df['QVM_Composite'] = (
        data_df['Quality_Composite'] * config.get('quality_weight', 0.4) +
        data_df['Value_Composite'] * config.get('value_weight', 0.3) +
        data_df['Momentum_Composite'] * config.get('momentum_weight', 0.3)
    )
    
    print(f"✅ Fixed factors calculated for {len(data_df)} stocks")
    print(f"📊 Factor ranges:")
    print(f"   Quality: {data_df['Quality_Composite'].min():.3f} to {data_df['Quality_Composite'].max():.3f}")
    print(f"   Value: {data_df['Value_Composite'].min():.3f} to {data_df['Value_Composite'].max():.3f}")
    print(f"   Momentum: {data_df['Momentum_Composite'].min():.3f} to {data_df['Momentum_Composite'].max():.3f}")
    print(f"   QVM: {data_df['QVM_Composite'].min():.3f} to {data_df['QVM_Composite'].max():.3f}")
    
    return data_df
'''
    
    print(code)

# %%
def main():
    """Main analysis function."""
    print("🔍 Starting Comprehensive Factor Analysis")
    print("=" * 80)
    
    try:
        # Create database connection
        db_engine = create_db_connection()
        
        # Analyze factor staleness
        analyze_factor_staleness(db_engine)
        
        # Analyze value factor issues
        analyze_value_factor_issues(db_engine)
        
        # Check other metrics
        check_other_metrics(db_engine)
        
        # Propose fixes
        propose_fixes()
        
        # Create fixed factor calculation
        create_fixed_factor_calculation()
        
        print(f"\n✅ Comprehensive analysis completed")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
