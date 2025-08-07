#!/usr/bin/env python3

# %% [markdown]
# # Factor Calculation Investigation
# 
# This script investigates how the composite factors are calculated by sampling real data from the database.

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

print("✅ Factor calculation investigation script initialized")

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

def sample_factor_data(db_engine, sample_date='2022-06-30', sample_tickers=None):
    """Sample factor data for investigation."""
    print(f"📊 Sampling factor data for {sample_date}...")
    
    if sample_tickers is None:
        sample_tickers = ['HPG', 'VNM', 'VCB', 'TCB', 'FPT', 'MWG', 'MSN', 'VIC', 'VHM', 'GAS']
    
    ticker_list = "', '".join(sample_tickers)
    
    # Get factor scores
    factor_query = f"""
    SELECT 
        date,
        ticker,
        Quality_Composite,
        Value_Composite,
        Momentum_Composite,
        QVM_Composite,
        strategy_version
    FROM factor_scores_qvm
    WHERE date = '{sample_date}'
    AND ticker IN ('{ticker_list}')
    ORDER BY ticker
    """
    
    # Get fundamental data for validation
    fundamental_query = f"""
    SELECT 
        ticker,
        roaa,
        roe,
        pe_ratio,
        pb_ratio,
        debt_to_equity,
        current_ratio,
        gross_margin,
        net_margin,
        market_cap
    FROM fundamental_data
    WHERE date = '{sample_date}'
    AND ticker IN ('{ticker_list}')
    ORDER BY ticker
    """
    
    # Get price data for momentum validation
    price_query = f"""
    SELECT 
        trading_date as date,
        ticker,
        close_price_adjusted as close_price,
        total_volume as volume
    FROM vcsc_daily_data_complete
    WHERE ticker IN ('{ticker_list}')
    AND trading_date BETWEEN DATE_SUB('{sample_date}', INTERVAL 365 DAY) AND '{sample_date}'
    ORDER BY ticker, trading_date
    """
    
    try:
        factor_data = pd.read_sql(factor_query, db_engine)
        factor_data['date'] = pd.to_datetime(factor_data['date'])
        
        # Try to load fundamental data
        try:
            fundamental_data = pd.read_sql(fundamental_query, db_engine)
        except:
            print("   ⚠️ Fundamental data not available")
            fundamental_data = pd.DataFrame()
        
        # Load price data
        price_data = pd.read_sql(price_query, db_engine)
        price_data['date'] = pd.to_datetime(price_data['date'])
        
        print(f"   ✅ Loaded factor data for {len(factor_data)} stocks")
        if not fundamental_data.empty:
            print(f"   ✅ Loaded fundamental data for {len(fundamental_data)} stocks")
        print(f"   ✅ Loaded price data for {len(price_data)} records")
        
        return factor_data, fundamental_data, price_data
        
    except Exception as e:
        print(f"   ❌ Failed to load data: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def analyze_factor_distributions(factor_data):
    """Analyze factor distributions and check for issues."""
    print("🔍 Analyzing factor distributions...")
    
    print(f"   📊 Factor data summary:")
    print(f"      Total records: {len(factor_data)}")
    print(f"      Strategy version: {factor_data['strategy_version'].iloc[0] if len(factor_data) > 0 else 'Unknown'}")
    
    # Basic statistics for each factor
    for factor in ['Quality_Composite', 'Value_Composite', 'Momentum_Composite', 'QVM_Composite']:
        if factor in factor_data.columns:
            stats = factor_data[factor].describe()
            print(f"   📈 {factor}:")
            print(f"      Mean: {stats['mean']:.4f}")
            print(f"      Std: {stats['std']:.4f}")
            print(f"      Min: {stats['min']:.4f}")
            print(f"      Max: {stats['max']:.4f}")
            print(f"      Range: {stats['max'] - stats['min']:.4f}")
            
            # Check for extreme values
            extreme_high = factor_data.nlargest(3, factor)[['ticker', factor]]
            extreme_low = factor_data.nsmallest(3, factor)[['ticker', factor]]
            
            print(f"      Top 3: {extreme_high.to_dict('records')}")
            print(f"      Bottom 3: {extreme_low.to_dict('records')}")
    
    # Check factor correlations
    print(f"   📊 Factor correlations:")
    factor_cols = ['Quality_Composite', 'Value_Composite', 'Momentum_Composite', 'QVM_Composite']
    factor_cols = [col for col in factor_cols if col in factor_data.columns]
    
    if len(factor_cols) > 1:
        corr_matrix = factor_data[factor_cols].corr()
        print(corr_matrix.round(3))
    
    return factor_data

def validate_factor_calculations(factor_data, fundamental_data, price_data):
    """Validate factor calculations with common sense checks."""
    print("🔍 Validating factor calculations...")
    
    if fundamental_data.empty:
        print("   ⚠️ No fundamental data available for validation")
        return
    
    # Merge data for validation
    validation_data = factor_data.merge(fundamental_data, on='ticker', how='inner')
    
    if len(validation_data) == 0:
        print("   ⚠️ No overlapping data for validation")
        return
    
    print(f"   📊 Validation data: {len(validation_data)} stocks")
    
    # Quality factor validation
    print(f"   🔍 Quality factor validation:")
    if 'Quality_Composite' in validation_data.columns and 'roaa' in validation_data.columns:
        quality_roaa_corr = validation_data[['Quality_Composite', 'roaa']].corr().iloc[0, 1]
        print(f"      Quality vs ROAA correlation: {quality_roaa_corr:.3f}")
        
        # Check if high quality stocks have high ROAA
        high_quality = validation_data.nlargest(3, 'Quality_Composite')[['ticker', 'Quality_Composite', 'roaa']]
        low_quality = validation_data.nsmallest(3, 'Quality_Composite')[['ticker', 'Quality_Composite', 'roaa']]
        
        print(f"      High quality stocks: {high_quality.to_dict('records')}")
        print(f"      Low quality stocks: {low_quality.to_dict('records')}")
    
    # Value factor validation
    print(f"   🔍 Value factor validation:")
    if 'Value_Composite' in validation_data.columns and 'pe_ratio' in validation_data.columns:
        value_pe_corr = validation_data[['Value_Composite', 'pe_ratio']].corr().iloc[0, 1]
        print(f"      Value vs P/E correlation: {value_pe_corr:.3f}")
        
        # Check if high value stocks have low P/E
        high_value = validation_data.nlargest(3, 'Value_Composite')[['ticker', 'Value_Composite', 'pe_ratio']]
        low_value = validation_data.nsmallest(3, 'Value_Composite')[['ticker', 'Value_Composite', 'pe_ratio']]
        
        print(f"      High value stocks: {high_value.to_dict('records')}")
        print(f"      Low value stocks: {low_value.to_dict('records')}")
    
    # Momentum factor validation
    print(f"   🔍 Momentum factor validation:")
    if 'Momentum_Composite' in validation_data.columns and not price_data.empty:
        # Calculate simple momentum for validation
        momentum_validation = []
        
        for ticker in validation_data['ticker']:
            ticker_prices = price_data[price_data['ticker'] == ticker].copy()
            if len(ticker_prices) > 20:  # Need at least 20 days
                ticker_prices = ticker_prices.sort_values('date')
                # Calculate 3-month momentum
                start_price = ticker_prices.iloc[-63]['close_price'] if len(ticker_prices) >= 63 else ticker_prices.iloc[0]['close_price']
                end_price = ticker_prices.iloc[-1]['close_price']
                simple_momentum = (end_price / start_price) - 1
                
                momentum_validation.append({
                    'ticker': ticker,
                    'simple_momentum': simple_momentum,
                    'factor_momentum': validation_data[validation_data['ticker'] == ticker]['Momentum_Composite'].iloc[0]
                })
        
        if momentum_validation:
            momentum_df = pd.DataFrame(momentum_validation)
            momentum_corr = momentum_df[['simple_momentum', 'factor_momentum']].corr().iloc[0, 1]
            print(f"      Simple momentum vs Factor momentum correlation: {momentum_corr:.3f}")
            
            # Show examples
            high_momentum = momentum_df.nlargest(3, 'factor_momentum')[['ticker', 'simple_momentum', 'factor_momentum']]
            low_momentum = momentum_df.nsmallest(3, 'factor_momentum')[['ticker', 'simple_momentum', 'factor_momentum']]
            
            print(f"      High momentum stocks: {high_momentum.to_dict('records')}")
            print(f"      Low momentum stocks: {low_momentum.to_dict('records')}")

def check_qvm_composite_calculation(factor_data):
    """Check if QVM_Composite is calculated correctly from components."""
    print("🔍 Checking QVM_Composite calculation...")
    
    if len(factor_data) == 0:
        print("   ⚠️ No factor data available")
        return
    
    # Check if all required columns exist
    required_cols = ['Quality_Composite', 'Value_Composite', 'Momentum_Composite', 'QVM_Composite']
    missing_cols = [col for col in required_cols if col not in factor_data.columns]
    
    if missing_cols:
        print(f"   ❌ Missing columns: {missing_cols}")
        return
    
    # Calculate expected QVM_Composite using standard weights
    # Based on the documentation: QVM_Composite = 0.40 * Quality + 0.30 * Value + 0.30 * Momentum
    factor_data['expected_qvm'] = (
        0.40 * factor_data['Quality_Composite'] +
        0.30 * factor_data['Value_Composite'] +
        0.30 * factor_data['Momentum_Composite']
    )
    
    # Compare actual vs expected
    factor_data['qvm_difference'] = factor_data['QVM_Composite'] - factor_data['expected_qvm']
    
    print(f"   📊 QVM_Composite calculation check:")
    print(f"      Average difference: {factor_data['qvm_difference'].mean():.6f}")
    print(f"      Max difference: {factor_data['qvm_difference'].abs().max():.6f}")
    print(f"      Standard deviation of difference: {factor_data['qvm_difference'].std():.6f}")
    
    # Show examples
    large_diff = factor_data.nlargest(3, 'qvm_difference')[['ticker', 'Quality_Composite', 'Value_Composite', 'Momentum_Composite', 'QVM_Composite', 'expected_qvm', 'qvm_difference']]
    print(f"   📊 Largest positive differences: {large_diff.to_dict('records')}")
    
    large_neg_diff = factor_data.nsmallest(3, 'qvm_difference')[['ticker', 'Quality_Composite', 'Value_Composite', 'Momentum_Composite', 'QVM_Composite', 'expected_qvm', 'qvm_difference']]
    print(f"   📊 Largest negative differences: {large_neg_diff.to_dict('records')}")

def sample_multiple_dates(db_engine):
    """Sample multiple dates to check consistency."""
    print("📊 Sampling multiple dates for consistency check...")
    
    dates = ['2022-06-30', '2022-12-30', '2023-06-30', '2023-12-29']
    
    for date in dates:
        print(f"   📅 Checking {date}...")
        
        # Get factor data for this date
        factor_query = f"""
        SELECT 
            ticker,
            Quality_Composite,
            Value_Composite,
            Momentum_Composite,
            QVM_Composite
        FROM factor_scores_qvm
        WHERE date = '{date}'
        ORDER BY ticker
        LIMIT 10
        """
        
        try:
            factor_data = pd.read_sql(factor_query, db_engine)
            
            if len(factor_data) > 0:
                # Basic statistics
                for factor in ['Quality_Composite', 'Value_Composite', 'Momentum_Composite', 'QVM_Composite']:
                    if factor in factor_data.columns:
                        stats = factor_data[factor].describe()
                        print(f"      {factor}: Mean={stats['mean']:.3f}, Std={stats['std']:.3f}, Min={stats['min']:.3f}, Max={stats['max']:.3f}")
            else:
                print(f"      No data found for {date}")
                
        except Exception as e:
            print(f"      Error loading data for {date}: {e}")

# %%
def main():
    """Main investigation function."""
    print("🔍 Starting Factor Calculation Investigation")
    print("="*60)
    
    try:
        # Create database connection
        db_engine = create_db_connection()
        
        # Sample factor data
        factor_data, fundamental_data, price_data = sample_factor_data(db_engine)
        
        if factor_data.empty:
            print("❌ No factor data available")
            return
        
        # Analyze factor distributions
        factor_data = analyze_factor_distributions(factor_data)
        
        # Validate factor calculations
        validate_factor_calculations(factor_data, fundamental_data, price_data)
        
        # Check QVM_Composite calculation
        check_qvm_composite_calculation(factor_data)
        
        # Sample multiple dates
        sample_multiple_dates(db_engine)
        
        # Save results
        results_dir = Path("insights")
        results_dir.mkdir(exist_ok=True)
        
        factor_data.to_csv(results_dir / "factor_calculation_sample.csv", index=False)
        
        print(f"\n✅ Investigation completed and saved to {results_dir}/")
        print(f"📊 Key findings:")
        print(f"   - Analyzed {len(factor_data)} factor records")
        print(f"   - Strategy version: {factor_data['strategy_version'].iloc[0] if len(factor_data) > 0 else 'Unknown'}")
        
    except Exception as e:
        print(f"❌ Investigation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

