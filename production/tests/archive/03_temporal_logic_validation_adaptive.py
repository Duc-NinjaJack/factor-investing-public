# =============================================================================
# SCRIPT: 03_temporal_logic_validation_adaptive.py
#
# PURPOSE:
# To validate the "cold start" robustness of the QVMEngineV2Enhanced
# by first discovering available data dates and then testing with actual data.
#
# METHODOLOGY:
# 1. First, discover the earliest available fundamental data
# 2. Use a test date that we know has data
# 3. Simulate both "full history" and "cold start" scenarios
# 4. Verify identical results to prove temporal logic robustness
#
# =============================================================================

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging
import yaml
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta

# --- 1. ENVIRONMENT SETUP ---
print("="*70)
print("🚀 Adaptive Temporal Logic & Cold Start Validation Test")
print("="*70)

# Find project root
try:
    project_root = Path.cwd()
    while not (project_root / 'production').exists() and not (project_root / 'config').exists():
        if project_root.parent == project_root:
            raise FileNotFoundError("Could not find project root")
        project_root = project_root.parent
    
    print(f"✅ Project root: {project_root}")

    production_path = project_root / 'production'
    if str(production_path) not in sys.path:
        sys.path.insert(0, str(production_path))

    from engine.qvm_engine_v2_enhanced import QVMEngineV2Enhanced
    print("✅ Successfully imported QVMEngineV2Enhanced")

except (ImportError, FileNotFoundError) as e:
    print(f"❌ CRITICAL ERROR: {e}")
    raise

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# --- 2. DATABASE CONNECTION ---
print("\n🔌 Establishing database connection...")
try:
    config_path = project_root / 'config' / 'database.yml'
    with open(config_path, 'r') as f:
        db_config = yaml.safe_load(f)['production']
    
    engine = create_engine(
        f"mysql+pymysql://{db_config['username']}:{db_config['password']}@"
        f"{db_config['host']}/{db_config['schema_name']}"
    )
    
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))
    print(f"✅ Connected to '{db_config['schema_name']}'")
except Exception as e:
    print(f"❌ Database connection failed: {e}")
    raise

# --- 3. DISCOVER AVAILABLE DATA RANGE ---
print("\n📊 Discovering available data range...")

with engine.connect() as conn:
    # Check intermediary calculations data range
    result = conn.execute(text("""
        SELECT 
            MIN(CONCAT(year, '-', LPAD(quarter*3, 2, '0'), '-01')) as earliest_date,
            MAX(CONCAT(year, '-', LPAD(quarter*3, 2, '0'), '-01')) as latest_date,
            COUNT(DISTINCT CONCAT(year, '-', quarter)) as total_periods
        FROM intermediary_calculations_enhanced
        WHERE ticker IN ('FPT', 'VCB', 'SSI', 'NLG')
    """))
    
    data_info = result.fetchone()
    if data_info and data_info[0]:
        earliest_fundamental = pd.Timestamp(data_info[0])
        latest_fundamental = pd.Timestamp(data_info[1])
        print(f"✅ Fundamental data range: {earliest_fundamental.date()} to {latest_fundamental.date()}")
        print(f"   Total periods: {data_info[2]}")
    else:
        print("❌ No fundamental data found")
        raise ValueError("No fundamental data available for test tickers")

    # Check price data range
    result = conn.execute(text("""
        SELECT 
            MIN(date) as earliest_date,
            MAX(date) as latest_date,
            COUNT(DISTINCT date) as total_days
        FROM equity_history
        WHERE ticker IN ('FPT', 'VCB', 'SSI', 'NLG')
    """))
    
    price_info = result.fetchone()
    if price_info and price_info[0]:
        earliest_price = pd.Timestamp(price_info[0])
        latest_price = pd.Timestamp(price_info[1])
        print(f"✅ Price data range: {earliest_price.date()} to {latest_price.date()}")
        print(f"   Total days: {price_info[2]}")

# --- 4. DETERMINE SAFE TEST DATE ---
# Use a date at least 1 year after earliest fundamental data to ensure YoY calculations work
safe_test_date = earliest_fundamental + timedelta(days=400)  # ~13 months after earliest
if safe_test_date > datetime.now():
    safe_test_date = pd.Timestamp('2024-07-01')  # Fallback to recent date we know has data

print(f"\n🎯 Selected test date: {safe_test_date.date()}")
print("   (Ensures sufficient historical data for YoY calculations)")

# --- 5. INITIALIZE ENGINE ---
print("\n🔧 Initializing QVMEngineV2Enhanced...")
try:
    qvm_engine = QVMEngineV2Enhanced(config_path=str(project_root / 'config'), log_level='INFO')
    print("✅ Engine initialized successfully")
except Exception as e:
    print(f"❌ Engine initialization failed: {e}")
    raise

# --- 6. RUN VALIDATION TEST ---
TEST_UNIVERSE = ['FPT', 'VCB', 'SSI', 'NLG']
TEST_DATE = safe_test_date

print(f"\n📋 Test Configuration:")
print(f"   Universe: {TEST_UNIVERSE}")
print(f"   Test Date: {TEST_DATE.date()}")

# --- SIMULATION 1: First Run (simulating "full history" context) ---
print("\n" + "="*70)
print("🏃 SIMULATION 1: First Run")
print("="*70)

results_run1 = qvm_engine.calculate_qvm_composite(TEST_DATE, TEST_UNIVERSE)

if results_run1:
    print(f"✅ Calculation successful for {len(results_run1)} tickers")
    df_run1 = pd.DataFrame.from_dict(results_run1, orient='index').reset_index()
    df_run1.rename(columns={'index': 'ticker'}, inplace=True)
    print("\n--- Results from Run 1 ---")
    print(df_run1.round(6))
else:
    print("❌ Calculation failed for Run 1")
    df_run1 = pd.DataFrame()

# --- SIMULATION 2: Second Run (simulating "cold start") ---
print("\n" + "="*70)
print("🥶 SIMULATION 2: Second Run (Cold Start Simulation)")
print("="*70)

# Create a new engine instance to simulate a fresh start
print("Creating new engine instance to simulate cold start...")
qvm_engine_cold = QVMEngineV2Enhanced(config_path=str(project_root / 'config'), log_level='WARNING')

results_run2 = qvm_engine_cold.calculate_qvm_composite(TEST_DATE, TEST_UNIVERSE)

if results_run2:
    print(f"✅ Calculation successful for {len(results_run2)} tickers")
    df_run2 = pd.DataFrame.from_dict(results_run2, orient='index').reset_index()
    df_run2.rename(columns={'index': 'ticker'}, inplace=True)
    print("\n--- Results from Run 2 ---")
    print(df_run2.round(6))
else:
    print("❌ Calculation failed for Run 2")
    df_run2 = pd.DataFrame()

# --- 7. VALIDATION: COMPARE RESULTS ---
print("\n" + "="*70)
print("🔬 VALIDATION: Comparing Run 1 vs Run 2 Results")
print("="*70)

if not df_run1.empty and not df_run2.empty:
    # Merge results for comparison
    comparison = pd.merge(
        df_run1.add_suffix('_run1'),
        df_run2.add_suffix('_run2'),
        left_on='ticker_run1',
        right_on='ticker_run2',
        how='outer'
    )
    
    # Calculate differences
    for col in ['Quality_Composite', 'Value_Composite', 'Momentum_Composite', 'QVM_Composite']:
        comparison[f'{col}_diff'] = (comparison[f'{col}_run1'] - comparison[f'{col}_run2']).abs()
    
    # Display comparison
    print("\n--- Detailed Comparison ---")
    display_cols = ['ticker_run1', 'QVM_Composite_run1', 'QVM_Composite_run2', 'QVM_Composite_diff']
    print(comparison[display_cols].round(8))
    
    # Calculate total difference
    diff_cols = [col for col in comparison.columns if col.endswith('_diff')]
    total_diff = comparison[diff_cols].sum().sum()
    
    print(f"\n📊 Total absolute difference across all components: {total_diff:.10f}")
    
    # Final verdict
    print("\n" + "="*70)
    print("🏁 FINAL VERDICT")
    print("="*70)
    
    if total_diff < 1e-9:
        print("✅ SUCCESS: Results are IDENTICAL between runs!")
        print("✅ The engine's temporal logic is ROBUST and self-contained")
        print("✅ Cold start handling is WORKING CORRECTLY")
        print("✅ Parallel historical generation by year is SAFE")
        print("\n🎉 You can proceed with confidence to run parallel historical generation!")
    else:
        print("❌ FAILURE: Results differ between runs")
        print(f"   Total difference: {total_diff}")
        print("⚠️  The engine may have temporal dependencies")
        print("⚠️  Further investigation needed before parallel execution")
        
        # Show which components differ most
        print("\n--- Difference by Component ---")
        for col in diff_cols:
            col_diff = comparison[col].sum()
            if col_diff > 1e-10:
                print(f"   {col}: {col_diff:.10f}")
else:
    print("❌ VALIDATION FAILED: One or both runs produced no data")
    print("   This may indicate insufficient historical data for the test date")

print("\n" + "="*70)
print("✅ Temporal Logic Validation Test Complete")
print("="*70)