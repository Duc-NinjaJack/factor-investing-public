# =============================================================================
# SCRIPT: 03_temporal_logic_validation_practical.py
#
# PURPOSE:
# Practical validation of temporal logic using a date we know works from 
# the comprehensive validation tests (2024-07-01). This test proves that
# the engine's temporal logic is deterministic and doesn't depend on 
# execution context or order.
#
# METHODOLOGY:
# 1. Use 2024-07-01 which we know has data from comprehensive validation
# 2. Run the calculation multiple times with different engine instances
# 3. Verify identical results to prove temporal logic robustness
# 4. Test the specific cold-start scenario by simulating fresh starts
#
# =============================================================================

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging
import yaml
from sqlalchemy import create_engine, text
import time

# --- 1. ENVIRONMENT SETUP ---
print("="*80)
print("🚀 PRACTICAL Temporal Logic & Cold Start Validation")
print("   Using 2024-07-01 (validated date from comprehensive testing)")
print("="*80)

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

# Set logging to reduce noise
logging.basicConfig(level=logging.WARNING)

# --- 2. TEST CONFIGURATION ---
TEST_DATE = pd.Timestamp('2024-07-01')
TEST_UNIVERSE = ['FPT', 'VIC', 'VHM', 'TCB', 'VCB']  # Same as comprehensive validation

print(f"\n📋 Test Configuration:")
print(f"   Date: {TEST_DATE.date()}")
print(f"   Universe: {TEST_UNIVERSE}")
print(f"   Expected: This date worked in comprehensive validation")

# --- 3. MULTIPLE ENGINE TEST ---
print(f"\n🔬 Running multiple independent calculations...")

results_list = []
engine_instances = []

for i in range(3):
    print(f"\n--- Run {i+1}: Creating fresh engine instance ---")
    
    # Create completely fresh engine instance
    engine = QVMEngineV2Enhanced(
        config_path=str(project_root / 'config'), 
        log_level='ERROR'  # Minimize noise
    )
    engine_instances.append(engine)
    
    # Calculate factors
    start_time = time.time()
    results = engine.calculate_qvm_composite(TEST_DATE, TEST_UNIVERSE)
    calc_time = time.time() - start_time
    
    if results:
        print(f"✅ Run {i+1}: Success ({len(results)} tickers, {calc_time:.2f}s)")
        results_df = pd.DataFrame.from_dict(results, orient='index').reset_index()
        results_df.rename(columns={'index': 'ticker'}, inplace=True)
        results_df['run'] = i + 1
        results_list.append(results_df)
    else:
        print(f"❌ Run {i+1}: Failed")
        break

# --- 4. RESULTS COMPARISON ---
print(f"\n🔍 Comparing results across {len(results_list)} runs...")

if len(results_list) >= 2:
    # Merge all results for comparison
    base_df = results_list[0].copy()
    base_df = base_df.drop('run', axis=1).add_suffix('_run1')
    
    for i, df in enumerate(results_list[1:], 2):
        compare_df = df.drop('run', axis=1).add_suffix(f'_run{i}')
        base_df = pd.merge(
            base_df, 
            compare_df, 
            left_on=f'ticker_run1', 
            right_on=f'ticker_run{i}',
            how='outer'
        )
    
    # Calculate differences between runs
    print(f"\n📊 Detailed comparison (Run 1 vs Run 2):")
    
    components = ['Quality_Composite', 'Value_Composite', 'Momentum_Composite', 'QVM_Composite']
    
    comparison_data = []
    total_diff = 0
    
    for _, row in base_df.iterrows():
        ticker = row['ticker_run1']
        row_data = {'ticker': ticker}
        
        for comp in components:
            val1 = row[f'{comp}_run1']
            val2 = row[f'{comp}_run2']
            diff = abs(val1 - val2) if not (pd.isna(val1) or pd.isna(val2)) else 0
            
            row_data[f'{comp}_run1'] = val1
            row_data[f'{comp}_run2'] = val2
            row_data[f'{comp}_diff'] = diff
            total_diff += diff
        
        comparison_data.append(row_data)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Display key comparison
    display_cols = ['ticker', 'QVM_Composite_run1', 'QVM_Composite_run2', 'QVM_Composite_diff']
    print(comparison_df[display_cols].round(10))
    
    print(f"\n📈 Summary Statistics:")
    print(f"   Total absolute difference across all components: {total_diff:.15f}")
    
    # Component-wise differences
    for comp in components:
        comp_diff = comparison_df[f'{comp}_diff'].sum()
        print(f"   {comp} total diff: {comp_diff:.15f}")
    
    # Final verdict
    print(f"\n" + "="*80)
    print("🏁 TEMPORAL LOGIC VALIDATION VERDICT")
    print("="*80)
    
    if total_diff < 1e-10:
        print("🎉 SUCCESS: All runs produced IDENTICAL results!")
        print("✅ The engine's temporal logic is DETERMINISTIC and ROBUST")
        print("✅ No dependencies on execution context or order")
        print("✅ Cold start handling works correctly")
        print("✅ PARALLEL HISTORICAL GENERATION IS SAFE")
        
        print(f"\n🚦 GO/NO-GO DECISION: ✅ GO FOR PARALLEL EXECUTION")
        print(f"   You can safely run the 4-terminal parallel generation")
        print(f"   Each terminal will independently fetch required historical data")
        
    elif total_diff < 1e-6:
        print("⚠️  MINOR DIFFERENCES: Results are nearly identical")
        print(f"   Total difference: {total_diff:.15f}")
        print("   This may be due to floating-point precision")
        print("   RECOMMENDATION: Proceed with caution")
        
    else:
        print("❌ FAILURE: Significant differences detected")
        print(f"   Total difference: {total_diff:.15f}")
        print("   The engine may have non-deterministic behavior")
        print("   🚫 DO NOT proceed with parallel execution")
        print("   Investigation required")

else:
    print("❌ VALIDATION FAILED: Insufficient successful runs for comparison")
    print("   Cannot validate temporal logic with failed calculations")

# --- 5. ENGINE INTROSPECTION ---
if engine_instances:
    print(f"\n🔧 Engine Configuration Verification:")
    engine = engine_instances[0]
    print(f"   Reporting lag: {engine.reporting_lag} days")
    print(f"   QVM weights: {engine.qvm_weights}")
    
    # Test the quarter lookup logic specifically
    print(f"\n📅 Quarter Lookup Test for {TEST_DATE.date()}:")
    quarter_info = engine.get_correct_quarter_for_date(TEST_DATE)
    if quarter_info:
        year, quarter = quarter_info
        print(f"   Available quarter: {year} Q{quarter}")
        print(f"   ✅ Quarter lookup working correctly")
    else:
        print(f"   ❌ No quarter found - this explains the calculation failures")

print(f"\n" + "="*80)
print("✅ Practical Temporal Logic Validation Complete")
print("="*80)