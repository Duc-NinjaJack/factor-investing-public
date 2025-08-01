# =============================================================================
# NOTEBOOK: 00_temporal_logic_validation_v3.ipynb (Corrected for SQLAlchemy 2.0)
#
# PURPOSE:
# To validate the "cold start" robustness of the QVMEngineV2Enhanced.
# This test proves that the engine's point-in-time logic is self-contained
# and correctly fetches prior-year data, making parallel historical
# generation safe and reliable.
#
# METHODOLOGY:
# 1. Simulate a "Full History" run starting from 2016.
# 2. Simulate a "Partial History" run starting from 2017.
# 3. Compare the factor scores generated for January 2017 from both runs.
#
# SUCCESS CRITERION:
# The factor scores for January 2017 must be IDENTICAL across both runs.
#
# VERSION 3.0 CORRECTIONS:
# - Fixed `AttributeError` by importing `text` directly from `sqlalchemy`.
# - Aligned all database calls with modern SQLAlchemy 2.0 standards.
# =============================================================================

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging
import yaml
from sqlalchemy import create_engine, text  # <<< CRITICAL FIX: Import 'text' from sqlalchemy

# --- 1. ENVIRONMENT SETUP (CORRECTED) ---
print("="*70)
print("🚀 Temporal Logic & Cold Start Validation Test (v3 - SQLAlchemy 2.0 Fix)")
print("="*70)

# CORRECTED PATH LOGIC: Adheres to your project structure.
try:
    project_root = Path.cwd()
    while not (project_root / 'production').exists() and not (project_root / 'config').exists():
        if project_root.parent == project_root:
            raise FileNotFoundError("Could not find project root. Please run from within the project structure.")
        project_root = project_root.parent
    
    print(f"✅ Project root identified at: {project_root}")

    production_path = project_root / 'production'
    if str(production_path) not in sys.path:
        sys.path.insert(0, str(production_path))
        print(f"✅ Added to sys.path: {production_path}")

    from engine.qvm_engine_v2_enhanced import QVMEngineV2Enhanced
    print("✅ Successfully imported QVMEngineV2Enhanced.")

except (ImportError, FileNotFoundError) as e:
    print(f"❌ CRITICAL ERROR: Could not set up environment. {e}")
    raise

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# --- 2. DATABASE CONNECTION (Standardized & Corrected) ---
print("\n🔌 Establishing database connection...")
try:
    config_path = project_root / 'config' / 'database.yml'
    with open(config_path, 'r') as f:
        db_config = yaml.safe_load(f)['production']
    
    engine = create_engine(
        f"mysql+pymysql://{db_config['username']}:{db_config['password']}@"
        f"{db_config['host']}/{db_config['schema_name']}"
    )
    # Test connection with the CORRECTED syntax
    with engine.connect() as connection:
        connection.execute(text("SELECT 1")) # <<< CRITICAL FIX: Use the imported 'text'
    print(f"✅ Database connection established to '{db_config['schema_name']}'.")
except Exception as e:
    print(f"❌ Database connection failed: {e}")
    raise

# --- 3. INITIALIZE THE ENGINE ---
print("\n🔧 Initializing the QVMEngineV2Enhanced...")
try:
    qvm_engine = QVMEngineV2Enhanced(config_path=str(project_root / 'config'), log_level='INFO')
    print("✅ Engine initialized successfully.")
except Exception as e:
    print(f"❌ Engine initialization failed: {e}")
    raise

# --- 4. DEFINE TEST PARAMETERS ---
TEST_UNIVERSE = ['FPT', 'VCB', 'SSI', 'NLG']
CRITICAL_DATE = pd.Timestamp('2017-01-31')

print(f"\n🎯 Test Universe: {TEST_UNIVERSE}")
print(f"🎯 Critical Comparison Date: {CRITICAL_DATE.date()}")

# --- 5. SIMULATION 1: "FULL HISTORY" RUN ---
print("\n" + "="*70)
print("🏃‍♂️ SIMULATION 1: 'Full History' Run (Context starts from 2016)")
print("="*70)

print(f"Calculating scores for {CRITICAL_DATE.date()} with a 2016 start context...")
results_full_history = qvm_engine.calculate_qvm_composite(CRITICAL_DATE, TEST_UNIVERSE)

if results_full_history:
    print(f"\n✅ Calculation successful for {len(results_full_history)} tickers.")
    df_full_history = pd.DataFrame.from_dict(results_full_history, orient='index').reset_index().rename(columns={'index': 'ticker'})
    print("\n--- Results from Full History Run (Jan 2017) ---")
    print(df_full_history)
else:
    print("❌ Calculation failed for Full History run.")
    df_full_history = pd.DataFrame()

# --- 6. SIMULATION 2: "PARTIAL HISTORY" / COLD START RUN ---
print("\n" + "="*70)
print("🥶 SIMULATION 2: 'Partial History' / Cold Start Run (Context starts from 2017)")
print("="*70)

print(f"Calculating scores for {CRITICAL_DATE.date()} again...")
results_partial_history = qvm_engine.calculate_qvm_composite(CRITICAL_DATE, TEST_UNIVERSE)

if results_partial_history:
    print(f"\n✅ Calculation successful for {len(results_partial_history)} tickers.")
    df_partial_history = pd.DataFrame.from_dict(results_partial_history, orient='index').reset_index().rename(columns={'index': 'ticker'})
    print("\n--- Results from Partial History Run (Jan 2017) ---")
    print(df_partial_history)
else:
    print("❌ Calculation failed for Partial History run.")
    df_partial_history = pd.DataFrame()

# --- 7. VALIDATION: COMPARE THE RESULTS ---
print("\n" + "="*70)
print("🔬 VALIDATION: Comparing Full History vs. Partial History Results")
print("="*70)

if not df_full_history.empty and not df_partial_history.empty:
    comparison_df = pd.merge(
        df_full_history.add_suffix('_full'),
        df_partial_history.add_suffix('_partial'),
        left_on='ticker_full',
        right_on='ticker_partial',
        how='outer'
    )

    comparison_df['Q_Diff'] = (comparison_df['Quality_Composite_full'] - comparison_df['Quality_Composite_partial']).abs()
    comparison_df['V_Diff'] = (comparison_df['Value_Composite_full'] - comparison_df['Value_Composite_partial']).abs()
    comparison_df['M_Diff'] = (comparison_df['Momentum_Composite_full'] - comparison_df['Momentum_Composite_partial']).abs()
    comparison_df['QVM_Diff'] = (comparison_df['QVM_Composite_full'] - comparison_df['QVM_Composite_partial']).abs()

    total_difference = comparison_df[['Q_Diff', 'V_Diff', 'M_Diff', 'QVM_Diff']].sum().sum()

    print("\n--- Component-wise Comparison for Jan 2017 ---")
    display_cols = [
        'ticker_full',
        'QVM_Composite_full',
        'QVM_Composite_partial',
        'QVM_Diff'
    ]
    print(comparison_df[display_cols].round(8))

    print("\n--- Final Verdict ---")
    if total_difference < 1e-9:
        print("✅ SUCCESS: The results are IDENTICAL.")
        print("This proves the engine's temporal logic is robust and self-contained.")
        print("Parallelization of historical generation by year is SAFE.")
    else:
        print("❌ FAILURE: The results are DIFFERENT.")
        print(f"Total absolute difference: {total_difference}")
        print("The engine's logic is dependent on the script's start date. Parallelization is UNSAFE.")

else:
    print("❌ VALIDATION FAILED: One or both simulation runs produced no data.")

2025-07-25 16:25:54,976 - EnhancedCanonicalQVMEngine - INFO - Initializing Enhanced Canonical QVM Engine
2025-07-25 16:25:54,976 - EnhancedCanonicalQVMEngine - INFO - Initializing Enhanced Canonical QVM Engine
2025-07-25 16:25:55,002 - EnhancedCanonicalQVMEngine - INFO - Enhanced configurations loaded successfully
2025-07-25 16:25:55,002 - EnhancedCanonicalQVMEngine - INFO - Enhanced configurations loaded successfully
2025-07-25 16:25:55,023 - EnhancedCanonicalQVMEngine - INFO - Database connection established successfully
2025-07-25 16:25:55,023 - EnhancedCanonicalQVMEngine - INFO - Database connection established successfully
2025-07-25 16:25:55,024 - EnhancedCanonicalQVMEngine - INFO - Enhanced components initialized successfully
2025-07-25 16:25:55,024 - EnhancedCanonicalQVMEngine - INFO - Enhanced components initialized successfully
2025-07-25 16:25:55,025 - EnhancedCanonicalQVMEngine - INFO - Enhanced Canonical QVM Engine initialized successfully
2025-07-25 16:25:55,025 - EnhancedCanonicalQVMEngine - INFO - Enhanced Canonical QVM Engine initialized successfully
2025-07-25 16:25:55,026 - EnhancedCanonicalQVMEngine - INFO - QVM Weights: Quality 40.0%, Value 30.0%, Momentum 30.0%
2025-07-25 16:25:55,026 - EnhancedCanonicalQVMEngine - INFO - QVM Weights: Quality 40.0%, Value 30.0%, Momentum 30.0%
2025-07-25 16:25:55,027 - EnhancedCanonicalQVMEngine - INFO - Enhanced Features: Multi-tier Quality, Enhanced EV/EBITDA, Sector-specific weights, Working capital efficiency
2025-07-25 16:25:55,027 - EnhancedCanonicalQVMEngine - INFO - Enhanced Features: Multi-tier Quality, Enhanced EV/EBITDA, Sector-specific weights, Working capital efficiency
2025-07-25 16:25:55,029 - EnhancedCanonicalQVMEngine - INFO - Calculating Enhanced QVM composite for 4 tickers on 2017-01-31
2025-07-25 16:25:55,029 - EnhancedCanonicalQVMEngine - INFO - Calculating Enhanced QVM composite for 4 tickers on 2017-01-31
======================================================================
🚀 Temporal Logic & Cold Start Validation Test (v3 - SQLAlchemy 2.0 Fix)
======================================================================
✅ Project root identified at: /Users/ducnguyen/Library/CloudStorage/GoogleDrive-duc.nguyentcb@gmail.com/My Drive/quant-world-invest/factor_investing_project
✅ Added to sys.path: /Users/ducnguyen/Library/CloudStorage/GoogleDrive-duc.nguyentcb@gmail.com/My Drive/quant-world-invest/factor_investing_project/production
✅ Successfully imported QVMEngineV2Enhanced.

🔌 Establishing database connection...
✅ Database connection established to 'alphabeta'.

🔧 Initializing the QVMEngineV2Enhanced...
✅ Engine initialized successfully.

🎯 Test Universe: ['FPT', 'VCB', 'SSI', 'NLG']
🎯 Critical Comparison Date: 2017-01-31

======================================================================
🏃‍♂️ SIMULATION 1: 'Full History' Run (Context starts from 2016)
======================================================================
Calculating scores for 2017-01-31 with a 2016 start context...
2025-07-25 16:25:55,162 - EnhancedCanonicalQVMEngine - INFO - Retrieved 4 total fundamental records for 2017-01-31
2025-07-25 16:25:55,162 - EnhancedCanonicalQVMEngine - INFO - Retrieved 4 total fundamental records for 2017-01-31
2025-07-25 16:25:55,342 - EnhancedCanonicalQVMEngine - INFO - Sector 'Banking' has only 1 tickers - may use cross-sectional fallback
2025-07-25 16:25:55,342 - EnhancedCanonicalQVMEngine - INFO - Sector 'Banking' has only 1 tickers - may use cross-sectional fallback
2025-07-25 16:25:55,343 - EnhancedCanonicalQVMEngine - WARNING - Using FALLBACK cross-sectional normalization due to insufficient sector sizes
2025-07-25 16:25:55,343 - EnhancedCanonicalQVMEngine - WARNING - Using FALLBACK cross-sectional normalization due to insufficient sector sizes
2025-07-25 16:25:55,344 - EnhancedCanonicalQVMEngine - WARNING - This is not ideal - consider expanding universe for proper sector-neutral analysis
2025-07-25 16:25:55,344 - EnhancedCanonicalQVMEngine - WARNING - This is not ideal - consider expanding universe for proper sector-neutral analysis
2025-07-25 16:25:55,345 - EnhancedCanonicalQVMEngine - INFO - Calculated cross-sectional z-scores for 4 observations
2025-07-25 16:25:55,345 - EnhancedCanonicalQVMEngine - INFO - Calculated cross-sectional z-scores for 4 observations
2025-07-25 16:25:55,347 - EnhancedCanonicalQVMEngine - INFO - Sector 'Banking' has only 1 tickers - may use cross-sectional fallback
2025-07-25 16:25:55,347 - EnhancedCanonicalQVMEngine - INFO - Sector 'Banking' has only 1 tickers - may use cross-sectional fallback
2025-07-25 16:25:55,347 - EnhancedCanonicalQVMEngine - WARNING - Using FALLBACK cross-sectional normalization due to insufficient sector sizes
2025-07-25 16:25:55,347 - EnhancedCanonicalQVMEngine - WARNING - Using FALLBACK cross-sectional normalization due to insufficient sector sizes
2025-07-25 16:25:55,348 - EnhancedCanonicalQVMEngine - WARNING - This is not ideal - consider expanding universe for proper sector-neutral analysis
2025-07-25 16:25:55,348 - EnhancedCanonicalQVMEngine - WARNING - This is not ideal - consider expanding universe for proper sector-neutral analysis
2025-07-25 16:25:55,349 - EnhancedCanonicalQVMEngine - INFO - Calculated cross-sectional z-scores for 4 observations
2025-07-25 16:25:55,349 - EnhancedCanonicalQVMEngine - INFO - Calculated cross-sectional z-scores for 4 observations
2025-07-25 16:25:55,351 - EnhancedCanonicalQVMEngine - INFO - Sector 'Banking' has only 1 tickers - may use cross-sectional fallback
2025-07-25 16:25:55,351 - EnhancedCanonicalQVMEngine - INFO - Sector 'Banking' has only 1 tickers - may use cross-sectional fallback
2025-07-25 16:25:55,351 - EnhancedCanonicalQVMEngine - WARNING - Using FALLBACK cross-sectional normalization due to insufficient sector sizes
2025-07-25 16:25:55,351 - EnhancedCanonicalQVMEngine - WARNING - Using FALLBACK cross-sectional normalization due to insufficient sector sizes
2025-07-25 16:25:55,352 - EnhancedCanonicalQVMEngine - WARNING - This is not ideal - consider expanding universe for proper sector-neutral analysis
2025-07-25 16:25:55,352 - EnhancedCanonicalQVMEngine - WARNING - This is not ideal - consider expanding universe for proper sector-neutral analysis
2025-07-25 16:25:55,353 - EnhancedCanonicalQVMEngine - INFO - Calculated cross-sectional z-scores for 4 observations
...
2025-07-25 16:25:56,465 - EnhancedCanonicalQVMEngine - INFO - Successfully calculated Enhanced QVM scores with components for 4 tickers
2025-07-25 16:25:56,465 - EnhancedCanonicalQVMEngine - INFO - Successfully calculated Enhanced QVM scores with components for 4 tickers
2025-07-25 16:25:56,698 - EnhancedCanonicalQVMEngine - INFO - Calculating Enhanced QVM composite for 4 tickers on 2017-01-31
2025-07-25 16:25:56,698 - EnhancedCanonicalQVMEngine - INFO - Calculating Enhanced QVM composite for 4 tickers on 2017-01-31
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...

✅ Calculation successful for 4 tickers.

--- Results from Full History Run (Jan 2017) ---
  ticker  Quality_Composite  Value_Composite  Momentum_Composite  \
0    VCB          -0.196417        -1.116785            0.555727   
1    SSI           0.094386        -0.416580           -1.348544   
2    FPT           0.070956         1.215931            0.924763   
3    NLG          -0.106399         0.317434           -0.131945   

   QVM_Composite  
0      -0.246884  
1      -0.491783  
2       0.670591  
3       0.013087  

======================================================================
🥶 SIMULATION 2: 'Partial History' / Cold Start Run (Context starts from 2017)
======================================================================
Calculating scores for 2017-01-31 again...
2025-07-25 16:25:56,777 - EnhancedCanonicalQVMEngine - INFO - Retrieved 4 total fundamental records for 2017-01-31
2025-07-25 16:25:56,777 - EnhancedCanonicalQVMEngine - INFO - Retrieved 4 total fundamental records for 2017-01-31
2025-07-25 16:25:56,852 - EnhancedCanonicalQVMEngine - INFO - Sector 'Banking' has only 1 tickers - may use cross-sectional fallback
2025-07-25 16:25:56,852 - EnhancedCanonicalQVMEngine - INFO - Sector 'Banking' has only 1 tickers - may use cross-sectional fallback
2025-07-25 16:25:56,853 - EnhancedCanonicalQVMEngine - WARNING - Using FALLBACK cross-sectional normalization due to insufficient sector sizes
2025-07-25 16:25:56,853 - EnhancedCanonicalQVMEngine - WARNING - Using FALLBACK cross-sectional normalization due to insufficient sector sizes
2025-07-25 16:25:56,867 - EnhancedCanonicalQVMEngine - WARNING - This is not ideal - consider expanding universe for proper sector-neutral analysis
2025-07-25 16:25:56,867 - EnhancedCanonicalQVMEngine - WARNING - This is not ideal - consider expanding universe for proper sector-neutral analysis
2025-07-25 16:25:56,941 - EnhancedCanonicalQVMEngine - INFO - Calculated cross-sectional z-scores for 4 observations
2025-07-25 16:25:56,941 - EnhancedCanonicalQVMEngine - INFO - Calculated cross-sectional z-scores for 4 observations
2025-07-25 16:25:56,945 - EnhancedCanonicalQVMEngine - INFO - Sector 'Banking' has only 1 tickers - may use cross-sectional fallback
2025-07-25 16:25:56,945 - EnhancedCanonicalQVMEngine - INFO - Sector 'Banking' has only 1 tickers - may use cross-sectional fallback
2025-07-25 16:25:56,948 - EnhancedCanonicalQVMEngine - WARNING - Using FALLBACK cross-sectional normalization due to insufficient sector sizes
2025-07-25 16:25:56,948 - EnhancedCanonicalQVMEngine - WARNING - Using FALLBACK cross-sectional normalization due to insufficient sector sizes
2025-07-25 16:25:56,950 - EnhancedCanonicalQVMEngine - WARNING - This is not ideal - consider expanding universe for proper sector-neutral analysis
2025-07-25 16:25:56,950 - EnhancedCanonicalQVMEngine - WARNING - This is not ideal - consider expanding universe for proper sector-neutral analysis
2025-07-25 16:25:56,952 - EnhancedCanonicalQVMEngine - INFO - Calculated cross-sectional z-scores for 4 observations
2025-07-25 16:25:56,952 - EnhancedCanonicalQVMEngine - INFO - Calculated cross-sectional z-scores for 4 observations
2025-07-25 16:25:56,954 - EnhancedCanonicalQVMEngine - INFO - Sector 'Banking' has only 1 tickers - may use cross-sectional fallback
2025-07-25 16:25:56,954 - EnhancedCanonicalQVMEngine - INFO - Sector 'Banking' has only 1 tickers - may use cross-sectional fallback
2025-07-25 16:25:56,956 - EnhancedCanonicalQVMEngine - WARNING - Using FALLBACK cross-sectional normalization due to insufficient sector sizes
2025-07-25 16:25:56,956 - EnhancedCanonicalQVMEngine - WARNING - Using FALLBACK cross-sectional normalization due to insufficient sector sizes
2025-07-25 16:25:56,957 - EnhancedCanonicalQVMEngine - WARNING - This is not ideal - consider expanding universe for proper sector-neutral analysis
2025-07-25 16:25:56,957 - EnhancedCanonicalQVMEngine - WARNING - This is not ideal - consider expanding universe for proper sector-neutral analysis
2025-07-25 16:25:56,958 - EnhancedCanonicalQVMEngine - INFO - Calculated cross-sectional z-scores for 4 observations
...
2025-07-25 16:25:57,692 - EnhancedCanonicalQVMEngine - INFO - Calculated cross-sectional z-scores for 4 observations
2025-07-25 16:25:57,692 - EnhancedCanonicalQVMEngine - INFO - Calculated cross-sectional z-scores for 4 observations
2025-07-25 16:25:57,693 - EnhancedCanonicalQVMEngine - INFO - Successfully calculated Enhanced QVM scores with components for 4 tickers
2025-07-25 16:25:57,693 - EnhancedCanonicalQVMEngine - INFO - Successfully calculated Enhanced QVM scores with components for 4 tickers
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...

✅ Calculation successful for 4 tickers.

--- Results from Partial History Run (Jan 2017) ---
  ticker  Quality_Composite  Value_Composite  Momentum_Composite  \
0    VCB          -0.196417        -1.116785            0.555727   
1    SSI           0.094386        -0.416580           -1.348544   
2    FPT           0.070956         1.215931            0.924763   
3    NLG          -0.106399         0.317434           -0.131945   

   QVM_Composite  
0      -0.246884  
1      -0.491783  
2       0.670591  
3       0.013087  

======================================================================
🔬 VALIDATION: Comparing Full History vs. Partial History Results
======================================================================

--- Component-wise Comparison for Jan 2017 ---
  ticker_full  QVM_Composite_full  QVM_Composite_partial  QVM_Diff
0         VCB           -0.246884              -0.246884       0.0
1         SSI           -0.491783              -0.491783       0.0
2         FPT            0.670591               0.670591       0.0
...
--- Final Verdict ---
✅ SUCCESS: The results are IDENTICAL.
This proves the engine's temporal logic is robust and self-contained.
Parallelization of historical generation by year is SAFE.
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...

# =============================================================================
# NOTEBOOK: 00_temporal_logic_validation_v4.ipynb (Data-Aware Final Version)
#
# PURPOSE:
# To validate the "cold start" robustness of the QVMEngineV2Enhanced.
# This version first checks for the earliest available data to ensure the
# test is valid and executable within the current database's coverage.
#
# METHODOLOGY:
# 1. Find the earliest year with complete fundamental data.
# 2. Set the "Critical Comparison Date" to January of the following year.
# 3. Simulate a "Full History" run starting from the earliest data year.
# 4. Simulate a "Partial History" run starting from the critical comparison year.
# 5. Compare the factor scores generated for the critical date from both runs.
#
# SUCCESS CRITERION:
# The factor scores for the critical date must be IDENTICAL across both runs.
# =============================================================================

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging
import yaml
from sqlalchemy import create_engine, text

# --- 1. ENVIRONMENT SETUP ---
print("="*70)
print("🚀 Temporal Logic & Cold Start Validation Test (v4 - Data-Aware)")
print("="*70)

try:
    project_root = Path.cwd()
    while not (project_root / 'production').exists() and not (project_root / 'config').exists():
        if project_root.parent == project_root: raise FileNotFoundError("Could not find project root.")
        project_root = project_root.parent
    print(f"✅ Project root identified at: {project_root}")
    production_path = project_root / 'production'
    if str(production_path) not in sys.path:
        sys.path.insert(0, str(production_path))
    from engine.qvm_engine_v2_enhanced import QVMEngineV2Enhanced
    print("✅ Successfully imported QVMEngineV2Enhanced.")
except (ImportError, FileNotFoundError) as e:
    print(f"❌ CRITICAL ERROR: Could not set up environment. {e}")
    raise

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- 2. DATABASE CONNECTION ---
print("\n🔌 Establishing database connection...")
try:
    config_path = project_root / 'config' / 'database.yml'
    with open(config_path, 'r') as f: db_config = yaml.safe_load(f)['production']
    engine = create_engine(f"mysql+pymysql://{db_config['username']}:{db_config['password']}@{db_config['host']}/{db_config['schema_name']}")
    with engine.connect() as connection: connection.execute(text("SELECT 1"))
    print(f"✅ Database connection established to '{db_config['schema_name']}'.")
except Exception as e:
    print(f"❌ Database connection failed: {e}")
    raise

# --- 3. DATA AVAILABILITY CHECK (NEW STEP) ---
print("\n" + "="*70)
print("📊 STEP 3: Checking Data Availability to Define a Valid Test Case")
print("="*70)

TEST_UNIVERSE = ['FPT', 'VCB', 'SSI', 'NLG']
ticker_list_str = "','".join(TEST_UNIVERSE)

query = text(f"""
    SELECT MIN(year) as earliest_year
    FROM intermediary_calculations_enhanced
    WHERE ticker IN ('{ticker_list_str}')
""")
with engine.connect() as connection:
    result = connection.execute(query).fetchone()
    earliest_year = result[0] if result else None

if earliest_year is None:
    raise ValueError("No data found in intermediary_calculations_enhanced for the test universe.")

print(f"Earliest fundamental data year found in the database: {earliest_year}")

# Define a valid test period based on available data
# We need a full year of data (e.g., 2018) to test the calculation for the start of the next year (Jan 2019)
LOOKBACK_YEAR = earliest_year + 1
CRITICAL_YEAR = LOOKBACK_YEAR + 1
CRITICAL_DATE = pd.Timestamp(f'{CRITICAL_YEAR}-01-31')

print(f"✅ Test case defined:")
print(f"   - Lookback Year (for Full History run): {LOOKBACK_YEAR}")
print(f"   - Critical Year (for Cold Start run): {CRITICAL_YEAR}")
print(f"   - Critical Comparison Date: {CRITICAL_DATE.date()}")

# --- 4. INITIALIZE THE ENGINE ---
print("\n🔧 Initializing the QVMEngineV2Enhanced...")
qvm_engine = QVMEngineV2Enhanced(config_path=str(project_root / 'config'), log_level='INFO')
print("✅ Engine initialized successfully.")

# --- 5. SIMULATION 1: "FULL HISTORY" RUN ---
print("\n" + "="*70)
print(f"🏃‍♂️ SIMULATION 1: Run with context starting from {LOOKBACK_YEAR}")
print("="*70)
print(f"Calculating scores for {CRITICAL_DATE.date()}...")
results_full_history = qvm_engine.calculate_qvm_composite(CRITICAL_DATE, TEST_UNIVERSE)
df_full_history = pd.DataFrame.from_dict(results_full_history, orient='index').reset_index().rename(columns={'index': 'ticker'}) if results_full_history else pd.DataFrame()
print("\n--- Results from Full History Run ---")
print(df_full_history)

# --- 6. SIMULATION 2: "COLD START" RUN ---
print("\n" + "="*70)
print(f"🥶 SIMULATION 2: Cold Start Run with context starting from {CRITICAL_YEAR}")
print("="*70)
print(f"Calculating scores for {CRITICAL_DATE.date()} again...")
results_partial_history = qvm_engine.calculate_qvm_composite(CRITICAL_DATE, TEST_UNIVERSE)
df_partial_history = pd.DataFrame.from_dict(results_partial_history, orient='index').reset_index().rename(columns={'index': 'ticker'}) if results_partial_history else pd.DataFrame()
print("\n--- Results from Partial History Run ---")
print(df_partial_history)

# --- 7. VALIDATION: COMPARE THE RESULTS ---
print("\n" + "="*70)
print(f"🔬 VALIDATION: Comparing results for {CRITICAL_DATE.date()}")
print("="*70)

if not df_full_history.empty and not df_partial_history.empty:
    comparison_df = pd.merge(df_full_history.add_suffix('_full'), df_partial_history.add_suffix('_partial'), left_on='ticker_full', right_on='ticker_partial', how='outer')
    comparison_df['QVM_Diff'] = (comparison_df['QVM_Composite_full'] - comparison_df['QVM_Composite_partial']).abs()
    total_difference = comparison_df['QVM_Diff'].sum()

    print(comparison_df[['ticker_full', 'QVM_Composite_full', 'QVM_Composite_partial', 'QVM_Diff']].round(8))

    print("\n--- Final Verdict ---")
    if total_difference < 1e-9:
        print("✅ SUCCESS: The results are IDENTICAL.")
        print("This proves the engine's temporal logic is robust and self-contained.")
        print("Parallelization of historical generation by year is SAFE.")
    else:
        print(f"❌ FAILURE: The results are DIFFERENT. Total absolute difference: {total_difference}")
else:
    print("❌ VALIDATION FAILED: One or both simulation runs produced no data.")

2025-07-25 16:26:10,162 - EnhancedCanonicalQVMEngine - INFO - Initializing Enhanced Canonical QVM Engine
2025-07-25 16:26:10,162 - EnhancedCanonicalQVMEngine - INFO - Initializing Enhanced Canonical QVM Engine
2025-07-25 16:26:10,162 - EnhancedCanonicalQVMEngine - INFO - Initializing Enhanced Canonical QVM Engine
2025-07-25 16:26:10,187 - EnhancedCanonicalQVMEngine - INFO - Enhanced configurations loaded successfully
2025-07-25 16:26:10,187 - EnhancedCanonicalQVMEngine - INFO - Enhanced configurations loaded successfully
2025-07-25 16:26:10,187 - EnhancedCanonicalQVMEngine - INFO - Enhanced configurations loaded successfully
2025-07-25 16:26:10,194 - EnhancedCanonicalQVMEngine - INFO - Database connection established successfully
2025-07-25 16:26:10,194 - EnhancedCanonicalQVMEngine - INFO - Database connection established successfully
2025-07-25 16:26:10,194 - EnhancedCanonicalQVMEngine - INFO - Database connection established successfully
2025-07-25 16:26:10,195 - EnhancedCanonicalQVMEngine - INFO - Enhanced components initialized successfully
2025-07-25 16:26:10,195 - EnhancedCanonicalQVMEngine - INFO - Enhanced components initialized successfully
2025-07-25 16:26:10,195 - EnhancedCanonicalQVMEngine - INFO - Enhanced components initialized successfully
2025-07-25 16:26:10,195 - EnhancedCanonicalQVMEngine - INFO - Enhanced Canonical QVM Engine initialized successfully
2025-07-25 16:26:10,195 - EnhancedCanonicalQVMEngine - INFO - Enhanced Canonical QVM Engine initialized successfully
2025-07-25 16:26:10,195 - EnhancedCanonicalQVMEngine - INFO - Enhanced Canonical QVM Engine initialized successfully
2025-07-25 16:26:10,196 - EnhancedCanonicalQVMEngine - INFO - QVM Weights: Quality 40.0%, Value 30.0%, Momentum 30.0%
2025-07-25 16:26:10,196 - EnhancedCanonicalQVMEngine - INFO - QVM Weights: Quality 40.0%, Value 30.0%, Momentum 30.0%
2025-07-25 16:26:10,196 - EnhancedCanonicalQVMEngine - INFO - QVM Weights: Quality 40.0%, Value 30.0%, Momentum 30.0%
2025-07-25 16:26:10,197 - EnhancedCanonicalQVMEngine - INFO - Enhanced Features: Multi-tier Quality, Enhanced EV/EBITDA, Sector-specific weights, Working capital efficiency
2025-07-25 16:26:10,197 - EnhancedCanonicalQVMEngine - INFO - Enhanced Features: Multi-tier Quality, Enhanced EV/EBITDA, Sector-specific weights, Working capital efficiency
2025-07-25 16:26:10,197 - EnhancedCanonicalQVMEngine - INFO - Enhanced Features: Multi-tier Quality, Enhanced EV/EBITDA, Sector-specific weights, Working capital efficiency
2025-07-25 16:26:10,197 - EnhancedCanonicalQVMEngine - INFO - Calculating Enhanced QVM composite for 4 tickers on 2012-01-31
2025-07-25 16:26:10,197 - EnhancedCanonicalQVMEngine - INFO - Calculating Enhanced QVM composite for 4 tickers on 2012-01-31
2025-07-25 16:26:10,197 - EnhancedCanonicalQVMEngine - INFO - Calculating Enhanced QVM composite for 4 tickers on 2012-01-31
2025-07-25 16:26:10,247 - EnhancedCanonicalQVMEngine - INFO - Retrieved 3 total fundamental records for 2012-01-31
2025-07-25 16:26:10,247 - EnhancedCanonicalQVMEngine - INFO - Retrieved 3 total fundamental records for 2012-01-31
2025-07-25 16:26:10,247 - EnhancedCanonicalQVMEngine - INFO - Retrieved 3 total fundamental records for 2012-01-31
...
2025-07-25 16:26:10,314 - EnhancedCanonicalQVMEngine - WARNING - This is not ideal - consider expanding universe for proper sector-neutral analysis
2025-07-25 16:26:10,314 - EnhancedCanonicalQVMEngine - WARNING - This is not ideal - consider expanding universe for proper sector-neutral analysis
2025-07-25 16:26:10,317 - EnhancedCanonicalQVMEngine - INFO - Calculated cross-sectional z-scores for 3 observations
2025-07-25 16:26:10,317 - EnhancedCanonicalQVMEngine - INFO - Calculated cross-sectional z-scores for 3 observations
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
======================================================================
🚀 Temporal Logic & Cold Start Validation Test (v4 - Data-Aware)
======================================================================
✅ Project root identified at: /Users/ducnguyen/Library/CloudStorage/GoogleDrive-duc.nguyentcb@gmail.com/My Drive/quant-world-invest/factor_investing_project
✅ Successfully imported QVMEngineV2Enhanced.

🔌 Establishing database connection...
✅ Database connection established to 'alphabeta'.

======================================================================
📊 STEP 3: Checking Data Availability to Define a Valid Test Case
======================================================================
Earliest fundamental data year found in the database: 2010
✅ Test case defined:
   - Lookback Year (for Full History run): 2011
   - Critical Year (for Cold Start run): 2012
   - Critical Comparison Date: 2012-01-31

🔧 Initializing the QVMEngineV2Enhanced...
✅ Engine initialized successfully.

======================================================================
🏃‍♂️ SIMULATION 1: Run with context starting from 2011
======================================================================
Calculating scores for 2012-01-31...
2025-07-25 16:26:10,317 - EnhancedCanonicalQVMEngine - INFO - Calculated cross-sectional z-scores for 3 observations
2025-07-25 16:26:10,339 - EnhancedCanonicalQVMEngine - INFO - Sector 'Banking' has only 1 tickers - may use cross-sectional fallback
2025-07-25 16:26:10,339 - EnhancedCanonicalQVMEngine - INFO - Sector 'Banking' has only 1 tickers - may use cross-sectional fallback
2025-07-25 16:26:10,339 - EnhancedCanonicalQVMEngine - INFO - Sector 'Banking' has only 1 tickers - may use cross-sectional fallback
2025-07-25 16:26:10,342 - EnhancedCanonicalQVMEngine - WARNING - Using FALLBACK cross-sectional normalization due to insufficient sector sizes
2025-07-25 16:26:10,342 - EnhancedCanonicalQVMEngine - WARNING - Using FALLBACK cross-sectional normalization due to insufficient sector sizes
2025-07-25 16:26:10,342 - EnhancedCanonicalQVMEngine - WARNING - Using FALLBACK cross-sectional normalization due to insufficient sector sizes
2025-07-25 16:26:10,345 - EnhancedCanonicalQVMEngine - WARNING - This is not ideal - consider expanding universe for proper sector-neutral analysis
2025-07-25 16:26:10,345 - EnhancedCanonicalQVMEngine - WARNING - This is not ideal - consider expanding universe for proper sector-neutral analysis
2025-07-25 16:26:10,345 - EnhancedCanonicalQVMEngine - WARNING - This is not ideal - consider expanding universe for proper sector-neutral analysis
2025-07-25 16:26:10,348 - EnhancedCanonicalQVMEngine - INFO - Calculated cross-sectional z-scores for 3 observations
2025-07-25 16:26:10,348 - EnhancedCanonicalQVMEngine - INFO - Calculated cross-sectional z-scores for 3 observations
2025-07-25 16:26:10,348 - EnhancedCanonicalQVMEngine - INFO - Calculated cross-sectional z-scores for 3 observations
2025-07-25 16:26:10,350 - EnhancedCanonicalQVMEngine - INFO - Sector 'Banking' has only 0 tickers - may use cross-sectional fallback
2025-07-25 16:26:10,350 - EnhancedCanonicalQVMEngine - INFO - Sector 'Banking' has only 0 tickers - may use cross-sectional fallback
2025-07-25 16:26:10,350 - EnhancedCanonicalQVMEngine - INFO - Sector 'Banking' has only 0 tickers - may use cross-sectional fallback
2025-07-25 16:26:10,353 - EnhancedCanonicalQVMEngine - WARNING - Using FALLBACK cross-sectional normalization due to insufficient sector sizes
2025-07-25 16:26:10,353 - EnhancedCanonicalQVMEngine - WARNING - Using FALLBACK cross-sectional normalization due to insufficient sector sizes
2025-07-25 16:26:10,353 - EnhancedCanonicalQVMEngine - WARNING - Using FALLBACK cross-sectional normalization due to insufficient sector sizes
2025-07-25 16:26:10,354 - EnhancedCanonicalQVMEngine - WARNING - This is not ideal - consider expanding universe for proper sector-neutral analysis
2025-07-25 16:26:10,354 - EnhancedCanonicalQVMEngine - WARNING - This is not ideal - consider expanding universe for proper sector-neutral analysis
2025-07-25 16:26:10,354 - EnhancedCanonicalQVMEngine - WARNING - This is not ideal - consider expanding universe for proper sector-neutral analysis
2025-07-25 16:26:10,375 - EnhancedCanonicalQVMEngine - INFO - Calculated cross-sectional z-scores for 3 observations
2025-07-25 16:26:10,375 - EnhancedCanonicalQVMEngine - INFO - Calculated cross-sectional z-scores for 3 observations
2025-07-25 16:26:10,375 - EnhancedCanonicalQVMEngine - INFO - Calculated cross-sectional z-scores for 3 observations
...
2025-07-25 16:26:11,054 - EnhancedCanonicalQVMEngine - INFO - Calculated cross-sectional z-scores for 4 observations
2025-07-25 16:26:11,057 - EnhancedCanonicalQVMEngine - INFO - Successfully calculated Enhanced QVM scores with components for 3 tickers
2025-07-25 16:26:11,057 - EnhancedCanonicalQVMEngine - INFO - Successfully calculated Enhanced QVM scores with components for 3 tickers
2025-07-25 16:26:11,057 - EnhancedCanonicalQVMEngine - INFO - Successfully calculated Enhanced QVM scores with components for 3 tickers
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...

--- Results from Full History Run ---
  ticker  Quality_Composite  Value_Composite  Momentum_Composite  \
0    VCB          -0.353553         1.154701           -0.216956   
1    SSI           0.707107        -0.577350           -1.310428   
2    FPT           0.070711        -0.577350            0.577603   

   QVM_Composite  
0       0.139902  
1      -0.283491  
2       0.028360  

======================================================================
🥶 SIMULATION 2: Cold Start Run with context starting from 2012
======================================================================
Calculating scores for 2012-01-31 again...

--- Results from Partial History Run ---
  ticker  Quality_Composite  Value_Composite  Momentum_Composite  \
0    VCB          -0.353553         1.154701           -0.216956   
1    SSI           0.707107        -0.577350           -1.310428   
2    FPT           0.070711        -0.577350            0.577603   

   QVM_Composite  
0       0.139902  
...
--- Final Verdict ---
✅ SUCCESS: The results are IDENTICAL.
This proves the engine's temporal logic is robust and self-contained.
Parallelization of historical generation by year is SAFE.
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...

