# ============================================================================
# Aureus Sigma Capital - Phase 22: Canonical Backtest Engine
# Notebook: 22_canonical_backtest_engine.ipynb
#
# Objective:
#   To build a single, definitive, and reusable backtesting engine for a
#   production-style strategy. This engine will serve as the "golden path"
#   template for the full 12-cell test matrix.
#
#   This notebook will first implement and validate the engine for the
#   "Standalone Value" strategy, then apply the same engine to the
#   "QVR_60_20_20" composite for a direct, apples-to-apples comparison.
# ============================================================================
#
# --- BASELINE STRATEGY REFERENCE ---
#
# This engine is built upon the validated methodologies from our most successful prior work:
#
# 1.  **Standalone Value Baseline:**
#     - **Source:** `16b_extended_backtest_2016_2025.md`
#     - **Key Result:** Sharpe Ratio of 2.60 over the full 2016-2025 period.
#     - **Inherited Logic:** Full-period backtesting structure, dynamic universe construction.
#
# 2.  **Composite Logic Baseline:**
#     - **Source:** `16_weighted_composite_engineering.md`
#     - **Key Result:** Validated the methodology for creating weighted composites.
#     - **Inherited Logic:** The critical process of re-normalizing individual factor
#       scores within the liquid universe *before* combining them.
#
# --- KEY UPGRADES IN THIS ENGINE ---
#
# This backtest is NOT a replication. It introduces critical upgrades to bridge
# the gap from theoretical research to practical implementation:
#
# 1.  **Portfolio Construction -> From Quintile to Concentrated:**
#     - **Baseline:** Used the top quintile (top 20% of the liquid universe), which
#       resulted in a variable portfolio size (e.g., 15 to 40 stocks).
#     - **Upgrade:** Implements a **fixed portfolio size of 20 stocks**. This tests
#       a more concentrated, practical strategy suitable for our fund size and
#       management capacity.
#
# 2.  **Engine Design -> From Procedural to Object-Oriented:**
#     - **Baseline:** Procedural scripts tailored to a specific test.
#     - **Upgrade:** A more robust, reusable `CanonicalBacktester` class structure
#       that can be easily configured to run different strategies, forming the
#       foundation for the full test matrix.
#
# 3.  **Focus -> From Discovery to Auditability:**
#     - **Baseline:** Focused on discovering alpha signals.
#     - **Upgrade:** Focused on creating a clean, auditable, end-to-end process
#       that can be handed to a risk manager or investor for due diligence.
#
# --- METHODOLOGY OVERVIEW (THE 5 PHASES) ---
#
# This notebook will execute the following five phases in a clear, sequential manner:
#
#   - **Phase 1: Setup & Configuration:** Define all backtest parameters in one place.
#   - **Phase 2: Data Preparation:** Load all raw factor, price, and benchmark data.
#   - **Phase 3: The Canonical Backtesting Loop:** The core engine that iterates through
#     rebalance periods, constructs the universe, re-normalizes factors, and forms
#     the target portfolio.
#   - **Phase 4: Return Calculation:** Compute daily gross and net returns, accounting
#     for transaction costs and preventing look-ahead bias.
#   - **Phase 5: Performance Analysis:** Generate a full institutional tearsheet to
#     evaluate the final performance.
#

# ============================================================================
# PHASE 1: SETUP & CONFIGURATION (REFACTORED)
# ============================================================================

# Core scientific libraries
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import pickle
from pathlib import Path
import sys

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Database connectivity
import yaml
from sqlalchemy import create_engine, text

# --- Environment Setup ---
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# --- CORRECTED Module Import Logic ---
# The goal is to add the project's root directory (e.g., 'factor_investing_project')
# to the system path, so that we can import from the 'production' package.
try:
    # This assumes the notebook is in: .../factor_investing_project/production/tests/phase22...
    # We need to go up three levels to reach the project root.
    project_root = Path.cwd().parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        print(f"✅ Project root added to sys.path: {project_root}")

    from production.universe.constructors import get_liquid_universe_dataframe
    print("✅ Successfully imported production modules.")

except ImportError as e:
    print(f"❌ ERROR: Could not import production modules. Please check your directory structure.")
    print(f"   - Attempted Project Root: {project_root}")
    print(f"   - Error: {e}")
except Exception as e:
    print(f"❌ An unexpected error occurred during import: {e}")


# --- Standardized Institutional Visualization Palette ---
# Inherited from Phase 16b for consistency
PALETTE = {
    'primary': '#16A085', 'secondary': '#34495E', 'positive': '#27AE60',
    'negative': '#C0392B', 'highlight_1': '#2980B9', 'highlight_2': '#E67E22',
    'neutral': '#7F8C8D', 'grid': '#BDC3C7', 'text': '#2C3E50'
}
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.dpi': 300, 'figure.figsize': (14, 8), 'font.size': 11,
    'axes.facecolor': 'white', 'axes.edgecolor': PALETTE['text'],
    'axes.grid': True, 'axes.axisbelow': True, 'axes.labelcolor': PALETTE['text'],
    'axes.titlepad': 15, 'axes.titlesize': 16, 'axes.titleweight': 'bold',
    'axes.titlecolor': PALETTE['text'], 'grid.color': PALETTE['grid'],
    'legend.frameon': False, 'xtick.color': PALETTE['text'], 'ytick.color': PALETTE['text'],
    'lines.linewidth': 2.0, 'lines.solid_capstyle': 'round'
})
print("✅ Visualization settings configured.")


# --- Core Backtest Configuration ---
# This dictionary will drive the entire backtest.
# We will start with the 'Standalone Value' strategy.
CONFIG = {
    "strategy_name": "Standalone_Value",
    "factor_to_use": "Value_Composite",
    "backtest_start_date": "2015-12-01",
    "backtest_end_date": "2025-07-28",
    "rebalance_frequency": "Q",
    "portfolio_size": 20,
    "transaction_cost_bps": 30,
    "strategy_version_db": "qvm_v2.0_enhanced" # The version to pull from the database
}

print("✅ Backtest configuration defined for 'Standalone Value' strategy.")
print(f"   - Portfolio Size: {CONFIG['portfolio_size']} stocks")
print(f"   - Rebalancing: {CONFIG['rebalance_frequency']} (Quarterly)")
print(f"   - Period: {CONFIG['backtest_start_date']} to {CONFIG['backtest_end_date']}")

✅ Project root added to sys.path: /Users/ducnguyen/Library/CloudStorage/GoogleDrive-duc.nguyentcb@gmail.com/My Drive/quant-world-invest/factor_investing_project
✅ Successfully imported production modules.
✅ Visualization settings configured.
✅ Backtest configuration defined for 'Standalone Value' strategy.
   - Portfolio Size: 20 stocks
   - Rebalancing: Q (Quarterly)
   - Period: 2015-12-01 to 2025-07-28

# ============================================================================
# PHASE 2: DATA PREPARATION
# ============================================================================

# --- Database Connection ---
def create_db_connection():
    """
    Establishes a SQLAlchemy database engine connection using the central config file.
    """
    try:
        # Path is relative to the project root, which we added to sys.path
        config_path = project_root / 'config' / 'database.yml'
        with open(config_path, 'r') as f:
            db_config = yaml.safe_load(f)['production']
        
        connection_string = (
            f"mysql+pymysql://{db_config['username']}:{db_config['password']}"
            f"@{db_config['host']}/{db_config['schema_name']}"
        )
        
        engine = create_engine(connection_string, pool_pre_ping=True)
        
        # Test the connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            
        print(f"✅ Database connection established successfully to schema '{db_config['schema_name']}'.")
        return engine

    except Exception as e:
        print(f"❌ FAILED to connect to the database.")
        print(f"   - Config path checked: {config_path}")
        print(f"   - Error: {e}")
        return None

# Create the engine for this session
engine = create_db_connection()

# --- Load Raw Data for the Full Backtest Period ---
if engine:
    print("\n📂 Loading all raw data for the full backtest period...")

    db_params = {
        'start_date': CONFIG['backtest_start_date'],
        'end_date': CONFIG['backtest_end_date'],
        'strategy_version': CONFIG['strategy_version_db']
    }

    # 1. Factor Scores (Full Universe for all three factors for future use)
    # We load all three now to make the engine easily adaptable for the QVR test later.
    factor_query = text("""
        SELECT date, ticker, Quality_Composite, Value_Composite, Momentum_Composite
        FROM factor_scores_qvm
        WHERE date BETWEEN :start_date AND :end_date 
          AND strategy_version = :strategy_version
    """)
    factor_data_raw = pd.read_sql(factor_query, engine, params=db_params, parse_dates=['date'])
    print(f"   - ✅ Loaded {len(factor_data_raw):,} raw factor observations.")

    # 2. Price Data
    price_query = text("""
        SELECT date, ticker, close 
        FROM equity_history
        WHERE date BETWEEN :start_date AND :end_date
    """)
    price_data_raw = pd.read_sql(price_query, engine, params=db_params, parse_dates=['date'])
    print(f"   - ✅ Loaded {len(price_data_raw):,} raw price observations.")

    # 3. Benchmark Data
    benchmark_query = text("""
        SELECT date, close
        FROM etf_history
        WHERE ticker = 'VNINDEX' AND date BETWEEN :start_date AND :end_date
    """)
    benchmark_data_raw = pd.read_sql(benchmark_query, engine, params=db_params, parse_dates=['date'])
    print(f"   - ✅ Loaded {len(benchmark_data_raw):,} benchmark observations.")

    # --- Prepare Data Structures for Backtesting ---
    print("\n🛠️  Preparing data structures for the backtesting engine...")

    # 1. Calculate daily returns and create the returns matrix
    price_data_raw['return'] = price_data_raw.groupby('ticker')['close'].pct_change()
    daily_returns_matrix = price_data_raw.pivot(index='date', columns='ticker', values='return')
    print(f"   - ✅ Daily returns matrix constructed. Shape: {daily_returns_matrix.shape}")

    # 2. Calculate benchmark returns
    benchmark_returns = benchmark_data_raw.set_index('date')['close'].pct_change().rename('VN-Index')
    print(f"   - ✅ Benchmark returns calculated. Days: {len(benchmark_returns)}")

    print("\n✅ All data prepared. Ready for the canonical backtesting loop.")
else:
    print("\n❌ Halting execution due to database connection failure.")

✅ Database connection established successfully to schema 'alphabeta'.

📂 Loading all raw data for the full backtest period...
   - ✅ Loaded 1,567,488 raw factor observations.
   - ✅ Loaded 1,623,168 raw price observations.
   - ✅ Loaded 2,411 benchmark observations.

🛠️  Preparing data structures for the backtesting engine...
   - ✅ Daily returns matrix constructed. Shape: (2408, 728)
   - ✅ Benchmark returns calculated. Days: 2411

✅ All data prepared. Ready for the canonical backtesting loop.

