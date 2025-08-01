# ============================================================================
# Aureus Sigma Capital - Phase 27e: Walk-Forward Adaptive Engine
# Notebook: 28_walk_forward_adaptive_engine.ipynb
#
# Objective:
#   To build and validate the definitive, institutional-grade production model.
#   This notebook addresses the core failure of the static regime model by
#   implementing a Walk-Forward Optimizer. This system will dynamically learn
#   and adapt factor weights based on recent market history, creating a proactive
#   strategy. This is the final sprint to meet all Investment Committee hurdles.
# ============================================================================
#
# --- STRATEGIC DIRECTIVE & DIAGNOSIS ---
#
# The Phase 27c Regime-Adaptive model failed (Sharpe 0.72, Return 8.89%) due to
# a lagging regime indicator and a flawed risk overlay that resulted in permanent
# de-risking. The root cause is that our static, rule-based approach cannot adapt
# to the changing efficacy of factors in the Vietnamese market.
#
# --- METHODOLOGY: THE WALK-FORWARD OPTIMIZER ---
#
# This notebook will build the `PortfolioEngine_v3.0` with a new core architecture:
#
# 1.  **Walk-Forward Framework**: The backtest will be structured as a series of
#     expanding windows.
#     -   **Training Period**: A 24-month lookback window.
#     -   **Trading Period**: A 6-month "out-of-sample" trading period.
# 2.  **Dynamic Weight Optimization**: At the start of each 6-month trading period,
#     the engine will run an optimization process on the preceding 24 months of
#     data to find the factor weights (for V, Q, M, R) that maximize the Sharpe Ratio.
# 3.  **Bayesian Priors**: The optimizer will be constrained with sensible priors
#     (e.g., Value weight must be ≥ 30%) to ensure stability and prevent extreme,
#     uninvestable allocations.
# 4.  **Full Risk Management**: The final, fully-managed version will apply our
#     validated Hybrid Volatility Overlay and Portfolio Stop-Loss to the P&L
#     generated by the superior, dynamically-weighted alpha signal.
#
# --- SUCCESS CRITERIA ---
#
# The final, fully-managed, walk-forward optimized strategy must meet all
# Investment Committee hurdles over the full 2016-2025 backtest period:
#
#   -   **Sharpe Ratio**: ≥ 1.0
#   -   **Maximum Drawdown**: ≤ -35%
#

# ============================================================================
# CELL 2: SETUP & WALK-FORWARD CONFIGURATION
# ============================================================================

# Core scientific libraries
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
from pathlib import Path
import sys
import yaml

# Optimization and statistical libraries
from scipy.optimize import minimize

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Database connectivity
from sqlalchemy import create_engine, text

# --- Environment Setup ---
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', lambda x: f'{x:,.2f}')

# --- Add Project Root to Python Path ---
try:
    current_path = Path.cwd()
    while not (current_path / 'production').is_dir():
        if current_path.parent == current_path:
            raise FileNotFoundError("Could not find the 'production' directory.")
        current_path = current_path.parent
    
    project_root = current_path
    
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    from production.universe.constructors import get_liquid_universe_dataframe
    print(f"✅ Successfully imported production modules.")
    print(f"   - Project Root set to: {project_root}")

except (ImportError, FileNotFoundError) as e:
    print(f"❌ ERROR: Could not import production modules. Please check your directory structure.")
    print(f"   - Final Path Searched: {project_root}")
    print(f"   - Error: {e}")
    raise

# --- Walk-Forward Adaptive v1.0 Configuration ---
# This configuration defines our most sophisticated, production-candidate strategy.
WF_CONFIG = {
    # --- Backtest Parameters ---
    "strategy_name": "WalkForward_Adaptive_v1.0",
    "backtest_start_date": "2016-03-01",
    "backtest_end_date": "2025-07-28",
    "rebalance_frequency": "Q",
    "transaction_cost_bps": 30,

    # --- Universe & Portfolio Construction ---
    "universe": { "lookback_days": 63, "adtv_threshold_bn": 10.0, "top_n": 200, "min_trading_coverage": 0.6 },
    "portfolio": { "portfolio_size": 20 },

    # --- Walk-Forward Optimization ---
    "walk_forward": {
        "train_months": 24,
        "test_months": 6, # This is the weight lock-in period
        "factors_to_optimize": ['Value_Composite', 'Quality_Composite', 'Momentum_Composite', 'Momentum_Reversal'],
        # Bayesian Priors (Constraints for the optimizer)
        "bounds": {
            'Value_Composite': (0.30, 0.70),    # Value is our anchor, must be at least 30%
            'Quality_Composite': (0.10, 0.40),
            'Momentum_Composite': (0.00, 0.50),
            'Momentum_Reversal': (0.00, 0.30)
        }
    },

    # --- Factor Signal Source ---
    "signal": { "db_strategy_version": "qvm_v2.0_enhanced" },
    
    # --- Risk Overlay Parameters ---
    "risk_overlay": {
        "volatility_target": 0.15,
        "regime_dd_threshold": -0.07,
        "stop_loss_threshold": -0.15,
        "de_risk_level": 0.3
    }
}

print("\n⚙️  Walk-Forward Adaptive v1.0 Configuration Loaded:")
print(f"   - Strategy: {WF_CONFIG['strategy_name']}")
print(f"   - Period: {WF_CONFIG['backtest_start_date']} to {WF_CONFIG['backtest_end_date']}")
print(f"   - Optimization: {WF_CONFIG['walk_forward']['train_months']}mo train / {WF_CONFIG['walk_forward']['test_months']}mo trade")
print(f"   - Value Factor Constraint: {WF_CONFIG['walk_forward']['bounds']['Value_Composite'][0]:.0%} - {WF_CONFIG['walk_forward']['bounds']['Value_Composite'][1]:.0%}")

# --- Database Connection ---
def create_db_connection(project_root_path: Path):
    """Establishes a SQLAlchemy database engine connection."""
    try:
        config_path = project_root_path / 'config' / 'database.yml'
        with open(config_path, 'r') as f:
            db_config = yaml.safe_load(f)['production']
        connection_string = (f"mysql+pymysql://{db_config['username']}:{db_config['password']}@{db_config['host']}/{db_config['schema_name']}")
        engine = create_engine(connection_string, pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print(f"\n✅ Database connection established successfully to schema '{db_config['schema_name']}'.")
        return engine
    except Exception as e:
        print(f"❌ FAILED to connect to the database: {e}")
        return None

# Create the engine for this session
engine = create_db_connection(project_root)
if engine is None:
    raise ConnectionError("Database connection failed. Halting execution.")

✅ Successfully imported production modules.
   - Project Root set to: /Users/ducnguyen/Library/CloudStorage/GoogleDrive-duc.nguyentcb@gmail.com/My Drive/quant-world-invest/factor_investing_project

⚙️  Walk-Forward Adaptive v1.0 Configuration Loaded:
   - Strategy: WalkForward_Adaptive_v1.0
   - Period: 2016-03-01 to 2025-07-28
   - Optimization: 24mo train / 6mo trade
   - Value Factor Constraint: 30% - 70%

✅ Database connection established successfully to schema 'alphabeta'.

# ============================================================================
# CELL 3: DATA INGESTION FOR WALK-FORWARD MODEL
# ============================================================================

def load_all_data_for_wf_model(config: dict, db_engine):
    """
    Loads all necessary data for the full backtest period, including an
    initial buffer for the first training window.
    """
    # The true backtest starts later, but we need data from the beginning for the first optimization
    train_months = config['walk_forward']['train_months']
    buffer_start_date = pd.Timestamp(config['backtest_start_date']) - pd.DateOffset(months=train_months)
    end_date = config['backtest_end_date']
    db_version = config['signal']['db_strategy_version']
    
    print(f"📂 Loading all data for period: {buffer_start_date.date()} to {end_date}...")
    print(f"   (Includes a {train_months}-month buffer for the initial training period)")

    db_params = {
        'start_date': buffer_start_date,
        'end_date': pd.Timestamp(end_date),
        'strategy_version': db_version
    }

    # 1. Factor Scores
    print("   - Loading factor scores...")
    factor_query = text("""
        SELECT date, ticker, Quality_Composite, Value_Composite, Momentum_Composite
        FROM factor_scores_qvm
        WHERE date BETWEEN :start_date AND :end_date 
          AND strategy_version = :strategy_version
    """)
    factor_data = pd.read_sql(factor_query, db_engine, params=db_params, parse_dates=['date'])
    print(f"     ✅ Loaded {len(factor_data):,} factor observations.")

    # 2. Price Data
    print("   - Loading price data...")
    price_query = text("""
        SELECT date, ticker, close 
        FROM equity_history
        WHERE date BETWEEN :start_date AND :end_date
    """)
    price_data = pd.read_sql(price_query, db_engine, params=db_params, parse_dates=['date'])
    print(f"     ✅ Loaded {len(price_data):,} price observations.")

    # 3. Benchmark Data
    print("   - Loading benchmark data (VN-Index)...")
    benchmark_query = text("""
        SELECT date, close
        FROM etf_history
        WHERE ticker = 'VNINDEX' AND date BETWEEN :start_date AND :end_date
    """)
    benchmark_data = pd.read_sql(benchmark_query, db_engine, params=db_params, parse_dates=['date'])
    print(f"     ✅ Loaded {len(benchmark_data):,} benchmark observations.")

    # --- Data Preparation ---
    print("\n🛠️  Preparing data structures...")
    price_data['return'] = price_data.groupby('ticker')['close'].pct_change()
    daily_returns_matrix = price_data.pivot(index='date', columns='ticker', values='return')
    benchmark_returns = benchmark_data.set_index('date')['close'].pct_change().rename('VN-Index')
    
    print("   ✅ Data preparation complete.")
    return factor_data, daily_returns_matrix, benchmark_returns

# Execute the data loading
try:
    factor_data_raw, daily_returns_matrix, benchmark_returns = load_all_data_for_wf_model(WF_CONFIG, engine)
    print("\n✅ All data successfully loaded and prepared for the backtest.")
    print(f"   - Factor Data Shape: {factor_data_raw.shape}")
    print(f"   - Returns Matrix Shape: {daily_returns_matrix.shape}")
    print(f"   - Benchmark Returns: {len(benchmark_returns)} days")
except Exception as e:
    print(f"❌ ERROR during data ingestion: {e}")
    raise

📂 Loading all data for period: 2014-03-01 to 2025-07-28...
   (Includes a 24-month buffer for the initial training period)
   - Loading factor scores...
     ✅ Loaded 1,567,488 factor observations.
   - Loading price data...
     ✅ Loaded 1,847,773 price observations.
   - Loading benchmark data (VN-Index)...
     ✅ Loaded 2,848 benchmark observations.

🛠️  Preparing data structures...
   ✅ Data preparation complete.

✅ All data successfully loaded and prepared for the backtest.
   - Factor Data Shape: (1567488, 5)
   - Returns Matrix Shape: (2845, 728)
   - Benchmark Returns: 2848 days

# ============================================================================
# CELL 3: DATA INGESTION FOR WALK-FORWARD MODEL
# ============================================================================

def load_all_data_for_wf_model(config: dict, db_engine):
    """
    Loads all necessary data for the full backtest period, including an
    initial buffer for the first training window.
    """
    # The true backtest starts later, but we need data from the beginning for the first optimization
    train_months = config['walk_forward']['train_months']
    buffer_start_date = pd.Timestamp(config['backtest_start_date']) - pd.DateOffset(months=train_months)
    end_date = config['backtest_end_date']
    db_version = config['signal']['db_strategy_version']
    
    print(f"📂 Loading all data for period: {buffer_start_date.date()} to {end_date}...")
    print(f"   (Includes a {train_months}-month buffer for the initial training period)")

    db_params = {
        'start_date': buffer_start_date,
        'end_date': pd.Timestamp(end_date),
        'strategy_version': db_version
    }

    # 1. Factor Scores
    print("   - Loading factor scores...")
    factor_query = text("""
        SELECT date, ticker, Quality_Composite, Value_Composite, Momentum_Composite
        FROM factor_scores_qvm
        WHERE date BETWEEN :start_date AND :end_date 
          AND strategy_version = :strategy_version
    """)
    factor_data = pd.read_sql(factor_query, db_engine, params=db_params, parse_dates=['date'])
    print(f"     ✅ Loaded {len(factor_data):,} factor observations.")

    # 2. Price Data
    print("   - Loading price data...")
    price_query = text("""
        SELECT date, ticker, close 
        FROM equity_history
        WHERE date BETWEEN :start_date AND :end_date
    """)
    price_data = pd.read_sql(price_query, db_engine, params=db_params, parse_dates=['date'])
    print(f"     ✅ Loaded {len(price_data):,} price observations.")

    # 3. Benchmark Data
    print("   - Loading benchmark data (VN-Index)...")
    benchmark_query = text("""
        SELECT date, close
        FROM etf_history
        WHERE ticker = 'VNINDEX' AND date BETWEEN :start_date AND :end_date
    """)
    benchmark_data = pd.read_sql(benchmark_query, db_engine, params=db_params, parse_dates=['date'])
    print(f"     ✅ Loaded {len(benchmark_data):,} benchmark observations.")

    # --- Data Preparation ---
    print("\n🛠️  Preparing data structures...")
    price_data['return'] = price_data.groupby('ticker')['close'].pct_change()
    daily_returns_matrix = price_data.pivot(index='date', columns='ticker', values='return')
    benchmark_returns = benchmark_data.set_index('date')['close'].pct_change().rename('VN-Index')
    
    print("   ✅ Data preparation complete.")
    return factor_data, daily_returns_matrix, benchmark_returns

# Execute the data loading
try:
    factor_data_raw, daily_returns_matrix, benchmark_returns = load_all_data_for_wf_model(WF_CONFIG, engine)
    print("\n✅ All data successfully loaded and prepared for the backtest.")
    print(f"   - Factor Data Shape: {factor_data_raw.shape}")
    print(f"   - Returns Matrix Shape: {daily_returns_matrix.shape}")
    print(f"   - Benchmark Returns: {len(benchmark_returns)} days")
except Exception as e:
    print(f"❌ ERROR during data ingestion: {e}")
    raise

📂 Loading all data for period: 2014-03-01 to 2025-07-28...
   (Includes a 24-month buffer for the initial training period)
   - Loading factor scores...
     ✅ Loaded 1,567,488 factor observations.
   - Loading price data...
     ✅ Loaded 1,847,773 price observations.
   - Loading benchmark data (VN-Index)...
     ✅ Loaded 2,848 benchmark observations.

🛠️  Preparing data structures...
   ✅ Data preparation complete.

✅ All data successfully loaded and prepared for the backtest.
   - Factor Data Shape: (1567488, 5)
   - Returns Matrix Shape: (2845, 728)
   - Benchmark Returns: 2848 days

# ============================================================================
# CELL 4: IMPORT ENGINES & DEFINE ANALYTICS SUITE
# ============================================================================
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

# --- Import the parallelizable function and engines from our new module ---
try:
    from production.engine.adaptive_engine import run_optimization_for_period, PortfolioEngine_v2_0
    print("✅ Successfully imported the parallel optimization function and engine.")
except ImportError as e:
    print(f"❌ ERROR: Could not import from 'production.engine.adaptive_engine'. Please ensure the file is created correctly.")
    raise

# --- WALK-FORWARD ENGINE (FINAL VERSION) ---
class WalkForwardEngine:
    def __init__(self, config: dict, factor_data: pd.DataFrame, returns_matrix: pd.DataFrame,
                 benchmark_returns: pd.Series, db_engine):
        self.config = config; self.engine = db_engine; self.factor_data_raw = factor_data
        self.daily_returns_matrix = returns_matrix; self.benchmark_returns = benchmark_returns
        self._universe_cache = {}
        print(f"✅ WalkForwardEngine (Optimized) initialized for '{config['strategy_name']}'.")

    def run_backtest(self) -> (pd.Series, pd.DataFrame, pd.Series, pd.DataFrame):
        print("\n🚀 Starting walk-forward backtest execution (with Caching & Parallelization)...")
        start_date = pd.Timestamp(self.config['backtest_start_date']); end_date = pd.Timestamp(self.config['backtest_end_date'])
        train_months = self.config['walk_forward']['train_months']; test_months = self.config['walk_forward']['test_months']
        periods = []
        current_start = start_date
        while current_start < end_date:
            train_start = current_start - pd.DateOffset(months=train_months)
            train_end = current_start - pd.DateOffset(days=1)
            test_end = current_start + pd.DateOffset(months=test_months) - pd.DateOffset(days=1)
            periods.append((train_start, train_end, current_start, min(test_end, end_date)))
            current_start += pd.DateOffset(months=test_months)
        print(f"   - Generated {len(periods)} walk-forward periods.")

        print("   - Starting parallel optimization across all periods...")
        with open(project_root / 'config' / 'database.yml', 'r') as f:
            db_engine_config = yaml.safe_load(f)['production']
        
        tasks = [(p[0], p[1], self.config, self.factor_data_raw, self.daily_returns_matrix, self.benchmark_returns, db_engine_config) for p in periods]
        
        optimal_weights_list = []
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(run_optimization_for_period, task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(tasks), desc="Optimizing Periods"):
                optimal_weights_list.append(future.result())
        
        all_optimal_weights = [{'start_date': p[2], **w} for p, w in zip(periods, optimal_weights_list)]
        
        print("\n   - Running sequential daily backtest with optimized weights...")
        all_net_returns = []; all_diagnostics = []; all_daily_exposure = []
        for i, (train_start, train_end, test_start, test_end) in enumerate(periods):
            optimal_weights = optimal_weights_list[i]
            print(f"     - Testing Period {i+1}/{len(periods)} ({test_start.date()} to {test_end.date()}) with optimized weights...")
            period_returns, period_diagnostics, period_exposure = self._run_daily_iterative_test(test_start, test_end, optimal_weights)
            all_net_returns.append(period_returns); all_diagnostics.append(period_diagnostics); all_daily_exposure.append(period_exposure)

        final_net_returns = pd.concat(all_net_returns); final_diagnostics = pd.concat(all_diagnostics)
        final_daily_exposure = pd.concat(all_daily_exposure); optimal_weights_df = pd.DataFrame(all_optimal_weights).set_index('start_date')
        print("\n✅ Walk-forward backtest execution complete.")
        return final_net_returns, final_diagnostics, final_daily_exposure, optimal_weights_df

    def _run_daily_iterative_test(self, test_start: pd.Timestamp, test_end: pd.Timestamp, factor_weights: dict):
        test_config = self.config.copy()
        test_config['backtest_start_date'] = test_start.strftime('%Y-%m-%d')
        test_config['backtest_end_date'] = test_end.strftime('%Y-%m-%d')
        test_config['signal']['factors_to_combine'] = factor_weights
        engine = PortfolioEngine_v2_0(config=test_config, factor_data=self.factor_data_raw, returns_matrix=self.daily_returns_matrix,
                                      benchmark_returns=self.benchmark_returns, db_engine=self.engine, universe_cache=self._universe_cache)
        return engine.run_backtest(mode='iterative')

# --- Analytics Suite ---
def calculate_official_metrics(returns: pd.Series, benchmark: pd.Series, periods_per_year: int = 252) -> dict:
    first_trade_date = returns.loc[returns.ne(0)].index.min()
    if pd.isna(first_trade_date): return {metric: 0.0 for metric in ['Annualized Return (%)', 'Annualized Volatility (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 'Calmar Ratio', 'Information Ratio', 'Beta']}
    aligned_returns = returns.loc[first_trade_date:]; aligned_benchmark = benchmark.loc[first_trade_date:]
    n_years = len(aligned_returns) / periods_per_year; annualized_return = ((1 + aligned_returns).prod() ** (1 / n_years) - 1) if n_years > 0 else 0
    annualized_volatility = aligned_returns.std() * np.sqrt(periods_per_year); sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else 0.0
    cumulative_returns = (1 + aligned_returns).cumprod(); max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min(); calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0.0
    excess_returns = aligned_returns - aligned_benchmark; information_ratio = (excess_returns.mean() * periods_per_year) / (excess_returns.std() * np.sqrt(periods_per_year)) if excess_returns.std() > 0 else 0.0
    beta = aligned_returns.cov(aligned_benchmark) / aligned_benchmark.var() if aligned_benchmark.var() > 0 else 0.0
    return {'Annualized Return (%)': annualized_return * 100, 'Annualized Volatility (%)': annualized_volatility * 100, 'Sharpe Ratio': sharpe_ratio, 'Max Drawdown (%)': max_drawdown * 100, 'Calmar Ratio': calmar_ratio, 'Information Ratio': information_ratio, 'Beta': beta}

def generate_full_tearsheet(strategy_returns: pd.Series, benchmark_returns: pd.Series, diagnostics: pd.DataFrame, title: str, exposure: pd.Series = None, weights_df: pd.DataFrame = None):
    metrics = calculate_official_metrics(strategy_returns, benchmark_returns)
    gs_rows = 8 if weights_df is not None else 6
    height_ratios = [1.2, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 1.2] if weights_df is not None else [1.2, 0.8, 0.8, 0.8, 0.8, 1.2]
    fig = plt.figure(figsize=(18, 36 if weights_df is not None else 28))
    gs = fig.add_gridspec(gs_rows, 2, height_ratios=height_ratios, hspace=0.9, wspace=0.2); fig.suptitle(title, fontsize=20, fontweight='bold')
    # Full plotting logic here...
    plt.show()

print("✅ All engine classes and the analytics suite are defined and ready for the final execution.")

✅ Successfully imported the parallel optimization function and engine.
✅ All engine classes and the analytics suite are defined and ready for the final execution.

# ============================================================================
# CELL 5: EXECUTION & FINAL VERDICT (v3.2 - Refactored)
# ============================================================================

# --- Instantiate the WalkForwardEngine ---
final_engine = WalkForwardEngine(
    config=WF_CONFIG,
    factor_data=factor_data_raw,
    returns_matrix=daily_returns_matrix,
    benchmark_returns=benchmark_returns,
    db_engine=engine
)

# --- Run the Definitive Backtest ---
# This will now run in parallel and should be much faster.
final_net_returns, final_diagnostics, final_daily_exposure, optimal_weights_df = final_engine.run_backtest()

# --- Generate the Final Report ---
print("\n" + "="*80)
print("📊 FINAL PERFORMANCE REPORT: Walk-Forward Adaptive Strategy")
print("="*80)

final_metrics = generate_full_tearsheet(
    strategy_returns=final_net_returns,
    benchmark_returns=benchmark_returns,
    diagnostics=final_diagnostics,
    title="Final Approved Strategy: Walk-Forward Adaptive Model",
    exposure=final_daily_exposure,
    weights_df=optimal_weights_df
)

# --- INSTITUTIONAL VERDICT ---
print("\n" + "="*80)
print("🏆 INSTITUTIONAL VERDICT & RECOMMENDATION")
print("="*80)
sharpe_target = 1.0
dd_target = -35.0

sharpe_ok = final_metrics['Sharpe Ratio'] >= sharpe_target
dd_ok = final_metrics['Max Drawdown (%)'] >= dd_target

if sharpe_ok and dd_ok:
    print(f"✅✅✅ SUCCESS: The Walk-Forward Adaptive strategy meets all Investment Committee hurdles.")
    print(f"   - Final Sharpe Ratio: {final_metrics['Sharpe Ratio']:.2f} (Target: ≥{sharpe_target})")
    print(f"   - Final Max Drawdown: {final_metrics['Max Drawdown (%)']:.2f}% (Target: ≥{dd_target}%)")
    print("\nRECOMMENDATION: This configuration is APPROVED for pilot deployment.")
    print("This is our definitive production model.")
else:
    print(f"❌ FAILURE: The final strategy still does not meet the full IC hurdles.")
    if not sharpe_ok: print(f"   - Sharpe Ratio: {final_metrics['Sharpe Ratio']:.2f} (Target: ≥{sharpe_target})")
    if not dd_ok: print(f"   - Max Drawdown: {final_metrics['Max Drawdown (%)']:.2f}% (Target: ≥{dd_target}%)")
    print("\nRECOMMENDATION: The Deep Alpha Enhancement sprint to add new, uncorrelated factors is now the mandatory next step.")


================================================================================
📊 FINAL PERFORMANCE REPORT: Walk-Forward Adaptive Strategy
================================================================================
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
<Figure size 1800x3600 with 0 Axes>

================================================================================
🏆 INSTITUTIONAL VERDICT & RECOMMENDATION
================================================================================
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
Cell In[10], line 39
     36 sharpe_target = 1.0
     37 dd_target = -35.0
---> 39 sharpe_ok = final_metrics['Sharpe Ratio'] >= sharpe_target
     40 dd_ok = final_metrics['Max Drawdown (%)'] >= dd_target
     42 if sharpe_ok and dd_ok:

TypeError: 'NoneType' object is not subscriptable

