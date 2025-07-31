# ============================================================================
# Aureus Sigma Capital - Phase 27: Official Baseline v1.0
# Notebook: 27_official_baseline_standalone_value.ipynb
#
# Objective:
#   To establish the single, definitive, and immutable performance baseline for
#   all future strategy development. This notebook will execute a methodologically
#   pure backtest of our most robust standalone signal (`Standalone Value`) using
#   a clean engine that contains all critical mechanical fixes but NO risk overlays.
#   The results of this backtest will serve as the "Official Baseline v1.0"
#   against which all subsequent enhancements (e.g., risk overlays, factor
#   enhancements) will be measured.
# ============================================================================
#
# --- STRATEGY & ENGINE SPECIFICATION ---
#
# *   **Strategy**: `A_Standalone_Value`
#     -   **Backtest Period**: 2016-03-01 to 2025-07-28 (Full Period)
#     -   **Signal**: `Value_Composite` from the `qvm_v2.0_enhanced` engine.
#
# *   **Execution Engine**: `PortfolioEngine_v3.1` (from Phase 23)
#     -   **P0 Fix**: Correct turnover calculation (`turnover / 2`).
#     -   **P1 Fixes**: Hybrid portfolio construction (Fixed-N for small universes,
#       Percentile for large) and z-score safeguards.
#     -   **EXCLUDED**: No volatility targeting, no regime filters, no stop-losses.
#       The portfolio is fully invested at all times.
#
# --- METHODOLOGY WORKFLOW ---
#
# 1.  **Setup & Configuration**: Define the single configuration for the baseline run.
# 2.  **Data Ingestion**: Load all required data for the full 2016-2025 period.
# 3.  **Engine Definition**: Define the clean `PortfolioEngine_v3.1` class.
# 4.  **Backtest Execution**: Run the full-period backtest.
# 5.  **Performance Analysis & Reporting**:
#     -   Generate a full institutional tearsheet with a **CORRECTLY ALIGNED**
#       benchmark, ensuring both strategy and index start their cumulative
#       return calculation from the date of the first trade.
#     -   Produce the final, official performance metrics table.
#
# --- DATA DEPENDENCIES ---
#
# *   **Database**: `alphabeta` (Production)
# *   **Tables**:
#     -   `factor_scores_qvm` (strategy_version='qvm_v2.0_enhanced')
#     -   `equity_history`
#     -   `vcsc_daily_data_complete`
#
# --- EXPECTED OUTPUTS ---
#
# 1.  **Primary Deliverable**: The Official Baseline v1.0 Tearsheet. This chart
#     must show a perfectly aligned benchmark and a complete drawdown curve.
# 2.  **Secondary Deliverable**: The Official Baseline v1.0 performance table.
#     These metrics (Sharpe, Max DD, etc.) will become the formal hurdles that
#     all future strategy improvements must beat.
#

# ============================================================================
# CELL 2: SETUP & CONFIGURATION
# ============================================================================

# Core scientific libraries
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
from pathlib import Path
import sys
import yaml

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Database connectivity
from sqlalchemy import create_engine, text

# --- Environment Setup ---
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', lambda x: f'{x:,.2f}')

# --- Add Project Root to Python Path (Corrected Logic) ---
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

# --- Official Baseline v1.0 Configuration ---
# This is the single source of truth for the baseline strategy.
BASELINE_CONFIG = {
    # --- Backtest Parameters ---
    "strategy_name": "Official_Baseline_v1.0_Value",
    "backtest_start_date": "2016-03-01",
    "backtest_end_date": "2025-07-28",
    "rebalance_frequency": "Q", # Quarterly
    "transaction_cost_bps": 30, # Flat 30bps for simplicity in baseline

    # --- Universe Construction ---
    "universe": {
        "lookback_days": 63,
        "adtv_threshold_bn": 10.0,
        "top_n": 200,
        "min_trading_coverage": 0.6,
    },

    # --- Factor & Signal Generation ---
    "signal": {
        "factors_to_combine": {
            'Value_Composite': 1.0
        },
        "db_strategy_version": "qvm_v2.0_enhanced"
    },

    # --- Portfolio Construction (Hybrid Method from v3.1) ---
    "portfolio": {
        "construction_method": "hybrid",
        "portfolio_size_small_universe": 20, # Fixed-N for small universes
        "selection_percentile": 0.8, # Top 20% for large universes
    }
}

print("\n⚙️  Official Baseline v1.0 Configuration Loaded:")
print(f"   - Strategy: {BASELINE_CONFIG['strategy_name']}")
print(f"   - Period: {BASELINE_CONFIG['backtest_start_date']} to {BASELINE_CONFIG['backtest_end_date']}")
print(f"   - Engine Logic: P0 (Turnover) + P1 (Hybrid Portfolio) fixes included.")
print(f"   - Risk Overlay: NONE (Fully invested)")

# --- Database Connection ---
def create_db_connection(project_root_path: Path):
    """Establishes a SQLAlchemy database engine connection."""
    try:
        config_path = project_root_path / 'config' / 'database.yml'
        with open(config_path, 'r') as f:
            db_config = yaml.safe_load(f)['production']
        
        connection_string = (
            f"mysql+pymysql://{db_config['username']}:{db_config['password']}"
            f"@{db_config['host']}/{db_config['schema_name']}"
        )
        engine = create_engine(connection_string, pool_pre_ping=True)
        
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print(f"\n✅ Database connection established successfully to schema '{db_config['schema_name']}'.")
        return engine

    except Exception as e:
        print(f"❌ FAILED to connect to the database.")
        print(f"   - Config path checked: {config_path}")
        print(f"   - Error: {e}")
        return None

# Create the engine for this session
engine = create_db_connection(project_root)

if engine is None:
    raise ConnectionError("Database connection failed. Halting execution.")

✅ Successfully imported production modules.
   - Project Root set to: /Users/ducnguyen/Library/CloudStorage/GoogleDrive-duc.nguyentcb@gmail.com/My Drive/quant-world-invest/factor_investing_project

⚙️  Official Baseline v1.0 Configuration Loaded:
   - Strategy: Official_Baseline_v1.0_Value
   - Period: 2016-03-01 to 2025-07-28
   - Engine Logic: P0 (Turnover) + P1 (Hybrid Portfolio) fixes included.
   - Risk Overlay: NONE (Fully invested)

✅ Database connection established successfully to schema 'alphabeta'.

# ============================================================================
# CELL 3: DATA INGESTION FOR FULL BACKTEST PERIOD
# ============================================================================

def load_all_data_for_backtest(config: dict, db_engine):
    """
    Loads all necessary data (factors, prices, benchmark) for the
    specified backtest period.
    """
    start_date = config['backtest_start_date']
    end_date = config['backtest_end_date']
    db_version = config['signal']['db_strategy_version']
    
    # Add a buffer to the start date for rolling calculations
    buffer_start_date = pd.Timestamp(start_date) - pd.DateOffset(months=3)
    
    print(f"📂 Loading all data for period: {buffer_start_date.date()} to {end_date}...")

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
    print("\n🛠️  Preparing data structures for backtesting engine...")

    # Create returns matrix
    price_data['return'] = price_data.groupby('ticker')['close'].pct_change()
    daily_returns_matrix = price_data.pivot(index='date', columns='ticker', values='return')

    # Create benchmark returns series
    benchmark_returns = benchmark_data.set_index('date')['close'].pct_change().rename('VN-Index')

    print("   ✅ Data preparation complete.")
    return factor_data, daily_returns_matrix, benchmark_returns

# Execute the data loading
try:
    factor_data_raw, daily_returns_matrix, benchmark_returns = load_all_data_for_backtest(BASELINE_CONFIG, engine)
    print("\n✅ All data successfully loaded and prepared for the backtest.")
    print(f"   - Factor Data Shape: {factor_data_raw.shape}")
    print(f"   - Returns Matrix Shape: {daily_returns_matrix.shape}")
    print(f"   - Benchmark Returns: {len(benchmark_returns)} days")
except Exception as e:
    print(f"❌ ERROR during data ingestion: {e}")
    raise

📂 Loading all data for period: 2015-12-01 to 2025-07-28...
   - Loading factor scores...
     ✅ Loaded 1,567,488 factor observations.
   - Loading price data...
     ✅ Loaded 1,623,168 price observations.
   - Loading benchmark data (VN-Index)...
     ✅ Loaded 2,411 benchmark observations.

🛠️  Preparing data structures for backtesting engine...
   ✅ Data preparation complete.

✅ All data successfully loaded and prepared for the backtest.
   - Factor Data Shape: (1567488, 5)
   - Returns Matrix Shape: (2408, 728)
   - Benchmark Returns: 2411 days

# ============================================================================
# CELL 4: BASELINE PORTFOLIO ENGINE (v3.1 LOGIC)
# ============================================================================

class BaselinePortfolioEngine:
    """
    A clean backtesting engine implementing the logic from PortfolioEngine_v3.1.
    This engine includes P0 (turnover) and P1 (hybrid portfolio construction)
    fixes but contains NO risk overlays. It is used to establish the official
    performance baseline of the fully-invested strategy.
    """
    def __init__(self, config: dict, factor_data: pd.DataFrame, returns_matrix: pd.DataFrame,
                 benchmark_returns: pd.Series, db_engine):
        
        self.config = config
        self.engine = db_engine
        
        # Slice data to the exact backtest window
        start = pd.Timestamp(config['backtest_start_date'])
        end = pd.Timestamp(config['backtest_end_date'])
        
        self.factor_data_raw = factor_data[factor_data['date'].between(start, end)].copy()
        self.daily_returns_matrix = returns_matrix.loc[start:end].copy()
        self.benchmark_returns = benchmark_returns.loc[start:end].copy()
        
        print("✅ BaselinePortfolioEngine initialized.")
        print(f"   - Strategy: {config['strategy_name']}")
        print(f"   - Period: {self.daily_returns_matrix.index.min().date()} to {self.daily_returns_matrix.index.max().date()}")

    def run_backtest(self) -> (pd.Series, pd.DataFrame):
        """Executes the full backtesting pipeline."""
        print("\n🚀 Starting baseline backtest execution...")
        
        rebalance_dates = self._generate_rebalance_dates()
        daily_holdings, diagnostics = self._run_backtesting_loop(rebalance_dates)
        net_returns = self._calculate_net_returns(daily_holdings)
        
        print("✅ Baseline backtest execution complete.")
        return net_returns, diagnostics

    def _generate_rebalance_dates(self) -> list:
        """Generates quarterly rebalance dates based on actual trading days."""
        all_trading_dates = self.daily_returns_matrix.index
        rebal_dates_calendar = pd.date_range(
            start=self.config['backtest_start_date'],
            end=self.config['backtest_end_date'],
            freq=self.config['rebalance_frequency']
        )
        actual_rebal_dates = [all_trading_dates[all_trading_dates.searchsorted(d, side='left')-1] for d in rebal_dates_calendar if d >= all_trading_dates.min()]
        print(f"   - Generated {len(actual_rebal_dates)} quarterly rebalance dates.")
        return sorted(list(set(actual_rebal_dates)))

    def _run_backtesting_loop(self, rebalance_dates: list) -> (pd.DataFrame, pd.DataFrame):
        """The core loop for portfolio construction at each rebalance date."""
        daily_holdings = pd.DataFrame(0.0, index=self.daily_returns_matrix.index, columns=self.daily_returns_matrix.columns)
        diagnostics_log = []
        
        for i, rebal_date in enumerate(rebalance_dates):
            print(f"   - Processing rebalance {i+1}/{len(rebalance_dates)}: {rebal_date.date()}...", end="")
            
            universe_df = get_liquid_universe_dataframe(rebal_date, self.engine, self.config['universe'])
            if universe_df.empty:
                print(" ⚠️ Universe empty. Skipping.")
                continue
            
            factors_on_date = self.factor_data_raw[self.factor_data_raw['date'] == rebal_date]
            liquid_factors = factors_on_date[factors_on_date['ticker'].isin(universe_df['ticker'])].copy()
            
            if len(liquid_factors) < 10:
                print(f" ⚠️ Insufficient stocks ({len(liquid_factors)}). Skipping.")
                continue

            target_portfolio = self._calculate_target_portfolio(liquid_factors)
            if target_portfolio.empty:
                print(" ⚠️ Portfolio empty. Skipping.")
                continue
            
            start_period = rebal_date + pd.Timedelta(days=1)
            end_period = rebalance_dates[i+1] if i + 1 < len(rebalance_dates) else self.daily_returns_matrix.index.max()
            holding_dates = self.daily_returns_matrix.index[(self.daily_returns_matrix.index >= start_period) & (self.daily_returns_matrix.index <= end_period)]
            
            daily_holdings.loc[holding_dates] = 0.0
            valid_tickers = target_portfolio.index.intersection(daily_holdings.columns)
            daily_holdings.loc[holding_dates, valid_tickers] = target_portfolio[valid_tickers].values
            
            if i > 0:
                prev_holdings_idx = self.daily_returns_matrix.index.get_loc(rebal_date, method='ffill') - 1
                prev_holdings = daily_holdings.iloc[prev_holdings_idx] if prev_holdings_idx >= 0 else pd.Series(dtype='float64')
            else:
                prev_holdings = pd.Series(dtype='float64')

            turnover = (target_portfolio - prev_holdings.reindex(target_portfolio.index).fillna(0)).abs().sum() / 2.0
            
            diagnostics_log.append({
                'date': rebal_date,
                'universe_size': len(universe_df),
                'portfolio_size': len(target_portfolio),
                'turnover': turnover
            })
            print(f" ✅ Universe: {len(universe_df)}, Portfolio: {len(target_portfolio)}, Turnover: {turnover:.1%}")

        return daily_holdings, pd.DataFrame(diagnostics_log).set_index('date')

    def _calculate_target_portfolio(self, factors_df: pd.DataFrame) -> pd.Series:
        """Constructs the portfolio using the hybrid method."""
        factors_to_combine = self.config['signal']['factors_to_combine']
        
        # Engineer signals (in this case, none are needed for pure value)
        # Re-normalize and combine
        weighted_scores = []
        for factor, weight in factors_to_combine.items():
            scores = factors_df[factor]
            mean, std = scores.mean(), scores.std()
            if std > 1e-8: # Z-score safeguard
                weighted_scores.append(((scores - mean) / std) * weight)
        
        if not weighted_scores: return pd.Series(dtype='float64')
        factors_df['final_signal'] = pd.concat(weighted_scores, axis=1).sum(axis=1)
        
        # Hybrid portfolio construction
        universe_size = len(factors_df)
        if universe_size < 100:
            # Fixed-N for small universes
            portfolio_size = self.config['portfolio']['portfolio_size_small_universe']
            selected_stocks = factors_df.nlargest(portfolio_size, 'final_signal')
        else:
            # Percentile for large universes
            percentile = self.config['portfolio']['selection_percentile']
            score_cutoff = factors_df['final_signal'].quantile(percentile)
            selected_stocks = factors_df[factors_df['final_signal'] >= score_cutoff]
            
        if selected_stocks.empty: return pd.Series(dtype='float64')
        return pd.Series(1.0 / len(selected_stocks), index=selected_stocks['ticker'])

    def _calculate_net_returns(self, daily_holdings: pd.DataFrame) -> pd.Series:
        """Calculates net returns with P0 turnover fix."""
        holdings_shifted = daily_holdings.shift(1).fillna(0.0)
        gross_returns = (holdings_shifted * self.daily_returns_matrix).sum(axis=1)
        
        # P0 Fix: Corrected turnover calculation
        turnover = (holdings_shifted - holdings_shifted.shift(1)).abs().sum(axis=1) / 2.0
        
        costs = turnover * (self.config['transaction_cost_bps'] / 10000)
        net_returns = (gross_returns - costs).rename(self.config['strategy_name'])
        
        print("\n💸 Net returns calculated.")
        print(f"   - Total Gross Return: {(1 + gross_returns).prod() - 1:.2%}")
        print(f"   - Total Net Return: {(1 + net_returns).prod() - 1:.2%}")
        print(f"   - Total Cost Drag: {gross_returns.sum() - net_returns.sum():.2%}")
        
        return net_returns

# ============================================================================
# CELL 5: EXECUTION & OFFICIAL BASELINE REPORTING (FULL TEARSHEET)
# ============================================================================

# --- Analytics Suite with Corrected Benchmark Alignment & Full Tearsheet ---
def calculate_official_metrics(returns: pd.Series, benchmark: pd.Series, periods_per_year: int = 252) -> dict:
    """Calculates a dictionary of institutional performance metrics with corrected alignment."""
    # --- CRITICAL FIX: ALIGN BENCHMARK ---
    first_trade_date = returns.loc[returns.ne(0)].index.min()
    if pd.isna(first_trade_date):
        return {metric: 0.0 for metric in ['Annualized Return (%)', 'Annualized Volatility (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 'Calmar Ratio', 'Information Ratio', 'Beta']}
    
    aligned_returns = returns.loc[first_trade_date:]
    aligned_benchmark = benchmark.loc[first_trade_date:]
    # --- END FIX ---

    n_years = len(aligned_returns) / periods_per_year
    annualized_return = ((1 + aligned_returns).prod() ** (1 / n_years) - 1) if n_years > 0 else 0
    annualized_volatility = aligned_returns.std() * np.sqrt(periods_per_year)
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else 0.0
    
    cumulative_returns = (1 + aligned_returns).cumprod()
    max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0.0
    
    excess_returns = aligned_returns - aligned_benchmark
    information_ratio = (excess_returns.mean() * periods_per_year) / (excess_returns.std() * np.sqrt(periods_per_year)) if excess_returns.std() > 0 else 0.0
    beta = aligned_returns.cov(aligned_benchmark) / aligned_benchmark.var() if aligned_benchmark.var() > 0 else 0.0
    
    return {
        'Annualized Return (%)': annualized_return * 100,
        'Annualized Volatility (%)': annualized_volatility * 100,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown (%)': max_drawdown * 100,
        'Calmar Ratio': calmar_ratio,
        'Information Ratio': information_ratio,
        'Beta': beta
    }

def generate_official_tearsheet(strategy_returns: pd.Series, benchmark_returns: pd.Series, diagnostics: pd.DataFrame, title: str):
    """Generates the full, multi-plot institutional tearsheet with a correctly aligned benchmark."""
    
    # --- CRITICAL FIX: ALIGN BENCHMARK FOR PLOTTING & METRICS ---
    first_trade_date = strategy_returns.loc[strategy_returns.ne(0)].index.min()
    aligned_strategy_returns = strategy_returns.loc[first_trade_date:]
    aligned_benchmark_returns = benchmark_returns.loc[first_trade_date:]
    # --- END FIX ---

    strategy_metrics = calculate_official_metrics(strategy_returns, benchmark_returns)
    benchmark_metrics = calculate_official_metrics(benchmark_returns, benchmark_returns) # Benchmark vs itself
    
    fig = plt.figure(figsize=(18, 26))
    gs = fig.add_gridspec(5, 2, height_ratios=[1.2, 0.8, 0.8, 0.8, 1.2], hspace=0.7, wspace=0.2)
    fig.suptitle(title, fontsize=20, fontweight='bold', color='#2C3E50')

    # 1. Cumulative Performance
    ax1 = fig.add_subplot(gs[0, :])
    (1 + aligned_strategy_returns).cumprod().plot(ax=ax1, label='Strategy Net Returns', color='#16A085', lw=2.5)
    (1 + aligned_benchmark_returns).cumprod().plot(ax=ax1, label='VN-Index (Aligned)', color='#34495E', linestyle='--', lw=2)
    ax1.set_title('Cumulative Performance (Log Scale)', fontweight='bold'); ax1.set_ylabel('Growth of 1 VND'); ax1.set_yscale('log'); ax1.legend(loc='upper left'); ax1.grid(True, which='both', linestyle='--', alpha=0.5)

    # 2. Drawdown
    ax2 = fig.add_subplot(gs[1, :])
    drawdown = ((1 + aligned_strategy_returns).cumprod() / (1 + aligned_strategy_returns).cumprod().cummax() - 1) * 100
    drawdown.plot(ax=ax2, color='#C0392B'); ax2.fill_between(drawdown.index, drawdown, 0, color='#C0392B', alpha=0.1)
    ax2.set_title('Drawdown Analysis', fontweight='bold'); ax2.set_ylabel('Drawdown (%)'); ax2.grid(True, linestyle='--', alpha=0.5)

    # 3. Annual Returns
    ax3 = fig.add_subplot(gs[2, 0])
    strat_annual = aligned_strategy_returns.resample('Y').apply(lambda x: (1+x).prod()-1) * 100
    bench_annual = aligned_benchmark_returns.resample('Y').apply(lambda x: (1+x).prod()-1) * 100
    pd.DataFrame({'Strategy': strat_annual, 'Benchmark': bench_annual}).plot(kind='bar', ax=ax3, color=['#16A085', '#34495E'])
    ax3.set_xticklabels([d.strftime('%Y') for d in strat_annual.index], rotation=45, ha='right'); ax3.set_title('Annual Returns', fontweight='bold'); ax3.grid(True, axis='y', linestyle='--', alpha=0.5)

    # 4. Rolling Sharpe Ratio
    ax4 = fig.add_subplot(gs[2, 1])
    rolling_sharpe = (aligned_strategy_returns.rolling(252).mean() * 252) / (aligned_strategy_returns.rolling(252).std() * np.sqrt(252))
    rolling_sharpe.plot(ax=ax4, color='#E67E22'); ax4.axhline(1.0, color='#27AE60', linestyle='--'); ax4.set_title('1-Year Rolling Sharpe Ratio', fontweight='bold'); ax4.grid(True, linestyle='--', alpha=0.5)

    # 5. Metrics Table
    ax5 = fig.add_subplot(gs[3:, :]); ax5.axis('off')
    summary_data = [['Metric', 'Strategy', 'Benchmark']]
    for key in strategy_metrics.keys():
        summary_data.append([key, f"{strategy_metrics[key]:.2f}", f"{benchmark_metrics.get(key, 0.0):.2f}"])
    
    table = ax5.table(cellText=summary_data[1:], colLabels=summary_data[0], loc='center', cellLoc='center')
    table.auto_set_font_size(False); table.set_fontsize(14); table.scale(1, 2.5)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97]); plt.show()

# --- Instantiate and Run the Baseline Engine ---
try:
    baseline_engine = BaselinePortfolioEngine(
        config=BASELINE_CONFIG,
        factor_data=factor_data_raw,
        returns_matrix=daily_returns_matrix,
        benchmark_returns=benchmark_returns,
        db_engine=engine
    )
    
    baseline_net_returns, baseline_diagnostics = baseline_engine.run_backtest()

    # --- Generate the Official Report ---
    print("\n" + "="*80)
    print("📊 OFFICIAL BASELINE V1.0: PERFORMANCE REPORT")
    print("="*80)
    generate_official_tearsheet(
        baseline_net_returns,
        benchmark_returns,
        baseline_diagnostics,
        "Official Baseline v1.0: Standalone Value (2016-2025)"
    )

except Exception as e:
    print(f"❌ An error occurred during the baseline execution: {e}")
    raise

✅ BaselinePortfolioEngine initialized.
   - Strategy: Official_Baseline_v1.0_Value
   - Period: 2016-03-01 to 2025-07-28

🚀 Starting baseline backtest execution...
   - Generated 38 quarterly rebalance dates.
   - Processing rebalance 1/38: 2016-03-30...Constructing liquid universe for 2016-03-30...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 552 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/12...
  Step 3: Filtering and ranking...
    Total batch results: 552
    Sample result: ('AAA', 41, 6.368384063414634, 758.1949381463414)
    Before filters: 552 stocks
    Trading days range: 1-41 (need >= 37)
    ADTV range: 0.000-190.911B VND (need >= 10.0)
    Stocks passing trading days filter: 315
    Stocks passing ADTV filter: 64
    After filters: 62 stocks
✅ Universe constructed: 62 stocks
  ADTV range: 10.0B - 190.9B VND
  Market cap range: 198.1B - 155696.1B VND
  Adding sector information...
 ✅ Universe: 62, Portfolio: 20, Turnover: 50.0%
   - Processing rebalance 2/38: 2016-06-29...Constructing liquid universe for 2016-06-29...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 566 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/12...
  Step 3: Filtering and ranking...
    Total batch results: 566
    Sample result: ('AAA', 44, 15.89246561818182, 1249.164935972727)
    Before filters: 566 stocks
    Trading days range: 1-44 (need >= 37)
    ADTV range: 0.001-149.550B VND (need >= 10.0)
    Stocks passing trading days filter: 377
    Stocks passing ADTV filter: 71
    After filters: 71 stocks
✅ Universe constructed: 71 stocks
  ADTV range: 10.1B - 149.5B VND
  Market cap range: 285.1B - 168860.7B VND
  Adding sector information...
 ✅ Universe: 71, Portfolio: 20, Turnover: 22.5%
   - Processing rebalance 3/38: 2016-09-29...Constructing liquid universe for 2016-09-29...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 565 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/12...
  Step 3: Filtering and ranking...
    Total batch results: 565
    Sample result: ('AAA', 45, 11.724057446666666, 1687.4416098400006)
    Before filters: 565 stocks
    Trading days range: 1-45 (need >= 37)
    ADTV range: 0.000-345.870B VND (need >= 10.0)
    Stocks passing trading days filter: 389
    Stocks passing ADTV filter: 62
    After filters: 62 stocks
✅ Universe constructed: 62 stocks
  ADTV range: 10.0B - 345.9B VND
  Market cap range: 339.0B - 204339.3B VND
  Adding sector information...
 ✅ Universe: 62, Portfolio: 20, Turnover: 32.5%
   - Processing rebalance 4/38: 2016-12-30...Constructing liquid universe for 2016-12-30...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 576 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/12...
  Step 3: Filtering and ranking...
    Total batch results: 576
    Sample result: ('AAA', 41, 7.417628868292683, 1405.1605287658538)
    Before filters: 576 stocks
    Trading days range: 1-46 (need >= 37)
    ADTV range: 0.000-312.245B VND (need >= 10.0)
    Stocks passing trading days filter: 381
    Stocks passing ADTV filter: 60
    After filters: 57 stocks
✅ Universe constructed: 57 stocks
  ADTV range: 10.0B - 257.0B VND
  Market cap range: 299.0B - 195134.4B VND
  Adding sector information...
 ✅ Universe: 57, Portfolio: 20, Turnover: 22.5%
   - Processing rebalance 5/38: 2017-03-30...Constructing liquid universe for 2017-03-30...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 590 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/12...
  Step 3: Filtering and ranking...
    Total batch results: 590
    Sample result: ('AAA', 41, 28.556135121951222, 1416.4830552682927)
    Before filters: 590 stocks
    Trading days range: 1-41 (need >= 37)
    ADTV range: 0.001-189.465B VND (need >= 10.0)
    Stocks passing trading days filter: 377
    Stocks passing ADTV filter: 74
    After filters: 73 stocks
✅ Universe constructed: 73 stocks
  ADTV range: 10.5B - 189.5B VND
  Market cap range: 322.6B - 193774.8B VND
  Adding sector information...
 ✅ Universe: 73, Portfolio: 20, Turnover: 20.0%
   - Processing rebalance 6/38: 2017-06-29...Constructing liquid universe for 2017-06-29...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 603 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/13...
  Step 3: Filtering and ranking...
    Total batch results: 603
    Sample result: ('AAA', 44, 70.45970019886366, 1811.3177437318182)
    Before filters: 603 stocks
    Trading days range: 1-44 (need >= 37)
    ADTV range: 0.000-163.047B VND (need >= 10.0)
    Stocks passing trading days filter: 407
    Stocks passing ADTV filter: 91
    After filters: 88 stocks
✅ Universe constructed: 88 stocks
  ADTV range: 10.3B - 163.0B VND
  Market cap range: 224.0B - 218285.6B VND
  Adding sector information...
 ✅ Universe: 88, Portfolio: 20, Turnover: 20.0%
   - Processing rebalance 7/38: 2017-09-29...Constructing liquid universe for 2017-09-29...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 618 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/13...
  Step 3: Filtering and ranking...
    Total batch results: 618
    Sample result: ('AAA', 45, 43.5869311111111, 1968.5479346400004)
    Before filters: 618 stocks
    Trading days range: 1-45 (need >= 37)
    ADTV range: 0.000-179.877B VND (need >= 10.0)
    Stocks passing trading days filter: 424
    Stocks passing ADTV filter: 89
    After filters: 88 stocks
✅ Universe constructed: 88 stocks
  ADTV range: 10.2B - 179.9B VND
  Market cap range: 434.9B - 217170.0B VND
  Adding sector information...
 ✅ Universe: 88, Portfolio: 20, Turnover: 22.5%
   - Processing rebalance 8/38: 2017-12-29...Constructing liquid universe for 2017-12-29...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 630 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/13...
  Step 3: Filtering and ranking...
    Total batch results: 630
    Sample result: ('AAA', 46, 45.83964149999999, 2099.0708189826087)
    Before filters: 630 stocks
    Trading days range: 1-46 (need >= 37)
    ADTV range: 0.000-723.486B VND (need >= 10.0)
    Stocks passing trading days filter: 428
    Stocks passing ADTV filter: 95
    After filters: 94 stocks
✅ Universe constructed: 94 stocks
  ADTV range: 10.2B - 723.5B VND
  Market cap range: 440.6B - 268262.7B VND
  Adding sector information...
 ✅ Universe: 94, Portfolio: 20, Turnover: 12.5%
   - Processing rebalance 9/38: 2018-03-30...Constructing liquid universe for 2018-03-30...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 645 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/13...
  Step 3: Filtering and ranking...
    Total batch results: 645
    Sample result: ('AAA', 41, 34.33390243902439, 2298.99967)
    Before filters: 645 stocks
    Trading days range: 1-41 (need >= 37)
    ADTV range: 0.000-417.736B VND (need >= 10.0)
    Stocks passing trading days filter: 401
    Stocks passing ADTV filter: 97
    After filters: 95 stocks
✅ Universe constructed: 95 stocks
  ADTV range: 10.6B - 417.7B VND
  Market cap range: 304.2B - 296549.8B VND
  Adding sector information...
 ✅ Universe: 95, Portfolio: 20, Turnover: 12.5%
   - Processing rebalance 10/38: 2018-06-29...Constructing liquid universe for 2018-06-29...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 647 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/13...
  Step 3: Filtering and ranking...
    Total batch results: 647
    Sample result: ('AAA', 44, 25.543715625, 3345.32951980909)
    Before filters: 647 stocks
    Trading days range: 1-44 (need >= 37)
    ADTV range: 0.000-1114.965B VND (need >= 10.0)
    Stocks passing trading days filter: 411
    Stocks passing ADTV filter: 79
    After filters: 77 stocks
✅ Universe constructed: 77 stocks
  ADTV range: 10.1B - 399.9B VND
  Market cap range: 229.6B - 320538.5B VND
  Adding sector information...
 ✅ Universe: 77, Portfolio: 20, Turnover: 17.5%
   - Processing rebalance 11/38: 2018-09-28...Constructing liquid universe for 2018-09-28...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 655 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 655
    Sample result: ('AAA', 45, 33.14820583333334, 2873.066256266666)
    Before filters: 655 stocks
    Trading days range: 1-45 (need >= 37)
    ADTV range: 0.000-234.621B VND (need >= 10.0)
    Stocks passing trading days filter: 418
    Stocks passing ADTV filter: 85
    After filters: 85 stocks
✅ Universe constructed: 85 stocks
  ADTV range: 10.1B - 234.6B VND
  Market cap range: 580.9B - 328302.6B VND
  Adding sector information...
 ✅ Universe: 85, Portfolio: 20, Turnover: 10.0%
   - Processing rebalance 12/38: 2018-12-28...Constructing liquid universe for 2018-12-28...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 663 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 663
    Sample result: ('AAA', 46, 27.68439130434782, 2572.0935524695647)
    Before filters: 663 stocks
    Trading days range: 1-46 (need >= 37)
    ADTV range: 0.000-253.780B VND (need >= 10.0)
    Stocks passing trading days filter: 404
    Stocks passing ADTV filter: 85
    After filters: 82 stocks
✅ Universe constructed: 82 stocks
  ADTV range: 10.5B - 253.8B VND
  Market cap range: 891.6B - 316157.8B VND
  Adding sector information...
 ✅ Universe: 82, Portfolio: 20, Turnover: 5.0%
   - Processing rebalance 13/38: 2019-03-29...Constructing liquid universe for 2019-03-29...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 664 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 664
    Sample result: ('AAA', 41, 34.701419512195116, 2677.4006002731708)
    Before filters: 664 stocks
    Trading days range: 1-41 (need >= 37)
    ADTV range: 0.000-200.491B VND (need >= 10.0)
    Stocks passing trading days filter: 385
    Stocks passing ADTV filter: 84
    After filters: 82 stocks
✅ Universe constructed: 82 stocks
  ADTV range: 10.3B - 200.5B VND
  Market cap range: 868.3B - 364171.8B VND
  Adding sector information...
 ✅ Universe: 82, Portfolio: 20, Turnover: 15.0%
   - Processing rebalance 14/38: 2019-06-28...Constructing liquid universe for 2019-06-28...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 668 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 668
    Sample result: ('AAA', 43, 56.586420023255805, 3043.3781780093022)
    Before filters: 668 stocks
    Trading days range: 1-43 (need >= 37)
    ADTV range: 0.000-201.426B VND (need >= 10.0)
    Stocks passing trading days filter: 404
    Stocks passing ADTV filter: 75
    After filters: 73 stocks
✅ Universe constructed: 73 stocks
  ADTV range: 10.1B - 201.4B VND
  Market cap range: 655.4B - 384768.2B VND
  Adding sector information...
 ✅ Universe: 73, Portfolio: 20, Turnover: 10.0%
   - Processing rebalance 15/38: 2019-09-27...Constructing liquid universe for 2019-09-27...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 666 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 666
    Sample result: ('AAA', 45, 36.92309141111112, 2856.566710657777)
    Before filters: 666 stocks
    Trading days range: 1-45 (need >= 37)
    ADTV range: 0.000-174.884B VND (need >= 10.0)
    Stocks passing trading days filter: 426
    Stocks passing ADTV filter: 87
    After filters: 86 stocks
✅ Universe constructed: 86 stocks
  ADTV range: 10.6B - 174.9B VND
  Market cap range: 788.6B - 406880.6B VND
  Adding sector information...
 ✅ Universe: 86, Portfolio: 20, Turnover: 12.5%
   - Processing rebalance 16/38: 2019-12-30...Constructing liquid universe for 2019-12-30...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 667 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 667
    Sample result: ('AAA', 46, 35.899058913043476, 2463.2326981652172)
    Before filters: 667 stocks
    Trading days range: 1-46 (need >= 37)
    ADTV range: 0.000-236.615B VND (need >= 10.0)
    Stocks passing trading days filter: 405
    Stocks passing ADTV filter: 83
    After filters: 81 stocks
✅ Universe constructed: 81 stocks
  ADTV range: 10.1B - 236.6B VND
  Market cap range: 342.3B - 393224.6B VND
  Adding sector information...
 ✅ Universe: 81, Portfolio: 20, Turnover: 17.5%
   - Processing rebalance 17/38: 2020-03-30...Constructing liquid universe for 2020-03-30...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 674 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 674
    Sample result: ('AAA', 43, 24.245858627906976, 1987.9502329432557)
    Before filters: 674 stocks
    Trading days range: 1-43 (need >= 37)
    ADTV range: 0.000-213.480B VND (need >= 10.0)
    Stocks passing trading days filter: 394
    Stocks passing ADTV filter: 79
    After filters: 79 stocks
✅ Universe constructed: 79 stocks
  ADTV range: 10.1B - 213.5B VND
  Market cap range: 536.4B - 342364.9B VND
  Adding sector information...
 ✅ Universe: 79, Portfolio: 20, Turnover: 25.0%
   - Processing rebalance 18/38: 2020-06-29...Constructing liquid universe for 2020-06-29...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 676 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 676
    Sample result: ('AAA', 44, 30.83602238636364, 2167.6251506727276)
    Before filters: 676 stocks
    Trading days range: 1-44 (need >= 37)
    ADTV range: 0.000-683.076B VND (need >= 10.0)
    Stocks passing trading days filter: 454
    Stocks passing ADTV filter: 115
    After filters: 114 stocks
✅ Universe constructed: 114 stocks
  ADTV range: 10.1B - 683.1B VND
  Market cap range: 301.1B - 321092.6B VND
  Adding sector information...
 ✅ Universe: 114, Portfolio: 23, Turnover: 27.8%
   - Processing rebalance 19/38: 2020-09-29...Constructing liquid universe for 2020-09-29...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 685 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 685
    Sample result: ('AAA', 45, 32.20547535555555, 2555.75437624)
    Before filters: 685 stocks
    Trading days range: 1-45 (need >= 37)
    ADTV range: 0.000-329.080B VND (need >= 10.0)
    Stocks passing trading days filter: 470
    Stocks passing ADTV filter: 120
    After filters: 117 stocks
✅ Universe constructed: 117 stocks
  ADTV range: 10.2B - 329.1B VND
  Market cap range: 230.6B - 306633.5B VND
  Adding sector information...
 ✅ Universe: 117, Portfolio: 23, Turnover: 15.2%
   - Processing rebalance 20/38: 2020-12-30...Constructing liquid universe for 2020-12-30...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 695 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 695
    Sample result: ('AAA', 46, 33.98864422826088, 2755.608720400001)
    Before filters: 695 stocks
    Trading days range: 1-46 (need >= 37)
    ADTV range: 0.000-804.722B VND (need >= 10.0)
    Stocks passing trading days filter: 498
    Stocks passing ADTV filter: 153
    After filters: 150 stocks
✅ Universe constructed: 150 stocks
  ADTV range: 10.1B - 804.7B VND
  Market cap range: 348.5B - 356265.5B VND
  Adding sector information...
 ✅ Universe: 150, Portfolio: 30, Turnover: 29.1%
   - Processing rebalance 21/38: 2021-03-30...Constructing liquid universe for 2021-03-30...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 707 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 707
    Sample result: ('AAA', 41, 49.53687585365853, 3277.450152324391)
    Before filters: 707 stocks
    Trading days range: 1-41 (need >= 37)
    ADTV range: 0.001-999.126B VND (need >= 10.0)
    Stocks passing trading days filter: 507
    Stocks passing ADTV filter: 170
    After filters: 168 stocks
✅ Universe constructed: 168 stocks
  ADTV range: 10.1B - 999.1B VND
  Market cap range: 250.0B - 360443.4B VND
  Adding sector information...
 ✅ Universe: 168, Portfolio: 33, Turnover: 26.8%
   - Processing rebalance 22/38: 2021-06-29...Constructing liquid universe for 2021-06-29...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 710 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 710
    Sample result: ('AAA', 44, 144.37945613636361, 4268.378297297727)
    Before filters: 710 stocks
    Trading days range: 1-44 (need >= 37)
    ADTV range: 0.001-2225.420B VND (need >= 10.0)
    Stocks passing trading days filter: 552
    Stocks passing ADTV filter: 185
    After filters: 183 stocks
✅ Universe constructed: 183 stocks
  ADTV range: 10.0B - 2225.4B VND
  Market cap range: 404.4B - 414770.6B VND
  Adding sector information...
 ✅ Universe: 183, Portfolio: 36, Turnover: 19.7%
   - Processing rebalance 23/38: 2021-09-29...Constructing liquid universe for 2021-09-29...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 715 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 715
    Sample result: ('AAA', 44, 112.56472204545457, 5153.735775311365)
    Before filters: 715 stocks
    Trading days range: 1-44 (need >= 37)
    ADTV range: 0.000-1356.183B VND (need >= 10.0)
    Stocks passing trading days filter: 577
    Stocks passing ADTV filter: 235
    After filters: 235 stocks
✅ Universe constructed: 200 stocks
  ADTV range: 14.9B - 1356.2B VND
  Market cap range: 208.1B - 366740.5B VND
  Adding sector information...
 ✅ Universe: 200, Portfolio: 39, Turnover: 20.6%
   - Processing rebalance 24/38: 2021-12-30...Constructing liquid universe for 2021-12-30...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 719 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 719
    Sample result: ('AAA', 45, 148.09710811111108, 5877.634452977779)
    Before filters: 719 stocks
    Trading days range: 1-46 (need >= 37)
    ADTV range: 0.004-1271.870B VND (need >= 10.0)
    Stocks passing trading days filter: 622
    Stocks passing ADTV filter: 281
    After filters: 278 stocks
✅ Universe constructed: 200 stocks
  ADTV range: 23.4B - 1271.9B VND
  Market cap range: 447.0B - 375177.6B VND
  Adding sector information...
 ✅ Universe: 200, Portfolio: 39, Turnover: 20.5%
   - Processing rebalance 25/38: 2022-03-30...Constructing liquid universe for 2022-03-30...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 718 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 718
    Sample result: ('AAA', 41, 100.7600319512195, 5832.827116331708)
    Before filters: 718 stocks
    Trading days range: 2-41 (need >= 37)
    ADTV range: 0.001-1116.303B VND (need >= 10.0)
    Stocks passing trading days filter: 577
    Stocks passing ADTV filter: 256
    After filters: 255 stocks
✅ Universe constructed: 200 stocks
  ADTV range: 16.4B - 1116.3B VND
  Market cap range: 398.0B - 406396.2B VND
  Adding sector information...
 ✅ Universe: 200, Portfolio: 39, Turnover: 19.2%
   - Processing rebalance 26/38: 2022-06-29...Constructing liquid universe for 2022-06-29...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 720 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 720
    Sample result: ('AAA', 44, 49.13806886363637, 3979.1623165818182)
    Before filters: 720 stocks
    Trading days range: 1-44 (need >= 37)
    ADTV range: 0.000-733.900B VND (need >= 10.0)
    Stocks passing trading days filter: 553
    Stocks passing ADTV filter: 181
    After filters: 180 stocks
✅ Universe constructed: 180 stocks
  ADTV range: 10.0B - 733.9B VND
  Market cap range: 470.9B - 366963.6B VND
  Adding sector information...
 ✅ Universe: 180, Portfolio: 35, Turnover: 23.1%
   - Processing rebalance 27/38: 2022-09-29...Constructing liquid universe for 2022-09-29...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 721 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 721
    Sample result: ('AAA', 44, 45.93358894318182, 4507.3638301090905)
    Before filters: 721 stocks
    Trading days range: 1-44 (need >= 37)
    ADTV range: 0.000-599.143B VND (need >= 10.0)
    Stocks passing trading days filter: 545
    Stocks passing ADTV filter: 182
    After filters: 181 stocks
✅ Universe constructed: 181 stocks
  ADTV range: 10.4B - 599.1B VND
  Market cap range: 274.7B - 377429.0B VND
  Adding sector information...
 ✅ Universe: 181, Portfolio: 36, Turnover: 17.6%
   - Processing rebalance 28/38: 2022-12-30...Constructing liquid universe for 2022-12-30...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 717 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 717
    Sample result: ('AAA', 46, 21.876608707608707, 2738.9967638400008)
    Before filters: 717 stocks
    Trading days range: 1-46 (need >= 37)
    ADTV range: 0.000-698.257B VND (need >= 10.0)
    Stocks passing trading days filter: 529
    Stocks passing ADTV filter: 148
    After filters: 147 stocks
✅ Universe constructed: 147 stocks
  ADTV range: 10.4B - 698.3B VND
  Market cap range: 508.7B - 364136.3B VND
  Adding sector information...
 ✅ Universe: 147, Portfolio: 29, Turnover: 16.7%
   - Processing rebalance 29/38: 2023-03-30...Constructing liquid universe for 2023-03-30...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 712 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 712
    Sample result: ('AAA', 45, 30.817742834444452, 3314.659679872)
    Before filters: 712 stocks
    Trading days range: 1-45 (need >= 37)
    ADTV range: 0.000-501.358B VND (need >= 10.0)
    Stocks passing trading days filter: 518
    Stocks passing ADTV filter: 137
    After filters: 136 stocks
✅ Universe constructed: 136 stocks
  ADTV range: 10.3B - 501.4B VND
  Market cap range: 402.7B - 434865.7B VND
  Adding sector information...
 ✅ Universe: 136, Portfolio: 27, Turnover: 15.5%
   - Processing rebalance 30/38: 2023-06-29...Constructing liquid universe for 2023-06-29...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 717 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 717
    Sample result: ('AAA', 43, 67.9175810960465, 4225.022191255815)
    Before filters: 717 stocks
    Trading days range: 1-43 (need >= 37)
    ADTV range: 0.000-538.011B VND (need >= 10.0)
    Stocks passing trading days filter: 535
    Stocks passing ADTV filter: 187
    After filters: 186 stocks
✅ Universe constructed: 186 stocks
  ADTV range: 10.1B - 538.0B VND
  Market cap range: 374.7B - 455015.0B VND
  Adding sector information...
 ✅ Universe: 186, Portfolio: 37, Turnover: 34.7%
   - Processing rebalance 31/38: 2023-09-29...Constructing liquid universe for 2023-09-29...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 716 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 716
    Sample result: ('AAA', 44, 88.42057762522725, 4172.091721003634)
    Before filters: 716 stocks
    Trading days range: 2-44 (need >= 37)
    ADTV range: 0.000-1009.327B VND (need >= 10.0)
    Stocks passing trading days filter: 567
    Stocks passing ADTV filter: 207
    After filters: 205 stocks
✅ Universe constructed: 200 stocks
  ADTV range: 10.7B - 1009.3B VND
  Market cap range: 403.7B - 498242.3B VND
  Adding sector information...
 ✅ Universe: 200, Portfolio: 40, Turnover: 15.5%
   - Processing rebalance 32/38: 2023-12-29...Constructing liquid universe for 2023-12-29...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 710 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 710
    Sample result: ('AAA', 46, 21.983487449999995, 3496.814400584348)
    Before filters: 710 stocks
    Trading days range: 1-46 (need >= 37)
    ADTV range: 0.000-716.519B VND (need >= 10.0)
    Stocks passing trading days filter: 553
    Stocks passing ADTV filter: 154
    After filters: 152 stocks
✅ Universe constructed: 152 stocks
  ADTV range: 10.2B - 716.5B VND
  Market cap range: 441.7B - 475911.1B VND
  Adding sector information...
 ✅ Universe: 152, Portfolio: 31, Turnover: 23.8%
   - Processing rebalance 33/38: 2024-03-29...Constructing liquid universe for 2024-03-29...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 714 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 714
    Sample result: ('AAA', 41, 51.41185883292683, 4149.543035239025)
    Before filters: 714 stocks
    Trading days range: 1-41 (need >= 37)
    ADTV range: 0.000-911.981B VND (need >= 10.0)
    Stocks passing trading days filter: 528
    Stocks passing ADTV filter: 170
    After filters: 167 stocks
✅ Universe constructed: 167 stocks
  ADTV range: 10.0B - 912.0B VND
  Market cap range: 313.4B - 520153.5B VND
  Adding sector information...
 ✅ Universe: 167, Portfolio: 34, Turnover: 19.4%
   - Processing rebalance 34/38: 2024-06-28...Constructing liquid universe for 2024-06-28...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 711 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 711
    Sample result: ('AAA', 43, 66.10686307418604, 4305.7443406437205)
    Before filters: 711 stocks
    Trading days range: 1-43 (need >= 37)
    ADTV range: 0.000-849.669B VND (need >= 10.0)
    Stocks passing trading days filter: 547
    Stocks passing ADTV filter: 194
    After filters: 191 stocks
✅ Universe constructed: 191 stocks
  ADTV range: 10.1B - 849.7B VND
  Market cap range: 385.1B - 499092.9B VND
  Adding sector information...
 ✅ Universe: 191, Portfolio: 38, Turnover: 21.0%
   - Processing rebalance 35/38: 2024-09-27...Constructing liquid universe for 2024-09-27...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 707 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 707
    Sample result: ('AAA', 44, 50.1624685509091, 3958.6261672145465)
    Before filters: 707 stocks
    Trading days range: 1-44 (need >= 37)
    ADTV range: 0.000-589.261B VND (need >= 10.0)
    Stocks passing trading days filter: 524
    Stocks passing ADTV filter: 155
    After filters: 153 stocks
✅ Universe constructed: 153 stocks
  ADTV range: 10.0B - 589.3B VND
  Market cap range: 401.1B - 502294.2B VND
  Adding sector information...
 ✅ Universe: 153, Portfolio: 31, Turnover: 23.7%
   - Processing rebalance 36/38: 2024-12-30...Constructing liquid universe for 2024-12-30...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 702 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 702
    Sample result: ('AAA', 46, 13.785921829347828, 3290.8016885008706)
    Before filters: 702 stocks
    Trading days range: 1-46 (need >= 37)
    ADTV range: 0.000-753.529B VND (need >= 10.0)
    Stocks passing trading days filter: 534
    Stocks passing ADTV filter: 158
    After filters: 156 stocks
✅ Universe constructed: 156 stocks
  ADTV range: 10.1B - 753.5B VND
  Market cap range: 471.1B - 517221.8B VND
  Adding sector information...
 ✅ Universe: 156, Portfolio: 31, Turnover: 14.5%
   - Processing rebalance 37/38: 2025-03-28...Constructing liquid universe for 2025-03-28...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 699 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 699
    Sample result: ('AAA', 41, 15.05166797317073, 3316.5575846868296)
    Before filters: 699 stocks
    Trading days range: 1-41 (need >= 37)
    ADTV range: 0.000-820.823B VND (need >= 10.0)
    Stocks passing trading days filter: 510
    Stocks passing ADTV filter: 164
    After filters: 163 stocks
✅ Universe constructed: 163 stocks
  ADTV range: 10.2B - 820.8B VND
  Market cap range: 319.8B - 529831.9B VND
  Adding sector information...
 ✅ Universe: 163, Portfolio: 33, Turnover: 18.8%
   - Processing rebalance 38/38: 2025-06-26...Constructing liquid universe for 2025-06-26...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 696 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 696
    Sample result: ('AAA', 43, 13.450654152790696, 2756.3769182511624)
    Before filters: 696 stocks
    Trading days range: 1-43 (need >= 37)
    ADTV range: 0.000-917.006B VND (need >= 10.0)
    Stocks passing trading days filter: 526
    Stocks passing ADTV filter: 164
    After filters: 164 stocks
✅ Universe constructed: 164 stocks
  ADTV range: 10.3B - 917.0B VND
  Market cap range: 439.2B - 474971.5B VND
  Adding sector information...
 ✅ Universe: 164, Portfolio: 32, Turnover: 10.6%

💸 Net returns calculated.
   - Total Gross Return: 288.18%
   - Total Net Return: 272.44%
   - Total Cost Drag: 4.16%
✅ Baseline backtest execution complete.

================================================================================
📊 OFFICIAL BASELINE V1.0: PERFORMANCE REPORT
================================================================================

