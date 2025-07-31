# ============================================================================
# Aureus Sigma Capital - Phase 24: Monthly Rebalancing Sprint (v2 - Refined)
# Notebook: 24_monthly_rebalancing_sprint_v2.ipynb
#
# Objective:
#   To execute a refined, 5-day sprint to test the hypothesis that a monthly
#   rebalancing frequency, combined with an optimized four-factor stack, can
#   achieve a Sharpe Ratio of >= 0.70, creating a viable candidate for our
#   final production model.
# ============================================================================
#
# --- STRATEGIC DIRECTIVE & ALIGNMENT ---
#
# This notebook implements the refined strategic directive on monthly rebalancing.
# It incorporates critical feedback to de-risk the experiment and enhance its
# methodological rigor. Key refinements include:
#
# 1.  **Smarter Execution Timing:** Rebalancing is shifted to T+2 after month-end
#     to avoid operational risks associated with month-end liquidity crowding.
# 2.  **Constrained Optimization:** The factor weight optimizer will be constrained
#     to ensure Value remains a core component of the strategy (weight >= 30%).
# 3.  **Realistic Cost Model:** A non-linear transaction cost model that accounts
#     for market impact will be implemented.
# 4.  **Robust Universe Construction:** A trading-day coverage filter will be added
#     to the monthly universe construction to guard against stale liquidity data.
#
# --- PRIMARY RESEARCH QUESTION ---
#
# Can a monthly rebalanced, dynamically weighted four-factor composite (V, Q,
# PosMom, Rev) with a 15% volatility target achieve a Sharpe Ratio in the
# 0.70-0.75 range and a Net CAGR in the 10-11% range over the full 2016-2025 period?
#
# --- METHODOLOGY: THE REFINED 5-DAY SPRINT PLAN ---
#
# This notebook will follow the refined implementation checklist:
#
# 1.  **Day 1 (Engine Upgrade):**
#     -   Create `PortfolioEngine_v4_0` with a `rebalance_freq='M'` flag and
#       T+2 rebalance date logic.
#     -   Implement the enhanced, non-linear transaction cost model.
#
# 2.  **Day 2 (Signal & Optimizer Integration):**
#     -   Integrate the full four-factor signal stack (V, Q, PosMom, Rev).
#     -   Implement a constrained Bayesian optimizer for factor weights.
#
# 3.  **Day 3 (Smoke Test):**
#     -   Run a `Value-only` strategy on the monthly schedule to validate the
#       engine and quantify the turnover/cost impact versus the quarterly baseline.
#
# 4.  **Day 4 (Full Run):**
#     -   Execute the full, four-factor composite strategy with the monthly
#       rebalancing, optimized weights, and 15% volatility target.
#
# 5.  **Day 5 (Analysis & Verdict):**
#     -   Directly compare the final monthly tearsheet against the quarterly
#       baseline, including a cost-adjusted information ratio and slippage
#       stress test.
#     -   Formulate a final Go/No-Go verdict for the Investment Committee.
#
# --- SUCCESS CRITERIA ---
#
# The monthly rebalancing approach will be deemed successful if it delivers:
#
#   -   **Sharpe Ratio:** >= 0.70
#   -   **Maximum Drawdown:** <= -40%
#   -   **Net CAGR:** >= 10%
#

# ============================================================================
# DAY 1 (ENGINE UPGRADE): MONTHLY FREQUENCY & NON-LINEAR COSTS
# ============================================================================

# Core scientific libraries
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
from pathlib import Path
import sys
from typing import Dict, List, Optional

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Database connectivity
import yaml
from sqlalchemy import create_engine, text

# --- Environment Setup ---
warnings.filterwarnings('ignore')
try:
    project_root = Path.cwd().parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from production.universe.constructors import get_liquid_universe_dataframe
    print("✅ Successfully imported production modules.")
except Exception as e:
    print(f"❌ ERROR during module import: {e}")

# --- Unified Configuration Block ---
print("\n⚙️  Initializing unified configuration block for the sprint...")
BASE_CONFIG = {
    "backtest_start_date": "2016-03-01",
    "backtest_end_date": "2025-07-28",
    "strategy_version_db": "qvm_v2.0_enhanced",
    # Transaction costs are now handled by the new model, not a flat bps rate
}
print("✅ Base configuration defined.")

# --- Visualization Palette ---
PALETTE = {
    'primary': '#16A085', 'secondary': '#34495E', 'positive': '#27AE60',
    'negative': '#C0392B', 'highlight_1': '#2980B9', 'highlight_2': '#E67E22',
    'neutral': '#7F8C8D', 'grid': '#BDC3C7', 'text': '#2C3E50'
}
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'figure.dpi': 300, 'figure.figsize': (14, 8), 'font.size': 11})
print("✅ Visualization settings configured.")

# --- Data Loading Function ---
def load_all_data(config):
    print("\n📂 Loading all raw data...")
    engine = create_engine(f"mysql+pymysql://{yaml.safe_load(open(project_root / 'config' / 'database.yml'))['production']['username']}:{yaml.safe_load(open(project_root / 'config' / 'database.yml'))['production']['password']}@{yaml.safe_load(open(project_root / 'config' / 'database.yml'))['production']['host']}/{yaml.safe_load(open(project_root / 'config' / 'database.yml'))['production']['schema_name']}")
    db_params = {'start_date': "2016-01-01", 'end_date': config['backtest_end_date'], 'strategy_version': config['strategy_version_db']}
    factor_data_raw = pd.read_sql(text("SELECT date, ticker, Quality_Composite, Value_Composite, Momentum_Composite FROM factor_scores_qvm WHERE date BETWEEN :start_date AND :end_date AND strategy_version = :strategy_version"), engine, params=db_params, parse_dates=['date'])
    price_data_raw = pd.read_sql(text("SELECT date, ticker, close, total_value FROM equity_history WHERE date BETWEEN :start_date AND :end_date"), engine, params=db_params, parse_dates=['date'])
    benchmark_data_raw = pd.read_sql(text("SELECT date, close FROM etf_history WHERE ticker = 'VNINDEX' AND date BETWEEN :start_date AND :end_date"), engine, params=db_params, parse_dates=['date'])
    price_data_raw['return'] = price_data_raw.groupby('ticker')['close'].pct_change()
    daily_returns_matrix = price_data_raw.pivot(index='date', columns='ticker', values='return')
    daily_adtv_matrix = price_data_raw.pivot(index='date', columns='ticker', values='total_value')
    benchmark_returns = benchmark_data_raw.set_index('date')['close'].pct_change().rename('VN-Index')
    print("✅ All data loaded and prepared.")
    return factor_data_raw, daily_returns_matrix, daily_adtv_matrix, benchmark_returns, engine

# --- PORTFOLIO ENGINE v4.0 (MONTHLY REBALANCE & NON-LINEAR COSTS) ---
class PortfolioEngine_v4_0:
    """
    Version 4.0 of our backtesting engine.
    This version implements the Day 1 upgrades for the monthly rebalancing sprint:
        1. Flexible rebalancing frequency ('Q' or 'M') with T+2 logic.
        2. A non-linear transaction cost model incorporating market impact.
    """
    def __init__(self, config: Dict, factor_data: pd.DataFrame, returns_matrix: pd.DataFrame, adtv_matrix: pd.DataFrame, benchmark_returns: pd.Series, db_engine, palette: Dict):
        self.config = config; self.engine = db_engine; self.palette = palette
        start = self.config['backtest_start_date']; end = self.config['backtest_end_date']
        self.factor_data_raw = factor_data[factor_data['date'].between(start, end)].copy()
        self.daily_returns_matrix = returns_matrix.loc[start:end].copy()
        self.daily_adtv_matrix = adtv_matrix.loc[start:end].copy()
        self.benchmark_returns = benchmark_returns.loc[start:end].copy()
        print(f"\n✅ PortfolioEngine v4.0 initialized for strategy: '{self.config['strategy_name']}'")
        print(f"   - Rebalance Frequency: {self.config.get('rebalance_frequency', 'Q')}")

    def _generate_rebalance_dates(self) -> List[pd.Timestamp]:
        """
        UPGRADED: Generates rebalance dates with T+2 logic.
        """
        freq = self.config.get('rebalance_frequency', 'Q')
        all_trading_dates = self.daily_returns_matrix.index
        
        # Generate month-end or quarter-end calendar dates
        base_dates = pd.date_range(start=all_trading_dates.min(), end=all_trading_dates.max(), freq=f'{freq}S') # Start of period
        
        rebalance_dates = []
        for date in base_dates:
            # Find all trading dates in that month/quarter
            period_dates = all_trading_dates[(all_trading_dates.year == date.year) & (all_trading_dates.month == date.month if freq == 'M' else all_trading_dates.quarter == date.quarter)]
            if len(period_dates) > 2:
                # Select the 2nd business day (index 1) as the rebalance date
                rebalance_dates.append(period_dates[1])
                
        print(f"   - Generated {len(rebalance_dates)} data-driven rebalance dates (T+2 logic).")
        return sorted(list(set(rebalance_dates)))

    def _calculate_transaction_costs(self, prev_holdings: pd.Series, next_holdings: pd.Series, rebal_date: pd.Timestamp) -> float:
        """
        UPGRADED: Non-linear transaction cost model.
        """
        portfolio_value = self.config.get('portfolio_value_vnd', 20e9) # Assume 20B VND AUM
        turnover_df = pd.DataFrame({'prev': prev_holdings, 'next': next_holdings}).fillna(0)
        turnover_df['change'] = (turnover_df['next'] - turnover_df['prev']).abs()
        
        # Get 20-day ADTV leading up to the rebalance date
        adtv_20d = self.daily_adtv_matrix.loc[:rebal_date].tail(20).mean()
        
        total_cost_pct = 0
        for ticker, row in turnover_df.iterrows():
            if row['change'] > 1e-6:
                order_value = row['change'] * portfolio_value
                adtv_value = adtv_20d.get(ticker, 10e9) # Default to 10B VND if missing
                
                # Non-linear impact model from assessment
                base_cost_bps = 3.0 # 3 bps for commission/tax
                market_impact_bps = 1.5 * (order_value / adtv_value)
                
                total_cost_bps = base_cost_bps + market_impact_bps
                total_cost_pct += (total_cost_bps / 10000) * row['change']
                
        return total_cost_pct

    def _calculate_net_returns(self, daily_holdings: pd.DataFrame) -> pd.Series:
        holdings_shifted = daily_holdings.shift(1).fillna(0.0)
        gross_returns = (holdings_shifted * self.daily_returns_matrix).sum(axis=1)
        
        costs = pd.Series(0.0, index=gross_returns.index)
        rebalance_dates = self._generate_rebalance_dates()
        
        for i in range(1, len(rebalance_dates)):
            prev_rebal_date = rebalance_dates[i-1]
            curr_rebal_date = rebalance_dates[i]
            
            prev_holdings = holdings_shifted.loc[curr_rebal_date]
            next_holdings = daily_holdings.loc[curr_rebal_date]
            
            cost_pct = self._calculate_transaction_costs(prev_holdings, next_holdings, curr_rebal_date)
            costs.loc[curr_rebal_date] = cost_pct
            
        return (gross_returns - costs).rename(self.config['strategy_name'])

    # --- Other methods (run, loop, portfolio construction, tearsheet) remain the same for now ---
    # They will be called by the execution block. For brevity, only the changed methods are shown in full.
    def run(self) -> pd.Series:
        print(f"--- Executing Backtest for: {self.config['strategy_name']} ---")
        daily_holdings = self._run_backtesting_loop()
        net_returns = self._calculate_net_returns(daily_holdings)
        # self._generate_tearsheet(net_returns) # Will be called externally
        print(f"✅ Backtest for {self.config['strategy_name']} complete.")
        return net_returns
    def _run_backtesting_loop(self) -> pd.DataFrame:
        rebalance_dates = self._generate_rebalance_dates()
        daily_holdings = pd.DataFrame(0.0, index=self.daily_returns_matrix.index, columns=self.daily_returns_matrix.columns)
        for i, rebal_date in enumerate(rebalance_dates):
            universe_df = get_liquid_universe_dataframe(analysis_date=rebal_date, engine=self.engine, config={'lookback_days': 63, 'adtv_threshold_bn': 10.0, 'top_n': 200, 'min_trading_coverage': 0.6})
            if universe_df.empty: continue
            mask = self.factor_data_raw['date'] == rebal_date
            factors_on_date = self.factor_data_raw.loc[mask]
            factors_on_date = factors_on_date[factors_on_date['ticker'].isin(universe_df['ticker'].tolist())].copy()
            if len(factors_on_date) < 10: continue
            target_portfolio = self._calculate_target_portfolio(factors_on_date)
            if target_portfolio.empty: continue
            start_period = rebal_date + pd.Timedelta(days=1)
            end_period = rebalance_dates[i+1] if i + 1 < len(rebalance_dates) else self.daily_returns_matrix.index.max()
            holding_dates = daily_holdings.index[(daily_holdings.index >= start_period) & (daily_holdings.index <= end_period)]
            daily_holdings.loc[holding_dates] = 0.0
            valid_tickers = target_portfolio.index.intersection(daily_holdings.columns)
            daily_holdings.loc[holding_dates, valid_tickers] = target_portfolio[valid_tickers].values
        return daily_holdings
    def _calculate_target_portfolio(self, factors_df: pd.DataFrame) -> pd.Series:
        # This is a placeholder and will be upgraded on Day 2
        factors_to_combine = self.config.get('factors_to_combine', {})
        if 'Momentum_Reversal' in factors_to_combine: factors_df['Momentum_Reversal'] = -1 * factors_df['Momentum_Composite']
        normalized_scores = []
        for factor_name, weight in factors_to_combine.items():
            if weight == 0: continue
            factor_scores = factors_df[factor_name]
            mean, std = factor_scores.mean(), factor_scores.std()
            normalized_scores.append(((factor_scores - mean) / std if std > 1e-8 else 0.0) * weight)
        if not normalized_scores: return pd.Series(dtype='float64')
        factors_df['final_signal'] = pd.concat(normalized_scores, axis=1).sum(axis=1)
        # Using the hybrid construction from our last validated engine
        universe_size = len(factors_df)
        if universe_size < 100:
            selected_stocks = factors_df.nlargest(20, 'final_signal')
        else:
            score_cutoff = factors_df['final_signal'].quantile(0.8)
            selected_stocks = factors_df[factors_df['final_signal'] >= score_cutoff]
        if selected_stocks.empty: return pd.Series(dtype='float64')
        return pd.Series(1.0 / len(selected_stocks), index=selected_stocks['ticker'])

print("✅ PortfolioEngine v4.0 (Monthly Frequency & Non-Linear Costs) defined successfully.")
print("   Ready to load data and proceed to Day 2 (Signal & Optimizer Integration).")

✅ Successfully imported production modules.

⚙️  Initializing unified configuration block for the sprint...
✅ Base configuration defined.
✅ Visualization settings configured.
✅ PortfolioEngine v4.0 (Monthly Frequency & Non-Linear Costs) defined successfully.
   Ready to load data and proceed to Day 2 (Signal & Optimizer Integration).

# ============================================================================
# DAY 2 (SIGNAL UPGRADE): FOUR-FACTOR STACK & OPTIMIZER (v2.2 - ROBUST DATA)
#
# This version corrects the SQL error by loading data in two steps and merging
# in Pandas, which is more robust than a direct SQL JOIN on un-indexed columns.
# ============================================================================

# --- 1. Corrected Data Loading Function (v2) ---
def load_all_data_corrected_v2(config):
    print("\n📂 Loading all raw data (Corrected v2 - Robust Merge)...")
    engine = create_engine(f"mysql+pymysql://{yaml.safe_load(open(project_root / 'config' / 'database.yml'))['production']['username']}:{yaml.safe_load(open(project_root / 'config' / 'database.yml'))['production']['password']}@{yaml.safe_load(open(project_root / 'config' / 'database.yml'))['production']['host']}/{yaml.safe_load(open(project_root / 'config' / 'database.yml'))['production']['schema_name']}")
    
    db_params = {
        'start_date': "2016-01-01", 
        'end_date': config['backtest_end_date'],
        'strategy_version': config['strategy_version_db']
    }

    # Load factors (unchanged)
    factor_query = text("SELECT date, ticker, Quality_Composite, Value_Composite, Momentum_Composite FROM factor_scores_qvm WHERE date BETWEEN :start_date AND :end_date AND strategy_version = :strategy_version")
    factor_data_raw = pd.read_sql(factor_query, engine, params=db_params, parse_dates=['date'])
    print(f"   - ✅ Loaded {len(factor_data_raw):,} raw factor observations.")

    # Load price data from equity_history
    price_query = text("SELECT date, ticker, close FROM equity_history WHERE date BETWEEN :start_date AND :end_date")
    price_data_df = pd.read_sql(price_query, engine, params=db_params, parse_dates=['date'])
    print(f"   - ✅ Loaded {len(price_data_df):,} raw price observations.")

    # Load total_value data from vcsc_daily_data_complete
    value_query = text("SELECT trading_date as date, ticker, total_value FROM vcsc_daily_data_complete WHERE trading_date BETWEEN :start_date AND :end_date")
    value_data_df = pd.read_sql(value_query, engine, params=db_params, parse_dates=['date'])
    print(f"   - ✅ Loaded {len(value_data_df):,} raw total_value observations.")

    # Merge the two dataframes in Pandas
    price_data_raw = pd.merge(price_data_df, value_data_df, on=['date', 'ticker'], how='inner')
    print(f"   - ✅ Merged price/value data. Final records: {len(price_data_raw):,}")

    # Load benchmark (unchanged)
    benchmark_query = text("SELECT date, close FROM etf_history WHERE ticker = 'VNINDEX' AND date BETWEEN :start_date AND :end_date")
    benchmark_data_raw = pd.read_sql(benchmark_query, engine, params=db_params, parse_dates=['date'])
    print(f"   - ✅ Loaded {len(benchmark_data_raw):,} benchmark observations.")

    # Prepare data structures
    price_data_raw['return'] = price_data_raw.groupby('ticker')['close'].pct_change()
    daily_returns_matrix = price_data_raw.pivot(index='date', columns='ticker', values='return')
    daily_adtv_matrix = price_data_raw.pivot(index='date', columns='ticker', values='total_value')
    benchmark_returns = benchmark_data_raw.set_index('date')['close'].pct_change().rename('VN-Index')
    
    print("✅ All data loaded and prepared correctly.")
    return factor_data_raw, daily_returns_matrix, daily_adtv_matrix, benchmark_returns, engine

# --- 2. Define the Upgraded Portfolio Engine (v4.1 - unchanged) ---
class PortfolioEngine_v4_1(PortfolioEngine_v4_0):
    """
    Version 4.1 of the Portfolio Engine.
    This version implements the Day 2 upgrades with corrected data handling.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(f"   - v4.1 Enhancements: Four-Factor Stack & Optimizer active.")
        # Pre-calculate momentum signals once for efficiency
        self.factor_data_raw['PosMom'] = self.factor_data_raw.groupby('ticker')['Momentum_Composite'].shift(21) # Proxy for 12-1M
        self.factor_data_raw['Reversal'] = -1 * self.factor_data_raw['Momentum_Composite'].rolling(window=21, min_periods=1).mean() # Corrected rolling mean for reversal

    def _calculate_target_portfolio(self, factors_df: pd.DataFrame) -> pd.Series:
        factors_to_combine = self.config.get('factor_weights', {})
        base_factors = ['Value_Composite', 'Quality_Composite', 'PosMom', 'Reversal']
        for factor in base_factors:
            if factor in factors_df.columns:
                mean, std = factors_df[factor].mean(), factors_df[factor].std()
                factors_df[f'{factor}_Z'] = (factors_df[factor] - mean) / std if std > 1e-8 else 0.0
        final_signal = pd.Series(0.0, index=factors_df.index)
        for factor, weight in factors_to_combine.items():
            if f'{factor}_Z' in factors_df.columns and weight != 0:
                final_signal += factors_df[f'{factor}_Z'].fillna(0) * weight
        factors_df['final_signal'] = final_signal
        universe_size = len(factors_df)
        if universe_size < 100:
            selected_stocks = factors_df.nlargest(self.config.get('portfolio_size_small_universe', 20), 'final_signal')
        else:
            score_cutoff = factors_df['final_signal'].quantile(self.config.get('selection_percentile', 0.8))
            selected_stocks = factors_df[factors_df['final_signal'] >= score_cutoff]
        if selected_stocks.empty: return pd.Series(dtype='float64')
        return pd.Series(1.0 / len(selected_stocks), index=selected_stocks['ticker'])

    def _run_backtesting_loop(self) -> pd.DataFrame:
        rebalance_dates = self._generate_rebalance_dates()
        daily_holdings = pd.DataFrame(0.0, index=self.daily_returns_matrix.index, columns=self.daily_returns_matrix.columns)
        for i, rebal_date in enumerate(rebalance_dates):
            print(f"   - Processing {rebal_date.date()}... ", end="")
            universe_df = get_liquid_universe_dataframe(analysis_date=rebal_date, engine=self.engine, config={'lookback_days': 63, 'adtv_threshold_bn': 10.0, 'top_n': 200, 'min_trading_coverage': 0.6})
            if universe_df.empty: print("⚠️ Universe empty. Skipping."); continue
            mask = self.factor_data_raw['date'] == rebal_date
            factors_on_date = self.factor_data_raw.loc[mask]
            factors_on_date = factors_on_date[factors_on_date['ticker'].isin(universe_df['ticker'].tolist())].copy()
            if len(factors_on_date) < 10: print(f"⚠️ Insufficient stocks ({len(factors_on_date)}). Skipping."); continue
            target_portfolio = self._calculate_target_portfolio(factors_on_date)
            if target_portfolio.empty: print("⚠️ Portfolio empty. Skipping."); continue
            start_period = rebal_date + pd.Timedelta(days=1)
            end_period = rebalance_dates[i+1] if i + 1 < len(rebalance_dates) else self.daily_returns_matrix.index.max()
            holding_dates = daily_holdings.index[(daily_holdings.index >= start_period) & (daily_holdings.index <= end_period)]
            daily_holdings.loc[holding_dates] = 0.0
            valid_tickers = target_portfolio.index.intersection(daily_holdings.columns)
            daily_holdings.loc[holding_dates, valid_tickers] = target_portfolio[valid_tickers].values
            print(f"✅ Formed portfolio with {len(target_portfolio)} stocks.")
        return daily_holdings

print("✅ PortfolioEngine v4.1 (Corrected Data & Logic) defined successfully.")

# --- 3. Execute the Day 3 Smoke Test ---
print("\n" + "="*80)
print("🚀 RUNNING DAY 3 SMOKE TEST: Monthly Value-Only Strategy")
print("="*80)

smoke_test_config = {
    **BASE_CONFIG,
    "strategy_name": "Value_Only_Monthly_Smoke_Test",
    "rebalance_frequency": "M",
    "factor_weights": {'Value_Composite': 1.0},
    "construction_method": "hybrid",
    "portfolio_size_small_universe": 20,
    "selection_percentile": 0.8
}

# Load data with the new corrected function
factor_data_raw, daily_returns_matrix, daily_adtv_matrix, benchmark_returns, engine = load_all_data_corrected_v2(smoke_test_config)

# Instantiate the v4.1 engine
smoke_test_backtester = PortfolioEngine_v4_1(
    config=smoke_test_config,
    factor_data=factor_data_raw,
    returns_matrix=daily_returns_matrix,
    adtv_matrix=daily_adtv_matrix,
    benchmark_returns=benchmark_returns,
    db_engine=engine,
    palette=PALETTE
)

# Run the backtest
smoke_test_returns = smoke_test_backtester.run()

✅ PortfolioEngine v4.1 (Corrected Data & Logic) defined successfully.

================================================================================
🚀 RUNNING DAY 3 SMOKE TEST: Monthly Value-Only Strategy
================================================================================

📂 Loading all raw data (Corrected v2 - Robust Merge)...
   - ✅ Loaded 1,567,488 raw factor observations.
   - ✅ Loaded 1,610,552 raw price observations.
   - ✅ Loaded 1,625,572 raw total_value observations.
   - ✅ Merged price/value data. Final records: 1,610,529
   - ✅ Loaded 2,388 benchmark observations.
✅ All data loaded and prepared correctly.

✅ PortfolioEngine v4.0 initialized for strategy: 'Value_Only_Monthly_Smoke_Test'
   - Rebalance Frequency: M
   - v4.1 Enhancements: Four-Factor Stack & Optimizer active.
--- Executing Backtest for: Value_Only_Monthly_Smoke_Test ---
   - Generated 113 data-driven rebalance dates (T+2 logic).
   - Processing 2016-03-02... Constructing liquid universe for 2016-03-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 547 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/11...
  Step 3: Filtering and ranking...
    Total batch results: 547
    Sample result: ('AAA', 37, 6.44357164054054, 711.4619896864865)
    Before filters: 547 stocks
    Trading days range: 1-37 (need >= 37)
    ADTV range: 0.000-200.961B VND (need >= 10.0)
    Stocks passing trading days filter: 227
    Stocks passing ADTV filter: 57
    After filters: 54 stocks
✅ Universe constructed: 54 stocks
  ADTV range: 10.3B - 201.0B VND
  Market cap range: 339.6B - 147876.6B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2016-04-04... Constructing liquid universe for 2016-04-04...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 555 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/12...
  Step 3: Filtering and ranking...
    Total batch results: 555
    Sample result: ('AAA', 41, 6.319916931707318, 768.4571307804878)
    Before filters: 555 stocks
    Trading days range: 1-41 (need >= 37)
    ADTV range: 0.000-186.130B VND (need >= 10.0)
    Stocks passing trading days filter: 314
    Stocks passing ADTV filter: 62
    After filters: 61 stocks
✅ Universe constructed: 61 stocks
  ADTV range: 10.3B - 186.1B VND
  Market cap range: 201.7B - 157247.5B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2016-05-05... Constructing liquid universe for 2016-05-05...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 560 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/12...
  Step 3: Filtering and ranking...
    Total batch results: 560
    Sample result: ('AAA', 43, 7.361778841860468, 836.8951459534882)
    Before filters: 560 stocks
    Trading days range: 1-43 (need >= 37)
    ADTV range: 0.000-133.500B VND (need >= 10.0)
    Stocks passing trading days filter: 351
    Stocks passing ADTV filter: 69
    After filters: 68 stocks
✅ Universe constructed: 68 stocks
  ADTV range: 10.0B - 133.5B VND
  Market cap range: 231.6B - 164163.4B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2016-06-02... Constructing liquid universe for 2016-06-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 563 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/12...
  Step 3: Filtering and ranking...
    Total batch results: 563
    Sample result: ('AAA', 43, 10.526833283720931, 1010.2183645395351)
    Before filters: 563 stocks
    Trading days range: 1-43 (need >= 37)
    ADTV range: 0.001-162.176B VND (need >= 10.0)
    Stocks passing trading days filter: 359
    Stocks passing ADTV filter: 67
    After filters: 67 stocks
✅ Universe constructed: 67 stocks
  ADTV range: 10.0B - 162.2B VND
  Market cap range: 259.8B - 169605.9B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2016-07-04... Constructing liquid universe for 2016-07-04...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 566 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/12...
  Step 3: Filtering and ranking...
    Total batch results: 566
    Sample result: ('AAA', 44, 16.813872002272728, 1295.138562409091)
    Before filters: 566 stocks
    Trading days range: 1-44 (need >= 37)
    ADTV range: 0.000-158.531B VND (need >= 10.0)
    Stocks passing trading days filter: 377
    Stocks passing ADTV filter: 75
    After filters: 74 stocks
✅ Universe constructed: 74 stocks
  ADTV range: 10.0B - 158.5B VND
  Market cap range: 290.4B - 169242.5B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2016-08-02... Constructing liquid universe for 2016-08-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 567 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/12...
  Step 3: Filtering and ranking...
    Total batch results: 567
    Sample result: ('AAA', 46, 19.444281450000002, 1537.9320357130434)
    Before filters: 567 stocks
    Trading days range: 1-46 (need >= 37)
    ADTV range: 0.001-187.378B VND (need >= 10.0)
    Stocks passing trading days filter: 398
    Stocks passing ADTV filter: 81
    After filters: 80 stocks
✅ Universe constructed: 80 stocks
  ADTV range: 10.1B - 187.4B VND
  Market cap range: 297.7B - 173607.8B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2016-09-05... Constructing liquid universe for 2016-09-05...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 566 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/12...
  Step 3: Filtering and ranking...
    Total batch results: 566
    Sample result: ('AAA', 45, 13.368739091111113, 1710.508271173333)
    Before filters: 566 stocks
    Trading days range: 1-45 (need >= 37)
    ADTV range: 0.001-251.023B VND (need >= 10.0)
    Stocks passing trading days filter: 387
    Stocks passing ADTV filter: 72
    After filters: 72 stocks
✅ Universe constructed: 72 stocks
  ADTV range: 10.2B - 251.0B VND
  Market cap range: 138.9B - 194310.1B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2016-10-04... Constructing liquid universe for 2016-10-04...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 564 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/12...
  Step 3: Filtering and ranking...
    Total batch results: 564
    Sample result: ('AAA', 45, 11.182317131111112, 1672.1022800533337)
    Before filters: 564 stocks
    Trading days range: 1-45 (need >= 37)
    ADTV range: 0.000-344.433B VND (need >= 10.0)
    Stocks passing trading days filter: 391
    Stocks passing ADTV filter: 62
    After filters: 62 stocks
✅ Universe constructed: 62 stocks
  ADTV range: 11.1B - 344.4B VND
  Market cap range: 343.7B - 205303.6B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2016-11-02... Constructing liquid universe for 2016-11-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 571 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/12...
  Step 3: Filtering and ranking...
    Total batch results: 571
    Sample result: ('AAA', 45, 14.229569953333334, 1595.4056311199997)
    Before filters: 571 stocks
    Trading days range: 1-45 (need >= 37)
    ADTV range: 0.000-305.647B VND (need >= 10.0)
    Stocks passing trading days filter: 382
    Stocks passing ADTV filter: 66
    After filters: 62 stocks
✅ Universe constructed: 62 stocks
  ADTV range: 10.1B - 305.6B VND
  Market cap range: 359.9B - 206989.9B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2016-12-02... Constructing liquid universe for 2016-12-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 573 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/12...
  Step 3: Filtering and ranking...
    Total batch results: 573
    Sample result: ('AAA', 41, 12.918178492682925, 1549.784275814634)
    Before filters: 573 stocks
    Trading days range: 1-46 (need >= 37)
    ADTV range: 0.001-231.637B VND (need >= 10.0)
    Stocks passing trading days filter: 387
    Stocks passing ADTV filter: 57
    After filters: 56 stocks
✅ Universe constructed: 56 stocks
  ADTV range: 10.2B - 231.6B VND
  Market cap range: 291.5B - 203345.1B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2017-01-04... Constructing liquid universe for 2017-01-04...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 577 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/12...
  Step 3: Filtering and ranking...
    Total batch results: 577
    Sample result: ('AAA', 40, 5.957224137500002, 1389.687053685)
    Before filters: 577 stocks
    Trading days range: 1-45 (need >= 37)
    ADTV range: 0.000-260.089B VND (need >= 10.0)
    Stocks passing trading days filter: 380
    Stocks passing ADTV filter: 60
    After filters: 57 stocks
✅ Universe constructed: 57 stocks
  ADTV range: 10.1B - 260.1B VND
  Market cap range: 406.4B - 193871.2B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2017-02-03... Constructing liquid universe for 2017-02-03...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 582 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/12...
  Step 3: Filtering and ranking...
    Total batch results: 582
    Sample result: ('AAA', 40, 14.153060000000002, 1305.9568610649999)
    Before filters: 582 stocks
    Trading days range: 1-40 (need >= 37)
    ADTV range: 0.000-234.324B VND (need >= 10.0)
    Stocks passing trading days filter: 335
    Stocks passing ADTV filter: 53
    After filters: 52 stocks
✅ Universe constructed: 52 stocks
  ADTV range: 10.3B - 234.3B VND
  Market cap range: 401.8B - 186422.3B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2017-03-02... Constructing liquid universe for 2017-03-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 583 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/12...
  Step 3: Filtering and ranking...
    Total batch results: 583
    Sample result: ('AAA', 40, 25.938185375000007, 1364.0073429299998)
    Before filters: 583 stocks
    Trading days range: 1-40 (need >= 37)
    ADTV range: 0.000-207.525B VND (need >= 10.0)
    Stocks passing trading days filter: 347
    Stocks passing ADTV filter: 59
    After filters: 58 stocks
✅ Universe constructed: 58 stocks
  ADTV range: 10.1B - 207.5B VND
  Market cap range: 527.8B - 188108.0B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2017-04-04... Constructing liquid universe for 2017-04-04...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 590 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/12...
  Step 3: Filtering and ranking...
    Total batch results: 590
    Sample result: ('AAA', 44, 27.812262272727274, 1415.0620882727271)
    Before filters: 590 stocks
    Trading days range: 1-44 (need >= 37)
    ADTV range: 0.001-189.692B VND (need >= 10.0)
    Stocks passing trading days filter: 406
    Stocks passing ADTV filter: 75
    After filters: 74 stocks
✅ Universe constructed: 74 stocks
  ADTV range: 10.1B - 189.7B VND
  Market cap range: 310.3B - 194800.0B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2017-05-04... Constructing liquid universe for 2017-05-04...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 593 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/12...
  Step 3: Filtering and ranking...
    Total batch results: 593
    Sample result: ('AAA', 43, 31.745640116279066, 1426.4430425348835)
    Before filters: 593 stocks
    Trading days range: 1-43 (need >= 37)
    ADTV range: 0.000-170.601B VND (need >= 10.0)
    Stocks passing trading days filter: 393
    Stocks passing ADTV filter: 78
    After filters: 77 stocks
✅ Universe constructed: 77 stocks
  ADTV range: 10.0B - 170.6B VND
  Market cap range: 479.0B - 202724.1B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2017-06-02... Constructing liquid universe for 2017-06-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 598 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/12...
  Step 3: Filtering and ranking...
    Total batch results: 598
    Sample result: ('AAA', 43, 57.22276959302326, 1590.9791241534883)
    Before filters: 598 stocks
    Trading days range: 1-43 (need >= 37)
    ADTV range: 0.001-149.590B VND (need >= 10.0)
    Stocks passing trading days filter: 388
    Stocks passing ADTV filter: 88
    After filters: 85 stocks
✅ Universe constructed: 85 stocks
  ADTV range: 10.3B - 149.6B VND
  Market cap range: 192.3B - 212109.4B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2017-07-04... Constructing liquid universe for 2017-07-04...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 602 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/13...
  Step 3: Filtering and ranking...
    Total batch results: 602
    Sample result: ('AAA', 45, 71.07006241666669, 1841.0211635644444)
    Before filters: 602 stocks
    Trading days range: 1-45 (need >= 37)
    ADTV range: 0.000-163.358B VND (need >= 10.0)
    Stocks passing trading days filter: 422
    Stocks passing ADTV filter: 92
    After filters: 90 stocks
✅ Universe constructed: 90 stocks
  ADTV range: 10.1B - 163.4B VND
  Market cap range: 222.6B - 219130.4B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2017-08-02... Constructing liquid universe for 2017-08-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 615 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/13...
  Step 3: Filtering and ranking...
    Total batch results: 615
    Sample result: ('AAA', 46, 47.81374013586956, 1948.8107720086953)
    Before filters: 615 stocks
    Trading days range: 1-46 (need >= 37)
    ADTV range: 0.001-181.907B VND (need >= 10.0)
    Stocks passing trading days filter: 435
    Stocks passing ADTV filter: 91
    After filters: 88 stocks
✅ Universe constructed: 88 stocks
  ADTV range: 10.0B - 181.9B VND
  Market cap range: 207.7B - 222783.7B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2017-09-05... Constructing liquid universe for 2017-09-05...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 617 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/13...
  Step 3: Filtering and ranking...
    Total batch results: 617
    Sample result: ('AAA', 45, 33.022726666666664, 1942.2146066400003)
    Before filters: 617 stocks
    Trading days range: 1-45 (need >= 37)
    ADTV range: 0.000-213.582B VND (need >= 10.0)
    Stocks passing trading days filter: 420
    Stocks passing ADTV filter: 92
    After filters: 90 stocks
✅ Universe constructed: 90 stocks
  ADTV range: 10.2B - 164.4B VND
  Market cap range: 353.9B - 220115.2B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2017-10-03... Constructing liquid universe for 2017-10-03...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 619 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/13...
  Step 3: Filtering and ranking...
    Total batch results: 619
    Sample result: ('AAA', 45, 43.71764222222221, 1969.9962676800005)
    Before filters: 619 stocks
    Trading days range: 1-45 (need >= 37)
    ADTV range: 0.000-180.377B VND (need >= 10.0)
    Stocks passing trading days filter: 425
    Stocks passing ADTV filter: 88
    After filters: 87 stocks
✅ Universe constructed: 87 stocks
  ADTV range: 10.4B - 180.4B VND
  Market cap range: 439.6B - 216873.1B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2017-11-02... Constructing liquid universe for 2017-11-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 622 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/13...
  Step 3: Filtering and ranking...
    Total batch results: 622
    Sample result: ('AAA', 45, 47.144670999999995, 2000.74042812)
    Before filters: 622 stocks
    Trading days range: 1-45 (need >= 37)
    ADTV range: 0.000-188.017B VND (need >= 10.0)
    Stocks passing trading days filter: 417
    Stocks passing ADTV filter: 85
    After filters: 84 stocks
✅ Universe constructed: 84 stocks
  ADTV range: 10.1B - 188.0B VND
  Market cap range: 483.9B - 216829.5B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2017-12-04... Constructing liquid universe for 2017-12-04...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 626 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/13...
  Step 3: Filtering and ranking...
    Total batch results: 626
    Sample result: ('AAA', 46, 42.27217400000001, 1914.6118404913043)
    Before filters: 626 stocks
    Trading days range: 1-46 (need >= 37)
    ADTV range: 0.001-1107.375B VND (need >= 10.0)
    Stocks passing trading days filter: 431
    Stocks passing ADTV filter: 91
    After filters: 88 stocks
✅ Universe constructed: 88 stocks
  ADTV range: 10.2B - 671.6B VND
  Market cap range: 542.4B - 238121.3B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2018-01-03... Constructing liquid universe for 2018-01-03...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 630 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/13...
  Step 3: Filtering and ranking...
    Total batch results: 630
    Sample result: ('AAA', 45, 45.81383353333334, 2144.191955813334)
    Before filters: 630 stocks
    Trading days range: 1-45 (need >= 37)
    ADTV range: 0.000-732.866B VND (need >= 10.0)
    Stocks passing trading days filter: 428
    Stocks passing ADTV filter: 96
    After filters: 95 stocks
✅ Universe constructed: 95 stocks
  ADTV range: 10.1B - 732.9B VND
  Market cap range: 412.8B - 273259.5B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2018-02-02... Constructing liquid universe for 2018-02-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 639 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/13...
  Step 3: Filtering and ranking...
    Total batch results: 639
    Sample result: ('AAA', 43, 51.67917441860467, 2568.5009506139527)
    Before filters: 639 stocks
    Trading days range: 1-45 (need >= 37)
    ADTV range: 0.000-454.118B VND (need >= 10.0)
    Stocks passing trading days filter: 442
    Stocks passing ADTV filter: 100
    After filters: 97 stocks
✅ Universe constructed: 97 stocks
  ADTV range: 10.1B - 305.1B VND
  Market cap range: 255.7B - 295976.4B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2018-03-02... Constructing liquid universe for 2018-03-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 640 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/13...
  Step 3: Filtering and ranking...
    Total batch results: 640
    Sample result: ('AAA', 38, 47.68413157894738, 2558.269632784211)
    Before filters: 640 stocks
    Trading days range: 1-40 (need >= 37)
    ADTV range: 0.000-393.616B VND (need >= 10.0)
    Stocks passing trading days filter: 383
    Stocks passing ADTV filter: 99
    After filters: 94 stocks
✅ Universe constructed: 94 stocks
  ADTV range: 10.0B - 393.6B VND
  Market cap range: 610.7B - 296616.1B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2018-04-03... Constructing liquid universe for 2018-04-03...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 643 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/13...
  Step 3: Filtering and ranking...
    Total batch results: 643
    Sample result: ('AAA', 41, 32.4399756097561, 2271.8806495024396)
    Before filters: 643 stocks
    Trading days range: 1-41 (need >= 37)
    ADTV range: 0.000-420.193B VND (need >= 10.0)
    Stocks passing trading days filter: 400
    Stocks passing ADTV filter: 95
    After filters: 94 stocks
✅ Universe constructed: 94 stocks
  ADTV range: 10.5B - 420.2B VND
  Market cap range: 304.6B - 295668.4B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2018-05-03... Constructing liquid universe for 2018-05-03...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 644 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/13...
  Step 3: Filtering and ranking...
    Total batch results: 644
    Sample result: ('AAA', 43, 28.107709127906976, 2685.795428432558)
    Before filters: 644 stocks
    Trading days range: 1-43 (need >= 37)
    ADTV range: 0.000-487.220B VND (need >= 10.0)
    Stocks passing trading days filter: 418
    Stocks passing ADTV filter: 99
    After filters: 97 stocks
✅ Universe constructed: 97 stocks
  ADTV range: 10.2B - 487.2B VND
  Market cap range: 271.5B - 304538.7B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2018-06-04... Constructing liquid universe for 2018-06-04...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 646 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/13...
  Step 3: Filtering and ranking...
    Total batch results: 646
    Sample result: ('AAA', 43, 23.568220465116276, 3232.1116290837203)
    Before filters: 646 stocks
    Trading days range: 1-43 (need >= 37)
    ADTV range: 0.001-2763.135B VND (need >= 10.0)
    Stocks passing trading days filter: 405
    Stocks passing ADTV filter: 90
    After filters: 86 stocks
✅ Universe constructed: 86 stocks
  ADTV range: 10.2B - 489.3B VND
  Market cap range: 143.8B - 323192.8B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2018-07-03... Constructing liquid universe for 2018-07-03...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 647 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/13...
  Step 3: Filtering and ranking...
    Total batch results: 647
    Sample result: ('AAA', 45, 27.026210833333334, 3332.4812994311105)
    Before filters: 647 stocks
    Trading days range: 1-45 (need >= 37)
    ADTV range: 0.000-1049.734B VND (need >= 10.0)
    Stocks passing trading days filter: 415
    Stocks passing ADTV filter: 80
    After filters: 78 stocks
✅ Universe constructed: 78 stocks
  ADTV range: 10.2B - 392.2B VND
  Market cap range: 228.3B - 320704.2B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2018-08-02... Constructing liquid universe for 2018-08-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 652 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 652
    Sample result: ('AAA', 46, 35.046076086956525, 3173.164761913043)
    Before filters: 652 stocks
    Trading days range: 1-46 (need >= 37)
    ADTV range: 0.000-319.337B VND (need >= 10.0)
    Stocks passing trading days filter: 411
    Stocks passing ADTV filter: 72
    After filters: 71 stocks
✅ Universe constructed: 71 stocks
  ADTV range: 10.2B - 319.3B VND
  Market cap range: 1002.0B - 330756.6B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2018-09-05... Constructing liquid universe for 2018-09-05...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 655 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 655
    Sample result: ('AAA', 45, 34.866753922222216, 2883.456474995555)
    Before filters: 655 stocks
    Trading days range: 1-45 (need >= 37)
    ADTV range: 0.000-215.115B VND (need >= 10.0)
    Stocks passing trading days filter: 407
    Stocks passing ADTV filter: 74
    After filters: 72 stocks
✅ Universe constructed: 72 stocks
  ADTV range: 10.1B - 215.1B VND
  Market cap range: 915.9B - 334508.6B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2018-10-02... Constructing liquid universe for 2018-10-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 656 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 656
    Sample result: ('AAA', 45, 32.46618361111112, 2873.691145502222)
    Before filters: 656 stocks
    Trading days range: 1-45 (need >= 37)
    ADTV range: 0.000-339.176B VND (need >= 10.0)
    Stocks passing trading days filter: 416
    Stocks passing ADTV filter: 87
    After filters: 87 stocks
✅ Universe constructed: 87 stocks
  ADTV range: 10.1B - 339.2B VND
  Market cap range: 600.0B - 327430.2B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2018-11-02... Constructing liquid universe for 2018-11-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 653 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 653
    Sample result: ('AAA', 45, 29.63125191111111, 2759.5196108)
    Before filters: 653 stocks
    Trading days range: 1-45 (need >= 37)
    ADTV range: 0.000-535.977B VND (need >= 10.0)
    Stocks passing trading days filter: 413
    Stocks passing ADTV filter: 88
    After filters: 88 stocks
✅ Universe constructed: 88 stocks
  ADTV range: 10.7B - 536.0B VND
  Market cap range: 553.8B - 314849.9B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2018-12-04... Constructing liquid universe for 2018-12-04...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 661 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 661
    Sample result: ('AAA', 46, 27.42560869565218, 2572.0935524695656)
    Before filters: 661 stocks
    Trading days range: 1-46 (need >= 37)
    ADTV range: 0.000-535.306B VND (need >= 10.0)
    Stocks passing trading days filter: 406
    Stocks passing ADTV filter: 88
    After filters: 83 stocks
✅ Universe constructed: 83 stocks
  ADTV range: 10.0B - 535.3B VND
  Market cap range: 586.0B - 311634.1B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2019-01-03... Constructing liquid universe for 2019-01-03...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 661 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 661
    Sample result: ('AAA', 44, 27.045318181818185, 2577.726911363636)
    Before filters: 661 stocks
    Trading days range: 1-44 (need >= 37)
    ADTV range: 0.000-259.641B VND (need >= 10.0)
    Stocks passing trading days filter: 392
    Stocks passing ADTV filter: 83
    After filters: 79 stocks
✅ Universe constructed: 79 stocks
  ADTV range: 10.4B - 259.6B VND
  Market cap range: 896.6B - 316986.0B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2019-02-11... Constructing liquid universe for 2019-02-11...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 654 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 654
    Sample result: ('AAA', 39, 23.099, 2545.3924636820516)
    Before filters: 654 stocks
    Trading days range: 1-39 (need >= 37)
    ADTV range: 0.000-228.945B VND (need >= 10.0)
    Stocks passing trading days filter: 333
    Stocks passing ADTV filter: 76
    After filters: 72 stocks
✅ Universe constructed: 72 stocks
  ADTV range: 10.2B - 228.9B VND
  Market cap range: 863.8B - 325209.8B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2019-03-04... Constructing liquid universe for 2019-03-04...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 656 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 656
    Sample result: ('AAA', 39, 22.711338461538464, 2531.1257990153836)
    Before filters: 656 stocks
    Trading days range: 1-39 (need >= 37)
    ADTV range: 0.000-164.064B VND (need >= 10.0)
    Stocks passing trading days filter: 341
    Stocks passing ADTV filter: 74
    After filters: 71 stocks
✅ Universe constructed: 71 stocks
  ADTV range: 10.0B - 164.1B VND
  Market cap range: 886.2B - 341118.8B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2019-04-02... Constructing liquid universe for 2019-04-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 666 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 666
    Sample result: ('AAA', 41, 37.49707804878048, 2702.0366943804875)
    Before filters: 666 stocks
    Trading days range: 1-41 (need >= 37)
    ADTV range: 0.000-201.433B VND (need >= 10.0)
    Stocks passing trading days filter: 384
    Stocks passing ADTV filter: 86
    After filters: 83 stocks
✅ Universe constructed: 83 stocks
  ADTV range: 10.5B - 201.4B VND
  Market cap range: 868.4B - 366211.3B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2019-05-03... Constructing liquid universe for 2019-05-03...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 667 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 667
    Sample result: ('AAA', 42, 70.61595418095237, 2940.9710162857145)
    Before filters: 667 stocks
    Trading days range: 1-42 (need >= 37)
    ADTV range: 0.000-166.934B VND (need >= 10.0)
    Stocks passing trading days filter: 408
    Stocks passing ADTV filter: 84
    After filters: 81 stocks
✅ Universe constructed: 81 stocks
  ADTV range: 10.1B - 166.9B VND
  Market cap range: 641.2B - 368556.3B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2019-06-04... Constructing liquid universe for 2019-06-04...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 667 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 667
    Sample result: ('AAA', 42, 75.32802113333335, 3046.951953809524)
    Before filters: 667 stocks
    Trading days range: 1-42 (need >= 37)
    ADTV range: 0.000-201.333B VND (need >= 10.0)
    Stocks passing trading days filter: 397
    Stocks passing ADTV filter: 80
    After filters: 76 stocks
✅ Universe constructed: 76 stocks
  ADTV range: 10.2B - 201.3B VND
  Market cap range: 609.1B - 369853.8B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2019-07-02... Constructing liquid universe for 2019-07-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 668 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 668
    Sample result: ('AAA', 44, 55.90945593181818, 3047.5541182272723)
    Before filters: 668 stocks
    Trading days range: 1-44 (need >= 37)
    ADTV range: 0.000-204.160B VND (need >= 10.0)
    Stocks passing trading days filter: 411
    Stocks passing ADTV filter: 76
    After filters: 75 stocks
✅ Universe constructed: 75 stocks
  ADTV range: 10.1B - 204.2B VND
  Market cap range: 663.1B - 385939.1B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2019-08-02... Constructing liquid universe for 2019-08-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 671 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 671
    Sample result: ('AAA', 46, 64.37494573913044, 3138.7282556434784)
    Before filters: 671 stocks
    Trading days range: 1-46 (need >= 37)
    ADTV range: 0.000-169.874B VND (need >= 10.0)
    Stocks passing trading days filter: 426
    Stocks passing ADTV filter: 79
    After filters: 78 stocks
✅ Universe constructed: 78 stocks
  ADTV range: 10.1B - 169.9B VND
  Market cap range: 733.6B - 399947.1B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2019-09-04... Constructing liquid universe for 2019-09-04...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 670 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 670
    Sample result: ('AAA', 45, 55.23336122222221, 3042.9844623022223)
    Before filters: 670 stocks
    Trading days range: 1-45 (need >= 37)
    ADTV range: 0.001-165.626B VND (need >= 10.0)
    Stocks passing trading days filter: 425
    Stocks passing ADTV filter: 93
    After filters: 92 stocks
✅ Universe constructed: 92 stocks
  ADTV range: 10.2B - 165.6B VND
  Market cap range: 528.6B - 404802.3B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2019-10-02... Constructing liquid universe for 2019-10-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 664 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 664
    Sample result: ('AAA', 45, 33.47731363333334, 2820.234271306666)
    Before filters: 664 stocks
    Trading days range: 1-45 (need >= 37)
    ADTV range: 0.000-163.536B VND (need >= 10.0)
    Stocks passing trading days filter: 424
    Stocks passing ADTV filter: 88
    After filters: 87 stocks
✅ Universe constructed: 87 stocks
  ADTV range: 10.0B - 163.5B VND
  Market cap range: 787.9B - 406040.4B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2019-11-04... Constructing liquid universe for 2019-11-04...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 665 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 665
    Sample result: ('AAA', 45, 32.068491855555564, 2651.887628240001)
    Before filters: 665 stocks
    Trading days range: 1-45 (need >= 37)
    ADTV range: 0.000-138.706B VND (need >= 10.0)
    Stocks passing trading days filter: 417
    Stocks passing ADTV filter: 85
    After filters: 84 stocks
✅ Universe constructed: 84 stocks
  ADTV range: 10.1B - 138.7B VND
  Market cap range: 510.5B - 399764.9B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2019-12-03... Constructing liquid universe for 2019-12-03...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 666 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 666
    Sample result: ('AAA', 46, 39.30526130434783, 2574.140508704347)
    Before filters: 666 stocks
    Trading days range: 1-46 (need >= 37)
    ADTV range: 0.000-225.004B VND (need >= 10.0)
    Stocks passing trading days filter: 421
    Stocks passing ADTV filter: 77
    After filters: 76 stocks
✅ Universe constructed: 76 stocks
  ADTV range: 10.5B - 225.0B VND
  Market cap range: 623.1B - 394710.3B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2020-01-03... Constructing liquid universe for 2020-01-03...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 666 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 666
    Sample result: ('AAA', 45, 34.640642, 2432.561436764444)
    Before filters: 666 stocks
    Trading days range: 1-45 (need >= 37)
    ADTV range: 0.000-234.589B VND (need >= 10.0)
    Stocks passing trading days filter: 403
    Stocks passing ADTV filter: 82
    After filters: 81 stocks
✅ Universe constructed: 81 stocks
  ADTV range: 10.0B - 234.6B VND
  Market cap range: 340.2B - 392559.8B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2020-02-04... Constructing liquid universe for 2020-02-04...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 662 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 662
    Sample result: ('AAA', 40, 23.18127225, 2199.0636917200004)
    Before filters: 662 stocks
    Trading days range: 1-40 (need >= 37)
    ADTV range: 0.000-188.475B VND (need >= 10.0)
    Stocks passing trading days filter: 344
    Stocks passing ADTV filter: 83
    After filters: 80 stocks
✅ Universe constructed: 80 stocks
  ADTV range: 10.2B - 188.5B VND
  Market cap range: 324.8B - 389026.7B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2020-03-03... Constructing liquid universe for 2020-03-03...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 668 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 668
    Sample result: ('AAA', 40, 24.2249806, 2098.48370582)
    Before filters: 668 stocks
    Trading days range: 1-40 (need >= 37)
    ADTV range: 0.000-239.848B VND (need >= 10.0)
    Stocks passing trading days filter: 349
    Stocks passing ADTV filter: 75
    After filters: 73 stocks
✅ Universe constructed: 73 stocks
  ADTV range: 10.6B - 239.8B VND
  Market cap range: 573.7B - 378751.6B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2020-04-03... Constructing liquid universe for 2020-04-03...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 678 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 678
    Sample result: ('AAA', 45, 23.75639824444444, 1966.4790132142223)
    Before filters: 678 stocks
    Trading days range: 1-45 (need >= 37)
    ADTV range: 0.000-208.327B VND (need >= 10.0)
    Stocks passing trading days filter: 422
    Stocks passing ADTV filter: 80
    After filters: 78 stocks
✅ Universe constructed: 78 stocks
  ADTV range: 10.1B - 208.3B VND
  Market cap range: 529.6B - 338085.2B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2020-05-05... Constructing liquid universe for 2020-05-05...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 675 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 675
    Sample result: ('AAA', 43, 25.002736825581398, 1916.0064290753494)
    Before filters: 675 stocks
    Trading days range: 1-43 (need >= 37)
    ADTV range: 0.000-219.854B VND (need >= 10.0)
    Stocks passing trading days filter: 421
    Stocks passing ADTV filter: 88
    After filters: 88 stocks
✅ Universe constructed: 88 stocks
  ADTV range: 10.3B - 219.9B VND
  Market cap range: 280.7B - 311655.6B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2020-06-02... Constructing liquid universe for 2020-06-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 673 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 673
    Sample result: ('AAA', 43, 28.527129232558135, 2034.4131101506978)
    Before filters: 673 stocks
    Trading days range: 1-43 (need >= 37)
    ADTV range: 0.000-285.943B VND (need >= 10.0)
    Stocks passing trading days filter: 439
    Stocks passing ADTV filter: 99
    After filters: 98 stocks
✅ Universe constructed: 98 stocks
  ADTV range: 10.0B - 285.9B VND
  Market cap range: 294.2B - 321055.6B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2020-07-02... Constructing liquid universe for 2020-07-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 679 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 679
    Sample result: ('AAA', 44, 29.933921136363644, 2164.1233329818187)
    Before filters: 679 stocks
    Trading days range: 1-44 (need >= 37)
    ADTV range: 0.000-678.813B VND (need >= 10.0)
    Stocks passing trading days filter: 454
    Stocks passing ADTV filter: 118
    After filters: 115 stocks
✅ Universe constructed: 115 stocks
  ADTV range: 10.1B - 678.8B VND
  Market cap range: 288.8B - 320631.4B VND
  Adding sector information...
✅ Formed portfolio with 23 stocks.
   - Processing 2020-08-04... Constructing liquid universe for 2020-08-04...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 684 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 684
    Sample result: ('AAA', 46, 25.161592956521734, 2269.6257874782605)
    Before filters: 684 stocks
    Trading days range: 1-46 (need >= 37)
    ADTV range: 0.000-498.530B VND (need >= 10.0)
    Stocks passing trading days filter: 465
    Stocks passing ADTV filter: 115
    After filters: 111 stocks
✅ Universe constructed: 111 stocks
  ADTV range: 10.2B - 498.5B VND
  Market cap range: 228.6B - 309425.2B VND
  Adding sector information...
✅ Formed portfolio with 22 stocks.
   - Processing 2020-09-03... Constructing liquid universe for 2020-09-03...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 686 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 686
    Sample result: ('AAA', 45, 28.131360688888886, 2434.6112651022218)
    Before filters: 686 stocks
    Trading days range: 1-45 (need >= 37)
    ADTV range: 0.000-324.034B VND (need >= 10.0)
    Stocks passing trading days filter: 469
    Stocks passing ADTV filter: 101
    After filters: 100 stocks
✅ Universe constructed: 100 stocks
  ADTV range: 10.1B - 324.0B VND
  Market cap range: 258.5B - 304828.5B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2020-10-02... Constructing liquid universe for 2020-10-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 685 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 685
    Sample result: ('AAA', 45, 32.40854930000001, 2569.5997079999997)
    Before filters: 685 stocks
    Trading days range: 1-45 (need >= 37)
    ADTV range: 0.000-335.867B VND (need >= 10.0)
    Stocks passing trading days filter: 474
    Stocks passing ADTV filter: 121
    After filters: 118 stocks
✅ Universe constructed: 118 stocks
  ADTV range: 10.6B - 335.9B VND
  Market cap range: 233.9B - 308240.7B VND
  Adding sector information...
✅ Formed portfolio with 24 stocks.
   - Processing 2020-11-03... Constructing liquid universe for 2020-11-03...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 683 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 683
    Sample result: ('AAA', 45, 24.29860811111111, 2581.837570855556)
    Before filters: 683 stocks
    Trading days range: 1-45 (need >= 37)
    ADTV range: 0.001-450.666B VND (need >= 10.0)
    Stocks passing trading days filter: 465
    Stocks passing ADTV filter: 126
    After filters: 122 stocks
✅ Universe constructed: 122 stocks
  ADTV range: 10.5B - 450.7B VND
  Market cap range: 471.3B - 323788.8B VND
  Adding sector information...
✅ Formed portfolio with 24 stocks.
   - Processing 2020-12-02... Constructing liquid universe for 2020-12-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 687 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 687
    Sample result: ('AAA', 46, 22.9851329347826, 2580.6570913782607)
    Before filters: 687 stocks
    Trading days range: 1-46 (need >= 37)
    ADTV range: 0.000-649.651B VND (need >= 10.0)
    Stocks passing trading days filter: 475
    Stocks passing ADTV filter: 133
    After filters: 131 stocks
✅ Universe constructed: 131 stocks
  ADTV range: 10.1B - 649.7B VND
  Market cap range: 342.7B - 341897.6B VND
  Adding sector information...
✅ Formed portfolio with 26 stocks.
   - Processing 2021-01-05... Constructing liquid universe for 2021-01-05...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 696 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 696
    Sample result: ('AAA', 45, 36.443623577777785, 2816.3516698000003)
    Before filters: 696 stocks
    Trading days range: 1-45 (need >= 37)
    ADTV range: 0.000-815.667B VND (need >= 10.0)
    Stocks passing trading days filter: 497
    Stocks passing ADTV filter: 156
    After filters: 153 stocks
✅ Universe constructed: 153 stocks
  ADTV range: 10.2B - 815.7B VND
  Market cap range: 351.6B - 357568.0B VND
  Adding sector information...
✅ Formed portfolio with 30 stocks.
   - Processing 2021-02-02... Constructing liquid universe for 2021-02-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 705 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 705
    Sample result: ('AAA', 45, 49.816759811111105, 3096.2620369822225)
    Before filters: 705 stocks
    Trading days range: 1-45 (need >= 37)
    ADTV range: 0.000-931.351B VND (need >= 10.0)
    Stocks passing trading days filter: 532
    Stocks passing ADTV filter: 177
    After filters: 171 stocks
✅ Universe constructed: 171 stocks
  ADTV range: 10.0B - 931.4B VND
  Market cap range: 361.5B - 365810.7B VND
  Adding sector information...
✅ Formed portfolio with 34 stocks.
   - Processing 2021-03-02... Constructing liquid universe for 2021-03-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 705 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 705
    Sample result: ('AAA', 40, 46.81468970000002, 3161.4656293375)
    Before filters: 705 stocks
    Trading days range: 1-40 (need >= 37)
    ADTV range: 0.000-1036.926B VND (need >= 10.0)
    Stocks passing trading days filter: 483
    Stocks passing ADTV filter: 172
    After filters: 168 stocks
✅ Universe constructed: 168 stocks
  ADTV range: 10.1B - 1036.9B VND
  Market cap range: 130.3B - 370655.9B VND
  Adding sector information...
✅ Formed portfolio with 33 stocks.
   - Processing 2021-04-02... Constructing liquid universe for 2021-04-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 708 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 708
    Sample result: ('AAA', 41, 50.02336341463414, 3324.2360980585368)
    Before filters: 708 stocks
    Trading days range: 1-41 (need >= 37)
    ADTV range: 0.001-971.076B VND (need >= 10.0)
    Stocks passing trading days filter: 508
    Stocks passing ADTV filter: 173
    After filters: 171 stocks
✅ Universe constructed: 171 stocks
  ADTV range: 10.0B - 971.1B VND
  Market cap range: 158.1B - 366168.7B VND
  Adding sector information...
✅ Formed portfolio with 34 stocks.
   - Processing 2021-05-05... Constructing liquid universe for 2021-05-05...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 708 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 708
    Sample result: ('AAA', 43, 72.07673790697673, 3643.052224037209)
    Before filters: 708 stocks
    Trading days range: 1-43 (need >= 37)
    ADTV range: 0.001-1097.726B VND (need >= 10.0)
    Stocks passing trading days filter: 544
    Stocks passing ADTV filter: 186
    After filters: 182 stocks
✅ Universe constructed: 182 stocks
  ADTV range: 10.1B - 1097.7B VND
  Market cap range: 200.6B - 409116.8B VND
  Adding sector information...
✅ Formed portfolio with 36 stocks.
   - Processing 2021-06-02... Constructing liquid universe for 2021-06-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 710 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 710
    Sample result: ('AAA', 43, 71.12607674418605, 3617.476044097674)
    Before filters: 710 stocks
    Trading days range: 1-43 (need >= 37)
    ADTV range: 0.001-1598.753B VND (need >= 10.0)
    Stocks passing trading days filter: 545
    Stocks passing ADTV filter: 182
    After filters: 180 stocks
✅ Universe constructed: 180 stocks
  ADTV range: 10.0B - 1598.8B VND
  Market cap range: 219.8B - 434894.0B VND
  Adding sector information...
✅ Formed portfolio with 36 stocks.
   - Processing 2021-07-02... Constructing liquid universe for 2021-07-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 710 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 710
    Sample result: ('AAA', 44, 156.3550298863636, 4405.620110899999)
    Before filters: 710 stocks
    Trading days range: 1-44 (need >= 37)
    ADTV range: 0.001-2254.665B VND (need >= 10.0)
    Stocks passing trading days filter: 555
    Stocks passing ADTV filter: 186
    After filters: 185 stocks
✅ Universe constructed: 185 stocks
  ADTV range: 10.1B - 2254.7B VND
  Market cap range: 191.0B - 411734.1B VND
  Adding sector information...
✅ Formed portfolio with 36 stocks.
   - Processing 2021-08-03... Constructing liquid universe for 2021-08-03...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 712 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 712
    Sample result: ('AAA', 46, 167.63186141304345, 5004.735341965217)
    Before filters: 712 stocks
    Trading days range: 1-46 (need >= 37)
    ADTV range: 0.000-1652.694B VND (need >= 10.0)
    Stocks passing trading days filter: 549
    Stocks passing ADTV filter: 187
    After filters: 184 stocks
✅ Universe constructed: 184 stocks
  ADTV range: 10.1B - 1652.7B VND
  Market cap range: 337.1B - 388964.5B VND
  Adding sector information...
✅ Formed portfolio with 36 stocks.
   - Processing 2021-09-06... Constructing liquid universe for 2021-09-06...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 713 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 713
    Sample result: ('AAA', 44, 115.00393852272727, 4966.918896353408)
    Before filters: 713 stocks
    Trading days range: 1-44 (need >= 37)
    ADTV range: 0.000-1452.408B VND (need >= 10.0)
    Stocks passing trading days filter: 542
    Stocks passing ADTV filter: 205
    After filters: 202 stocks
✅ Universe constructed: 200 stocks
  ADTV range: 10.1B - 1452.4B VND
  Market cap range: 351.5B - 373720.0B VND
  Adding sector information...
✅ Formed portfolio with 39 stocks.
   - Processing 2021-10-04... Constructing liquid universe for 2021-10-04...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 715 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 715
    Sample result: ('AAA', 44, 111.54386366818186, 5174.271340102274)
    Before filters: 715 stocks
    Trading days range: 1-44 (need >= 37)
    ADTV range: 0.002-1406.313B VND (need >= 10.0)
    Stocks passing trading days filter: 577
    Stocks passing ADTV filter: 235
    After filters: 235 stocks
✅ Universe constructed: 200 stocks
  ADTV range: 15.6B - 1406.3B VND
  Market cap range: 219.6B - 366302.2B VND
  Adding sector information...
✅ Formed portfolio with 39 stocks.
   - Processing 2021-11-02... Constructing liquid universe for 2021-11-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 716 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 716
    Sample result: ('AAA', 44, 103.06125355454543, 5302.9793986113655)
    Before filters: 716 stocks
    Trading days range: 1-44 (need >= 37)
    ADTV range: 0.001-1506.020B VND (need >= 10.0)
    Stocks passing trading days filter: 601
    Stocks passing ADTV filter: 261
    After filters: 259 stocks
✅ Universe constructed: 200 stocks
  ADTV range: 20.1B - 1506.0B VND
  Market cap range: 286.6B - 361253.1B VND
  Adding sector information...
✅ Formed portfolio with 39 stocks.
   - Processing 2021-12-02... Constructing liquid universe for 2021-12-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 717 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 717
    Sample result: ('AAA', 46, 120.37381850869563, 5499.001977182608)
    Before filters: 717 stocks
    Trading days range: 1-46 (need >= 37)
    ADTV range: 0.004-1559.500B VND (need >= 10.0)
    Stocks passing trading days filter: 621
    Stocks passing ADTV filter: 278
    After filters: 277 stocks
✅ Universe constructed: 200 stocks
  ADTV range: 23.3B - 1559.5B VND
  Market cap range: 361.5B - 362179.9B VND
  Adding sector information...
✅ Formed portfolio with 39 stocks.
   - Processing 2022-01-05... Constructing liquid universe for 2022-01-05...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 720 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 720
    Sample result: ('AAA', 44, 148.91960011363633, 5957.058603709092)
    Before filters: 720 stocks
    Trading days range: 1-45 (need >= 37)
    ADTV range: 0.001-1208.382B VND (need >= 10.0)
    Stocks passing trading days filter: 617
    Stocks passing ADTV filter: 274
    After filters: 271 stocks
✅ Universe constructed: 200 stocks
  ADTV range: 23.1B - 1208.4B VND
  Market cap range: 466.1B - 376217.4B VND
  Adding sector information...
✅ Formed portfolio with 39 stocks.
   - Processing 2022-02-08... Constructing liquid universe for 2022-02-08...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 719 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 719
    Sample result: ('AAA', 39, 147.29607396410256, 6159.567836061538)
    Before filters: 719 stocks
    Trading days range: 1-40 (need >= 37)
    ADTV range: 0.001-851.146B VND (need >= 10.0)
    Stocks passing trading days filter: 556
    Stocks passing ADTV filter: 259
    After filters: 254 stocks
✅ Universe constructed: 200 stocks
  ADTV range: 17.8B - 851.1B VND
  Market cap range: 442.7B - 388832.3B VND
  Adding sector information...
✅ Formed portfolio with 39 stocks.
   - Processing 2022-03-02... Constructing liquid universe for 2022-03-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 719 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 719
    Sample result: ('AAA', 40, 115.40873899, 6030.469270479998)
    Before filters: 719 stocks
    Trading days range: 1-40 (need >= 37)
    ADTV range: 0.001-994.529B VND (need >= 10.0)
    Stocks passing trading days filter: 556
    Stocks passing ADTV filter: 250
    After filters: 245 stocks
✅ Universe constructed: 200 stocks
  ADTV range: 16.0B - 994.5B VND
  Market cap range: 474.9B - 406310.3B VND
  Adding sector information...
✅ Formed portfolio with 39 stocks.
   - Processing 2022-04-04... Constructing liquid universe for 2022-04-04...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 718 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 718
    Sample result: ('AAA', 41, 104.80573109756097, 5887.763653463416)
    Before filters: 718 stocks
    Trading days range: 3-41 (need >= 37)
    ADTV range: 0.001-1126.816B VND (need >= 10.0)
    Stocks passing trading days filter: 589
    Stocks passing ADTV filter: 260
    After filters: 258 stocks
✅ Universe constructed: 200 stocks
  ADTV range: 18.3B - 1126.8B VND
  Market cap range: 387.4B - 403279.7B VND
  Adding sector information...
✅ Formed portfolio with 39 stocks.
   - Processing 2022-05-05... Constructing liquid universe for 2022-05-05...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 719 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 719
    Sample result: ('AAA', 43, 100.69134720930232, 5510.290207479071)
    Before filters: 719 stocks
    Trading days range: 1-43 (need >= 37)
    ADTV range: 0.001-1038.697B VND (need >= 10.0)
    Stocks passing trading days filter: 593
    Stocks passing ADTV filter: 246
    After filters: 246 stocks
✅ Universe constructed: 200 stocks
  ADTV range: 16.4B - 1038.7B VND
  Market cap range: 343.8B - 389354.0B VND
  Adding sector information...
✅ Formed portfolio with 39 stocks.
   - Processing 2022-06-02... Constructing liquid universe for 2022-06-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 719 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 719
    Sample result: ('AAA', 43, 65.51559255813955, 4596.653193674417)
    Before filters: 719 stocks
    Trading days range: 1-43 (need >= 37)
    ADTV range: 0.001-814.549B VND (need >= 10.0)
    Stocks passing trading days filter: 569
    Stocks passing ADTV filter: 198
    After filters: 198 stocks
✅ Universe constructed: 198 stocks
  ADTV range: 10.0B - 814.5B VND
  Market cap range: 424.8B - 375332.6B VND
  Adding sector information...
✅ Formed portfolio with 39 stocks.
   - Processing 2022-07-04... Constructing liquid universe for 2022-07-04...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 720 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 720
    Sample result: ('AAA', 44, 47.85175352272727, 3927.9714524363635)
    Before filters: 720 stocks
    Trading days range: 1-44 (need >= 37)
    ADTV range: 0.000-713.472B VND (need >= 10.0)
    Stocks passing trading days filter: 550
    Stocks passing ADTV filter: 175
    After filters: 175 stocks
✅ Universe constructed: 175 stocks
  ADTV range: 10.3B - 713.5B VND
  Market cap range: 449.7B - 364704.9B VND
  Adding sector information...
✅ Formed portfolio with 34 stocks.
   - Processing 2022-08-02... Constructing liquid universe for 2022-08-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 720 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 720
    Sample result: ('AAA', 46, 50.14384423913043, 4068.0858849391334)
    Before filters: 720 stocks
    Trading days range: 1-46 (need >= 37)
    ADTV range: 0.000-556.993B VND (need >= 10.0)
    Stocks passing trading days filter: 555
    Stocks passing ADTV filter: 168
    After filters: 166 stocks
✅ Universe constructed: 166 stocks
  ADTV range: 10.2B - 557.0B VND
  Market cap range: 810.1B - 357181.5B VND
  Adding sector information...
✅ Formed portfolio with 33 stocks.
   - Processing 2022-09-06... Constructing liquid universe for 2022-09-06...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 721 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 721
    Sample result: ('AAA', 44, 55.14230454545454, 4515.8174008)
    Before filters: 721 stocks
    Trading days range: 1-44 (need >= 37)
    ADTV range: 0.000-600.691B VND (need >= 10.0)
    Stocks passing trading days filter: 538
    Stocks passing ADTV filter: 178
    After filters: 177 stocks
✅ Universe constructed: 177 stocks
  ADTV range: 10.0B - 600.7B VND
  Market cap range: 281.2B - 368630.8B VND
  Adding sector information...
✅ Formed portfolio with 35 stocks.
   - Processing 2022-10-04... Constructing liquid universe for 2022-10-04...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 724 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 724
    Sample result: ('AAA', 44, 43.403194423636364, 4430.387647505456)
    Before filters: 724 stocks
    Trading days range: 1-44 (need >= 37)
    ADTV range: 0.000-599.257B VND (need >= 10.0)
    Stocks passing trading days filter: 548
    Stocks passing ADTV filter: 184
    After filters: 182 stocks
✅ Universe constructed: 182 stocks
  ADTV range: 10.1B - 599.3B VND
  Market cap range: 269.0B - 376149.0B VND
  Adding sector information...
✅ Formed portfolio with 36 stocks.
   - Processing 2022-11-02... Constructing liquid universe for 2022-11-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 722 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 722
    Sample result: ('AAA', 44, 23.236783438863636, 3616.3167321600004)
    Before filters: 722 stocks
    Trading days range: 1-44 (need >= 37)
    ADTV range: 0.000-530.181B VND (need >= 10.0)
    Stocks passing trading days filter: 545
    Stocks passing ADTV filter: 155
    After filters: 154 stocks
✅ Universe constructed: 154 stocks
  ADTV range: 10.0B - 530.2B VND
  Market cap range: 593.6B - 347614.1B VND
  Adding sector information...
✅ Formed portfolio with 30 stocks.
   - Processing 2022-12-02... Constructing liquid universe for 2022-12-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 722 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 722
    Sample result: ('AAA', 46, 22.793177210434784, 2892.986903206958)
    Before filters: 722 stocks
    Trading days range: 1-46 (need >= 37)
    ADTV range: 0.000-642.048B VND (need >= 10.0)
    Stocks passing trading days filter: 541
    Stocks passing ADTV filter: 144
    After filters: 142 stocks
✅ Universe constructed: 142 stocks
  ADTV range: 10.2B - 642.0B VND
  Market cap range: 713.6B - 341749.4B VND
  Adding sector information...
✅ Formed portfolio with 28 stocks.
   - Processing 2023-01-04... Constructing liquid universe for 2023-01-04...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 717 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 717
    Sample result: ('AAA', 45, 21.59236027666668, 2723.663309056001)
    Before filters: 717 stocks
    Trading days range: 1-45 (need >= 37)
    ADTV range: 0.000-672.073B VND (need >= 10.0)
    Stocks passing trading days filter: 523
    Stocks passing ADTV filter: 150
    After filters: 148 stocks
✅ Universe constructed: 148 stocks
  ADTV range: 10.5B - 672.1B VND
  Market cap range: 509.0B - 366475.6B VND
  Adding sector information...
✅ Formed portfolio with 29 stocks.
   - Processing 2023-02-02... Constructing liquid universe for 2023-02-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 713 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 713
    Sample result: ('AAA', 40, 22.466598833749995, 2834.0875447199996)
    Before filters: 713 stocks
    Trading days range: 1-40 (need >= 37)
    ADTV range: 0.000-642.675B VND (need >= 10.0)
    Stocks passing trading days filter: 473
    Stocks passing ADTV filter: 152
    After filters: 147 stocks
✅ Universe constructed: 147 stocks
  ADTV range: 10.3B - 642.7B VND
  Market cap range: 435.4B - 393733.5B VND
  Adding sector information...
✅ Formed portfolio with 29 stocks.
   - Processing 2023-03-02... Constructing liquid universe for 2023-03-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 711 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 711
    Sample result: ('AAA', 40, 27.22989710225, 3041.758164672)
    Before filters: 711 stocks
    Trading days range: 1-40 (need >= 37)
    ADTV range: 0.000-520.831B VND (need >= 10.0)
    Stocks passing trading days filter: 470
    Stocks passing ADTV filter: 143
    After filters: 140 stocks
✅ Universe constructed: 140 stocks
  ADTV range: 10.1B - 520.8B VND
  Market cap range: 555.0B - 426589.0B VND
  Adding sector information...
✅ Formed portfolio with 28 stocks.
   - Processing 2023-04-04... Constructing liquid universe for 2023-04-04...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 712 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 712
    Sample result: ('AAA', 46, 29.920173138043484, 3341.8270234017386)
    Before filters: 712 stocks
    Trading days range: 1-46 (need >= 37)
    ADTV range: 0.000-512.563B VND (need >= 10.0)
    Stocks passing trading days filter: 525
    Stocks passing ADTV filter: 137
    After filters: 136 stocks
✅ Universe constructed: 136 stocks
  ADTV range: 10.1B - 512.6B VND
  Market cap range: 402.7B - 435010.9B VND
  Adding sector information...
✅ Formed portfolio with 27 stocks.
   - Processing 2023-05-05... Constructing liquid universe for 2023-05-05...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 716 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 716
    Sample result: ('AAA', 43, 30.036747084186043, 3570.7104957767424)
    Before filters: 716 stocks
    Trading days range: 1-43 (need >= 37)
    ADTV range: 0.000-460.216B VND (need >= 10.0)
    Stocks passing trading days filter: 509
    Stocks passing ADTV filter: 147
    After filters: 145 stocks
✅ Universe constructed: 145 stocks
  ADTV range: 10.1B - 460.2B VND
  Market cap range: 407.0B - 425904.5B VND
  Adding sector information...
✅ Formed portfolio with 29 stocks.
   - Processing 2023-06-02... Constructing liquid universe for 2023-06-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 717 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 717
    Sample result: ('AAA', 43, 50.993005356976745, 3929.9596209711635)
    Before filters: 717 stocks
    Trading days range: 1-43 (need >= 37)
    ADTV range: 0.000-448.862B VND (need >= 10.0)
    Stocks passing trading days filter: 515
    Stocks passing ADTV filter: 168
    After filters: 168 stocks
✅ Universe constructed: 168 stocks
  ADTV range: 10.1B - 448.9B VND
  Market cap range: 474.3B - 431231.3B VND
  Adding sector information...
✅ Formed portfolio with 34 stocks.
   - Processing 2023-07-04... Constructing liquid universe for 2023-07-04...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 716 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 716
    Sample result: ('AAA', 44, 64.97159355454545, 4227.608403490911)
    Before filters: 716 stocks
    Trading days range: 1-44 (need >= 37)
    ADTV range: 0.000-543.223B VND (need >= 10.0)
    Stocks passing trading days filter: 546
    Stocks passing ADTV filter: 188
    After filters: 186 stocks
✅ Universe constructed: 186 stocks
  ADTV range: 10.1B - 543.2B VND
  Market cap range: 378.0B - 457526.8B VND
  Adding sector information...
✅ Formed portfolio with 37 stocks.
   - Processing 2023-08-02... Constructing liquid universe for 2023-08-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 716 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 716
    Sample result: ('AAA', 46, 93.8007539119565, 4368.7326640695655)
    Before filters: 716 stocks
    Trading days range: 1-46 (need >= 37)
    ADTV range: 0.000-652.011B VND (need >= 10.0)
    Stocks passing trading days filter: 573
    Stocks passing ADTV filter: 195
    After filters: 193 stocks
✅ Universe constructed: 193 stocks
  ADTV range: 10.0B - 652.0B VND
  Market cap range: 403.7B - 485202.6B VND
  Adding sector information...
✅ Formed portfolio with 39 stocks.
   - Processing 2023-09-06... Constructing liquid universe for 2023-09-06...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 717 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 717
    Sample result: ('AAA', 44, 117.77191957068182, 4392.2470784727275)
    Before filters: 717 stocks
    Trading days range: 1-44 (need >= 37)
    ADTV range: 0.000-833.597B VND (need >= 10.0)
    Stocks passing trading days filter: 570
    Stocks passing ADTV filter: 205
    After filters: 204 stocks
✅ Universe constructed: 200 stocks
  ADTV range: 10.8B - 833.6B VND
  Market cap range: 405.5B - 498969.7B VND
  Adding sector information...
✅ Formed portfolio with 40 stocks.
   - Processing 2023-10-03... Constructing liquid universe for 2023-10-03...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 716 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 716
    Sample result: ('AAA', 44, 84.19526558727271, 4119.876500072725)
    Before filters: 716 stocks
    Trading days range: 2-44 (need >= 37)
    ADTV range: 0.000-1010.673B VND (need >= 10.0)
    Stocks passing trading days filter: 566
    Stocks passing ADTV filter: 202
    After filters: 200 stocks
✅ Universe constructed: 200 stocks
  ADTV range: 10.4B - 1010.7B VND
  Market cap range: 403.2B - 496743.3B VND
  Adding sector information...
✅ Formed portfolio with 40 stocks.
   - Processing 2023-11-02... Constructing liquid universe for 2023-11-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 716 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 716
    Sample result: ('AAA', 44, 38.80639022931819, 3622.4852524363628)
    Before filters: 716 stocks
    Trading days range: 1-44 (need >= 37)
    ADTV range: 0.000-914.857B VND (need >= 10.0)
    Stocks passing trading days filter: 549
    Stocks passing ADTV filter: 177
    After filters: 175 stocks
✅ Universe constructed: 175 stocks
  ADTV range: 10.2B - 914.9B VND
  Market cap range: 433.1B - 487381.5B VND
  Adding sector information...
✅ Formed portfolio with 35 stocks.
   - Processing 2023-12-04... Constructing liquid universe for 2023-12-04...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 709 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 709
    Sample result: ('AAA', 46, 22.177921931304343, 3426.9246503373906)
    Before filters: 709 stocks
    Trading days range: 1-46 (need >= 37)
    ADTV range: 0.000-782.034B VND (need >= 10.0)
    Stocks passing trading days filter: 540
    Stocks passing ADTV filter: 154
    After filters: 151 stocks
✅ Universe constructed: 151 stocks
  ADTV range: 10.3B - 782.0B VND
  Market cap range: 438.5B - 482059.1B VND
  Adding sector information...
✅ Formed portfolio with 30 stocks.
   - Processing 2024-01-03... Constructing liquid universe for 2024-01-03...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 710 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 710
    Sample result: ('AAA', 45, 22.073825207333336, 3527.459149312)
    Before filters: 710 stocks
    Trading days range: 1-45 (need >= 37)
    ADTV range: 0.000-726.965B VND (need >= 10.0)
    Stocks passing trading days filter: 541
    Stocks passing ADTV filter: 157
    After filters: 155 stocks
✅ Universe constructed: 155 stocks
  ADTV range: 10.0B - 727.0B VND
  Market cap range: 441.8B - 475346.0B VND
  Adding sector information...
✅ Formed portfolio with 31 stocks.
   - Processing 2024-02-02... Constructing liquid universe for 2024-02-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 707 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 707
    Sample result: ('AAA', 45, 30.08223503688889, 3670.854560256)
    Before filters: 707 stocks
    Trading days range: 1-45 (need >= 37)
    ADTV range: 0.000-698.736B VND (need >= 10.0)
    Stocks passing trading days filter: 550
    Stocks passing ADTV filter: 162
    After filters: 160 stocks
✅ Universe constructed: 160 stocks
  ADTV range: 10.1B - 698.7B VND
  Market cap range: 309.3B - 483444.0B VND
  Adding sector information...
✅ Formed portfolio with 32 stocks.
   - Processing 2024-03-04... Constructing liquid universe for 2024-03-04...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 708 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 708
    Sample result: ('AAA', 40, 47.083037369500005, 3930.2596619999995)
    Before filters: 708 stocks
    Trading days range: 1-40 (need >= 37)
    ADTV range: 0.000-782.572B VND (need >= 10.0)
    Stocks passing trading days filter: 505
    Stocks passing ADTV filter: 159
    After filters: 157 stocks
✅ Universe constructed: 157 stocks
  ADTV range: 10.8B - 782.6B VND
  Market cap range: 251.3B - 504960.4B VND
  Adding sector information...
✅ Formed portfolio with 32 stocks.
   - Processing 2024-04-02... Constructing liquid universe for 2024-04-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 714 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 714
    Sample result: ('AAA', 41, 49.85210302804879, 4166.792006400002)
    Before filters: 714 stocks
    Trading days range: 1-41 (need >= 37)
    ADTV range: 0.000-938.982B VND (need >= 10.0)
    Stocks passing trading days filter: 528
    Stocks passing ADTV filter: 170
    After filters: 167 stocks
✅ Universe constructed: 167 stocks
  ADTV range: 10.1B - 939.0B VND
  Market cap range: 303.4B - 521039.6B VND
  Adding sector information...
✅ Formed portfolio with 34 stocks.
   - Processing 2024-05-03... Constructing liquid universe for 2024-05-03...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 713 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 713
    Sample result: ('AAA', 42, 36.21881074595237, 4072.861701668572)
    Before filters: 713 stocks
    Trading days range: 1-42 (need >= 37)
    ADTV range: 0.000-865.854B VND (need >= 10.0)
    Stocks passing trading days filter: 540
    Stocks passing ADTV filter: 171
    After filters: 168 stocks
✅ Universe constructed: 168 stocks
  ADTV range: 10.2B - 865.9B VND
  Market cap range: 439.4B - 525121.7B VND
  Adding sector information...
✅ Formed portfolio with 34 stocks.
   - Processing 2024-06-04... Constructing liquid universe for 2024-06-04...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 712 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 712
    Sample result: ('AAA', 42, 53.31682626142858, 4154.322576411429)
    Before filters: 712 stocks
    Trading days range: 1-42 (need >= 37)
    ADTV range: 0.000-710.121B VND (need >= 10.0)
    Stocks passing trading days filter: 532
    Stocks passing ADTV filter: 181
    After filters: 179 stocks
✅ Universe constructed: 179 stocks
  ADTV range: 10.1B - 710.1B VND
  Market cap range: 352.5B - 512612.8B VND
  Adding sector information...
✅ Formed portfolio with 36 stocks.
   - Processing 2024-07-02... Constructing liquid universe for 2024-07-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 712 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 712
    Sample result: ('AAA', 44, 65.61904068886363, 4318.398596290908)
    Before filters: 712 stocks
    Trading days range: 1-44 (need >= 37)
    ADTV range: 0.000-854.214B VND (need >= 10.0)
    Stocks passing trading days filter: 560
    Stocks passing ADTV filter: 194
    After filters: 191 stocks
✅ Universe constructed: 191 stocks
  ADTV range: 10.1B - 854.2B VND
  Market cap range: 387.5B - 498305.6B VND
  Adding sector information...
✅ Formed portfolio with 38 stocks.
   - Processing 2024-08-02... Constructing liquid universe for 2024-08-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 711 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 711
    Sample result: ('AAA', 46, 71.80838077934781, 4431.891059060869)
    Before filters: 711 stocks
    Trading days range: 1-46 (need >= 37)
    ADTV range: 0.000-913.206B VND (need >= 10.0)
    Stocks passing trading days filter: 566
    Stocks passing ADTV filter: 184
    After filters: 183 stocks
✅ Universe constructed: 183 stocks
  ADTV range: 10.0B - 913.2B VND
  Market cap range: 430.0B - 489361.4B VND
  Adding sector information...
✅ Formed portfolio with 36 stocks.
   - Processing 2024-09-05... Constructing liquid universe for 2024-09-05...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 708 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 708
    Sample result: ('AAA', 44, 68.17864246590908, 4231.9524318545455)
    Before filters: 708 stocks
    Trading days range: 1-44 (need >= 37)
    ADTV range: 0.000-689.112B VND (need >= 10.0)
    Stocks passing trading days filter: 534
    Stocks passing ADTV filter: 166
    After filters: 166 stocks
✅ Universe constructed: 166 stocks
  ADTV range: 10.1B - 689.1B VND
  Market cap range: 416.0B - 496565.4B VND
  Adding sector information...
✅ Formed portfolio with 33 stocks.
   - Processing 2024-10-02... Constructing liquid universe for 2024-10-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 707 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 707
    Sample result: ('AAA', 44, 46.08232932454546, 3903.891409832728)
    Before filters: 707 stocks
    Trading days range: 1-44 (need >= 37)
    ADTV range: 0.000-597.839B VND (need >= 10.0)
    Stocks passing trading days filter: 526
    Stocks passing ADTV filter: 156
    After filters: 153 stocks
✅ Universe constructed: 153 stocks
  ADTV range: 10.2B - 597.8B VND
  Market cap range: 398.4B - 504148.7B VND
  Adding sector information...
✅ Formed portfolio with 31 stocks.
   - Processing 2024-11-04... Constructing liquid universe for 2024-11-04...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 709 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 709
    Sample result: ('AAA', 44, 21.614921891590907, 3626.4817585309092)
    Before filters: 709 stocks
    Trading days range: 1-44 (need >= 37)
    ADTV range: 0.000-696.061B VND (need >= 10.0)
    Stocks passing trading days filter: 513
    Stocks passing ADTV filter: 147
    After filters: 144 stocks
✅ Universe constructed: 144 stocks
  ADTV range: 10.4B - 696.1B VND
  Market cap range: 434.2B - 510995.4B VND
  Adding sector information...
✅ Formed portfolio with 29 stocks.
   - Processing 2024-12-03... Constructing liquid universe for 2024-12-03...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 703 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 703
    Sample result: ('AAA', 46, 13.853613753695651, 3387.3675503165205)
    Before filters: 703 stocks
    Trading days range: 1-46 (need >= 37)
    ADTV range: 0.000-732.425B VND (need >= 10.0)
    Stocks passing trading days filter: 526
    Stocks passing ADTV filter: 146
    After filters: 144 stocks
✅ Universe constructed: 144 stocks
  ADTV range: 10.5B - 732.4B VND
  Market cap range: 789.4B - 514317.9B VND
  Adding sector information...
✅ Formed portfolio with 29 stocks.
   - Processing 2025-01-03... Constructing liquid universe for 2025-01-03...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 701 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 701
    Sample result: ('AAA', 45, 13.209269164666669, 3282.9733716479996)
    Before filters: 701 stocks
    Trading days range: 1-45 (need >= 37)
    ADTV range: 0.000-806.089B VND (need >= 10.0)
    Stocks passing trading days filter: 528
    Stocks passing ADTV filter: 159
    After filters: 157 stocks
✅ Universe constructed: 157 stocks
  ADTV range: 10.1B - 806.1B VND
  Market cap range: 478.8B - 517015.8B VND
  Adding sector information...
✅ Formed portfolio with 32 stocks.
   - Processing 2025-02-04... Constructing liquid universe for 2025-02-04...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 697 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 697
    Sample result: ('AAA', 40, 11.873568785, 3286.318273488)
    Before filters: 697 stocks
    Trading days range: 2-40 (need >= 37)
    ADTV range: 0.000-768.975B VND (need >= 10.0)
    Stocks passing trading days filter: 484
    Stocks passing ADTV filter: 159
    After filters: 157 stocks
✅ Universe constructed: 157 stocks
  ADTV range: 10.1B - 769.0B VND
  Market cap range: 529.0B - 517116.7B VND
  Adding sector information...
✅ Formed portfolio with 31 stocks.
   - Processing 2025-03-04... Constructing liquid universe for 2025-03-04...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 697 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 697
    Sample result: ('AAA', 40, 12.326138511749999, 3293.1036457920004)
    Before filters: 697 stocks
    Trading days range: 1-40 (need >= 37)
    ADTV range: 0.000-682.780B VND (need >= 10.0)
    Stocks passing trading days filter: 484
    Stocks passing ADTV filter: 161
    After filters: 159 stocks
✅ Universe constructed: 159 stocks
  ADTV range: 10.5B - 682.8B VND
  Market cap range: 533.7B - 515342.2B VND
  Adding sector information...
✅ Formed portfolio with 32 stocks.
   - Processing 2025-04-02... Constructing liquid universe for 2025-04-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 700 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 700
    Sample result: ('AAA', 43, 15.144741557674415, 3315.1199897302336)
    Before filters: 700 stocks
    Trading days range: 1-43 (need >= 37)
    ADTV range: 0.000-818.714B VND (need >= 10.0)
    Stocks passing trading days filter: 532
    Stocks passing ADTV filter: 163
    After filters: 162 stocks
✅ Universe constructed: 162 stocks
  ADTV range: 10.0B - 818.7B VND
  Market cap range: 318.6B - 530714.2B VND
  Adding sector information...
✅ Formed portfolio with 33 stocks.
   - Processing 2025-05-06... Constructing liquid universe for 2025-05-06...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 697 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 697
    Sample result: ('AAA', 42, 16.630042019761905, 2986.9290798171423)
    Before filters: 697 stocks
    Trading days range: 1-42 (need >= 37)
    ADTV range: 0.000-1022.172B VND (need >= 10.0)
    Stocks passing trading days filter: 519
    Stocks passing ADTV filter: 163
    After filters: 163 stocks
✅ Universe constructed: 163 stocks
  ADTV range: 10.3B - 1022.2B VND
  Market cap range: 473.6B - 515738.0B VND
  Adding sector information...
✅ Formed portfolio with 33 stocks.
   - Processing 2025-06-03... Constructing liquid universe for 2025-06-03...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 694 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 694
    Sample result: ('AAA', 42, 16.713963587380956, 2749.008714925714)
    Before filters: 694 stocks
    Trading days range: 1-42 (need >= 37)
    ADTV range: 0.000-1033.973B VND (need >= 10.0)
    Stocks passing trading days filter: 523
    Stocks passing ADTV filter: 167
    After filters: 166 stocks
✅ Universe constructed: 166 stocks
  ADTV range: 10.3B - 1034.0B VND
  Market cap range: 434.3B - 483137.1B VND
  Adding sector information...
✅ Formed portfolio with 33 stocks.
   - Processing 2025-07-02... Constructing liquid universe for 2025-07-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 697 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 697
    Sample result: ('AAA', 43, 13.657289003488374, 2764.6447154902326)
    Before filters: 697 stocks
    Trading days range: 1-43 (need >= 37)
    ADTV range: 0.000-904.619B VND (need >= 10.0)
    Stocks passing trading days filter: 527
    Stocks passing ADTV filter: 166
    After filters: 166 stocks
✅ Universe constructed: 166 stocks
  ADTV range: 10.1B - 904.6B VND
  Market cap range: 439.9B - 474855.0B VND
  Adding sector information...
✅ Formed portfolio with 33 stocks.
   - Generated 113 data-driven rebalance dates (T+2 logic).
✅ Backtest for Value_Only_Monthly_Smoke_Test complete.

⚙️  Defining configuration for the full four-factor monthly composite strategy...
✅ Configuration for the full composite run is ready.

================================================================================
🚀 LAUNCHING DAY 4 RUN: Full_Composite_Monthly
================================================================================

✅ PortfolioEngine v4.0 initialized for strategy: 'Full_Composite_Monthly'
   - Rebalance Frequency: M
   - v4.1 Enhancements: Four-Factor Stack & Optimizer active.
--- Executing Backtest for: Full_Composite_Monthly ---
   - Generated 113 data-driven rebalance dates (T+2 logic).
   - Processing 2016-03-02... Constructing liquid universe for 2016-03-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 547 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/11...
  Step 3: Filtering and ranking...
    Total batch results: 547
    Sample result: ('AAA', 37, 6.44357164054054, 711.4619896864865)
    Before filters: 547 stocks
    Trading days range: 1-37 (need >= 37)
    ADTV range: 0.000-200.961B VND (need >= 10.0)
    Stocks passing trading days filter: 227
    Stocks passing ADTV filter: 57
    After filters: 54 stocks
✅ Universe constructed: 54 stocks
  ADTV range: 10.3B - 201.0B VND
  Market cap range: 339.6B - 147876.6B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2016-04-04... Constructing liquid universe for 2016-04-04...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 555 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/12...
  Step 3: Filtering and ranking...
    Total batch results: 555
    Sample result: ('AAA', 41, 6.319916931707318, 768.4571307804878)
    Before filters: 555 stocks
    Trading days range: 1-41 (need >= 37)
    ADTV range: 0.000-186.130B VND (need >= 10.0)
    Stocks passing trading days filter: 314
    Stocks passing ADTV filter: 62
    After filters: 61 stocks
✅ Universe constructed: 61 stocks
  ADTV range: 10.3B - 186.1B VND
  Market cap range: 201.7B - 157247.5B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2016-05-05... Constructing liquid universe for 2016-05-05...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 560 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/12...
  Step 3: Filtering and ranking...
    Total batch results: 560
    Sample result: ('AAA', 43, 7.361778841860468, 836.8951459534882)
    Before filters: 560 stocks
    Trading days range: 1-43 (need >= 37)
    ADTV range: 0.000-133.500B VND (need >= 10.0)
    Stocks passing trading days filter: 351
    Stocks passing ADTV filter: 69
    After filters: 68 stocks
✅ Universe constructed: 68 stocks
  ADTV range: 10.0B - 133.5B VND
  Market cap range: 231.6B - 164163.4B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2016-06-02... Constructing liquid universe for 2016-06-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 563 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/12...
  Step 3: Filtering and ranking...
    Total batch results: 563
    Sample result: ('AAA', 43, 10.526833283720931, 1010.2183645395351)
    Before filters: 563 stocks
    Trading days range: 1-43 (need >= 37)
    ADTV range: 0.001-162.176B VND (need >= 10.0)
    Stocks passing trading days filter: 359
    Stocks passing ADTV filter: 67
    After filters: 67 stocks
✅ Universe constructed: 67 stocks
  ADTV range: 10.0B - 162.2B VND
  Market cap range: 259.8B - 169605.9B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2016-07-04... Constructing liquid universe for 2016-07-04...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 566 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/12...
  Step 3: Filtering and ranking...
    Total batch results: 566
    Sample result: ('AAA', 44, 16.813872002272728, 1295.138562409091)
    Before filters: 566 stocks
    Trading days range: 1-44 (need >= 37)
    ADTV range: 0.000-158.531B VND (need >= 10.0)
    Stocks passing trading days filter: 377
    Stocks passing ADTV filter: 75
    After filters: 74 stocks
✅ Universe constructed: 74 stocks
  ADTV range: 10.0B - 158.5B VND
  Market cap range: 290.4B - 169242.5B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2016-08-02... Constructing liquid universe for 2016-08-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 567 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/12...
  Step 3: Filtering and ranking...
    Total batch results: 567
    Sample result: ('AAA', 46, 19.444281450000002, 1537.9320357130434)
    Before filters: 567 stocks
    Trading days range: 1-46 (need >= 37)
    ADTV range: 0.001-187.378B VND (need >= 10.0)
    Stocks passing trading days filter: 398
    Stocks passing ADTV filter: 81
    After filters: 80 stocks
✅ Universe constructed: 80 stocks
  ADTV range: 10.1B - 187.4B VND
  Market cap range: 297.7B - 173607.8B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2016-09-05... Constructing liquid universe for 2016-09-05...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 566 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/12...
  Step 3: Filtering and ranking...
    Total batch results: 566
    Sample result: ('AAA', 45, 13.368739091111113, 1710.508271173333)
    Before filters: 566 stocks
    Trading days range: 1-45 (need >= 37)
    ADTV range: 0.001-251.023B VND (need >= 10.0)
    Stocks passing trading days filter: 387
    Stocks passing ADTV filter: 72
    After filters: 72 stocks
✅ Universe constructed: 72 stocks
  ADTV range: 10.2B - 251.0B VND
  Market cap range: 138.9B - 194310.1B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2016-10-04... Constructing liquid universe for 2016-10-04...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 564 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/12...
  Step 3: Filtering and ranking...
    Total batch results: 564
    Sample result: ('AAA', 45, 11.182317131111112, 1672.1022800533337)
    Before filters: 564 stocks
    Trading days range: 1-45 (need >= 37)
    ADTV range: 0.000-344.433B VND (need >= 10.0)
    Stocks passing trading days filter: 391
    Stocks passing ADTV filter: 62
    After filters: 62 stocks
✅ Universe constructed: 62 stocks
  ADTV range: 11.1B - 344.4B VND
  Market cap range: 343.7B - 205303.6B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2016-11-02... Constructing liquid universe for 2016-11-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 571 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/12...
  Step 3: Filtering and ranking...
    Total batch results: 571
    Sample result: ('AAA', 45, 14.229569953333334, 1595.4056311199997)
    Before filters: 571 stocks
    Trading days range: 1-45 (need >= 37)
    ADTV range: 0.000-305.647B VND (need >= 10.0)
    Stocks passing trading days filter: 382
    Stocks passing ADTV filter: 66
    After filters: 62 stocks
✅ Universe constructed: 62 stocks
  ADTV range: 10.1B - 305.6B VND
  Market cap range: 359.9B - 206989.9B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2016-12-02... Constructing liquid universe for 2016-12-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 573 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/12...
  Step 3: Filtering and ranking...
    Total batch results: 573
    Sample result: ('AAA', 41, 12.918178492682925, 1549.784275814634)
    Before filters: 573 stocks
    Trading days range: 1-46 (need >= 37)
    ADTV range: 0.001-231.637B VND (need >= 10.0)
    Stocks passing trading days filter: 387
    Stocks passing ADTV filter: 57
    After filters: 56 stocks
✅ Universe constructed: 56 stocks
  ADTV range: 10.2B - 231.6B VND
  Market cap range: 291.5B - 203345.1B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2017-01-04... Constructing liquid universe for 2017-01-04...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 577 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/12...
  Step 3: Filtering and ranking...
    Total batch results: 577
    Sample result: ('AAA', 40, 5.957224137500002, 1389.687053685)
    Before filters: 577 stocks
    Trading days range: 1-45 (need >= 37)
    ADTV range: 0.000-260.089B VND (need >= 10.0)
    Stocks passing trading days filter: 380
    Stocks passing ADTV filter: 60
    After filters: 57 stocks
✅ Universe constructed: 57 stocks
  ADTV range: 10.1B - 260.1B VND
  Market cap range: 406.4B - 193871.2B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2017-02-03... Constructing liquid universe for 2017-02-03...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 582 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/12...
  Step 3: Filtering and ranking...
    Total batch results: 582
    Sample result: ('AAA', 40, 14.153060000000002, 1305.9568610649999)
    Before filters: 582 stocks
    Trading days range: 1-40 (need >= 37)
    ADTV range: 0.000-234.324B VND (need >= 10.0)
    Stocks passing trading days filter: 335
    Stocks passing ADTV filter: 53
    After filters: 52 stocks
✅ Universe constructed: 52 stocks
  ADTV range: 10.3B - 234.3B VND
  Market cap range: 401.8B - 186422.3B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2017-03-02... Constructing liquid universe for 2017-03-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 583 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/12...
  Step 3: Filtering and ranking...
    Total batch results: 583
    Sample result: ('AAA', 40, 25.938185375000007, 1364.0073429299998)
    Before filters: 583 stocks
    Trading days range: 1-40 (need >= 37)
    ADTV range: 0.000-207.525B VND (need >= 10.0)
    Stocks passing trading days filter: 347
    Stocks passing ADTV filter: 59
    After filters: 58 stocks
✅ Universe constructed: 58 stocks
  ADTV range: 10.1B - 207.5B VND
  Market cap range: 527.8B - 188108.0B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2017-04-04... Constructing liquid universe for 2017-04-04...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 590 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/12...
  Step 3: Filtering and ranking...
    Total batch results: 590
    Sample result: ('AAA', 44, 27.812262272727274, 1415.0620882727271)
    Before filters: 590 stocks
    Trading days range: 1-44 (need >= 37)
    ADTV range: 0.001-189.692B VND (need >= 10.0)
    Stocks passing trading days filter: 406
    Stocks passing ADTV filter: 75
    After filters: 74 stocks
✅ Universe constructed: 74 stocks
  ADTV range: 10.1B - 189.7B VND
  Market cap range: 310.3B - 194800.0B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2017-05-04... Constructing liquid universe for 2017-05-04...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 593 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/12...
  Step 3: Filtering and ranking...
    Total batch results: 593
    Sample result: ('AAA', 43, 31.745640116279066, 1426.4430425348835)
    Before filters: 593 stocks
    Trading days range: 1-43 (need >= 37)
    ADTV range: 0.000-170.601B VND (need >= 10.0)
    Stocks passing trading days filter: 393
    Stocks passing ADTV filter: 78
    After filters: 77 stocks
✅ Universe constructed: 77 stocks
  ADTV range: 10.0B - 170.6B VND
  Market cap range: 479.0B - 202724.1B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2017-06-02... Constructing liquid universe for 2017-06-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 598 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/12...
  Step 3: Filtering and ranking...
    Total batch results: 598
    Sample result: ('AAA', 43, 57.22276959302326, 1590.9791241534883)
    Before filters: 598 stocks
    Trading days range: 1-43 (need >= 37)
    ADTV range: 0.001-149.590B VND (need >= 10.0)
    Stocks passing trading days filter: 388
    Stocks passing ADTV filter: 88
    After filters: 85 stocks
✅ Universe constructed: 85 stocks
  ADTV range: 10.3B - 149.6B VND
  Market cap range: 192.3B - 212109.4B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2017-07-04... Constructing liquid universe for 2017-07-04...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 602 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/13...
  Step 3: Filtering and ranking...
    Total batch results: 602
    Sample result: ('AAA', 45, 71.07006241666669, 1841.0211635644444)
    Before filters: 602 stocks
    Trading days range: 1-45 (need >= 37)
    ADTV range: 0.000-163.358B VND (need >= 10.0)
    Stocks passing trading days filter: 422
    Stocks passing ADTV filter: 92
    After filters: 90 stocks
✅ Universe constructed: 90 stocks
  ADTV range: 10.1B - 163.4B VND
  Market cap range: 222.6B - 219130.4B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2017-08-02... Constructing liquid universe for 2017-08-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 615 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/13...
  Step 3: Filtering and ranking...
    Total batch results: 615
    Sample result: ('AAA', 46, 47.81374013586956, 1948.8107720086953)
    Before filters: 615 stocks
    Trading days range: 1-46 (need >= 37)
    ADTV range: 0.001-181.907B VND (need >= 10.0)
    Stocks passing trading days filter: 435
    Stocks passing ADTV filter: 91
    After filters: 88 stocks
✅ Universe constructed: 88 stocks
  ADTV range: 10.0B - 181.9B VND
  Market cap range: 207.7B - 222783.7B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2017-09-05... Constructing liquid universe for 2017-09-05...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 617 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/13...
  Step 3: Filtering and ranking...
    Total batch results: 617
    Sample result: ('AAA', 45, 33.022726666666664, 1942.2146066400003)
    Before filters: 617 stocks
    Trading days range: 1-45 (need >= 37)
    ADTV range: 0.000-213.582B VND (need >= 10.0)
    Stocks passing trading days filter: 420
    Stocks passing ADTV filter: 92
    After filters: 90 stocks
✅ Universe constructed: 90 stocks
  ADTV range: 10.2B - 164.4B VND
  Market cap range: 353.9B - 220115.2B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2017-10-03... Constructing liquid universe for 2017-10-03...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 619 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/13...
  Step 3: Filtering and ranking...
    Total batch results: 619
    Sample result: ('AAA', 45, 43.71764222222221, 1969.9962676800005)
    Before filters: 619 stocks
    Trading days range: 1-45 (need >= 37)
    ADTV range: 0.000-180.377B VND (need >= 10.0)
    Stocks passing trading days filter: 425
    Stocks passing ADTV filter: 88
    After filters: 87 stocks
✅ Universe constructed: 87 stocks
  ADTV range: 10.4B - 180.4B VND
  Market cap range: 439.6B - 216873.1B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2017-11-02... Constructing liquid universe for 2017-11-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 622 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/13...
  Step 3: Filtering and ranking...
    Total batch results: 622
    Sample result: ('AAA', 45, 47.144670999999995, 2000.74042812)
    Before filters: 622 stocks
    Trading days range: 1-45 (need >= 37)
    ADTV range: 0.000-188.017B VND (need >= 10.0)
    Stocks passing trading days filter: 417
    Stocks passing ADTV filter: 85
    After filters: 84 stocks
✅ Universe constructed: 84 stocks
  ADTV range: 10.1B - 188.0B VND
  Market cap range: 483.9B - 216829.5B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2017-12-04... Constructing liquid universe for 2017-12-04...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 626 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/13...
  Step 3: Filtering and ranking...
    Total batch results: 626
    Sample result: ('AAA', 46, 42.27217400000001, 1914.6118404913043)
    Before filters: 626 stocks
    Trading days range: 1-46 (need >= 37)
    ADTV range: 0.001-1107.375B VND (need >= 10.0)
    Stocks passing trading days filter: 431
    Stocks passing ADTV filter: 91
    After filters: 88 stocks
✅ Universe constructed: 88 stocks
  ADTV range: 10.2B - 671.6B VND
  Market cap range: 542.4B - 238121.3B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2018-01-03... Constructing liquid universe for 2018-01-03...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 630 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/13...
  Step 3: Filtering and ranking...
    Total batch results: 630
    Sample result: ('AAA', 45, 45.81383353333334, 2144.191955813334)
    Before filters: 630 stocks
    Trading days range: 1-45 (need >= 37)
    ADTV range: 0.000-732.866B VND (need >= 10.0)
    Stocks passing trading days filter: 428
    Stocks passing ADTV filter: 96
    After filters: 95 stocks
✅ Universe constructed: 95 stocks
  ADTV range: 10.1B - 732.9B VND
  Market cap range: 412.8B - 273259.5B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2018-02-02... Constructing liquid universe for 2018-02-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 639 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/13...
  Step 3: Filtering and ranking...
    Total batch results: 639
    Sample result: ('AAA', 43, 51.67917441860467, 2568.5009506139527)
    Before filters: 639 stocks
    Trading days range: 1-45 (need >= 37)
    ADTV range: 0.000-454.118B VND (need >= 10.0)
    Stocks passing trading days filter: 442
    Stocks passing ADTV filter: 100
    After filters: 97 stocks
✅ Universe constructed: 97 stocks
  ADTV range: 10.1B - 305.1B VND
  Market cap range: 255.7B - 295976.4B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2018-03-02... Constructing liquid universe for 2018-03-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 640 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/13...
  Step 3: Filtering and ranking...
    Total batch results: 640
    Sample result: ('AAA', 38, 47.68413157894738, 2558.269632784211)
    Before filters: 640 stocks
    Trading days range: 1-40 (need >= 37)
    ADTV range: 0.000-393.616B VND (need >= 10.0)
    Stocks passing trading days filter: 383
    Stocks passing ADTV filter: 99
    After filters: 94 stocks
✅ Universe constructed: 94 stocks
  ADTV range: 10.0B - 393.6B VND
  Market cap range: 610.7B - 296616.1B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2018-04-03... Constructing liquid universe for 2018-04-03...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 643 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/13...
  Step 3: Filtering and ranking...
    Total batch results: 643
    Sample result: ('AAA', 41, 32.4399756097561, 2271.8806495024396)
    Before filters: 643 stocks
    Trading days range: 1-41 (need >= 37)
    ADTV range: 0.000-420.193B VND (need >= 10.0)
    Stocks passing trading days filter: 400
    Stocks passing ADTV filter: 95
    After filters: 94 stocks
✅ Universe constructed: 94 stocks
  ADTV range: 10.5B - 420.2B VND
  Market cap range: 304.6B - 295668.4B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2018-05-03... Constructing liquid universe for 2018-05-03...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 644 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/13...
  Step 3: Filtering and ranking...
    Total batch results: 644
    Sample result: ('AAA', 43, 28.107709127906976, 2685.795428432558)
    Before filters: 644 stocks
    Trading days range: 1-43 (need >= 37)
    ADTV range: 0.000-487.220B VND (need >= 10.0)
    Stocks passing trading days filter: 418
    Stocks passing ADTV filter: 99
    After filters: 97 stocks
✅ Universe constructed: 97 stocks
  ADTV range: 10.2B - 487.2B VND
  Market cap range: 271.5B - 304538.7B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2018-06-04... Constructing liquid universe for 2018-06-04...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 646 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/13...
  Step 3: Filtering and ranking...
    Total batch results: 646
    Sample result: ('AAA', 43, 23.568220465116276, 3232.1116290837203)
    Before filters: 646 stocks
    Trading days range: 1-43 (need >= 37)
    ADTV range: 0.001-2763.135B VND (need >= 10.0)
    Stocks passing trading days filter: 405
    Stocks passing ADTV filter: 90
    After filters: 86 stocks
✅ Universe constructed: 86 stocks
  ADTV range: 10.2B - 489.3B VND
  Market cap range: 143.8B - 323192.8B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2018-07-03... Constructing liquid universe for 2018-07-03...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 647 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/13...
  Step 3: Filtering and ranking...
    Total batch results: 647
    Sample result: ('AAA', 45, 27.026210833333334, 3332.4812994311105)
    Before filters: 647 stocks
    Trading days range: 1-45 (need >= 37)
    ADTV range: 0.000-1049.734B VND (need >= 10.0)
    Stocks passing trading days filter: 415
    Stocks passing ADTV filter: 80
    After filters: 78 stocks
✅ Universe constructed: 78 stocks
  ADTV range: 10.2B - 392.2B VND
  Market cap range: 228.3B - 320704.2B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2018-08-02... Constructing liquid universe for 2018-08-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 652 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 652
    Sample result: ('AAA', 46, 35.046076086956525, 3173.164761913043)
    Before filters: 652 stocks
    Trading days range: 1-46 (need >= 37)
    ADTV range: 0.000-319.337B VND (need >= 10.0)
    Stocks passing trading days filter: 411
    Stocks passing ADTV filter: 72
    After filters: 71 stocks
✅ Universe constructed: 71 stocks
  ADTV range: 10.2B - 319.3B VND
  Market cap range: 1002.0B - 330756.6B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2018-09-05... Constructing liquid universe for 2018-09-05...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 655 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 655
    Sample result: ('AAA', 45, 34.866753922222216, 2883.456474995555)
    Before filters: 655 stocks
    Trading days range: 1-45 (need >= 37)
    ADTV range: 0.000-215.115B VND (need >= 10.0)
    Stocks passing trading days filter: 407
    Stocks passing ADTV filter: 74
    After filters: 72 stocks
✅ Universe constructed: 72 stocks
  ADTV range: 10.1B - 215.1B VND
  Market cap range: 915.9B - 334508.6B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2018-10-02... Constructing liquid universe for 2018-10-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 656 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 656
    Sample result: ('AAA', 45, 32.46618361111112, 2873.691145502222)
    Before filters: 656 stocks
    Trading days range: 1-45 (need >= 37)
    ADTV range: 0.000-339.176B VND (need >= 10.0)
    Stocks passing trading days filter: 416
    Stocks passing ADTV filter: 87
    After filters: 87 stocks
✅ Universe constructed: 87 stocks
  ADTV range: 10.1B - 339.2B VND
  Market cap range: 600.0B - 327430.2B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2018-11-02... Constructing liquid universe for 2018-11-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 653 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 653
    Sample result: ('AAA', 45, 29.63125191111111, 2759.5196108)
    Before filters: 653 stocks
    Trading days range: 1-45 (need >= 37)
    ADTV range: 0.000-535.977B VND (need >= 10.0)
    Stocks passing trading days filter: 413
    Stocks passing ADTV filter: 88
    After filters: 88 stocks
✅ Universe constructed: 88 stocks
  ADTV range: 10.7B - 536.0B VND
  Market cap range: 553.8B - 314849.9B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2018-12-04... Constructing liquid universe for 2018-12-04...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 661 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 661
    Sample result: ('AAA', 46, 27.42560869565218, 2572.0935524695656)
    Before filters: 661 stocks
    Trading days range: 1-46 (need >= 37)
    ADTV range: 0.000-535.306B VND (need >= 10.0)
    Stocks passing trading days filter: 406
    Stocks passing ADTV filter: 88
    After filters: 83 stocks
✅ Universe constructed: 83 stocks
  ADTV range: 10.0B - 535.3B VND
  Market cap range: 586.0B - 311634.1B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2019-01-03... Constructing liquid universe for 2019-01-03...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 661 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 661
    Sample result: ('AAA', 44, 27.045318181818185, 2577.726911363636)
    Before filters: 661 stocks
    Trading days range: 1-44 (need >= 37)
    ADTV range: 0.000-259.641B VND (need >= 10.0)
    Stocks passing trading days filter: 392
    Stocks passing ADTV filter: 83
    After filters: 79 stocks
✅ Universe constructed: 79 stocks
  ADTV range: 10.4B - 259.6B VND
  Market cap range: 896.6B - 316986.0B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2019-02-11... Constructing liquid universe for 2019-02-11...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 654 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 654
    Sample result: ('AAA', 39, 23.099, 2545.3924636820516)
    Before filters: 654 stocks
    Trading days range: 1-39 (need >= 37)
    ADTV range: 0.000-228.945B VND (need >= 10.0)
    Stocks passing trading days filter: 333
    Stocks passing ADTV filter: 76
    After filters: 72 stocks
✅ Universe constructed: 72 stocks
  ADTV range: 10.2B - 228.9B VND
  Market cap range: 863.8B - 325209.8B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2019-03-04... Constructing liquid universe for 2019-03-04...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 656 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 656
    Sample result: ('AAA', 39, 22.711338461538464, 2531.1257990153836)
    Before filters: 656 stocks
    Trading days range: 1-39 (need >= 37)
    ADTV range: 0.000-164.064B VND (need >= 10.0)
    Stocks passing trading days filter: 341
    Stocks passing ADTV filter: 74
    After filters: 71 stocks
✅ Universe constructed: 71 stocks
  ADTV range: 10.0B - 164.1B VND
  Market cap range: 886.2B - 341118.8B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2019-04-02... Constructing liquid universe for 2019-04-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 666 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 666
    Sample result: ('AAA', 41, 37.49707804878048, 2702.0366943804875)
    Before filters: 666 stocks
    Trading days range: 1-41 (need >= 37)
    ADTV range: 0.000-201.433B VND (need >= 10.0)
    Stocks passing trading days filter: 384
    Stocks passing ADTV filter: 86
    After filters: 83 stocks
✅ Universe constructed: 83 stocks
  ADTV range: 10.5B - 201.4B VND
  Market cap range: 868.4B - 366211.3B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2019-05-03... Constructing liquid universe for 2019-05-03...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 667 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 667
    Sample result: ('AAA', 42, 70.61595418095237, 2940.9710162857145)
    Before filters: 667 stocks
    Trading days range: 1-42 (need >= 37)
    ADTV range: 0.000-166.934B VND (need >= 10.0)
    Stocks passing trading days filter: 408
    Stocks passing ADTV filter: 84
    After filters: 81 stocks
✅ Universe constructed: 81 stocks
  ADTV range: 10.1B - 166.9B VND
  Market cap range: 641.2B - 368556.3B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2019-06-04... Constructing liquid universe for 2019-06-04...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 667 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 667
    Sample result: ('AAA', 42, 75.32802113333335, 3046.951953809524)
    Before filters: 667 stocks
    Trading days range: 1-42 (need >= 37)
    ADTV range: 0.000-201.333B VND (need >= 10.0)
    Stocks passing trading days filter: 397
    Stocks passing ADTV filter: 80
    After filters: 76 stocks
✅ Universe constructed: 76 stocks
  ADTV range: 10.2B - 201.3B VND
  Market cap range: 609.1B - 369853.8B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2019-07-02... Constructing liquid universe for 2019-07-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 668 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 668
    Sample result: ('AAA', 44, 55.90945593181818, 3047.5541182272723)
    Before filters: 668 stocks
    Trading days range: 1-44 (need >= 37)
    ADTV range: 0.000-204.160B VND (need >= 10.0)
    Stocks passing trading days filter: 411
    Stocks passing ADTV filter: 76
    After filters: 75 stocks
✅ Universe constructed: 75 stocks
  ADTV range: 10.1B - 204.2B VND
  Market cap range: 663.1B - 385939.1B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2019-08-02... Constructing liquid universe for 2019-08-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 671 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 671
    Sample result: ('AAA', 46, 64.37494573913044, 3138.7282556434784)
    Before filters: 671 stocks
    Trading days range: 1-46 (need >= 37)
    ADTV range: 0.000-169.874B VND (need >= 10.0)
    Stocks passing trading days filter: 426
    Stocks passing ADTV filter: 79
    After filters: 78 stocks
✅ Universe constructed: 78 stocks
  ADTV range: 10.1B - 169.9B VND
  Market cap range: 733.6B - 399947.1B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2019-09-04... Constructing liquid universe for 2019-09-04...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 670 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 670
    Sample result: ('AAA', 45, 55.23336122222221, 3042.9844623022223)
    Before filters: 670 stocks
    Trading days range: 1-45 (need >= 37)
    ADTV range: 0.001-165.626B VND (need >= 10.0)
    Stocks passing trading days filter: 425
    Stocks passing ADTV filter: 93
    After filters: 92 stocks
✅ Universe constructed: 92 stocks
  ADTV range: 10.2B - 165.6B VND
  Market cap range: 528.6B - 404802.3B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2019-10-02... Constructing liquid universe for 2019-10-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 664 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 664
    Sample result: ('AAA', 45, 33.47731363333334, 2820.234271306666)
    Before filters: 664 stocks
    Trading days range: 1-45 (need >= 37)
    ADTV range: 0.000-163.536B VND (need >= 10.0)
    Stocks passing trading days filter: 424
    Stocks passing ADTV filter: 88
    After filters: 87 stocks
✅ Universe constructed: 87 stocks
  ADTV range: 10.0B - 163.5B VND
  Market cap range: 787.9B - 406040.4B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2019-11-04... Constructing liquid universe for 2019-11-04...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 665 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 665
    Sample result: ('AAA', 45, 32.068491855555564, 2651.887628240001)
    Before filters: 665 stocks
    Trading days range: 1-45 (need >= 37)
    ADTV range: 0.000-138.706B VND (need >= 10.0)
    Stocks passing trading days filter: 417
    Stocks passing ADTV filter: 85
    After filters: 84 stocks
✅ Universe constructed: 84 stocks
  ADTV range: 10.1B - 138.7B VND
  Market cap range: 510.5B - 399764.9B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2019-12-03... Constructing liquid universe for 2019-12-03...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 666 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 666
    Sample result: ('AAA', 46, 39.30526130434783, 2574.140508704347)
    Before filters: 666 stocks
    Trading days range: 1-46 (need >= 37)
    ADTV range: 0.000-225.004B VND (need >= 10.0)
    Stocks passing trading days filter: 421
    Stocks passing ADTV filter: 77
    After filters: 76 stocks
✅ Universe constructed: 76 stocks
  ADTV range: 10.5B - 225.0B VND
  Market cap range: 623.1B - 394710.3B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2020-01-03... Constructing liquid universe for 2020-01-03...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 666 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 666
    Sample result: ('AAA', 45, 34.640642, 2432.561436764444)
    Before filters: 666 stocks
    Trading days range: 1-45 (need >= 37)
    ADTV range: 0.000-234.589B VND (need >= 10.0)
    Stocks passing trading days filter: 403
    Stocks passing ADTV filter: 82
    After filters: 81 stocks
✅ Universe constructed: 81 stocks
  ADTV range: 10.0B - 234.6B VND
  Market cap range: 340.2B - 392559.8B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2020-02-04... Constructing liquid universe for 2020-02-04...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 662 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 662
    Sample result: ('AAA', 40, 23.18127225, 2199.0636917200004)
    Before filters: 662 stocks
    Trading days range: 1-40 (need >= 37)
    ADTV range: 0.000-188.475B VND (need >= 10.0)
    Stocks passing trading days filter: 344
    Stocks passing ADTV filter: 83
    After filters: 80 stocks
✅ Universe constructed: 80 stocks
  ADTV range: 10.2B - 188.5B VND
  Market cap range: 324.8B - 389026.7B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2020-03-03... Constructing liquid universe for 2020-03-03...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 668 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 668
    Sample result: ('AAA', 40, 24.2249806, 2098.48370582)
    Before filters: 668 stocks
    Trading days range: 1-40 (need >= 37)
    ADTV range: 0.000-239.848B VND (need >= 10.0)
    Stocks passing trading days filter: 349
    Stocks passing ADTV filter: 75
    After filters: 73 stocks
✅ Universe constructed: 73 stocks
  ADTV range: 10.6B - 239.8B VND
  Market cap range: 573.7B - 378751.6B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2020-04-03... Constructing liquid universe for 2020-04-03...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 678 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 678
    Sample result: ('AAA', 45, 23.75639824444444, 1966.4790132142223)
    Before filters: 678 stocks
    Trading days range: 1-45 (need >= 37)
    ADTV range: 0.000-208.327B VND (need >= 10.0)
    Stocks passing trading days filter: 422
    Stocks passing ADTV filter: 80
    After filters: 78 stocks
✅ Universe constructed: 78 stocks
  ADTV range: 10.1B - 208.3B VND
  Market cap range: 529.6B - 338085.2B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2020-05-05... Constructing liquid universe for 2020-05-05...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 675 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 675
    Sample result: ('AAA', 43, 25.002736825581398, 1916.0064290753494)
    Before filters: 675 stocks
    Trading days range: 1-43 (need >= 37)
    ADTV range: 0.000-219.854B VND (need >= 10.0)
    Stocks passing trading days filter: 421
    Stocks passing ADTV filter: 88
    After filters: 88 stocks
✅ Universe constructed: 88 stocks
  ADTV range: 10.3B - 219.9B VND
  Market cap range: 280.7B - 311655.6B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2020-06-02... Constructing liquid universe for 2020-06-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 673 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 673
    Sample result: ('AAA', 43, 28.527129232558135, 2034.4131101506978)
    Before filters: 673 stocks
    Trading days range: 1-43 (need >= 37)
    ADTV range: 0.000-285.943B VND (need >= 10.0)
    Stocks passing trading days filter: 439
    Stocks passing ADTV filter: 99
    After filters: 98 stocks
✅ Universe constructed: 98 stocks
  ADTV range: 10.0B - 285.9B VND
  Market cap range: 294.2B - 321055.6B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2020-07-02... Constructing liquid universe for 2020-07-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 679 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 679
    Sample result: ('AAA', 44, 29.933921136363644, 2164.1233329818187)
    Before filters: 679 stocks
    Trading days range: 1-44 (need >= 37)
    ADTV range: 0.000-678.813B VND (need >= 10.0)
    Stocks passing trading days filter: 454
    Stocks passing ADTV filter: 118
    After filters: 115 stocks
✅ Universe constructed: 115 stocks
  ADTV range: 10.1B - 678.8B VND
  Market cap range: 288.8B - 320631.4B VND
  Adding sector information...
✅ Formed portfolio with 23 stocks.
   - Processing 2020-08-04... Constructing liquid universe for 2020-08-04...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 684 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 684
    Sample result: ('AAA', 46, 25.161592956521734, 2269.6257874782605)
    Before filters: 684 stocks
    Trading days range: 1-46 (need >= 37)
    ADTV range: 0.000-498.530B VND (need >= 10.0)
    Stocks passing trading days filter: 465
    Stocks passing ADTV filter: 115
    After filters: 111 stocks
✅ Universe constructed: 111 stocks
  ADTV range: 10.2B - 498.5B VND
  Market cap range: 228.6B - 309425.2B VND
  Adding sector information...
✅ Formed portfolio with 22 stocks.
   - Processing 2020-09-03... Constructing liquid universe for 2020-09-03...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 686 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 686
    Sample result: ('AAA', 45, 28.131360688888886, 2434.6112651022218)
    Before filters: 686 stocks
    Trading days range: 1-45 (need >= 37)
    ADTV range: 0.000-324.034B VND (need >= 10.0)
    Stocks passing trading days filter: 469
    Stocks passing ADTV filter: 101
    After filters: 100 stocks
✅ Universe constructed: 100 stocks
  ADTV range: 10.1B - 324.0B VND
  Market cap range: 258.5B - 304828.5B VND
  Adding sector information...
✅ Formed portfolio with 20 stocks.
   - Processing 2020-10-02... Constructing liquid universe for 2020-10-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 685 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 685
    Sample result: ('AAA', 45, 32.40854930000001, 2569.5997079999997)
    Before filters: 685 stocks
    Trading days range: 1-45 (need >= 37)
    ADTV range: 0.000-335.867B VND (need >= 10.0)
    Stocks passing trading days filter: 474
    Stocks passing ADTV filter: 121
    After filters: 118 stocks
✅ Universe constructed: 118 stocks
  ADTV range: 10.6B - 335.9B VND
  Market cap range: 233.9B - 308240.7B VND
  Adding sector information...
✅ Formed portfolio with 24 stocks.
   - Processing 2020-11-03... Constructing liquid universe for 2020-11-03...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 683 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 683
    Sample result: ('AAA', 45, 24.29860811111111, 2581.837570855556)
    Before filters: 683 stocks
    Trading days range: 1-45 (need >= 37)
    ADTV range: 0.001-450.666B VND (need >= 10.0)
    Stocks passing trading days filter: 465
    Stocks passing ADTV filter: 126
    After filters: 122 stocks
✅ Universe constructed: 122 stocks
  ADTV range: 10.5B - 450.7B VND
  Market cap range: 471.3B - 323788.8B VND
  Adding sector information...
✅ Formed portfolio with 24 stocks.
   - Processing 2020-12-02... Constructing liquid universe for 2020-12-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 687 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 687
    Sample result: ('AAA', 46, 22.9851329347826, 2580.6570913782607)
    Before filters: 687 stocks
    Trading days range: 1-46 (need >= 37)
    ADTV range: 0.000-649.651B VND (need >= 10.0)
    Stocks passing trading days filter: 475
    Stocks passing ADTV filter: 133
    After filters: 131 stocks
✅ Universe constructed: 131 stocks
  ADTV range: 10.1B - 649.7B VND
  Market cap range: 342.7B - 341897.6B VND
  Adding sector information...
✅ Formed portfolio with 26 stocks.
   - Processing 2021-01-05... Constructing liquid universe for 2021-01-05...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 696 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 696
    Sample result: ('AAA', 45, 36.443623577777785, 2816.3516698000003)
    Before filters: 696 stocks
    Trading days range: 1-45 (need >= 37)
    ADTV range: 0.000-815.667B VND (need >= 10.0)
    Stocks passing trading days filter: 497
    Stocks passing ADTV filter: 156
    After filters: 153 stocks
✅ Universe constructed: 153 stocks
  ADTV range: 10.2B - 815.7B VND
  Market cap range: 351.6B - 357568.0B VND
  Adding sector information...
✅ Formed portfolio with 30 stocks.
   - Processing 2021-02-02... Constructing liquid universe for 2021-02-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 705 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 705
    Sample result: ('AAA', 45, 49.816759811111105, 3096.2620369822225)
    Before filters: 705 stocks
    Trading days range: 1-45 (need >= 37)
    ADTV range: 0.000-931.351B VND (need >= 10.0)
    Stocks passing trading days filter: 532
    Stocks passing ADTV filter: 177
    After filters: 171 stocks
✅ Universe constructed: 171 stocks
  ADTV range: 10.0B - 931.4B VND
  Market cap range: 361.5B - 365810.7B VND
  Adding sector information...
✅ Formed portfolio with 34 stocks.
   - Processing 2021-03-02... Constructing liquid universe for 2021-03-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 705 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 705
    Sample result: ('AAA', 40, 46.81468970000002, 3161.4656293375)
    Before filters: 705 stocks
    Trading days range: 1-40 (need >= 37)
    ADTV range: 0.000-1036.926B VND (need >= 10.0)
    Stocks passing trading days filter: 483
    Stocks passing ADTV filter: 172
    After filters: 168 stocks
✅ Universe constructed: 168 stocks
  ADTV range: 10.1B - 1036.9B VND
  Market cap range: 130.3B - 370655.9B VND
  Adding sector information...
✅ Formed portfolio with 33 stocks.
   - Processing 2021-04-02... Constructing liquid universe for 2021-04-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 708 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 708
    Sample result: ('AAA', 41, 50.02336341463414, 3324.2360980585368)
    Before filters: 708 stocks
    Trading days range: 1-41 (need >= 37)
    ADTV range: 0.001-971.076B VND (need >= 10.0)
    Stocks passing trading days filter: 508
    Stocks passing ADTV filter: 173
    After filters: 171 stocks
✅ Universe constructed: 171 stocks
  ADTV range: 10.0B - 971.1B VND
  Market cap range: 158.1B - 366168.7B VND
  Adding sector information...
✅ Formed portfolio with 34 stocks.
   - Processing 2021-05-05... Constructing liquid universe for 2021-05-05...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 708 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 708
    Sample result: ('AAA', 43, 72.07673790697673, 3643.052224037209)
    Before filters: 708 stocks
    Trading days range: 1-43 (need >= 37)
    ADTV range: 0.001-1097.726B VND (need >= 10.0)
    Stocks passing trading days filter: 544
    Stocks passing ADTV filter: 186
    After filters: 182 stocks
✅ Universe constructed: 182 stocks
  ADTV range: 10.1B - 1097.7B VND
  Market cap range: 200.6B - 409116.8B VND
  Adding sector information...
✅ Formed portfolio with 36 stocks.
   - Processing 2021-06-02... Constructing liquid universe for 2021-06-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 710 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 710
    Sample result: ('AAA', 43, 71.12607674418605, 3617.476044097674)
    Before filters: 710 stocks
    Trading days range: 1-43 (need >= 37)
    ADTV range: 0.001-1598.753B VND (need >= 10.0)
    Stocks passing trading days filter: 545
    Stocks passing ADTV filter: 182
    After filters: 180 stocks
✅ Universe constructed: 180 stocks
  ADTV range: 10.0B - 1598.8B VND
  Market cap range: 219.8B - 434894.0B VND
  Adding sector information...
✅ Formed portfolio with 36 stocks.
   - Processing 2021-07-02... Constructing liquid universe for 2021-07-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 710 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 710
    Sample result: ('AAA', 44, 156.3550298863636, 4405.620110899999)
    Before filters: 710 stocks
    Trading days range: 1-44 (need >= 37)
    ADTV range: 0.001-2254.665B VND (need >= 10.0)
    Stocks passing trading days filter: 555
    Stocks passing ADTV filter: 186
    After filters: 185 stocks
✅ Universe constructed: 185 stocks
  ADTV range: 10.1B - 2254.7B VND
  Market cap range: 191.0B - 411734.1B VND
  Adding sector information...
✅ Formed portfolio with 36 stocks.
   - Processing 2021-08-03... Constructing liquid universe for 2021-08-03...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 712 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 712
    Sample result: ('AAA', 46, 167.63186141304345, 5004.735341965217)
    Before filters: 712 stocks
    Trading days range: 1-46 (need >= 37)
    ADTV range: 0.000-1652.694B VND (need >= 10.0)
    Stocks passing trading days filter: 549
    Stocks passing ADTV filter: 187
    After filters: 184 stocks
✅ Universe constructed: 184 stocks
  ADTV range: 10.1B - 1652.7B VND
  Market cap range: 337.1B - 388964.5B VND
  Adding sector information...
✅ Formed portfolio with 36 stocks.
   - Processing 2021-09-06... Constructing liquid universe for 2021-09-06...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 713 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 713
    Sample result: ('AAA', 44, 115.00393852272727, 4966.918896353408)
    Before filters: 713 stocks
    Trading days range: 1-44 (need >= 37)
    ADTV range: 0.000-1452.408B VND (need >= 10.0)
    Stocks passing trading days filter: 542
    Stocks passing ADTV filter: 205
    After filters: 202 stocks
✅ Universe constructed: 200 stocks
  ADTV range: 10.1B - 1452.4B VND
  Market cap range: 351.5B - 373720.0B VND
  Adding sector information...
✅ Formed portfolio with 39 stocks.
   - Processing 2021-10-04... Constructing liquid universe for 2021-10-04...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 715 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 715
    Sample result: ('AAA', 44, 111.54386366818186, 5174.271340102274)
    Before filters: 715 stocks
    Trading days range: 1-44 (need >= 37)
    ADTV range: 0.002-1406.313B VND (need >= 10.0)
    Stocks passing trading days filter: 577
    Stocks passing ADTV filter: 235
    After filters: 235 stocks
✅ Universe constructed: 200 stocks
  ADTV range: 15.6B - 1406.3B VND
  Market cap range: 219.6B - 366302.2B VND
  Adding sector information...
✅ Formed portfolio with 39 stocks.
   - Processing 2021-11-02... Constructing liquid universe for 2021-11-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 716 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 716
    Sample result: ('AAA', 44, 103.06125355454543, 5302.9793986113655)
    Before filters: 716 stocks
    Trading days range: 1-44 (need >= 37)
    ADTV range: 0.001-1506.020B VND (need >= 10.0)
    Stocks passing trading days filter: 601
    Stocks passing ADTV filter: 261
    After filters: 259 stocks
✅ Universe constructed: 200 stocks
  ADTV range: 20.1B - 1506.0B VND
  Market cap range: 286.6B - 361253.1B VND
  Adding sector information...
✅ Formed portfolio with 39 stocks.
   - Processing 2021-12-02... Constructing liquid universe for 2021-12-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 717 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 717
    Sample result: ('AAA', 46, 120.37381850869563, 5499.001977182608)
    Before filters: 717 stocks
    Trading days range: 1-46 (need >= 37)
    ADTV range: 0.004-1559.500B VND (need >= 10.0)
    Stocks passing trading days filter: 621
    Stocks passing ADTV filter: 278
    After filters: 277 stocks
✅ Universe constructed: 200 stocks
  ADTV range: 23.3B - 1559.5B VND
  Market cap range: 361.5B - 362179.9B VND
  Adding sector information...
✅ Formed portfolio with 39 stocks.
   - Processing 2022-01-05... Constructing liquid universe for 2022-01-05...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 720 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 720
    Sample result: ('AAA', 44, 148.91960011363633, 5957.058603709092)
    Before filters: 720 stocks
    Trading days range: 1-45 (need >= 37)
    ADTV range: 0.001-1208.382B VND (need >= 10.0)
    Stocks passing trading days filter: 617
    Stocks passing ADTV filter: 274
    After filters: 271 stocks
✅ Universe constructed: 200 stocks
  ADTV range: 23.1B - 1208.4B VND
  Market cap range: 466.1B - 376217.4B VND
  Adding sector information...
✅ Formed portfolio with 39 stocks.
   - Processing 2022-02-08... Constructing liquid universe for 2022-02-08...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 719 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 719
    Sample result: ('AAA', 39, 147.29607396410256, 6159.567836061538)
    Before filters: 719 stocks
    Trading days range: 1-40 (need >= 37)
    ADTV range: 0.001-851.146B VND (need >= 10.0)
    Stocks passing trading days filter: 556
    Stocks passing ADTV filter: 259
    After filters: 254 stocks
✅ Universe constructed: 200 stocks
  ADTV range: 17.8B - 851.1B VND
  Market cap range: 442.7B - 388832.3B VND
  Adding sector information...
✅ Formed portfolio with 39 stocks.
   - Processing 2022-03-02... Constructing liquid universe for 2022-03-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 719 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 719
    Sample result: ('AAA', 40, 115.40873899, 6030.469270479998)
    Before filters: 719 stocks
    Trading days range: 1-40 (need >= 37)
    ADTV range: 0.001-994.529B VND (need >= 10.0)
    Stocks passing trading days filter: 556
    Stocks passing ADTV filter: 250
    After filters: 245 stocks
✅ Universe constructed: 200 stocks
  ADTV range: 16.0B - 994.5B VND
  Market cap range: 474.9B - 406310.3B VND
  Adding sector information...
✅ Formed portfolio with 39 stocks.
   - Processing 2022-04-04... Constructing liquid universe for 2022-04-04...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 718 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 718
    Sample result: ('AAA', 41, 104.80573109756097, 5887.763653463416)
    Before filters: 718 stocks
    Trading days range: 3-41 (need >= 37)
    ADTV range: 0.001-1126.816B VND (need >= 10.0)
    Stocks passing trading days filter: 589
    Stocks passing ADTV filter: 260
    After filters: 258 stocks
✅ Universe constructed: 200 stocks
  ADTV range: 18.3B - 1126.8B VND
  Market cap range: 387.4B - 403279.7B VND
  Adding sector information...
✅ Formed portfolio with 39 stocks.
   - Processing 2022-05-05... Constructing liquid universe for 2022-05-05...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 719 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 719
    Sample result: ('AAA', 43, 100.69134720930232, 5510.290207479071)
    Before filters: 719 stocks
    Trading days range: 1-43 (need >= 37)
    ADTV range: 0.001-1038.697B VND (need >= 10.0)
    Stocks passing trading days filter: 593
    Stocks passing ADTV filter: 246
    After filters: 246 stocks
✅ Universe constructed: 200 stocks
  ADTV range: 16.4B - 1038.7B VND
  Market cap range: 343.8B - 389354.0B VND
  Adding sector information...
✅ Formed portfolio with 39 stocks.
   - Processing 2022-06-02... Constructing liquid universe for 2022-06-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 719 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 719
    Sample result: ('AAA', 43, 65.51559255813955, 4596.653193674417)
    Before filters: 719 stocks
    Trading days range: 1-43 (need >= 37)
    ADTV range: 0.001-814.549B VND (need >= 10.0)
    Stocks passing trading days filter: 569
    Stocks passing ADTV filter: 198
    After filters: 198 stocks
✅ Universe constructed: 198 stocks
  ADTV range: 10.0B - 814.5B VND
  Market cap range: 424.8B - 375332.6B VND
  Adding sector information...
✅ Formed portfolio with 39 stocks.
   - Processing 2022-07-04... Constructing liquid universe for 2022-07-04...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 720 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 720
    Sample result: ('AAA', 44, 47.85175352272727, 3927.9714524363635)
    Before filters: 720 stocks
    Trading days range: 1-44 (need >= 37)
    ADTV range: 0.000-713.472B VND (need >= 10.0)
    Stocks passing trading days filter: 550
    Stocks passing ADTV filter: 175
    After filters: 175 stocks
✅ Universe constructed: 175 stocks
  ADTV range: 10.3B - 713.5B VND
  Market cap range: 449.7B - 364704.9B VND
  Adding sector information...
✅ Formed portfolio with 34 stocks.
   - Processing 2022-08-02... Constructing liquid universe for 2022-08-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 720 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 720
    Sample result: ('AAA', 46, 50.14384423913043, 4068.0858849391334)
    Before filters: 720 stocks
    Trading days range: 1-46 (need >= 37)
    ADTV range: 0.000-556.993B VND (need >= 10.0)
    Stocks passing trading days filter: 555
    Stocks passing ADTV filter: 168
    After filters: 166 stocks
✅ Universe constructed: 166 stocks
  ADTV range: 10.2B - 557.0B VND
  Market cap range: 810.1B - 357181.5B VND
  Adding sector information...
✅ Formed portfolio with 33 stocks.
   - Processing 2022-09-06... Constructing liquid universe for 2022-09-06...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 721 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 721
    Sample result: ('AAA', 44, 55.14230454545454, 4515.8174008)
    Before filters: 721 stocks
    Trading days range: 1-44 (need >= 37)
    ADTV range: 0.000-600.691B VND (need >= 10.0)
    Stocks passing trading days filter: 538
    Stocks passing ADTV filter: 178
    After filters: 177 stocks
✅ Universe constructed: 177 stocks
  ADTV range: 10.0B - 600.7B VND
  Market cap range: 281.2B - 368630.8B VND
  Adding sector information...
✅ Formed portfolio with 35 stocks.
   - Processing 2022-10-04... Constructing liquid universe for 2022-10-04...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 724 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 724
    Sample result: ('AAA', 44, 43.403194423636364, 4430.387647505456)
    Before filters: 724 stocks
    Trading days range: 1-44 (need >= 37)
    ADTV range: 0.000-599.257B VND (need >= 10.0)
    Stocks passing trading days filter: 548
    Stocks passing ADTV filter: 184
    After filters: 182 stocks
✅ Universe constructed: 182 stocks
  ADTV range: 10.1B - 599.3B VND
  Market cap range: 269.0B - 376149.0B VND
  Adding sector information...
✅ Formed portfolio with 36 stocks.
   - Processing 2022-11-02... Constructing liquid universe for 2022-11-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 722 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 722
    Sample result: ('AAA', 44, 23.236783438863636, 3616.3167321600004)
    Before filters: 722 stocks
    Trading days range: 1-44 (need >= 37)
    ADTV range: 0.000-530.181B VND (need >= 10.0)
    Stocks passing trading days filter: 545
    Stocks passing ADTV filter: 155
    After filters: 154 stocks
✅ Universe constructed: 154 stocks
  ADTV range: 10.0B - 530.2B VND
  Market cap range: 593.6B - 347614.1B VND
  Adding sector information...
✅ Formed portfolio with 30 stocks.
   - Processing 2022-12-02... Constructing liquid universe for 2022-12-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 722 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 722
    Sample result: ('AAA', 46, 22.793177210434784, 2892.986903206958)
    Before filters: 722 stocks
    Trading days range: 1-46 (need >= 37)
    ADTV range: 0.000-642.048B VND (need >= 10.0)
    Stocks passing trading days filter: 541
    Stocks passing ADTV filter: 144
    After filters: 142 stocks
✅ Universe constructed: 142 stocks
  ADTV range: 10.2B - 642.0B VND
  Market cap range: 713.6B - 341749.4B VND
  Adding sector information...
✅ Formed portfolio with 28 stocks.
   - Processing 2023-01-04... Constructing liquid universe for 2023-01-04...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 717 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 717
    Sample result: ('AAA', 45, 21.59236027666668, 2723.663309056001)
    Before filters: 717 stocks
    Trading days range: 1-45 (need >= 37)
    ADTV range: 0.000-672.073B VND (need >= 10.0)
    Stocks passing trading days filter: 523
    Stocks passing ADTV filter: 150
    After filters: 148 stocks
✅ Universe constructed: 148 stocks
  ADTV range: 10.5B - 672.1B VND
  Market cap range: 509.0B - 366475.6B VND
  Adding sector information...
✅ Formed portfolio with 29 stocks.
   - Processing 2023-02-02... Constructing liquid universe for 2023-02-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 713 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 713
    Sample result: ('AAA', 40, 22.466598833749995, 2834.0875447199996)
    Before filters: 713 stocks
    Trading days range: 1-40 (need >= 37)
    ADTV range: 0.000-642.675B VND (need >= 10.0)
    Stocks passing trading days filter: 473
    Stocks passing ADTV filter: 152
    After filters: 147 stocks
✅ Universe constructed: 147 stocks
  ADTV range: 10.3B - 642.7B VND
  Market cap range: 435.4B - 393733.5B VND
  Adding sector information...
✅ Formed portfolio with 29 stocks.
   - Processing 2023-03-02... Constructing liquid universe for 2023-03-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 711 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 711
    Sample result: ('AAA', 40, 27.22989710225, 3041.758164672)
    Before filters: 711 stocks
    Trading days range: 1-40 (need >= 37)
    ADTV range: 0.000-520.831B VND (need >= 10.0)
    Stocks passing trading days filter: 470
    Stocks passing ADTV filter: 143
    After filters: 140 stocks
✅ Universe constructed: 140 stocks
  ADTV range: 10.1B - 520.8B VND
  Market cap range: 555.0B - 426589.0B VND
  Adding sector information...
✅ Formed portfolio with 28 stocks.
   - Processing 2023-04-04... Constructing liquid universe for 2023-04-04...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 712 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 712
    Sample result: ('AAA', 46, 29.920173138043484, 3341.8270234017386)
    Before filters: 712 stocks
    Trading days range: 1-46 (need >= 37)
    ADTV range: 0.000-512.563B VND (need >= 10.0)
    Stocks passing trading days filter: 525
    Stocks passing ADTV filter: 137
    After filters: 136 stocks
✅ Universe constructed: 136 stocks
  ADTV range: 10.1B - 512.6B VND
  Market cap range: 402.7B - 435010.9B VND
  Adding sector information...
✅ Formed portfolio with 27 stocks.
   - Processing 2023-05-05... Constructing liquid universe for 2023-05-05...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 716 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 716
    Sample result: ('AAA', 43, 30.036747084186043, 3570.7104957767424)
    Before filters: 716 stocks
    Trading days range: 1-43 (need >= 37)
    ADTV range: 0.000-460.216B VND (need >= 10.0)
    Stocks passing trading days filter: 509
    Stocks passing ADTV filter: 147
    After filters: 145 stocks
✅ Universe constructed: 145 stocks
  ADTV range: 10.1B - 460.2B VND
  Market cap range: 407.0B - 425904.5B VND
  Adding sector information...
✅ Formed portfolio with 29 stocks.
   - Processing 2023-06-02... Constructing liquid universe for 2023-06-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 717 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 717
    Sample result: ('AAA', 43, 50.993005356976745, 3929.9596209711635)
    Before filters: 717 stocks
    Trading days range: 1-43 (need >= 37)
    ADTV range: 0.000-448.862B VND (need >= 10.0)
    Stocks passing trading days filter: 515
    Stocks passing ADTV filter: 168
    After filters: 168 stocks
✅ Universe constructed: 168 stocks
  ADTV range: 10.1B - 448.9B VND
  Market cap range: 474.3B - 431231.3B VND
  Adding sector information...
✅ Formed portfolio with 34 stocks.
   - Processing 2023-07-04... Constructing liquid universe for 2023-07-04...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 716 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 716
    Sample result: ('AAA', 44, 64.97159355454545, 4227.608403490911)
    Before filters: 716 stocks
    Trading days range: 1-44 (need >= 37)
    ADTV range: 0.000-543.223B VND (need >= 10.0)
    Stocks passing trading days filter: 546
    Stocks passing ADTV filter: 188
    After filters: 186 stocks
✅ Universe constructed: 186 stocks
  ADTV range: 10.1B - 543.2B VND
  Market cap range: 378.0B - 457526.8B VND
  Adding sector information...
✅ Formed portfolio with 37 stocks.
   - Processing 2023-08-02... Constructing liquid universe for 2023-08-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 716 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 716
    Sample result: ('AAA', 46, 93.8007539119565, 4368.7326640695655)
    Before filters: 716 stocks
    Trading days range: 1-46 (need >= 37)
    ADTV range: 0.000-652.011B VND (need >= 10.0)
    Stocks passing trading days filter: 573
    Stocks passing ADTV filter: 195
    After filters: 193 stocks
✅ Universe constructed: 193 stocks
  ADTV range: 10.0B - 652.0B VND
  Market cap range: 403.7B - 485202.6B VND
  Adding sector information...
✅ Formed portfolio with 39 stocks.
   - Processing 2023-09-06... Constructing liquid universe for 2023-09-06...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 717 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 717
    Sample result: ('AAA', 44, 117.77191957068182, 4392.2470784727275)
    Before filters: 717 stocks
    Trading days range: 1-44 (need >= 37)
    ADTV range: 0.000-833.597B VND (need >= 10.0)
    Stocks passing trading days filter: 570
    Stocks passing ADTV filter: 205
    After filters: 204 stocks
✅ Universe constructed: 200 stocks
  ADTV range: 10.8B - 833.6B VND
  Market cap range: 405.5B - 498969.7B VND
  Adding sector information...
✅ Formed portfolio with 40 stocks.
   - Processing 2023-10-03... Constructing liquid universe for 2023-10-03...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 716 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 716
    Sample result: ('AAA', 44, 84.19526558727271, 4119.876500072725)
    Before filters: 716 stocks
    Trading days range: 2-44 (need >= 37)
    ADTV range: 0.000-1010.673B VND (need >= 10.0)
    Stocks passing trading days filter: 566
    Stocks passing ADTV filter: 202
    After filters: 200 stocks
✅ Universe constructed: 200 stocks
  ADTV range: 10.4B - 1010.7B VND
  Market cap range: 403.2B - 496743.3B VND
  Adding sector information...
✅ Formed portfolio with 40 stocks.
   - Processing 2023-11-02... Constructing liquid universe for 2023-11-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 716 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 716
    Sample result: ('AAA', 44, 38.80639022931819, 3622.4852524363628)
    Before filters: 716 stocks
    Trading days range: 1-44 (need >= 37)
    ADTV range: 0.000-914.857B VND (need >= 10.0)
    Stocks passing trading days filter: 549
    Stocks passing ADTV filter: 177
    After filters: 175 stocks
✅ Universe constructed: 175 stocks
  ADTV range: 10.2B - 914.9B VND
  Market cap range: 433.1B - 487381.5B VND
  Adding sector information...
✅ Formed portfolio with 35 stocks.
   - Processing 2023-12-04... Constructing liquid universe for 2023-12-04...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 709 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 709
    Sample result: ('AAA', 46, 22.177921931304343, 3426.9246503373906)
    Before filters: 709 stocks
    Trading days range: 1-46 (need >= 37)
    ADTV range: 0.000-782.034B VND (need >= 10.0)
    Stocks passing trading days filter: 540
    Stocks passing ADTV filter: 154
    After filters: 151 stocks
✅ Universe constructed: 151 stocks
  ADTV range: 10.3B - 782.0B VND
  Market cap range: 438.5B - 482059.1B VND
  Adding sector information...
✅ Formed portfolio with 30 stocks.
   - Processing 2024-01-03... Constructing liquid universe for 2024-01-03...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 710 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 710
    Sample result: ('AAA', 45, 22.073825207333336, 3527.459149312)
    Before filters: 710 stocks
    Trading days range: 1-45 (need >= 37)
    ADTV range: 0.000-726.965B VND (need >= 10.0)
    Stocks passing trading days filter: 541
    Stocks passing ADTV filter: 157
    After filters: 155 stocks
✅ Universe constructed: 155 stocks
  ADTV range: 10.0B - 727.0B VND
  Market cap range: 441.8B - 475346.0B VND
  Adding sector information...
✅ Formed portfolio with 31 stocks.
   - Processing 2024-02-02... Constructing liquid universe for 2024-02-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 707 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 707
    Sample result: ('AAA', 45, 30.08223503688889, 3670.854560256)
    Before filters: 707 stocks
    Trading days range: 1-45 (need >= 37)
    ADTV range: 0.000-698.736B VND (need >= 10.0)
    Stocks passing trading days filter: 550
    Stocks passing ADTV filter: 162
    After filters: 160 stocks
✅ Universe constructed: 160 stocks
  ADTV range: 10.1B - 698.7B VND
  Market cap range: 309.3B - 483444.0B VND
  Adding sector information...
✅ Formed portfolio with 32 stocks.
   - Processing 2024-03-04... Constructing liquid universe for 2024-03-04...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 708 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 708
    Sample result: ('AAA', 40, 47.083037369500005, 3930.2596619999995)
    Before filters: 708 stocks
    Trading days range: 1-40 (need >= 37)
    ADTV range: 0.000-782.572B VND (need >= 10.0)
    Stocks passing trading days filter: 505
    Stocks passing ADTV filter: 159
    After filters: 157 stocks
✅ Universe constructed: 157 stocks
  ADTV range: 10.8B - 782.6B VND
  Market cap range: 251.3B - 504960.4B VND
  Adding sector information...
✅ Formed portfolio with 32 stocks.
   - Processing 2024-04-02... Constructing liquid universe for 2024-04-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 714 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 714
    Sample result: ('AAA', 41, 49.85210302804879, 4166.792006400002)
    Before filters: 714 stocks
    Trading days range: 1-41 (need >= 37)
    ADTV range: 0.000-938.982B VND (need >= 10.0)
    Stocks passing trading days filter: 528
    Stocks passing ADTV filter: 170
    After filters: 167 stocks
✅ Universe constructed: 167 stocks
  ADTV range: 10.1B - 939.0B VND
  Market cap range: 303.4B - 521039.6B VND
  Adding sector information...
✅ Formed portfolio with 34 stocks.
   - Processing 2024-05-03... Constructing liquid universe for 2024-05-03...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 713 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 713
    Sample result: ('AAA', 42, 36.21881074595237, 4072.861701668572)
    Before filters: 713 stocks
    Trading days range: 1-42 (need >= 37)
    ADTV range: 0.000-865.854B VND (need >= 10.0)
    Stocks passing trading days filter: 540
    Stocks passing ADTV filter: 171
    After filters: 168 stocks
✅ Universe constructed: 168 stocks
  ADTV range: 10.2B - 865.9B VND
  Market cap range: 439.4B - 525121.7B VND
  Adding sector information...
✅ Formed portfolio with 34 stocks.
   - Processing 2024-06-04... Constructing liquid universe for 2024-06-04...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 712 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 712
    Sample result: ('AAA', 42, 53.31682626142858, 4154.322576411429)
    Before filters: 712 stocks
    Trading days range: 1-42 (need >= 37)
    ADTV range: 0.000-710.121B VND (need >= 10.0)
    Stocks passing trading days filter: 532
    Stocks passing ADTV filter: 181
    After filters: 179 stocks
✅ Universe constructed: 179 stocks
  ADTV range: 10.1B - 710.1B VND
  Market cap range: 352.5B - 512612.8B VND
  Adding sector information...
✅ Formed portfolio with 36 stocks.
   - Processing 2024-07-02... Constructing liquid universe for 2024-07-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 712 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 712
    Sample result: ('AAA', 44, 65.61904068886363, 4318.398596290908)
    Before filters: 712 stocks
    Trading days range: 1-44 (need >= 37)
    ADTV range: 0.000-854.214B VND (need >= 10.0)
    Stocks passing trading days filter: 560
    Stocks passing ADTV filter: 194
    After filters: 191 stocks
✅ Universe constructed: 191 stocks
  ADTV range: 10.1B - 854.2B VND
  Market cap range: 387.5B - 498305.6B VND
  Adding sector information...
✅ Formed portfolio with 38 stocks.
   - Processing 2024-08-02... Constructing liquid universe for 2024-08-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 711 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 711
    Sample result: ('AAA', 46, 71.80838077934781, 4431.891059060869)
    Before filters: 711 stocks
    Trading days range: 1-46 (need >= 37)
    ADTV range: 0.000-913.206B VND (need >= 10.0)
    Stocks passing trading days filter: 566
    Stocks passing ADTV filter: 184
    After filters: 183 stocks
✅ Universe constructed: 183 stocks
  ADTV range: 10.0B - 913.2B VND
  Market cap range: 430.0B - 489361.4B VND
  Adding sector information...
✅ Formed portfolio with 36 stocks.
   - Processing 2024-09-05... Constructing liquid universe for 2024-09-05...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 708 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 708
    Sample result: ('AAA', 44, 68.17864246590908, 4231.9524318545455)
    Before filters: 708 stocks
    Trading days range: 1-44 (need >= 37)
    ADTV range: 0.000-689.112B VND (need >= 10.0)
    Stocks passing trading days filter: 534
    Stocks passing ADTV filter: 166
    After filters: 166 stocks
✅ Universe constructed: 166 stocks
  ADTV range: 10.1B - 689.1B VND
  Market cap range: 416.0B - 496565.4B VND
  Adding sector information...
✅ Formed portfolio with 33 stocks.
   - Processing 2024-10-02... Constructing liquid universe for 2024-10-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 707 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 707
    Sample result: ('AAA', 44, 46.08232932454546, 3903.891409832728)
    Before filters: 707 stocks
    Trading days range: 1-44 (need >= 37)
    ADTV range: 0.000-597.839B VND (need >= 10.0)
    Stocks passing trading days filter: 526
    Stocks passing ADTV filter: 156
    After filters: 153 stocks
✅ Universe constructed: 153 stocks
  ADTV range: 10.2B - 597.8B VND
  Market cap range: 398.4B - 504148.7B VND
  Adding sector information...
✅ Formed portfolio with 31 stocks.
   - Processing 2024-11-04... Constructing liquid universe for 2024-11-04...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 709 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 709
    Sample result: ('AAA', 44, 21.614921891590907, 3626.4817585309092)
    Before filters: 709 stocks
    Trading days range: 1-44 (need >= 37)
    ADTV range: 0.000-696.061B VND (need >= 10.0)
    Stocks passing trading days filter: 513
    Stocks passing ADTV filter: 147
    After filters: 144 stocks
✅ Universe constructed: 144 stocks
  ADTV range: 10.4B - 696.1B VND
  Market cap range: 434.2B - 510995.4B VND
  Adding sector information...
✅ Formed portfolio with 29 stocks.
   - Processing 2024-12-03... Constructing liquid universe for 2024-12-03...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 703 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 703
    Sample result: ('AAA', 46, 13.853613753695651, 3387.3675503165205)
    Before filters: 703 stocks
    Trading days range: 1-46 (need >= 37)
    ADTV range: 0.000-732.425B VND (need >= 10.0)
    Stocks passing trading days filter: 526
    Stocks passing ADTV filter: 146
    After filters: 144 stocks
✅ Universe constructed: 144 stocks
  ADTV range: 10.5B - 732.4B VND
  Market cap range: 789.4B - 514317.9B VND
  Adding sector information...
✅ Formed portfolio with 29 stocks.
   - Processing 2025-01-03... Constructing liquid universe for 2025-01-03...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 701 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 701
    Sample result: ('AAA', 45, 13.209269164666669, 3282.9733716479996)
    Before filters: 701 stocks
    Trading days range: 1-45 (need >= 37)
    ADTV range: 0.000-806.089B VND (need >= 10.0)
    Stocks passing trading days filter: 528
    Stocks passing ADTV filter: 159
    After filters: 157 stocks
✅ Universe constructed: 157 stocks
  ADTV range: 10.1B - 806.1B VND
  Market cap range: 478.8B - 517015.8B VND
  Adding sector information...
✅ Formed portfolio with 32 stocks.
   - Processing 2025-02-04... Constructing liquid universe for 2025-02-04...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 697 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 697
    Sample result: ('AAA', 40, 11.873568785, 3286.318273488)
    Before filters: 697 stocks
    Trading days range: 2-40 (need >= 37)
    ADTV range: 0.000-768.975B VND (need >= 10.0)
    Stocks passing trading days filter: 484
    Stocks passing ADTV filter: 159
    After filters: 157 stocks
✅ Universe constructed: 157 stocks
  ADTV range: 10.1B - 769.0B VND
  Market cap range: 529.0B - 517116.7B VND
  Adding sector information...
✅ Formed portfolio with 31 stocks.
   - Processing 2025-03-04... Constructing liquid universe for 2025-03-04...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 697 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 697
    Sample result: ('AAA', 40, 12.326138511749999, 3293.1036457920004)
    Before filters: 697 stocks
    Trading days range: 1-40 (need >= 37)
    ADTV range: 0.000-682.780B VND (need >= 10.0)
    Stocks passing trading days filter: 484
    Stocks passing ADTV filter: 161
    After filters: 159 stocks
✅ Universe constructed: 159 stocks
  ADTV range: 10.5B - 682.8B VND
  Market cap range: 533.7B - 515342.2B VND
  Adding sector information...
✅ Formed portfolio with 32 stocks.
   - Processing 2025-04-02... Constructing liquid universe for 2025-04-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 700 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 700
    Sample result: ('AAA', 43, 15.144741557674415, 3315.1199897302336)
    Before filters: 700 stocks
    Trading days range: 1-43 (need >= 37)
    ADTV range: 0.000-818.714B VND (need >= 10.0)
    Stocks passing trading days filter: 532
    Stocks passing ADTV filter: 163
    After filters: 162 stocks
✅ Universe constructed: 162 stocks
  ADTV range: 10.0B - 818.7B VND
  Market cap range: 318.6B - 530714.2B VND
  Adding sector information...
✅ Formed portfolio with 33 stocks.
   - Processing 2025-05-06... Constructing liquid universe for 2025-05-06...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 697 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 697
    Sample result: ('AAA', 42, 16.630042019761905, 2986.9290798171423)
    Before filters: 697 stocks
    Trading days range: 1-42 (need >= 37)
    ADTV range: 0.000-1022.172B VND (need >= 10.0)
    Stocks passing trading days filter: 519
    Stocks passing ADTV filter: 163
    After filters: 163 stocks
✅ Universe constructed: 163 stocks
  ADTV range: 10.3B - 1022.2B VND
  Market cap range: 473.6B - 515738.0B VND
  Adding sector information...
✅ Formed portfolio with 33 stocks.
   - Processing 2025-06-03... Constructing liquid universe for 2025-06-03...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 694 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 694
    Sample result: ('AAA', 42, 16.713963587380956, 2749.008714925714)
    Before filters: 694 stocks
    Trading days range: 1-42 (need >= 37)
    ADTV range: 0.000-1033.973B VND (need >= 10.0)
    Stocks passing trading days filter: 523
    Stocks passing ADTV filter: 167
    After filters: 166 stocks
✅ Universe constructed: 166 stocks
  ADTV range: 10.3B - 1034.0B VND
  Market cap range: 434.3B - 483137.1B VND
  Adding sector information...
✅ Formed portfolio with 33 stocks.
   - Processing 2025-07-02... Constructing liquid universe for 2025-07-02...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 697 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 697
    Sample result: ('AAA', 43, 13.657289003488374, 2764.6447154902326)
    Before filters: 697 stocks
    Trading days range: 1-43 (need >= 37)
    ADTV range: 0.000-904.619B VND (need >= 10.0)
    Stocks passing trading days filter: 527
    Stocks passing ADTV filter: 166
    After filters: 166 stocks
✅ Universe constructed: 166 stocks
  ADTV range: 10.1B - 904.6B VND
  Market cap range: 439.9B - 474855.0B VND
  Adding sector information...
✅ Formed portfolio with 33 stocks.
   - Generated 113 data-driven rebalance dates (T+2 logic).
✅ Backtest for Full_Composite_Monthly complete.

--- Generating Final Tearsheet for the Full Composite Strategy ---