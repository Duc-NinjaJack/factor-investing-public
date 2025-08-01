# ============================================================================
# Aureus Sigma Capital - Phase 22d: Mechanical Fixes & Strategy Rebuild
# Notebook: 22d_mechanical_fixes_and_rebuild.ipynb
#
# Objective:
#   To systematically implement the high-priority mechanical and logical fixes
#   identified in the post-bake-off strategic assessment. This notebook will
#   correct known flaws in the backtesting engine and portfolio construction,
#   establish a new, robust performance baseline, and begin the process of
#   enhancing the factor stack.
# ============================================================================
#
# --- STRATEGIC DIRECTIVE & ALIGNMENT ---
#
# This notebook directly implements the "Recommended Fixes" and "Implementation
# Checklist" from the definitive strategic assessment memo. The core findings
# being addressed are:
#
# 1.  **Mechanical Flaws:** A turnover cost calculation error and portfolio
#     hyper-concentration in early years have artificially depressed performance.
# 2.  **Signal Shallowness:** The existing QVR composite is not sufficiently
#     differentiated from pure Value and lacks adaptability to different regimes.
#
# This notebook will proceed sequentially through the P0, P1, and P2 priority fixes.
#
# --- METHODOLOGY: SEQUENTIAL REFINEMENT ---
#
# This notebook will execute the following logical steps, aligned with the 5-day plan:
#
# 1.  **P0 Fixes (Day 1):**
#     -   Correct the turnover calculation in the backtesting engine to divide by 2.
#     -   Refactor all configuration into a single, unified block.
#
# 2.  **P1 Fixes (Day 2):**
#     -   Implement a **hybrid portfolio construction method**:
#         -   If `liquid_universe_size < 100`, select the **Top 20 stocks**.
#         -   If `liquid_universe_size >= 100`, select the **Top 20% (percentile)**.
#     -   Safeguard the z-score calculation against zero-standard-deviation events.
#     -   Re-run the `Standalone Value` strategy to establish a new, corrected baseline.
#
# 3.  **P2 Enhancements (Day 3-5):**
#     -   Introduce a standard `Momentum_Composite` as a potential signal.
#     -   Begin testing more diversified composites and applying a simple
#       volatility targeting overlay.
#
# --- SUCCESS CRITERIA ---
#
# The primary goal is to create a corrected baseline strategy that demonstrates
# a significant improvement over the previous result (Sharpe 0.52). Based on the
# assessment's back-of-the-envelope calculation, we are targeting:
#
#   -   **Corrected Baseline Sharpe Ratio:** ~0.6 - 0.65
#   -   **Enhanced Strategy Sharpe Ratio:** Approaching ~0.75
#
# This will provide the necessary foundation to proceed to Phase 23 for the
# application of more advanced risk overlays and optimization.
#

# ============================================================================
# DAY 1: P0 FIXES - UNIFIED CONFIG & CORRECTED TURNOVER LOGIC
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

# --- P0 FIX 1: SINGLE SOURCE OF TRUTH FOR CONFIGURATION ---
# All configurations are now centralized here.
print("\n⚙️  Initializing unified configuration block...")
BASE_CONFIG = {
    "backtest_start_date": "2016-03-01",
    "backtest_end_date": "2025-07-28",
    "rebalance_frequency": "Q",
    "strategy_version_db": "qvm_v2.0_enhanced",
    "transaction_cost_bps": 30,
}
ALL_CONFIGS = {
    "A_Standalone_Value": {
        **BASE_CONFIG,
        "strategy_name": "A_Standalone_Value",
        "factors_to_combine": {'Value_Composite': 1.0}
    },
    "B_Equal_Weighted_QVM": {
        **BASE_CONFIG,
        "strategy_name": "B_Equal_Weighted_QVM",
        "factors_to_combine": {'Value_Composite': 1/3, 'Quality_Composite': 1/3, 'Momentum_Composite': 1/3}
    },
    "C_Equal_Weighted_QVR": {
        **BASE_CONFIG,
        "strategy_name": "C_Equal_Weighted_QVR",
        "factors_to_combine": {'Value_Composite': 1/3, 'Quality_Composite': 1/3, 'Momentum_Reversal': 1/3}
    }
}
print("✅ Unified configurations defined.")

# --- Data Loading (Condensed into a single function for cleanliness) ---
def load_all_data(config):
    print("\n📂 Loading all raw data...")
    engine = create_engine(f"mysql+pymysql://{yaml.safe_load(open(project_root / 'config' / 'database.yml'))['production']['username']}:{yaml.safe_load(open(project_root / 'config' / 'database.yml'))['production']['password']}@{yaml.safe_load(open(project_root / 'config' / 'database.yml'))['production']['host']}/{yaml.safe_load(open(project_root / 'config' / 'database.yml'))['production']['schema_name']}")
    db_params = {'start_date': "2016-01-01", 'end_date': config['backtest_end_date'], 'strategy_version': config['strategy_version_db']}
    factor_data_raw = pd.read_sql(text("SELECT date, ticker, Quality_Composite, Value_Composite, Momentum_Composite FROM factor_scores_qvm WHERE date BETWEEN :start_date AND :end_date AND strategy_version = :strategy_version"), engine, params=db_params, parse_dates=['date'])
    price_data_raw = pd.read_sql(text("SELECT date, ticker, close FROM equity_history WHERE date BETWEEN :start_date AND :end_date"), engine, params=db_params, parse_dates=['date'])
    benchmark_data_raw = pd.read_sql(text("SELECT date, close FROM etf_history WHERE ticker = 'VNINDEX' AND date BETWEEN :start_date AND :end_date"), engine, params=db_params, parse_dates=['date'])
    price_data_raw['return'] = price_data_raw.groupby('ticker')['close'].pct_change()
    daily_returns_matrix = price_data_raw.pivot(index='date', columns='ticker', values='return')
    benchmark_returns = benchmark_data_raw.set_index('date')['close'].pct_change().rename('VN-Index')
    print("✅ All data loaded and prepared.")
    return factor_data_raw, daily_returns_matrix, benchmark_returns, engine

# --- P0 FIX 2: UNIFIED BACKTESTER v2.0 WITH CORRECTED TURNOVER ---
class UnifiedBacktester_v2_0:
    """
    Version 2.0 of the backtesting engine.
    - Implements the critical turnover / 2 fix.
    - Ready for P1 fixes (hybrid portfolio construction).
    """
    def __init__(self, config: Dict, factor_data: pd.DataFrame, returns_matrix: pd.DataFrame, benchmark_returns: pd.Series, db_engine):
        self.config = config; self.engine = db_engine
        start = self.config['backtest_start_date']; end = self.config['backtest_end_date']
        self.factor_data_raw = factor_data[factor_data['date'].between(start, end)].copy()
        self.daily_returns_matrix = returns_matrix.loc[start:end].copy()
        self.benchmark_returns = benchmark_returns.loc[start:end].copy()
        print(f"\n✅ UnifiedBacktester v2.0 initialized for strategy: '{self.config['strategy_name']}'")

    def run(self) -> pd.Series:
        print(f"--- Executing Backtest for: {self.config['strategy_name']} ---")
        daily_holdings = self._run_backtesting_loop()
        net_returns = self._calculate_net_returns(daily_holdings)
        # self._generate_tearsheet(net_returns) # Tearsheet generation will be done in a separate step
        print(f"✅ Backtest for {self.config['strategy_name']} complete.")
        return net_returns

    def _calculate_net_returns(self, daily_holdings: pd.DataFrame) -> pd.Series:
        """
        Calculates final net returns with the corrected turnover logic.
        """
        holdings_shifted = daily_holdings.shift(1).fillna(0.0)
        gross_returns = (holdings_shifted * self.daily_returns_matrix).sum(axis=1)
        
        # --- P0 FIX: CORRECTED TURNOVER CALCULATION ---
        # Turnover is the sum of absolute weight changes, divided by 2 for round-trip trades.
        turnover = (holdings_shifted - holdings_shifted.shift(1)).abs().sum(axis=1) / 2.0
        
        costs = turnover * (self.config['transaction_cost_bps'] / 10000)
        return (gross_returns - costs).rename(self.config['strategy_name'])

    # --- Other methods (unchanged from previous correct version) ---
    def _generate_rebalance_dates(self) -> List[pd.Timestamp]:
        all_trading_dates = self.daily_returns_matrix.index
        freq_ends = pd.date_range(start=all_trading_dates.min(), end=all_trading_dates.max(), freq=self.config['rebalance_frequency'])
        return [all_trading_dates[all_trading_dates.searchsorted(q_end, side='right') - 1] for q_end in freq_ends]
    def _run_backtesting_loop(self) -> pd.DataFrame:
        rebalance_dates = self._generate_rebalance_dates()
        daily_holdings = pd.DataFrame(0.0, index=self.daily_returns_matrix.index, columns=self.daily_returns_matrix.columns)
        for i in range(len(rebalance_dates)):
            rebal_date = rebalance_dates[i]
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
        # This method will be upgraded in the next step (P1 Fix)
        factors_to_combine = self.config['factors_to_combine']
        if 'Momentum_Reversal' in factors_to_combine: factors_df['Momentum_Reversal'] = -1 * factors_df['Momentum_Composite']
        normalized_scores = []
        for factor_name, weight in factors_to_combine.items():
            if weight == 0: continue
            factor_scores = factors_df[factor_name]
            mean, std = factor_scores.mean(), factor_scores.std()
            normalized_scores.append(((factor_scores - mean) / std if std > 0 else 0.0) * weight)
        if not normalized_scores: return pd.Series(dtype='float64')
        factors_df['final_signal'] = pd.concat(normalized_scores, axis=1).sum(axis=1)
        percentile_cutoff = self.config.get('selection_percentile', 0.8)
        score_cutoff = factors_df['final_signal'].quantile(percentile_cutoff)
        selected_stocks = factors_df[factors_df['final_signal'] >= score_cutoff]
        if selected_stocks.empty: return pd.Series(dtype='float64')
        return pd.Series(1.0 / len(selected_stocks), index=selected_stocks['ticker'])

print("✅ UnifiedBacktester v2.0 (with Corrected Costs) defined successfully.")
print("   Ready to load data and proceed to Day 2 (P1 Fixes).")

✅ Successfully imported production modules.

⚙️  Initializing unified configuration block...
✅ Unified configurations defined.
✅ UnifiedBacktester v2.0 (with Corrected Costs) defined successfully.
   Ready to load data and proceed to Day 2 (P1 Fixes).

# ============================================================================
# DAY 2: P1 FIXES - UNIFIED BACKTESTER v2.2 (DEFINITIVE & SCOPE-CORRECTED)
#
# This version corrects the NameError by properly passing the PALETTE
# dictionary into the class during initialization.
# ============================================================================

from typing import Dict, List, Optional

class UnifiedBacktester_v2_2:
    """
    The definitive backtesting engine incorporating all P0 and P1 fixes.
    Version 2.2 corrects the variable scope issue for the visualization palette.
    """
    def __init__(self, config: Dict, factor_data: pd.DataFrame, returns_matrix: pd.DataFrame, benchmark_returns: pd.Series, db_engine, palette: Dict):
        self.config = config
        self.engine = db_engine
        self.palette = palette # Store the palette
        
        start = self.config['backtest_start_date']
        end = self.config['backtest_end_date']
        
        self.factor_data_raw = factor_data[factor_data['date'].between(start, end)].copy()
        self.daily_returns_matrix = returns_matrix.loc[start:end].copy()
        self.benchmark_returns = benchmark_returns.loc[start:end].copy()
        
        print(f"✅ UnifiedBacktester v2.2 initialized for strategy: '{self.config['strategy_name']}'")
        print(f"   - Data sliced to period: {self.daily_returns_matrix.index.min().date()} to {self.daily_returns_matrix.index.max().date()}")

    def run(self) -> pd.Series:
        print(f"\n--- Executing Backtest for: {self.config['strategy_name']} ---")
        daily_holdings = self._run_backtesting_loop()
        net_returns = self._calculate_net_returns(daily_holdings)
        self._generate_tearsheet(net_returns)
        print(f"\n✅ Backtest for {self.config['strategy_name']} complete.")
        return net_returns

    def _generate_rebalance_dates(self) -> List[pd.Timestamp]:
        all_trading_dates = self.daily_returns_matrix.index
        freq_ends = pd.date_range(start=all_trading_dates.min(), end=all_trading_dates.max(), freq=self.config['rebalance_frequency'])
        return [all_trading_dates[all_trading_dates.searchsorted(q_end, side='right') - 1] for q_end in freq_ends]

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
            
            target_portfolio = self._calculate_target_portfolio(factors_on_date, i, rebal_date)
            if target_portfolio.empty: print("⚠️ Portfolio empty. Skipping."); continue
            
            start_period = rebal_date + pd.Timedelta(days=1)
            end_period = rebalance_dates[i+1] if i + 1 < len(rebalance_dates) else self.daily_returns_matrix.index.max()
            holding_dates = daily_holdings.index[(daily_holdings.index >= start_period) & (daily_holdings.index <= end_period)]
            
            daily_holdings.loc[holding_dates] = 0.0
            valid_tickers = target_portfolio.index.intersection(daily_holdings.columns)
            daily_holdings.loc[holding_dates, valid_tickers] = target_portfolio[valid_tickers].values
            print(f"✅ Formed portfolio.")
            
        return daily_holdings

    def _calculate_target_portfolio(self, factors_df: pd.DataFrame, i: int, rebal_date: pd.Timestamp) -> pd.Series:
        factors_to_combine = self.config['factors_to_combine']
        if 'Momentum_Reversal' in factors_to_combine: factors_df['Momentum_Reversal'] = -1 * factors_df['Momentum_Composite']
        
        normalized_scores = []
        for factor_name, weight in factors_to_combine.items():
            if weight == 0: continue
            factor_scores = factors_df[factor_name]
            mean, std = factor_scores.mean(), factor_scores.std()
            
            if std > 1e-8:
                normalized_score = (factor_scores - mean) / std
            else:
                normalized_score = 0.0
            normalized_scores.append(normalized_score * weight)
            
        if not normalized_scores: return pd.Series(dtype='float64')
        factors_df['final_signal'] = pd.concat(normalized_scores, axis=1).sum(axis=1)
        
        universe_size = len(factors_df)
        if universe_size < 100:
            construction_method = 'fixed_n'
            portfolio_size = self.config.get('portfolio_size_small_universe', 20)
            selected_stocks = factors_df.nlargest(portfolio_size, 'final_signal')
        else:
            construction_method = 'percentile'
            percentile_cutoff = self.config.get('selection_percentile', 0.8)
            score_cutoff = factors_df['final_signal'].quantile(percentile_cutoff)
            selected_stocks = factors_df[factors_df['final_signal'] >= score_cutoff]
            
        if i == 0 or (i+1) % 5 == 0:
             print(f"\n     - Rebalance {rebal_date.date()}: Universe Size={universe_size}, Method='{construction_method}', Portfolio Size={len(selected_stocks)}")
             
        if selected_stocks.empty: return pd.Series(dtype='float64')
        return pd.Series(1.0 / len(selected_stocks), index=selected_stocks['ticker'])

    def _calculate_net_returns(self, daily_holdings: pd.DataFrame) -> pd.Series:
        holdings_shifted = daily_holdings.shift(1).fillna(0.0)
        gross_returns = (holdings_shifted * self.daily_returns_matrix).sum(axis=1)
        turnover = (holdings_shifted - holdings_shifted.shift(1)).abs().sum(axis=1) / 2.0
        costs = turnover * (self.config['transaction_cost_bps'] / 10000)
        return (gross_returns - costs).rename(self.config['strategy_name'])

    def _calculate_performance_metrics(self, returns: pd.Series) -> Dict:
        benchmark = self.benchmark_returns; first_trade_date = returns.loc[returns.ne(0)].index.min(); common_index = returns.loc[first_trade_date:].index.intersection(benchmark.index); returns, benchmark = returns.loc[common_index], benchmark.loc[common_index]; n_years = len(returns) / 252; annual_return = ((1 + returns).prod() ** (1 / n_years) - 1) if n_years > 0 else 0; annual_vol = returns.std() * np.sqrt(252); sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0; cumulative = (1 + returns).cumprod(); drawdown = (cumulative / cumulative.cummax() - 1); max_drawdown = drawdown.min(); calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0; excess_returns = returns - benchmark; information_ratio = (excess_returns.mean() * 252) / (excess_returns.std() * np.sqrt(252)) if excess_returns.std() > 0 else 0; beta = np.cov(returns.fillna(0), benchmark.fillna(0))[0, 1] / np.var(benchmark.fillna(0)); alpha = (returns.mean() - beta * benchmark.mean()) * 252; return {'Annual Return (%)': annual_return*100, 'Annual Volatility (%)': annual_vol*100, 'Sharpe Ratio': sharpe_ratio, 'Max Drawdown (%)': max_drawdown*100, 'Calmar Ratio': calmar_ratio, 'Beta': beta, 'Alpha (%)': alpha*100, 'Information Ratio': information_ratio}

    def _generate_tearsheet(self, strategy_returns: pd.Series):
        # This method now correctly uses self.palette
        portfolio_metrics = self._calculate_performance_metrics(strategy_returns); benchmark_metrics = self._calculate_performance_metrics(self.benchmark_returns); first_trade_date = strategy_returns.loc[strategy_returns.ne(0)].index.min(); strategy_cum = (1 + strategy_returns.loc[first_trade_date:]).cumprod(); benchmark_cum = (1 + self.benchmark_returns.loc[first_trade_date:]).cumprod(); fig = plt.figure(figsize=(18, 22)); gs = fig.add_gridspec(5, 2, height_ratios=[1.2, 0.8, 0.8, 0.8, 1.2], hspace=0.7, wspace=0.2); title = f"Institutional Tearsheet: {self.config['strategy_name']} ({first_trade_date.year}-{strategy_returns.index.max().year})"; fig.suptitle(title, fontsize=20, fontweight='bold', color=self.palette['text']); ax1 = fig.add_subplot(gs[0, :]); ax1.plot(strategy_cum.index, strategy_cum, label=self.config['strategy_name'], color=self.palette['primary'], linewidth=2.5); ax1.plot(benchmark_cum.index, benchmark_cum, label='VN-Index', color=self.palette['secondary'], linestyle='--', linewidth=2); ax1.set_yscale('log'); ax1.set_title('Cumulative Performance (Log Scale)', fontweight='bold'); ax1.legend(loc='upper left'); ax2 = fig.add_subplot(gs[1, :]); strategy_dd = (strategy_cum / strategy_cum.cummax() - 1) * 100; ax2.plot(strategy_dd.index, strategy_dd, color=self.palette['negative'], linewidth=2); ax2.fill_between(strategy_dd.index, strategy_dd, 0, color=self.palette['negative'], alpha=0.1); ax2.set_title('Drawdown Analysis', fontweight='bold'); ax2.set_ylabel('Drawdown (%)'); plot_row = 2;
        ax3 = fig.add_subplot(gs[plot_row, 0]); strat_annual = strategy_returns.resample('Y').apply(lambda x: (1+x).prod()-1) * 100; bench_annual = self.benchmark_returns.resample('Y').apply(lambda x: (1+x).prod()-1) * 100; pd.DataFrame({'Strategy': strat_annual, 'Benchmark': bench_annual}).plot(kind='bar', ax=ax3, color=[self.palette['primary'], self.palette['secondary']]); ax3.set_xticklabels([d.strftime('%Y') for d in strat_annual.index], rotation=45, ha='right'); ax3.set_title('Annual Returns', fontweight='bold'); ax4 = fig.add_subplot(gs[plot_row, 1]); rolling_sharpe = (strategy_returns.rolling(252).mean() * 252) / (strategy_returns.rolling(252).std() * np.sqrt(252)); ax4.plot(rolling_sharpe.index, rolling_sharpe, color=self.palette['highlight_2']); ax4.axhline(1.0, color=self.palette['positive'], linestyle='--'); ax4.set_title('1-Year Rolling Sharpe Ratio', fontweight='bold'); ax5 = fig.add_subplot(gs[plot_row+1:, :]); ax5.axis('off'); summary_data = [['Metric', 'Strategy', 'Benchmark']];
        for key in portfolio_metrics.keys(): summary_data.append([key, f"{portfolio_metrics[key]:.2f}", f"{benchmark_metrics.get(key, 0.0):.2f}"])
        table = ax5.table(cellText=summary_data[1:], colLabels=summary_data[0], loc='center', cellLoc='center'); table.auto_set_font_size(False); table.set_fontsize(14); table.scale(1, 2.5); plt.tight_layout(rect=[0, 0, 1, 0.97]); plt.show()

print("✅ UnifiedBacktester v2.2 (Scope-Corrected) defined successfully.")

# ============================================================================
# DAY 2: P1 FIXES - UNIFIED BACKTESTER v2.2 (DEFINITIVE & SCOPE-CORRECTED)
#
# This version corrects the NameError by properly passing the PALETTE
# dictionary into the class during initialization.
# ============================================================================

from typing import Dict, List, Optional

class UnifiedBacktester_v2_2:
    """
    The definitive backtesting engine incorporating all P0 and P1 fixes.
    Version 2.2 corrects the variable scope issue for the visualization palette.
    """
    def __init__(self, config: Dict, factor_data: pd.DataFrame, returns_matrix: pd.DataFrame, benchmark_returns: pd.Series, db_engine, palette: Dict):
        self.config = config
        self.engine = db_engine
        self.palette = palette # Store the palette
        
        start = self.config['backtest_start_date']
        end = self.config['backtest_end_date']
        
        self.factor_data_raw = factor_data[factor_data['date'].between(start, end)].copy()
        self.daily_returns_matrix = returns_matrix.loc[start:end].copy()
        self.benchmark_returns = benchmark_returns.loc[start:end].copy()
        
        print(f"✅ UnifiedBacktester v2.2 initialized for strategy: '{self.config['strategy_name']}'")
        print(f"   - Data sliced to period: {self.daily_returns_matrix.index.min().date()} to {self.daily_returns_matrix.index.max().date()}")

    def run(self) -> pd.Series:
        print(f"\n--- Executing Backtest for: {self.config['strategy_name']} ---")
        daily_holdings = self._run_backtesting_loop()
        net_returns = self._calculate_net_returns(daily_holdings)
        self._generate_tearsheet(net_returns)
        print(f"\n✅ Backtest for {self.config['strategy_name']} complete.")
        return net_returns

    def _generate_rebalance_dates(self) -> List[pd.Timestamp]:
        all_trading_dates = self.daily_returns_matrix.index
        freq_ends = pd.date_range(start=all_trading_dates.min(), end=all_trading_dates.max(), freq=self.config['rebalance_frequency'])
        return [all_trading_dates[all_trading_dates.searchsorted(q_end, side='right') - 1] for q_end in freq_ends]

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
            
            target_portfolio = self._calculate_target_portfolio(factors_on_date, i, rebal_date)
            if target_portfolio.empty: print("⚠️ Portfolio empty. Skipping."); continue
            
            start_period = rebal_date + pd.Timedelta(days=1)
            end_period = rebalance_dates[i+1] if i + 1 < len(rebalance_dates) else self.daily_returns_matrix.index.max()
            holding_dates = daily_holdings.index[(daily_holdings.index >= start_period) & (daily_holdings.index <= end_period)]
            
            daily_holdings.loc[holding_dates] = 0.0
            valid_tickers = target_portfolio.index.intersection(daily_holdings.columns)
            daily_holdings.loc[holding_dates, valid_tickers] = target_portfolio[valid_tickers].values
            print(f"✅ Formed portfolio.")
            
        return daily_holdings

    def _calculate_target_portfolio(self, factors_df: pd.DataFrame, i: int, rebal_date: pd.Timestamp) -> pd.Series:
        factors_to_combine = self.config['factors_to_combine']
        if 'Momentum_Reversal' in factors_to_combine: factors_df['Momentum_Reversal'] = -1 * factors_df['Momentum_Composite']
        
        normalized_scores = []
        for factor_name, weight in factors_to_combine.items():
            if weight == 0: continue
            factor_scores = factors_df[factor_name]
            mean, std = factor_scores.mean(), factor_scores.std()
            
            if std > 1e-8:
                normalized_score = (factor_scores - mean) / std
            else:
                normalized_score = 0.0
            normalized_scores.append(normalized_score * weight)
            
        if not normalized_scores: return pd.Series(dtype='float64')
        factors_df['final_signal'] = pd.concat(normalized_scores, axis=1).sum(axis=1)
        
        universe_size = len(factors_df)
        if universe_size < 100:
            construction_method = 'fixed_n'
            portfolio_size = self.config.get('portfolio_size_small_universe', 20)
            selected_stocks = factors_df.nlargest(portfolio_size, 'final_signal')
        else:
            construction_method = 'percentile'
            percentile_cutoff = self.config.get('selection_percentile', 0.8)
            score_cutoff = factors_df['final_signal'].quantile(percentile_cutoff)
            selected_stocks = factors_df[factors_df['final_signal'] >= score_cutoff]
            
        if i == 0 or (i+1) % 5 == 0:
             print(f"\n     - Rebalance {rebal_date.date()}: Universe Size={universe_size}, Method='{construction_method}', Portfolio Size={len(selected_stocks)}")
             
        if selected_stocks.empty: return pd.Series(dtype='float64')
        return pd.Series(1.0 / len(selected_stocks), index=selected_stocks['ticker'])

    def _calculate_net_returns(self, daily_holdings: pd.DataFrame) -> pd.Series:
        holdings_shifted = daily_holdings.shift(1).fillna(0.0)
        gross_returns = (holdings_shifted * self.daily_returns_matrix).sum(axis=1)
        turnover = (holdings_shifted - holdings_shifted.shift(1)).abs().sum(axis=1) / 2.0
        costs = turnover * (self.config['transaction_cost_bps'] / 10000)
        return (gross_returns - costs).rename(self.config['strategy_name'])

    def _calculate_performance_metrics(self, returns: pd.Series) -> Dict:
        benchmark = self.benchmark_returns; first_trade_date = returns.loc[returns.ne(0)].index.min(); common_index = returns.loc[first_trade_date:].index.intersection(benchmark.index); returns, benchmark = returns.loc[common_index], benchmark.loc[common_index]; n_years = len(returns) / 252; annual_return = ((1 + returns).prod() ** (1 / n_years) - 1) if n_years > 0 else 0; annual_vol = returns.std() * np.sqrt(252); sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0; cumulative = (1 + returns).cumprod(); drawdown = (cumulative / cumulative.cummax() - 1); max_drawdown = drawdown.min(); calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0; excess_returns = returns - benchmark; information_ratio = (excess_returns.mean() * 252) / (excess_returns.std() * np.sqrt(252)) if excess_returns.std() > 0 else 0; beta = np.cov(returns.fillna(0), benchmark.fillna(0))[0, 1] / np.var(benchmark.fillna(0)); alpha = (returns.mean() - beta * benchmark.mean()) * 252; return {'Annual Return (%)': annual_return*100, 'Annual Volatility (%)': annual_vol*100, 'Sharpe Ratio': sharpe_ratio, 'Max Drawdown (%)': max_drawdown*100, 'Calmar Ratio': calmar_ratio, 'Beta': beta, 'Alpha (%)': alpha*100, 'Information Ratio': information_ratio}

    def _generate_tearsheet(self, strategy_returns: pd.Series):
        # This method now correctly uses self.palette
        portfolio_metrics = self._calculate_performance_metrics(strategy_returns); benchmark_metrics = self._calculate_performance_metrics(self.benchmark_returns); first_trade_date = strategy_returns.loc[strategy_returns.ne(0)].index.min(); strategy_cum = (1 + strategy_returns.loc[first_trade_date:]).cumprod(); benchmark_cum = (1 + self.benchmark_returns.loc[first_trade_date:]).cumprod(); fig = plt.figure(figsize=(18, 22)); gs = fig.add_gridspec(5, 2, height_ratios=[1.2, 0.8, 0.8, 0.8, 1.2], hspace=0.7, wspace=0.2); title = f"Institutional Tearsheet: {self.config['strategy_name']} ({first_trade_date.year}-{strategy_returns.index.max().year})"; fig.suptitle(title, fontsize=20, fontweight='bold', color=self.palette['text']); ax1 = fig.add_subplot(gs[0, :]); ax1.plot(strategy_cum.index, strategy_cum, label=self.config['strategy_name'], color=self.palette['primary'], linewidth=2.5); ax1.plot(benchmark_cum.index, benchmark_cum, label='VN-Index', color=self.palette['secondary'], linestyle='--', linewidth=2); ax1.set_yscale('log'); ax1.set_title('Cumulative Performance (Log Scale)', fontweight='bold'); ax1.legend(loc='upper left'); ax2 = fig.add_subplot(gs[1, :]); strategy_dd = (strategy_cum / strategy_cum.cummax() - 1) * 100; ax2.plot(strategy_dd.index, strategy_dd, color=self.palette['negative'], linewidth=2); ax2.fill_between(strategy_dd.index, strategy_dd, 0, color=self.palette['negative'], alpha=0.1); ax2.set_title('Drawdown Analysis', fontweight='bold'); ax2.set_ylabel('Drawdown (%)'); plot_row = 2;
        ax3 = fig.add_subplot(gs[plot_row, 0]); strat_annual = strategy_returns.resample('Y').apply(lambda x: (1+x).prod()-1) * 100; bench_annual = self.benchmark_returns.resample('Y').apply(lambda x: (1+x).prod()-1) * 100; pd.DataFrame({'Strategy': strat_annual, 'Benchmark': bench_annual}).plot(kind='bar', ax=ax3, color=[self.palette['primary'], self.palette['secondary']]); ax3.set_xticklabels([d.strftime('%Y') for d in strat_annual.index], rotation=45, ha='right'); ax3.set_title('Annual Returns', fontweight='bold'); ax4 = fig.add_subplot(gs[plot_row, 1]); rolling_sharpe = (strategy_returns.rolling(252).mean() * 252) / (strategy_returns.rolling(252).std() * np.sqrt(252)); ax4.plot(rolling_sharpe.index, rolling_sharpe, color=self.palette['highlight_2']); ax4.axhline(1.0, color=self.palette['positive'], linestyle='--'); ax4.set_title('1-Year Rolling Sharpe Ratio', fontweight='bold'); ax5 = fig.add_subplot(gs[plot_row+1:, :]); ax5.axis('off'); summary_data = [['Metric', 'Strategy', 'Benchmark']];
        for key in portfolio_metrics.keys(): summary_data.append([key, f"{portfolio_metrics[key]:.2f}", f"{benchmark_metrics.get(key, 0.0):.2f}"])
        table = ax5.table(cellText=summary_data[1:], colLabels=summary_data[0], loc='center', cellLoc='center'); table.auto_set_font_size(False); table.set_fontsize(14); table.scale(1, 2.5); plt.tight_layout(rect=[0, 0, 1, 0.97]); plt.show()

print("✅ UnifiedBacktester v2.2 (Scope-Corrected) defined successfully.")

# ============================================================================
# FINAL EXECUTION: CORRECTED BASELINE (P0 & P1 Fixes)
# This cell is self-contained to prevent scope errors.
# ============================================================================

# --- 1. Define All Necessary Variables ---

# Visualization Palette
PALETTE = {
    'primary': '#16A085', 'secondary': '#34495E', 'positive': '#27AE60',
    'negative': '#C0392B', 'highlight_1': '#2980B9', 'highlight_2': '#E67E22',
    'neutral': '#7F8C8D', 'grid': '#BDC3C7', 'text': '#2C3E50'
}

# Strategy Configurations
BASE_CONFIG = {
    "backtest_start_date": "2016-03-01",
    "backtest_end_date": "2025-07-28",
    "rebalance_frequency": "Q",
    "strategy_version_db": "qvm_v2.0_enhanced",
    "transaction_cost_bps": 30,
}
ALL_CONFIGS = {
    "A_Standalone_Value": {
        **BASE_CONFIG,
        "strategy_name": "A_Standalone_Value",
        "factors_to_combine": {'Value_Composite': 1.0},
        "construction_method": "hybrid", # Use hybrid method
        "portfolio_size_small_universe": 20,
        "selection_percentile": 0.8
    }
}

# --- 2. Load All Data ---
# This function needs to be defined or accessible in this scope.
# Assuming it's in a cell above. If not, we should redefine it here.
factor_data_raw, daily_returns_matrix, benchmark_returns, engine = load_all_data(BASE_CONFIG)


# --- 3. Run the Backtest ---
print("\n" + "="*80)
print("🚀 RUNNING CORRECTED BASELINE: Standalone Value with P0 & P1 Fixes")
print("="*80)

# Use the 'A_Standalone_Value' config
corrected_baseline_config = ALL_CONFIGS['A_Standalone_Value']

# Instantiate the v2.2 backtester, passing the now-defined data and PALETTE
# (Ensure the UnifiedBacktester_v2_2 class is defined in a cell above)
corrected_baseline_backtester = UnifiedBacktester_v2_2(
    config=corrected_baseline_config,
    factor_data=factor_data_raw,
    returns_matrix=daily_returns_matrix,
    benchmark_returns=benchmark_returns,
    db_engine=engine,
    palette=PALETTE
)

# Run the backtest and store the results
corrected_baseline_returns = corrected_baseline_backtester.run()


📂 Loading all raw data...
✅ All data loaded and prepared.

================================================================================
🚀 RUNNING CORRECTED BASELINE: Standalone Value with P0 & P1 Fixes
================================================================================
✅ UnifiedBacktester v2.2 initialized for strategy: 'A_Standalone_Value'
   - Data sliced to period: 2016-03-01 to 2025-07-28

--- Executing Backtest for: A_Standalone_Value ---
   - Processing 2016-03-31... Constructing liquid universe for 2016-03-31...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 554 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/12...
  Step 3: Filtering and ranking...
    Total batch results: 554
    Sample result: ('AAA', 41, 6.3474130902439025, 761.4546934536584)
    Before filters: 554 stocks
    Trading days range: 1-41 (need >= 37)
    ADTV range: 0.000-192.100B VND (need >= 10.0)
    Stocks passing trading days filter: 318
    Stocks passing ADTV filter: 63
    After filters: 62 stocks
✅ Universe constructed: 62 stocks
  ADTV range: 10.0B - 192.1B VND
  Market cap range: 199.2B - 156193.8B VND
  Adding sector information...

     - Rebalance 2016-03-31: Universe Size=59, Method='fixed_n', Portfolio Size=20
✅ Formed portfolio.
   - Processing 2016-06-30... Constructing liquid universe for 2016-06-30...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 566 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/12...
  Step 3: Filtering and ranking...
    Total batch results: 566
    Sample result: ('AAA', 44, 16.14919344318182, 1263.9637964181818)
    Before filters: 566 stocks
    Trading days range: 1-44 (need >= 37)
    ADTV range: 0.001-151.061B VND (need >= 10.0)
    Stocks passing trading days filter: 376
    Stocks passing ADTV filter: 69
    After filters: 69 stocks
✅ Universe constructed: 69 stocks
  ADTV range: 10.0B - 151.1B VND
  Market cap range: 286.9B - 168942.5B VND
  Adding sector information...
✅ Formed portfolio.
   - Processing 2016-09-30... Constructing liquid universe for 2016-09-30...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 564 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/12...
  Step 3: Filtering and ranking...
    Total batch results: 564
    Sample result: ('AAA', 45, 11.573230091111112, 1681.9056111200007)
    Before filters: 564 stocks
    Trading days range: 1-45 (need >= 37)
    ADTV range: 0.000-344.721B VND (need >= 10.0)
    Stocks passing trading days filter: 391
    Stocks passing ADTV filter: 63
    After filters: 63 stocks
✅ Universe constructed: 63 stocks
  ADTV range: 10.0B - 344.7B VND
  Market cap range: 340.7B - 204587.8B VND
  Adding sector information...
✅ Formed portfolio.
   - Processing 2016-12-30... Constructing liquid universe for 2016-12-30...
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
✅ Formed portfolio.
   - Processing 2017-03-31... Constructing liquid universe for 2017-03-31...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 590 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/12...
  Step 3: Filtering and ranking...
    Total batch results: 590
    Sample result: ('AAA', 42, 28.265655714285717, 1415.9868445714285)
    Before filters: 590 stocks
    Trading days range: 1-42 (need >= 37)
    ADTV range: 0.001-189.960B VND (need >= 10.0)
    Stocks passing trading days filter: 390
    Stocks passing ADTV filter: 75
    After filters: 74 stocks
✅ Universe constructed: 74 stocks
  ADTV range: 10.4B - 190.0B VND
  Market cap range: 318.5B - 194113.2B VND
  Adding sector information...

     - Rebalance 2017-03-31: Universe Size=72, Method='fixed_n', Portfolio Size=20
✅ Formed portfolio.
   - Processing 2017-06-30... Constructing liquid universe for 2017-06-30...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 602 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/13...
  Step 3: Filtering and ranking...
    Total batch results: 602
    Sample result: ('AAA', 44, 71.05920019886365, 1824.7639174136364)
    Before filters: 602 stocks
    Trading days range: 1-44 (need >= 37)
    ADTV range: 0.000-165.398B VND (need >= 10.0)
    Stocks passing trading days filter: 408
    Stocks passing ADTV filter: 91
    After filters: 89 stocks
✅ Universe constructed: 89 stocks
  ADTV range: 10.1B - 165.4B VND
  Market cap range: 224.2B - 218651.7B VND
  Adding sector information...
✅ Formed portfolio.
   - Processing 2017-09-29... Constructing liquid universe for 2017-09-29...
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
✅ Formed portfolio.
   - Processing 2017-12-29... Constructing liquid universe for 2017-12-29...
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
✅ Formed portfolio.
   - Processing 2018-03-30... Constructing liquid universe for 2018-03-30...
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
✅ Formed portfolio.
   - Processing 2018-06-29... Constructing liquid universe for 2018-06-29...
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

     - Rebalance 2018-06-29: Universe Size=74, Method='fixed_n', Portfolio Size=20
✅ Formed portfolio.
   - Processing 2018-09-28... Constructing liquid universe for 2018-09-28...
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
✅ Formed portfolio.
   - Processing 2018-12-28... Constructing liquid universe for 2018-12-28...
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
✅ Formed portfolio.
   - Processing 2019-03-29... Constructing liquid universe for 2019-03-29...
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
✅ Formed portfolio.
   - Processing 2019-06-28... Constructing liquid universe for 2019-06-28...
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
✅ Formed portfolio.
   - Processing 2019-09-30... Constructing liquid universe for 2019-09-30...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 667 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 667
    Sample result: ('AAA', 45, 36.296758077777795, 2843.8218235555546)
    Before filters: 667 stocks
    Trading days range: 1-45 (need >= 37)
    ADTV range: 0.000-164.927B VND (need >= 10.0)
    Stocks passing trading days filter: 426
    Stocks passing ADTV filter: 87
    After filters: 86 stocks
✅ Universe constructed: 86 stocks
  ADTV range: 10.9B - 164.9B VND
  Market cap range: 787.8B - 406709.6B VND
  Adding sector information...

     - Rebalance 2019-09-30: Universe Size=84, Method='fixed_n', Portfolio Size=20
✅ Formed portfolio.
   - Processing 2019-12-31... Constructing liquid universe for 2019-12-31...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 666 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 666
    Sample result: ('AAA', 46, 35.48351934782609, 2454.1144385739126)
    Before filters: 666 stocks
    Trading days range: 1-46 (need >= 37)
    ADTV range: 0.000-236.047B VND (need >= 10.0)
    Stocks passing trading days filter: 405
    Stocks passing ADTV filter: 83
    After filters: 81 stocks
✅ Universe constructed: 81 stocks
  ADTV range: 10.2B - 236.0B VND
  Market cap range: 342.0B - 393084.8B VND
  Adding sector information...
✅ Formed portfolio.
   - Processing 2020-03-31... Constructing liquid universe for 2020-03-31...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 675 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 675
    Sample result: ('AAA', 44, 24.098066386363634, 1979.3051770727272)
    Before filters: 675 stocks
    Trading days range: 1-44 (need >= 37)
    ADTV range: 0.000-213.031B VND (need >= 10.0)
    Stocks passing trading days filter: 404
    Stocks passing ADTV filter: 80
    After filters: 79 stocks
✅ Universe constructed: 79 stocks
  ADTV range: 10.0B - 213.0B VND
  Market cap range: 533.7B - 340995.1B VND
  Adding sector information...
✅ Formed portfolio.
   - Processing 2020-06-30... Constructing liquid universe for 2020-06-30...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 677 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 677
    Sample result: ('AAA', 44, 30.761108318181822, 2165.0960601181823)
    Before filters: 677 stocks
    Trading days range: 1-44 (need >= 37)
    ADTV range: 0.000-685.229B VND (need >= 10.0)
    Stocks passing trading days filter: 454
    Stocks passing ADTV filter: 118
    After filters: 114 stocks
✅ Universe constructed: 114 stocks
  ADTV range: 10.1B - 685.2B VND
  Market cap range: 296.2B - 320862.0B VND
  Adding sector information...
✅ Formed portfolio.
   - Processing 2020-09-30... Constructing liquid universe for 2020-09-30...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 685 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 685
    Sample result: ('AAA', 45, 32.046618433333336, 2558.80504256)
    Before filters: 685 stocks
    Trading days range: 1-45 (need >= 37)
    ADTV range: 0.000-328.211B VND (need >= 10.0)
    Stocks passing trading days filter: 469
    Stocks passing ADTV filter: 121
    After filters: 118 stocks
✅ Universe constructed: 118 stocks
  ADTV range: 10.3B - 328.2B VND
  Market cap range: 231.7B - 307095.1B VND
  Adding sector information...
✅ Formed portfolio.
   - Processing 2020-12-31... Constructing liquid universe for 2020-12-31...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 696 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 696
    Sample result: ('AAA', 46, 34.23234689130436, 2772.9638488000005)
    Before filters: 696 stocks
    Trading days range: 1-46 (need >= 37)
    ADTV range: 0.000-798.382B VND (need >= 10.0)
    Stocks passing trading days filter: 500
    Stocks passing ADTV filter: 154
    After filters: 150 stocks
✅ Universe constructed: 150 stocks
  ADTV range: 10.0B - 798.4B VND
  Market cap range: 349.5B - 356853.8B VND
  Adding sector information...

     - Rebalance 2020-12-31: Universe Size=146, Method='percentile', Portfolio Size=30
✅ Formed portfolio.
   - Processing 2021-03-31... Constructing liquid universe for 2021-03-31...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 707 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 707
    Sample result: ('AAA', 41, 49.81973512195122, 3289.3494680024396)
    Before filters: 707 stocks
    Trading days range: 1-41 (need >= 37)
    ADTV range: 0.001-992.963B VND (need >= 10.0)
    Stocks passing trading days filter: 509
    Stocks passing ADTV filter: 170
    After filters: 168 stocks
✅ Universe constructed: 168 stocks
  ADTV range: 10.0B - 993.0B VND
  Market cap range: 249.2B - 361796.3B VND
  Adding sector information...
✅ Formed portfolio.
   - Processing 2021-06-30... Constructing liquid universe for 2021-06-30...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 710 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 710
    Sample result: ('AAA', 44, 147.30311397727272, 4315.728932006818)
    Before filters: 710 stocks
    Trading days range: 1-44 (need >= 37)
    ADTV range: 0.001-2228.627B VND (need >= 10.0)
    Stocks passing trading days filter: 551
    Stocks passing ADTV filter: 187
    After filters: 185 stocks
✅ Universe constructed: 185 stocks
  ADTV range: 10.0B - 2228.6B VND
  Market cap range: 406.1B - 413763.5B VND
  Adding sector information...
✅ Formed portfolio.
   - Processing 2021-09-30... Constructing liquid universe for 2021-09-30...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 715 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 715
    Sample result: ('AAA', 44, 111.40748261363638, 5160.277457379547)
    Before filters: 715 stocks
    Trading days range: 1-44 (need >= 37)
    ADTV range: 0.000-1363.272B VND (need >= 10.0)
    Stocks passing trading days filter: 574
    Stocks passing ADTV filter: 234
    After filters: 234 stocks
✅ Universe constructed: 200 stocks
  ADTV range: 14.9B - 1363.3B VND
  Market cap range: 212.2B - 366757.4B VND
  Adding sector information...
✅ Formed portfolio.
   - Processing 2021-12-31... Constructing liquid universe for 2021-12-31...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 719 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 719
    Sample result: ('AAA', 45, 150.17773177777775, 5901.572982684445)
    Before filters: 719 stocks
    Trading days range: 1-46 (need >= 37)
    ADTV range: 0.003-1259.464B VND (need >= 10.0)
    Stocks passing trading days filter: 623
    Stocks passing ADTV filter: 279
    After filters: 276 stocks
✅ Universe constructed: 200 stocks
  ADTV range: 23.5B - 1259.5B VND
  Market cap range: 452.5B - 375185.9B VND
  Adding sector information...
✅ Formed portfolio.
   - Processing 2022-03-31... Constructing liquid universe for 2022-03-31...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 718 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 718
    Sample result: ('AAA', 41, 101.89924853658535, 5849.945022829269)
    Before filters: 718 stocks
    Trading days range: 2-41 (need >= 37)
    ADTV range: 0.001-1118.662B VND (need >= 10.0)
    Stocks passing trading days filter: 578
    Stocks passing ADTV filter: 257
    After filters: 256 stocks
✅ Universe constructed: 200 stocks
  ADTV range: 16.9B - 1118.7B VND
  Market cap range: 394.2B - 404964.9B VND
  Adding sector information...

     - Rebalance 2022-03-31: Universe Size=192, Method='percentile', Portfolio Size=39
✅ Formed portfolio.
   - Processing 2022-06-30... Constructing liquid universe for 2022-06-30...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 720 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 720
    Sample result: ('AAA', 44, 48.95811068181819, 3962.8405917818177)
    Before filters: 720 stocks
    Trading days range: 1-44 (need >= 37)
    ADTV range: 0.000-725.799B VND (need >= 10.0)
    Stocks passing trading days filter: 556
    Stocks passing ADTV filter: 180
    After filters: 179 stocks
✅ Universe constructed: 179 stocks
  ADTV range: 10.0B - 725.8B VND
  Market cap range: 464.4B - 366243.0B VND
  Adding sector information...
✅ Formed portfolio.
   - Processing 2022-09-30... Constructing liquid universe for 2022-09-30...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 722 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 722
    Sample result: ('AAA', 44, 45.33385690386364, 4486.8600162327275)
    Before filters: 722 stocks
    Trading days range: 1-44 (need >= 37)
    ADTV range: 0.000-598.923B VND (need >= 10.0)
    Stocks passing trading days filter: 542
    Stocks passing ADTV filter: 183
    After filters: 182 stocks
✅ Universe constructed: 182 stocks
  ADTV range: 10.0B - 598.9B VND
  Market cap range: 273.3B - 377203.1B VND
  Adding sector information...
✅ Formed portfolio.
   - Processing 2022-12-30... Constructing liquid universe for 2022-12-30...
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
✅ Formed portfolio.
   - Processing 2023-03-31... Constructing liquid universe for 2023-03-31...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 713 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 713
    Sample result: ('AAA', 46, 30.509317390000007, 3319.804688306087)
    Before filters: 713 stocks
    Trading days range: 1-46 (need >= 37)
    ADTV range: 0.000-503.301B VND (need >= 10.0)
    Stocks passing trading days filter: 527
    Stocks passing ADTV filter: 137
    After filters: 136 stocks
✅ Universe constructed: 136 stocks
  ADTV range: 10.4B - 503.3B VND
  Market cap range: 402.8B - 434815.4B VND
  Adding sector information...
✅ Formed portfolio.
   - Processing 2023-06-30... Constructing liquid universe for 2023-06-30...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 717 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 717
    Sample result: ('AAA', 43, 67.13501987906976, 4226.3557069395365)
    Before filters: 717 stocks
    Trading days range: 1-43 (need >= 37)
    ADTV range: 0.000-543.053B VND (need >= 10.0)
    Stocks passing trading days filter: 536
    Stocks passing ADTV filter: 188
    After filters: 186 stocks
✅ Universe constructed: 186 stocks
  ADTV range: 10.1B - 543.1B VND
  Market cap range: 376.1B - 456115.5B VND
  Adding sector information...

     - Rebalance 2023-06-30: Universe Size=184, Method='percentile', Portfolio Size=37
✅ Formed portfolio.
   - Processing 2023-09-29... Constructing liquid universe for 2023-09-29...
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
✅ Formed portfolio.
   - Processing 2023-12-29... Constructing liquid universe for 2023-12-29...
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
✅ Formed portfolio.
   - Processing 2024-03-29... Constructing liquid universe for 2024-03-29...
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
✅ Formed portfolio.
   - Processing 2024-06-28... Constructing liquid universe for 2024-06-28...
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
✅ Formed portfolio.
   - Processing 2024-09-30... Constructing liquid universe for 2024-09-30...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 707 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 707
    Sample result: ('AAA', 44, 50.20643357272727, 3941.163173192728)
    Before filters: 707 stocks
    Trading days range: 1-44 (need >= 37)
    ADTV range: 0.000-590.433B VND (need >= 10.0)
    Stocks passing trading days filter: 524
    Stocks passing ADTV filter: 156
    After filters: 154 stocks
✅ Universe constructed: 154 stocks
  ADTV range: 10.2B - 590.4B VND
  Market cap range: 400.2B - 502891.2B VND
  Adding sector information...

     - Rebalance 2024-09-30: Universe Size=152, Method='percentile', Portfolio Size=31
✅ Formed portfolio.
   - Processing 2024-12-31... Constructing liquid universe for 2024-12-31...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 702 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/15...
  Step 3: Filtering and ranking...
    Total batch results: 702
    Sample result: ('AAA', 46, 13.83696037804348, 3289.0565223234785)
    Before filters: 702 stocks
    Trading days range: 1-46 (need >= 37)
    ADTV range: 0.000-765.066B VND (need >= 10.0)
    Stocks passing trading days filter: 534
    Stocks passing ADTV filter: 157
    After filters: 155 stocks
✅ Universe constructed: 155 stocks
  ADTV range: 10.3B - 765.1B VND
  Market cap range: 473.4B - 517124.6B VND
  Adding sector information...
✅ Formed portfolio.
   - Processing 2025-03-31... Constructing liquid universe for 2025-03-31...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 699 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 699
    Sample result: ('AAA', 41, 15.314483515853656, 3317.6764368702447)
    Before filters: 699 stocks
    Trading days range: 1-41 (need >= 37)
    ADTV range: 0.000-822.524B VND (need >= 10.0)
    Stocks passing trading days filter: 510
    Stocks passing ADTV filter: 164
    After filters: 163 stocks
✅ Universe constructed: 163 stocks
  ADTV range: 10.0B - 822.5B VND
  Market cap range: 319.9B - 530251.8B VND
  Adding sector information...
✅ Formed portfolio.
   - Processing 2025-06-30... Constructing liquid universe for 2025-06-30...
  Lookback: 63 days
  ADTV threshold: 10.0B VND
  Target size: 200 stocks
  Step 1: Loading ticker list...
    Found 697 active tickers
  Step 2: Calculating ADTV in batches...
    Processing batch 10/14...
  Step 3: Filtering and ranking...
    Total batch results: 697
    Sample result: ('AAA', 43, 13.422274973023258, 2760.821970530232)
    Before filters: 697 stocks
    Trading days range: 1-43 (need >= 37)
    ADTV range: 0.000-908.189B VND (need >= 10.0)
    Stocks passing trading days filter: 528
    Stocks passing ADTV filter: 164
    After filters: 164 stocks
✅ Universe constructed: 164 stocks
  ADTV range: 10.4B - 908.2B VND
  Market cap range: 439.6B - 474582.9B VND
  Adding sector information...
✅ Formed portfolio.

✅ Backtest for A_Standalone_Value complete.

# ============================================================================
# FINAL EXECUTION: THE FULL FIVE-STRATEGY BAKE-OFF (DEFINITIVE VERSION)
# This cell is fully self-contained to prevent all scope errors.
# ============================================================================

# --- 1. Define All Configurations for the Bake-Off ---
# This ensures all CONFIG variables are in the local scope of this cell.
BASE_CONFIG = {
    "backtest_start_date": "2016-03-01",
    "backtest_end_date": "2025-07-28",
    "rebalance_frequency": "Q",
    "strategy_version_db": "qvm_v2.0_enhanced",
    "transaction_cost_bps": 30,
    "construction_method": "hybrid",
    "portfolio_size_small_universe": 20,
    "selection_percentile": 0.8
}
CONFIG_A = {**BASE_CONFIG, "strategy_name": "A_Standalone_Value", "factors_to_combine": {'Value_Composite': 1.0}}
CONFIG_B = {**BASE_CONFIG, "strategy_name": "B_Equal_Weighted_QVM", "factors_to_combine": {'Value_Composite': 1/3, 'Quality_Composite': 1/3, 'Momentum_Composite': 1/3}}
CONFIG_C = {**BASE_CONFIG, "strategy_name": "C_Equal_Weighted_QVR", "factors_to_combine": {'Value_Composite': 1/3, 'Quality_Composite': 1/3, 'Momentum_Reversal': 1/3}}
CONFIG_D = {**BASE_CONFIG, "strategy_name": "D_Value_Weighted_QVM", "factors_to_combine": {'Value_Composite': 0.6, 'Quality_Composite': 0.2, 'Momentum_Composite': 0.2}}
CONFIG_E = {**BASE_CONFIG, "strategy_name": "E_Value_Weighted_QVR", "factors_to_combine": {'Value_Composite': 0.6, 'Quality_Composite': 0.2, 'Momentum_Reversal': 0.2}}
ALL_CONFIGS_LIST = [CONFIG_A, CONFIG_B, CONFIG_C, CONFIG_D, CONFIG_E]


# --- 2. Store results for final comparison ---
bake_off_results = {}
all_returns_series = {}

# --- 3. Loop through each configuration and run a full backtest ---
for config in ALL_CONFIGS_LIST:
    print("="*80)
    print(f"🚀 LAUNCHING BAKE-OFF RUN FOR: {config['strategy_name']}")
    print("="*80)
    
    # Instantiate the v2.2 backtester
    # (Ensure the UnifiedBacktester_v2_2 class is defined in a cell above)
    backtester = UnifiedBacktester_v2_2(
        config=config,
        factor_data=factor_data_raw,
        returns_matrix=daily_returns_matrix,
        benchmark_returns=benchmark_returns,
        db_engine=engine,
        palette=PALETTE
    )
    
    net_returns = backtester.run()
    
    # Store the results
    all_returns_series[config['strategy_name']] = net_returns
    bake_off_results[config['strategy_name']] = backtester._calculate_performance_metrics(net_returns)
    
    print(f"\n✅ COMPLETED BAKE-OFF RUN FOR: {config['strategy_name']}")

print("\n" + "="*80)
print("🎉 ALL BAKE-OFF BACKTESTS COMPLETED.")
print("="*80)

# --- 4. Final Summary Comparison ---
summary_df = pd.DataFrame(bake_off_results).T
summary_df.index.name = "Strategy"

print("\n\n--- FINAL BAKE-OFF PERFORMANCE SUMMARY ---")
display(summary_df[['Annual Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 'Information Ratio', 'Alpha (%)']].round(2))

# --- 5. Visualize the Equity Curves for a clear comparison ---
fig, ax = plt.subplots(figsize=(14, 8))

# Use a predefined color cycle for clarity
color_cycle = [PALETTE['primary'], PALETTE['highlight_2'], PALETTE['negative'], '#9B59B6', '#3498DB'] # Teal, Orange, Red, Purple, Blue

for i, (name, returns) in enumerate(all_returns_series.items()):
    first_trade = returns.loc[returns != 0].index.min()
    (1 + returns.loc[first_trade:]).cumprod().plot(ax=ax, label=name, color=color_cycle[i % len(color_cycle)], linewidth=2)

(1 + benchmark_returns.loc[first_trade:]).cumprod().plot(ax=ax, label='VN-Index', color=PALETTE['secondary'], linestyle=':', linewidth=2.5)

ax.set_title('Bake-Off: Cumulative Performance of All Strategies (Log Scale)', fontweight='bold')
ax.set_ylabel('Growth of $1 (Log Scale)')
ax.set_yscale('log')
ax.legend()
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.show()

================================================================================
🎉 ALL BAKE-OFF BACKTESTS COMPLETED.
================================================================================


--- FINAL BAKE-OFF PERFORMANCE SUMMARY ---
Annual Return (%)	Sharpe Ratio	Max Drawdown (%)	Information Ratio	Alpha (%)
Strategy					
A_Standalone_Value	15.03	0.58	-66.90	0.32	2.62
B_Equal_Weighted_QVM	11.45	0.45	-68.65	0.10	-0.77
C_Equal_Weighted_QVR	12.74	0.52	-65.41	0.18	0.65
D_Value_Weighted_QVM	12.41	0.48	-66.85	0.17	0.17
E_Value_Weighted_QVR	13.51	0.53	-66.60	0.24	1.12
