# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Lag Sensitivity Analysis (Standalone)
# 
# This notebook tests the sensitivity of the strategy to the lag period for fundamentals (45, 30, 60, 75, 90 days) using the full backtest period.
# 
# ---

# %%
# --- Imports & Setup ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# TODO: Import your QVM Engine or backtest function here
# from qvm_engine_v3 import QVMEngineV3AdoptedInsights

# --- Load Data & Config ---
# TODO: Load your price_data, fundamental_data, returns_matrix, benchmark_returns, config, engine, etc.

LAG_PERIODS = [45, 30, 60, 75, 90]  # days

# %% [markdown]
# ## Run Backtest for Each Lag

# %%
lag_results = {}
for lag in LAG_PERIODS:
    print(f"\n=== Running backtest for lag: {lag} days ===")
    config_run = config.copy()
    config_run['factors'] = config['factors'].copy()
    config_run['factors']['fundamental_lag_days'] = lag
    # TODO: Run your QVM Engine or backtest here
    # net_returns, diagnostics = engine.run_backtest()
    net_returns = pd.Series(dtype='float64')  # TODO: Replace
    diagnostics = pd.DataFrame()  # TODO: Replace
    lag_results[lag] = {'net_returns': net_returns, 'diagnostics': diagnostics}

# %% [markdown]
# ## Plot Results

# %%
plt.figure(figsize=(12,6))
for lag, res in lag_results.items():
    net = res['net_returns']
    if not net.empty:
        cum = (1 + net).cumprod()
        plt.plot(cum, label=f'Lag {lag}d')
plt.title('Cumulative Returns by Lag Period')
plt.legend()
plt.show()

# %% [markdown]
# ## Performance Metrics Comparison

# %%
# Calculate performance metrics for each lag
performance_summary = {}
for lag, res in lag_results.items():
    net = res['net_returns']
    if not net.empty:
        # Basic metrics
        total_return = (1 + net).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(net)) - 1
        volatility = net.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Drawdown
        cumulative = (1 + net).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        performance_summary[lag] = {
            'Total Return': total_return,
            'Annualized Return': annualized_return,
            'Volatility': volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown
        }

# Display results
if performance_summary:
    summary_df = pd.DataFrame(performance_summary).T
    print("Performance Summary by Lag Period:")
    print(summary_df.round(4))

# %% [markdown]
# ## Sensitivity Analysis Summary
# 
# - Summarize performance metrics (return, Sharpe, drawdown, etc.) for each lag
# - Discuss how lag period affects strategy performance
# - Add your own analysis and conclusions here. 