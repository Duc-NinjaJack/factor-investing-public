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
# # Walkforward Validation (2016 Onwards)
# 
# This notebook implements a walkforward (rolling out-of-sample) validation for the QVM Engine v3 (or any strategy), starting from 2016.
# 
# It also performs a sensitivity analysis on the lag period for fundamentals (45, 30, 60, 75, 90 days).
# 
# ---
# 
# **Key Steps:**
# - Define rolling train/test windows
# - For each lag period, run walkforward backtest
# - Aggregate and compare results
# 
# ---

# %%
# --- Imports & Setup ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

# TODO: Import your QVM Engine or backtest function here
# from qvm_engine_v3 import QVMEngineV3AdoptedInsights

# --- Load Data & Config ---
# TODO: Load your price_data, fundamental_data, returns_matrix, benchmark_returns, config, engine, etc.
# Example:
# price_data, fundamental_data, returns_matrix, benchmark_returns = load_all_data_for_backtest(config, engine)

# Set the full backtest period
BACKTEST_START = pd.Timestamp('2016-01-01')
BACKTEST_END = pd.Timestamp('2025-07-31')

# --- Walkforward Window Parameters ---
TRAIN_YEARS = 3
TEST_YEARS = 1

# --- Lag Periods to Test (in days) ---
LAG_PERIODS = [45, 30, 60, 75, 90]  # days

# %% [markdown]
# ## Walkforward Window Generator

# %%
def generate_walkforward_windows(start, end, train_years=3, test_years=1):
    windows = []
    current_train_start = start
    while True:
        train_end = current_train_start + pd.DateOffset(years=train_years) - timedelta(days=1)
        test_start = train_end + timedelta(days=1)
        test_end = test_start + pd.DateOffset(years=test_years) - timedelta(days=1)
        if test_end > end:
            break
        windows.append({
            'train_start': current_train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end
        })
        current_train_start = current_train_start + pd.DateOffset(years=test_years)
    return windows

walk_windows = generate_walkforward_windows(BACKTEST_START, BACKTEST_END, TRAIN_YEARS, TEST_YEARS)
print(f'Generated {len(walk_windows)} walkforward windows:')
for w in walk_windows:
    print(w)

# %% [markdown]
# ## Walkforward Backtest Function (Template)

# %%
def run_walkforward_for_lag(lag_days, walk_windows, config, price_data, fundamental_data, returns_matrix, benchmark_returns, engine):
    results = []
    for i, window in enumerate(walk_windows):
        print(f"\n=== Walk {i+1}/{len(walk_windows)}: Train {window['train_start'].date()} to {window['train_end'].date()}, Test {window['test_start'].date()} to {window['test_end'].date()} (Lag: {lag_days}d) ===")
        config_run = config.copy()
        config_run['factors'] = config['factors'].copy()
        config_run['factors']['fundamental_lag_days'] = lag_days
        config_run['backtest_start_date'] = window['test_start'].strftime('%Y-%m-%d')
        config_run['backtest_end_date'] = window['test_end'].strftime('%Y-%m-%d')
        # TODO: Slice data for this window if needed
        # TODO: Run your QVM Engine or backtest here
        # Example (replace with your actual call):
        # engine = QVMEngineV3AdoptedInsights(config_run, price_data, fundamental_data, returns_matrix, benchmark_returns, db_engine)
        # net_returns, diagnostics = engine.run_backtest()
        net_returns = pd.Series(dtype='float64')  # TODO: Replace
        diagnostics = pd.DataFrame()  # TODO: Replace
        results.append({
            'window': window,
            'lag_days': lag_days,
            'net_returns': net_returns,
            'diagnostics': diagnostics
        })
    return results

# %% [markdown]
# ## Run Walkforward for All Lag Periods

# %%
all_results = {}
for lag in LAG_PERIODS:
    print(f"\n=== Running walkforward for lag: {lag} days ===")
    # TODO: Pass your actual data and config
    results = run_walkforward_for_lag(lag, walk_windows, config, price_data, fundamental_data, returns_matrix, benchmark_returns, engine)
    all_results[lag] = results

# %% [markdown]
# ## Aggregate and Plot Results

# %%
# TODO: Aggregate net_returns and diagnostics for each lag
plt.figure(figsize=(12,6))
for lag, results in all_results.items():
    all_net = pd.concat([r['net_returns'] for r in results])
    if not all_net.empty:
        all_net = all_net.sort_index()
        cum = (1 + all_net).cumprod()
        plt.plot(cum, label=f'Lag {lag}d')
plt.title('Walkforward Cumulative Returns by Lag Period')
plt.legend()
plt.show()

# %% [markdown]
# ## Sensitivity Analysis Summary
# 
# - Summarize performance metrics (return, Sharpe, drawdown, etc.) for each lag
# - Discuss how lag period affects strategy performance
# - Add your own analysis and conclusions here. 