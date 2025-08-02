---
# Walkforward Validation (2016 Onwards)

This notebook implements a walkforward (rolling out-of-sample) validation for the QVM Engine v3 (or any strategy), starting from 2016.

It also performs a sensitivity analysis on the lag period for fundamentals (45, 30, 60, 75, 90 days).

---

```python
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
```

---
## Walkforward Window Generator
---

```python
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
```

---
## Walkforward Backtest Function (Template)
---

```python
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
```

---
## Run Walkforward for All Lag Periods
---

```python
all_results = {}
for lag in LAG_PERIODS:
    print(f"\n=== Running walkforward for lag: {lag} days ===")
    # TODO: Pass your actual data and config
    results = run_walkforward_for_lag(lag, walk_windows, config, price_data, fundamental_data, returns_matrix, benchmark_returns, engine)
    all_results[lag] = results
```

---
## Aggregate and Plot Results
---

```python
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
```

---
## Sensitivity Analysis Summary
---

- Summarize performance metrics (return, Sharpe, drawdown, etc.) for each lag
- Discuss how lag period affects strategy performance
- Add your own analysis and conclusions here.


---
# Lag Sensitivity Analysis (Standalone)

This notebook tests the sensitivity of the strategy to the lag period for fundamentals (45, 30, 60, 75, 90 days) using the full backtest period.

---

```python
# --- Imports & Setup ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# TODO: Import your QVM Engine or backtest function here
# from qvm_engine_v3 import QVMEngineV3AdoptedInsights

# --- Load Data & Config ---
# TODO: Load your price_data, fundamental_data, returns_matrix, benchmark_returns, config, engine, etc.

LAG_PERIODS = [45, 30, 60, 75, 90]  # days
```

---
## Run Backtest for Each Lag
---

```python
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
```

---
## Plot Results
---

```python
plt.figure(figsize=(12,6))
for lag, res in lag_results.items():
    net = res['net_returns']
    if not net.empty:
        cum = (1 + net).cumprod()
        plt.plot(cum, label=f'Lag {lag}d')
plt.title('Cumulative Returns by Lag Period')
plt.legend()
plt.show()
```

---
## Sensitivity Analysis Summary
---

- Summarize performance metrics (return, Sharpe, drawdown, etc.) for each lag
- Discuss how lag period affects strategy performance
- Add your own analysis and conclusions here.


---
# Min ADTV (10B VND) Adjustment (Clone of 28_qvm_engine_v3c.ipynb)

**Instructions:**
- Clone your `28_qvm_engine_v3c.ipynb` notebook.
- Change the ADTV threshold from shares to VND in the config and SQL logic.

---

```python
# --- In your config ---
"universe": {
    "lookback_days": 60,
    "adtv_threshold_vnd": 10_000_000_000,  # 10 billion VND
    "min_market_cap_bn": 1.0,
    "target_portfolio_size": 25
}

# --- In your SQL query for universe selection ---
SELECT 
    ticker,
    AVG(total_volume * close_price_adjusted) as avg_adtv_vnd,
    AVG(market_cap) as avg_market_cap
FROM vcsc_daily_data_complete
WHERE trading_date <= :analysis_date
  AND trading_date >= DATE_SUB(:analysis_date, INTERVAL :lookback_days DAY)
GROUP BY ticker
HAVING avg_adtv_vnd >= :adtv_threshold AND avg_market_cap >= :min_market_cap

# --- In your Python code ---
adtv_threshold = self.config['universe']['adtv_threshold_vnd']
```

---
# Composite vs. Single-Factor Comparison

This notebook compares the composite QVM strategy to standalone Quality, Value, and Momentum factor strategies.

---

```python
# --- Imports & Setup ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# TODO: Import your QVM Engine and single-factor engines here
# from qvm_engine_v3 import QVMEngineV3AdoptedInsights
# from single_factors import QualityFactorEngine, ValueFactorEngine, MomentumFactorEngine

# --- Load Data & Config ---
# TODO: Load your price_data, fundamental_data, returns_matrix, benchmark_returns, config, engine, etc.

# --- Run Composite Strategy ---
# qvm_engine = QVMEngineV3AdoptedInsights(config, price_data, fundamental_data, returns_matrix, benchmark_returns, engine)
# qvm_returns, qvm_diag = qvm_engine.run_backtest()
qvm_returns = pd.Series(dtype='float64')  # TODO: Replace

# --- Run Single-Factor Strategies ---
# quality_engine = QualityFactorEngine(...)
# value_engine = ValueFactorEngine(...)
# momentum_engine = MomentumFactorEngine(...)
# quality_returns = quality_engine.run_backtest()
# value_returns = value_engine.run_backtest()
# momentum_returns = momentum_engine.run_backtest()
quality_returns = pd.Series(dtype='float64')  # TODO: Replace
value_returns = pd.Series(dtype='float64')    # TODO: Replace
momentum_returns = pd.Series(dtype='float64') # TODO: Replace

# --- Plot Comparison ---
plt.figure(figsize=(12,6))
for label, returns in zip(['QVM Composite', 'Quality', 'Value', 'Momentum'], [qvm_returns, quality_returns, value_returns, momentum_returns]):
    if not returns.empty:
        cum = (1 + returns).cumprod()
        plt.plot(cum, label=label)
plt.title('Composite vs. Single-Factor Cumulative Returns')
plt.legend()
plt.show()
```

---
## Analysis
---

- Compare performance metrics (return, Sharpe, drawdown, etc.) for each strategy
- Discuss which factor(s) drive performance
- Add your own analysis and conclusions here.

---

# Instructions for Conversion

- Save this file as `notebook_templates.md`.
- Use [Jupytext](https://jupytext.readthedocs.io/en/latest/) or VSCode/JupyterLab to convert each section to a `.ipynb` notebook.
- Each section above can be split into a separate notebook as needed.
- Fill in the TODOs with your actual data and engine code.