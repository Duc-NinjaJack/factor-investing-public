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
# # Composite vs. Single-Factor Comparison
# 
# This notebook compares the composite QVM strategy to standalone Quality, Value, and Momentum factor strategies.
# 
# **Purpose:**
# - Validate the composite strategy against individual factors
# - Identify which factor(s) drive performance
# - Detect potential overfitting in the composite approach
# 
# ---

# %%
# --- Imports & Setup ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# TODO: Import your QVM Engine and single-factor engines here
# from qvm_engine_v3 import QVMEngineV3AdoptedInsights
# from single_factors import QualityFactorEngine, ValueFactorEngine, MomentumFactorEngine

# --- Load Data & Config ---
# TODO: Load your price_data, fundamental_data, returns_matrix, benchmark_returns, config, engine, etc.

# %%
# --- Run Composite Strategy ---
print("Running QVM Composite Strategy...")
# qvm_engine = QVMEngineV3AdoptedInsights(config, price_data, fundamental_data, returns_matrix, benchmark_returns, engine)
# qvm_returns, qvm_diag = qvm_engine.run_backtest()
qvm_returns = pd.Series(dtype='float64')  # TODO: Replace
qvm_diag = pd.DataFrame()  # TODO: Replace

# %%
# --- Run Single-Factor Strategies ---
print("Running Single-Factor Strategies...")

# Quality Factor (ROAA-based)
print("  - Quality Factor (ROAA)...")
# quality_engine = QualityFactorEngine(config, price_data, fundamental_data, returns_matrix, benchmark_returns, engine)
# quality_returns, quality_diag = quality_engine.run_backtest()
quality_returns = pd.Series(dtype='float64')  # TODO: Replace
quality_diag = pd.DataFrame()  # TODO: Replace

# Value Factor (P/E-based)
print("  - Value Factor (P/E)...")
# value_engine = ValueFactorEngine(config, price_data, fundamental_data, returns_matrix, benchmark_returns, engine)
# value_returns, value_diag = value_engine.run_backtest()
value_returns = pd.Series(dtype='float64')  # TODO: Replace
value_diag = pd.DataFrame()  # TODO: Replace

# Momentum Factor (Multi-horizon)
print("  - Momentum Factor (Multi-horizon)...")
# momentum_engine = MomentumFactorEngine(config, price_data, fundamental_data, returns_matrix, benchmark_returns, engine)
# momentum_returns, momentum_diag = momentum_engine.run_backtest()
momentum_returns = pd.Series(dtype='float64')  # TODO: Replace
momentum_diag = pd.DataFrame()  # TODO: Replace

# %%
# --- Plot Comparison ---
plt.figure(figsize=(12,6))
strategies = {
    'QVM Composite': qvm_returns,
    'Quality': quality_returns,
    'Value': value_returns,
    'Momentum': momentum_returns
}

for label, returns in strategies.items():
    if not returns.empty:
        cum = (1 + returns).cumprod()
        plt.plot(cum, label=label, linewidth=2)

plt.title('Composite vs. Single-Factor Cumulative Returns')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %%
# --- Performance Metrics Comparison ---
def calculate_performance_metrics(returns, name):
    if returns.empty:
        return {}
    
    # Basic metrics
    total_return = (1 + returns).prod() - 1
    annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
    volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
    
    # Drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Information ratio (vs benchmark)
    # excess_returns = returns - benchmark_returns
    # information_ratio = excess_returns.mean() * 252 / (excess_returns.std() * np.sqrt(252))
    
    return {
        'Strategy': name,
        'Total Return': total_return,
        'Annualized Return': annualized_return,
        'Volatility': volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown
        # 'Information Ratio': information_ratio
    }

# Calculate metrics for all strategies
all_metrics = []
for name, returns in strategies.items():
    metrics = calculate_performance_metrics(returns, name)
    if metrics:
        all_metrics.append(metrics)

# Display comparison table
if all_metrics:
    comparison_df = pd.DataFrame(all_metrics).set_index('Strategy')
    print("Performance Comparison:")
    print(comparison_df.round(4))

# %%
# --- Factor Contribution Analysis ---
# Analyze which factors contribute most to the composite performance
print("\nFactor Contribution Analysis:")
print("=" * 50)

# Calculate correlation between composite and individual factors
if not qvm_returns.empty:
    correlations = {}
    for name, returns in strategies.items():
        if name != 'QVM Composite' and not returns.empty:
            # Align dates
            aligned_qvm = qvm_returns.reindex(returns.index).dropna()
            aligned_factor = returns.reindex(aligned_qvm.index).dropna()
            if len(aligned_qvm) > 0:
                corr = aligned_qvm.corr(aligned_factor)
                correlations[name] = corr
    
    if correlations:
        print("Correlation with QVM Composite:")
        for factor, corr in correlations.items():
            print(f"  {factor}: {corr:.4f}")

# %%
# --- Risk-Return Scatter Plot ---
plt.figure(figsize=(10,8))
for name, returns in strategies.items():
    if not returns.empty:
        metrics = calculate_performance_metrics(returns, name)
        if metrics:
            plt.scatter(metrics['Volatility'], metrics['Annualized Return'], 
                       label=name, s=100, alpha=0.7)

plt.xlabel('Annualized Volatility')
plt.ylabel('Annualized Return')
plt.title('Risk-Return Profile: Composite vs. Single Factors')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %%
# --- Rolling Performance Analysis ---
# Compare rolling performance over time
if not qvm_returns.empty:
    window = 252  # 1 year rolling window
    
    plt.figure(figsize=(12,6))
    for name, returns in strategies.items():
        if not returns.empty:
            rolling_sharpe = returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(252)
            plt.plot(rolling_sharpe, label=f'{name} (Rolling Sharpe)', alpha=0.7)
    
    plt.title(f'{window}-Day Rolling Sharpe Ratio')
    plt.xlabel('Date')
    plt.ylabel('Rolling Sharpe Ratio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# %% [markdown]
# ## Analysis Summary
# 
# **Key Questions to Answer:**
# 
# 1. **Factor Dominance**: Which single factor performs best?
# 2. **Composite Benefits**: Does the composite strategy outperform individual factors?
# 3. **Diversification**: Are the factors sufficiently uncorrelated?
# 4. **Overfitting Risk**: Is the composite performance driven by one dominant factor?
# 
# **Expected Outcomes:**
# - If composite significantly outperforms all single factors → Good diversification
# - If composite performance is similar to best single factor → Potential overfitting
# - If factors are highly correlated → Limited diversification benefit
# 
# **Next Steps:**
# - Analyze factor correlations over time
# - Test different factor weights
# - Consider regime-dependent factor allocation 