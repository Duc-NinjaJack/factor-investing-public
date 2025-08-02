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
# # Min ADTV (10B VND) Adjustment
# 
# **Instructions:**
# - Clone your `28_qvm_engine_v3c.ipynb` notebook.
# - Change the ADTV threshold from shares to VND in the config and SQL logic.
# 
# **Key Changes:**
# 1. Update config: `adtv_threshold_vnd: 10_000_000_000` (10 billion VND)
# 2. Update SQL query: Use `total_volume * close_price_adjusted` instead of just `total_volume`
# 3. Update Python code: Use the new config key
# 
# ---

# %%
# --- Configuration Changes ---
# 
# OLD CONFIG (incorrect):
# "universe": {
#     "lookback_days": 60,
#     "adtv_threshold_shares": 1000000,  # 1 million shares
#     "min_market_cap_bn": 1.0,
#     "target_portfolio_size": 25
# }
# 
# NEW CONFIG (correct):
# "universe": {
#     "lookback_days": 60,
#     "adtv_threshold_vnd": 10_000_000_000,  # 10 billion VND
#     "min_market_cap_bn": 1.0,
#     "target_portfolio_size": 25
# }

# %%
# --- SQL Query Changes ---
# 
# OLD QUERY (incorrect):
# SELECT 
#     ticker,
#     AVG(total_volume) as avg_volume,
#     AVG(market_cap) as avg_market_cap
# FROM vcsc_daily_data_complete
# WHERE trading_date <= :analysis_date
#   AND trading_date >= DATE_SUB(:analysis_date, INTERVAL :lookback_days DAY)
# GROUP BY ticker
# HAVING avg_volume >= :adtv_threshold AND avg_market_cap >= :min_market_cap
# 
# NEW QUERY (correct):
# SELECT 
#     ticker,
#     AVG(total_volume * close_price_adjusted) as avg_adtv_vnd,
#     AVG(market_cap) as avg_market_cap
# FROM vcsc_daily_data_complete
# WHERE trading_date <= :analysis_date
#   AND trading_date >= DATE_SUB(:analysis_date, INTERVAL :lookback_days DAY)
# GROUP BY ticker
# HAVING avg_adtv_vnd >= :adtv_threshold AND avg_market_cap >= :min_market_cap

# %%
# --- Python Code Changes ---
# 
# OLD CODE (incorrect):
# adtv_threshold = self.config['universe']['adtv_threshold_shares']  # Already in shares
# 
# NEW CODE (correct):
# adtv_threshold = self.config['universe']['adtv_threshold_vnd']  # Now in VND

# %%
# --- Implementation Steps ---
# 
# 1. Copy your 28_qvm_engine_v3c.ipynb to a new file
# 2. Update the config section with the new ADTV threshold
# 3. Update the _get_universe method SQL query
# 4. Update the Python code to use the new config key
# 5. Test the changes

# %% [markdown]
# ## Benefits of VND-based ADTV
# 
# 1. **Proper Liquidity Measurement**: ADTV in VND measures actual trading value, not just share count
# 2. **Price-Aware Filtering**: Accounts for both volume and price, filtering out low-value high-volume stocks
# 3. **Market Standard**: Most institutional investors use currency-based liquidity measures
# 4. **Better Risk Management**: Prevents investing in stocks with low trading value despite high share volume
# 5. **Consistent with Market Cap**: Both ADTV and market cap are in currency terms
# 
# ## Example
# 
# - Stock A: 1M shares traded at 10,000 VND = 10B VND ADTV ✅ (passes 10B threshold)
# - Stock B: 1M shares traded at 1,000 VND = 1B VND ADTV ❌ (fails 10B threshold)
# 
# This is more meaningful than just counting shares!

# %%
# --- Suggested Threshold Values ---
# 
# thresholds = {
#     "Conservative": 20_000_000_000,  # 20B VND - very liquid stocks only
#     "Balanced": 10_000_000_000,      # 10B VND - good liquidity (recommended)
#     "Aggressive": 5_000_000_000,     # 5B VND - more stocks included
#     "Very Aggressive": 2_000_000_000 # 2B VND - maximum universe
# }
# 
# Recommendation: Start with 10B VND (Balanced) and adjust based on:
# - Desired universe size
# - Liquidity requirements
# - Market conditions 