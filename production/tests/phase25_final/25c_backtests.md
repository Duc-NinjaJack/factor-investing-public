# Phase 25c: Institutional Grade Composite - Structural
Refactoring & Multi-Window Analysis

## 🎯 **MISSION STATEMENT**
Implement structural refactoring with centralized
configuration to enable rapid testing across multiple time
windows and systematic activation of performance-critical
components. This notebook represents Day 1-7 of the
institutional sprint to achieve IC hurdles.

## 📊 **PREVIOUS RESULTS SUMMARY (Phase 25b)**
**Current Best Model: `Composite_Q_20_1.25×`**
- Annual Return (net): **13.0%** ❌ (Target: ≥15%)
- Annual Volatility: **19.8%** ❌ (Target: 15%)
- Sharpe Ratio (net): **0.65** ❌ (Target: ≥1.0)
- Max Drawdown: **-46.3%** ❌ (Limit: ≥-35%)
- Beta vs VN-Index: **0.85** ⚠️ (Target: ≤0.75)
- Information Ratio: **0.12** ❌ (Target: ≥0.8)

**ROOT CAUSE ANALYSIS:**
- Insufficient gross alpha density due to static V:Q:M:R ≈ 
50:25:20:5 weights
- Missing walk-forward optimizer, hybrid regime filter, 
non-linear cost model
- Liquidity regime shift around 2020 not properly handled

## 🔧 **STRUCTURAL ENHANCEMENTS (Phase 25c)**

### **1. Multi-Window Configuration**
- **FULL_2016_2025**: Complete historical record
- **LIQUID_2018_2025**: Post-IPO spike, includes 2018 
stress
- **POST_DERIV_2020_2025**: High-liquidity era (VN30
derivatives launch)
- **ADAPTIVE_2016_2025**: Full period with liquidity-aware
weighting

### **2. Infrastructure Activation Sequence**
1. **Liquidity-aware universe & cost model** → Realistic
net returns
2. **Walk-forward factor optimizer** → Adaptive alpha
density
3. **Hybrid volatility ⊕ regime overlay** → Risk-adjusted
performance

### **3. Investment Committee Gates**
| Metric | Target | Current | Gap |
|--------|--------|---------|-----|
| Sharpe Ratio (net) | ≥1.0 | 0.65 | **+54%** |
| Max Drawdown | ≥-35% | -46.3% | **+32%** |
| Annual Return (net) | ≥15% | 13.0% | **+15%** |
| Information Ratio | ≥0.8 | 0.12 | **+567%** |

## 🎯 **SUCCESS CRITERIA**
- At least one time window achieves Sharpe ≥ 1.0 (net,
unlevered)
- Max drawdown ≤ -35% across all viable windows
- Demonstrate alpha persistence in high-liquidity regime
(2020-2025)
- Generate audit-ready comparative tearsheets

## 📋 **NOTEBOOK STRUCTURE**
1. **Configuration & Setup** - Centralized config loading
2. **Data Pipeline** - Multi-window data preparation
3. **Universe Construction** - Liquidity-aware filtering
4. **Cost Model Integration** - Non-linear ADTV impact
5. **Walk-Forward Optimization** - Bayesian factor
weighting
6. **Hybrid Risk Overlay** - Volatility + regime detection
7. **Multi-Window Backtesting** - Comparative analysis
8. **Performance Attribution** - IC gate assessment
9. **Institutional Tearsheets** - Audit-ready reporting

---
**Author:** Vietnam Factor Investing Platform
**Date:** July 30, 2025
**Version:** 25c (Structural Refactoring)
**Status:** 🔄 ACTIVE DEVELOPMENT

# ===============================================================
# PHASE 25c: CELL 1 - CENTRALIZED CONFIGURATION & SETUP
# ===============================================================

import pandas as pd
import numpy as np
import warnings
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import yaml
from typing import Dict, List, Optional, Tuple, Any
import logging

# Add project root to path
project_root = Path.cwd().parent.parent.parent
sys.path.append(str(project_root))

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)

# ===============================================================
# 1. MULTI-WINDOW CONFIGURATION SYSTEM
# ===============================================================

# Central configuration dictionary - single source of truth
PHASE_25C_CONFIG = {
# === BACKTEST WINDOWS ===
"backtest_windows": {
    "FULL_2016_2025": {
        "start": "2016-01-01",
        "end": "2025-12-31",
        "description": "Complete historical record",
        "liquidity_regime": "mixed"
    },
    "LIQUID_2018_2025": {
        "start": "2018-01-01",
        "end": "2025-12-31",
        "description": "Post-IPO spike, includes 2018 stress",
        "liquidity_regime": "improving"
    },
    "POST_DERIV_2020_2025": {
        "start": "2020-01-01",
        "end": "2025-12-31",
        "description": "High-liquidity era (VN30 derivatives launch)",
        "liquidity_regime": "high"
    },
    "ADAPTIVE_2016_2025": {
        "start": "2016-01-01",
        "end": "2025-12-31",
        "description": "Full period with liquidity-aware weighting",
        "liquidity_regime": "adaptive"
    }
},

# === ACTIVE CONFIGURATION ===
"active_window": "LIQUID_2018_2025",  # Primary test window
"rebalance_frequency": "Q",  # Quarterly rebalancing
"portfolio_size": 20,  # Fixed 20 names

# === INVESTMENT COMMITTEE GATES ===
"ic_hurdles": {
    "sharpe_ratio_net": 1.0,
    "max_drawdown_limit": -0.35,  # -35%
    "annual_return_net": 0.15,  # 15%
    "information_ratio": 0.8,
    "beta_vs_vnindex": 0.75,  # ≤0.75
    "volatility_target": 0.15  # 15%
},

# === LIQUIDITY CONSTRAINTS ===
"liquidity_filters": {
    "min_adtv_vnd": 10_000_000_000,  # 10 billion VND
    "adtv_to_mcap_ratio": 0.0004,  # 0.04% of market cap
    "max_position_vs_adtv": 0.05,  # 5% of daily volume
    "rolling_adtv_days": 20
},

# === COST MODEL PARAMETERS ===
"cost_model": {
    "base_cost_bps": 3.0,  # 3 bps base cost
    "impact_coefficient": 0.15,  # sqrt coefficient for market impact
    "max_participation_rate": 0.05,  # 5% of ADTV
    "bid_ask_spread_bps": 8.0  # Average bid-ask spread
},

# === FACTOR OPTIMIZATION ===
"optimization": {
    "lookback_months": 24,  # 24-month fitting window
    "lockout_months": 6,   # 6-month lock period
    "bayesian_priors": {
        "value_min": 0.30,    # Value ≥ 30%
        "quality_max": 0.25,  # Quality ≤ 25%
        "momentum_min": 0.25, # Momentum ≥ 25%
        "reversal_max": 0.10  # Reversal ≤ 10%
    },
    "regularization_lambda": 0.05
},

# === RISK OVERLAY ===
"risk_overlay": {
    "volatility_target": 0.15,
    "regime_detection": {
        "vol_threshold": 0.25,  # 25% realized vol threshold
        "drawdown_threshold": -0.10,  # -10% drawdown threshold
        "lookback_days": 63,
        "cooldown_days": 5
    }
}
}

# ===============================================================
# 2. CONFIGURATION VALIDATION & UTILITIES
# ===============================================================

def validate_config(config: Dict) -> bool:
"""Validate configuration integrity"""
required_keys = ['backtest_windows', 'active_window', 'ic_hurdles']

for key in required_keys:
    if key not in config:
        raise ValueError(f"Missing required config key: {key}")

# Validate active window exists
if config['active_window'] not in config['backtest_windows']:
    raise ValueError(f"Active window '{config['active_window']}' not found in backtest_windows")

# Validate date formats
for window_name, window_config in config['backtest_windows'].items():
    try:
        pd.Timestamp(window_config['start'])
        pd.Timestamp(window_config['end'])
    except Exception as e:
        raise ValueError(f"Invalid date format in window {window_name}: {e}")

return True

def get_active_window_config(config: Dict) -> Dict:
"""Get configuration for active window"""
active_window = config['active_window']
window_config = config['backtest_windows'][active_window].copy()

# Add parsed timestamps
window_config['start_date'] = pd.Timestamp(window_config['start'])
window_config['end_date'] = pd.Timestamp(window_config['end'])

return window_config

def setup_logging() -> logging.Logger:
"""Setup structured logging for the notebook"""
logger = logging.getLogger('phase25c')
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

return logger

# ===============================================================
# 3. INITIALIZE CONFIGURATION
# ===============================================================

# Validate configuration
validate_config(PHASE_25C_CONFIG)

# Get active window details
ACTIVE_CONFIG = get_active_window_config(PHASE_25C_CONFIG)

# Setup logging
logger = setup_logging()

# ===============================================================
# 4. CONFIGURATION SUMMARY
# ===============================================================

print("=" * 80)
print("PHASE 25C: INSTITUTIONAL GRADE COMPOSITE - CONFIGURATION LOADED")
print("=" * 80)
print(f"📅 Active Window: {PHASE_25C_CONFIG['active_window']}")
print(f"📊 Period: {ACTIVE_CONFIG['start']} to {ACTIVE_CONFIG['end']}")
print(f"📈 Description: {ACTIVE_CONFIG['description']}")
print(f"🔄 Rebalance: {PHASE_25C_CONFIG['rebalance_frequency']} (Quarterly)")
print(f"📋 Portfolio Size: {PHASE_25C_CONFIG['portfolio_size']} names")
print()
print("🎯 INVESTMENT COMMITTEE HURDLES:")
for metric, target in PHASE_25C_CONFIG['ic_hurdles'].items():
if isinstance(target, float) and target < 1:
    print(f"   • {metric.replace('_', ' ').title()}: {target:.1%}")
else:
    print(f"   • {metric.replace('_', ' ').title()}: {target}")
print()
print("💧 LIQUIDITY CONSTRAINTS:")
print(f"   • Min ADTV: {PHASE_25C_CONFIG['liquidity_filters']['min_adtv_vnd']:,} VND")
print(f"   • ADTV/MCap: {PHASE_25C_CONFIG['liquidity_filters']['adtv_to_mcap_ratio']:.2%}")
print(f"   • Max Position: {PHASE_25C_CONFIG['liquidity_filters']['max_position_vs_adtv']:.1%} of ADTV")
print()
print("🔧 Available Windows:")
for window_name, window_info in PHASE_25C_CONFIG['backtest_windows'].items():
status = ">>> ACTIVE <<<" if window_name == PHASE_25C_CONFIG['active_window'] else ""
print(f"   • {window_name}: {window_info['start']} to {window_info['end']} {status}")
print("=" * 80)

# Configuration validation checkpoint
logger.info(f"Phase 25c configuration loaded successfully")
logger.info(f"Active window: {PHASE_25C_CONFIG['active_window']} "
        f"({ACTIVE_CONFIG['start']} to {ACTIVE_CONFIG['end']})")

2025-07-30 19:34:01,408 - phase25c - INFO - Phase 25c configuration loaded successfully
2025-07-30 19:34:01,412 - phase25c - INFO - Active window: LIQUID_2018_2025 (2018-01-01 to 2025-12-31)
================================================================================
PHASE 25C: INSTITUTIONAL GRADE COMPOSITE - CONFIGURATION LOADED
================================================================================
📅 Active Window: LIQUID_2018_2025
📊 Period: 2018-01-01 to 2025-12-31
📈 Description: Post-IPO spike, includes 2018 stress
🔄 Rebalance: Q (Quarterly)
📋 Portfolio Size: 20 names

🎯 INVESTMENT COMMITTEE HURDLES:
• Sharpe Ratio Net: 1.0
• Max Drawdown Limit: -35.0%
• Annual Return Net: 15.0%
• Information Ratio: 80.0%
• Beta Vs Vnindex: 75.0%
• Volatility Target: 15.0%

💧 LIQUIDITY CONSTRAINTS:
• Min ADTV: 10,000,000,000 VND
• ADTV/MCap: 0.04%
• Max Position: 5.0% of ADTV

🔧 Available Windows:
• FULL_2016_2025: 2016-01-01 to 2025-12-31 
• LIQUID_2018_2025: 2018-01-01 to 2025-12-31 >>> ACTIVE <<<
• POST_DERIV_2020_2025: 2020-01-01 to 2025-12-31 
• ADAPTIVE_2016_2025: 2016-01-01 to 2025-12-31 
================================================================================

# ===========================================================
===========
# PHASE 25c: CELL 5 - DAY 1 (CORRECTED): NON-LINEAR ADTV COST
MODEL
# ===========================================================
===========

print("🚀 DAY 1 (CORRECTED): EMBEDDING CALIBRATED ADTV COST 
MODEL")
print("=" * 70)
print("CORRECTION: Impact coefficient 0.002 (20 bps), 
turnover-based calculation")
print("=" * 70)

# Update configuration with correct calibration
PHASE_25C_CONFIG['cost_model'] = {
    'base_cost_bps': 5.0,  # 5 bps (commission + fees)
    'impact_coefficient': 0.002,  # 0.2% (20 bps) - realistic
for Vietnam
    'max_participation_rate': 0.15,  # 15% of ADTV
    'days_to_trade': 2.2  # Effective days when splitting 
orders
}

def calculate_adtv_based_costs_corrected(
    portfolio_weights_new: pd.Series,
    portfolio_weights_old: pd.Series,
    adtv_data: pd.DataFrame,
    portfolio_value_vnd: float,
    config: Dict
) -> pd.Series:
    """
    Calculate CORRECTED non-linear ADTV-based transaction 
costs.
    
    Cost Model: total_cost_pct = base_cost_bps/10000 + 
impact_coeff * sqrt(turnover / (adtv * days_to_trade))
    
    CRITICAL FIXES:
    - Impact coefficient: 0.002 (not 0.15)
    - Order size: turnover only (not full position)
    - ADV divisor: 2.2 for multi-day execution
    """

    base_cost_bps = config['cost_model']['base_cost_bps']  # 
5 bps
    impact_coeff = config['cost_model']['impact_coefficient']
# 0.002 (20 bps)
    days_to_trade = config['cost_model'].get('days_to_trade',
2.2)  # Split across days

    logger.info(f"💰 Calculating CORRECTED ADTV-based 
transaction costs")
    logger.info(f"   Base cost: {base_cost_bps} bps")
    logger.info(f"   Impact coefficient: {impact_coeff:.4f} 
({impact_coeff*100:.1f} bps)")
    logger.info(f"   Days to trade: {days_to_trade}")

    # Calculate TURNOVER (delta weights) - this is what we 
actually trade
    all_tickers = set(portfolio_weights_new.index) |
set(portfolio_weights_old.index)

    turnover_data = []
    for ticker in all_tickers:
        weight_new = portfolio_weights_new.get(ticker, 0.0)
        weight_old = portfolio_weights_old.get(ticker, 0.0)
        delta_weight = abs(weight_new - weight_old)

        if delta_weight > 1e-6:  # Only if there's actual 
trading
            turnover_data.append({
                'ticker': ticker,
                'weight_new': weight_new,
                'weight_old': weight_old,
                'delta_weight': delta_weight
            })

    if not turnover_data:
        return pd.Series(dtype='float64')

    turnover_df = pd.DataFrame(turnover_data)

    # Merge with ADTV data
    turnover_df = turnover_df.merge(
        adtv_data[['ticker', 'adtv_vnd']],
        on='ticker',
        how='left'
    )

    # Handle missing ADTV (conservative assumption)
    missing_adtv = turnover_df['adtv_vnd'].isna()
    if missing_adtv.any():
        logger.warning(f"⚠️ {missing_adtv.sum()} tickers 
missing ADTV - using conservative estimate")
        turnover_df.loc[missing_adtv, 'adtv_vnd'] = 5e9  # 5B
VND conservative estimate

    # Calculate traded value (turnover portion only)
    turnover_df['traded_value_vnd'] =
turnover_df['delta_weight'] * portfolio_value_vnd

    # Calculate costs with CORRECTED formula
    base_cost_pct = base_cost_bps / 10000

    # Market impact with multi-day execution
    effective_adtv = turnover_df['adtv_vnd'] * days_to_trade
    impact_ratio = turnover_df['traded_value_vnd'] /
effective_adtv
    turnover_df['impact_cost_pct'] = impact_coeff *
np.sqrt(impact_ratio)

    # Total cost per position
    turnover_df['total_cost_pct'] = base_cost_pct +
turnover_df['impact_cost_pct']

    # Apply costs only to positions with turnover
    cost_series = pd.Series(0.0, index=all_tickers)
    for _, row in turnover_df.iterrows():
        cost_series[row['ticker']] = row['total_cost_pct']

    # Summary statistics
    if len(turnover_df) > 0:
        avg_cost = turnover_df['total_cost_pct'].mean()
        max_cost = turnover_df['total_cost_pct'].max()
        total_turnover = turnover_df['delta_weight'].sum()

        logger.info(f"   Turnover: {total_turnover:.1%} 
(two-way)")
        logger.info(f"   Positions traded: 
{len(turnover_df)}")
        logger.info(f"   Cost range: {base_cost_pct:.2%} - 
{max_cost:.2%}")
        logger.info(f"   Average cost: {avg_cost:.2%}")

    return cost_series

# Update PortfolioEngine to use corrected cost calculation
class PortfolioEngine_v5_2_corrected(PortfolioEngine_v5_2):
    """Corrected PortfolioEngine with proper cost 
calibration"""

    def _calculate_net_returns_with_costs(self, 
daily_holdings: pd.DataFrame, 
                                        rebalance_dates: 
List[pd.Timestamp]) -> pd.Series:
        """Calculate net returns with CORRECTED ADTV-based 
cost model"""

        self.logger.info(f"💰 Calculating net returns with 
CORRECTED cost model")

        # Calculate gross returns
        holdings_shifted =
daily_holdings.shift(1).fillna(0.0)
        gross_returns = (holdings_shifted *
self.daily_returns_matrix).sum(axis=1)

        # Track costs
        cost_series = pd.Series(0.0,
index=gross_returns.index)
        total_rebalance_costs = []

        # Calculate costs at each rebalance
        prev_weights = pd.Series(dtype='float64')

        for i, rebal_date in enumerate(rebalance_dates):
            try:
                next_day = rebal_date + pd.Timedelta(days=1)
                if next_day in daily_holdings.index:
                    # Get new weights
                    new_weights =
daily_holdings.loc[next_day]
                    new_weights = new_weights[new_weights >
0]

                    if len(new_weights) > 0:
                        # Load ADTV data
                        adtv_data =
self._load_adtv_data(rebal_date)

                        if not adtv_data.empty:
                            # Calculate turnover-based costs
                            position_costs =
calculate_adtv_based_costs_corrected(

portfolio_weights_new=new_weights,

portfolio_weights_old=prev_weights,
                                adtv_data=adtv_data,

portfolio_value_vnd=self.portfolio_value_vnd,
                                config=self.config
                            )

                            # Portfolio-weighted cost
                            portfolio_cost = (new_weights *
position_costs).sum()
                            cost_series.loc[next_day] =
portfolio_cost

total_rebalance_costs.append(portfolio_cost)

                            self.logger.info(f"   
{rebal_date.date()}: {portfolio_cost:.3%} cost")

                        # Update previous weights for next 
iteration
                        prev_weights = new_weights.copy()

            except Exception as e:
                self.logger.warning(f"   Cost calculation 
failed for {rebal_date.date()}: {e}")
                continue

        # Apply costs
        net_returns = gross_returns - cost_series

        # Summary
        if total_rebalance_costs:
            avg_rebalance_cost =
np.mean(total_rebalance_costs)
            annual_cost_drag = sum(total_rebalance_costs) /
(len(net_returns) / 252)

            self.logger.info(f"✅ Net returns calculated:")
            self.logger.info(f"   Average rebalance cost: 
{avg_rebalance_cost:.3%}")
            self.logger.info(f"   Annual cost drag: 
{annual_cost_drag:.3%}")
            self.logger.info(f"   Total rebalances: 
{len(total_rebalance_costs)}")

        return net_returns.rename('Net_Returns_Corrected')

# Test corrected cost model
def test_corrected_costs():
    """Validate corrected cost model produces realistic 
costs"""

    print(f"\n🧪 TESTING CORRECTED COST MODEL (VIETNAM 
CALIBRATION)")
    print("-" * 60)

    # Test scenario: 5% position with 50% turnover at 15% ADV
    test_adtv = pd.DataFrame({
        'ticker': ['HPG', 'VNM', 'DXG'],
        'adtv_vnd': [300e9, 150e9, 50e9]  # Different 
liquidity levels
    })

    # Old and new weights (50% turnover)
    old_weights = pd.Series([0.05, 0.05, 0.00], index=['HPG',
'VNM', 'DXG'])
    new_weights = pd.Series([0.025, 0.05, 0.025],
index=['HPG', 'VNM', 'DXG'])

    portfolio_value = 50e9  # 50B VND

    costs = calculate_adtv_based_costs_corrected(
        portfolio_weights_new=new_weights,
        portfolio_weights_old=old_weights,
        adtv_data=test_adtv,
        portfolio_value_vnd=portfolio_value,
        config=PHASE_25C_CONFIG
    )

    print(f"\nTest Results (50B VND portfolio):")
    for ticker in ['HPG', 'VNM', 'DXG']:
        old_w = old_weights.get(ticker, 0)
        new_w = new_weights.get(ticker, 0)
        delta_w = abs(new_w - old_w)
        traded_value = delta_w * portfolio_value / 1e9
        participation = (delta_w * portfolio_value) /
(test_adtv.loc[test_adtv['ticker']==ticker,
'adtv_vnd'].iloc[0] * 2.2)

        print(f"\n{ticker}:")
        print(f"   Weight: {old_w:.1%} → {new_w:.1%} 
(Δ={delta_w:.1%})")
        print(f"   Traded: {traded_value:.1f}B VND")
        print(f"   Participation: {participation:.1%} of 
daily volume")
        print(f"   Cost: {costs[ticker]:.3%} 
({costs[ticker]*100:.1f} bps)")

    # Validate range
    avg_cost = costs.mean()
    max_cost = costs.max()

    print(f"\nSummary:")
    print(f"   Average cost: {avg_cost:.3%} 
({avg_cost*100:.1f} bps)")
    print(f"   Max cost: {max_cost:.3%} ({max_cost*100:.1f} 
bps)")

    # Pass/fail check
    if 0.0015 <= avg_cost <= 0.0040:  # 15-40 bps range
        print(f"   ✅ PASSED: Costs in realistic range for 
Vietnam")
        return True
    else:
        print(f"   ❌ FAILED: Costs outside expected range")
        return False

# Execute test
test_passed = test_corrected_costs()

if test_passed:
    print(f"\n✅ CORRECTED COST MODEL VALIDATED")
    print(f"🎯 Ready to re-run backtests with realistic 
transaction costs")
else:
    print(f"\n❌ Cost model calibration needs adjustment")