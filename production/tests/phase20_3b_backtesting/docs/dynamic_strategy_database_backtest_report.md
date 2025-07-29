# Dynamic Strategy Database Backtest Report

**Date:** 2025-07-29 23:01:12
**Purpose:** Compare dynamic regime-switching vs static QVM strategies
**Data Source:** Database (vcsc_daily_data_complete)

## 🎯 Executive Summary

This analysis compares the performance of:
- **Dynamic Strategy:** Regime-switching factor weights (QVM vs QV-Reversal)
- **Static Strategy:** Fixed QVM_Composite factor weights
- **Parameters:** Top 25 stocks, monthly rebalancing, 30 bps costs

## 📊 Performance Comparison

| Strategy | Annual Return (%) | Sharpe Ratio | Max Drawdown (%) | Alpha (%) | Beta | Information Ratio |
|----------|------------------|--------------|------------------|-----------|------|-------------------|
| 10B_VND_Dynamic | 8.25 | 0.33 | -65.73 | -4.78 | 1.16 | -0.22 |
| 10B_VND_Static | 3.48 | 0.14 | -67.06 | -9.35 | 1.14 | -0.54 |

## 🔍 Key Findings

**Best Annual Return:** 10B_VND_Dynamic (8.25%)
**Best Sharpe Ratio:** 10B_VND_Dynamic (0.33)
**Best Risk Control:** 10B_VND_Static (-67.06%)

## 📈 Regime Analysis (Dynamic Strategies)

### 10B_VND_Dynamic

- **Sideways:** 12 rebalances (52.2%)
- **Bull:** 7 rebalances (30.4%)
- **Stress:** 2 rebalances (8.7%)
- **Bear:** 2 rebalances (8.7%)

## 🎯 Conclusions

✅ **Dynamic strategy outperforms static strategy**
- Dynamic Sharpe: 0.33
- Static Sharpe: 0.14
- Regime-switching logic provides better risk-adjusted returns

## 📋 Recommendations

1. **Strategy Selection:** Choose based on risk tolerance and performance objectives
2. **Regime Detection:** Refine regime detection methodology if needed
3. **Parameter Tuning:** Optimize factor weights for different regimes
4. **Risk Management:** Implement additional risk controls for stress periods

---
**Analysis completed:** 2025-07-29 23:01:12