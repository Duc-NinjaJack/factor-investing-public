# Regime-Based Momentum Factor Methodology

This document outlines the methodology for testing the Information Coefficient (IC) and regime effectiveness of the momentum factor across different market regimes.

## Objective

- Assess the predictive power of the momentum factor in different market regimes (pre- and post-COVID)
- Compare IC, t-statistics, and hit rates across regimes
- Identify regime-dependent changes in momentum effectiveness

## Methodology

### 1. Regime Definition
- **Pre-COVID Regime**: 2016-2020
- **Post-COVID Regime**: 2021-2025

### 2. Factor Construction
- **Momentum Factor**: Multi-timeframe returns (1M, 3M, 6M, 12M) with skip-1-month convention
- **Weights**: 1M=15%, 3M=25%, 6M=30%, 12M=30%
- **Sector-neutral normalization**

### 3. IC Calculation
- At each rebalance date (quarterly/monthly), calculate the cross-sectional Spearman rank correlation between momentum scores and forward returns (1M, 3M, 6M, 12M)
- Store IC time series for each regime and horizon

### 4. Quality Assessment
- **Mean IC**: Should exceed 0.02
- **T-statistic**: Should exceed 2.0
- **Hit rate**: Should exceed 55%

### 5. Regime Comparison
- Compare mean IC, t-stat, and hit rate between regimes
- Identify statistically significant differences

## References
- See phase21_regime_switching_effectiveness/docs/regime_switching_methodology.md for general regime-switching methodology
- See docs/3_operational_framework/03a_factor_validation_playbook.md for IC standards