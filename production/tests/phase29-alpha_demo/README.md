# QVM Engine v3j - Component Analysis

## Overview

This directory contains a comprehensive component analysis of the QVM Engine v3j, breaking down the strategy into its core components to understand the contribution of each component to overall performance and Sharpe ratio.

## File Structure

```
phase29-alpha_demo/
├── 01_base_strategy.py              # Base strategy (equal weight, no factors, no regime)
├── 01_base_strategy.ipynb           # Jupyter notebook version
├── 02_regime_only.py                # Regime-only strategy ✅
├── 02_regime_only.ipynb             # Jupyter notebook version ✅
├── 03_factors_only.py               # Factors-only strategy ✅
├── 03_factors_only.ipynb            # Jupyter notebook version ✅
├── 04_integrated_strategy.py        # Integrated strategy (full implementation)
├── 04_integrated_strategy.ipynb     # Jupyter notebook version
├── component_comparison_results.csv # Performance comparison results ✅
├── components/
│   ├── base_engine.py               # Shared engine functionality
│   ├── regime_detector.py           # Regime detection component
│   └── factor_calculator.py         # Factor calculation component
├── analysis/
│   ├── component_comparison.py      # Component comparison analysis ✅
│   └── performance_visualization.py # Performance charts generation ✅
├── insights/
│   ├── component_contribution_analysis.md      # Analysis methodology ✅
│   ├── component_contribution_analysis_results.md # Results documentation ✅
│   ├── performance_visualization_insights.md   # Visualization insights ✅
│   └── component_performance_comparison.png    # Performance charts ✅
└── README.md                        # This file
```

## Components Analyzed

### 1. Base Strategy (01_base_strategy.py)
- **Status**: ✅ Complete
- **Description**: Equal weight allocation with no factors or regime detection
- **Purpose**: Serves as the baseline for comparison
- **Features**:
  - Top 200 stocks by ADTV
  - Equal weight allocation (20 stocks)
  - Monthly rebalancing
  - No factor analysis
  - No regime detection
  - Always 100% invested

### 2. Regime-Only Strategy (02_regime_only.py)
- **Status**: ✅ Complete
- **Description**: Regime detection only, no factor analysis
- **Purpose**: Isolate the contribution of regime detection
- **Features**:
  - Same universe as base strategy
  - Regime-based allocation (Bull: 100%, Bear: 80%, Sideways: 60%, Stress: 40%)
  - Equal weight within each regime
  - No factor analysis

### 3. Factors-Only Strategy (03_factors_only.py)
- **Status**: ✅ Complete
- **Description**: Factor analysis only, no regime detection
- **Purpose**: Isolate the contribution of factor analysis
- **Features**:
  - Same universe as base strategy
  - Factor-based stock selection (ROAA + P/E + Momentum)
  - Always 100% invested
  - No regime detection

### 4. Integrated Strategy (04_integrated_strategy.py)
- **Status**: ✅ Complete
- **Description**: Full strategy with both regime detection and factor analysis
- **Purpose**: Show the combined effect of all components
- **Features**:
  - Regime-based allocation
  - Factor-based stock selection
  - Combined approach

## Shared Components

### Base Engine (components/base_engine.py)
- **Status**: ✅ Complete
- **Purpose**: Shared functionality for all strategy variants
- **Features**:
  - Data loading and pre-computation
  - Performance metrics calculation
  - Database connection management

### Regime Detector (components/regime_detector.py)
- **Status**: ✅ Complete
- **Purpose**: Regime detection logic
- **Features**:
  - 4-regime classification (Bull, Bear, Sideways, Stress)
  - Configurable thresholds
  - Regime-based allocation rules

### Factor Calculator (components/factor_calculator.py)
- **Status**: ✅ Complete
- **Purpose**: Factor calculation logic
- **Features**:
  - Sector-aware P/E calculation
  - Multi-horizon momentum scoring
  - Composite score calculation
  - Entry criteria application

## Analysis Tools

### Component Comparison (analysis/component_comparison.py)
- **Status**: ✅ Complete
- **Purpose**: Run all strategy variants and compare performance
- **Features**:
  - Automated execution of all strategies
  - Performance comparison and visualization
  - Component contribution analysis
  - Results export to CSV

## Analysis Results ✅ COMPLETED

### Component Performance Summary
| Strategy | Sharpe Ratio | Annualized Return (%) | Annualized Volatility (%) | Max Drawdown (%) |
|----------|--------------|----------------------|---------------------------|------------------|
| **Base** | 0.091 | 2.22 | 24.43 | -65.55 |
| **Regime_Only** | 0.234 | 3.41 | 14.59 | -46.98 |
| **Factors_Only** | 0.304 | 6.96 | 22.89 | -60.30 |
| **Integrated** | **0.393** | **5.29** | **13.47** | **-44.44** |

### Key Findings
- **Integrated Strategy** achieves the best Sharpe ratio (0.393) through synergistic combination
- **Regime Detection** provides superior risk management (40.3% volatility reduction)
- **Factor Analysis** delivers superior returns (213.5% return improvement)
- **Component Synergy** yields 29.3% improvement over best individual component

### Component Contributions
- **Regime_Only**: +157.5% Sharpe improvement over Base
- **Factors_Only**: +235.3% Sharpe improvement over Base  
- **Integrated**: +333.2% Sharpe improvement over Base

## Next Steps

### 1. Advanced Analysis (Optional)
- [ ] Sensitivity analysis of regime thresholds
- [ ] Factor weight optimization
- [ ] Out-of-sample validation
- [ ] Transaction cost impact analysis

### 2. Production Implementation
- [ ] Real-time regime detection system
- [ ] Factor data pipeline optimization
- [ ] Risk management overlays
- [ ] Performance monitoring dashboard

### 3. Strategy Enhancement
- [ ] Additional factor exploration
- [ ] Regime transition analysis
- [ ] Sector-specific optimizations
- [ ] Dynamic rebalancing frequency

## Execution Instructions

### Individual Strategy Execution
```bash
# Run base strategy
python 01_base_strategy.py

# Run integrated strategy
python 04_integrated_strategy.py

# Convert to notebooks (if needed)
jupytext --to notebook 01_base_strategy.py
jupytext --to notebook 04_integrated_strategy.py
```

### Component Comparison Execution
```bash
# Run comprehensive comparison
python analysis/component_comparison.py
```

## Expected Outcomes

### Performance Hierarchy (Expected)
1. **Integrated Strategy** (Highest Sharpe)
2. **Factors-Only Strategy** (Medium-High Sharpe)
3. **Regime-Only Strategy** (Medium Sharpe)
4. **Base Strategy** (Lowest Sharpe)

### Key Metrics to Analyze
- **Sharpe Ratio**: Primary measure of risk-adjusted returns
- **Maximum Drawdown**: Risk management effectiveness
- **Information Ratio**: Alpha generation relative to benchmark
- **Calmar Ratio**: Return per unit of maximum drawdown
- **Annualized Return**: Absolute performance
- **Annualized Volatility**: Risk measure

## Technical Notes

### Performance Optimization
- All strategies use pre-computed data for faster execution
- Database queries reduced from 342 to 4 (98.8% reduction)
- Vectorized operations for momentum calculations
- Expected 5-10x speed improvement

### Data Consistency
- All strategies use the same universe (Top 200 stocks by ADTV)
- Same rebalancing frequency (monthly)
- Same transaction costs (30 bps)
- Same time period (2016-2025)

### Jupytext Format
- All Python files use jupytext-compatible format
- Proper cell markers: `# %% [markdown]` and `# %%`
- Markdown headers: `# # HEADER_NAME`
- Clean notebook conversion

## Contact

For questions or issues with this component analysis, please refer to the insights documentation or create an issue in the project repository. 