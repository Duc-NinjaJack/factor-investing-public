# QVM Engine v3j - Component Analysis

## Overview

This directory contains a comprehensive component analysis of the QVM Engine v3j, breaking down the strategy into its core components to understand the contribution of each component to overall performance and Sharpe ratio.

## File Structure

```
phase29-alpha_demo/
├── 01_base_strategy.py              # Base strategy (equal weight, no factors, no regime)
├── 01_base_strategy.ipynb           # Jupyter notebook version
├── 02_regime_only.py                # Regime-only strategy (to be created)
├── 02_regime_only.ipynb             # Jupyter notebook version (to be created)
├── 03_factors_only.py               # Factors-only strategy (to be created)
├── 03_factors_only.ipynb            # Jupyter notebook version (to be created)
├── 04_integrated_strategy.py        # Integrated strategy (full implementation)
├── 04_integrated_strategy.ipynb     # Jupyter notebook version
├── components/
│   ├── base_engine.py               # Shared engine functionality
│   ├── regime_detector.py           # Regime detection component
│   └── factor_calculator.py         # Factor calculation component
├── analysis/
│   └── component_comparison.py      # Component comparison analysis
├── insights/
│   └── component_contribution_analysis.md  # Analysis methodology and insights
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
- **Status**: ⏳ To be created
- **Description**: Regime detection only, no factor analysis
- **Purpose**: Isolate the contribution of regime detection
- **Features**:
  - Same universe as base strategy
  - Regime-based allocation (Bull: 100%, Bear: 80%, Sideways: 60%, Stress: 40%)
  - Equal weight within each regime
  - No factor analysis

### 3. Factors-Only Strategy (03_factors_only.py)
- **Status**: ⏳ To be created
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

## Next Steps

### 1. Create Remaining Strategy Files
- [ ] Create `02_regime_only.py` (regime detection only)
- [ ] Create `03_factors_only.py` (factors only)
- [ ] Convert both to Jupyter notebooks using jupytext

### 2. Run Component Analysis
- [ ] Execute individual strategy files to generate baseline results
- [ ] Run component comparison analysis for comprehensive comparison
- [ ] Analyze results and document insights

### 3. Documentation and Insights
- [ ] Update insights documentation with actual results
- [ ] Create performance comparison visualizations
- [ ] Document component contribution findings

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