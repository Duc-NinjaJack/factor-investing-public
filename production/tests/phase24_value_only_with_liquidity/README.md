# Phase 24: Value-Only Factor Backtesting

## Overview

This phase implements a **pure value factor strategy** using only the `Value_Composite` factor for stock selection, with no quality or momentum components. This provides a clean test of the standalone value factor performance.

## Implementation

### Files Created

1. **`production/scripts/value_only_backtesting.py`** - Subclass of `RealDataBacktesting` that uses `Value_Composite` instead of `QVM_Composite`
2. **`production/tests/phase24_value_only_with_liquidity/run_value_only_backtest.py`** - Simple script to run the value-only backtest
3. **`production/tests/phase24_value_only_with_liquidity/24_phase24_value_only_tearsheet.py`** - Comprehensive analysis script
4. **`production/tests/phase24_value_only_with_liquidity/24_phase24_value_only_tearsheet.ipynb`** - Jupyter notebook for interactive analysis

### Key Features

- **Pure Value Strategy**: Uses only `Value_Composite` factor scores
- **No Quality/Momentum**: Excludes Quality_Composite and Momentum_Composite components
- **Real Data**: Uses actual market data from database
- **Liquidity Filtering**: Applies ADTV-based liquidity thresholds
- **Comprehensive Analysis**: Performance metrics, visualizations, and risk analysis

## Results

### Performance Summary (10B VND Threshold)

| Metric | Value Strategy | Benchmark (VNINDEX) |
|--------|----------------|---------------------|
| Annual Return | 0.49% | 14.94% |
| Sharpe Ratio | 0.02 | N/A |
| Max Drawdown | -31.41% | N/A |
| Alpha | -12.96% | N/A |
| Beta | 0.85 | 1.00 |
| Information Ratio | -0.52 | N/A |

### Key Findings

1. **Underperformance**: Pure value factor underperforms the benchmark significantly
2. **Low Sharpe Ratio**: Very poor risk-adjusted returns (0.02)
3. **High Drawdown**: Maximum drawdown of -31.41%
4. **Negative Alpha**: -12.96% alpha indicates poor stock selection
5. **Low Beta**: 0.85 beta suggests defensive positioning

### Strategic Insights

- **Value Factor Alone is Insufficient**: The pure value strategy shows that value factors alone may not be sufficient for good performance
- **Need for Multi-Factor Approach**: Results suggest that combining value with quality and momentum factors (as in QVM) provides better performance
- **Risk Management Required**: High drawdown levels indicate need for risk management overlays

## Usage

### Run Basic Backtest

```bash
cd production/tests/phase24_value_only_with_liquidity
python run_value_only_backtest.py
```

### Run Comprehensive Analysis

```bash
python 24_phase24_value_only_tearsheet.py
```

### Interactive Analysis

Open `24_phase24_value_only_tearsheet.ipynb` in Jupyter for interactive analysis.

## Configuration

The strategy uses:
- **Portfolio Size**: 25 stocks
- **Rebalancing**: Monthly
- **Transaction Costs**: 20 bps
- **Liquidity Thresholds**: 3B VND and 10B VND
- **Date Range**: 2017-12-01 to 2025-07-28

## Data Sources

- **Price Data**: `vcsc_daily_data_complete` table
- **Factor Scores**: `factor_scores_qvm` table (Value_Composite only)
- **Benchmark**: VNINDEX from `etf_history` table
- **Liquidity**: ADTV data from pickle file

## Comparison with Phase 20

Phase 20 (QVM Composite) showed:
- **Dynamic Strategy**: 8.25% annual return, 0.33 Sharpe ratio
- **Static Strategy**: 3.48% annual return, 0.14 Sharpe ratio

Phase 24 (Value Only) shows:
- **Value Strategy**: 0.49% annual return, 0.02 Sharpe ratio

This demonstrates the importance of multi-factor approaches over single-factor strategies.

## Files Generated

- `value_only_performance_plots.png` - Performance visualizations
- `value_only_backtest_report.txt` - Detailed performance report
- `phase24_value_only_results.pkl` - Pickled results for further analysis

## Strategic Documentation

### Critical Insights Documents

1. **`STRATEGIC_INSIGHTS_ANALYSIS.md`** - Comprehensive strategic insights analysis
   - Performance hierarchy: Dynamic QVM > Static QVM > Value-Only
   - Regime switching effectiveness: Phase 20 vs Phase 21
   - Momentum factor evolution: Regime shift evidence
   - Liquidity filter performance: 10B vs 3B VND

2. **`FACTOR_PERFORMANCE_COMPARISON.md`** - Quantitative performance analysis
   - Detailed performance metrics across strategies
   - Statistical significance of regime shifts
   - Liquidity premium analysis
   - Strategic recommendations

### Key Strategic Findings

#### **Performance Hierarchy**
- **Dynamic QVM**: 8.25% return, 0.33 Sharpe ratio (Best)
- **Static QVM**: 3.48% return, 0.14 Sharpe ratio (Good)
- **Value-Only**: 0.49% return, 0.02 Sharpe ratio (Poor)

#### **Regime Switching Effectiveness**
- **Phase 20 (Simple)**: 8.25% return ✅
- **Phase 21 (Complex)**: Failed all criteria ❌
- **Principle**: Simple, practical approaches work better

#### **Momentum Factor Evolution**
- **2016-2020**: Strong mean reversion regime
- **2021-2025**: Weak momentum regime
- **Impact**: Momentum factor effectiveness has fundamentally changed

#### **Liquidity Filter Performance**
- **10B VND**: Better performance, lower risk
- **3B VND**: Lower performance, higher liquidity risk
- **Principle**: Quality over quantity

## Next Steps

1. **Compare with Quality-Only and Momentum-Only strategies**
2. **Analyze value factor performance across different market regimes**
3. **Investigate value factor timing and regime-switching approaches**
4. **Consider enhanced value factor construction methods**

## Strategic Recommendations

### **Factor Strategy Selection**
1. **Dynamic QVM** (8.25% return) - Best overall performance
2. **Static QVM** (3.48% return) - Good baseline performance  
3. **Avoid Value-Only** (0.49% return) - Poor performance

### **Implementation Guidelines**
- **Never use single-factor strategies**
- **Use simple, practical regime detection**
- **Prefer 10B VND liquidity threshold**
- **Account for transaction costs**
- **Monitor factor effectiveness continuously**

### **Risk Management**
- **Implement regime-aware risk management**
- **Use liquidity filters for better execution**
- **Monitor factor correlations across regimes**
- **Diversify across factor types** 