# Momentum Factor Information Coefficient (IC) Tests

This directory contains comprehensive tests for the Information Coefficient (IC) of the momentum factor for two periods: **2016-2020** and **2021-2025**.

## Overview

The Information Coefficient (IC) measures the correlation between factor values and forward returns, providing a key metric for validating the predictive power of the momentum factor.

### What is IC?

- **Definition**: IC is the Spearman rank correlation between factor scores and subsequent forward returns
- **Interpretation**: Higher IC values indicate better predictive power
- **Quality Thresholds**: 
  - Mean IC > 0.02 (2%)
  - T-statistic > 2.0
  - Hit rate > 55%

## Test Scripts

### 1. `run_momentum_ic_test.py` (Simple Test)
A straightforward test script that can be run quickly to validate momentum factor IC.

**Features:**
- Tests both periods (2016-2020, 2021-2025)
- Multiple forward horizons (1M, 3M, 6M, 12M)
- Quarterly rebalancing for efficiency
- Limited universe (50 stocks) for quick testing

**Usage:**
```bash
cd production/tests/phase22_pure_value_with_liquidity
python run_momentum_ic_test.py
```

### 2. `test_momentum_ic_analysis.py` (Comprehensive Analysis)
A full-featured analysis script with detailed reporting and visualizations.

**Features:**
- Monthly rebalancing for more granular analysis
- Comprehensive statistical analysis
- Detailed reporting with quality assessments
- Visualization of results
- Comparative analysis between periods

**Usage:**
```bash
cd production/tests/phase22_pure_value_with_liquidity
python test_momentum_ic_analysis.py
```

## Momentum Factor Methodology

The momentum factor uses the enhanced QVM engine v2 methodology:

### Factor Construction
- **Multi-timeframe returns**: 1M (15%), 3M (25%), 6M (30%), 12M (30%)
- **Skip-1-month convention**: Avoids microstructure noise
- **Sector-neutral normalization**: Pure alpha signal extraction
- **Weights**: 1M=15%, 3M=25%, 6M=30%, 12M=30%

### Return Calculation
```
R(t1,t2) = P(t2)/P(t1) - 1
```
Where:
- P(t1) = First available adjusted close on or after start date
- P(t2) = Last available adjusted close on or before end date

### Momentum Composite
```
M_composite = w_1M * R_1M + w_3M * R_3M + w_6M * R_6M + w_12M * R_12M
```

## Test Periods

### Period 1: 2016-2020 (Pre-COVID)
- **Description**: Pre-COVID market conditions
- **Characteristics**: Normal market dynamics, steady growth
- **Expected**: Standard momentum effectiveness

### Period 2: 2021-2025 (Post-COVID)
- **Description**: Post-COVID recovery and volatility
- **Characteristics**: High volatility, regime changes, recovery patterns
- **Expected**: Potentially different momentum effectiveness due to market regime changes

## Expected Results

### Quality Thresholds
Based on institutional standards:

| Metric | Threshold | Description |
|--------|-----------|-------------|
| Mean IC | > 0.02 | Average correlation should exceed 2% |
| T-statistic | > 2.0 | Statistical significance |
| Hit rate | > 55% | More than 55% of periods should have positive IC |

### Typical Results
- **1M forward returns**: Usually highest IC, but may be noisy
- **3M forward returns**: Good balance of signal and stability
- **6M forward returns**: More stable, moderate IC
- **12M forward returns**: Most stable, but may have lower IC due to signal decay

## Interpreting Results

### Good IC Results
```
✅ Mean IC: 0.035 (3.5%)
✅ T-statistic: 3.2
✅ Hit rate: 62%
```

### Poor IC Results
```
❌ Mean IC: 0.008 (0.8%)
❌ T-statistic: 1.1
❌ Hit rate: 48%
```

### Comparative Analysis
The tests compare IC between periods to identify:
- **Regime changes**: Different IC patterns between pre/post-COVID
- **Signal persistence**: How IC varies across forward horizons
- **Statistical significance**: Whether differences are meaningful

## Output Files

### Simple Test (`run_momentum_ic_test.py`)
- Console output with summary statistics
- No files generated

### Comprehensive Analysis (`test_momentum_ic_analysis.py`)
- **Report**: Detailed text report with all statistics
- **Visualizations**: 4-panel plot showing:
  - IC time series
  - IC distribution
  - Mean IC comparison
  - Hit rate comparison

## Dependencies

### Required Python Packages
```bash
pip install pandas numpy matplotlib seaborn scipy sqlalchemy pymysql pyyaml
```

### Database Access
- Requires access to the factor investing database
- Tables needed: `equity_history`, `intermediary_calculations_*`
- Database configuration in `config/database.yml`

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```
   ❌ Import error: No module named 'database.connection'
   ```
   **Solution**: Ensure you're running from the correct directory and paths are set correctly.

2. **Database Connection Errors**
   ```
   ❌ Failed to create database engine
   ```
   **Solution**: Check database configuration in `config/database.yml`

3. **Insufficient Data**
   ```
   ⚠️ Universe too small: 15 stocks
   ```
   **Solution**: The test requires at least 20 stocks. Check if data exists for the test period.

4. **No Fundamental Data**
   ```
   ⚠️ No fundamental data for 2020-01-01
   ```
   **Solution**: Ensure fundamental data tables are populated for the test period.

### Performance Tips

1. **Use Simple Test First**: Run `run_momentum_ic_test.py` for quick validation
2. **Adjust Universe Size**: Modify the `LIMIT 50` in `get_test_universe()` for faster/slower testing
3. **Change Frequency**: Use quarterly instead of monthly for faster testing
4. **Limit Periods**: Test one period at a time for debugging

## Example Output

```
🚀 MOMENTUM FACTOR IC TEST
============================================================
Testing Information Coefficient (IC) for momentum factor
Periods: 2016-2020 and 2021-2025
============================================================

🔧 Initializing QVM Engine...
✅ QVM Engine initialized

============================================================
🧪 TESTING: 2016-2020
📝 Pre-COVID period
============================================================

📈 Testing 1M forward returns...
📊 Generated 16 test dates

📅 Processing 2017-01-01 (1/16)
📊 Calculating momentum IC for 2017-01-01
✅ Calculated momentum factors for 45 stocks
✅ Calculated forward returns for 42 stocks
✅ IC: 0.0423 (n=40)

📊 IC ANALYSIS FOR 2016-2020 (1M)
==================================================
Number of observations: 15
Mean IC: 0.0387
Std IC: 0.0234
Min IC: -0.0123
Max IC: 0.0891
T-statistic: 6.412
Hit rate: 73.3%

🎯 QUALITY ASSESSMENT:
IC Quality: ✅ GOOD
Hit Rate Quality: ✅ GOOD
```

## Next Steps

After running the tests:

1. **Review Results**: Check if IC meets quality thresholds
2. **Compare Periods**: Analyze differences between pre/post-COVID periods
3. **Investigate Anomalies**: Look into periods with poor IC performance
4. **Optimize Parameters**: Adjust momentum weights or lookback periods if needed
5. **Extend Analysis**: Run comprehensive analysis for detailed insights

## References

- **Factor Validation Playbook**: `docs/3_operational_framework/03a_factor_validation_playbook.md`
- **Momentum Configuration**: `config/momentum.yml`
- **QVM Engine Documentation**: `docs/2_technical_implementation/02d_qvm_engine_v2_enhanced_factor_definitions.md` 