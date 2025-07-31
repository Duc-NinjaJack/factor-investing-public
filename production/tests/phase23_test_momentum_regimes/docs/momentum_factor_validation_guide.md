# Momentum Factor Validation Guide

**Date**: July 30, 2025  
**Author**: Factor Investing Team  
**Version**: 1.0  

## Overview

This guide provides comprehensive documentation for validating momentum factors across different market regimes, with specific focus on the Vietnamese equity market. The framework includes statistical analysis, regime detection, and parameter optimization.

## Table of Contents

1. [Factor Construction](#factor-construction)
2. [Validation Methodology](#validation-methodology)
3. [Regime Analysis](#regime-analysis)
4. [Statistical Framework](#statistical-framework)
5. [Implementation Guide](#implementation-guide)
6. [Troubleshooting](#troubleshooting)

## Factor Construction

### Momentum Factor Definition

The momentum factor is constructed using multi-timeframe returns with skip-1-month convention and sector-neutral normalization:

```python
# Multi-timeframe momentum construction
periods = [
    ('1M', 1, 1, 0.15),   # 1-month lookback, 1-month skip, 15% weight
    ('3M', 3, 1, 0.25),   # 3-month lookback, 1-month skip, 25% weight
    ('6M', 6, 1, 0.30),   # 6-month lookback, 1-month skip, 30% weight
    ('12M', 12, 1, 0.30)  # 12-month lookback, 1-month skip, 30% weight
]
```

### Key Components

1. **Lookback Periods**: 1M, 3M, 6M, 12M
2. **Skip Convention**: 1-month skip to avoid microstructure noise
3. **Weighting**: Sophisticated weighting across timeframes
4. **Normalization**: Sector-neutral z-score normalization

### Implementation in QVM Engine

```python
def _calculate_enhanced_momentum_composite(self, data, analysis_date, universe):
    """
    Enhanced momentum calculation with skip-1-month convention.
    """
    for period_name, lookback, skip, weight in periods:
        end_date = analysis_date - pd.DateOffset(months=skip)
        start_date_period = analysis_date - pd.DateOffset(months=lookback + skip)
        returns = self._calculate_returns_fixed(price_data, start_date_period, end_date)
```

## Validation Methodology

### Information Coefficient (IC) Analysis

The primary validation metric is the Information Coefficient, calculated as the Spearman rank correlation between factor scores and forward returns:

```python
def calculate_momentum_ic(engine, analysis_date, universe, forward_months=1):
    """
    Calculate momentum IC for given date and universe.
    """
    # Get momentum scores
    momentum_scores = engine._calculate_enhanced_momentum_composite(...)
    
    # Calculate forward returns
    forward_returns = calculate_forward_returns(...)
    
    # Calculate Spearman correlation
    ic = spearman_correlation(momentum_scores, forward_returns)
    return ic
```

### Quality Gates

| Metric | Threshold | Interpretation |
|--------|-----------|----------------|
| Mean IC | > 0.02 | Factor shows predictive power |
| T-statistic | > 2.0 | Statistically significant |
| Hit Rate | > 55% | Factor direction correct >50% |

### Forward Return Horizons

- **1M**: 1-month forward returns
- **3M**: 3-month forward returns  
- **6M**: 6-month forward returns
- **12M**: 12-month forward returns

## Regime Analysis

### Regime Detection

Regimes are identified based on IC patterns across different time periods:

1. **Mean Reversion Regime**: Negative IC values
2. **Momentum Regime**: Positive IC values
3. **Neutral Regime**: IC values near zero

### Statistical Tests

```python
def test_regime_differences(regime1_ics, regime2_ics):
    """
    Test statistical significance of regime differences.
    """
    ic_difference = np.mean(regime2_ics) - np.mean(regime1_ics)
    
    if abs(ic_difference) > 0.02:
        significance = "SIGNIFICANT"
    elif abs(ic_difference) > 0.01:
        significance = "MODERATE"
    else:
        significance = "WEAK"
    
    return significance, ic_difference
```

### Confidence Intervals

```python
def calculate_confidence_intervals(regime1_ic, regime2_ic, n1=15, n2=15):
    """
    Calculate 95% confidence intervals for regime differences.
    """
    se1 = 0.05  # Approximate standard error
    se2 = 0.05
    se_diff = np.sqrt(se1**2 + se2**2)
    
    ci_lower = (regime2_ic - regime1_ic) - 1.96 * se_diff
    ci_upper = (regime2_ic - regime1_ic) + 1.96 * se_diff
    
    return ci_lower, ci_upper
```

## Statistical Framework

### Manual Spearman Correlation

Due to scipy dependency issues, we implement manual Spearman correlation:

```python
def _calculate_spearman_correlation(x, y):
    """
    Calculate Spearman correlation manually to avoid scipy dependency.
    """
    # Get ranks
    x_ranks = x.rank()
    y_ranks = y.rank()
    
    # Calculate correlation using Pearson formula on ranks
    n = len(x)
    if n < 2:
        return 0.0
    
    x_mean = x_ranks.mean()
    y_mean = y_ranks.mean()
    
    numerator = ((x_ranks - x_mean) * (y_ranks - y_mean)).sum()
    x_var = ((x_ranks - x_mean) ** 2).sum()
    y_var = ((y_ranks - y_mean) ** 2).sum()
    
    if x_var == 0 or y_var == 0:
        return 0.0
    
    correlation = numerator / (x_var * y_var) ** 0.5
    return correlation
```

### T-Statistic Calculation

```python
def calculate_t_statistic(ic_values):
    """
    Calculate t-statistic for IC values.
    """
    mean_ic = np.mean(ic_values)
    std_ic = np.std(ic_values, ddof=1)
    n = len(ic_values)
    
    if std_ic == 0:
        return 0.0
    
    t_stat = mean_ic / (std_ic / np.sqrt(n))
    return t_stat
```

### Hit Rate Calculation

```python
def calculate_hit_rate(ic_values):
    """
    Calculate hit rate (percentage of positive IC values).
    """
    positive_ics = sum(1 for ic in ic_values if ic > 0)
    hit_rate = positive_ics / len(ic_values)
    return hit_rate
```

## Implementation Guide

### Running Validation Tests

1. **Basic Validation**:
```bash
python 01_momentum_regime_validation_tests.py
```

2. **Parameter Optimization**:
```bash
python 02_momentum_parameter_optimization.py
```

3. **Weight Optimization**:
```bash
python 03_momentum_weight_optimization.py
```

4. **Transaction Cost Analysis**:
```bash
python 04_momentum_transaction_cost_analysis.py
```

5. **Comprehensive Analysis**:
```bash
python 05_comprehensive_momentum_regime_analysis.py
```

### Configuration

Update configuration files as needed:

```yaml
# config/strategy_config.yml
momentum:
  timeframe_weights:
    "1M": 0.15
    "3M": 0.25
    "6M": 0.30
    "12M": 0.30
  lookback_periods:
    "1M": 1
    "3M": 3
    "6M": 6
    "12M": 12
  skip_months: 1
```

### Data Requirements

1. **Price Data**: `equity_history` table with columns:
   - `ticker`: Stock symbol
   - `date`: Trading date
   - `close`: Adjusted close price

2. **Fundamental Data**: `intermediary_calculations_*` tables with:
   - Sector information
   - Market cap data
   - Financial ratios

3. **Universe Selection**: Top 100 stocks by market cap with minimum liquidity

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError: No module named 'database'**
   - **Solution**: Update `sys.path.append('../../../production')`

2. **scipy Library Loading Error**
   - **Solution**: Use manual Spearman correlation implementation

3. **Empty Results**
   - **Check**: Universe size, data availability, date ranges
   - **Solution**: Verify database connections and data completeness

4. **Poor Performance**
   - **Check**: Factor construction, normalization, skip periods
   - **Solution**: Run parameter optimization

### Performance Optimization

1. **Database Queries**: Use indexed columns and limit date ranges
2. **Memory Usage**: Process data in chunks for large universes
3. **Computation**: Vectorize calculations where possible

### Debugging Tips

1. **Log Analysis**: Check engine logs for data issues
2. **Sample Testing**: Test with small universe first
3. **Data Validation**: Verify price data completeness
4. **Factor Inspection**: Examine raw factor scores distribution

## Best Practices

### Validation Frequency

- **Daily**: Monitor factor performance
- **Weekly**: Check regime stability
- **Monthly**: Full validation analysis
- **Quarterly**: Parameter optimization

### Documentation

- Document all parameter changes
- Track regime shifts over time
- Maintain validation history
- Update quality gates as needed

### Risk Management

- Monitor factor stability
- Implement regime detection
- Use multiple validation metrics
- Consider transaction costs

## References

1. **Academic Literature**:
   - Jegadeesh & Titman (1993): Momentum returns
   - Carhart (1997): Four-factor model
   - Asness et al. (2013): Value and momentum

2. **Implementation Papers**:
   - Fama & French (2015): Five-factor model
   - Novy-Marx (2012): Momentum and mean reversion

3. **Market Regime Literature**:
   - Ang & Bekaert (2002): Regime switching
   - Hamilton (1989): Markov switching models

---

**Contact**: Factor Investing Team  
**Last Updated**: July 30, 2025  
**Next Review**: August 30, 2025 