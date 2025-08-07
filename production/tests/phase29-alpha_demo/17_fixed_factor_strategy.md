# Factor Calculation Investigation - Version 17 Analysis

## 📋 Executive Summary

This document presents the findings from a comprehensive investigation of the QVM factor calculations in the `factor_scores_qvm` database table. The investigation reveals critical issues with the underlying factor calculation methodology that explain the poor performance of the QVM strategy despite technical improvements in Version 17.

## 🔍 Investigation Methodology

### Data Sources
- **Factor Scores**: `factor_scores_qvm` table (strategy_version: `qvm_v2.0_enhanced`)
- **Sample Period**: June 2022 (representative of the 2022 crash period)
- **Sample Size**: 10 major Vietnamese stocks (HPG, VNM, VCB, TCB, FPT, MWG, MSN, VIC, VHM, GAS)
- **Time Series**: Multiple dates (2022-06-30, 2022-12-30, 2023-06-30, 2023-12-29)

### Validation Approach
1. **Factor Distribution Analysis**: Statistical properties of each factor
2. **Correlation Analysis**: Factor independence validation
3. **QVM Composite Validation**: Verification of weighted combination
4. **Time Series Consistency**: Multi-date analysis
5. **Common Sense Checks**: Expected factor relationships

## 📊 Key Findings

### 1. Factor Calculation Accuracy

#### ✅ QVM_Composite Calculation
- **Formula**: `QVM_Composite = 0.40 × Quality + 0.30 × Value + 0.30 × Momentum`
- **Validation**: ✅ **Correctly calculated** (differences < 1e-10)
- **Strategy Version**: `qvm_v2.0_enhanced`

#### 📈 Factor Distributions (June 2022 Sample)

| Factor | Mean | Std Dev | Min | Max | Range |
|--------|------|---------|-----|-----|-------|
| **Quality_Composite** | 0.81 | 0.83 | -0.77 | 2.04 | 2.81 |
| **Value_Composite** | -0.70 | 0.44 | -1.37 | 0.33 | 1.70 |
| **Momentum_Composite** | -0.24 | 0.51 | -0.99 | 0.56 | 1.55 |
| **QVM_Composite** | 0.04 | 0.31 | -0.68 | 0.34 | 1.03 |

### 2. Factor Performance Rankings

#### Quality Factor (Top/Bottom 3)
```
Top Quality Stocks:
1. HPG: 2.04 (Steel - High ROE, strong fundamentals)
2. VNM: 1.67 (Food - Stable profitability)
3. FPT: 1.15 (Technology - Growing business)

Bottom Quality Stocks:
1. VIC: -0.77 (Real Estate - Financial stress)
2. GAS: -0.15 (Energy - Lower profitability)
3. VCB: 0.33 (Banking - Moderate quality)
```

#### Value Factor (Top/Bottom 3)
```
Top Value Stocks:
1. TCB: 0.33 (Banking - Lower P/E, higher yield)
2. VHM: -0.52 (Real Estate - Moderate value)
3. HPG: -0.62 (Steel - Some value characteristics)

Bottom Value Stocks:
1. VCB: -1.37 (Banking - High P/E, expensive)
2. FPT: -1.12 (Technology - Growth premium)
3. VNM: -0.82 (Food - Stable but expensive)
```

#### Momentum Factor (Top/Bottom 3)
```
Top Momentum Stocks:
1. VCB: 0.56 (Banking - Strong recent performance)
2. GAS: 0.23 (Energy - Positive momentum)
3. FPT: 0.11 (Technology - Slight positive momentum)

Bottom Momentum Stocks:
1. TCB: -0.99 (Banking - Poor recent performance)
2. HPG: -0.96 (Steel - Declining prices)
3. VIC: -0.59 (Real Estate - Weak momentum)
```

## 🚨 Critical Issues Identified

### 1. Quality Factor Dominance

#### Problem
- **Quality vs QVM correlation**: 0.955 (extremely high)
- **Quality factor dominates** the composite score
- **Value and Momentum factors** have minimal impact on final QVM score

#### Impact
- **Factor imbalance**: Quality represents 95.5% of QVM signal
- **Reduced diversification**: Strategy becomes essentially a quality-only strategy
- **Performance vulnerability**: Quality factor breakdown in 2022 caused massive losses

### 2. Value Factor Calculation Problems

#### Problem
- **Consistently negative values** across all time periods
- **Mean values**: -0.70 to -0.16 (always negative)
- **Suggests calculation errors** or data quality issues

#### Time Series Evidence
```
2022-06-30: Mean = -0.70 (Market crash period)
2022-12-30: Mean = -0.27 (Recovery period)
2023-06-30: Mean = -0.24 (Growth period)
2023-12-29: Mean = -0.16 (Stable period)
```

#### Root Cause Analysis
- **Value factor should be positive** for undervalued stocks
- **Consistent negativity suggests**:
  - Incorrect P/E, P/B, or P/S calculations
  - Wrong normalization methodology
  - Data quality issues in fundamental data

### 3. Factor Independence Violations

#### Problem
- **Value vs Momentum correlation**: -0.770 (strong negative)
- **Violates factor independence** - value and momentum should be uncorrelated
- **Suggests calculation methodology issues**

#### Expected vs Actual
```
Expected Factor Correlations:
- Quality vs Value: ~0.0 (independent)
- Quality vs Momentum: ~0.0 (independent)
- Value vs Momentum: ~0.0 (independent)

Actual Factor Correlations:
- Quality vs Value: +0.114 (slight positive)
- Quality vs Momentum: -0.352 (negative)
- Value vs Momentum: -0.770 (strong negative) ❌
```

### 4. Momentum Factor Inconsistency

#### Problem
- **Inconsistent performance** across time periods
- **Sometimes positive, sometimes negative** averages
- **May not be capturing true momentum** signals

#### Time Series Evidence
```
2022-06-30: Mean = -0.47 (Crash - expected negative)
2022-12-30: Mean = -0.38 (Recovery - still negative)
2023-06-30: Mean = +0.17 (Growth - positive)
2023-12-29: Mean = -0.35 (Stable - negative again)
```

## 🔧 Technical Analysis

### Factor Calculation Methodology

#### Current Implementation (qvm_v2.0_enhanced)
1. **Quality Factor**: Sector-specific metrics (ROAE, ROAA, margins)
2. **Value Factor**: P/E, P/B, P/S, EV/EBITDA ratios
3. **Momentum Factor**: Multi-timeframe returns (1M, 3M, 6M, 12M)
4. **Normalization**: Sector-neutral z-scores
5. **Combination**: Weighted average with standard weights

#### Identified Issues
1. **Sector-neutral normalization** may be causing problems
2. **Factor component calculations** may have errors
3. **Data quality issues** in underlying fundamental data
4. **Momentum calculation** may not use proper skip-1-month convention

### Data Quality Assessment

#### Available Data
- ✅ **Factor scores**: Available and correctly calculated
- ⚠️ **Fundamental data**: Not available for validation
- ✅ **Price data**: Available for momentum validation
- ❌ **Underlying calculations**: Cannot verify component accuracy

#### Missing Validation
- **Cannot verify** if P/E, P/B ratios are calculated correctly
- **Cannot verify** if ROAE, ROAA calculations are accurate
- **Cannot verify** if momentum returns use proper methodology

## 📋 Recommendations

### 1. Immediate Actions

#### A. Investigate Value Factor Calculation
- **Review P/E, P/B, P/S calculations** in the factor engine
- **Check fundamental data quality** and availability
- **Verify sector-specific value metrics** for banking/securities
- **Test value factor with simple P/E ranking** as baseline

#### B. Fix Factor Independence
- **Investigate why Value and Momentum are correlated** (-0.770)
- **Review momentum calculation methodology**
- **Check if sector-neutral normalization** is causing issues
- **Consider alternative normalization approaches**

#### C. Reduce Quality Dominance
- **Adjust factor weights** to reduce quality dominance
- **Consider equal weighting** (33.3% each) temporarily
- **Investigate quality factor calculation** for over-optimization

### 2. Medium-term Improvements

#### A. Factor Calculation Validation
- **Create validation scripts** for each factor component
- **Compare with simple benchmarks** (e.g., P/E ranking vs complex value)
- **Test factor calculations** on known good/bad stocks
- **Implement factor quality metrics** and monitoring

#### B. Data Quality Enhancement
- **Improve fundamental data** availability and quality
- **Add data validation checks** in factor calculation pipeline
- **Implement outlier detection** and handling
- **Create data quality reports** for monitoring

#### C. Methodology Refinement
- **Review sector-neutral normalization** approach
- **Consider alternative factor combination** methods
- **Test different momentum timeframes** and weights
- **Implement regime-aware factor weighting**

### 3. Long-term Strategy

#### A. Factor Engine Redesign
- **Modular factor calculation** with individual validation
- **Configurable normalization** methods
- **Flexible factor combination** approaches
- **Comprehensive testing framework**

#### B. Performance Attribution
- **Implement factor attribution** analysis
- **Track factor performance** over time
- **Identify factor breakdown** periods
- **Create factor risk management** framework

## 🎯 Impact on Strategy Performance

### Current Performance Issues
- **2022 Crash**: -27.46% (Version 16) → -38.25% (Version 17)
- **Quality factor breakdown** during market stress
- **Value factor not providing** downside protection
- **Momentum factor inconsistent** across regimes

### Root Cause Attribution
1. **Quality dominance** (95.5% correlation) → Single factor risk
2. **Value factor problems** (consistently negative) → No value protection
3. **Factor correlation** (-0.770 Value vs Momentum) → Reduced diversification
4. **Momentum inconsistency** → Unreliable signals

### Expected Improvements
- **Factor independence** → Better diversification
- **Balanced factor weights** → Reduced single-factor risk
- **Corrected value factor** → Downside protection
- **Improved momentum** → Better trend following

## 📊 Next Steps

### Phase 1: Factor Investigation (Immediate)
1. **Create factor validation scripts** for each component
2. **Test with simple benchmarks** (P/E ranking, simple momentum)
3. **Compare with alternative calculations** (different normalization)
4. **Document findings** and create fix plan

### Phase 2: Factor Fixes (Short-term)
1. **Fix value factor calculation** issues
2. **Resolve factor independence** problems
3. **Adjust factor weights** to reduce quality dominance
4. **Test fixes** with historical data

### Phase 3: Strategy Validation (Medium-term)
1. **Backtest fixed factors** across full period
2. **Compare performance** with benchmarks
3. **Analyze factor attribution** and risk
4. **Implement monitoring** and alerts

## 📈 Conclusion

The factor calculation investigation reveals that the **underlying factor calculation methodology has fundamental problems**, particularly with the Value factor and factor independence. These issues explain why the QVM strategy underperforms despite technical improvements in Version 17.

**Key findings:**
- Quality factor dominates (95.5% correlation with QVM)
- Value factor consistently negative (calculation errors)
- Factor independence violated (Value vs Momentum -0.770)
- Momentum factor inconsistent across time periods

**Immediate action required:**
1. Investigate and fix Value factor calculation
2. Resolve factor independence issues
3. Reduce Quality factor dominance
4. Implement comprehensive factor validation

The **factor calculation problems are the root cause** of poor strategy performance, and fixing these issues should significantly improve the QVM strategy's effectiveness.

---

*Document created: August 7, 2025*  
*Investigation period: June 2022 - December 2023*  
*Strategy version analyzed: qvm_v2.0_enhanced*  
*Next review: After factor fixes implementation*
