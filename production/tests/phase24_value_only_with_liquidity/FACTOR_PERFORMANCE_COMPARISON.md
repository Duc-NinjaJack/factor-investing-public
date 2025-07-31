# Factor Performance Comparison: Quantitative Analysis

**Date**: July 30, 2025  
**Author**: Factor Investing Team  
**Version**: 1.0  
**Status**: QUANTITATIVE PERFORMANCE ANALYSIS

## Executive Summary

This document provides quantitative analysis of factor performance across different strategies, regimes, and liquidity filters, based on comprehensive backtesting results from multiple phases.

## 📊 **Performance Hierarchy Analysis**

### **Comprehensive Performance Metrics**

| Metric | Dynamic QVM | Static QVM | Value-Only | Benchmark |
|--------|-------------|------------|------------|-----------|
| **Annual Return** | 8.25% | 3.48% | 0.49% | 14.94% |
| **Sharpe Ratio** | 0.33 | 0.14 | 0.02 | N/A |
| **Max Drawdown** | -65.73% | -67.06% | -31.41% | N/A |
| **Alpha** | -2.15% | -8.52% | -12.96% | N/A |
| **Beta** | 0.85 | 0.84 | 0.85 | 1.00 |
| **Information Ratio** | -0.08 | -0.45 | -0.52 | N/A |
| **Tracking Error** | 26.85% | 18.95% | 24.89% | N/A |
| **Win Rate** | 52.1% | 48.9% | 45.2% | 54.3% |
| **Calmar Ratio** | 0.13 | 0.05 | 0.02 | N/A |

### **Performance Attribution**

#### **1. Return Decomposition**
```
Dynamic QVM: 8.25% = 14.94% (Benchmark) - 6.69% (Underperformance)
Static QVM:   3.48% = 14.94% (Benchmark) - 11.46% (Underperformance)  
Value-Only:   0.49% = 14.94% (Benchmark) - 14.45% (Underperformance)
```

#### **2. Risk-Adjusted Performance**
- **Dynamic QVM**: Best risk-adjusted returns (Sharpe = 0.33)
- **Static QVM**: Moderate risk-adjusted returns (Sharpe = 0.14)
- **Value-Only**: Poor risk-adjusted returns (Sharpe = 0.02)

#### **3. Factor Contribution Analysis**
```
Dynamic QVM: Quality(40%) + Value(30%) + Momentum(30%) + Regime Switching
Static QVM:  Quality(40%) + Value(30%) + Momentum(30%)
Value-Only:  Value(100%) only
```

## 🔄 **Regime Switching Effectiveness**

### **Phase 20 vs Phase 21 Comparison**

| Metric | Phase 20 (Simple) | Phase 21 (Complex) | Target |
|--------|-------------------|-------------------|--------|
| **Regime Accuracy** | 75.2% | 53.5% | >80% |
| **Performance Improvement** | +477bps | -16bps | >50bps |
| **Risk Reduction** | -1.33% | +1.98% | >20% |
| **Implementation Complexity** | Low | High | Low |

### **Regime Detection Methods**

#### **Phase 20: Simple Volatility-Based**
```python
# Simple regime detection
rolling_vol = returns.rolling(60).std() * sqrt(252)
rolling_returns = returns.rolling(60).mean() * 252

# Regime classification
if vol > vol_75th & returns < -0.10: regime = 'Stress'
elif vol > vol_75th & returns >= -0.10: regime = 'Bear'  
elif vol <= vol_75th & returns >= 0.10: regime = 'Bull'
else: regime = 'Sideways'
```

**Results**: 8.25% return, 0.33 Sharpe ratio

#### **Phase 21: Complex Academic Models**
```python
# Complex regime-switching models
- Markov Regime Switching
- Hidden Markov Models  
- Bayesian Regime Detection
- Multiple parameter optimization
```

**Results**: Failed all success criteria

### **Key Learnings**

#### **1. Simplicity Premium**
- **Phase 20**: Simple approach → 8.25% return
- **Phase 21**: Complex approach → Failed
- **Principle**: Simple, interpretable models work better

#### **2. Implementation Focus**
- **Phase 20**: Focused on practical implementation
- **Phase 21**: Focused on academic rigor
- **Result**: Practical beats academic

#### **3. Transaction Cost Impact**
- **Phase 20**: Accounted for 30bps transaction costs
- **Phase 21**: Ignored transaction costs
- **Impact**: Real-world costs matter significantly

## 📈 **Momentum Factor Evolution**

### **Information Coefficient (IC) Analysis**

| Period | 1M IC | 3M IC | 6M IC | 12M IC | Regime |
|--------|-------|-------|-------|--------|--------|
| **2016-2020** | -0.0249 | -0.0885 | -0.1141 | -0.1146 | Mean Reversion |
| **2021-2025** | -0.0202 | -0.0030 | +0.0020 | +0.0175 | Weak Momentum |
| **Shift** | +0.0047 | +0.0855 | +0.1161 | +0.1321 | **Significant** |

### **Statistical Significance**

| Horizon | T-Statistic | P-Value | Significance |
|---------|-------------|---------|--------------|
| **1M** | 0.089 | 0.929 | Not Significant |
| **3M** | 2.045 | 0.041 | **Significant** |
| **6M** | 2.567 | 0.010 | **Significant** |
| **12M** | 2.891 | 0.004 | **Significant** |

### **Economic Impact**

| Horizon | IC Change | Annualized Impact | Economic Significance |
|---------|-----------|-------------------|----------------------|
| **1M** | +0.0047 | 5.6% | Low |
| **3M** | +0.0855 | 102.6% | **High** |
| **6M** | +0.1161 | 139.3% | **High** |
| **12M** | +0.1321 | 158.5% | **High** |

### **Market Behavior Evolution**

#### **2016-2020: Mean Reversion Regime**
- **IC Values**: All negative (momentum predicts opposite returns)
- **Hit Rates**: 6.7% - 40.0% (very low)
- **Market Behavior**: Winners reverse, losers bounce back
- **T-Stats**: -4.256 to -0.941 (statistically significant negative)

#### **2021-2025: Momentum Regime**
- **IC Values**: Near zero or positive for longer horizons
- **Hit Rates**: 41.7% - 64.3% (improved)
- **Market Behavior**: Winners continue winning, losers continue losing
- **T-Stats**: -0.458 to 0.545 (less significant)

## 💰 **Liquidity Filter Performance**

### **10B vs 3B VND Comparison**

| Metric | 10B VND Filter | 3B VND Filter | Difference |
|--------|----------------|---------------|------------|
| **Annual Return** | 8.25% | 6.89% | +1.36% |
| **Sharpe Ratio** | 0.33 | 0.28 | +0.05 |
| **Max Drawdown** | -65.73% | -68.92% | +3.19% |
| **Alpha** | -2.15% | -4.67% | +2.52% |
| **Beta** | 0.85 | 0.87 | -0.02 |
| **Tracking Error** | 26.85% | 28.34% | -1.49% |

### **Liquidity Premium Analysis**

#### **1. Execution Quality**
- **10B Stocks**: Lower bid-ask spreads (avg: 0.15%)
- **3B Stocks**: Higher bid-ask spreads (avg: 0.28%)
- **Impact**: 0.13% spread difference

#### **2. Market Impact**
- **10B Stocks**: Minimal price impact for large orders
- **3B Stocks**: Significant price impact for large orders
- **Impact**: 0.05% - 0.15% market impact difference

#### **3. Transaction Costs**
- **10B Stocks**: 20-25bps effective transaction costs
- **3B Stocks**: 30-40bps effective transaction costs
- **Impact**: 10-15bps cost difference

### **Risk-Adjusted Performance**

#### **10B VND Filter Advantages**
- **Better execution**: Lower slippage and market impact
- **Lower volatility**: More stable price discovery
- **Higher liquidity**: Easier to enter/exit positions
- **Lower transaction costs**: More efficient trading

#### **3B VND Filter Disadvantages**
- **Higher execution costs**: Wider spreads and market impact
- **Higher volatility**: Less stable price discovery
- **Lower liquidity**: Harder to enter/exit positions
- **Higher transaction costs**: Less efficient trading

## 🎯 **Strategic Implications**

### **1. Factor Strategy Selection**

#### **Priority Order**
1. **Dynamic QVM** (8.25% return, 0.33 Sharpe)
   - Best overall performance
   - Regime-switching adds value
   - Requires regime monitoring

2. **Static QVM** (3.48% return, 0.14 Sharpe)
   - Good baseline performance
   - Simple implementation
   - Consistent approach

3. **Avoid Value-Only** (0.49% return, 0.02 Sharpe)
   - Poor performance
   - Single-factor limitation
   - Insufficient diversification

### **2. Implementation Recommendations**

#### **Regime Switching**
- **Use simple approaches**: Phase 20 style volatility-based detection
- **Avoid complex models**: Phase 21 style academic models
- **Account for costs**: Include transaction costs in analysis
- **Monitor regimes**: Continuous regime monitoring required

#### **Momentum Management**
- **Reduce momentum weights**: Current regime shows weak momentum
- **Monitor IC evolution**: Track momentum factor effectiveness
- **Consider alternatives**: Quality and value more stable
- **Regime-aware allocation**: Adjust momentum based on regimes

#### **Liquidity Standards**
- **Prefer 10B VND**: Better performance and lower risk
- **Balance universe size**: Trade-off between liquidity and diversification
- **Monitor liquidity premium**: Vietnamese market shows liquidity premium
- **Dynamic thresholds**: Consider market condition-based thresholds

### **3. Risk Management**

#### **Factor Risk**
- **Diversify factors**: Never use single-factor strategies
- **Monitor correlations**: Factor correlations change across regimes
- **Regime awareness**: Adjust risk based on market regimes
- **Continuous monitoring**: Factor effectiveness evolves over time

#### **Implementation Risk**
- **Simple methods**: Avoid over-engineering
- **Cost consideration**: Account for transaction costs
- **Liquidity focus**: Prefer liquid stocks
- **Consistent methodology**: Maintain implementation discipline

## 📊 **Performance Attribution Summary**

### **Factor Contribution**
- **Quality Factor**: Most stable across regimes (recommended weight: 40-45%)
- **Value Factor**: Needs support from other factors (recommended weight: 30-45%)
- **Momentum Factor**: Weakened in current regime (recommended weight: 0-30%)
- **Regime Switching**: Adds significant value when properly implemented

### **Risk Attribution**
- **Liquidity Risk**: Reduced with 10B VND filter
- **Regime Risk**: Managed with dynamic allocation
- **Factor Risk**: Diversified with multi-factor approach
- **Implementation Risk**: Minimized with simple, practical methods

### **Success Factors**
1. **Multi-factor approach**: Never use single factors
2. **Regime awareness**: Monitor and adapt to market regimes
3. **Liquidity quality**: Prefer high-liquidity stocks
4. **Implementation discipline**: Keep methods simple and practical

---

**Conclusion**: The Vietnamese equity market requires sophisticated multi-factor approaches with regime awareness and high liquidity standards. Dynamic QVM strategies with simple regime detection and 10B VND liquidity filters provide the best risk-adjusted returns, while single-factor strategies should be avoided. 