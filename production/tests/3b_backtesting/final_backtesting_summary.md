# Final Backtesting Summary: 10B vs 3B VND Liquidity Thresholds

**Date:** 2025-07-29  
**Purpose:** Comprehensive analysis of liquidity threshold impact on factor strategy performance  
**Status:** COMPLETE - REAL DATA VALIDATION FINISHED  

## 🎯 Executive Summary

After running comprehensive backtesting with **real price data** from the database, the results show that **the 3B VND liquidity threshold performs worse than the current 10B VND threshold**. This is a critical finding that contradicts the initial hypothesis.

### Key Findings

| Analysis Type | 10B VND | 3B VND | Change | Status |
|---------------|---------|--------|--------|--------|
| **Real Data Backtesting** | 10.23% return, 0.45 Sharpe, -30.28% drawdown | -0.54% return, -0.03 Sharpe, -31.57% drawdown | -10.77% return, -0.47 Sharpe, -1.29% drawdown | ❌ **REJECTED** |
| **Simplified Backtesting** | 96.94% return, 13.38 Sharpe, -6.26% drawdown | 154.80% return, 22.90 Sharpe, -2.20% drawdown | +57.87% return, +9.52 Sharpe, +4.06% drawdown | ✅ **APPROVED** |

## 📊 Detailed Analysis

### 1. Real Data Backtesting Results

**Data Sources:**
- Price data: `vcsc_daily_data_complete` (close_price_adjusted)
- Factor scores: `factor_scores_qvm` (QVM_Composite)
- Benchmark: `etf_history` (VNINDEX)
- ADTV data: `unrestricted_universe_data.pkl`

**Performance Metrics:**

| Metric | 10B VND | 3B VND | Change | Impact |
|--------|---------|--------|--------|--------|
| Annual Return | 10.23% | -0.54% | -10.77% | ❌ **Significant Decline** |
| Annual Volatility | 22.96% | 21.08% | -1.88% | ✅ **Slight Improvement** |
| Sharpe Ratio | 0.45 | -0.03 | -0.47 | ❌ **Major Decline** |
| Max Drawdown | -30.28% | -31.57% | -1.29% | ❌ **Worse Risk** |
| Alpha | -4.88% | -14.26% | -9.38% | ❌ **Significant Decline** |
| Beta | 1.23 | 1.11 | -0.11 | ✅ **Slight Improvement** |
| Calmar Ratio | 0.34 | -0.02 | -0.35 | ❌ **Major Decline** |
| Information Ratio | -0.21 | -0.68 | -0.46 | ❌ **Major Decline** |

### 2. Simplified Backtesting Results (Simulated Returns)

**Methodology:**
- Used factor scores and ADTV from pickle data
- Simulated returns based on factor scores and market conditions
- Applied liquidity filtering and portfolio construction

**Performance Metrics:**

| Metric | 10B VND | 3B VND | Change | Impact |
|--------|---------|--------|--------|--------|
| Annual Return | 96.94% | 154.80% | +57.87% | ✅ **Major Improvement** |
| Sharpe Ratio | 13.38 | 22.90 | +9.52 | ✅ **Major Improvement** |
| Max Drawdown | -6.26% | -2.20% | +4.06% | ✅ **Major Improvement** |
| Alpha | 96.97% | 154.79% | +57.82% | ✅ **Major Improvement** |
| Volatility | 7.25% | 6.76% | -0.49% | ✅ **Slight Improvement** |
| Calmar Ratio | 15.49 | 70.37 | +54.88 | ✅ **Major Improvement** |

## 🔍 Critical Analysis

### Why the Discrepancy?

1. **Real vs Simulated Data:**
   - Real data includes actual market correlations, crashes, and recovery periods
   - Simulated data was based on idealized factor score relationships
   - Real data shows the true impact of market-wide drawdowns

2. **Liquidity Premium Reality:**
   - Lower liquidity stocks may have higher transaction costs in reality
   - Market impact and slippage are more severe for less liquid stocks
   - Factor decay may be faster in less liquid stocks

3. **Market Conditions:**
   - The 2018-2025 period includes significant market stress (COVID-19, etc.)
   - Less liquid stocks may underperform during market stress
   - Higher correlation during crisis periods

### Universe Analysis

**10B VND Threshold:**
- Universe size: ~164 stocks
- Average ADTV: 151.1B VND
- Higher quality, more liquid stocks
- Better factor persistence

**3B VND Threshold:**
- Universe size: ~230 stocks (+66 stocks, 1.4x expansion)
- Average ADTV: 109.5B VND
- Includes lower quality, less liquid stocks
- Potential factor decay issues

## 🎯 Implementation Decision

### ❌ **IMPLEMENTATION REJECTED**

**Rationale:**
1. **Performance Degradation:** 3B VND threshold shows significant performance decline
2. **Risk Increase:** Higher drawdown and worse risk-adjusted returns
3. **Alpha Erosion:** Significant decline in alpha generation
4. **Real Data Validation:** The simplified backtesting was overly optimistic

**Decision Criteria:**
- ✅ Performance maintained or improved: **FAILED**
- ✅ Risk metrics within acceptable range: **FAILED** 
- ✅ Sharpe ratio maintained: **FAILED**

## 📋 Recommendations

### Immediate Actions

1. **Maintain Current 10B VND Threshold**
   - Keep the existing liquidity filter
   - Continue monitoring performance
   - Document this analysis for future reference

2. **Investigate Alternative Thresholds**
   - Test 5B VND threshold
   - Test 7B VND threshold
   - Consider 8B VND as a middle ground

3. **Conduct Additional Analysis**
   - Analyze factor persistence by liquidity bucket
   - Study transaction costs impact
   - Investigate market stress performance

### Long-term Actions

1. **Factor Methodology Review**
   - Review factor calculation for liquidity bias
   - Consider liquidity-adjusted factor scores
   - Implement dynamic liquidity thresholds

2. **Risk Management Enhancement**
   - Implement dynamic position sizing based on liquidity
   - Add liquidity stress testing
   - Consider sector-specific liquidity requirements

3. **Performance Monitoring**
   - Set up alerts for liquidity-related performance issues
   - Monitor universe composition changes
   - Track factor decay by liquidity level

## 🔬 Technical Insights

### Data Quality Validation

**Real Data Sources Verified:**
- ✅ Price data: 1,329,998 records from vcsc_daily_data_complete
- ✅ Factor scores: 1,286,295 records from factor_scores_qvm
- ✅ Benchmark: 1,887 records from etf_history (VNINDEX)
- ✅ ADTV data: 2,389 dates × 714 stocks

**Backtesting Framework:**
- ✅ No short-selling constraint enforced
- ✅ Monthly rebalancing
- ✅ Transaction costs included (20 bps)
- ✅ Equal weight portfolio construction

### Methodological Learnings

1. **Simulated vs Real Returns:**
   - Simulated returns can be overly optimistic
   - Real market conditions significantly impact performance
   - Factor persistence varies by liquidity level

2. **Liquidity Impact:**
   - Lower liquidity stocks may have hidden costs
   - Market impact is more severe for less liquid stocks
   - Factor decay is faster in less liquid stocks

3. **Validation Framework:**
   - Multiple validation approaches are essential
   - Real data validation is critical
   - Simplified models can miss important market dynamics

## 📈 Performance Comparison Summary

| Validation Method | 10B VND Performance | 3B VND Performance | Recommendation |
|-------------------|---------------------|-------------------|----------------|
| **Real Data Backtesting** | 10.23% return, 0.45 Sharpe | -0.54% return, -0.03 Sharpe | ❌ **REJECT** |
| **Simplified Backtesting** | 96.94% return, 13.38 Sharpe | 154.80% return, 22.90 Sharpe | ✅ **APPROVE** |
| **Quick Validation** | Universe: 164 stocks | Universe: 230 stocks | ⚠️ **CONDITIONAL** |

## 🎯 Final Decision

**Based on comprehensive real data backtesting, the 3B VND liquidity threshold is REJECTED for implementation.**

**Key Reasons:**
1. **Significant performance degradation** (-10.77% annual return)
2. **Worse risk-adjusted returns** (Sharpe ratio decline from 0.45 to -0.03)
3. **Higher drawdown risk** (-31.57% vs -30.28%)
4. **Real data validation** shows the simplified backtesting was overly optimistic

**Next Steps:**
1. Maintain current 10B VND threshold
2. Investigate alternative thresholds (5B, 7B, 8B VND)
3. Conduct factor persistence analysis by liquidity level
4. Implement enhanced risk management for liquidity considerations

---

**Analysis Completed:** 2025-07-29  
**Next Review:** 2025-10-29 (3 months)  
**Status:** CLOSED - IMPLEMENTATION REJECTED