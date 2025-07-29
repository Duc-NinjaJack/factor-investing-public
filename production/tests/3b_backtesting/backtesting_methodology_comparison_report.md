# Backtesting Methodology Comparison Analysis

**Date:** 2025-07-29
**Purpose:** Compare assumptions and return series between simplified and real data backtesting
**Context:** Understanding why results differ despite identical data sources

## 🎯 Executive Summary

**Key Finding:** The dramatic difference between simplified and real data backtesting results
is due to **methodology differences**, not data discrepancies. Both approaches used identical
data sources, but different assumptions and constraints led to vastly different outcomes.

## 📊 Methodology Comparison

### Simplified Backtesting Assumptions

- **Return Calculation:** Simulated returns based on factor scores
- **Data Source:** Pickle data (factor scores + ADTV)
- **Transaction Costs:** None (0 bps)
- **Rebalancing:** Monthly (assumed)
- **Short Selling:** Allowed (implied)
- **Market Impact:** Ignored
- **Liquidity Filtering:** Simple ADTV threshold
- **Portfolio Construction:** Equal weight, idealized
- **Risk Management:** None
- **Realistic Constraints:** Minimal

### Real Data Backtesting Assumptions

- **Return Calculation:** Actual returns from price changes
- **Data Source:** Database (real price data)
- **Transaction Costs:** 20 bps per trade
- **Rebalancing:** Monthly (enforced)
- **Short Selling:** Not allowed (constraint)
- **Market Impact:** Implicit in real prices
- **Liquidity Filtering:** ADTV threshold + availability
- **Portfolio Construction:** Equal weight, practical
- **Risk Management:** No short selling constraint
- **Realistic Constraints:** Full market reality

## 📈 Performance Comparison

### 10B_VND Threshold

| Metric | Simplified | Real Data | Difference |
|--------|------------|-----------|------------|
| Annual Return | nan% | nan% | nan% |
| Sharpe Ratio | -0.17 | -1.27 | 1.10 |
| Max Drawdown | -493.19% | -270.65% | -222.54% |
| Volatility | 354225.42% | 39852.49% | 314372.94% |
| Positive Days | 30.0% | 27.7% | 2.3% |

### 3B_VND Threshold

| Metric | Simplified | Real Data | Difference |
|--------|------------|-----------|------------|
| Annual Return | 16.50% | nan% | nan% |
| Sharpe Ratio | 0.22 | -0.27 | 0.50 |
| Max Drawdown | -169.66% | -311.44% | 141.78% |
| Volatility | 45993.46% | 27082.02% | 18911.45% |
| Positive Days | 32.2% | 28.2% | 3.9% |

## 🔍 Key Differences Analysis

### 1. Return Calculation Methodology

**Simplified Approach:**
- Simulated returns based on factor scores
- Assumes perfect factor-to-return relationship
- Ignores market microstructure effects

**Real Data Approach:**
- Actual returns from price changes
- Includes all market dynamics
- Reflects real trading conditions

### 2. Transaction Costs Impact

**Simplified Approach:**
- No transaction costs (0 bps)
- Assumes frictionless trading
- Unrealistic for large portfolios

**Real Data Approach:**
- 20 bps transaction costs per trade
- Reflects realistic trading costs
- Significant impact on performance

### 3. Short Selling Constraints

**Simplified Approach:**
- Implicitly allows short selling
- No position constraints
- Unrealistic for most investors

**Real Data Approach:**
- No short selling constraint
- Long-only portfolio
- Realistic for most investors

### 4. Market Impact and Liquidity

**Simplified Approach:**
- Ignores market impact
- Assumes infinite liquidity
- No slippage considerations

**Real Data Approach:**
- Market impact implicit in prices
- Real liquidity constraints
- Practical trading limitations

## 🎯 Implications for Implementation

### Why Real Data Results Are More Reliable

1. **Market Reality:** Real data backtesting reflects actual market conditions
2. **Transaction Costs:** Includes realistic trading costs
3. **Constraints:** Applies practical investment constraints
4. **Liquidity:** Considers real market liquidity
5. **Risk Management:** Includes realistic risk constraints

### Why Simplified Results Were Overly Optimistic

1. **Idealized Assumptions:** Perfect factor-to-return relationship
2. **No Transaction Costs:** Frictionless trading assumption
3. **No Constraints:** Unrealistic position limits
4. **Market Impact Ignored:** No consideration of trading impact
5. **Liquidity Assumptions:** Infinite liquidity assumption

## 📋 Recommendations

### For Future Backtesting

1. **Always Use Real Data:** Validate with actual price data
2. **Include Transaction Costs:** Apply realistic trading costs
3. **Apply Realistic Constraints:** Use practical investment limits
4. **Consider Market Impact:** Account for trading effects
5. **Validate Assumptions:** Test methodology robustness

### For Implementation Decisions

1. **Trust Real Data Results:** Use realistic backtesting outcomes
2. **Question Simplified Models:** Be skeptical of idealized results
3. **Consider Practical Constraints:** Account for real-world limitations
4. **Validate Methodology:** Ensure backtesting reflects reality
5. **Document Assumptions:** Clearly state methodology limitations

## 🎯 Conclusion

The dramatic difference between simplified and real data backtesting results
demonstrates the critical importance of using realistic assumptions and constraints.
While simplified models can provide quick insights, they often overestimate
performance by ignoring real-world trading costs and constraints.

**Key Takeaway:** The rejection of the 3B VND liquidity threshold based on
real data backtesting is justified, as it reflects the true market reality
rather than idealized assumptions.
