# 2018 Underperformance Analysis - Root Cause Investigation

## Executive Summary

The QVM strategy experienced significant underperformance in 2018, with the equity curve dropping below its starting value and showing a deep drawdown of approximately -20%. This analysis identifies the root causes and provides actionable recommendations for improvement.

## Key Findings

### 1. Market Conditions in 2018
- **Total trading days**: 248 days
- **Average rolling return**: 12.8% (positive but volatile)
- **Average rolling volatility**: 21.4%
- **Market behavior**: Started strong (Jan-Mar) but declined significantly from April onwards

### 2. Regime Detection Analysis

#### Regime Distribution:
- **Correction**: 105 days (42.3%) - HIGH BEAR MARKET EXPOSURE
- **Growth**: 86 days (34.7%)
- **Bull**: 57 days (23.0%)
- **Crisis**: 0 days (0.0%) - NO CRISIS DETECTED

#### Critical Issues:
- **42.3% of days in bear markets** (correction regime)
- **No crisis regime detected** despite severe market stress
- **Only 3 regime changes** throughout the year (excessive smoothing)
- **Average regime duration**: 62 days (too long for rapid market changes)

### 3. Threshold Effectiveness Analysis

#### Current Thresholds:
- **Return thresholds**: Strong+ (15%), Mod+ (5%), Mod- (-5%), Strong- (-15%)
- **Volatility thresholds**: Low (20%), Medium (30%), High (40%)

#### Threshold Problems in 2018:
- **85 days with rolling_return < -15%** but no crisis detected
- **114 days with rolling_return < -5%** but limited correction detection
- **0 days with rolling_vol > 40%** - volatility threshold too high
- **0 days with rolling_vol > 30%** - medium threshold also too high

### 4. Monthly Performance Breakdown

| Month | Regime | Allocation | Avg Return | Avg Vol | Monthly Return | Cumulative |
|-------|--------|------------|------------|---------|----------------|------------|
| Jan   | Bull   | 100%       | 80.5%      | 12.7%   | 12.2%          | 12.2%      |
| Feb   | Bull   | 100%       | 80.6%      | 18.9%   | 1.4%           | 13.8%      |
| Mar   | Bull   | 100%       | 88.2%      | 21.5%   | 4.7%           | 19.2%      |
| Apr   | Growth | 90%        | 60.5%      | 22.6%   | -10.9%         | 6.3%       |
| May   | Growth | 90%        | 14.0%      | 26.1%   | -7.4%          | -1.6%      |
| Jun   | Growth | 90%        | -14.2%     | 27.4%   | -0.9%          | -2.5%      |
| Jul   | Growth | 90%        | -52.0%     | 25.7%   | -0.2%          | -2.7%      |
| Aug   | Correction | 50%    | -45.8%     | 25.2%   | 3.5%           | 0.7%       |
| Sep   | Correction | 50%    | -9.7%      | 21.5%   | 2.8%           | 3.5%       |
| Oct   | Correction | 50%    | -8.0%      | 18.8%   | -10.3%         | -7.2%      |
| Nov   | Correction | 50%    | -1.1%      | 17.4%   | 1.4%           | -5.9%      |
| Dec   | Correction | 50%    | -7.8%      | 17.1%   | -3.6%          | -9.3%      |

## Root Cause Analysis

### Primary Issues:

1. **Delayed Regime Detection**
   - 90-day lookback period too long for rapid market changes
   - April-July period should have been classified as correction/crisis earlier
   - Strategy remained in "growth" regime during significant declines

2. **Overly Conservative Thresholds**
   - Volatility thresholds (30%, 40%) too high for Vietnamese market
   - Return thresholds (-5%, -15%) not sensitive enough
   - No crisis regime detected despite 85 days with < -15% returns

3. **Excessive Smoothing**
   - Only 3 regime changes in 248 days
   - 30-day minimum regime duration prevents quick defensive moves
   - Smoothing logic eliminates important regime signals

4. **Insufficient Defensive Positioning**
   - 50% allocation during corrections may not be defensive enough
   - No crisis regime allocation (30%) was ever triggered
   - Momentum factor (20%) still active during corrections

### Secondary Issues:

5. **Factor Weight Problems**
   - Momentum factor (20%) in corrections may amplify losses
   - Quality and Value weights (40% each) may not be defensive enough
   - No defensive factors (low volatility, low beta) during stress

6. **Timing Issues**
   - Regime detection lag allows initial losses before defensive moves
   - Monthly rebalancing may be too infrequent for volatile periods
   - No intra-month regime change detection

## Recommendations

### Immediate Fixes (High Impact):

1. **Reduce Lookback Period**
   - Change from 90 to 60 days for faster regime detection
   - Expected impact: Earlier detection of market stress

2. **Lower Thresholds**
   - Volatility: High (40% → 30%), Medium (30% → 20%)
   - Returns: Strong- (-15% → -10%), Mod- (-5% → -3%)
   - Expected impact: More sensitive stress detection

3. **Reduce Minimum Regime Duration**
   - Change from 30 to 15 days
   - Expected impact: Faster regime transitions

### Medium-term Improvements:

4. **Add Crisis Regime Detection**
   - Implement drawdown-based classification
   - Add volatility spike detection
   - Expected impact: Better stress period identification

5. **Optimize Factor Weights**
   - Reduce momentum in corrections (20% → 10%)
   - Increase quality in stress (40% → 50%)
   - Add defensive factors during bear markets
   - Expected impact: Better risk-adjusted returns

6. **Improve Allocation Strategy**
   - Reduce correction allocation (50% → 40%)
   - Add dynamic allocation based on drawdown severity
   - Expected impact: Better capital preservation

### Long-term Enhancements:

7. **Risk Management Framework**
   - Maximum drawdown limits
   - Volatility targeting
   - Correlation-based position sizing
   - Expected impact: Systematic risk control

8. **Enhanced Regime Detection**
   - Multi-factor regime classification
   - Machine learning-based regime prediction
   - Real-time regime monitoring
   - Expected impact: More accurate market state identification

## Expected Performance Improvement

With the recommended changes, we expect:

1. **Earlier Stress Detection**: 2-4 weeks faster regime classification
2. **Better Capital Preservation**: 10-15% reduction in drawdown during stress periods
3. **Improved Risk-Adjusted Returns**: 0.1-0.2 improvement in Sharpe ratio
4. **More Consistent Performance**: Reduced volatility in regime transitions

## Implementation Priority

1. **Phase 1** (Week 1): Implement threshold adjustments and reduced lookback
2. **Phase 2** (Week 2): Add crisis regime detection and factor weight optimization
3. **Phase 3** (Week 3): Implement enhanced allocation strategy
4. **Phase 4** (Week 4): Add comprehensive risk management framework

## Conclusion

The 2018 underperformance was primarily caused by conservative regime detection parameters that failed to identify market stress early enough. The strategy remained in growth/bull regimes during significant market declines, leading to excessive exposure and amplified losses.

The recommended changes focus on making the regime detection more sensitive to Vietnamese market conditions while maintaining stability through improved risk management. These improvements should significantly enhance the strategy's performance during stress periods while preserving its strong performance during bull markets.
