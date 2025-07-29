# Alternative Liquidity Thresholds Investigation Roadmap

**Date:** 2025-07-29  
**Purpose:** Investigate alternative liquidity thresholds following 3B VND rejection  
**Status:** PLANNING PHASE  
**Previous Analysis:** 3B VND threshold rejected due to performance degradation  

## 🎯 Objective

Based on the comprehensive analysis that rejected the 3B VND threshold, investigate alternative liquidity thresholds that may provide better balance between universe expansion and performance quality.

## 📊 Analysis Results Summary

### 3B VND Threshold Results (REJECTED)
- **Performance:** -10.77% return decline vs 10B VND
- **Risk:** -31.57% drawdown vs -30.28% (worse)
- **Sharpe:** -0.03 vs 0.45 (major decline)
- **Universe:** 230 stocks vs 164 stocks (+66 stocks)

### Key Insights from 3B VND Analysis
1. **Real data validation is critical** - simulated returns were overly optimistic
2. **Liquidity impact is complex** - lower liquidity stocks show hidden costs
3. **Factor persistence varies** by liquidity level
4. **Market conditions matter** - real market dynamics significantly impact performance

## 🎯 Alternative Thresholds to Investigate

### Primary Candidates
1. **5B VND Threshold** (Priority: HIGH)
   - Middle ground between 3B and 10B VND
   - Expected universe: ~180-200 stocks
   - Potential balance of expansion vs quality

2. **7B VND Threshold** (Priority: HIGH)
   - Closer to current 10B VND
   - Expected universe: ~170-185 stocks
   - Conservative expansion approach

3. **8B VND Threshold** (Priority: MEDIUM)
   - Minimal expansion from current
   - Expected universe: ~165-175 stocks
   - Very conservative approach

### Secondary Candidates
4. **6B VND Threshold** (Priority: MEDIUM)
   - Between 5B and 7B VND
   - Expected universe: ~175-190 stocks

5. **9B VND Threshold** (Priority: LOW)
   - Very close to current 10B VND
   - Expected universe: ~160-170 stocks
   - Minimal change approach

## 📋 Investigation Plan

### Phase 1: Quick Validation (Week 1)
**Objective:** Rapid assessment of universe expansion and factor quality

**Tasks:**
1. **5B VND Quick Validation**
   - Update configuration to 5B VND
   - Run universe expansion analysis
   - Calculate QVM score impact
   - Generate quick validation report

2. **7B VND Quick Validation**
   - Update configuration to 7B VND
   - Run universe expansion analysis
   - Calculate QVM score impact
   - Generate quick validation report

3. **8B VND Quick Validation**
   - Update configuration to 8B VND
   - Run universe expansion analysis
   - Calculate QVM score impact
   - Generate quick validation report

**Success Criteria:**
- Universe expansion <1.5x for all thresholds
- QVM score impact positive or neutral
- Average ADTV >50B VND for all thresholds

### Phase 2: Real Data Backtesting (Week 2-3)
**Objective:** Comprehensive performance validation with real price data

**Tasks:**
1. **5B VND Real Data Backtesting**
   - Run full backtesting with real price data
   - Compare vs 10B VND baseline
   - Generate performance report

2. **7B VND Real Data Backtesting**
   - Run full backtesting with real price data
   - Compare vs 10B VND baseline
   - Generate performance report

3. **8B VND Real Data Backtesting**
   - Run full backtesting with real price data
   - Compare vs 10B VND baseline
   - Generate performance report

**Success Criteria:**
- Annual return >= 10B VND performance
- Sharpe ratio >= 0.45
- Max drawdown <= -30.28%
- Alpha >= -4.88%

### Phase 3: Comparative Analysis (Week 4)
**Objective:** Comprehensive comparison and recommendation

**Tasks:**
1. **Performance Comparison Matrix**
   - Compare all thresholds vs 10B VND baseline
   - Rank by performance metrics
   - Identify best performing threshold

2. **Risk-Return Analysis**
   - Create risk-return scatter plots
   - Analyze drawdown characteristics
   - Evaluate risk-adjusted returns

3. **Universe Quality Analysis**
   - Analyze factor persistence by threshold
   - Study sector diversification
   - Evaluate liquidity characteristics

4. **Final Recommendation**
   - Select optimal threshold
   - Document rationale
   - Create implementation plan

## 🔬 Methodology

### Data Sources
- **Price data:** `vcsc_daily_data_complete` (close_price_adjusted)
- **Factor scores:** `factor_scores_qvm` (QVM_Composite)
- **Benchmark:** `etf_history` (VNINDEX)
- **ADTV data:** `unrestricted_universe_data.pkl`

### Backtesting Framework
- **No short-selling constraint**
- **Monthly rebalancing**
- **Transaction costs:** 20 bps
- **Equal weight portfolio construction**
- **Portfolio size:** 25 stocks
- **Initial capital:** 100M VND

### Performance Metrics
- Annual Return
- Annual Volatility
- Sharpe Ratio
- Max Drawdown
- Alpha
- Beta
- Calmar Ratio
- Information Ratio

## 📊 Expected Outcomes

### Best Case Scenario
- Find threshold with 5-10% universe expansion
- Maintain or improve performance vs 10B VND
- Better risk-adjusted returns
- **Recommendation:** Implement optimal threshold

### Realistic Scenario
- Find threshold with 2-5% universe expansion
- Maintain performance vs 10B VND
- Similar risk metrics
- **Recommendation:** Implement if benefits outweigh costs

### Worst Case Scenario
- All thresholds show performance degradation
- No viable alternative to 10B VND
- **Recommendation:** Maintain current threshold, investigate other optimization approaches

## 🎯 Success Criteria

### Primary Criteria
1. **Performance Maintenance:** Annual return >= 10.23%
2. **Risk Control:** Max drawdown <= -30.28%
3. **Risk-Adjusted Returns:** Sharpe ratio >= 0.45
4. **Alpha Generation:** Alpha >= -4.88%

### Secondary Criteria
1. **Universe Expansion:** 1.1x to 1.3x (modest expansion)
2. **Factor Quality:** QVM score impact neutral or positive
3. **Liquidity Standards:** Average ADTV >= 100B VND
4. **Implementation Feasibility:** Minimal operational impact

## 📋 Implementation Timeline

### Week 1: Quick Validation
- Day 1-2: 5B VND quick validation
- Day 3-4: 7B VND quick validation
- Day 5: 8B VND quick validation

### Week 2-3: Real Data Backtesting
- Week 2: 5B VND and 7B VND backtesting
- Week 3: 8B VND backtesting and analysis

### Week 4: Comparative Analysis
- Day 1-3: Performance comparison and analysis
- Day 4-5: Final recommendation and documentation

## 🔍 Risk Considerations

### Technical Risks
- **Database connectivity issues** - Mitigation: Use existing working framework
- **Data quality issues** - Mitigation: Validate data before analysis
- **Computational complexity** - Mitigation: Optimize backtesting scripts

### Analysis Risks
- **Overfitting to historical data** - Mitigation: Use out-of-sample validation
- **Market regime changes** - Mitigation: Analyze different time periods
- **Factor decay** - Mitigation: Monitor factor persistence

### Implementation Risks
- **Performance degradation** - Mitigation: Conservative threshold selection
- **Operational complexity** - Mitigation: Minimal configuration changes
- **Monitoring requirements** - Mitigation: Leverage existing monitoring

## 📈 Expected Benefits

### If Successful Implementation
1. **Universe Expansion:** 10-30% more investment opportunities
2. **Performance Improvement:** Better risk-adjusted returns
3. **Alpha Enhancement:** Improved factor capture
4. **Operational Efficiency:** Optimized liquidity management

### Knowledge Benefits
1. **Liquidity Threshold Optimization:** Better understanding of optimal levels
2. **Factor Persistence Analysis:** Insights into factor decay by liquidity
3. **Risk Management Enhancement:** Improved liquidity risk controls
4. **Methodology Validation:** Confirmed real data validation importance

## 🎯 Next Steps

### Immediate Actions (This Week)
1. **Create investigation scripts** for alternative thresholds
2. **Set up analysis framework** for comparative studies
3. **Prepare data validation** procedures

### Short-term Actions (Next 2 Weeks)
1. **Execute quick validation** for 5B, 7B, and 8B VND thresholds
2. **Run real data backtesting** for promising candidates
3. **Generate comparative analysis** reports

### Medium-term Actions (Next Month)
1. **Select optimal threshold** based on analysis
2. **Create implementation plan** if threshold approved
3. **Document lessons learned** and methodology

---

**Status:** PLANNING PHASE  
**Next Review:** 2025-08-05 (1 week)  
**Expected Completion:** 2025-08-26 (4 weeks)