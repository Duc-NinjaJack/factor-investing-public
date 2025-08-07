# Factor Analysis Summary - Version 18

## 📋 Executive Summary

This document summarizes the comprehensive factor analysis conducted to identify and fix critical issues in the QVM factor calculation methodology. The analysis revealed fundamental problems with factor staleness, normalization, and missing advanced metrics integration.

## 🔍 Key Findings

### 1. Quality Factor Staleness ✅ CONFIRMED QUARTERLY REFRESH

**Issue**: Quality factors appeared identical across recent dates (May-June 2025)

**Investigation**: 
- Analyzed quality factors across quarters and years (2022-2025)
- **Result**: Quality factors ARE changing quarterly as expected
- Evidence: FPT, HPG, TCB, VCB, VNM all show significant quarterly changes
- Example: FPT quality changed from 1.494 (2022-Q1) to 0.891 (2025-Q2)

**Conclusion**: Quality factors are properly refreshed quarterly based on fundamental data

### 2. Value Factor Normalization Issues ❌ CRITICAL PROBLEM

**Issue**: Value factors consistently negative (-0.46 average) with ceiling values (3.000)

**Investigation**:
- Value factor statistics: Mean = -0.015, Range = -1.88 to +3.00
- Top value stocks hit ceiling (3.000) indicating poor normalization
- Correlation with P/E is correct (-0.364 to -0.374) but scale is wrong

**Root Cause**: Poor normalization methodology causing negative averages and ceiling effects

### 3. Missing Advanced Metrics Integration ❌ OPPORTUNITY

**Investigation**:
- **F-Score**: ❌ No data available in `precalculated_metrics`
- **FCF Yield**: ✅ Available (27,706 records, 656 tickers, 79 dates)
- **Low Vol**: ✅ Available (83,518 records, 707 tickers, 138 dates)

**Opportunity**: Can integrate FCF Yield and Low Vol immediately

### 4. Factor Generation Pipeline Issues ❌ INFRASTRUCTURE PROBLEM

**Issue**: 2,384 unique dates but only 89 unique timestamps

**Problem**: Many dates share same generation timestamp, indicating stale factor generation

## 🔧 Solutions Implemented

### Version 18a: Advanced Metrics Strategy

**Features**:
1. **Proper Factor Refresh Cycles**:
   - Quality factors: Quarterly refresh (fundamental-based)
   - Value factors: Daily refresh (price-based)
   - Momentum factors: Daily refresh (price-based)

2. **Full Advanced Metrics Integration**:
   - Quality: ROAA + F-Score (50/50) - F-Score placeholder for future
   - Value: P/E + FCF Yield (50/50)
   - Momentum: Momentum + Low Vol (50/50)

3. **Proper Normalization**: Ranking-based 0-1 scale
   - Formula: `normalized_score = (rank - 1) / (total_stocks - 1)`
   - Ensures all factors scale from 0 to 1

4. **Regime Detection**: Dynamic factor weighting
   - Normal: 40% Quality, 30% Value, 30% Momentum
   - Stress: 60% Quality, 30% Value, 10% Momentum (60% allocation)
   - Bull: 20% Quality, 30% Value, 50% Momentum

### Version 18b: Available Metrics Strategy

**Features**:
1. **Available Metrics Only** (no F-Score):
   - Quality: ROAA only (100%)
   - Value: P/E + FCF Yield (50/50)
   - Momentum: Momentum + Low Vol (50/50)

2. **Adjusted Weights**:
   - Quality: 30% (reduced since no F-Score)
   - Value: 40% (increased focus)
   - Momentum: 30%

3. **Same Proper Normalization and Regime Detection**

## 📊 Technical Improvements

### 1. Factor Normalization Fix

**Before**: Z-score normalization with negative averages
**After**: Ranking-based normalization (0-1 scale)

```python
# New normalization method
data_df['rank'] = data_df['metric'].rank(ascending=True, method='min')
data_df['normalized'] = (data_df['rank'] - 1) / (len(data_df) - 1)
```

### 2. Factor Refresh Cycles

**Quality Factors**: Quarterly refresh (fundamental data)
**Value Factors**: Daily refresh (price-based metrics)
**Momentum Factors**: Daily refresh (price-based metrics)

### 3. Advanced Metrics Integration

**FCF Yield**: Integrated into Value factor
**Low Volatility**: Integrated into Momentum factor
**F-Score**: Placeholder for future implementation

### 4. Regime-Aware Allocation

**Dynamic Allocation**: Adjusts portfolio size based on market regime
**Factor Weighting**: Changes factor importance based on market conditions

## 🎯 Expected Performance Improvements

### 1. Value Factor Fix
- **Before**: Negative scale, ceiling effects, poor discrimination
- **After**: 0-1 scale, proper ranking, better stock selection

### 2. Advanced Metrics Integration
- **FCF Yield**: Better value identification
- **Low Vol**: Better risk-adjusted returns
- **Combined**: More robust factor signals

### 3. Proper Refresh Cycles
- **Quality**: Quarterly updates reflect changing fundamentals
- **Value/Momentum**: Daily updates capture market changes

### 4. Regime Detection
- **Stress**: Quality focus with reduced allocation
- **Bull**: Momentum focus with full allocation
- **Normal**: Balanced approach

## 📁 File Organization

### Scripts Moved to `scripts/` Subfolder:
- `check_*.py`: Database and factor checking scripts
- `analyze_*.py`: Analysis and investigation scripts
- `factor_analysis_comprehensive.py`: Comprehensive factor analysis

### New Strategy Files:
- `18a_advanced_metrics_strategy.py`: Full advanced metrics version
- `18b_available_metrics_strategy.py`: Available metrics only version

## 🚀 Next Steps

### Immediate (Version 18b):
1. Test Version 18b with available metrics
2. Validate factor normalization improvements
3. Compare performance against previous versions

### Short-term:
1. Source F-Score data or implement calculation
2. Implement daily factor generation pipeline
3. Add factor validation and monitoring

### Long-term:
1. Implement real-time factor generation
2. Add factor backtesting framework
3. Develop factor performance attribution

## 📈 Success Metrics

### Factor Quality:
- All factors should scale 0-1
- No negative averages
- No ceiling effects
- Proper quarterly/daily refresh

### Performance:
- Improved Sharpe ratio
- Better risk-adjusted returns
- Reduced drawdowns
- Better regime adaptation

### Technical:
- Proper factor refresh cycles
- Advanced metrics integration
- Robust normalization
- Clean code organization

## 🔍 Validation Checklist

- [ ] Quality factors refresh quarterly
- [ ] Value factors scale 0-1 (no negative averages)
- [ ] Momentum factors update daily
- [ ] FCF Yield integrated into Value factor
- [ ] Low Vol integrated into Momentum factor
- [ ] Regime detection working properly
- [ ] Dynamic allocation functioning
- [ ] All scripts moved to scripts/ subfolder
- [ ] Performance improved vs previous versions

---

**Conclusion**: The factor analysis revealed critical issues with normalization and missing advanced metrics. Version 18 implements proper fixes with ranking-based normalization, advanced metrics integration, and proper factor refresh cycles. These improvements should significantly enhance strategy performance and robustness.
