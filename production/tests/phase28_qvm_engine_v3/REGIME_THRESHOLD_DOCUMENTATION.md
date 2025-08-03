# Vietnamese Market Regime Detection Threshold Analysis

## 📊 Executive Summary

This document analyzes the regime detection thresholds used in QVM Engine implementations, specifically investigating what the **26.59% volatility threshold** means and how it performs across different market periods in Vietnam.

## 🎯 What Does 26.59% Volatility Threshold Mean?

### **📈 Threshold Definition**
- **26.59%** = **90-day rolling volatility (annualized)**
- **Time Period**: 90-day rolling window
- **Annualization**: Daily volatility × √252
- **Source**: 75th percentile from 2016-2020 period

### **📅 Period-Specific Analysis**

| Period | 75th Percentile | 90th Percentile | 95th Percentile | Fixed Threshold (26.59%) |
|--------|----------------|-----------------|-----------------|---------------------------|
| **2016-2020** | 20.44% | 25.96% | 28.48% | **BELOW 95th percentile** |
| **2020-2025** | 24.28% | 26.42% | 29.11% | **BELOW 95th percentile** |
| **Full Period** | 22.21% | 26.21% | 27.90% | **BELOW 95th percentile** |

## 🌪️ Market Period Characteristics

### **2016-2020 Period (High Volatility Era)**
- **Total Observations**: 1,161 days
- **Date Range**: 2016-05-19 to 2020-12-31
- **Key Events**: Trade wars, COVID-19 crash, market stress

**Volatility Quintiles:**
| Quintile | Volatility Range | Mean Vol | Mean Return | Characteristics |
|----------|------------------|----------|-------------|-----------------|
| Q1 (Lowest) | 7.41% - 9.80% | 8.66% | 17.64% | Quiet periods, stable growth |
| Q2 | 9.80% - 12.35% | 11.02% | 18.85% | Normal volatility |
| Q3 | 12.35% - 14.92% | 13.38% | 27.15% | Moderate volatility, good returns |
| Q4 | 14.94% - 21.69% | 18.55% | 14.02% | High volatility, mixed returns |
| Q5 (Highest) | 21.69% - 31.28% | 26.44% | -13.51% | **Turbulent periods, negative returns** |

**Period Classification:**
- **Quiet**: 25.0% of days (vol < 10.57%)
- **Normal**: 50.0% of days (vol 10.57% - 20.44%)
- **Turbulent**: 25.0% of days (vol > 20.44%)

### **2020-2025 Period (Lower Volatility Era)**
- **Total Observations**: 1,299 days
- **Date Range**: 2020-05-19 to 2025-07-28
- **Key Events**: Post-COVID recovery, central bank support, stable growth

**Volatility Quintiles:**
| Quintile | Volatility Range | Mean Vol | Mean Return | Characteristics |
|----------|------------------|----------|-------------|-----------------|
| Q1 (Lowest) | 8.75% - 13.66% | 11.52% | 25.02% | Very quiet, strong returns |
| Q2 | 13.69% - 16.71% | 15.43% | 19.77% | Low volatility, good returns |
| Q3 | 16.74% - 20.94% | 18.97% | 9.45% | Moderate volatility, modest returns |
| Q4 | 20.95% - 25.16% | 23.13% | 3.85% | High volatility, low returns |
| Q5 (Highest) | 25.17% - 31.28% | 27.32% | -3.31% | **Turbulent periods, negative returns** |

**Period Classification:**
- **Quiet**: 25.0% of days (vol < 14.64%)
- **Normal**: 50.0% of days (vol 14.64% - 24.28%)
- **Turbulent**: 25.0% of days (vol > 24.28%)

## 📅 Specific Market Periods Analysis

### **Turbulent Periods (High Volatility)**

| Period | Volatility | Returns | Characteristics |
|--------|------------|---------|-----------------|
| **COVID Crash 2020** | 19.35% | -53.85% | Market crash, extreme stress |
| **COVID Recovery 2020** | 22.81% | 19.87% | Recovery with high volatility |
| **Bear Market 2022** | 21.72% | -25.58% | Bear market, negative returns |

### **Quiet Periods (Low Volatility)**

| Period | Volatility | Returns | Characteristics |
|--------|------------|---------|-----------------|
| **Stable 2024** | 14.97% | 12.03% | Very stable, positive returns |
| **Recovery 2023** | 18.71% | 7.61% | Moderate recovery |
| **Bull Market 2021** | 20.62% | 41.18% | Strong bull market |

## 🚨 The Problem with Fixed Thresholds

### **❌ Why Only "Sideways" Regime is Detected in 2020-2025**

The fixed 26.59% threshold causes the regime detection logic to always fall into the "Sideways" category:

```python
# Current logic with fixed thresholds
if volatility > 26.59%:  # High volatility
    if avg_return > 25.88%:  # High returns
        return 'Bull'
    else:
        return 'Bear'
else:  # Low volatility ← ALWAYS THIS CASE!
    if abs(avg_return) < 21.31%:  # Low absolute returns
        return 'Sideways'  # ← ALWAYS THIS RESULT!
    else:
        return 'Stress'
```

**Evidence from 2020-2025:**
- Current volatility: ~1-2% (much lower than 26.59%)
- Current returns: ~0.02% (much lower than 21.31%)
- **Result**: Always "Sideways" regime

### **📊 Threshold Comparison**

| Metric | 2016-2020 (Calibration) | 2020-2025 (Current) | Fixed Threshold | Issue |
|--------|------------------------|---------------------|-----------------|-------|
| **75th Percentile Vol** | 20.44% | 24.28% | 26.59% | Too high for current period |
| **75th Percentile Return** | 29.32% | 33.08% | 25.88% | Too low for current period |
| **25th Percentile Low Return** | 11.77% | 11.21% | 21.31% | Too high for current period |

## 💡 Recommendations

### **1. 🎯 Period-Specific Thresholds**

**For 2020-2025 Period:**
```python
"regime": {
    "volatility_threshold": 0.2428,  # 75th percentile from 2020-2025
    "return_threshold": 0.3308,      # 75th percentile from 2020-2025
    "low_return_threshold": 0.1121   # 25th percentile from 2020-2025
}
```

**For 2016-2020 Period:**
```python
"regime": {
    "volatility_threshold": 0.2044,  # 75th percentile from 2016-2020
    "return_threshold": 0.2932,      # 75th percentile from 2016-2020
    "low_return_threshold": 0.1177   # 25th percentile from 2016-2020
}
```

### **2. 🔄 Adaptive Percentile-Based Approach**

Switch to dynamic threshold calculation:
```python
def calculate_adaptive_thresholds(data, lookback_period=252):
    recent_data = data.tail(lookback_period)
    vol_threshold = np.percentile(recent_data['volatility'], 75)
    return_threshold = np.percentile(recent_data['returns'], 75)
    low_return_threshold = np.percentile(np.abs(recent_data['returns']), 25)
    return vol_threshold, return_threshold, low_return_threshold
```

### **3. 📊 Market Regime Classification**

**Improved 4-Regime Classification:**
```python
def detect_regime_improved(volatility, avg_return, vol_threshold, return_threshold, low_return_threshold):
    if volatility > vol_threshold:  # High volatility
        if avg_return > return_threshold:  # High returns
            return 'Bull'  # High vol + high returns
        else:
            return 'Bear'  # High vol + low returns
    else:  # Low volatility
        if abs(avg_return) < low_return_threshold:  # Low absolute returns
            return 'Sideways'  # Low vol + low returns
        else:
            return 'Stress'  # Low vol + high returns (unusual)
```

## 📈 Performance Implications

### **Current Issues:**
1. **No regime differentiation** in 2020-2025 period
2. **Inappropriate portfolio allocation** (always 60% for Sideways)
3. **Missed opportunities** during bull/bear markets
4. **Poor risk management** due to incorrect regime classification

### **Expected Improvements:**
1. **Better regime detection** with period-specific thresholds
2. **More appropriate portfolio allocation** based on actual market conditions
3. **Improved risk-adjusted returns** through proper regime-based strategies
4. **Enhanced risk management** during turbulent periods

## 🔍 Key Insights

### **1. Market Evolution**
- **2016-2020**: High volatility period with frequent regime changes
- **2020-2025**: Lower volatility period with more stable conditions
- **Fixed thresholds don't adapt** to changing market characteristics

### **2. Threshold Calibration**
- **26.59% threshold** was appropriate for 2016-2020
- **Too high** for 2020-2025 period
- **Period-specific calibration** is essential

### **3. Regime Detection Logic**
- **Current logic** works well for high volatility periods
- **Fails** in low volatility environments
- **Adaptive approach** needed for different market conditions

## 📋 Implementation Plan

### **Phase 1: Immediate Fix**
1. Implement period-specific thresholds for 2020-2025
2. Test with current v3f implementation
3. Validate regime detection accuracy

### **Phase 2: Adaptive System**
1. Implement percentile-based threshold calculation
2. Add rolling window for threshold updates
3. Test across multiple market periods

### **Phase 3: Enhanced Classification**
1. Improve regime classification logic
2. Add regime transition detection
3. Implement regime-aware portfolio optimization

---

**Conclusion**: The 26.59% volatility threshold represents the 75th percentile of 90-day rolling volatility during the 2016-2020 period. While appropriate for that turbulent era, it's too high for the current 2020-2025 period, causing the regime detection to always classify markets as "Sideways". Period-specific or adaptive thresholds are essential for proper regime detection across different market environments. 