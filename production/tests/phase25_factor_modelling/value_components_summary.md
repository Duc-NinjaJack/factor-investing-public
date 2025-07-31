# Value Components IC Analysis Summary

**Project:** Vietnam Factor Investing - Phase 25 Factor Modelling  
**Analysis:** Individual Contributions of Value Measures (P/E, P/B, P/S) to Value Factor IC  
**Date:** January 2025  
**Data Source:** Real market data (no simulations)  

---

## 🎯 **Key Findings**

### **Individual Component ICs:**

| Component | IC | N | Mean Score | Behavior |
|-----------|----|---|------------|----------|
| **P/B Score** | **-0.2038** | 133 | 0.1039 | **Strongest Contrarian** |
| **P/E Score** | -0.0644 | 238 | 0.0188 | Moderate Contrarian |
| **P/S Score** | -0.0241 | 238 | 0.0669 | Weakest Contrarian |

### **Composite Value Score IC: -0.2066**
- **Theoretical Composite IC:** -0.1202 (weighted average)
- **Actual Composite IC:** -0.2066
- **Difference:** -0.0864 (interaction effects)

---

## 🧠 **Critical Insights**

### **1. P/B Score Dominates the Contrarian Signal**
💡 **P/B Score is the primary driver of value factor contrarian behavior**
- **P/B Contribution:** -0.0815 (67.8% of theoretical composite)
- **P/E Contribution:** -0.0387 (32.2% of theoretical composite)
- **P/B IC:** -0.2038 (strongest negative signal)
- **P/E IC:** -0.0644 (moderate negative signal)

### **2. Interaction Effects Between Components**
⚠️ **Theoretical vs Actual Composite ICs differ significantly**
- **Difference:** -0.0864 (72% stronger than expected)
- **Implication:** P/E and P/B components interact synergistically
- **Effect:** Amplifies the contrarian signal beyond simple weighted average

### **3. Component-Specific Patterns by Quality Quintile**

#### **P/E Score Pattern:**
- **Range:** -0.3391 to +0.1016
- **Best:** Q2 (+0.1016) - Positive signal in moderate quality
- **Worst:** Q1 (-0.3391) - Strong contrarian in low quality

#### **P/B Score Pattern:**
- **Range:** -0.4504 to +0.0457
- **Best:** Q2 (+0.0457) - Slightly positive in moderate quality
- **Worst:** Q3 (-0.4504) - Very strong contrarian in moderate-low quality

#### **P/S Score Pattern:**
- **Range:** -0.3101 to +0.1359
- **Best:** Q3 (+0.1359) - Positive signal in moderate-low quality
- **Worst:** Q1 (-0.3101) - Strong contrarian in low quality

---

## 🎯 **Strategy Implications**

### **Component-Specific Value Strategy:**

#### **1. P/B-Focused Approach (Primary Driver):**
```python
def pb_focused_value_strategy(data):
    """
    P/B-focused value strategy (primary contrarian signal)
    """
    # P/B score is the strongest contrarian signal
    pb_signal = -data['pb_score']  # Strong contrarian
    
    # Quality-adjusted P/B approach
    if data['roaa_quintile'] == 'Q2':
        pb_signal = data['pb_score'] * 0.1  # Weak positive
    elif data['roaa_quintile'] == 'Q3':
        pb_signal = -data['pb_score'] * 2.0  # Very strong contrarian
    
    return pb_signal
```

#### **2. P/E-Adjusted Approach (Secondary Driver):**
```python
def pe_adjusted_value_strategy(data):
    """
    P/E-adjusted value strategy (moderate contrarian signal)
    """
    # P/E score shows moderate contrarian behavior
    pe_signal = -data['pe_score']  # Moderate contrarian
    
    # Quality-adjusted P/E approach
    if data['roaa_quintile'] == 'Q2':
        pe_signal = data['pe_score']  # Positive signal
    elif data['roaa_quintile'] == 'Q1 (Lowest)':
        pe_signal = -data['pe_score'] * 1.5  # Strong contrarian
    
    return pe_signal
```

#### **3. P/S Opportunistic Approach (Weakest Signal):**
```python
def ps_opportunistic_strategy(data):
    """
    P/S opportunistic strategy (weakest but positive in some regimes)
    """
    # P/S score shows regime-dependent behavior
    if data['roaa_quintile'] == 'Q3':
        ps_signal = data['ps_score']  # Positive signal
    else:
        ps_signal = -data['ps_score'] * 0.5  # Weak contrarian
    
    return ps_signal
```

#### **4. Enhanced Composite Strategy:**
```python
def enhanced_composite_strategy(data):
    """
    Enhanced composite strategy leveraging component interactions
    """
    # Base weights (banking sector)
    pb_weight = 0.4
    pe_weight = 0.6
    
    # Quality-adjusted weights
    if data['roaa_quintile'] == 'Q2':
        pb_weight = 0.3
        pe_weight = 0.7
    elif data['roaa_quintile'] == 'Q3':
        pb_weight = 0.6  # Increase P/B weight for stronger contrarian
        pe_weight = 0.4
    
    # Component signals
    pb_signal = -data['pb_score']  # Contrarian
    pe_signal = -data['pe_score']  # Contrarian
    
    # Enhanced composite (accounting for interactions)
    enhanced_score = (
        pb_signal * pb_weight +
        pe_signal * pe_weight
    ) * 1.2  # Amplify for interaction effects
    
    return enhanced_score
```

---

## 📈 **Performance Expectations**

### **By Component:**

| Component | Expected Performance | Strategy Weight | Quality Interaction |
|-----------|---------------------|-----------------|-------------------|
| **P/B Score** | **Strong Contrarian** (-20.38% IC) | **40%** | **High** |
| **P/E Score** | Moderate Contrarian (-6.44% IC) | **60%** | **Medium** |
| **P/S Score** | Weak Contrarian (-2.41% IC) | **0%** | **Low** |

### **By Quality Quintile:**

| Quintile | P/B Performance | P/E Performance | P/S Performance | Strategy |
|----------|----------------|-----------------|-----------------|----------|
| **Q2** | **Positive** (+4.57% IC) | **Positive** (+10.16% IC) | **Positive** (+8.08% IC) | Use all positively |
| **Q3** | **Very Strong Negative** (-45.04% IC) | **Weak Positive** (+2.62% IC) | **Positive** (+13.59% IC) | P/B contrarian, P/E/P/S positive |
| **Q1** | **Strong Negative** (-26.77% IC) | **Strong Negative** (-33.91% IC) | **Strong Negative** (-31.01% IC) | All contrarian |
| **Q5** | **Weak Negative** (-3.25% IC) | **Moderate Negative** (-11.18% IC) | **Weak Negative** (-3.89% IC) | All weak contrarian |
| **Q4** | **Strong Negative** (-20.24% IC) | **Moderate Negative** (-12.63% IC) | **Positive** (+4.39% IC) | P/B/P/E contrarian, P/S positive |

---

## 🔍 **Economic Interpretation**

### **Why P/B Score Shows Strongest Contrarian Signal:**
1. **Book Value Stability:** Book values are more stable than earnings
2. **Value Trap Detection:** High P/B ratios often indicate overvaluation
3. **Asset Quality:** P/B reflects asset quality and efficiency
4. **Market Sentiment:** P/B captures market sentiment about asset values

### **Why P/E Score Shows Moderate Contrarian Signal:**
1. **Earnings Volatility:** Earnings are more volatile than book values
2. **Growth Expectations:** P/E reflects growth expectations
3. **Quality Premium:** High P/E may reflect quality premium
4. **Earnings Quality:** P/E depends on earnings quality

### **Why P/S Score Shows Weakest Signal:**
1. **Revenue Stability:** Sales are more stable than earnings
2. **Margin Effects:** P/S doesn't account for profit margins
3. **Industry Differences:** P/S varies significantly by industry
4. **Growth Bias:** P/S may favor growth over value

### **Why Interaction Effects Amplify the Signal:**
1. **Synergistic Effects:** P/E and P/B reinforce each other
2. **Quality Confirmation:** Both ratios confirm quality assessment
3. **Market Inefficiency:** Market misprices multiple ratios simultaneously
4. **Risk Amplification:** Multiple contrarian signals amplify risk

---

## ⚠️ **Risk Considerations**

### **Component-Specific Risks:**
1. **P/B Risk:** Book value write-downs, asset quality deterioration
2. **P/E Risk:** Earnings manipulation, cyclical earnings
3. **P/S Risk:** Revenue recognition issues, margin compression
4. **Interaction Risk:** Correlation breakdown between components

### **Quality Quintile Risks:**
1. **Q2 Risk:** Moderate quality may deteriorate
2. **Q3 Risk:** Value trap in moderate-low quality
3. **Q1 Risk:** Low quality may become uninvestable
4. **Q5 Risk:** Quality premium may expand further

### **Implementation Risks:**
1. **Data Quality:** Component calculation accuracy
2. **Weight Stability:** Optimal weights may change over time
3. **Regime Changes:** Component relationships may shift
4. **Liquidity Constraints:** Component-specific liquidity issues

---

## 📋 **Implementation Recommendations**

### **Phase 1: Component Validation**
- [ ] Validate P/B calculation accuracy
- [ ] Monitor P/E earnings quality
- [ ] Track P/S revenue recognition
- [ ] Test component interaction stability

### **Phase 2: Quality-Adjusted Implementation**
- [ ] Implement P/B-focused strategy in Q3
- [ ] Use P/E positively in Q2
- [ ] Apply P/S opportunistically in Q3
- [ ] Adjust weights by quality quintile

### **Phase 3: Enhanced Composite Strategy**
- [ ] Implement enhanced composite scoring
- [ ] Account for interaction effects
- [ ] Monitor component correlations
- [ ] Optimize weights dynamically

---

## 🎯 **Key Takeaways**

### **1. P/B Score is the Primary Driver**
- **Strongest contrarian signal** (-20.38% IC)
- **Dominates composite value factor** (67.8% contribution)
- **Quality-dependent behavior** (Q2 positive, Q3 very negative)

### **2. Interaction Effects Amplify Signals**
- **72% stronger** than theoretical composite
- **Synergistic effects** between P/E and P/B
- **Enhanced contrarian signal** through component interaction

### **3. Quality Quintile Strategy**
- **Q2:** Use all components positively
- **Q3:** P/B contrarian, P/E/P/S positive
- **Q1/Q4/Q5:** Component-specific contrarian approaches

### **4. Implementation Priority**
1. **Focus on P/B score** (primary driver)
2. **Quality-adjust P/E** (secondary driver)
3. **Opportunistic P/S** (weakest signal)
4. **Account for interactions** (amplification effects)

---

**Document Version:** 1.0  
**Last Updated:** January 2025  
**Next Review:** Monthly  
**Maintained By:** Quantitative Research Team  
**Key Insight:** P/B score dominates the value factor contrarian signal with significant interaction effects 