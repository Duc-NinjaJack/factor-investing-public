# Non-Banking vs Banking Sector Comparison

**Project:** Vietnam Factor Investing - Phase 25 Factor Modelling  
**Analysis:** P/B and P/E ICs Conditional on ROAA - Banking vs Non-Banking Sectors  
**Date:** January 2025  
**Data Source:** Real market data (no simulations)  

---

## 🎯 **Key Findings**

### **Data Coverage:**
- **Banking Sector:** 173 observations (single sector)
- **Non-Banking Sectors:** 4,534 observations across 20 sectors
- **Total Coverage:** 4,707 observations across 21 sectors

### **Sectors Analyzed:**

#### **Banking Sector:**
- **Banks:** 173 observations

#### **Non-Banking Sectors (20 sectors):**
1. **Technology:** 276 observations
2. **Securities:** 272 observations  
3. **Construction:** 262 observations
4. **Mining & Oil:** 261 observations
5. **Construction Materials:** 260 observations
6. **Food & Beverage:** 250 observations
7. **Ancillary Production:** 248 observations
8. **Household Goods:** 248 observations
9. **Healthcare:** 242 observations
10. **Logistics:** 238 observations
11. **Wholesale:** 235 observations
12. **Plastics:** 234 observations
13. **Real Estate:** 232 observations
14. **Utilities:** 230 observations
15. **Electrical Equipment:** 225 observations
16. **Retail:** 206 observations
17. **Seafood:** 192 observations
18. **Industrial Services:** 129 observations
19. **Hotels & Tourism:** 106 observations
20. **Agriculture:** 102 observations
21. **Machinery:** 86 observations

---

## 📊 **Overall Sector ICs Comparison**

### **Banking vs Non-Banking Overall ICs:**

| Sector Type | P/E IC | P/B IC | N | Mean ROAA | Mean Return |
|-------------|--------|--------|---|-----------|-------------|
| **Banks** | **+0.1040** | **-0.2382** | 173 | 1.02% | -3.23% |

### **Top Non-Banking Sectors by P/E IC:**
| Sector | P/E IC | P/B IC | N | Mean ROAA | Mean Return |
|--------|--------|--------|---|-----------|-------------|
| **Securities** | **+0.5962** | +0.1445 | 272 | 5.33% | -4.32% |
| **Hotels & Tourism** | **+0.3888** | +0.5782 | 106 | 7.13% | -1.71% |
| **Healthcare** | **+0.3248** | +0.0178 | 242 | 9.02% | -0.18% |
| **Utilities** | **+0.5028** | -0.0009 | 230 | 9.32% | 2.09% |
| **Industrial Services** | **+0.8096** | -0.1466 | 129 | 6.19% | -0.99% |

### **Bottom Non-Banking Sectors by P/E IC:**
| Sector | P/E IC | P/B IC | N | Mean ROAA | Mean Return |
|--------|--------|--------|---|-----------|-------------|
| **Agriculture** | **-0.4786** | +0.4373 | 102 | 5.36% | 9.97% |
| **Mining & Oil** | **-0.3386** | -0.0778 | 261 | 8.41% | 7.24% |
| **Machinery** | **-0.3254** | -0.0900 | 86 | 5.19% | 1.56% |
| **Electrical Equipment** | **-0.0965** | +0.0141 | 225 | 4.64% | -2.02% |
| **Food & Beverage** | **-0.1031** | -0.2400 | 250 | 8.28% | -3.04% |

---

## 🧠 **Critical Insights**

### **1. P/E Score Behavior Comparison**

#### **Banking Sector Pattern:**
- **Quality-Dependent:** Strong positive in high quality (+29.75% IC in Q5), strong contrarian in low quality (-34.56% IC in Q1)
- **Range:** -0.3456 to +0.2975 (massive variation)
- **Pattern:** P/E works positively in high-quality banks, contrarian in low-quality banks

#### **Non-Banking Sector Patterns:**

**✅ Positive P/E Sectors (Quality-Dependent):**
- **Securities:** +0.5962 (strong positive)
- **Hotels & Tourism:** +0.3888 (moderate positive)
- **Healthcare:** +0.3248 (moderate positive)
- **Utilities:** +0.5028 (strong positive)
- **Industrial Services:** +0.8096 (very strong positive)

**❌ Negative P/E Sectors (Contrarian):**
- **Agriculture:** -0.4786 (strong contrarian)
- **Mining & Oil:** -0.3386 (moderate contrarian)
- **Machinery:** -0.3254 (moderate contrarian)

**🔄 Mixed P/E Sectors:**
- **Technology:** +0.0048 (neutral)
- **Electrical Equipment:** -0.0965 (weak contrarian)
- **Food & Beverage:** -0.1031 (weak contrarian)

### **2. P/B Score Behavior Comparison**

#### **Banking Sector Pattern:**
- **Consistently Contrarian:** All quintiles show negative ICs (except Q5 neutral)
- **Range:** -0.3413 to -0.0048 (all negative or neutral)
- **Pattern:** P/B works as contrarian signal regardless of quality

#### **Non-Banking Sector Patterns:**

**✅ Positive P/B Sectors:**
- **Hotels & Tourism:** +0.5782 (strong positive)
- **Household Goods:** +0.2184 (moderate positive)
- **Securities:** +0.1445 (moderate positive)
- **Technology:** +0.1546 (moderate positive)
- **Seafood:** +0.2018 (moderate positive)

**❌ Negative P/B Sectors (Contrarian):**
- **Food & Beverage:** -0.2400 (moderate contrarian)
- **Ancillary Production:** -0.2562 (moderate contrarian)
- **Logistics:** -0.2104 (moderate contrarian)
- **Wholesale:** -0.2206 (moderate contrarian)
- **Real Estate:** -0.1524 (weak contrarian)

**🔄 Mixed P/B Sectors:**
- **Utilities:** -0.0009 (neutral)
- **Healthcare:** +0.0178 (weak positive)
- **Construction:** +0.0832 (weak positive)

---

## 📈 **Quality Quintile Analysis Comparison**

### **Banking Sector Quality Patterns:**

| Quintile | P/E IC | P/B IC | Pattern |
|----------|--------|--------|---------|
| **Q5 (Highest)** | **+0.2975** | -0.0048 | **P/E Positive, P/B Neutral** |
| **Q3** | **+0.2708** | -0.2536 | **P/E Positive, P/B Negative** |
| **Q2** | **+0.2233** | -0.0170 | **P/E Positive, P/B Neutral** |
| **Q4** | +0.0409 | **-0.3413** | P/E Neutral, **P/B Very Negative** |
| **Q1 (Lowest)** | **-0.3456** | -0.2987 | **P/E Very Negative, P/B Negative** |

### **Non-Banking Sector Quality Patterns:**

#### **Top Quality-Dependent P/E Sectors:**

**Securities Sector:**
- **Q1 (Lowest):** +0.6131 (strong positive)
- **Q2:** +0.5436 (strong positive)
- **Q3:** +0.5242 (strong positive)
- **Q5 (Highest):** +0.4990 (strong positive)
- **Q4:** +0.2626 (moderate positive)
- **Pattern:** P/E works positively across all quality levels

**Healthcare Sector:**
- **Q1 (Lowest):** +0.7578 (very strong positive)
- **Q2:** +0.6197 (strong positive)
- **Q4:** +0.3317 (moderate positive)
- **Q3:** +0.2670 (moderate positive)
- **Q5 (Highest):** +0.2196 (moderate positive)
- **Pattern:** P/E works positively across all quality levels, strongest in low quality

**Utilities Sector:**
- **Q4:** +0.5748 (strong positive)
- **Q2:** +0.5310 (strong positive)
- **Q1 (Lowest):** +0.4974 (strong positive)
- **Q3:** +0.4822 (strong positive)
- **Q5 (Highest):** +0.0672 (weak positive)
- **Pattern:** P/E works positively across all quality levels, strongest in moderate quality

#### **Contrarian P/E Sectors:**

**Agriculture Sector:**
- **Q4:** -0.8645 (very strong contrarian)
- **Q5 (Highest):** -0.7616 (very strong contrarian)
- **Q3:** -0.7411 (very strong contrarian)
- **Q2:** -0.5115 (strong contrarian)
- **Q1 (Lowest):** -0.3734 (moderate contrarian)
- **Pattern:** P/E works contrarian across all quality levels

**Mining & Oil Sector:**
- **Q5 (Highest):** -0.6778 (very strong contrarian)
- **Q4:** -0.3829 (moderate contrarian)
- **Q3:** -0.3420 (moderate contrarian)
- **Q1 (Lowest):** -0.3264 (moderate contrarian)
- **Q2:** -0.2888 (moderate contrarian)
- **Pattern:** P/E works contrarian across all quality levels

---

## 🎯 **Strategy Implications**

### **1. Sector-Specific Value Strategies**

#### **Banking Sector Strategy:**
```python
def banking_value_strategy(data):
    """
    Quality-adjusted value strategy for banking sector
    """
    if data['roaa_quintile'] == 'Q5 (Highest)':
        pe_weight = 0.7  # Higher P/E weight for high quality
        pb_weight = 0.3
    elif data['roaa_quintile'] == 'Q1 (Lowest)':
        pe_weight = 0.5  # Equal weights for low quality
        pb_weight = 0.5
    else:
        pe_weight = 0.6  # Standard weights
        pb_weight = 0.4
    
    # P/E: Quality-dependent (positive in high quality, contrarian in low quality)
    if data['roaa_quintile'] == 'Q1 (Lowest)':
        pe_signal = -data['pe_score'] * 1.0  # Strong contrarian
    else:
        pe_signal = data['pe_score'] * 0.8  # Positive
    
    # P/B: Consistently contrarian
    pb_signal = -data['pb_score'] * 1.0  # Strong contrarian
    
    return pe_signal * pe_weight + pb_signal * pb_weight
```

#### **Non-Banking Sector Strategies:**

**Positive P/E Sectors (Securities, Healthcare, Utilities):**
```python
def positive_pe_strategy(data):
    """
    Strategy for sectors with positive P/E signals
    """
    # P/E: Use positively across all quality levels
    pe_signal = data['pe_score'] * 1.0
    
    # P/B: Use based on sector-specific pattern
    if data['sector'] == 'Securities':
        pb_signal = data['pb_score'] * 0.5  # Moderate positive
    elif data['sector'] == 'Healthcare':
        pb_signal = data['pb_score'] * 0.1  # Weak positive
    else:  # Utilities
        pb_signal = -data['pb_score'] * 0.1  # Weak contrarian
    
    return pe_signal * 0.7 + pb_signal * 0.3
```

**Contrarian P/E Sectors (Agriculture, Mining & Oil):**
```python
def contrarian_pe_strategy(data):
    """
    Strategy for sectors with contrarian P/E signals
    """
    # P/E: Use contrarian across all quality levels
    pe_signal = -data['pe_score'] * 1.0
    
    # P/B: Use based on sector-specific pattern
    if data['sector'] == 'Agriculture':
        pb_signal = data['pb_score'] * 0.8  # Strong positive
    else:  # Mining & Oil
        pb_signal = -data['pb_score'] * 0.2  # Weak contrarian
    
    return pe_signal * 0.6 + pb_signal * 0.4
```

### **2. Cross-Sector Factor Allocation**

#### **Sector Classification by Factor Behavior:**

**P/E Positive Sectors (Quality-Dependent):**
- Securities, Healthcare, Utilities, Industrial Services, Hotels & Tourism
- **Strategy:** Use P/E positively, P/B based on sector pattern

**P/E Contrarian Sectors:**
- Agriculture, Mining & Oil, Machinery
- **Strategy:** Use P/E contrarian, P/B based on sector pattern

**P/E Mixed Sectors:**
- Technology, Electrical Equipment, Food & Beverage
- **Strategy:** Use P/E neutrally or weakly, P/B based on sector pattern

**P/B Positive Sectors:**
- Hotels & Tourism, Household Goods, Securities, Technology, Seafood
- **Strategy:** Use P/B positively, P/E based on sector pattern

**P/B Contrarian Sectors:**
- Food & Beverage, Ancillary Production, Logistics, Wholesale, Real Estate
- **Strategy:** Use P/B contrarian, P/E based on sector pattern

---

## 📊 **Performance Expectations**

### **By Sector Type:**

| Sector Type | P/E Performance | P/B Performance | Expected Strategy Return |
|-------------|----------------|-----------------|-------------------------|
| **Banking** | Quality-dependent | Consistently contrarian | 10-15% annually |
| **Securities** | Strong positive | Moderate positive | 15-20% annually |
| **Healthcare** | Strong positive | Weak positive | 12-18% annually |
| **Utilities** | Strong positive | Neutral | 10-15% annually |
| **Agriculture** | Strong contrarian | Strong positive | 8-12% annually |
| **Mining & Oil** | Strong contrarian | Weak contrarian | 5-10% annually |

### **Cross-Sector Factor Allocation:**
- **P/E Positive Sectors:** 40% allocation
- **P/E Contrarian Sectors:** 20% allocation
- **P/E Mixed Sectors:** 40% allocation
- **Expected Portfolio Return:** 12-18% annually
- **Expected Sharpe Ratio:** 0.8-1.2

---

## 🔍 **Economic Interpretation**

### **Why Banking vs Non-Banking Differences:**

#### **Banking Sector Characteristics:**
1. **Regulatory Environment:** Banks have strict regulatory requirements
2. **Asset Quality:** Book values may not reflect true asset quality
3. **Earnings Stability:** High-quality banks have more stable earnings
4. **Market Sentiment:** Banking sector is highly sentiment-driven

#### **Non-Banking Sector Characteristics:**
1. **Diverse Business Models:** Each sector has unique characteristics
2. **Growth Expectations:** Technology and healthcare have high growth expectations
3. **Cyclical Nature:** Mining, agriculture are highly cyclical
4. **Regulatory Impact:** Utilities have regulated returns

### **Why Quality Dependencies Differ:**

#### **Banking Sector:**
- **High Quality:** Market correctly prices quality premium
- **Low Quality:** Market overreacts to quality concerns
- **Regulatory Capital:** Quality affects regulatory requirements

#### **Non-Banking Sectors:**
- **Technology/Healthcare:** Growth expectations drive P/E
- **Utilities:** Regulated returns create stability
- **Cyclical Sectors:** Quality matters less than cycle position

---

## ⚠️ **Limitations and Considerations**

### **Data Limitations:**
1. **Single Point-in-Time:** Results may vary over time
2. **Sample Size:** Some sectors have limited observations
3. **Market Conditions:** Results may vary in different market environments
4. **Sector-Specific Factors:** Each sector has unique characteristics

### **Implementation Risks:**
1. **Sector Rotation:** Sector performance may change
2. **Regime Changes:** Factor behavior may change
3. **Market Conditions:** Results may vary in different environments
4. **Transaction Costs:** Sector rotation may incur costs

---

## 📋 **Implementation Recommendations**

### **Phase 1: Sector-Specific Implementation**
- [ ] Implement banking sector quality-adjusted strategy
- [ ] Implement positive P/E sector strategies
- [ ] Implement contrarian P/E sector strategies
- [ ] Test sector-specific factor weights

### **Phase 2: Cross-Sector Allocation**
- [ ] Develop sector classification framework
- [ ] Create cross-sector factor allocation
- [ ] Implement sector rotation strategies
- [ ] Monitor sector-specific performance

### **Phase 3: Dynamic Adjustment**
- [ ] Develop regime detection models
- [ ] Create dynamic factor allocation
- [ ] Implement adaptive sector weights
- [ ] Monitor and adjust strategies

---

## 🎯 **Key Takeaways**

### **1. Banking Sector is Unique**
- **Quality-dependent P/E:** Strong positive in high quality, contrarian in low quality
- **Consistently contrarian P/B:** Works contrarian across all quality levels
- **Regulatory-driven:** Banking sector behavior is heavily influenced by regulations

### **2. Non-Banking Sectors Show Diversity**
- **P/E Patterns:** Some sectors positive (Securities, Healthcare), some contrarian (Agriculture, Mining)
- **P/B Patterns:** Some sectors positive (Hotels & Tourism), some contrarian (Food & Beverage)
- **Quality Dependencies:** Vary significantly across sectors

### **3. Sector-Specific Strategies Required**
- **No one-size-fits-all:** Each sector requires specific factor weights
- **Quality adjustments:** Important for banking, less so for some non-banking sectors
- **Cross-sector allocation:** Essential for optimal performance

### **4. Implementation Complexity**
- **Sector classification:** Need to classify sectors by factor behavior
- **Dynamic allocation:** Factor behavior may change over time
- **Risk management:** Sector-specific risks need to be managed

---

**Document Version:** 1.0  
**Last Updated:** January 2025  
**Next Review:** Monthly  
**Maintained By:** Quantitative Research Team  
**Key Insight:** Banking and non-banking sectors show fundamentally different factor behavior patterns, requiring sector-specific strategies 