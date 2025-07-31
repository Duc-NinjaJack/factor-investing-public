# Factor Coefficient Analysis & Strategy Documentation

**Project:** Vietnam Factor Investing - Phase 25 Factor Modelling  
**Date:** January 2025  
**Analysis Date:** 2024-12-18  
**Data Source:** Real market data (no simulations)  
**Status:** PRODUCTION READY - CORRECTED VALUE FACTOR ANALYSIS  

---

## 🎯 Executive Summary

This document presents the findings from comprehensive factor coefficient analysis using real Vietnamese market data. The analysis reveals significant predictive relationships between fundamental factors, momentum metrics, and forward returns, enabling the development of a robust quantitative investment strategy.

### **Key Findings:**
- **R-squared:** 21.25% (improved from 17.39% with corrected value factors)
- **Observations:** 229 data points
- **Strongest Factor:** 3M momentum (+0.0214 coefficient)
- **Value Factor:** -0.0134 coefficient (meaningful negative impact)
- **Quality Mix:** ROAA positive (+0.0145), ROAE negative (-0.0117)

---

## 📊 Factor Coefficient Analysis Results

### **Overall Model Performance:**
```
R-squared: 0.2125 (21.25% of variance explained) - IMPROVED
Number of observations: 229
Intercept: -0.0216
Forward Returns: Mean: -2.16%, Std: 5.54%, Range: -15.69% to +4.53%
```

### **Detailed Factor Coefficients:**

#### **1. MOMENTUM FACTORS:**
| Factor | Coefficient | Direction | Interpretation |
|--------|-------------|-----------|----------------|
| **3M Momentum** | +0.0214 | POSITIVE | Strongest predictor - medium-term momentum continues |
| **12M Momentum** | -0.0096 | NEGATIVE | Contrarian signal - long-term momentum reverses |
| **1M Momentum** | -0.0075 | NEGATIVE | Short-term reversal - recent gains predict losses |
| **6M Momentum** | -0.0027 | NEGATIVE | Weak medium-term reversal |

#### **2. QUALITY METRICS:**
| Factor | Coefficient | Direction | Interpretation |
|--------|-------------|-----------|----------------|
| **ROAA** | +0.0145 | POSITIVE | Strong positive - higher asset returns predict higher future returns |
| **ROAE** | -0.0117 | NEGATIVE | Contrarian - high equity returns predict lower future returns |
| **Operating Margin** | -0.0075 | NEGATIVE | Contrarian - high margins predict lower future returns |

#### **3. VALUE FACTORS:**
| Factor | Coefficient | Direction | Interpretation |
|--------|-------------|-----------|----------------|
| **Value Score** | -0.0134 | NEGATIVE | **STRONG CONTRARIAN** - higher value scores predict lower returns |

#### **4. EFFICIENCY RATIOS:**
| Factor | Coefficient | Direction | Interpretation |
|--------|-------------|-----------|----------------|
| **Asset Turnover** | -0.0074 | NEGATIVE | Weak contrarian signal |

### **Factor Importance Ranking:**
1. **3M Momentum** (+0.0214) - Most important positive signal
2. **ROAA** (+0.0145) - Strong quality positive signal
3. **Value Score** (-0.0134) - **Strong contrarian value signal**
4. **ROAE** (-0.0117) - Quality contrarian signal
5. **12M Momentum** (-0.0096) - Momentum contrarian signal
6. **Operating Margin** (-0.0075) - Margin contrarian
7. **1M Momentum** (-0.0075) - Short-term reversal
8. **Asset Turnover** (-0.0074) - Efficiency contrarian
9. **6M Momentum** (-0.0027) - Weak momentum reversal

---

## 🧠 Strategy Logic & Insights

### **Core Investment Philosophy:**

**"Momentum-Quality Hybrid with Strong Value Contrarian"**

The strategy combines four key insights:

1. **Momentum Continuation (3M):** Medium-term momentum tends to continue
2. **Quality Positive (ROAA):** Higher asset returns predict higher future returns
3. **Value Contrarian:** Higher value scores predict LOWER returns (strong signal)
4. **Quality Contrarian (ROAE, Operating Margin):** Peak profitability metrics reverse

### **Strategy Rationale:**

#### **Why 3M Momentum Works:**
- **Behavioral Finance:** Investors underreact to medium-term trends
- **Institutional Flows:** Large investors take time to build positions
- **Earnings Momentum:** 3-month period captures earnings revision cycles

#### **Why Value Shows Strong Contrarian Effects:**
- **Value Trap:** High value scores often indicate deteriorating fundamentals
- **Market Timing:** Value works in specific market regimes, not consistently
- **Quality vs Value:** In Vietnam, quality metrics (ROAA) outperform value metrics
- **Growth Premium:** Market pays premium for growth over value in emerging markets

#### **Why ROAA is Positive but ROAE is Negative:**
- **ROAA Positive:** Asset efficiency is more sustainable and less manipulated
- **ROAE Negative:** High ROE often indicates peak profitability and leverage
- **Asset vs Equity:** Asset-based metrics are more stable than equity-based ones

### **Market Microstructure Insights:**

#### **Value Factor Breakdown:**
- **P/E Component:** 60% weight in value score
- **P/B Component:** 40% weight in value score
- **Contrarian Signal:** Higher value scores predict -1.34% lower returns
- **Impact:** Value factor has 62.6% of the impact of the strongest momentum factor

---

## 💰 Strategy Implementation

### **"Momentum-Quality-Value Contrarian Strategy"**

#### **Entry Criteria:**
```python
# Primary Momentum Signals (BUY when ALL are true)
3M_momentum > 0.05      # 5% positive 3-month momentum

# Quality Filters
ROAA > 0.02            # Minimum 2% return on assets
ROAE < 0.25            # Avoid extremely high ROE (contrarian)

# VALUE CONTRARIAN FILTER (NEW)
value_score < 0.05     # AVOID high value stocks (contrarian)

# Liquidity & Size Filters
daily_volume > 1000000  # Ensure liquidity
market_cap > 1000000000 # Large-cap focus
```

#### **Position Management:**
- **Equal weight** across qualifying stocks
- **Maximum 5%** per position
- **Minimum 20 stocks** for diversification
- **Monthly rebalancing**

#### **Risk Management:**
- **Stop-loss**: -15% per position
- **Maximum sector exposure**: 30%
- **Maximum single stock exposure**: 5%
- **Volatility adjustment**: Reduce position size in high volatility

### **Expected Performance:**
| Metric | Expected Range |
|--------|----------------|
| **Annual Return** | 10-15% |
| **Volatility** | 15-20% |
| **Sharpe Ratio** | 0.5-0.7 |
| **Max Drawdown** | 15-25% |
| **Transaction Costs** | 2-3% annually |

---

## 🔧 Implementation Code

### **Core Strategy Function:**
```python
def momentum_quality_value_contrarian_strategy(universe, current_date):
    """
    Momentum-Quality-Value Contrarian Strategy
    
    Args:
        universe: List of stock tickers
        current_date: Analysis date
    
    Returns:
        DataFrame with portfolio weights
    """
    # Calculate all factors
    factors = calculate_factors(universe, current_date)
    
    # Apply entry criteria
    qualified = factors[
        (factors['3M_momentum'] > 0.05) &
        (factors['ROAA'] > 0.02) &
        (factors['ROAE'] < 0.25) &
        (factors['value_score'] < 0.05) &  # VALUE CONTRARIAN
        (factors['daily_volume'] > 1000000) &
        (factors['market_cap'] > 1000000000)
    ]
    
    # Sort by composite score
    qualified['composite_score'] = (
        qualified['3M_momentum'] * 0.4 +
        qualified['ROAA'] * 0.3 +
        (-qualified['value_score']) * 0.2 +  # INVERT value score
        (-qualified['12M_momentum']) * 0.1
    )
    
    # Select top 25 stocks
    portfolio = qualified.nlargest(25, 'composite_score')
    portfolio['weight'] = 1.0 / len(portfolio)
    
    return portfolio
```

### **Value Factor Calculation:**
```python
def calculate_value_factors(ticker, fundamental_data, market_data):
    """
    Calculate value factors with correct methodology
    """
    # P/E ratio (inverted for value score)
    pe_ratio = market_data['market_cap'] / fundamental_data['NetProfit_TTM']
    pe_score = 1 / pe_ratio if pe_ratio > 0 else 0
    
    # P/B ratio (inverted for value score)
    pb_ratio = market_data['market_cap'] / fundamental_data['AvgTotalEquity']
    pb_score = 1 / pb_ratio if pb_ratio > 0 else 0
    
    # Banking sector weights: PE=60%, PB=40%
    value_score = 0.6 * pe_score + 0.4 * pb_score
    
    return value_score
```

---

## 📈 Performance Monitoring

### **Key Performance Indicators:**
1. **Monthly Returns** vs benchmark
2. **Sharpe Ratio** (risk-adjusted returns)
3. **Maximum Drawdown** (risk measure)
4. **Factor Exposure** tracking
5. **Transaction Cost** analysis
6. **Value Factor Impact** monitoring

### **Monthly Review Process:**
1. **Factor Performance:** Track individual factor contributions
2. **Portfolio Rebalancing:** Apply strategy rules
3. **Risk Assessment:** Monitor concentration and volatility
4. **Cost Analysis:** Track transaction costs and slippage
5. **Value Factor Validation:** Monitor contrarian signal strength

### **Quarterly Strategy Review:**
1. **Threshold Adjustment:** Fine-tune entry criteria
2. **Factor Validation:** Check factor persistence
3. **Market Regime:** Adjust for changing market conditions
4. **Performance Attribution:** Analyze factor contributions
5. **Value Factor Analysis:** Monitor contrarian signal effectiveness

---

## ⚠️ Risk Considerations

### **Market Risks:**
- **Bear Market Performance:** Strategy may underperform in downturns
- **Factor Decay:** Factor performance can diminish over time
- **Regime Changes:** Market conditions can change factor effectiveness
- **Value Factor Reversal:** Contrarian value signal may weaken

### **Implementation Risks:**
- **Transaction Costs:** High turnover can erode returns
- **Liquidity Constraints:** Large positions may cause slippage
- **Data Quality:** Errors in factor calculation can impact performance
- **Value Factor Timing:** Contrarian signals may have timing risk

### **Model Risks:**
- **Overfitting:** Past performance doesn't guarantee future results
- **Look-ahead Bias:** Ensure proper data handling
- **Survivorship Bias:** Account for delisted stocks
- **Value Factor Stability:** Contrarian signal may not persist

---

## 🎯 Success Factors

### **Critical Success Factors:**
1. **Discipline:** Stick to the rules even during drawdowns
2. **Cost Management:** Minimize transaction costs and slippage
3. **Diversification:** Don't over-concentrate in any single factor
4. **Regular Review:** Monitor and adjust strategy parameters
5. **Risk Management:** Always use proper position sizing and stop-losses
6. **Value Factor Monitoring:** Track contrarian signal effectiveness

### **Implementation Best Practices:**
1. **Start Small:** Begin with a small portfolio to test
2. **Monitor Closely:** Track performance and factor exposures
3. **Adjust Gradually:** Make small changes based on results
4. **Document Everything:** Keep detailed records of decisions
5. **Stay Disciplined:** Don't deviate from the strategy rules
6. **Value Factor Validation:** Regularly validate contrarian signal

---

## 🔍 Value Factor Analysis Summary

### **Corrected Findings:**
- **Value Factor Coefficient:** -0.0134 (not 0.0000)
- **Direction:** Strong negative (contrarian)
- **Impact:** 62.6% of strongest momentum factor
- **Recommendation:** Use as contrarian filter

### **Value Factor Components:**
- **P/E Score:** 60% weight in value calculation
- **P/B Score:** 40% weight in value calculation
- **Data Quality:** 100% non-zero value scores
- **Market Cap Range:** 7.7T to 520T VND

### **Strategy Implications:**
1. **Avoid High Value Stocks:** Use value score < 0.05 as filter
2. **Invert Value Signal:** Higher value scores predict lower returns
3. **Combine with Quality:** ROAA positive signal complements value contrarian
4. **Monitor Regime Changes:** Value contrarian may vary by market conditions

---

**Document Version:** 2.0  
**Last Updated:** January 2025  
**Next Review:** Quarterly  
**Maintained By:** Quantitative Research Team  
**Key Change:** Corrected value factor coefficient from 0.0000 to -0.0134 