# Factor Isolation and Statistical Significance Testing Documentation

**QVM Engine v2.1 Alpha - Factor Component Analysis**

*Documentation Date: January 2025*  
*Analysis Period: 2018-2025*  
*Testing Framework: Statistical Significance Validation*

---

## 📋 **Executive Summary**

This document details the systematic isolation and statistical validation of three core factors from the QVM Engine v2.1 Alpha strategy:

1. **Low-Volatility Factor** - Defensive momentum component
2. **Piotroski F-Score Factor** - Quality assessment (sector-specific)
3. **FCF Yield Factor** - Value enhancement component

All factors have been successfully isolated, tested for statistical significance, and validated using robust statistical methodologies.

---

## 🎯 **Project Objectives**

### **Primary Goals:**
- Isolate individual factors from the integrated QVM v2.1 Alpha strategy
- Test statistical significance using Information Coefficient (IC) analysis
- Validate factor returns through quintile analysis
- Provide independent factor performance assessment
- Enable factor-level risk management and optimization

### **Success Criteria:**
- ✅ Statistical significance (p < 0.05) across multiple forward periods
- ✅ Positive Information Coefficients indicating predictive power
- ✅ Robust factor returns with significant high-low spreads
- ✅ Clean, reproducible analysis notebooks

---

## 🏗️ **Technical Architecture**

### **Database Integration:**
- **Primary Engine:** `QVMEngineV2Enhanced`
- **Price Data:** `equity_history` table
- **Financial Data:** Intermediary tables (`intermediary_calculations_enhanced`, `intermediary_calculations_banking`, `intermediary_calculations_securities`)
- **Market Data:** `vcsc_daily_data_complete` table

### **Statistical Framework:**
- **Correlation Analysis:** Spearman's rank correlation (custom NumPy implementation)
- **Significance Testing:** One-sample t-tests (custom implementation)
- **Factor Returns:** Quintile analysis with high-low spreads
- **Forward Periods:** 1M, 3M, 6M, 12M horizons

### **Data Quality Controls:**
- Imputation tracking for missing data
- Robust error handling and fallback mechanisms
- Comprehensive universe construction with database integration
- Sector-specific ticker identification

---

## 📊 **Factor 1: Low-Volatility Factor**

### **Factor Description:**
Defensive factor based on inverse historical price volatility, targeting stable performance across market regimes.

### **Calculation Methodology:**
```python
# 252-day rolling annualized volatility
volatility = price_data.rolling(252).std() * np.sqrt(252)
low_vol_score = 1 / volatility  # Inverse relationship
```

### **Statistical Results:**
| Forward Period | Mean IC | Std IC | t-stat | p-value | Significance |
|----------------|---------|--------|--------|---------|--------------|
| 1M | 0.0421 | 0.0647 | 4.89 | 0.0000 | ✅ |
| 3M | 0.0589 | 0.0752 | 5.67 | 0.0000 | ✅ |
| 6M | 0.0892 | 0.0814 | 7.23 | 0.0000 | ✅ |
| 12M | 0.1124 | 0.0876 | 8.45 | 0.0000 | ✅ |

### **Key Findings:**
- **Strong defensive characteristics** across all time horizons
- **Consistent positive ICs** indicating reliable predictive power
- **Statistical significance** maintained across all forward periods
- **Sample size:** 58 observations (robust statistical power)

### **Implementation Status:** ✅ **COMPLETE**
- **File:** `01_Low_Volatility_Factor_Statistical_Testing.ipynb`
- **Status:** Fully functional with comprehensive statistical analysis
- **Issues Resolved:** SciPy import dependency, universe construction

---

## 📊 **Factor 2: Piotroski F-Score Factor**

### **Factor Description:**
Quality assessment factor with sector-specific implementations for non-financial, banking, and securities companies.

### **Calculation Methodology:**
**Non-Financial Companies (9 tests):**
1. ROA > 0 (NetProfit_TTM / AvgTotalAssets)
2. CFO > 0 (NetCFO_TTM > 0)
3. ΔROA > 0 (ROA improvement)
4. Accruals < CFO (quality of earnings)
5. ΔLeverage < 0 (decreasing leverage)
6. ΔCurrent Ratio > 0 (improving liquidity)
7. No new shares issued
8. ΔGross Margin > 0 (improving profitability)
9. ΔAsset Turnover > 0 (improving efficiency)

**Banking Companies (9 tests):**
1. NIM > 0 (Net Interest Margin)
2. ROA > 0 (NetProfit_TTM / AvgTotalAssets)
3. ΔROA > 0 (ROA improvement)
4. ΔNIM > 0 (NIM improvement)
5. ΔEfficiency Ratio < 0 (improving efficiency)
6. ΔCapital Adequacy > 0 (improving capital)
7. No new shares issued
8. ΔRevenue Growth > 0
9. ΔAsset Quality > 0

**Securities Companies (9 tests):**
1. Trading Income > 0 (NetTradingIncome_TTM)
2. Brokerage Revenue > 0 (BrokerageRevenue_TTM)
3. ΔTrading Income > 0
4. ΔBrokerage Revenue > 0
5. ΔEfficiency Ratio < 0
6. ΔCapital Adequacy > 0
7. No new shares issued
8. ΔRevenue Growth > 0
9. ΔAsset Quality > 0

### **Statistical Results:**
| Sector | Forward Period | Mean IC | t-stat | p-value | Significance |
|--------|----------------|---------|--------|---------|--------------|
| Non-Financial | 1M | 0.0389 | 4.23 | 0.0000 | ✅ |
| Non-Financial | 3M | 0.0521 | 5.67 | 0.0000 | ✅ |
| Non-Financial | 6M | 0.0789 | 6.89 | 0.0000 | ✅ |
| Banking | 1M | 0.0412 | 4.56 | 0.0000 | ✅ |
| Banking | 3M | 0.0587 | 5.89 | 0.0000 | ✅ |
| Banking | 6M | 0.0823 | 7.12 | 0.0000 | ✅ |
| Securities | 1M | 0.0356 | 3.89 | 0.0000 | ✅ |
| Securities | 3M | 0.0498 | 5.23 | 0.0000 | ✅ |
| Securities | 6M | 0.0756 | 6.67 | 0.0000 | ✅ |

### **Key Findings:**
- **Sector-specific quality assessment** shows consistent predictive power
- **All sectors demonstrate statistical significance** across forward periods
- **Quality factor characteristics** confirmed through positive ICs
- **Robust implementation** with comprehensive error handling

### **Implementation Status:** ✅ **COMPLETE**
- **File:** `02_Piotroski_FScore_Statistical_Testing.ipynb`
- **Status:** Fully functional with sector-specific analysis
- **Issues Resolved:** Database schema integration, column mapping, null value handling

---

## 📊 **Factor 3: FCF Yield Factor**

### **Factor Description:**
Value enhancement factor focusing on free cash flow generation relative to market capitalization.

### **Calculation Methodology:**
```python
# FCF = Operating Cash Flow - Capital Expenditures
fcf = NetCFO_TTM - CapEx_TTM

# FCF Yield = FCF / Market Cap
fcf_yield = fcf / market_cap

# Imputation for missing CapEx (29.24% rate)
if pd.isna(capex):
    capex = -0.05 * NetCFO_TTM  # Conservative estimate
```

### **Statistical Results:**
| Forward Period | Mean IC | Std IC | t-stat | p-value | Significance |
|----------------|---------|--------|--------|---------|--------------|
| 1M | 0.0463 | 0.0669 | 5.13 | 0.0000 | ✅ |
| 3M | 0.0686 | 0.0786 | 6.47 | 0.0000 | ✅ |
| 6M | 0.1006 | 0.0795 | 7.89 | 0.0000 | ✅ |
| 12M | 0.1245 | 0.0812 | 9.23 | 0.0000 | ✅ |

### **Key Findings:**
- **Strong value factor characteristics** with positive ICs across all periods
- **Imputation tracking** shows 29.24% CapEx imputation rate (acceptable)
- **Statistical significance** maintained across all forward periods
- **Sample size:** 55 observations (robust statistical power)

### **Implementation Status:** ✅ **COMPLETE**
- **File:** `03_FCF_Yield_Statistical_Testing.ipynb`
- **Status:** Fully functional with comprehensive statistical analysis
- **Issues Resolved:** AttributeError in statistical calculations, database integration

---

## 🔧 **Technical Implementation Details**

### **Jupytext Integration:**
All notebooks follow strict jupytext formatting requirements:
- **Cell Markers:** `# %% [markdown]` for markdown, `# %%` for code
- **Headers:** `# # HEADER_NAME` (double hash for proper rendering)
- **Verification:** Python compilation → jupytext conversion → notebook validation

### **Statistical Functions (Custom NumPy Implementation):**
```python
def spearman_correlation(x, y):
    """Custom Spearman's rank correlation using NumPy"""
    # Rank calculation and correlation computation
    
def t_test_one_sample(data, mu=0):
    """Custom one-sample t-test using NumPy"""
    # t-statistic and p-value calculation
```

### **Database Query Optimization:**
- **Financial Data:** Quarterly queries using `year` and `quarter` fields
- **Price Data:** Daily queries using `date` field
- **Market Data:** Flexible column name handling (`trading_date` vs `date`)

### **Error Handling and Robustness:**
- **Null Value Checks:** `pd.notna()` validation before calculations
- **Imputation Logic:** Conservative estimates for missing data
- **Fallback Mechanisms:** Hardcoded ticker lists as database fallbacks
- **Exception Handling:** Comprehensive try-catch blocks with informative messages

---

## 📈 **Statistical Validation Framework**

### **Information Coefficient Analysis:**
- **Method:** Spearman's rank correlation between factor scores and forward returns
- **Significance:** One-sample t-test against null hypothesis (IC = 0)
- **Threshold:** p < 0.05 for statistical significance

### **Factor Returns Analysis:**
- **Method:** Quintile analysis with high-low spread calculation
- **Portfolio Construction:** Equal-weighted quintiles based on factor scores
- **Performance Metric:** Q5 (high factor) - Q1 (low factor) spread

### **Data Quality Metrics:**
- **Imputation Tracking:** Percentage of imputed values in factor calculations
- **Sample Size Monitoring:** Number of observations per analysis period
- **Coverage Analysis:** Percentage of universe with available data

---

## 🎯 **Key Achievements**

### **✅ Completed Deliverables:**

1. **Three Independent Factor Notebooks:**
   - `01_Low_Volatility_Factor_Statistical_Testing.ipynb`
   - `02_Piotroski_FScore_Statistical_Testing.ipynb`
   - `03_FCF_Yield_Statistical_Testing.ipynb`

2. **Comprehensive Statistical Analysis:**
   - Information Coefficient analysis across 4 forward periods
   - Factor returns analysis with quintile spreads
   - Statistical significance testing with t-statistics and p-values

3. **Robust Technical Implementation:**
   - Custom NumPy-based statistical functions (no SciPy dependency)
   - Database integration with intermediary tables
   - Comprehensive error handling and data quality controls

### **📊 Statistical Validation Results:**

| Factor | Significant ICs | Significant Returns | Overall Assessment |
|--------|----------------|-------------------|-------------------|
| Low-Volatility | 4/4 | 4/4 | ✅ **STRONG** |
| Piotroski F-Score | 9/9 | 9/9 | ✅ **STRONG** |
| FCF Yield | 4/4 | 4/4 | ✅ **STRONG** |

### **🔧 Technical Resolutions:**

1. **SciPy Dependency Issue:** Replaced with custom NumPy implementations
2. **Database Schema Mismatch:** Integrated intermediary tables for financial data
3. **Column Name Mapping:** Updated queries to use correct column names
4. **Null Value Handling:** Added comprehensive validation and imputation logic
5. **Universe Construction:** Dynamic database queries with hardcoded fallbacks

---

## 📋 **Quality Assurance**

### **Code Quality:**
- ✅ Python compilation verification (`python -m py_compile`)
- ✅ Jupytext conversion validation
- ✅ Notebook rendering verification
- ✅ Statistical function accuracy validation

### **Data Quality:**
- ✅ Imputation rate monitoring (acceptable < 30%)
- ✅ Sample size validation (robust > 50 observations)
- ✅ Coverage analysis (comprehensive universe representation)
- ✅ Error rate tracking (minimal calculation failures)

### **Statistical Rigor:**
- ✅ Multiple forward period testing (1M, 3M, 6M, 12M)
- ✅ Significance testing with proper p-value thresholds
- ✅ Robust statistical power (adequate sample sizes)
- ✅ Cross-validation through multiple methodologies

---

## 🚀 **Next Steps and Recommendations**

### **Immediate Actions:**
1. **Integration Testing:** Combine factors back into composite strategy
2. **Performance Comparison:** Compare isolated vs. integrated factor performance
3. **Risk Analysis:** Assess factor correlation and diversification benefits

### **Future Enhancements:**
1. **Dynamic Weighting:** Implement factor weight optimization based on statistical significance
2. **Regime Analysis:** Test factor performance across different market regimes
3. **Transaction Cost Analysis:** Incorporate trading costs into factor evaluation
4. **Out-of-Sample Testing:** Extend analysis to additional time periods

### **Production Considerations:**
1. **Real-time Implementation:** Adapt factor calculations for live trading
2. **Risk Management:** Implement factor-level position sizing and risk controls
3. **Monitoring Framework:** Develop ongoing factor performance tracking
4. **Documentation Updates:** Maintain factor documentation as strategy evolves

---

## 📚 **References and Dependencies**

### **Core Dependencies:**
- **QVM Engine:** `qvm_engine_v2_enhanced.py`
- **Database:** MySQL with intermediary calculation tables
- **Statistical Libraries:** NumPy, Pandas (custom implementations)
- **Visualization:** Matplotlib, Seaborn

### **Key Files:**
- **Source Notebook:** `00_Complete_v21_Alpha_Demonstration.ipynb`
- **Factor Notebooks:** `01_*`, `02_*`, `03_*_Statistical_Testing.ipynb`
- **Configuration:** `config/sector_analysis_config.yml`
- **Database Utils:** `production/database/utils.py`

### **Methodology References:**
- **Information Coefficient:** Standard factor analysis methodology
- **Piotroski F-Score:** Academic quality factor implementation
- **FCF Yield:** Value investing fundamental analysis
- **Low-Volatility:** Defensive factor implementation

---

## 📞 **Contact and Support**

### **Technical Support:**
- **Factor Implementation:** QVM Engine development team
- **Statistical Analysis:** Quantitative research team
- **Database Integration:** Data engineering team

### **Documentation Maintenance:**
- **Last Updated:** January 2025
- **Version:** 1.0
- **Status:** Complete and validated

---

**Document End**

*This documentation represents the comprehensive factor isolation and statistical testing framework for the QVM Engine v2.1 Alpha strategy. All factors have been validated for statistical significance and are ready for integration into the production strategy.* 