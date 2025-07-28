Of course. As before, the `Factor Validation Playbook` is a cornerstone document whose principles are universal. It requires no substantive changes. We will issue it as Version 1.2 to confirm it has been reviewed and re-ratified as the official standard governing our current scientific bake-off.

---
---

# **Part 3: Operational Framework**

**Document Name:** `03a_factor_validation_playbook.md`
**Version:** 1.2 (Reviewed and Re-ratified for Bake-Off)
**Date:** July 22, 2025
**Status:** ✅ **ACTIVE - SINGLE SOURCE OF TRUTH**
**Owner:** Duc Nguyen, Principal Quantitative Strategist

## **1. Executive Summary**

### **1.1. Purpose and Scope**
This playbook establishes the **institutional-grade validation standards** for all factor development at Aureus Sigma Capital. It provides a systematic, repeatable framework for quality assurance, ensuring that every new factor demonstrates genuine predictive power, economic intuition, and robustness before it is considered for inclusion in our production models. Adherence to this playbook is mandatory for all quantitative research, including the evaluation of new engine versions (e.g., v1 vs. v2) and the exploration of new alpha sources.

### **1.2. Validation Philosophy: "In God we trust, all others must bring data."**
A factor is not an idea; it is a rigorously tested hypothesis. Every proposed factor must successfully pass through the five gates of validation:
1.  **Economic Rationale:** Does it make sense?
2.  **Statistical Significance:** Is the effect real and not random noise?
3.  **Orthogonality:** Is it a new source of alpha, or just a repackaging of an existing factor?
4.  **Signal Persistence:** Does the signal last long enough to be traded profitably?
5.  **Implementation Feasibility:** Can we capture the alpha after accounting for real-world frictions?

### **1.3. Institutional Quality Gates**
A factor is only promoted to "Production Ready" if it meets these minimum thresholds:
*   **Correlation Threshold:** Maximum absolute correlation of **< 0.4** with any existing production factor.
*   **Information Coefficient (IC):** Minimum monthly IC of **> 0.02** (2%) with a t-statistic **> 2.0**.
*   **Signal Decay:** IC half-life of at least **3 months**.
*   **Turnover:** Implied annual portfolio turnover of **< 200%**.
*   **Transaction Cost Impact:** Sharpe ratio degradation after costs of **< 30%**.

## **2. Section 1: Pre-Development Factor Specification**

### **2.1. Economic Hypothesis Framework**
No factor research begins without a documented hypothesis.
*   **Requirement:** Every factor must have a clear economic or behavioral rationale for why it should predict returns.
*   **Template:** A markdown document must be created specifying the factor name, category (Quality, Value, etc.), economic intuition, supporting academic literature, and specific relevance to the Vietnamese market.

### **2.2. Factor Construction Specification**
*   **Requirement:** The precise mathematical formula must be documented, including any Vietnam-specific accounting adjustments (e.g., calculated EBIT).
*   **Data Sources:** All required data items from the `intermediary_calculations_*` tables must be listed.

## **3. Section 2: Statistical Properties Validation**

### **3.1. Descriptive Statistics & Distribution Analysis**
The factor's raw values are analyzed to ensure they are well-behaved.
*   **Methodology:** Calculate mean, median, standard deviation, skewness, and kurtosis.
*   **Quality Gates:**
    *   Missing data must be **< 10%** of the universe.
    *   Infinite values must be **zero**.
    *   The factor must have a plausible distribution (e.g., a P/E ratio should not have a mean of -500).

### **3.2. Outlier Treatment Framework**
*   **Methodology:** We employ an intelligent, peer-benchmark-based outlier treatment strategy.
    1.  **Classification:** Outliers are first compared to their sector median. Values >50x the sector median are flagged as likely data errors. Values between 10-50x are flagged as potential operational distress.
    2.  **Treatment:** Data errors are capped using sector-specific business logic (e.g., DIO for a construction firm is allowed a much longer cycle than for a retailer). This prevents the destruction of legitimate economic information that naive winsorization would cause.

## **4. Section 3: Factor Orthogonality Analysis**

### **4.1. Correlation Matrix Analysis**
*   **Requirement:** The proposed factor must be tested against the full universe of existing production factors.
*   **Methodology:**
    1.  Calculate the Pearson and Spearman correlation matrix between the new factor and all existing factors.
    2.  The maximum absolute correlation must be **< 0.4**.
    3.  Any correlation > 0.2 must be investigated and explained.
*   **Rationale:** This ensures we are adding a diversified source of alpha, not just another proxy for Value or Momentum.

### **4.2. Fama-French Factor Regression**
*   **Requirement:** The new factor's returns must be regressed against a Vietnam-specific Fama-French 5-Factor model (Mkt-RF, SMB, HML, RMW, CMA) plus a proprietary SOE factor.
*   **Methodology:**
    $$ R_{\text{factor}} = \alpha + \beta_1(\text{Mkt-RF}) + \beta_2(\text{SMB}) + ... + \beta_6(\text{SOE}) + \epsilon $$
*   **Quality Gate:** The regression **alpha (α)** must be positive and statistically significant (t-stat > 2.0). This proves the factor generates excess returns even after accounting for common risk premia.

## **5. Section 4: Signal Predictive Power & Persistence**

### **5.1. Information Coefficient (IC) Analysis**
This is the most critical test of a factor's predictive power.
*   **Methodology:**
    1.  At each rebalance date (e.g., end of month), calculate the Spearman rank correlation between the factor's cross-sectional values and the subsequent forward returns (e.g., next month's returns).
    2.  This generates a time-series of monthly ICs.
*   **Quality Gates:**
    *   **Mean IC:** The average of the monthly ICs must be **> 0.02**.
    *   **IC t-statistic:** The mean IC divided by its standard error must be **> 2.0**. This confirms the predictive skill is not due to chance.
    *   **IC Hit Rate:** The percentage of periods with a positive IC should be **> 55%**.

### **5.2. Signal Decay Analysis (IC Half-Life)**
*   **Methodology:** Calculate the mean IC for multiple forward return horizons (1-month, 2-month, 3-month, etc.). The "half-life" is the point at which the IC decays to 50% of its 1-month value.
*   **Quality Gate:** The IC half-life must be at least **3 months**. This ensures the signal is persistent enough to be captured in a monthly or quarterly rebalanced portfolio, especially given Vietnam's T+2 settlement cycle.

## **6. Section 5: Implementation & Transaction Cost Analysis**

### **6.1. Quintile Portfolio Backtest**
*   **Methodology:** A simplified backtest is run to assess the factor's real-world efficacy.
    1.  At each rebalance date, rank all stocks in the universe by the factor value.
    2.  Form five quintile portfolios (Q1 = top 20%, Q5 = bottom 20%).
    3.  Simulate a long-short strategy: **Long Q1, Short Q5**.
*   **Analysis:** The long-short portfolio should exhibit a positive, statistically significant return stream. We analyze the Sharpe ratio, max drawdown, and Calmar ratio of this strategy.

### **6.2. Turnover and Cost Analysis**
*   **Methodology:**
    1.  Calculate the monthly turnover of the Q1-Q5 long-short portfolio.
    2.  Apply our Vietnam-specific transaction cost model to the simulated trades.
*   **Quality Gates:**
    *   Annual turnover should be **< 200%**.
    *   The Sharpe ratio of the strategy after costs should not degrade by more than **30%** from the gross Sharpe ratio.

## **7. Production Deployment Checklist**

A factor is only approved for production use after it has passed every gate in this playbook. The final sign-off requires a comprehensive validation report summarizing the results of each test.

| Validation Category | Status | Key Metric | Result |
| :--- | :--- | :--- | :--- |
| **Economic Hypothesis** | PASS/FAIL | Documented Rationale | Yes/No |
| **Statistical Properties** | PASS/FAIL | Outlier % | < 3% |
| **Orthogonality** | PASS/FAIL | Max Correlation | < 0.4 |
| **Fama-French Alpha** | PASS/FAIL | Alpha t-stat | > 2.0 |
| **Information Coefficient** | PASS/FAIL | Mean IC & t-stat | >0.02 & >2.0 |
| **Signal Persistence** | PASS/FAIL | IC Half-Life | > 3 months |
| **Implementation Costs** | PASS/FAIL | Sharpe Degradation | < 30% |
| **Final Decision** | **APPROVE/REJECT** | | |

---
---

This document has been reviewed and its standards remain the definitive guide for all factor validation at the firm. Please confirm, and I will provide the next complete file: **`04_qvm_backtesting_framework.md`**.