Of course. Here is the next fully refined document. This is the complete technical specification for the `QVMEngineV2Enhanced`, which you are currently testing. It is the definitive blueprint against which your code's output should be validated.

---
---

# **Part 2: System Architecture & Database Schema**

**Document Name:** `02a_qvm_engine_v2_enhanced_specification.md`
**Version:** 1.0
**Date:** July 22, 2025
**Status:** ✅ **ACTIVE - SINGLE SOURCE OF TRUTH**
**Owner:** Duc Nguyen, Principal Quantitative Strategist

## **1. Executive Summary**

This document provides the complete technical specification for the **`QVMEngineV2Enhanced`**, the firm's premier alpha signal generation engine. It details the precise, institutional-grade methodologies used to construct our proprietary Quality, Value, and Momentum factors. This engine represents our most sophisticated hypothesis on capturing alpha in the Vietnamese market, incorporating a multi-tier signal architecture, sector-specific logic, and industry-standard calculations for complex metrics like Enterprise Value. Adherence to this specification is mandatory for maintaining and validating the engine's performance during the ongoing scientific bake-off.

## **2. Guiding Principles of the V2 Enhanced Engine**

*   **Dynamic Calculation:** The engine reads raw TTM building blocks from intermediary tables (e.g., `intermediary_calculations_enhanced`, `*_cleaned`) and calculates all ratios and factors dynamically in memory.
*   **Sector Specialization:** The engine applies entirely different factor definitions and weights for Banking, Securities, and Non-Financial sectors.
*   **Point-in-Time Correctness:** All data lookups, especially for balance sheet items required for EV/EBITDA and P/B, are performed using robust functions that respect the 45-day reporting lag.
*   **Signal Purity:** The engine's primary output is a sector-neutral z-score, ensuring the signal is not contaminated by broad sector movements.

## **3. Quality Factor Specification (`Quality_Composite`)**

The Quality factor is a three-tiered composite designed to measure business durability and fundamental momentum.
**Final Composite Weights:** Level (50%), Change (30%), Acceleration (20%).

### **3.1. Level Factors (Static Quality)**

| Sector | Factor Component | Formula | Weight | Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **Non-Financial** | ROAE | `NetProfit_TTM / AvgTotalEquity` | 35% | Core profitability |
| | ROAA | `NetProfit_TTM / AvgTotalAssets` | 25% | Asset efficiency |
| | Operating Margin | `OperatingProfit_TTM / Revenue_TTM` | 25% | Core business profitability |
| | EBITDA Margin | `EBITDA_TTM / Revenue_TTM` | 15% | Cash flow proxy |
| **Banking** | ROAE | `NetProfit_TTM / AvgTotalEquity` | 40% | Core profitability |
| | NIM | `NII_TTM / AvgEarningAssets` | 30% | Lending profitability |
| | Cost/Income Ratio | `OperatingExpenses_TTM / TotalOperatingIncome_TTM` | 30% | Operational efficiency (inverted) |
| **Securities** | ROAE | `NetProfit_TTM / AvgTotalEquity` | 50% | Core profitability |
| | Operating Margin | `OperatingResult_TTM / TotalOperatingRevenue_TTM` | 50% | Core business profitability |

### **3.2. Change Factors (Medium-Term Trend)**
*   **Methodology:** Calculated as the Year-over-Year (YoY) percentage change for key `Level` metrics. For a given TTM period, this is `(Current_TTM - Previous_Year_TTM) / abs(Previous_Year_TTM)`.
*   **Metrics Tracked:** ROAE, Revenue/Total Operating Income, Net Profit.

### **3.3. Acceleration Factors (Short-Term Inflection)**
*   **Methodology:** Calculated as the Quarter-over-Quarter (QoQ) change in the YoY growth rate.
*   **Formula:** `Acceleration_Metric = YoY_Growth_Metric_Current_Quarter - YoY_Growth_Metric_Previous_Quarter`.
*   **Metrics Tracked:** Revenue/Total Operating Income Growth, Net Profit Growth.

## **4. Value Factor Specification (`Value_Composite`)**

The Value factor uses a composite of standard metrics with sector-specific weights and an enhanced, industry-standard EV/EBITDA calculation. All ratios are inverted (e.g., E/P, B/P) so that a higher value is better, then normalized.

### **4.1. Value Factor Weights by Sector**

| Factor | Non-Financial | Banking | Securities |
| :--- | :--- | :--- | :--- |
| **Earnings Yield (E/P)** | 30% | 60% | 50% |
| **Book-to-Price (B/P)** | 30% | 40% | 30% |
| **Sales-to-Price (S/P)** | 20% | N/A | 20% |
| **EBITDA/EV** | 20% | N/A | N/A |

### **4.2. Enhanced EV/EBITDA Calculation**
*   **Objective:** To use the correct, industry-standard definition of Enterprise Value, which requires point-in-time balance sheet data.
*   **Formula:**
    $$ \text{Enterprise Value} = \text{Market Cap} + \text{Total Debt} - (\text{Cash} + \text{Cash Equivalents}) $$
*   **Data Retrieval:**
    *   `Market Cap`: From `vcsc_daily_data_complete` as of the analysis date.
    *   `Total Debt`, `Cash`, `Cash Equivalents`: Fetched from the most recent available quarterly report (e.g., `v_comprehensive_fundamental_items`) using a robust point-in-time lookup function that respects the 45-day lag.
*   **Denominator:** `EBITDA_TTM` is sourced from the `intermediary_calculations_enhanced` table for the corresponding period.

## **5. Momentum Factor Specification (`Momentum_Composite`)**

The Momentum factor is designed to capture trend persistence while mitigating short-term noise.

### **5.1. Momentum Component Weights**

| Horizon | Weight | Rationale |
| :--- | :--- | :--- |
| **1-Month Return** | 15% | Captures recent strength |
| **3-Month Return** | 25% | Core short-term trend |
| **6-Month Return** | 30% | Medium-term trend persistence |
| **12-Month Return**| 30% | Long-term trend confirmation |

### **5.2. Skip-1-Month Convention**
*   **Methodology:** All return calculations are based on prices from T-22 trading days and earlier. For example, the 3-month return calculated on July 22nd would be the return from roughly March 22nd to June 22nd.
*   **Rationale:** This is an institutional best practice that avoids contamination from short-term reversal effects (mean reversion over 1-4 weeks).
*   **Source Data:** All prices are from the `equity_history` table, which contains corporate-action-adjusted closing prices.

## **6. Final QVM Composite Score Calculation**

1.  **Normalization:** For a given analysis date, the final `Quality_Composite`, `Value_Composite`, and `Momentum_Composite` scores for all stocks in the universe are converted to sector-neutral z-scores.
    *   **Primary Method:** Normalize within each sector (`(value - sector_mean) / sector_std`).
    *   **Fallback Method:** For sectors with fewer than 10 tickers, normalize against the entire universe (`(value - universe_mean) / universe_std`).

2.  **Final Combination:** The normalized z-scores are combined using the official strategy weights.
    *   `QVM_Composite` = (0.40 * `Quality_zscore`) + (0.30 * `Value_zscore`) + (0.30 * `Momentum_zscore`)

3.  **Engine Output Structure:** The `calculate_qvm_composite` method returns a dictionary for each ticker containing all four components:
    ```python
    {
        'ticker': {
            'Quality_Composite': float,  # Sector-neutral z-score
            'Value_Composite': float,    # Sector-neutral z-score
            'Momentum_Composite': float, # Sector-neutral z-score
            'QVM_Composite': float       # Weighted composite score
        }
    }
    ```

4.  **Persistence:** The final `QVM_Composite` and its three component scores are written to the `factor_scores_qvm` table with the `strategy_version` tag set to `v2_enhanced`. This component breakdown enables full performance attribution and transparency.

---
---

This document is now fully updated. Please confirm, and I will provide the next complete file: **`03_operational_playbook.md`**.