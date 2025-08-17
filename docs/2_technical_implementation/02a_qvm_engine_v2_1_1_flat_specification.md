# **QVM Engine v2.1.1 Flat - Technical Specification**

**Document Name:** `02a_qvm_engine_v2_1_1_flat_specification.md`
**Version:** 1.0 (Definitive - Flat Methodology)
**Date:** August 5, 2025
**Status:** ✅ **ACTIVE - SINGLE SOURCE OF TRUTH**
**Owner:** Duc Nguyen, Principal Quantitative Strategist

## **1. Executive Summary**

This document provides the complete technical specification for the **`QVMEngineV211Flat`**, the firm's canonical alpha signal generation engine. It details the precise, institutional-grade methodologies used to construct our proprietary Quality, Value, Momentum, and Defensive factors. This engine represents our most sophisticated hypothesis on capturing alpha in the Vietnamese market, incorporating a 4-pillar, 6-factor model built on a statistically superior **flat composite architecture**. Adherence to this specification is mandatory for maintaining and validating the engine's performance.

## **2. Guiding Principles of the `v2.1.1_flat` Engine**

*   **Dynamic Calculation:** The engine reads raw TTM building blocks from intermediary tables and calculates all ratios and factors dynamically in memory.
*   **Universal Sector Neutralization:** Every individual factor (e.g., ROAE, F-Score, FCF Yield, Low Volatility) is converted to a sector-neutral z-score before any combination occurs. This is the core of our alpha extraction process.
*   **Flat Composite Architecture:** Pillar composites are formed in a single, weighted step from the individual z-scored factors. There is no hierarchical nesting of composites, which preserves signal integrity.
*   **Point-in-Time Correctness:** All data lookups respect the 45-day reporting lag, ensuring zero look-ahead bias.

## **3. Factor Specification: The Four Pillars**

### **Pillar 1: Quality (`Quality_Composite`)**
*   **Objective:** To measure business durability, profitability, and fundamental health.
*   **Enhanced with:** Piotroski F-Score.

| Factor Component | Formula / Methodology | Sector Applicability |
| :--- | :--- | :--- |
| **ROAE** | `NetProfit_TTM / AvgTotalEquity` | All |
| **ROAA** | `NetProfit_TTM / AvgTotalAssets` | Banking |
| **Margins** | `NetProfit_Margin`, `Gross_Margin`, `Operating_Margin`, `EBITDA_Margin` | Non-Financial, Securities |
| **NIM** | `NII_TTM / AvgEarningAssets` | Banking |
| **Cost/Income Ratio** | `1 - (abs(OperatingExpenses_TTM) / TotalOperatingIncome_TTM)` | Banking |
| **Piotroski F-Score** | 9, 6, or 5 binary tests based on profitability, leverage, and efficiency. | All (with sector-specific variants) |

### **Pillar 2: Value (`Value_Composite`)**
*   **Objective:** To identify companies that are inexpensive relative to their fundamental value.
*   **Enhanced with:** FCF Yield.

| Factor Component | Formula / Methodology | Sector Applicability |
| :--- | :--- | :--- |
| **Earnings Yield (E/P)** | `NetProfit_TTM / MarketCap` | All |
| **Book-to-Price (B/P)** | `PointInTimeEquity / MarketCap` | All |
| **Sales-to-Price (S/P)** | `SectorSpecificRevenue_TTM / MarketCap` | All (with sector-specific revenue) |
| **EBITDA/EV** | `EBITDA_TTM / (MarketCap + Debt - Cash)` | Non-Financial |
| **FCF Yield** | `(NetCFO_TTM - Capex) / MarketCap` | Non-Financial |

### **Pillar 3: Momentum (`Momentum_Composite`)**
*   **Objective:** To capture trend persistence in stock prices.

| Factor Component | Formula / Methodology | Sector Applicability |
| :--- | :--- | :--- |
| **1, 3, 6, 12-Month Returns** | Total return over the specified lookback period, calculated with a **1-month skip** to avoid short-term reversal effects. Prices are from `equity_history` (adjusted). | All |

### **Pillar 4: Defensive (`Defensive_Composite`)**
*   **Objective:** To reduce portfolio volatility and improve risk-adjusted returns.
*   **New Pillar:** This is a new addition in v2.1.1.

| Factor Component | Formula / Methodology | Sector Applicability |
| :--- | :--- | :--- |
| **Low Volatility** | `-1 * σ(daily_returns_63D)` (Inverse of 63-day rolling volatility of daily returns). | All |

## **4. Calculation Flow: From Raw Data to Final Scores**

The engine follows a strict, five-step process for each rebalancing date:

**Step 1: Raw Factor Calculation**
For every stock in the universe, the engine calculates the raw value for all applicable factors.
*   *Example (HPG - Non-Financial):* Calculates raw ROAE, Gross Margin, E/P, FCF Yield, 12M Return, 63D Volatility, and a 9-point F-Score.
*   *Example (VCB - Banking):* Calculates raw ROAE, NIM, E/P, 12M Return, 63D Volatility, and a 6-point F-Score. FCF Yield is not calculated.

**Step 2: Universal Sector Neutralization**
Every raw factor value from Step 1 is converted into a z-score relative to its sector peers.
*   **Formula:** `z_score = (raw_value - sector_mean) / sector_std`
*   **Output:** A clean, comparable set of z-scores for every factor (e.g., `roae_z`, `fcf_yield_z`, `low_volatility_z`).

**Step 3: Pillar Composite Calculation (Flat Combination)**
The engine combines the individual z-scores into the four pillar composites using the weights defined in `strategy_config.yml`.
*   **Formula (`Quality_Composite` for a Non-Financial stock):**
    ```
    (roae_z * w_roae) + (gross_margin_z * w_gm) + ... + (f_score_z * w_fscore)
    ```
*   This is a single, flat weighted average.

**Step 4: Final QVM Composite Calculation**
The four pillar scores are combined using the top-level weights from the configuration file.
*   **Formula (`QVM_Composite`):**
    ```
    (Quality_Composite * 0.35) + (Value_Composite * 0.30) + 
    (Momentum_Composite * 0.20) + (Defensive_Composite * 0.15)
    ```

**Step 5: Persistence**
The engine's final output, a dictionary containing the four pillar scores and the final QVM composite for each ticker, is passed to the `run_factor_generation.py` script, which writes the data to the `factor_scores_qvm` table with `strategy_version = 'qvm_v2.1.1_flat'`.

## **5. Detailed Factor Notes & Special Cases**

*   **Piotroski F-Score:** The raw 0-9 (or 0-6, 0-5) score is first normalized to a 0-1 scale (`raw_score / max_score`) before being converted to a sector-neutral z-score. This makes the factor comparable across sectors with different maximum possible scores.
*   **FCF Yield & Capex:** The engine uses `CapEx_TTM` from `intermediary_calculations_enhanced` when available. If `CapEx_TTM` is null, it falls back to using `max(0, -NetCFI_TTM)` as a proxy and logs the imputation for data quality monitoring.
*   **EV/EBITDA:** This calculation requires a point-in-time lookup to `v_comprehensive_fundamental_items` to get the latest reported `TotalDebt` and `CashAndCashEquivalents`, ensuring the Enterprise Value is calculated with the correct balance sheet data.
*   **Sector-Specific Revenue:** The Sales-to-Price (S/P) calculation correctly uses `Revenue_TTM` for Non-Financials, `NII_TTM` for Banks, and `TotalOperatingRevenue_TTM` for Securities firms.

---
This document is now fully updated. Please confirm, and I will provide the next complete file: the **archival notice for `02a_qvm_engine_v2_enhanced_specification.md`**.