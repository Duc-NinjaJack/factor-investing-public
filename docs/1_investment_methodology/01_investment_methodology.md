# Part 1: Investment Methodology

**Document Name:** `01_investment_methodology.md`
**Version:** 4.0 (Canonical - Flat Methodology)
**Date:** August 5, 2025
**Status:** ✅ **ACTIVE - SINGLE SOURCE OF TRUTH**
**Owner:** Duc Nguyen, Principal Quantitative Strategist

## **Changelog (v3.0 -> v4.0)**
*   **Removed Scientific Bake-Off:** The bake-off is complete; `v2.1.1_flat` is the certified winner.
*   **Updated Master Signal Architecture:** Now describes the final 4-pillar flat composite methodology.
*   **Canonized Engine Reference:** All references updated to `qvm_engine_v2_1_1_flat`.
*   **Added Product Suite:** Formalized Aggressive Growth and Risk-Managed strategies.

## **1. Executive Summary**

This document outlines the complete investment methodology for Aureus Sigma Capital. It serves as the firm's investment constitution, detailing the theoretical foundation, signal construction architecture, and risk management principles that govern our systematic strategies in the Vietnamese equity market.

Our methodology is a synthesis of seminal academic research and proprietary, market-specific enhancements. It is built on a **4-pillar factor model** targeting Quality, Value, Momentum, and Defensive characteristics, implemented via our canonical **`qvm_engine_v2_1_1_flat`**. The core tenets are:

1.  **A Flat, Sector-Neutral Architecture:** We isolate pure, company-specific alpha by converting every individual factor into a sector-neutral z-score before combining them in a single, statistically robust step.
2.  **A Focus on Fundamental Change:** We posit that a significant source of alpha lies not just in a company's static quality or value, but in the *rate of change* of its fundamentals, which we capture via factors like the Piotroski F-Score.
3.  **Integrated Risk Management:** Risk management is integrated into every step, from factor construction (orthogonality) and portfolio construction (constraints) to execution (transaction cost modeling).

This document provides the definitive "Why" and "What" behind our alpha generation process.

## **2. Master Signal Architecture: The 4-Pillar Flat Composite**

Our alpha signal is a composite derived from four distinct, academically-grounded factor pillars. The architecture is a **flat composite**, which preserves the integrity of each underlying signal and provides clear performance attribution.

### **2.1. The Four Pillars**

| Pillar | Objective | Key Factors |
| :--- | :--- | :--- |
| **Quality** | To identify durable, profitable, and fundamentally healthy businesses. | ROAE, Margins, Piotroski F-Score |
| **Value** | To identify companies that are inexpensive relative to their fundamental value. | E/P, B/P, S/P, FCF Yield, EBITDA/EV |
| **Momentum** | To capture trend persistence in stock prices while avoiding short-term noise. | 1, 3, 6, 12-Month Returns (with 1-month skip) |
| **Defensive** | To reduce portfolio volatility and improve risk-adjusted returns. | Low Volatility (Inverse 63-day Volatility) |

### **2.2. The Flat Combination Process**
The final `QVM_Composite` score is generated through a systematic, three-step process executed by the `qvm_engine_v2_1_1_flat`:

1.  **Raw Factor Calculation:** The engine calculates the raw value for every individual factor (e.g., ROAE, FCF Yield, 6-Month Return).
2.  **Universal Sector Neutralization:** Every raw factor is converted into a sector-neutral z-score. This is the primary alpha extraction step.
    ```
    Factor Z-Score = (Factor Value - μ_sector) / σ_sector
    ```
3.  **Pillar & Final Combination:** The individual z-scores are combined in a single, weighted step to form the four pillar composites, which are then combined to create the final `QVM_Composite` score. All weights are sourced from the `strategy_config.yml` file.

## **3. Formalized Product Suite**

Our core engine serves as the foundation for two distinct investment strategies, allowing capital allocators to target specific risk-return profiles.

| Strategy Name | **Aureus Sigma Vietnam Aggressive Growth** | **Aureus Sigma Vietnam Risk-Managed** |
| :--- | :--- | :--- |
| **Core Engine** | `qvm_engine_v2_1_1_flat.py` | `qvm_engine_v2_1_1_flat.py` + Volatility Overlay |
| **Objective** | Maximize long-term absolute returns | Maximize risk-adjusted returns (Sharpe) |
| **Target Return** | 25%+ Annualized | 15-20% Annualized |
| **Target Drawdown** | < 50% (Accepts high volatility) | **< 25% (Strictly controlled)** |

## **4. Factor Construction & Normalization**

### **4.1. Key Factor Enhancements**
*   **Piotroski F-Score (Quality):** We use sector-specific variants (9-point for Non-Financial, 6-point for Banking, 5-point for Securities) to screen for fundamental health and avoid value traps.
*   **FCF Yield (Value):** We use a robust Free Cash Flow Yield calculation with a fallback to `max(0, -NetCFI_TTM)` as a proxy for Capex, ensuring broad coverage despite variations in Vietnamese accounting disclosures.
*   **Low Volatility (Defensive):** We use the inverse of 63-day rolling price volatility as a direct measure of a stock's recent risk profile.

### **4.2. Cross-Sectional Normalization**
All factors are normalized cross-sectionally to create a standardized, relative ranking.
*   **Primary Method (Z-Score):** We convert each factor value into a z-score relative to its sector peers.
*   **Vietnam-Specific Normalization:** Recognizing the distinct economic behavior of state-owned enterprises, we apply a dual-normalization process where SOEs are normalized only against other SOEs, and private companies are normalized only against other private companies.

## **5. Portfolio Construction & Risk Management**

### **5.1. Portfolio Construction**
Our primary portfolio construction method is a long-only approach based on factor rankings.
1.  **Universe Definition:** Filter for tradable stocks based on liquidity (`constructors.py`).
2.  **Signal Generation:** Calculate the final `QVM_Composite` score for each stock.
3.  **Ranking & Selection:** Go long the top quintile (top 20%) of stocks.
4.  **Weighting:** Positions are typically equal-weighted, subject to concentration limits.

### **5.2. Risk Management & Validation**
*   **Factor Orthogonality:** We enforce a strict orthogonality constraint. The absolute pairwise correlation between any two final pillar composites must be less than **0.4**. This is monitored via automated tests (e.g., `test_pillar_correlations.py`).
*   **Transaction Cost Modeling:** All backtests incorporate a realistic transaction cost model tailored for Vietnam, including commissions, taxes, and a market impact function based on a trade's percentage of average daily volume (ADV).
*   **Implementation Shortfall:** We monitor the live performance of our pilot portfolio against the backtest, targeting an implementation shortfall of less than **50 bps**.

## **6. Theoretical Foundation**

Our methodology synthesizes insights from seminal academic works, including Asness et al. (2019) on Quality, Piotroski (2000) on fundamental screening, and Frazzini et al. (2018) on transaction costs. All factors are required to have a sound economic or behavioral rationale.

---