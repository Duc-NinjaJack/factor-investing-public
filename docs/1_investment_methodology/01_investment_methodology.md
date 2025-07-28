# **Part 1: Investment Methodology**

**Document Name:** `01_investment_methodology.md`
**Version:** 3.0 (Definitive Merged Version)
**Date:** July 22, 2025
**Status:** ✅ **ACTIVE - SINGLE SOURCE OF TRUTH**
**Owner:** Duc Nguyen, Principal Quantitative Strategist

## **1. Executive Summary**

This document outlines the complete investment methodology for Aureus Sigma Capital. It serves as the firm's investment constitution, detailing the theoretical foundation, signal construction architecture, and risk management principles that govern our systematic strategies in the Vietnamese equity market.

Our methodology is a synthesis of seminal academic research and the proprietary best practices of world-leading quantitative firms. It is distinguished by three core pillars:
1.  **A Focus on Fundamental Change:** We posit that the greatest source of persistent alpha lies not in a company's static quality or value, but in the *rate of change* of its fundamental characteristics. Our models are explicitly designed to capture this dynamic improvement.
2.  **Rigorous Risk Architecture:** Risk management is not an overlay but is integrated into every step of the process, from factor construction (orthogonality) and portfolio construction (constraints) to execution (transaction cost modeling).
3.  **Vietnam-Specific Adaptation:** We apply global quantitative principles through the specific lens of the Vietnamese market structure, accounting for its unique regulatory environment, accounting standards, and retail-driven flow dynamics.

This document provides the definitive "Why" and "What" behind our alpha generation process.

## **2. Methodology Validation via Scientific Bake-Off**

The transition from a validated factor concept to a production-ready alpha signal is a critical research step. Having established *that* Quality, Value, and Momentum work in Vietnam, we must now determine the most robust and profitable way to *measure* them. To this end, we are conducting a formal **Scientific Bake-Off** to compare two distinct signal construction methodologies.

*   **Hypothesis v1 (The Baseline):** This methodology posits that a simple, direct measurement of our core factors provides a reliable and effective performance benchmark. It is implemented in our `QVMEngineV1Baseline` and serves as the **control group** for our experiment.

*   **Hypothesis v2 (The Enhanced):** This methodology posits that a more sophisticated, multi-tier, and sector-specific signal construction will capture the underlying factor premia more effectively, leading to superior risk-adjusted returns. It is implemented in our `QVMEngineV2Enhanced` and serves as the **experimental group**.

The winner of this bake-off will be determined not by opinion, but by the empirical evidence from parallel backtests. The engine that produces a statistically significant improvement in Sharpe Ratio will be promoted to become the firm's official production engine. This process ensures our final strategy is the product of rigorous, evidence-based selection.

## **3. Formalized Product Suite**

Our core QVM engine serves as the foundation for two distinct investment strategies, allowing capital allocators to target specific risk-return profiles.

| Strategy Name | **Aureus Sigma Vietnam Aggressive Growth** | **Aureus Sigma Vietnam Risk-Managed** |
| :--- | :--- | :--- |
| **Core Engine** | Unaltered QVM Factor Engine | QVM Factor Engine + Volatility Overlay |
| **Objective** | Maximize long-term absolute returns | Maximize risk-adjusted returns (Sharpe) |
| **Target Return** | 25%+ Annualized | 15-20% Annualized |
| **Target Drawdown** | < 50% (Accepts high volatility) | **< 25% (Strictly controlled)** |
| **Target Investor** | Family Offices, HNWIs, Aggressive Funds | Pensions, Endowments, Insurers, Conservative Funds |
| **Benchmark** | Absolute Return / VN-Index + 15% | VN-Index / Volatility-controlled benchmark |

## **4. Theoretical Foundation & Core Principles**

Our investment process is built upon a rigorous academic foundation, adapted for the unique microstructure of the Vietnamese market. We do not innovate for the sake of complexity; we stand on the shoulders of giants and apply proven principles with relentless discipline.

### **4.1. Core Research Citations**

Our methodology synthesizes insights from the following seminal works:
1.  **Asness, Frazzini, and Pedersen (2019). "Quality Minus Junk."** - *Foundation for multi-dimensional quality factor construction.*
2.  **Novy-Marx (2013). "The Other Side of Value: The Gross Profitability Premium."** - *Establishes the primacy of change signals over level signals for alpha generation.*
3.  **Frazzini, Israel, and Moskowitz (2018). "Trading Costs."** - *Provides the framework for our Vietnam-specific transaction cost modeling.*
4.  **Harvey, Liu, and Zhu (2016). "... and the Cross-Section of Expected Returns."** - *Informs our strict approach to factor orthogonality and mitigating data mining.*
5.  **Daniel & Moskowitz (2016). "Momentum crashes."** - *Underpins our use of acceleration signals to capture inflection points and manage risk.*

### **4.2. Core Investment Principles**
*   **Discipline is Alpha:** The consistent application of a sound process is more important than the intermittent discovery of a "perfect" signal.
*   **Risk Management is Paramount:** We seek to generate high risk-adjusted returns. Capital preservation is the primary objective.
*   **Economic Intuition First:** All factors must have a sound economic or behavioral rationale. We do not engage in black-box data mining.
*   **Data is Sovereign:** All hypotheses must be validated with rigorous, bias-free statistical testing.
*   **Costs are Critical:** Paper profits are an illusion. Net returns after realistic transaction costs are the only metric that matters.

## **5. Master Signal Architecture**

The core of our alpha generation process is a three-tier signal architecture. This structure is based on the empirical finding that while factor levels provide context, the most potent and persistent alpha is found in their rates of change. This architecture is a key feature of the `v2_enhanced` engine currently under evaluation.

### **5.1. Three-Tier Signal Construction**

The composite alpha score for any given factor is a weighted combination of three distinct signal types:

$$ \text{Composite Alpha Score} = w_L \cdot \text{Level} + w_C \cdot \text{Change} + w_A \cdot \text{Acceleration} $$

Where:
*   **Level Signal (Weight ≈ 50%):** The cross-sectionally normalized value of a factor at a point in time (e.g., ROAE Z-Score). Provides a baseline measure of quality or value.
*   **Change Signal (Weight ≈ 30%):** The first derivative of the factor (e.g., YoY change in ROAE). This is a **primary alpha source**, capturing fundamental improvement or deterioration.
*   **Acceleration Signal (Weight ≈ 20%):** The second derivative of the factor (e.g., the change in the QoQ change). Captures critical **inflection points** and helps manage risk by identifying when trends are strengthening or weakening.

*Note: The `v2_enhanced` engine currently being tested uses these fixed weights. A future research path involves making these weights dynamic based on market regimes, as detailed in Section 7.*

### **5.2. Separation of Fundamental vs. Price Momentum**

To ensure true factor diversification and orthogonality, we maintain a strict separation between momentum derived from fundamental data and momentum derived from price action.
*   **Fundamental Momentum:** Calculated from changes in ROAE, operating margins, working capital efficiency, and revenue growth. This captures the improving or deteriorating health of the underlying business.
*   **Price Momentum:** Calculated from risk-adjusted price returns over multiple lookback windows (e.g., 3, 6, 12 months), skipping the most recent month. This captures market sentiment and flow dynamics.

These two factors are constructed and tested independently before being combined in a portfolio context.

## **6. Factor Construction & Normalization**

### **6.1. Refined Factor Construction Workflow**
Factor construction follows a systematic approach where raw building blocks are stored, and all analytical transformations are performed dynamically in memory by the factor engines.

**Three-Pronged Composite Quality Factor (Calculated Dynamically):**
*   **Non-Financial Quality:** A weighted blend of ROAE, ROAA, Operating Margin, and EBITDA Margin.
*   **Banking Quality:** A weighted blend of ROAE, NIM, and Cost-to-Income Ratio.
*   **Securities Quality:** A weighted blend of ROAE and Operating Margin.

### **6.2. Working Capital Efficiency Signals**

The Cash Conversion Cycle (CCC) is a cornerstone of our quality and efficiency factors. A shorter, improving CCC is a strong positive signal.
*   **CCC Formula:**
    $$ \text{CCC} = \text{DSO} + \text{DIO} - \text{DPO}_{\text{Enhanced}} $$
*   **Enhanced DPO Calculation:** To eliminate inventory-driven distortions in cost of goods sold (COGS), we reconstruct purchases for a more precise Days Payable Outstanding calculation. This is a key proprietary enhancement.
    $$ \text{Purchases}_{\text{TTM}} = \text{COGS}_{\text{TTM}} + (\text{Inventory}_t - \text{Inventory}_{t-4}) $$
    $$ \text{DPO}_{\text{Enhanced}} = \frac{\text{Average Accounts Payable} \times 365}{\text{Purchases}_{\text{TTM}}} $$

### **6.3. Cross-Sectional Normalization**

Raw factor values are not comparable across different industries. Therefore, all factors are normalized cross-sectionally to create a standardized, relative ranking.
*   **Primary Method (Z-Score):** We convert each factor value into a z-score relative to its sector peers.
    $$ \text{Factor Z-Score} = \frac{\text{Factor Value} - \mu_{\text{sector}}}{\sigma_{\text{sector}}} $$
*   **Vietnam-Specific Normalization:** Recognizing the distinct economic behavior of state-owned enterprises, we apply a dual-normalization process.
    *   SOEs are normalized against a universe of only other SOEs.
    *   Private companies are normalized against a universe of only other private companies.
    This prevents distortions caused by the different capital structures and operating mandates of SOEs.

## **7. Dynamic Weighting & Portfolio Construction**

### **7.1. Portfolio Construction**
Our primary portfolio construction method is a long-only approach based on factor rankings.
1.  **Universe Definition:** Filter for tradable stocks based on liquidity and market capitalization.
2.  **Signal Generation:** Calculate the final composite alpha score for each stock in the universe using the winning factor engine.
3.  **Ranking:** Rank all stocks from best (highest score) to worst (lowest score).
4.  **Portfolio Formation:** Go long the top quintile (top 20%) of stocks.
5.  **Weighting:** Positions are typically equal-weighted, subject to single-stock and sector concentration limits.

### **7.2. Future Research: Dynamic Signal Weighting**
Static weights are robust but may not be optimal in all market regimes. A key future research initiative is to make the weights for Level, Change, and Acceleration signals dynamic based on their recent performance. A potential implementation could use a softmax normalization of rolling Sharpe ratios to adapt to changing market conditions, with appropriate constraints to prevent extreme allocations.

## **8. Risk Management & Validation**

### **8.1. Factor Orthogonality**
To ensure we are capturing distinct sources of alpha, we enforce a strict orthogonality constraint between our primary factors.
*   **Requirement:** The absolute pairwise correlation between any two final composite factors (e.g., Quality vs. Value) must be less than **0.4**.
*   **Rationale:** This threshold, informed by Harvey, Liu, and Zhu (2016), helps mitigate multicollinearity and reduces the risk of building a portfolio that is unintentionally over-exposed to a single, hidden macro factor.

### **8.2. Vietnam-Specific Hypothesis Validation**
*   **Hypothesis:** High retail participation (~70%) in Vietnam leads to slower incorporation of fundamental information, resulting in longer-lasting alpha signals compared to developed markets.
*   **Validation Protocol:**
    1.  **Signal Decay Analysis:** Calculate the half-life of our core factor signals using Information Coefficient (IC) decay over 1, 3, 6, and 12-month horizons.
    2.  **Benchmark Comparison:** Compare this decay rate to established benchmarks from developed market studies.

### **8.3. Transaction Cost Modeling**
All backtests must incorporate a realistic transaction cost model tailored for Vietnam.
$$ \text{Cost}_{\text{round-trip}} = \text{Commission} + \text{Market Impact} + \text{Slippage} + \text{Taxes} $$
*   **Model Inputs:** Brokerage commissions, exchange fees, bid-ask spreads, and a market impact function based on a trade's percentage of average daily volume (ADV).

## **9. Bibliography**

1.  Asness, C. S., Frazzini, A., & Pedersen, L. H. (2019). Quality minus junk. *Review of Accounting Studies*, 24(1), 34-112.
2.  Daniel, K., & Moskowitz, T. J. (2016). Momentum crashes. *Journal of Financial Economics*, 122(2), 221-247.
3.  Frazzini, A., Israel, R., & Moskowitz, T. J. (2018). Trading costs. *Journal of Finance*, 73(4), 1445-1508.
4.  Harvey, C. R., Liu, Y., & Zhu, H. (2016). ... and the cross-section of expected returns. *Review of Financial Studies*, 29(1), 5-68.
5.  Novy-Marx, R. (2013). The other side of value: The gross profitability premium. *Journal of Financial Economics*, 108(1), 1-28.

---
---

This definitive version is now complete. Please confirm, and I will provide the next fully refined document: **`02_system_architecture.md`**.