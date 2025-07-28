Of course. Here is the next fully refined document. This version of the `QVM Backtesting Framework` has been completely rewritten to serve as the official master plan for the scientific bake-off and the subsequent validation phases. It removes all outdated and contradictory information.

---
---

# **Part 4: QVM Backtesting & Research Framework**

**Document Name:** `04_qvm_backtesting_framework.md`
**Version:** 2.1 (Bake-Off Charter)
**Date:** July 22, 2025
**Status:** ✅ **ACTIVE - SINGLE SOURCE OF TRUTH**
**Owner:** Duc Nguyen, Principal Quantitative Strategist

## **1. Executive Summary**

This document provides the official framework and implementation roadmap for the validation of the Aureus Sigma QVM strategy. The immediate priority is to execute a **Scientific Bake-Off** between our two competing factor engines (`v1_baseline` and `v2_enhanced`) to empirically determine the superior signal construction methodology. Following the bake-off, this framework will guide the rigorous, multi-stage validation of the winning engine, culminating in a high-fidelity, event-driven simulation that serves as the final institutional sign-off before production deployment. This document corresponds to the research plan outlined in the `00_master_project_brief_v3.0.md`.

## **2. Core Backtesting Architecture: Vectorized vs. Event-Driven**

Our validation process employs two distinct but complementary backtesting methodologies.

### **2.1. Vectorized Backtester**
*   **Purpose:** Rapid, large-scale analysis, parameter sensitivity testing, and strategy prototyping. It operates on entire arrays of data at once, making it exceptionally fast.
*   **Role in Current Project:** The vectorized backtester is the **primary tool for the Scientific Bake-Off** due to its speed, which allows for quick comparison of the two engine outputs over the full historical period.
*   **Mechanism:**
    1.  Loads the complete `factor_scores_qvm` dataset for a given `strategy_version`.
    2.  At each monthly rebalance date, it ranks stocks based on the factor score.
    3.  It calculates portfolio returns based on the subsequent price movements of the selected stocks.

### **2.2. Event-Driven Backtester (`backtrader`)**
*   **Purpose:** High-fidelity simulation that mimics real-world trading. This is the gold standard for final validation of the winning strategy.
*   **Role in Current Project:** This will be used in **Phase 11** to confirm that the performance of the winning engine is achievable after accounting for real-world frictions like cash drag, slippage, and order management.
*   **Mechanism:**
    1.  Iterates through time one day at a time.
    2.  On a rebalance date, the strategy logic generates target portfolio weights.
    3.  The engine generates individual orders (buy/sell) and simulates their execution, including commissions and slippage.

## **3. The Scientific Bake-Off Framework (The Experiment)**

This is the highest priority research task. Its purpose is to select our production engine based on data, not opinion.

### **Step 1: Generate Datasets**
*   **Objective:** To create two complete, time-series datasets of factor scores, one for each engine.
*   **Action:**
    1.  Execute the production runner script using `QVMEngineV1Baseline` to populate the `factor_scores_qvm` table for the full historical period, tagging each record with `strategy_version = 'v1_baseline'`.
    2.  Execute the production runner script using `QVMEngineV2Enhanced` to populate the `factor_scores_qvm` table for the full historical period, tagging each record with `strategy_version = 'v2_enhanced'`.

### **Step 2: Run Comparative Backtests**
*   **Objective:** To run an identical, apples-to-apples backtest on both datasets.
*   **Action:**
    1.  Using the **vectorized backtester**, run a backtest on the `v1_baseline` data. The strategy will be a long-only, top-quintile, equal-weighted portfolio, rebalanced monthly.
    2.  Run the **exact same backtest** on the `v2_enhanced` data.

### **Step 3: Analyze Results & Select Winner**
*   **Objective:** To declare a winner based on superior risk-adjusted performance.
*   **Decision Metric:** The primary metric is the **Sharpe Ratio**. The engine that produces a strategy with a statistically significant higher Sharpe Ratio will be declared the winner.
*   **Secondary Metrics:** Annual Return, Maximum Drawdown, and Turnover will be analyzed to understand the character of the returns and any trade-offs.
*   **Outcome:** The winning engine is promoted and becomes the firm's single production engine for the QVM strategy.

## **4. Post-Bake-Off Validation Roadmap (The Plan for the Winner)**

Once a winning engine is selected, it will proceed through the following validation phases, as detailed in the `00_master_project_brief_v3.0.md`.

### **Phase 7: Single Factor Performance Analysis**
*   **Objective:** To deconstruct the performance of the **winning engine's** combined QVM model to understand the contribution of its individual Q, V, and M components.
*   **Methodology:** Run three separate vectorized backtests on the winning engine's `Quality_Composite`, `Value_Composite`, and `Momentum_Composite` signals.

### **Phase 8: Canonical "Aggressive Growth" Backtest**
*   **Objective:** To establish the single, immutable source of truth for the **winning engine's** performance.
*   **Methodology:** The full backtest of the winning engine's `QVM_Composite` score from the bake-off will be formally version-controlled and locked as the official performance record.

### **Phase 9: Strategy Robustness Analysis**
*   **Objective:** To prove that the **winning strategy's** performance is a persistent structural edge.
*   **Methodology:** Systematically re-run the canonical backtest, altering key parameters one at a time (e.g., transaction costs, rebalancing frequency, data lags).

### **Phase 11: Event-Driven Backtesting with `backtrader`**
*   **Objective:** To replicate the **winning strategy's** performance in a high-fidelity, event-driven simulator as the final institutional sign-off.
*   **Success Criterion:** The `backtrader` results (Annual Return, Sharpe, MDD) must reconcile with the canonical vectorized backtest results within a tight tolerance (e.g., +/- 5%).

---
---

This document now provides the clear, forward-looking plan for our backtesting and research efforts. Please confirm, and I will provide the final document in this series: the deprecation and replacement plan for **`04a_notebook_implementation_guide.md`**.