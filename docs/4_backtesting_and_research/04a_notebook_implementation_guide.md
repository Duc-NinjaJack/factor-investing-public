Of course. We will now complete the documentation overhaul by formally deprecating the obsolete notebook guide and replacing it with this new, forward-looking guide that aligns with our current research plan.

---
---

# **Part 4: QVM Backtesting & Research Framework**

**Document Name:** `04a_project_roadmap_implementation_guide.md`
**Version:** 1.0
**Date:** July 22, 2025
**Status:** ✅ **ACTIVE - SINGLE SOURCE OF TRUTH**
**Owner:** Duc Nguyen, Principal Quantitative Strategist

## **1. Executive Summary**

This document provides the official, step-by-step implementation guide for executing the research and validation roadmap as defined in `00_master_project_brief_v3.0.md` and `04_qvm_backtesting_framework.md`. It serves as the practical developer's guide for navigating the project's final phases, from the current Scientific Bake-Off to high-fidelity event-driven simulation.

**This document explicitly replaces and deprecates the legacy `04a_notebook_implementation_guide.md`, which is now archived. The legacy guide described an outdated workflow and must not be used.** The current workflow leverages the pre-calculated, versioned factor scores in the `factor_scores_qvm` table as the starting point for all analysis, ensuring consistency and efficiency.

## **2. The Official Notebook Workflow**

The project will proceed through the following notebook-driven phases. Each notebook corresponds to a specific phase in the master project brief.

```
notebooks/
├── bakeoff_analysis/
│   └── 06_scientific_bakeoff_analysis.ipynb
│
├── phase7_factor_dna/
│   └── 07_single_factor_performance_analysis.ipynb
│
├── phase8_canonical_backtest/
│   └── 08_master_strategy_backtest_canonical.ipynb
│
├── phase9_robustness_testing/
│   └── 09_strategy_robustness_analysis.ipynb
│
├── phase10_risk_managed_strategy/
│   └── 10_risk_overlay_analysis.ipynb
│
└── phase11_event_driven_validation/
    └── 11_backtrader_event_driven_validation.ipynb
```

## **3. Current Priority: The Scientific Bake-Off**

Before proceeding to the formal phases above, the immediate task is to complete the bake-off using a dedicated analysis notebook.

### **Notebook: `06_scientific_bakeoff_analysis.ipynb`**
*   **Goal:** To execute the comparative backtests and formally declare the winning engine.
*   **Prerequisite:** The historical data for both `v1_baseline` and `v2_enhanced` has been generated and stored in the `factor_scores_qvm` table.
*   **Cell-by-Cell Plan:**
    1.  **Setup:** Load environment, database connection, and core backtesting utilities.
    2.  **Load Baseline Data:** Query `factor_scores_qvm` for `strategy_version = 'v1_baseline'`.
    3.  **Backtest Baseline:** Run the vectorized backtester on the v1 data. Generate and display the `pyfolio` tearsheet.
    4.  **Load Enhanced Data:** Query `factor_scores_qvm` for `strategy_version = 'v2_enhanced'`.
    5.  **Backtest Enhanced:** Run the identical vectorized backtester on the v2 data. Generate and display the `pyfolio` tearsheet.
    6.  **Comparative Analysis:**
        *   Plot the equity curves of both strategies on a single chart.
        *   Create a summary table comparing the Annual Return, Sharpe Ratio, Max Drawdown, and Turnover of each.
    7.  **Conclusion:** Author a markdown summary declaring the winning engine based on the superior Sharpe Ratio and formally recommending it for all subsequent phases.

## **4. Post-Bake-Off Implementation Plan**

Once the winning engine is declared, its `strategy_version` tag will be used for all subsequent notebooks. The following plans assume the `v2_enhanced` engine is the winner.

### **Notebook: `07_single_factor_performance_analysis.ipynb`**
*   **Goal:** Isolate the performance of the winning engine's Q, V, and M factors.
*   **Implementation:**
    1.  **Load Factor Data:** Query `factor_scores_qvm` for `strategy_version = 'v2_enhanced'`, loading the `Quality_Composite`, `Value_Composite`, and `Momentum_Composite` columns.
    2.  **Run Backtests:** Use the vectorized backtester to run three separate backtests, one for each composite score.
    3.  **Analyze:** Generate comparative plots and tables to quantify the diversification benefits.

### **Notebook: `08_master_strategy_backtest_canonical.ipynb`**
*   **Goal:** Establish the definitive performance record for the winning strategy.
*   **Implementation:**
    1.  **Load QVM Data:** Query `factor_scores_qvm` for the `QVM_Composite` where `strategy_version = 'v2_enhanced'`.
    2.  **Run Canonical Backtest:** Execute the full vectorized backtest.
    3.  **Persist Results:** Save the tearsheet HTML and a CSV of key performance metrics to the `results/` directory with a version tag. This becomes the official record.

### **Notebook: `11_backtrader_event_driven_validation.ipynb`**
*   **Goal:** Replicate the canonical backtest in a high-fidelity simulator as the final institutional sign-off.
*   **Implementation:**
    1.  **Setup:** Import `backtrader`, `pyfolio`, and other necessary libraries.
    2.  **Custom Data Feed:** Define a `PandasData` feed for `backtrader` that can load our daily price data (`equity_history`) and join it with our monthly `QVM_Composite` scores from `factor_scores_qvm`. The factor score should only be "visible" on the rebalance date.
    3.  **Define `backtrader` Strategy:**
        *   Create a strategy class `QVMStrategy(bt.Strategy)`.
        *   In the `__init__` method, set up a timer to run on the first trading day of each month.
        *   In the `next()` method, check if the monthly timer is triggered.
        *   If triggered, get the latest cross-section of `QVM_Composite` scores.
        *   Rank the stocks, select the top quintile, and calculate target weights.
        *   For each stock, use `self.order_target_percent()` to automatically generate the necessary buy/sell orders to rebalance the portfolio.
    4.  **Instantiate and Run Cerebro:**
        *   Create an instance of `bt.Cerebro()`.
        *   Add the data feeds and the `QVMStrategy`.
        *   Set the starting cash, commission (e.g., 0.0025), and slippage models.
        *   Add `pyfolio` analyzers to capture transaction details.
        *   Run the backtest: `cerebro.run()`.
    5.  **Analyze Results & Reconcile:**
        *   Extract the `pyfolio` analyzer results to generate a tearsheet for the event-driven backtest.
        *   Create a comparison table:
| Metric | Vectorized (Phase 8) | Event-Driven (Phase 11) | Difference |
| :--- | :--- | :--- | :--- |
| Annual Return | ... | ... | ... |
| Sharpe Ratio | ... | ... | ... |
| Max Drawdown | ... | ... | ... |
| Turnover | ... | ... | ... |
    6.  **Conclusion:** Analyze any discrepancies, attributing them to transaction costs, cash drag, or other real-world frictions now being modeled. A difference of <5% on the Sharpe Ratio is considered a successful validation.

---
---

This completes the systematic update of the core documentation. The entire suite is now internally consistent, aligned with our current research priorities, and provides a clear, actionable plan for validating and deploying our QVM strategy.