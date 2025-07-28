Of course. Here is the next fully refined document. This version of the `Operational Playbook` is updated to reflect the current state of the project: the bake-off is the central research activity, and the production environment has been formalized.

---
---

# **Part 3: Operational Framework**

**Document Name:** `03_operational_playbook.md`
**Version:** 2.1 (Bake-Off Charter)
**Date:** July 22, 2025
**Status:** ✅ **ACTIVE - OFFICIAL STANDARD OPERATING PROCEDURE (SOP)**
**Owner:** Duc Nguyen, Principal Quantitative Strategist

## **1. Guiding Principle: Discipline is Alpha**

The most sophisticated models will fail without a disciplined operational framework. This playbook ensures that every action, from daily data checks to quarterly strategic reviews, is executed systematically, repeatably, and with full auditability. Adherence to this process is mandatory for all quantitative and portfolio management staff. It is our primary defense against operational risk, model decay, and behavioral biases. All production scripts referenced herein are located in the `production/scripts/` directory.

## **2. The Research & Trading Cadence**

### **2.1. Daily Tasks (The Pulse: 07:00 - 09:00 ICT)**
**Objective:** Ensure 100% system integrity and full situational awareness before the market opens (09:00 ICT).
**Responsibility:** Quant Developer on rotation, reviewed by Head of Trading.

*   **[ ] 1. System & Data Health Check (07:00 - 07:30)**
    *   **Action:** Execute `production/scripts/daily_system_check.py`.
    *   **Verification Checklist:**
        *   [ ] **Data Pipelines:** All overnight data pipelines (VCSC, Fundamentals) completed with `SUCCESS` status.
        *   [ ] **Database Replication & Backups:** Replication lag is `< 5 seconds`. Daily automated backup of `factor_scores_qvm` completed successfully.
        *   [ ] **Error Logs:** No `CRITICAL` or `ERROR` level messages in the last 24 hours.
        *   [ ] **Data Freshness:** `vcsc_daily_data_complete` and `equity_history` tables contain data for T-1.
    *   **Escalation:** Any failed check is immediately escalated to the Head of Technology. Trading is not permitted until all checks pass.

*   **[ ] 2. Portfolio Risk & P&L Review (07:30 - 08:30)**
    *   **Action:** Generate and review the "Daily Portfolio Risk Monitor" report.
    *   **Verification Checklist:**
        *   [ ] **P&L Attribution:** Is >85% of T-1 P&L explained by our target factor exposures? Any unexplained residual > 15% requires investigation.
        *   [ ] **Factor Exposure Drift:** Are all primary factor exposures (Quality, Value, Momentum) within ±10% of their target weights?
        *   [ ] **Risk Limits:**
            *   [ ] Gross Exposure: Within `180% - 200%` range.
            *   [ ] Net Exposure: Within `-5% to +5%` range.
            *   [ ] Single Stock Concentration: No position > `5%` of the portfolio.
            *   [ ] Sector Concentration: No net sector exposure > `10%`.
    *   **Escalation:** Any limit breach requires an immediate plan to bring the portfolio back into compliance, approved by the Head of Risk.

*   **[ ] 3. Market Environment Scan (08:30 - 09:00)**
    *   **Action:** Review market dashboard and pre-open news flow.
    *   **Checklist:**
        *   [ ] **Top 10 Holdings:** Scan for any material news (earnings pre-announcements, M&A, regulatory probes).
        *   [ ] **Market Volatility:** Note the VN-VIX level. If > 30, review strategy for potential momentum crash risk.
        *   [ ] **Liquidity Events:** Note any stocks on the HOSE/HNX warning list or with unusual trading halts.

### **2.2. Weekly Tasks (The Review: Friday 15:00 - 17:00 ICT)**
**Objective:** To analyze factor performance, detect signal decay, and maintain data quality.
**Responsibility:** Quantitative Research Team, reviewed by Principal Strategist.

*   **[ ] 1. Factor Performance Attribution**
    *   **Action:** Generate the "Weekly Factor Performance Report" for both `v1_baseline` and `v2_enhanced` strategies (once historical data is generated).
    *   **Analysis:**
        *   Decompose portfolio returns into contributions from Quality, Value, and Momentum.
        *   For the `v2_enhanced` strategy, analyze the performance of the three-tier signals (Level vs. Change vs. Acceleration).
        *   Compare the performance of the v1 and v2 engines. Are the results aligning with the bake-off hypothesis?

*   **[ ] 2. Signal Quality Control**
    *   **Action:** Execute `production/scripts/weekly_signal_outlier_check.py`.
    *   **Process:**
        1.  The script generates a list of all stocks with a final `QVM_Composite` z-score > +3.0 or < -3.0 from the `v2_enhanced` strategy.
        2.  **Mandatory Investigation:** The top 3 and bottom 3 stocks on this list must be manually investigated.
        3.  **Validation:** For each investigated stock, confirm the extreme score is driven by a genuine fundamental catalyst and not a data error.

### **2.3. Monthly Tasks (The Rebalance: First Trading Day of Month)**
**Objective:** To systematically rebalance the portfolio based on the latest available signals.
**Responsibility:** Head of Trading, executed by Quant Developer.

*   **[ ] 1. Signal Generation (T-2 Day)**
    *   **Action:** Execute `production/scripts/run_factor_generation.py` for the month-end `analysis_date`. 
        ```bash
        python run_factor_generation.py --start-date YYYY-MM-DD --end-date YYYY-MM-DD --mode incremental --version v2_enhanced
        ```
        *   **--mode**: Controls the data insertion behavior:
            *   `incremental` (default): Only adds missing dates, preserving existing data
            *   `refresh`: Clears existing data for the date range before inserting new data
        *   **--version**: Specifies the strategy version tag (default: `v2_enhanced`)
            *   Enables parallel testing of multiple strategy versions
            *   All versions are stored with complete isolation in the database
    *   **Verification:** Confirm that new scores are successfully written to the `factor_scores_qvm` table for the analysis date with the correct `strategy_version` tag.

*   **[ ] 2. Pre-Rebalance Checklist (T-1 Day)**
    *   **Action:** Execute `production/scripts/generate_target_portfolio.py` in "dry-run" mode, targeting the `v2_enhanced` strategy signals.
    *   **Verification Checklist:**
        *   [ ] **Signal Freshness:** The latest signals from the `factor_scores_qvm` table are being used.
        *   [ ] **Target Portfolio Generated:** A target portfolio file is created.
        *   [ ] **Trade List & Cost Estimation:** A preliminary trade list is generated. The estimated total transaction cost must be below the monthly budget.
        *   [ ] **Compliance Pre-Check:** The proposed target portfolio is checked against all risk limits. It must pass 100% of checks before proceeding.

*   **[ ] 3. Execution & Post-Trade Analysis (T Day)**
    *   **Action:** Execute trades based on the final, approved trade list.
    *   **Post-Trade Reconciliation:** Confirm the live portfolio matches the target portfolio weights to within a 0.1% tolerance.

### **2.4. Quarterly Tasks (The Strategic Review)**
**Objective:** To step back from day-to-day operations, validate our models against new data, and set the research agenda.
**Responsibility:** Investment Committee (led by Principal Strategist).

*   **[ ] 1. Out-of-Sample Validation**
    *   **Action:** Treat the most recent quarter as a new, unseen out-of-sample data point for the v1 vs. v2 scientific bake-off.
    *   **Process:** Compare the live performance of the v2 engine against the v1 engine and the backtested expectations. Update the bake-off scorecard.

*   **[ ] 2. Strategic Research & Backtesting Review**
    *   **Action:** Convene the quarterly Investment Committee meeting.
    *   **Agenda:**
        *   Review progress on the official research agenda (see below).
        *   Analyze the factor correlation matrix. Have any of our core factors become more correlated (> 0.4)?
        *   Review and approve the research priorities for the upcoming quarter.

## **3. The Aureus Sigma Research Agenda**

A quantitative firm that is not researching is dying. The following are the official, prioritized research initiatives, aligned with the `00_master_project_brief_v3.0.md`. All research must follow the `03a_factor_validation_playbook.md`.

#### **Priority 1: Complete the Scientific Bake-Off**
*   **Project:** Finalize validation of `qvm_engine_v2_enhanced.py`, generate historical datasets for both v1 and v2 engines, and execute comparative backtests.
*   **Objective:** Conclusively determine the superior engine based on empirical evidence (Sharpe Ratio) and promote it to be the single production engine.
*   **Status:** **IN PROGRESS - HIGHEST FIRM PRIORITY.**

#### **Priority 2: Canonical Validation & High-Fidelity Simulation (Phases 7, 8, 11)**
*   **Project:** Single Factor Performance Analysis, Canonical Backtesting, and Event-Driven Backtesting.
*   **Objective:** Once the winning engine is chosen, establish its immutable performance record and replicate it in an institutional-grade `backtrader` simulation as the final pre-production sign-off.
*   **Status:** Pending completion of Priority 1.

#### **Priority 3: New Product & Robustness Testing (Phases 9 & 10)**
*   **Project:** Risk-Managed Strategy Development & Strategy Robustness Analysis.
*   **Objective:** Develop the "Risk-Managed" strategy variant and stress-test the canonical strategy against different market assumptions.
*   **Status:** Pending completion of Priority 2.

#### **Priority 4: New Alpha Sources (Long-Term)**
*   **Project:** Crypto & On-Chain Factor Development.
*   **Objective:** Integrate on-chain data (e.g., active addresses, transaction volume, TVL) as new factors for digital asset trading strategies, applying the same rigorous validation playbook.
*   **Status:** Future research.

---
---

This operational playbook is now synchronized with our current production environment and strategic priorities. Please confirm, and I will provide the next complete file: **`03a_factor_validation_playbook.md`**.