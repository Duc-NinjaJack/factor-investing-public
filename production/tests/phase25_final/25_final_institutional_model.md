# ============================================================================
# Aureus Sigma Capital - Phase 25: Final Institutional Model Run
# Notebook: 25_final_institutional_model.ipynb
#
# Objective:
#   To execute the final 10-day sprint to produce an Investment Committee (IC)
#   -ready strategy. This notebook will integrate all previously built and
#   validated components—the unified backtest engine, the full factor stack,
#   the non-linear cost model, and the risk overlay suite—into a single,
#   production-grade backtest.
# ============================================================================
#
# --- STRATEGIC DIRECTIVE & ALIGNMENT ---
#
# This notebook directly implements the "Implementation Notes (Next 10 calendar days)"
# from the final strategic assessment. The project is now "engineering-complete
# but policy-incomplete." Our task is to activate and integrate all risk
# management and cost policies to transform our raw alpha signal into a robust,
# investable strategy.
#
# --- PRIMARY RESEARCH QUESTION ---
#
# Can the `Full_Composite_Monthly` strategy, when combined with the full suite
# of risk overlays (15% Vol Target, 25% Sector Cap, Tail Hedge) and a realistic
# cost model, achieve our institutional hurdles of a Sharpe Ratio >= 1.0 and a
# Maximum Drawdown <= -35%?
#
# --- METHODOLOGY: THE 10-DAY SPRINT PLAN ---
#
# This notebook will be structured to follow the implementation checklist precisely:
#
# 1.  **P0 Task (Day 1-2): Activate Existing Modules**
#     -   Create `PortfolioEngine_v5_0`, which fully integrates the non-linear
#       cost model and the volatility targeting overlay into the main event loop.
#     -   Implement the walk-forward logic for the factor weight optimizer.
#     -   Re-run the `Full_Composite_Monthly` strategy to establish a new,
#       risk-managed performance baseline.
#
# 2.  **P1 Tasks (Day 3-6): Automate & Test**
#     -   Launch the full 12-cell test matrix using the new, fully-featured engine
#       to find the optimal configuration of (stock count x rebalance freq x logic).
#     -   Implement CI checks for compliance with risk policies (e.g., sector caps).
#
# 3.  **P2 Tasks (Day 7-10): Final Validation**
#     -   Take the single winning configuration from the test matrix.
#     -   Conduct final validation tests: Monte Carlo Sharpe CI and IPO-cohort
#       attribution analysis.
#     -   Prepare the final IC memo and performance tearsheet.
#
# --- SUCCESS CRITERIA (from assessment) ---
#
# The final, winning strategy from this sprint must meet the following gates:
#
#   -   **Sharpe Ratio:** >= 1.0
#   -   **Maximum Drawdown:** <= -35%
#   -   **Annual Turnover:** <= 250%
#

