We will stick to the principle of **radical transparency**, but apply it to the full 8-ticker universe. This gives us the best of both worlds: a manageable test size and the ability to validate sector-specific logic correctly.

Here is the revised, comprehensive plan.

### **EXECUTIVE SUMMARY**

This plan outlines a simple, transparent, step-by-step validation of the refactored `qvm_engine_v2_enhanced.py` using our established 8-ticker universe (`OCB`, `VCB`, `NLG`, `VIC`, `FPT`, `CTR`, `SSI`, `VND`). We will create a new, clean validation script that forces the engine to "show its work" for the entire test set. The script will execute and display the results of each logical stage—data loading, raw ratio calculation, individual normalization across sectors, and final score combination—in plain, easy-to-verify tables. This approach provides complete transparency, allowing us to manually confirm every calculation against our specifications and rebuild absolute trust in the engine's correctness before scaling to the full universe.

### **DETAILED ANALYSIS: THE 8-TICKER "SHOW YOUR WORK" VALIDATION PLAN**

#### **Objective:**
To build complete confidence in the refactored engine by observing its calculations step-by-step across our 8-ticker, 4-sector universe and manually verifying them against our known formulas.

#### **Methodology:**
We will create a new script: `production/tests/validate_corrected_engine.py`. This script will be clean, focused, and will execute the following sequential validation steps, printing the full 8-ticker dataframe at each stage.

---

### **The Step-by-Step Validation Script Logic**

#### **Step 1: Data Loading & Preparation**

*   **Action:**
    1.  Initialize the corrected `QVMEngineV2Enhanced`.
    2.  Define the 8-ticker universe and `analysis_date = '2025-06-30'`.
    3.  Call the engine's data loading methods (`get_fundamentals_correct_timing`, `get_market_data`) to create the master `engine_combined` dataframe.
*   **Output:** Print the head and tail of the `engine_combined` dataframe, showing the raw TTM building blocks for all 8 tickers.

```
== [STEP 1] DATA LOADING & PREPARATION (8 Tickers) ==
(Prints the first 5 rows of the combined dataframe with columns like ticker, sector, NetProfit_TTM, AvgTotalEquity, OperatingExpenses_TTM, etc.)
...
✅ Data loaded successfully for 8 tickers across 4 sectors.
```
*   **Verification:** Confirm all 8 tickers are present and have the expected raw data.

#### **Step 2: Raw Quality Ratio Calculation**

*   **Action:**
    1.  Create a copy of the dataframe from Step 1.
    2.  Pass this copy to the `_calculate_enhanced_quality_composite` method.
    3.  Modify the engine method temporarily to return the dataframe *after* it has calculated the raw ratios (before normalization).
*   **Output:** Print the dataframe showing the newly added `_Raw` columns.

```
== [STEP 2] RAW QUALITY RATIO CALCULATION (8 Tickers) ==
  ticker         sector  ROAE_Raw  ROAA_Raw  Cost_Income_Raw  GrossMargin_Raw ...
0    OCB        Banking  0.095107  0.011185         0.608438              NaN
1    VCB        Banking  0.178973  0.017320         0.655413              NaN
2    NLG    Real Estate  0.112766  0.052783              NaN         0.409795
3    VIC    Real Estate  0.038723  0.007958              NaN         0.193625
4    FPT     Technology  0.283982  0.144548              NaN         0.379403
...
✅ CRITICAL CHECK: OCB Cost_Income_Raw is 0.6084 (60.84%), NOT > 1.0. Fix is working.
```
*   **Verification:** Manually calculate the `Cost_Income_Raw` for OCB and VCB to confirm the `abs()` fix. Spot-check a non-financial metric like FPT's `GrossMargin_Raw`.

#### **Step 3: Individual Metric Normalization**

*   **Action:**
    1.  Continue execution within the engine method.
    2.  Modify the engine to return the dataframe *after* it has calculated the individual z-scores for each raw metric.
*   **Output:** Print the dataframe showing the newly added `_ZScore` columns.

```
== [STEP 3] INDIVIDUAL METRIC NORMALIZATION (8 Tickers) ==
  ticker         sector  ROAE_Raw  ROAE_ZScore  Cost_Income_Raw  Cost_Income_ZScore ...
0    OCB        Banking  0.095107    -0.707107         0.608438           -0.707107
1    VCB        Banking  0.178973     0.707107         0.655413            0.707107
2    NLG    Real Estate  0.112766     0.707107              NaN                 NaN
3    VIC    Real Estate  0.038723    -0.707107              NaN                 NaN
...
✅ Logic Confirmed: Z-scores are calculated WITHIN each sector.
   - Banking: OCB and VCB are normalized against each other.
   - Real Estate: NLG and VIC are normalized against each other.
```
*   **Verification:** Manually verify the z-score for one banking metric (e.g., OCB's ROAE) and one non-financial metric (e.g., FPT's ROAE). The z-scores for pairs within a sector should be equal and opposite (e.g., +/- 0.7071).

#### **Step 4: Final Quality Score Combination**

*   **Action:**
    1.  Allow the engine method to complete its execution.
    2.  The method should return the final dictionary of `quality_scores`.
*   **Output:** Print a final, comprehensive table showing the full calculation for each ticker.

```
== [STEP 4] FINAL QUALITY SCORE COMBINATION (8 Tickers) ==
Ticker | Sector      | Metric             | Z-Score  | Weight | Weighted Score | Final Score
-------|-------------|--------------------|----------|--------|----------------|------------
OCB    | Banking     | ROAE_ZScore        | -0.7071  | 0.40   | -0.2828        |
       |             | ROAA_ZScore        | -0.7071  | 0.25   | -0.1768        |
       |             | Cost_Income_ZScore | -0.7071  | 0.15   | -0.1061        |
       |             |                    |          |        |                | **-0.6431**
-------|-------------|--------------------|----------|--------|----------------|------------
FPT    | Technology  | ROAE_ZScore        |  0.7071  | 0.35   |  0.2475        |
       |             | GrossMargin_ZScore |  0.7071  | 0.25   |  0.1768        |
       |             | ...                |  ...     | ...    |  ...           | **+0.5500** (example)
...
```
*   **Verification:** Manually sum the "Weighted Score" column for OCB and FPT to confirm the `Final Score` is correct.

---

### **IMPLEMENTATION NOTES**

1.  **Create the New Script:** Archive the old complex notebook. Create `production/tests/validate_corrected_engine.py`.
2.  **Engine Modification for Transparency:** The `_calculate_enhanced_quality_composite` method will need to be temporarily modified to `return` the intermediate dataframes at each stage, rather than just the final scores. This is crucial for the "show your work" approach.
3.  **Focus on Quality First:** This plan validates the most complex and previously flawed part of the engine. Once we have full confidence in the Quality factor, we can apply the same transparent validation process to the Value and Momentum factors.

### **RISK CONSIDERATIONS**

*   **Reduced Risk:** This plan systematically reduces risk by making every step of the calculation transparent. There are no "black box" steps.
*   **Builds Trust:** Successfully completing this validation will build the necessary trust to proceed with the large-scale historical data generation.
*   **Efficiency:** This approach is far more efficient than the previous complex audit. We will get a definitive pass/fail result on the engine's core logic quickly.

This plan is robust, transparent, and directly addresses your need to see and trust the engine's calculations before committing to a large-scale data recovery. It is the correct path forward.