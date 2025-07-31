> Please read the following and acknowledge that you fully understand before we continue: docs/session_handoff/2025-07-30_1723_phase25b_institutional_grade_composite_final_results.md, production/tests/phase8_risk_management/images/10_liquidity_deepdive_63day_rolling_ADTV.png, production/tests/phase8_risk_management/images/10_liquidity_deepdive_concentration.png, production/tests/phase8_risk_management/images/10_liquidity_deepdive_sector_composition.png, production/tests/phase8_risk_management/images/10_liquidity_deepdive_size_liquid.png, productiontests/phase25_final/25b_institutional_grade_composite_final.md, production/testsphase25_final/Composite_Q_20_1.25x_tearsheet.png, production/tests/phase25_final/images/25b_composite_best_ts.png

**EXECUTIVE SUMMARY**
Your latest run **`Composite_Q_20_1.25×`** (fixed‑20 names, quarterly rebalance, 15 % vol‑target, 1.25× leverage) still **falls short of every Investment‑Committee gate**:

| KPI                 | Result      | IC Hurdle     | Pass/Fail |
| ------------------- | ----------- | ------------- | --------- |
| Annual Return (net) | 13.0 %      | ≥ 15 %\*      | **Fail**  |
| Annual Volatility   | 19.8 %      | 15 % (target) | **Fail**  |
| Sharpe Ratio (net)  | **0.65**    | ≥ 1.0         | **Fail**  |
| Max Drawdown        | **‑46.3 %** | ≥ ‑35 %       | **Fail**  |
| Beta vs VN‑Index    | 0.85        | ≤ 0.75        | Marginal  |
| Information Ratio   | 0.12        | ≥ 0.8         | **Fail**  |

\*15 % net CAGR is implied by the 1 : 1 Sharpe‑to‑vol target in the Master Brief .

The 1.25× leverage lifted return but *did not* improve risk‑adjusted performance; the Max DD breach grew by >3 ppts. The core issue remains **insufficient gross alpha density**—the engine still uses static V\:Q:M\:R ≈ 50 : 25 : 20 : 5 weights and a pure volatility overlay. Until the walk‑forward optimiser, hybrid regime filter and non‑linear cost model outlined in Phase 25b are activated , Sharpe is very unlikely to break 0.75.

---

### 1 │ Liquidity‑Regime Concern (2016‑19 vs 2020‑25)

Your intuition is correct: Vietnam’s market microstructure changed materially around 2020 (launch of VN30 derivatives, higher retail turnover, improved settlement). Pre‑2020 the free‑float ADTV for many mid‑caps sat below 5 bn VND; sophisticated factor portfolios faced:

* **High slippage & queue‑risk** – orders >3 % of ADTV produce non‑linear impact.
* **Sparse fundamental coverage** – delayed filings widen the 45‑day lag window, degrading Quality/Value timeliness.
* **Crowded exits** in stress events (e.g., 2018 taper tantrum) inflate draw‑downs.

That said, abandoning 2016‑19 outright is **not** recommended; investors and regulators will view such exclusion as cherry‑picking. Institutional best practice is:

1. **Adaptive liquidity filter** – require each candidate’s 20‑day ADTV ≥ `max(10 bn VND, 0.04 % of free‑float mkt‑cap)` at each rebalance. In low‑liquidity eras this naturally reduces weight/exposure without discarding the period altogether.

2. **Liquidity‑regime attribution** – report performance split by “low‑liquidity” vs “high‑liquidity” buckets; this shows transparency and guides product positioning (e.g., Aggressive Growth may launch with 2020‑onward track‑record, Risk‑Managed keeps full history).

3. **Shadow‑benchmark** – use VNFIN LEAD (liquid‑50) in addition to VN‑Index; this limits benchmarking distortions due to micro‑cap drift (Assessment v2, Regime sensitivity comment) .

Empirically, QVM alpha persists in early years, but the *capturable* fraction after costs is small. A liquidity‑aware back‑test often shows **Sharpe improves by 0.15‑0.20** once pre‑2020 positions are down‑weighted instead of removed.

---

### 2 │ Refactoring *25b* Notebook to Test Sub‑Periods

Below is a minimally invasive pattern that keeps the notebook reproducible and audit‑friendly.

```python
# --- 0. Global config (add at top) -----------------------------------------
BACKTEST_WINDOWS = {
    "FULL_2016_2025":  {"start": "2016-01-01", "end": "2025-12-31"},
    "LIQUID_2018_2025":{"start": "2018-01-01", "end": "2025-12-31"},
    "POST_DERIV_2020_2025": {"start": "2020-01-01", "end": "2025-12-31"}
}
ACTIVE_WINDOW = "LIQUID_2018_2025"          # ← toggle here
# ---------------------------------------------------------------------------

# in Phase 1 – Setup & Configuration block:
bt_cfg = {
    **BASE_CONFIG,
    "analysis_start": pd.Timestamp(BACKTEST_WINDOWS[ACTIVE_WINDOW]["start"]),
    "analysis_end":   pd.Timestamp(BACKTEST_WINDOWS[ACTIVE_WINDOW]["end"]),
    # other params unchanged
}

# Inside Canonical/Unified backtester init:
self.dates = pd.date_range(bt_cfg["analysis_start"],
                           bt_cfg["analysis_end"],
                           freq=self.rebalance_freq)
```

**Key edits**

1. **Central date dictionary** – prevents silent drift across cells.
2. **Filter all input frames** once:

```python
factor_data  = factor_data_raw.loc[bt_cfg["analysis_start"]:bt_cfg["analysis_end"]]
price_matrix = price_matrix_raw.loc[bt_cfg["analysis_start"]:bt_cfg["analysis_end"]]
benchmark    = benchmark_raw.loc[bt_cfg["analysis_start"]:bt_cfg["analysis_end"]]
```

3. **Unit test** – add

```python
assert results.index.min() >= bt_cfg["analysis_start"]
assert results.index.max() <= bt_cfg["analysis_end"]
```

to CI.

---

### 3 │ Recommended Experiment Sequence

| Step  | Window                           | Objective                                                                   | Expected Insight                                         |
| ----- | -------------------------------- | --------------------------------------------------------------------------- | -------------------------------------------------------- |
| **A** | 2018‑25                          | Remove pre‑IPO illiquidity spike, keep 2018 crunch                          | Sharpe ↑ ≈ 0.10; DD ↓ 3‑5 ppts                           |
| **B** | 2020‑25                          | Pure high‑liquidity era                                                     | Sharpe ↑ 0.20‑0.25; Turnover ↑ 30 %                      |
| **C** | *(A) vs (B)*                     | Liquidity attribution table: Contribution by year, cost drag by ADTV decile | Quantify alpha loss in 2016‑19; supports LP discussion   |
| **D** | 2016‑25 with adaptive ADV filter | Full track‑record, but liquidity‑weighted                                   | Sharpe \~ midpoint of A & B with transparent methodology |

---

### 4 │ Remaining Structural Gaps Before Next Run

| Gap                                                    | Fix                                                                                                         | Source Doc                      |
| ------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------- | ------------------------------- |
| Walk‑forward optimiser still inactive → static weights | Activate 24 m / 6 m cycle with Bayesian priors (Value ≥ 0.30, Momentum ≥ 0.25)                              |                                 |
| Hybrid (vol ⊕ regime) overlay missing                  | Implement trigger: max(vol‑exposure, regime‑exposure) where regime = 63‑day σ > 25 % *or* 20‑day DD < ‑10 % |                                 |
| ADTV cost model dormant                                | Switch cost hook: `3 bps + 0.15×sqrt(order/ADTV)`; cap participation ≤ 5 %                                  |                                 |
| Foreign‑ownership (FOL) filter absent                  | Exclude holdings with free‑room < 3 %; prevents forced sells                                                | Master Brief §4 Data Governance |

---

### 5 │ Actionable Next‑Week Checklist

| Day | Task                                                      | Owner     | Exit Test              |
| --- | --------------------------------------------------------- | --------- | ---------------------- |
| 1‑2 | Add `analysis_start` / `analysis_end` to config & CI test | Quant dev | Dates assertion passes |
| 2‑4 | Activate optimiser & cost model; rerun **B** (2020‑25)    | Quant dev | Sharpe ≥ 0.80 (net)    |
| 4‑5 | Implement hybrid overlay; rerun **B**                     | Quant dev | Max DD ≤ ‑35 %         |
| 6‑7 | Generate side‑by‑side tearsheets for windows A–D          | Analyst   | IC appendix ready      |

---

### 6 │ Strategic Perspective

*Institutional investors seldom object to excluding **pre‑liquidity** windows—provided the methodology is objective and disclosed.* By parameterising liquidity (ADV filter) rather than calendar dates, you align with AQR/Citadel best practice: strategy automatically adapts as markets deepen, and you present a **robust full‑history** view that penalises early years only to the extent they were actually tradable.

Once the optimiser and hybrid overlay are live, the 2020‑25 Sharpe should approach **0.9–1.1 pre‑leverage**. A controlled 1.3× gross exposure—back‑tested under the realistic cost model—can then push the IC‑level Sharpe above 1.0 while keeping Max DD within ‑35 %. That remains the fastest path to an investable, audit‑ready product.

---

Please let me know when the refactored notebook is ready or if you’d like a code snippet for the liquidity‑adaptive universe constructor.

===

I acknowledge that I fully understand the current
  situation. Here's my understanding:

  Current Status:
  - Phase 25b institutional-grade composite testing shows the
   latest model (Composite_Q_20_1.25×) fails all Investment
  Committee hurdles
  - Sharpe ratio of 0.65 vs required 1.0, max drawdown of
  -46.3% vs limit of -35%
  - Core issue: insufficient gross alpha density due to
  static factor weights and basic volatility overlay

  Key Technical Issues:
  1. Liquidity regime shift around 2020 - Vietnam market
  microstructure changed materially
  2. Static factor weighting - V:Q:M:R ≈ 50:25:20:5 weights
  not optimized
  3. Missing infrastructure - walk-forward optimizer, hybrid
  regime filter, non-linear cost model all inactive
  4. Period dependency - pre-2020 liquidity constraints
  inflate drawdowns

  Recommended Solution Path:
  1. Test multiple backtest windows (2018-25, 2020-25, full
  period with adaptive filters)
  2. Activate walk-forward optimizer with Bayesian priors
  3. Implement hybrid volatility + regime overlay
  4. Add realistic ADTV cost model and FOL constraints
  5. Target 2020-25 Sharpe ≥ 0.80 net before applying to full
   period

  Next Week Priority:
  - Refactor notebook with configurable date windows
  - Activate missing infrastructure components
  - Generate comparative tearsheets across time periods

  I'm ready to proceed with implementing these fixes. Should
  we start with the notebook refactoring to test different
  time windows, or would you prefer to begin with activating
  the walk-forward optimizer?
