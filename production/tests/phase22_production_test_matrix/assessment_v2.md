**EXECUTIVE SUMMARY**
Your diagnosis is correct: *Phase 16b*’s results are distorted by a survivorship‑biased, quasi‑static universe, whereas *Phase 22* finally deploys a fully dynamic, point‑in‑time liquid universe at every rebalance. The canonical engine therefore provides the first clean read‑out of true Value‑factor performance. The Value alpha is real but clearly regime‑dependent and still fails your institutional hurdle (Sharpe ≥ 1.5, max‑DD ≤ 15 %). You should (i) discard 15 / 16 series for performance appraisal, (ii) port the dynamic‑universe logic into the composite engines, and (iii) add volatility‑targeting, sector‑neutrality and tail‑risk overlays before concluding on alpha quality.

---

### DETAILED ANALYSIS

| Aspect                      | Phase 16 / 16b                                                                                                                                                                                                                                                                                                                           | Phase 22                                                                                                                                         | Verdict                                                      |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------ |
| **Universe construction**   | Uses the *factor\_scores\_qvm* table pre‑filtered to tickers present across the *entire* 2016‑2025 span. No `get_liquid_universe_dataframe()` call inside the back‑test loop.  See weighting back‑test in *16\_weighted\_composite\_engineering* where the loop merely subsets `factor_data_all` but never rebuilds liquidity universe.  | Explicitly invokes `get_liquid_universe_dataframe()` at every rebalance inside the `CanonicalBacktester` class (Phase 22 notebook, Phases 3–4).  | 16 / 16b is survivorship‑biased; 22 is clean.                |
| **Portfolio formation**     | Top‑quintile, variable count (≈15–40 stocks).                                                                                                                                                                                                                                                                                            | Fixed 20‑stock portfolio — more realistic for execution sizing.                                                                                  | Prefer 22’s concentration plus explicit liquidity screen.    |
| **Performance consistency** | Inflated in stress regimes; inconsistent Sharpe ratios (e.g., Sharpe 0.48–0.67 on static universe vs 0.50 on dynamic).                                                                                                                                                                                                                   | Stand‑alone Value 2016‑2025: Ann. Ret 16 %, Sharpe 0.61, Max DD ‑68 %.                                                                           | After bias removal, Value edge shrinks but remains positive. |
| **Factor coverage**         | Q, V, Reversal composites but with biased universe.                                                                                                                                                                                                                                                                                      | Only Value currently implemented.                                                                                                                | Need to port Q & Reversal into Phase 22 engine.              |

*Regime sensitivity*: The rolling‑Sharpe chart in the tearsheet (2020‑2025 window) oscillates between ‑1 and +3, indicating heavy cyclicality — typical for pure Value strategies in EMs.&#x20;

---

### IMPLEMENTATION NOTES

1. **Unify back‑testing framework**

   * Fork the `CanonicalBacktester` class and add a `factor_combiner` hook so that you can flexibly feed **Value**, **Quality** and **Reversal** (or any ML meta‑signal) after per‑period z‑scoring.
   * Ensure that `get_liquid_universe_dataframe()` is always called *inside* the rebalance loop, using the same ADTV ≥ 10 bn VND & ≥ 60 % trading‑day coverage filters you validated in 16b.
   * Store the final holdings table to S3 / MinIO for reproducibility and downstream risk analytics.

2. **Risk & transaction‑cost overlays**

   * **Target‑volatility**: Scale daily position weights so that ex‑ante annualised σ ≈ 15 %.
   * **Sector/industry caps**: Max 25 % per GICS sector to avoid concentration drawdowns typical in VN30 banks & real‑estate clusters.
   * **Tail hedge**: 5 % allocation to VN30 index puts (1‑month ATM) during “Stress/Bear” regimes defined in Phase 8, financed by reducing lowest‑ranked longs.

3. **Composite fine‑tuning**

   * Start with equal‑weighted QVR; then grid‑search weights on an expanding‑window walk‑forward (e.g., every 3 years) using Bayesian optimisation on Sharpe × (1 − k·turnover).
   * Consider a *dynamic* regime switch:

     ```
     weights = {Bull: (Q=0.3, V=0.2, Rev=0.5),
                Sideways: (0.4, 0.4, 0.2),
                Bear/Stress: (0.45, 0.45, 0.1)}
     ```

     Regime defined by realised 63‑day VN‑Index volatility > 25 % *or* 20‑day drawdown < ‑10 %.

4. **Validation protocol**

   * **In‑sample**: 2016‑2020; **out‑of‑sample**: 2021‑YTD.
   * Report information ratio versus VN‑Index and versus equal‑weighted liquid universe to ensure genuine alpha, not style drift.
   * Run **Monte‑Carlo bootstrapped** return paths (blocking by 21‑day chunks) to estimate Sharpe confidence interval.

5. **Execution / scalability checks**

   * Average order size ≤ 5 % of 20‑day ADTV to stay within SBV market‑impact guidelines.
   * Slippage model: 5 bps + 0.1·(order \$/ADTV). Feed this into the canonical engine’s cost module.

---

### RISK CONSIDERATIONS

| Risk                                        | Mitigation                                                                                                                               |
| ------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| **Data snooping & hyper‑parameter overfit** | Strict walk‑forward validation; lock model parameters every January; no look‑ahead on regime labels.                                     |
| **Liquidity regime shift**                  | Re‑tier ADTV threshold (e.g., 15 bn VND) if VN market cap surpasses 15 % CAGR; monitor quarterly.                                        |
| **Macro shocks (FX, SBV policy)**           | Add macro carry factor (USDVND forward points) as hedge overlay; suspend new longs if SBV hikes ≥ 100 bps intra‑quarter.                 |
| **Drawdown spike**                          | Daily stop‑loss at portfolio level: reduce gross exposure 50 % if trailing 10‑day P\&L < ‑7 %.                                           |
| **Tax / foreign ownership**                 | Maintain < 45 % cumulative foreign quota per ticker; automatically exclude hitting 47 % threshold to avoid forced sells under SSC rules. |

---

**Next Steps & KPIs**

| Milestone                        | Metric                      | Target                                                       | Deadline |
| -------------------------------- | --------------------------- | ------------------------------------------------------------ | -------- |
| Port QVR into Canonical Engine   | Code merge request          | Passed CI tests & 100 % unit‑test cover on rebalancing logic | +5 days  |
| Risk‑overlay prototype           | Sharpe (2016‑2025)          | ≥ 1.0                                                        | +10 days |
| Final optimisation run           | Sharpe, Max‑DD              | ≥ 1.5, ≤ 15 %                                                | +20 days |
| Deployment to live paper‑trading | Tracking error vs back‑test | ≤ 5 bps / day                                                | +30 days |

By following this roadmap you should convert the demonstrable, but presently under‑powered, Value alpha into an institutional‑grade multi‑factor strategy that meets Aureus Sigma’s performance mandate.
