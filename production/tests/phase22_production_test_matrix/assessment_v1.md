**VERDICT**

| Question                                                                                   | Answer                                                                                                                                                   | Confidence (1‑10) | Key Evidence                                                                                                                                                                                                                                                                                                                                                    |
| ------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Is survivorship bias the root cause of the performance gap between notebooks 15/16 and 22? | **Yes – highly likely.**                                                                                                                                 | **8**             | 15 b and 16 code build portfolios from a *pre‑assembled* `factor_data_all` table without refreshing the liquid universe each rebalance, so any stock that was delisted, suspended or IPO‑after‑start is missing.  Notebook 22 switches to an in‑loop call to `get_liquid_universe_dataframe()` and re‑normalises factors *point‑in‑time*, eliminating the bias. |
| Should 15/16 results be discarded?                                                         | **Treat them as *exploratory only*.**  Keep for debugging factor definitions, but **do not use for performance claims or capital‑allocation decisions.** |                   |                                                                                                                                                                                                                                                                                                                                                                 |
| Additional validation tests?                                                               | See section “Next diagnostics” below.                                                                                                                    |                   |                                                                                                                                                                                                                                                                                                                                                                 |

---

## 1 Technical Validation against Institutional Standards

### 1.1 Universe‑construction comparison

| Step             | Notebook 15 / 16                                                                                                                                                                      | Notebook 22                                                                    |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| Universe refresh | Static – relies on `factor_data_all` extracted once from Phase 14. No IPOs, no delist handling.                                                                                       | Dynamic – calls `get_liquid_universe_dataframe()` every rebalance.             |
| Selection filter | Top‑quintile **of the *survivorship‑biased* universe**.                                                                                                                               | Fixed 20‑stock portfolio selected from current liquid names.                   |
| Look‑ahead guard | Not explicitly checked; potential silent failures if `factor_data_all` contains forward‑filled NAs.                                                                                   | Uses explicit date filters in SQL and rebuilds factor z‑scores each rebalance. |
| Impact           | • Over‑weights large, durable incumbents.  • Misses 2018–2022 high‑growth IPOs (e.g. FPT Retail, GVR, VHM).  • Lower realised volatility ⇒ artificially higher risk‑adjusted metrics. | Captures true investable set; performance drops but Sharpe becomes realistic.  |

### 1.2 Quantifying the bias

| Metric (Standalone Value) | Static (2016‑25)\*\* | Dynamic (2022 engine) |
| ------------------------- | -------------------- | --------------------- |
| Annual return             | \~16 %               | \~13 %                |
| Sharpe ratio              | 0.61                 | 0.48                  |
| Max drawdown              | ‑69 %                | ‑67 %                 |

> **≈ 300 bps of annual return and ≈ 0.1‑0.15 Sharpe are attributable to survivorship bias** (back‑of‑envelope, using your tearsheets 2016‑25 vs 22).

### 1.3 Other integrity checks

| Criterion               | Status 15/16                         | Status 22                                                                  | Recommendation                                                          |
| ----------------------- | ------------------------------------ | -------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| Transaction costs       | 30 bps per trade – appropriate.      | same                                                                       | Consider slippage model using *percentage of ADV* once live.            |
| Liquidity brake         | Not enforced besides initial filter. | Implicit in `get_liquid_universe_dataframe()` (ADTV ≥ 10 bn VND, top 200). | Add position‑sizing cap (e.g. ≤ 5 % ADV per name).                      |
| Rebalancing feasibility | Quarterly, equal weight.             | Quarterly, 20‑name.                                                        | Stress‑test weekly turnover & market‑impact in a Monte‑Carlo framework. |

---

## 2 Performance‑Attribution Findings

1. **Value factor robustness**: survives the universe correction – still delivers > 300 bps alpha over VN‑Index after fees.
2. **Momentum & Quality**: positive in static test, *neutral or slightly negative* in dynamic test – indicates those signals were primarily harvesting survivorship bias.
3. **Regime dependency**: Sharpe collapses during 2023 bear‑cycle in both static and dynamic runs, confirming value’s typical pro‑cyclical risk. Dynamic universe merely reduces amplitude.

---

## 3 Strategic Roadmap (12‑week sprint)

| Phase                            | Weeks | Deliverables                                                                                                                                    | KPI                                                   |
| -------------------------------- | ----- | ----------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| **A. Methodology hardening**     | 1‑2   | • Replace all Phase 14 artefacts with on‑the‑fly queries.<br>• Add unit test: *backtest raises* if any holding’s listing‑date > rebalance‑date. | All tests pass; code coverage > 90 %.                 |
| **B. Diagnostics & attribution** | 3‑4   | • Run *paired* static vs dynamic backtests on identical factor sets.<br>• Decompose active return using Brinson‑Fachler.                        | Quantified % alpha from universe bias (< 5 % target). |
| **C. Factor‑stack rebuild**      | 5‑8   | • Momentum & Quality: re‑train with survivorship‑free sample.<br>• Bayesian factor weighting (Hierarchical Risk Parity).                        | Composite Sharpe ≥ 1.0; Max DD ≤ ‑35 %.               |
| **D. Risk overlays**             | 9‑10  | • Volatility targeting (ex‑ante 20 %).<br>• Max 8 % single‑name weight, sector caps 25 %.                                                       | Ex‑post volatility within ±2 %.                       |
| **E. Production readiness**      | 11‑12 | • Jenkins pipeline with checksum of input data.<br>• SBV/SSC compliance checklist (daily price‑limit monitoring, foreign cap alert).            | Automated nightly backtest with < 15 min runtime.     |

---

## 4 Next Diagnostics to Confirm the Fix

1. **Leave‑one‑year‑out walk‑forward** (2016‑25): verify Sharpe stability.
2. **In‑sample vs out‑of‑sample factor IC decay** after universe correction.
3. **IPO cohort study**: track P\&L of stocks that entered universe each year to isolate contribution lost in static runs.
4. **Survivorship‑free benchmark**: compare to VNFIN LEAD (liquid 50) instead of VN‑Index to remove small‑cap tail.

---

## 5 Risk Mitigation & Blind Spots

| Area                  | Blind Spot                                                                                       | Mitigation                                                                                           |
| --------------------- | ------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------- |
| Data vendor revisions | Historical restatements can re‑introduce look‑ahead.                                             | Store raw exchange dumps; snapshot daily.                                                            |
| Corporate actions     | Backtester currently uses *close‑to‑close* returns; ignores rights issues & stock splits delays. | Implement *total‑return factor* from exchange announcements.                                         |
| Liquidity crises      | ADTV filter may fail during market stress (e.g. 2022).                                           | Dynamic ADV percentile filter (≥ 20th pct of rolling distribution).                                  |
| Regulatory changes    | VN foreign‑ownership room (FOL) can block trade.                                                 | Integrate FOL‑adjusted free float into universe constructor; alert if weight × FOL > available room. |

---

### Bottom Line

*Your diagnosis is correct.* The performance “edge” seen in notebooks 15/16 was largely an artefact of survivorship bias from a static universe. Notebook 22’s dynamic, point‑in‑time universe brings the framework back to institutional best practice. Move forward using the roadmap above, and regard any historical metrics from the static runs as non‑reliable benchmarks.
