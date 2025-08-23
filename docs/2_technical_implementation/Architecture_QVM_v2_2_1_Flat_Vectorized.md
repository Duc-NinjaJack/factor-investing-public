## QVM v2.2.1 Flat Vectorized Strategy — Architecture and Orchestration

Scope: Documents the current architecture as implemented by `production/tests/phase31_add_fscore/08_QVM_v221_flat_vectorized_strategy.py`. Focuses on phases, components, configuration, data dependencies, artifacts, and extension points.

### High-level Flow (Phased Orchestration)
1. Config + validation
2. Universe + factors (engine)
3. Holdings (top-N)
4. Prices + benchmark load
5. Portfolio returns (no-risk) + (with-risk overlay)
6. Tearsheet(s) and artifacts

### Core Responsibilities of the Runner
- Bootstrap project paths and logging with de-duplicated log output.
- Load and validate `strategy` and `backtest` configs; enforce version compatibility.
- Produce a content-addressed `run_id` (SHA-256 of normalized configs) and create artifact lineage.
- Construct rebalance calendar with flexible anchors (first trading day, mid-month, or fundamentals-lagged quarter anchors).
- For each rebalance date: build liquid universe, compute factors and composite, select top-N holdings.
- Build daily price matrix for selected tickers, align calendars, compute PnL with and without risk overlay.
- Persist all intermediate outputs and generate static and interactive tearsheets.

### Configuration & Validation
- Sources: YAML pointed via `--config` (see `production/scripts/configuration_manager.py`).
- Loaders: `load_strategy_config`, `load_backtest_config`.
- Validation: `validate_version_compatibility`, `validate_strategy_config`, `validate_backtest_config` (Pydantic schema: `rebalance.anchor`, `rebalance.lag_days`, `fundamentals.reporting_lag_days`, `slippage_bps`).
- Overrides: `--window YYYY-MM-DD:YYYY-MM-DD` sets `backtest.active_window`.
- Run lineage: `run_id = sha256(sorted(strategy_cfg, backtest_cfg))` → `artifacts/qvm_v221_flat_vectorized/{run_id}/` stores `strategy_config.json` and `backtest_config.json` snapshots.

### Database Layer
- Access: `production.database.connection.DatabaseManager().get_engine()`.
- Benchmark data (VNINDEX):
  - Primary: `etf_history (date, close)`
  - Fallback: `vcsc_daily_data_complete (trading_date, close_price)`
- Querying: parameterized SQL via `sqlalchemy.text`; timestamps normalized to `pd.DatetimeIndex`.

### Engine & Factors
- Engine: `production.engine.qvm_engine_v2_2_1_flat.QVMEngineV221Flat`.
- Vectorized F-Score path: installed by `production.engine.qvm_engine_v2_2_1_flat_vectorized.install_vectorized_fscore_221(engine)`.
- Feature flag: `strategy_cfg['f_score']['use_vectorized_fscore_221']` (default True); safe fallback to non-vectorized path with warning.
- Composite computation: `engine.calculate_qvm_composite_fixed(reb_date, universe)` returns per-ticker factor dict including `QVM_Composite`.
- Reference: see `docs/Migration_Notes_v2_2_1_Vectorized_FScore.md` and `docs/Factor_Generation.md` for factor specifics and v2.2.1 changes.

### Universe Construction
- Provider: `production.universe.constructors.get_liquid_universe(reb_date, db_engine)`.
- Error handling: empty universes are logged and skipped with diagnostics recorded.

### Rebalance Calendar
- Trading calendar basis: inferred from benchmark close series.
- Modes (precedence):
  - If `backtest.fundamentals.reporting_lag_days` set → `quarter_lag` using previous quarter end + lag days.
  - Else if `backtest.rebalance.anchor` in {`first_trading_day`, `mid_month`, `quarter_lag`} with optional `rebalance.lag_days` (strictly typed via schema).
  - Default: `first_trading_day` of each month.
- Post-generation alignment:
  - Build price matrix for unique holdings tickers: `production.backtester.core.build_daily_price_matrix`.
  - Align to trading days via `first_trading_day_calendar(price_matrix)` and keep only dates that have holdings.

### Holdings Selection
- Top-N selection by descending `QVM_Composite` using `portfolio_size = strategy_cfg['strategy']['portfolio']['portfolio_size']` (default 20).
- Persisted as `monthly_holdings.csv` with `(date, ticker)` rows after calendar alignment.

### PnL Computation
- Config: `BacktestConfig(transaction_cost_bps=backtest_cfg['transaction_cost_bps'], portfolio_size, slippage_bps=backtest_cfg.get('slippage_bps', 0))`.
- No-risk path: `run_daily_pnl(..., risk_overlay_fn=None)` → daily returns and equity curve.
- With-risk path: `run_daily_pnl(..., risk_overlay_fn=overlay_fn)` where `overlay_fn` is drawdown-aware cash allocation.
  - Risk overlay: `production.risk.overlay.drawdown_to_cash_allocation(benchmark_prices, current_date, rules)` with rules from `strategy_cfg['risk_management']['cash_allocation']`.
  - Optional overlays: `production.risk.overlay.ewma_drawdown_cash_allocation`, volatility targeting output can be logged to diagnostics.
- Outputs persisted as CSVs in artifacts directory:
  - `no_risk_returns.csv`, `with_risk_returns.csv`, `with_risk_cash.csv` (cash_allocation), `benchmark_returns.csv`.

### Tearsheet Generation
- Comparison: `generate_comparison_tearsheet(with_risk_returns, no_risk_returns, benchmark_returns, cash_allocations_df, strategy_cfg)`.
- Comprehensive views: `generate_comprehensive_tearsheet(...)` (with and without risk).
- Comparison plots: `create_comparison_plots(...)`.
- Static PNGs saved: `tearsheet_with_risk.png`, `tearsheet_without_risk.png`, `tearsheet_comparison.png`.
- Reference: `docs/Analytics_Sidecar.md` for reporting pipelines and sidecar usage.

### Diagnostics & Failure Handling
- Logging: single-stream, de-duplicated via custom `logging.Filter` to suppress repeated messages.
- Diagnostics CSV: `diagnostics.csv` tracks per-rebalance universe size, score availability, holdings count.
- Added telemetry: factor coverage rate, turnover per rebalance, factor calc latency (ms), optional non-vectorized vs vectorized timings (first date), optional SQL counters.
- Guardrails: the runner raises on missing benchmark series, empty holdings across window, or no valid aligned rebalance dates.

### Normalization Alignment Fix (v2.2.1)
- Sector-neutral z-scoring defensively coerces groupby results to a 1D `Series` and reindexes to the input row order to prevent accidental 2D alignment when a `DataFrame` leaks through from `groupby.apply`.
- This eliminates spurious "cannot align with a higher dimensional NDF" warnings without changing numerical results or sort order. Stable mergesort ordering is preserved.

### Data Lineage & Reproducibility
- Deterministic artifacts path via config hash.
- Persisted input configs and all derived series ensure experiment replayability.
- Calendar alignment ensures that holdings and price calendars are consistent.
- Enrichment: `environment_manifest.json` (Python/numpy/pandas versions, git SHA) and `integrity_manifest.json` (sha256 and size) accompany artifacts.

### Extension Points & Configuration Knobs
- Feature flags: enable/disable vectorized F-Score path.
- Rebalance anchors: `first_trading_day`, `mid_month`, or `quarter_lag` (+ configurable lag days).
- Portfolio sizing: `portfolio_size` for top-N selection.
- Transaction costs: `transaction_cost_bps` and `slippage_bps`.
- Windowing: `active_window` or `--window` override.
- Risk policy: drawdown-to-cash rules in strategy config.

### Interfaces and Key Functions (by responsibility)
- Config: `load_strategy_config`, `load_backtest_config`, `validate_version_compatibility`, `validate_strategy_config`.
- DB: `DatabaseManager.get_engine`.
- Engine: `QVMEngineV221Flat`, `install_vectorized_fscore_221`, `calculate_qvm_composite_fixed`.
- Universe: `get_liquid_universe(date, engine)`.
- Calendar & Prices: `build_daily_price_matrix`, `first_trading_day_calendar`.
- PnL: `run_daily_pnl`, `drawdown_to_cash_allocation`.
- Reporting: `generate_comparison_tearsheet`, `generate_comprehensive_tearsheet`, `create_comparison_plots`.

### CLI Usage
```bash
python production/tests/phase31_add_fscore/08_QVM_v221_flat_vectorized_strategy.py \
  --config /home/raymond/Documents/Projects/factor-investing-public/production/config/strategy_config_v2_0_1_simple.yml
```

### Cross-References
- `docs/Factor_Generation.md`: factor definitions and composite methodology.
- `docs/Migration_Notes_v2_2_1_Vectorized_FScore.md`: vectorized F-Score design and migration notes.
- `docs/Analytics_Sidecar.md`: reporting and analytics sidecar.
- `docs/QuickStart.md`: end-to-end run instructions.

### Artifact Naming & Layout
Artifacts for each run are saved under a timestamp-prefixed directory with a deterministic hash suffix:

```
artifacts/qvm_v221_flat_vectorized/
  latest -> 20250823T142000Z@942b59804339   # symlink to most recent run
  20250823T142000Z@942b59804339/            # {UTC timestamp}@{config-hash}
  20250820T101530Z@a1b2c3d4e5f6/
  ...
```

- Prefix: `YYYYMMDDTHHMMSSZ` in UTC for human-friendly chronological browsing.
- Suffix: First 12 hex chars of SHA-256 over normalized `strategy_config` + `backtest_config` for reproducibility/deduplication.
- Symlink: `latest` updated on each run to point to the newest directory.

Per-run directory contents:
```
artifacts/qvm_v221_flat_vectorized/{timestamp}@{run_id}/
  strategy_config.json
  backtest_config.json
  monthly_holdings.csv
  diagnostics.csv
  no_risk_returns.csv
  with_risk_returns.csv
  with_risk_cash.csv
  benchmark_returns.csv
  tearsheet_with_risk.png
  tearsheet_without_risk.png
  tearsheet_comparison.png
  environment_manifest.json
```


