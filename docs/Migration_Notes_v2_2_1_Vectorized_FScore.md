v2.2.1 Vectorized F-Score Migration Notes
=========================================

Overview
- Introduces vectorized F-Score calculators for Non-Financial (9), Banking (6), and Securities (5)
- Preserves v2.2.1 timing: fundamentals from lagged quarter; shares at analysis date and -1y
- Adds robust banking fallbacks using AvgCustomerDeposits when CustomerDeposits is unavailable
- Single-pass per sector group per date with cache priming

Enablement (config only)
Add to `config/strategy_config.yml` (or your active strategy config):
```yaml
f_score:
  use_vectorized_fscore_221: true
```

Observability
- New engine logs show: priming group sizes, factor coverage by family
- Vectorized calculators log: rows fetched, accrual rate, share tolerance rate, banking backfill counts

Performance
- Vectorized path targets ≤ 3 DB queries per sector group per date
- Warm cache per date improves hit rate across repeated runs

Testing
- Unit tests verify sector normalization, issuance tolerance, banking backfill, and securities deltas
- Integration tests verify cache hit behavior and fallback when banking views are absent
- Timing boundary tests assert lagged quarter selection near announcement delays

Rollout guidance
1) Keep flag off by default; run A/B using `production/scripts/run_ab_backtest_fscore_221.py`
2) Validate parity vs v2.1.1/neo-fix on a frozen snapshot for F-Score components
3) Enable flag in production once parity/performance confirmed


