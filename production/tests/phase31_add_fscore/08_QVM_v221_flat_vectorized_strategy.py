#!/usr/bin/env python3
"""
08_QVM_v221_flat_vectorized_strategy.py
=======================================

Thin orchestration runner for QVM momentum strategy using v2.2.1 flat vectorized engine.

Phases:
1) Config + validation
2) Universe + factors (engine)
3) Holdings (top-N)
4) Prices + benchmark load
5) Portfolio returns (no-risk) + (with-risk)
6) Tearsheet(s) and artifacts

Run:
  python production/tests/phase31_add_fscore/08_QVM_v221_flat_vectorized_strategy.py \
    --config /home/raymond/Documents/Projects/factor-investing-public/production/config/strategy_config_v2_0_1_simple.yml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import concurrent.futures
from multiprocessing import cpu_count
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import sys
# Bootstrap sys.path so this script and its jupytext notebook can import the project packages
try:
    THIS_FILE = Path(__file__).resolve()
    search_roots = [THIS_FILE] + list(THIS_FILE.parents)
except NameError:
    # __file__ is not defined in notebooks; fall back to CWD search
    cwd = Path.cwd().resolve()
    search_roots = [cwd] + list(cwd.parents)

# Find project root that contains the 'production' package
project_root = None
for candidate in search_roots:
    if (candidate / 'production' / '__init__.py').exists():
        project_root = candidate
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break

# Ensure local 'scripts' directory is importable without hardcoding absolute paths
scripts_dir_candidates = []
if 'THIS_FILE' in globals():
    scripts_dir_candidates.append((THIS_FILE.parent / 'scripts'))
if project_root is not None:
    scripts_dir_candidates.append(project_root / 'production' / 'tests' / 'phase31_add_fscore' / 'scripts')

for scripts_dir in scripts_dir_candidates:
    if scripts_dir.exists() and str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

from production.engine.qvm_engine_v2_2_1_flat import QVMEngineV221Flat
from production.database.connection import DatabaseManager
from production.backtester.core import (
    BacktestConfig,
    build_daily_price_matrix,
    first_trading_day_calendar,
    run_daily_pnl,
)
from production.risk.overlay import drawdown_to_cash_allocation
from production.engine.qvm_engine_v2_2_1_flat_vectorized import install_vectorized_fscore_221
from sqlalchemy import text, event
import matplotlib.pyplot as plt

# 07_ utilities consolidated under scripts
from production.scripts.configuration_manager import (
    load_strategy_config,
    load_backtest_config,
    validate_version_compatibility,
)
from production.scripts.validation_manager import (
    validate_strategy_config,
    validate_backtest_config,
)
from production.scripts.tearsheet_generator import (
    calculate_performance_metrics,
    generate_comprehensive_tearsheet,
    generate_comparison_tearsheet,
    create_comparison_plots,
)


# Module-level guard to avoid repeating prematerialization hints across fan-outs
_PARALLEL_PREMATERIALIZATION_LOGGED = False


def _setup_logger() -> logging.Logger:
    class _DedupFilter(logging.Filter):
        def __init__(self):
            super().__init__()
            self._last_message = None

        def filter(self, record: logging.LogRecord) -> bool:
            msg = record.getMessage()
            if msg == self._last_message:
                return False
            self._last_message = msg
            return True

    logger = logging.getLogger("QVM_v221_runner")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        ch.addFilter(_DedupFilter())
        logger.addHandler(ch)
    # Also add dedup filter to root to reduce duplicate engine logs
    root = logging.getLogger()
    if not any(isinstance(f, _DedupFilter) for f in root.filters):
        root.addFilter(_DedupFilter())
    return logger


def _set_deterministic_env(logger: logging.Logger, seed: int = 42) -> None:
    """Set environment for deterministic execution and seed global RNGs.

    - Force BLAS single-threading to avoid nondeterministic reductions
    - Seed numpy and Python RNGs
    - Prefer float64 everywhere
    """
    try:
        # BLAS/OMP single-threading inside the runner
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        # Global RNG seeds
        try:
            np.random.seed(int(seed))
        except Exception:
            pass
        try:
            random.seed(int(seed))
        except Exception:
            pass
        # Prefer float64
        try:
            np.set_printoptions(precision=12, floatmode='maxprec', suppress=False)
        except Exception:
            pass
        logger.info("Deterministic env set: MKL/OMP threads=1, seeds initialized")
    except Exception as _e:
        logger.debug(f"Deterministic env setup skipped: {_e}")


def _hash_config(cfg: Dict) -> str:
    norm = json.dumps(cfg, sort_keys=True).encode("utf-8")
    return hashlib.sha256(norm).hexdigest()[:12]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _select_top_n_holdings(scores: Dict[str, Dict[str, float]], n: int) -> List[str]:
    """Select top-N tickers by `QVM_Composite` with deterministic tie-break.

    This simple selector is used as a fallback when constraints are not
    provided. It returns a list of tickers only.
    """
    df = pd.DataFrame.from_dict(scores, orient="index")
    if df.empty or "QVM_Composite" not in df.columns:
        return []
    df = df.copy()
    df["ticker"] = df.index.astype(str)
    ranked = df.sort_values(by=["QVM_Composite", "ticker"], ascending=[False, True], kind="mergesort")
    return ranked["ticker"].tolist()[:n]


def _load_sector_map(engine) -> Dict[str, str]:
    """Load sector mapping for all tickers from `master_info`.

    Returns a dict mapping ticker -> sector. Missing sectors will be mapped to
    'Unknown'. Errors are caught and result in an empty dict.
    """
    try:
        q = text("""
            SELECT ticker, sector
            FROM master_info
            WHERE sector IS NOT NULL
            ORDER BY ticker
        """)
        df = pd.read_sql(q, engine)
        if df.empty:
            return {}
        df = df.drop_duplicates(subset=["ticker"]).fillna({"sector": "Unknown"})
        return dict(zip(df["ticker"].astype(str), df["sector"].astype(str)))
    except Exception:
        return {}


def _load_adv_20(engine, as_of: pd.Timestamp, tickers: List[str]) -> Dict[str, float]:
    """Compute 20-trading-day ADV (VND) for provided tickers ending at `as_of`.

    ADV is computed as the average of `total_value` over the last 20 trading days
    available in `vcsc_daily_data_complete`. Returns mapping ticker -> ADV_VND.
    Missing tickers will be absent from the result.
    """
    if not tickers:
        return {}
    try:
        q = text("""
            SELECT t.ticker, AVG(t.total_value) AS adv_vnd
            FROM (
              SELECT trading_date, ticker, total_value,
                     ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY trading_date DESC) AS rn
              FROM vcsc_daily_data_complete
              WHERE ticker IN :tickers AND trading_date <= :as_of
            ) t
            WHERE t.rn <= 20
            GROUP BY t.ticker
        """)
        df = pd.read_sql(q, engine, params={"tickers": tuple(set(map(str, tickers))), "as_of": pd.Timestamp(as_of)})
        if df.empty:
            return {}
        return dict(zip(df["ticker"].astype(str), df["adv_vnd"].astype(float)))
    except Exception:
        return {}


def _select_constrained_holdings(
    scores: Dict[str, Dict[str, float]],
    n: int,
    sector_map: Optional[Dict[str, str]] = None,
    sector_cap: Optional[float] = None,
    prev_holdings: Optional[set] = None,
    min_hold_months: int = 0,
    ages: Optional[Dict[str, int]] = None,
    adv_map_vnd: Optional[Dict[str, float]] = None,
    adv_participation_cap: Optional[float] = None,
    portfolio_notional_vnd: Optional[float] = None,
) -> List[str]:
    """Select holdings under sector/ADV caps and churn control.

    - Sector caps: enforce max per-sector count = floor(sector_cap * n), at least 1.
    - ADV cap: if `portfolio_notional_vnd` and `adv_participation_cap` provided, ensure
      equal-weight notional per name (portfolio_notional_vnd / n) <= adv_participation_cap * ADV_20d_vnd.
    - Churn control: if `min_hold_months` > 0 and `ages` provided, attempt to retain
      names with age < min_hold_months.
    """
    base = pd.DataFrame.from_dict(scores, orient="index")
    if base.empty or "QVM_Composite" not in base.columns:
        return []
    base = base.copy()
    base["ticker"] = base.index.astype(str)
    ranked = base.sort_values(["QVM_Composite", "ticker"], ascending=[False, True], kind="mergesort")["ticker"].tolist()

    # Pre-compute caps
    max_per_sector = None
    if sector_map and isinstance(sector_cap, (int, float)) and sector_cap > 0:
        max_per_sector = max(1, int(np.floor(float(sector_cap) * float(n))))

    # Determine forced-retain tickers based on min holding period
    forced: List[str] = []
    if min_hold_months and ages and prev_holdings:
        for t in sorted(prev_holdings):  # deterministic order
            if ages.get(t, 0) < int(min_hold_months):
                forced.append(t)

    # Build selection
    selected: List[str] = []
    sector_counts: Dict[str, int] = {}

    def _sector_ok(t: str) -> bool:
        if max_per_sector is None:
            return True
        sec = (sector_map or {}).get(t, "Unknown")
        return sector_counts.get(sec, 0) < max_per_sector

    def _adv_ok(t: str) -> bool:
        if portfolio_notional_vnd is None or adv_participation_cap is None or not adv_map_vnd:
            return True
        adv_vnd = float(adv_map_vnd.get(t, 0.0))
        if adv_vnd <= 0.0:
            return False
        equal_notional = float(portfolio_notional_vnd) / float(n)
        return equal_notional <= float(adv_participation_cap) * adv_vnd

    # 1) Retain forced holdings first while respecting caps
    for t in forced:
        if t in ranked and t not in selected and _sector_ok(t) and _adv_ok(t):
            selected.append(t)
            if max_per_sector is not None:
                sec = (sector_map or {}).get(t, "Unknown")
                sector_counts[sec] = sector_counts.get(sec, 0) + 1
        if len(selected) >= n:
            break

    # 2) Fill remaining slots by ranked order under caps
    if len(selected) < n:
        for t in ranked:
            if t in selected:
                continue
            if not _sector_ok(t):
                continue
            if not _adv_ok(t):
                continue
            selected.append(t)
            if max_per_sector is not None:
                sec = (sector_map or {}).get(t, "Unknown")
                sector_counts[sec] = sector_counts.get(sec, 0) + 1
            if len(selected) >= n:
                break

    return selected[:n]


def _compute_exposure_schedule(
    price_matrix: pd.DataFrame,
    benchmark_returns: pd.Series,
    monthly_holdings: Dict[pd.Timestamp, List[str]],
    monthly_calendar: List[pd.Timestamp],
    vol_target_ann: float = 0.12,
    vol_lookback_days: int = 63,
    beta_lookback_days: int = 126,
    exposure_bounds: Tuple[float, float] = (0.0, 1.0),
) -> pd.Series:
    """Create a daily exposure schedule to target volatility and beta.

    Returns a Series indexed by trading date with values in [0,1] indicating
    exposure (1 - cash). If insufficient history, defaults to 1.0 exposure.
    """
    if price_matrix.empty or not monthly_calendar:
        return pd.Series(dtype="float64")

    daily_ret = price_matrix.sort_index().pct_change(fill_method=None).fillna(0.0)
    bench_ret = benchmark_returns.sort_index()
    all_dates = daily_ret.index
    sched = pd.Series(index=all_dates, dtype="float64")

    # Build per-period equal-weight portfolio returns
    for i, start in enumerate(monthly_calendar):
        try:
            end = monthly_calendar[i + 1]
        except IndexError:
            end = all_dates[-1] + pd.Timedelta(days=1)
        period_mask = (all_dates >= start) & (all_dates < end)
        period_dates = all_dates[period_mask]
        if start not in monthly_holdings or len(period_dates) == 0:
            continue
        tickers = [t for t in monthly_holdings[start] if t in daily_ret.columns]
        if not tickers:
            continue
        w = np.full(shape=(len(tickers),), fill_value=1.0 / float(len(tickers)))
        port_ret = (daily_ret.loc[period_dates, tickers] @ w)

        for d in period_dates:
            # Vol targeting
            hist = port_ret.loc[:d].tail(vol_lookback_days)
            if len(hist) < max(20, int(vol_lookback_days/3)):
                e_vol = 1.0
            else:
                vol_ann = float(hist.std(ddof=0)) * np.sqrt(252.0)
                e_vol = 1.0 if vol_ann <= 1e-12 else float(vol_target_ann) / float(vol_ann)
            # Beta targeting
            hist_p = port_ret.loc[:d].tail(beta_lookback_days)
            # Align benchmark returns to portfolio history index; tolerate gaps at period start
            hist_b = bench_ret.reindex(hist_p.index)
            valid_mask = hist_p.notna() & hist_b.notna()
            if valid_mask.sum() < max(20, int(beta_lookback_days/3)) or hist_b[valid_mask].std(ddof=0) <= 1e-12:
                beta_hat = 1.0
            else:
                # OLS slope: cov(p,b)/var(b)
                p_vals = hist_p[valid_mask]
                b_vals = hist_b[valid_mask]
                beta_hat = float(np.cov(p_vals, b_vals, ddof=0)[0, 1]) / float(b_vals.var(ddof=0))
                if not np.isfinite(beta_hat) or beta_hat <= 0:
                    beta_hat = 1.0
            e_beta = 1.0 / beta_hat
            e = min(max(e_vol, exposure_bounds[0]), exposure_bounds[1])
            e = min(e, max(min(e_beta, exposure_bounds[1]), exposure_bounds[0]))
            sched.loc[d] = float(np.clip(e, exposure_bounds[0], exposure_bounds[1]))

    # Forward fill any gaps and clamp
    sched = sched.ffill().fillna(1.0).clip(lower=exposure_bounds[0], upper=exposure_bounds[1])
    return sched


def _parse_active_window(backtest_cfg: Dict) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Support both dict windows {start,end} and string keys into backtest_windows."""
    aw = backtest_cfg.get("active_window", {})
    if isinstance(aw, dict):
        start_date = pd.to_datetime(aw.get("start"))
        end_date = pd.to_datetime(aw.get("end"))
        return start_date, end_date
    windows = backtest_cfg.get("backtest_windows", {})
    if isinstance(aw, str) and aw in windows:
        start_date = pd.to_datetime(windows[aw].get("start"))
        end_date = pd.to_datetime(windows[aw].get("end"))
        return start_date, end_date
    # Fallback: first available window
    if windows:
        key = sorted(windows.keys())[0]
        start_date = pd.to_datetime(windows[key].get("start"))
        end_date = pd.to_datetime(windows[key].get("end"))
        return start_date, end_date
    raise ValueError("active_window not found and no backtest_windows available")


def _previous_quarter_end(d: pd.Timestamp) -> pd.Timestamp:
    """Return the previous quarter-end date for a given date's month."""
    y = d.year
    m = d.month
    if m <= 3:
        return pd.Timestamp(year=y - 1, month=12, day=31)
    if m <= 6:
        return pd.Timestamp(year=y, month=3, day=31)
    if m <= 9:
        return pd.Timestamp(year=y, month=6, day=30)
    return pd.Timestamp(year=y, month=9, day=30)


def _first_trading_on_or_after(date_like: pd.Timestamp, trading_index: pd.DatetimeIndex) -> pd.Timestamp:
    pos = trading_index.searchsorted(date_like, side="left")
    if pos >= len(trading_index):
        return pd.NaT
    return trading_index[pos]


def _build_rebalance_calendar(
    benchmark_prices: pd.Series,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    backtest_cfg: Dict,
) -> List[pd.Timestamp]:
    """Construct monthly rebalance dates with support for different anchors.

    Modes (by precedence):
      - If backtest_cfg['fundamentals']['reporting_lag_days'] is set: use 'quarter_lag' with that lag
      - Else if backtest_cfg['rebalance']['anchor'] in {'first_trading_day','mid_month','quarter_lag'}
        and optional 'lag_days': use specified
      - Else default to 'first_trading_day'
    """
    trading_index = benchmark_prices.index
    # Determine mode and lag
    mode = "first_trading_day"
    lag_days = None
    fundamentals_cfg = backtest_cfg.get("fundamentals", {})
    if isinstance(fundamentals_cfg, dict) and isinstance(fundamentals_cfg.get("reporting_lag_days"), (int, float)):
        mode = "quarter_lag"
        lag_days = int(fundamentals_cfg["reporting_lag_days"])
    else:
        rebalance_cfg = backtest_cfg.get("rebalance", {})
        if isinstance(rebalance_cfg, dict):
            mode = rebalance_cfg.get("anchor", mode)
            if isinstance(rebalance_cfg.get("lag_days"), (int, float)):
                lag_days = int(rebalance_cfg["lag_days"])

    # Generate per-month anchor dates within the requested window
    months = pd.period_range(start=start_date.to_period("M"), end=end_date.to_period("M"), freq="M")
    anchors: List[pd.Timestamp] = []
    for p in months:
        month_start = p.to_timestamp("D")
        if mode == "mid_month":
            candidate = pd.Timestamp(year=month_start.year, month=month_start.month, day=15)
            anchor = _first_trading_on_or_after(candidate, trading_index)
        elif mode == "quarter_lag":
            q_end = _previous_quarter_end(month_start)
            eff_lag = lag_days if lag_days is not None else 60
            candidate = q_end + pd.Timedelta(days=eff_lag)
            anchor = _first_trading_on_or_after(candidate, trading_index)
        else:  # first_trading_day
            candidate = month_start
            anchor = _first_trading_on_or_after(candidate, trading_index)
        if pd.isna(anchor):
            continue
        if anchor < start_date or anchor > end_date:
            continue
        anchors.append(anchor)
    # Deduplicate and sort
    anchors = sorted(pd.to_datetime(list(dict.fromkeys(anchors))))
    return anchors


def _compute_factor_coverage(scores: Dict[str, Dict[str, float]]) -> float:
    """Compute fraction of available individual_factor entries across tickers.
    Returns NaN-safe coverage in [0,1].
    """
    try:
        total = 0
        non_null = 0
        for _ticker, comp in scores.items():
            indiv = comp.get('individual_factors', {}) or {}
            for _name, val in indiv.items():
                total += 1
                if pd.notna(val):
                    non_null += 1
        return float(non_null) / float(total) if total > 0 else 0.0
    except Exception:
        return 0.0


def _install_sql_counters(sql_engine):
    counters = {"queries": 0, "rows": 0}

    @event.listens_for(sql_engine, "after_cursor_execute")
    def _count_calls(conn, cursor, statement, parameters, context, executemany):
        try:
            counters["queries"] += 1
            rc = getattr(cursor, 'rowcount', 0) or 0
            if isinstance(rc, int):
                counters["rows"] += max(0, rc)
        except Exception:
            pass
    return counters


def _worker_process(reb_date: pd.Timestamp, strategy_cfg_local: Dict, backtest_cfg_local: Dict, portfolio_size: int) -> Dict:
    import time as _time
    import os as _os
    import random as _random
    import numpy as _np
    from production.universe.constructors import get_liquid_universe as _get_univ
    from production.engine.qvm_engine_v2_2_1_flat import QVMEngineV221Flat as _Eng
    from production.database.connection import DatabaseManager as _DBM
    from production.engine.qvm_engine_v2_2_1_flat_vectorized import install_vectorized_fscore_221 as _install_vec
    # Deterministic per-worker environment
    try:
        _os.environ.setdefault("MKL_NUM_THREADS", "1")
        _os.environ.setdefault("OMP_NUM_THREADS", "1")
        _os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    except Exception:
        pass
    try:
        seed = int(_os.environ.get("QVM_SEED", "42"))
        _np.random.seed(seed)
        _random.seed(seed)
    except Exception:
        pass
    try:
        _np.set_printoptions(precision=12, floatmode='maxprec', suppress=False)
    except Exception:
        pass
    # Safety: prevent DB access in child process if parent requested
    import logging as _logging
    _wlog = _logging.getLogger("QVM_v221_worker")
    try:
        if _os.environ.get('DISABLE_DB_IN_CHILDREN') == '1':
            _wlog.error("DB access in child process is disabled by parent; short-circuiting worker without DB handle")
            return {
                "date": pd.Timestamp(reb_date),
                "universe_size": 0,
                "had_scores": False,
                "holdings": [],
                "elapsed_ms_universe": None,
                "elapsed_ms_factors": None,
                "factor_coverage_rate": 0.0,
                "sql_queries": 0,
                "sql_rows": 0,
                "candidates": 0,
                "fail_trading_days": 0,
                "fail_adtv": 0,
            }
    except Exception:
        pass
    # Per-worker DB engine
    _db = _DBM()
    _db_engine = _db.get_engine()
    db_counters = _install_sql_counters(_db_engine)
    # Engine instance per worker
    _engine = _Eng()
    eng_counters = _install_sql_counters(_engine.engine)
    try:
        use_vec_local = bool(strategy_cfg_local.get("f_score", {}).get("use_vectorized_fscore_221", True))
        _engine.use_vectorized_fscore_221 = use_vec_local
        if use_vec_local:
            try:
                _install_vec(_engine)
            except Exception:
                pass
    except Exception:
        pass

    # Optional warm-up window (compute-only, emit-only target date)
    warmup_months = 0
    try:
        # Default to 12 months warm-up unless configured otherwise
        warmup_months = int(backtest_cfg_local.get('parallel', {}).get('warmup_months', 12))
        warmup_months = max(0, min(24, warmup_months))
    except Exception:
        warmup_months = 0

    dates_to_compute = [reb_date]
    if warmup_months > 0:
        try:
            # Build prior monthly anchors using benchmark calendar if available in parent scope
            anchor_idx = None
            try:
                # When called from the parallel path, we don't have benchmark series; synthesize month starts
                anchor_idx = pd.date_range(end=reb_date, periods=warmup_months+1, freq='MS').to_list()
            except Exception:
                anchor_idx = [reb_date]
            candidates = sorted(set([d for d in anchor_idx if d < reb_date]))[-warmup_months:]
            dates_to_compute = candidates + [reb_date]
        except Exception:
            dates_to_compute = [reb_date]

    last_scores = None
    # Feature flags (default OFF)
    try:
        selection_enable_constraints = bool(backtest_cfg_local.get('selection', {}).get('enable_constraints', False))
    except Exception:
        selection_enable_constraints = False
    elapsed_universe_ms = None
    for d in dates_to_compute:
        # Universe
        t_u0 = _time.perf_counter()
        try:
            from production.universe.constructors import get_liquid_universe_and_counts as _get_univ_counts
            univ, u_counts = _get_univ_counts(d, _db_engine)
        except Exception:
            univ = _get_univ(d, _db_engine)
            u_counts = {'candidates': 0, 'fail_trading_days': 0, 'fail_adtv': 0, 'selected_count': int(len(univ) or 0)}
        elapsed_universe_ms = (_time.perf_counter() - t_u0) * 1000.0
        if not univ:
            if d != reb_date:
                # Warm-up miss; continue to next date
                continue
            return {
                "date": pd.Timestamp(reb_date),
                "universe_size": 0,
                "had_scores": False,
                "holdings": [],
                "elapsed_ms_universe": round(float(elapsed_universe_ms), 3),
                "elapsed_ms_factors": None,
                "factor_coverage_rate": 0.0,
                "sql_queries": db_counters["queries"] + eng_counters["queries"],
                "sql_rows": db_counters["rows"] + eng_counters["rows"],
                **u_counts,
            }
        # Factors
        t_f0 = _time.perf_counter()
        last_scores = _engine.calculate_qvm_composite_fixed(d, univ)
        elapsed_factors_ms = (_time.perf_counter() - t_f0) * 1000.0
        _et = getattr(_engine, '_last_timings', {}) or {}
        if d != reb_date:
            # Warm-up iteration – discard outputs but keep caches warm
            continue
        if not last_scores:
            return {
                "date": pd.Timestamp(reb_date),
                "universe_size": len(univ),
                "had_scores": False,
                "holdings": [],
                "elapsed_ms_universe": round(float(elapsed_universe_ms), 3),
                "elapsed_ms_factors": round(float(elapsed_factors_ms), 3),
                "elapsed_ms_quality": _et.get("elapsed_ms_quality"),
                "elapsed_ms_value": _et.get("elapsed_ms_value"),
                "elapsed_ms_momentum": _et.get("elapsed_ms_momentum"),
                "elapsed_ms_lowvol": _et.get("elapsed_ms_lowvol"),
                "elapsed_ms_fscore": _et.get("elapsed_ms_fscore"),
                "elapsed_ms_fcf": _et.get("elapsed_ms_fcf"),
                "factor_coverage_rate": 0.0,
                "sql_queries": db_counters["queries"] + eng_counters["queries"],
                "sql_rows": db_counters["rows"] + eng_counters["rows"],
                **u_counts,
            }
        coverage = _compute_factor_coverage(last_scores)
        # Apply selection rules in worker path; constraints optional
        try:
            _sector_map = _load_sector_map(_db_engine)
        except Exception:
            _sector_map = {}
        try:
            _sector_cap = float(backtest_cfg_local.get('universe', {}).get('sector_concentration_limit', 0.20))
        except Exception:
            _sector_cap = 0.20
        try:
            _assumed_notional_vnd = float(backtest_cfg_local.get('portfolio', {}).get('assumed_notional_vnd', None))
        except Exception:
            _assumed_notional_vnd = None
        try:
            _adv_participation_cap = float(backtest_cfg_local.get('cost_model', {}).get('max_participation_rate', 0.05))
        except Exception:
            _adv_participation_cap = 0.05
        try:
            _adv_map = _load_adv_20(_db_engine, reb_date, list(last_scores.keys())) if _assumed_notional_vnd else {}
        except Exception:
            _adv_map = {}
        if selection_enable_constraints:
            holdings = _select_constrained_holdings(
                scores=last_scores,
                n=portfolio_size,
                sector_map=_sector_map,
                sector_cap=_sector_cap,
                prev_holdings=None,
                min_hold_months=0,
                ages=None,
                adv_map_vnd=_adv_map,
                adv_participation_cap=_adv_participation_cap,
                portfolio_notional_vnd=_assumed_notional_vnd,
            )
        else:
            holdings = _select_top_n_holdings(last_scores, portfolio_size)[:portfolio_size]
        return {
            "date": pd.Timestamp(reb_date),
            "universe_size": len(univ),
            "had_scores": True,
            "holdings": holdings or [],
            "elapsed_ms_universe": round(float(elapsed_universe_ms), 3),
            "elapsed_ms_factors": round(float(elapsed_factors_ms), 3),
            "elapsed_ms_quality": _et.get("elapsed_ms_quality"),
            "elapsed_ms_value": _et.get("elapsed_ms_value"),
            "elapsed_ms_momentum": _et.get("elapsed_ms_momentum"),
            "elapsed_ms_lowvol": _et.get("elapsed_ms_lowvol"),
            "elapsed_ms_fscore": _et.get("elapsed_ms_fscore"),
            "elapsed_ms_fcf": _et.get("elapsed_ms_fcf"),
            "factor_coverage_rate": round(float(coverage), 4),
            "sql_queries": db_counters["queries"] + eng_counters["queries"],
            "sql_rows": db_counters["rows"] + eng_counters["rows"],
            **u_counts,
        }

def main():
    logger = _setup_logger()
    _set_deterministic_env(logger, seed=int(os.environ.get("QVM_SEED", "42")))
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False, default=None)
    parser.add_argument("--window", type=str, required=False, help="YYYY-MM-DD:YYYY-MM-DD override")
    parser.add_argument("--jobs", type=int, required=False, help="Parallel workers for monthly dates")
    parser.add_argument("--force-parallel", action="store_true", help="Enable ProcessPoolExecutor parallel path (unsafe) for baseline testing")
    args = parser.parse_args()

    # Phase 1: Config + validation
    strategy_cfg = load_strategy_config(args.config)
    backtest_cfg = load_backtest_config(args.config)
    validate_version_compatibility(strategy_cfg, backtest_cfg)
    validate_strategy_config(strategy_cfg, logger)
    # Enforce backtest schema validation here to surface errors early
    try:
        backtest_cfg = validate_backtest_config(backtest_cfg, logger)
    except Exception as e:
        logger.error(f"Backtest config validation error: {e}")
        raise

    if args.window:
        start_s, end_s = args.window.split(":")
        backtest_cfg["active_window"] = {"start": start_s, "end": end_s}

    run_id = _hash_config({"strategy": strategy_cfg, "backtest": backtest_cfg})
    # Time-prefixed run directory plus 'latest' symlink for discoverability
    from datetime import datetime as _dt
    ts_prefix = _dt.utcnow().strftime("%Y%m%dT%H%M%SZ")
    artifacts_dir = Path("artifacts/qvm_v221_flat_vectorized") / f"{ts_prefix}@{run_id}"
    _ensure_dir(artifacts_dir)
    try:
        latest_link = artifacts_dir.parent / 'latest'
        if latest_link.is_symlink() or latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(artifacts_dir.name)
    except Exception as _e:
        logger.debug(f"Could not update latest symlink: {_e}")
    # Persist config snapshot for lineage
    with open(artifacts_dir / 'strategy_config.json', 'w', encoding='utf-8') as f:
        json.dump(strategy_cfg, f, indent=2, sort_keys=True)
    with open(artifacts_dir / 'backtest_config.json', 'w', encoding='utf-8') as f:
        # Ensure JSON-serializable snapshot
        def _default(o):
            import datetime as _dt
            if isinstance(o, (_dt.date, _dt.datetime)):
                return o.isoformat()
            return str(o)
        json.dump(backtest_cfg, f, indent=2, sort_keys=True, default=_default)

    # DB engine
    db = DatabaseManager()
    db_engine = db.get_engine()

    # Strategy parameters
    portfolio_size = strategy_cfg.get("strategy", {}).get("portfolio", {}).get("portfolio_size", 50)
    tc_bps = backtest_cfg.get("transaction_cost_bps", 10.0)
    # Feature flags (default OFF for golden equivalence)
    try:
        selection_enable_constraints = bool(backtest_cfg.get('selection', {}).get('enable_constraints', False))
    except Exception:
        selection_enable_constraints = False
    try:
        risk_overlay_mode = str(backtest_cfg.get('risk_overlay', {}).get('mode', 'drawdown_based')).strip().lower()
    except Exception:
        risk_overlay_mode = 'drawdown_based'
    start_date, end_date = _parse_active_window(backtest_cfg)

    # Phase 2-3: Universe, factors, holdings per monthly rebalance date
    import time as _time
    _run_t0 = _time.perf_counter()
    engine = QVMEngineV221Flat()
    # Wire normalization control from config if present
    try:
        normalization_cfg = backtest_cfg.get('normalization', {}) if isinstance(backtest_cfg, dict) else {}
        min_sector_size_cfg = normalization_cfg.get('min_sector_size')
        if isinstance(min_sector_size_cfg, (int, float)) and min_sector_size_cfg > 0:
            engine.min_sector_size = int(min_sector_size_cfg)
            logger.info(f"Normalization control: min_sector_size={engine.min_sector_size}")
    except Exception as _e:
        logger.debug(f"Normalization config not applied: {_e}")

    # Align earnings announcement delay with fundamentals.reporting_lag_days if provided
    try:
        fundamentals_cfg = backtest_cfg.get('fundamentals', {}) if isinstance(backtest_cfg, dict) else {}
        lag_days = fundamentals_cfg.get('reporting_lag_days')
        if isinstance(lag_days, (int, float)) and lag_days > 0:
            engine.data_timing_config['earnings_announcement_delay_days'] = int(lag_days)
            logger.info(f"Data timing: earnings_announcement_delay_days set to {engine.data_timing_config['earnings_announcement_delay_days']} from backtest config")
        else:
            # fallback: retain engine default
            logger.debug("Data timing: using engine default earnings_announcement_delay_days")
    except Exception as _e:
        logger.debug(f"Data timing config not applied: {_e}")
    # Enable vectorized F-Score path (v2.2.1) via config flag (default ON)
    try:
        use_vec = bool(strategy_cfg.get("f_score", {}).get("use_vectorized_fscore_221", True))
        engine.use_vectorized_fscore_221 = use_vec
        if use_vec:
            install_vectorized_fscore_221(engine)
            logger.info("Feature Flag: USE_VECTORIZED_F_SCORE_221=ON — vectorized F-Score installed (runner)")
        else:
            logger.info("Feature Flag: USE_VECTORIZED_F_SCORE_221=OFF — using non-vectorized F-Score path")
    except Exception as e:
        logger.warning(f"Vectorized F-Score enablement failed; falling back to non-vectorized path: {e}")
    all_holdings: Dict[pd.Timestamp, List[str]] = {}
    monthly_dates: List[pd.Timestamp] = []

    # Build a trading calendar from benchmark to anchor dates (parameterized queries with fallbacks)
    def _load_benchmark_series(engine, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> Tuple[pd.Series, pd.Series]:
        # Try etf_history first (date, close)
        try:
            q1 = text("""
                SELECT date AS date, close AS close_price
                FROM etf_history
                WHERE ticker = 'VNINDEX' AND date BETWEEN :start AND :end
                ORDER BY date
            """)
            df1 = pd.read_sql(q1, engine, params={"start": start_dt, "end": end_dt})
            if not df1.empty:
                df1["date"] = pd.to_datetime(df1["date"]) 
                close = df1.set_index("date")["close_price"].sort_index()
                return close, close.pct_change().dropna()
        except Exception:
            pass

        # Fallback to vcsc_daily_data_complete (trading_date, close_price)
        q2 = text("""
            SELECT trading_date AS date, close_price AS close_price
            FROM vcsc_daily_data_complete
            WHERE ticker = 'VNINDEX' AND trading_date BETWEEN :start AND :end
            ORDER BY trading_date
        """)
        df2 = pd.read_sql(q2, engine, params={"start": start_dt, "end": end_dt})
        if df2.empty:
            return pd.Series(dtype=float), pd.Series(dtype=float)
        df2["date"] = pd.to_datetime(df2["date"]) 
        close = df2.set_index("date")["close_price"].sort_index()
        return close, close.pct_change().dropna()

    t_b0 = _time.perf_counter()
    benchmark_prices, benchmark_returns = _load_benchmark_series(db_engine, start_date, end_date)
    elapsed_ms_benchmark = (_time.perf_counter() - t_b0) * 1000.0
    if benchmark_prices.empty:
        # System hardening: counter + diagnostics
        logger.error("Missing benchmark series; aborting run")
        pd.DataFrame([{ 'date': start_date, 'error': 'missing_benchmark' }]).to_csv(artifacts_dir / 'diagnostics.csv', index=False)
        raise RuntimeError("Benchmark data not found; cannot proceed")

    # Rebalance calendar with fundamentals-aware options
    monthly_dates = _build_rebalance_calendar(
        benchmark_prices=benchmark_prices,
        start_date=start_date,
        end_date=end_date,
        backtest_cfg=backtest_cfg,
    )

    # Persist normalization summary and run metadata under this run's artifacts directory
    try:
        normalization_cfg = backtest_cfg.get('normalization', {}) if isinstance(backtest_cfg, dict) else {}
        norm_min_sector_size = normalization_cfg.get('min_sector_size', 'dynamic')
        norm_robust = normalization_cfg.get('robust', 'median_mad')
        norm_fallback = normalization_cfg.get('fallback', ['sector', 'industry', 'market'])
        norm_summary = {
            'min_sector_size': norm_min_sector_size,
            'robust': norm_robust,
            'fallback': norm_fallback,
            # Engine applies James–Stein style shrinkage for thin groups
            'james_stein_shrinkage_for_thin_groups': True,
        }
        cal_df = pd.DataFrame({'anchor': monthly_dates}) if monthly_dates else pd.DataFrame()
        from production.utils.run_artifacts import write_run_artifact as _write_run_artifact
        seeds_manifest = {
            'QVM_SEED': int(os.environ.get('QVM_SEED', '42'))
        }
        run_cfg_snapshot = {
            'strategy': strategy_cfg,
            'backtest': backtest_cfg,
        }
        _write_run_artifact(
            run_config=run_cfg_snapshot,
            calendar_anchors=cal_df,
            seeds=seeds_manifest,
            extra={'normalization': norm_summary},
            base_dir=str(artifacts_dir),
        )
        try:
            if isinstance(norm_fallback, list):
                fb_str = ','.join(norm_fallback)
            else:
                fb_str = str(norm_fallback)
            logger.info(f"Normalization summary: min_sector_size={norm_min_sector_size} | robust={norm_robust} | fallback={fb_str}")
        except Exception:
            pass
    except Exception as _e:
        logger.debug(f"Run artifact write skipped: {_e}")

    diag_rows = []
    universe_diag_rows = []
    empty_universe_count = 0
    empty_scores_count = 0
    empty_holdings_count = 0
    benchmark_both = bool(strategy_cfg.get("f_score", {}).get("benchmark_both", False))
    did_benchmark_both = False

    # Parallel or sequential execution
    jobs = int(args.jobs) if getattr(args, 'jobs', None) is not None else max(1, min(4, (cpu_count() or 1)))
    results: List[Dict] = []
    timings_rows: List[Dict] = []
    if jobs <= 1 or not getattr(args, 'force_parallel', False):
        import time
        # Install SQL counters on shared runner DB and engine DB
        db_counters = _install_sql_counters(db_engine)
        engine = QVMEngineV221Flat()
        try:
            use_vec = bool(strategy_cfg.get("f_score", {}).get("use_vectorized_fscore_221", True))
            engine.use_vectorized_fscore_221 = use_vec
            if use_vec:
                install_vectorized_fscore_221(engine)
                logger.info("Feature Flag: USE_VECTORIZED_F_SCORE_221=ON — vectorized F-Score installed (runner)")
            else:
                logger.info("Feature Flag: USE_VECTORIZED_F_SCORE_221=OFF — using non-vectorized F-Score path")
        except Exception as e:
            logger.warning(f"Vectorized F-Score enablement failed; falling back to non-vectorized path: {e}")
        eng_counters = _install_sql_counters(engine.engine)
        prev_holdings_set: set = set()
        # Constraints and churn control configuration
        try:
            min_hold_months = int(backtest_cfg.get('rebalance', {}).get('min_holding_months', 0))
        except Exception:
            min_hold_months = 0
        try:
            sector_cap = float(backtest_cfg.get('universe', {}).get('sector_concentration_limit', 0.20))
        except Exception:
            sector_cap = 0.20
        try:
            adv_participation_cap = float(backtest_cfg.get('cost_model', {}).get('max_participation_rate', 0.05))
        except Exception:
            adv_participation_cap = 0.05
        try:
            assumed_notional_vnd = float(backtest_cfg.get('portfolio', {}).get('assumed_notional_vnd', None))
        except Exception:
            assumed_notional_vnd = None
        # Load sector map once for the run
        try:
            sector_map_global = _load_sector_map(db_engine)
        except Exception:
            sector_map_global = {}
        # Track holding ages (in months) to enforce minimum holding period
        holding_ages: Dict[str, int] = {}
        # Enriched holdings for instrumentation panels
        enriched_rows: List[Dict] = []
        for reb_date in monthly_dates:
            # Snapshot SQL counters at loop start
            db_q0, db_r0 = db_counters["queries"], db_counters["rows"]
            eng_q0, eng_r0 = eng_counters["queries"], eng_counters["rows"]

            # Universe selection from DB
            from production.universe.constructors import get_liquid_universe_and_counts
            t_u0 = time.perf_counter()
            universe, u_counts = get_liquid_universe_and_counts(reb_date, db_engine)
            elapsed_universe_ms = (time.perf_counter() - t_u0) * 1000.0
            # Seed universe diagnostics row (selected_count populated after holdings selection below)
            universe_diag_rows.append({
                'date': pd.Timestamp(reb_date),
                **u_counts,
                'selected_count': None,
            })
            if not universe:
                logger.warning(f"Empty universe on {reb_date.date()} - skipping")
                diag_rows.append({
                    "date": pd.Timestamp(reb_date),
                    "universe_size": 0,
                    "had_scores": False,
                    "holdings_size": 0,
                    "elapsed_ms_universe": round(float(elapsed_universe_ms), 3),
                    "sql_queries": (db_counters["queries"] + eng_counters["queries"]) - (db_q0 + eng_q0),
                    "sql_rows": (db_counters["rows"] + eng_counters["rows"]) - (db_r0 + eng_r0),
                })
                empty_universe_count += 1
                timings_rows.append({
                    "date": pd.Timestamp(reb_date),
                    "elapsed_ms_universe": round(float(elapsed_universe_ms), 3),
                    "elapsed_ms_factors": None,
                    "elapsed_ms_quality": None,
                    "elapsed_ms_value": None,
                    "elapsed_ms_momentum": None,
                    "elapsed_ms_lowvol": None,
                    "elapsed_ms_fscore": None,
                    "elapsed_ms_fcf": None,
                    "sql_queries": (db_counters["queries"] + eng_counters["queries"]) - (db_q0 + eng_q0),
                    "sql_rows": (db_counters["rows"] + eng_counters["rows"]) - (db_r0 + eng_r0),
                })
                continue

            # Engine factor computation and composite with timing telemetry
            t0 = time.perf_counter()
            scores = engine.calculate_qvm_composite_fixed(reb_date, universe)
            elapsed_ms_vec = (time.perf_counter() - t0) * 1000.0
            _et = getattr(engine, '_last_timings', {}) or {}

            # Optional micro-benchmark: compare non-vectorized path on first date only
            elapsed_ms_nonvec = None
            if benchmark_both and not did_benchmark_both:
                try:
                    orig_flag = engine.use_vectorized_fscore_221
                    engine.use_vectorized_fscore_221 = False
                    t1 = time.perf_counter()
                    _ = engine.calculate_qvm_composite_fixed(reb_date, universe)
                    elapsed_ms_nonvec = (time.perf_counter() - t1) * 1000.0
                    engine.use_vectorized_fscore_221 = orig_flag
                    did_benchmark_both = True
                except Exception as _e:
                    try:
                        engine.use_vectorized_fscore_221 = orig_flag
                    except Exception:
                        pass
            if not scores:
                logger.warning(f"No scores on {reb_date.date()} - skipping")
                diag_rows.append({
                    "date": pd.Timestamp(reb_date),
                    "universe_size": len(universe) if universe else 0,
                    "had_scores": False,
                    "holdings_size": 0,
                    "elapsed_ms_universe": round(float(elapsed_universe_ms), 3),
                    "elapsed_ms_factors": round(float(elapsed_ms_vec), 3),
                    "sql_queries": (db_counters["queries"] + eng_counters["queries"]) - (db_q0 + eng_q0),
                    "sql_rows": (db_counters["rows"] + eng_counters["rows"]) - (db_r0 + eng_r0),
                })
                empty_scores_count += 1
                timings_rows.append({
                    "date": pd.Timestamp(reb_date),
                    "elapsed_ms_universe": round(float(elapsed_universe_ms), 3),
                    "elapsed_ms_factors": _et.get("elapsed_ms_factors", round(float(elapsed_ms_vec), 3)),
                    "elapsed_ms_quality": _et.get("elapsed_ms_quality"),
                    "elapsed_ms_value": _et.get("elapsed_ms_value"),
                    "elapsed_ms_momentum": _et.get("elapsed_ms_momentum"),
                    "elapsed_ms_lowvol": _et.get("elapsed_ms_lowvol"),
                    "elapsed_ms_fscore": _et.get("elapsed_ms_fscore"),
                    "elapsed_ms_fcf": _et.get("elapsed_ms_fcf"),
                    "sql_queries": (db_counters["queries"] + eng_counters["queries"]) - (db_q0 + eng_q0),
                    "sql_rows": (db_counters["rows"] + eng_counters["rows"]) - (db_r0 + eng_r0),
                })
                continue

            # Factor coverage before selection
            coverage = _compute_factor_coverage(scores)

            # Apply constrained selection: sector caps, ADV capacity, and churn control
            try:
                adv_map = _load_adv_20(db_engine, reb_date, list(scores.keys())) if assumed_notional_vnd else {}
            except Exception:
                adv_map = {}
            if selection_enable_constraints:
                holdings = _select_constrained_holdings(
                    scores=scores,
                    n=portfolio_size,
                    sector_map=sector_map_global,
                    sector_cap=sector_cap,
                    prev_holdings=prev_holdings_set,
                    min_hold_months=min_hold_months,
                    ages=holding_ages,
                    adv_map_vnd=adv_map,
                    adv_participation_cap=adv_participation_cap,
                    portfolio_notional_vnd=assumed_notional_vnd,
                )
            else:
                holdings = _select_top_n_holdings(scores, portfolio_size)[:portfolio_size]
            if not holdings:
                logger.warning(f"No holdings on {reb_date.date()} - skipping")
                diag_rows.append({
                    "date": pd.Timestamp(reb_date),
                    "universe_size": len(universe) if universe else 0,
                    "had_scores": True,
                    "holdings_size": 0,
                    "elapsed_ms_universe": round(float(elapsed_universe_ms), 3),
                    "elapsed_ms_factors": round(float(elapsed_ms_vec), 3),
                    "sql_queries": (db_counters["queries"] + eng_counters["queries"]) - (db_q0 + eng_q0),
                    "sql_rows": (db_counters["rows"] + eng_counters["rows"]) - (db_r0 + eng_r0),
                })
                empty_holdings_count += 1
                # Update corresponding universe diagnostics selected_count to 0
                try:
                    universe_diag_rows[-1]['selected_count'] = 0
                except Exception:
                    pass
                timings_rows.append({
                    "date": pd.Timestamp(reb_date),
                    "elapsed_ms_universe": round(float(elapsed_universe_ms), 3),
                    "elapsed_ms_factors": _et.get("elapsed_ms_factors", round(float(elapsed_ms_vec), 3)),
                    "elapsed_ms_quality": _et.get("elapsed_ms_quality"),
                    "elapsed_ms_value": _et.get("elapsed_ms_value"),
                    "elapsed_ms_momentum": _et.get("elapsed_ms_momentum"),
                    "elapsed_ms_lowvol": _et.get("elapsed_ms_lowvol"),
                    "elapsed_ms_fscore": _et.get("elapsed_ms_fscore"),
                    "elapsed_ms_fcf": _et.get("elapsed_ms_fcf"),
                    "sql_queries": (db_counters["queries"] + eng_counters["queries"]) - (db_q0 + eng_q0),
                    "sql_rows": (db_counters["rows"] + eng_counters["rows"]) - (db_r0 + eng_r0),
                })
                continue
            all_holdings[reb_date] = holdings
            # Update selected_count for universe diagnostics
            try:
                universe_diag_rows[-1]['selected_count'] = len(holdings)
            except Exception:
                pass
            # Turnover vs previous holdings
            current_set = set(holdings)
            overlap = len(prev_holdings_set.intersection(current_set)) if prev_holdings_set else 0
            turnover = 1.0 - (overlap / float(len(current_set))) if current_set else 0.0
            prev_holdings_set = current_set
            # Update ages map
            new_ages: Dict[str, int] = {}
            for t in current_set:
                new_ages[t] = int(holding_ages.get(t, 0)) + 1
            holding_ages = new_ages
            # Enrich per-name records for instrumentation (scores + sector)
            eq_w = (1.0 / float(len(holdings))) if holdings else 0.0
            for t in holdings:
                s = scores.get(t, {})
                enriched_rows.append({
                    'date': pd.Timestamp(reb_date),
                    'ticker': t,
                    'quality_score': s.get('Quality_Composite'),
                    'value_score': s.get('Value_Composite'),
                    'momentum_score': s.get('Momentum_Composite'),
                    'composite_score': s.get('QVM_Composite'),
                    'sector': sector_map_global.get(t, 'Unknown'),
                    'weight': eq_w,
                })
            diag_rows.append({
                "date": pd.Timestamp(reb_date),
                "universe_size": len(universe) if universe else 0,
                "had_scores": True,
                "holdings_size": len(holdings),
                "elapsed_ms_universe": round(float(elapsed_universe_ms), 3),
                "elapsed_ms_factors": round(float(elapsed_ms_vec), 3),
                "elapsed_ms_factors_nonvec": round(float(elapsed_ms_nonvec), 3) if elapsed_ms_nonvec is not None else None,
                "turnover": round(float(turnover), 4),
                "factor_coverage_rate": round(float(coverage), 4),
                "sql_queries": (db_counters["queries"] + eng_counters["queries"]) - (db_q0 + eng_q0),
                "sql_rows": (db_counters["rows"] + eng_counters["rows"]) - (db_r0 + eng_r0),
            })
            timings_rows.append({
                "date": pd.Timestamp(reb_date),
                "elapsed_ms_universe": round(float(elapsed_universe_ms), 3),
                "elapsed_ms_factors": _et.get("elapsed_ms_factors", round(float(elapsed_ms_vec), 3)),
                "elapsed_ms_quality": _et.get("elapsed_ms_quality"),
                "elapsed_ms_value": _et.get("elapsed_ms_value"),
                "elapsed_ms_momentum": _et.get("elapsed_ms_momentum"),
                "elapsed_ms_lowvol": _et.get("elapsed_ms_lowvol"),
                "elapsed_ms_fscore": _et.get("elapsed_ms_fscore"),
                "elapsed_ms_fcf": _et.get("elapsed_ms_fcf"),
                "sql_queries": (db_counters["queries"] + eng_counters["queries"]) - (db_q0 + eng_q0),
                "sql_rows": (db_counters["rows"] + eng_counters["rows"]) - (db_r0 + eng_r0),
            })
    else:
        logger.info(f"Parallel execution enabled with jobs={jobs} (unsafe mode)")
        try:
            # Warm-up length L can reduce cross-date dependencies per worker
            warmup_months = 3
            try:
                warmup_months = int(backtest_cfg.get('parallel', {}).get('warmup_months', warmup_months))
            except Exception:
                pass
            # Enforce safe process start and configure DB access policy in children
            try:
                from production.utils.parallel import (
                    ensure_spawn_start_method,
                    disable_db_access_in_children,
                    db_access_allowed,
                )
                ensure_spawn_start_method(logger)
                # Respect pre-set DISABLE_DB_IN_CHILDREN if provided; otherwise default to disabled
                disable_db_access_in_children(logger=logger)
                eff = "allowed" if db_access_allowed() else "disabled"
                logger.info("Parallel safety: spawn start method enforced; DB access %s in children (DISABLE_DB_IN_CHILDREN=%s)", eff, os.environ.get('DISABLE_DB_IN_CHILDREN', '0'))
            except Exception as _saf_e:
                logger.warning(f"Parallel safety setup failed: {_saf_e}")
            # Explicit pool launch context
            try:
                logger.info(f"Parallel launch: submitting {len(monthly_dates)} monthly dates with warmup_months={warmup_months}")
                logger.info("Worker inputs: no pre-materialized DataFrames provided; workers expected to use DB unless disabled via DISABLE_DB_IN_CHILDREN=1")
                # One-time guidance on Parquet prematerialization to reduce per-worker I/O
                global _PARALLEL_PREMATERIALIZATION_LOGGED
                if not _PARALLEL_PREMATERIALIZATION_LOGGED:
                    logger.info("Prematerialization hint: consider pre-saving factor inputs to Parquet and passing file paths to workers to minimize DB roundtrips and serialization costs.")
                    _PARALLEL_PREMATERIALIZATION_LOGGED = True
            except Exception:
                pass
            with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as ex:
                futs = [
                    ex.submit(_worker_process, d, strategy_cfg, backtest_cfg, portfolio_size)
                    for d in monthly_dates
                ]
                for fut in concurrent.futures.as_completed(futs):
                    try:
                        results.append(fut.result())
                    except Exception as _e:
                        logger.error(f"Worker failed: {_e}")
        except Exception as _e:
            logger.warning(f"Parallel path failed, falling back to sequential workers: {_e}")
            for d in monthly_dates:
                res = _worker_process(d, strategy_cfg, backtest_cfg, portfolio_size)
                results.append(res)

        # Merge results in date order and build diagnostics
        results.sort(key=lambda r: r["date"]) 
        prev_holdings_set: set = set()
        for res in results:
            d = pd.Timestamp(res["date"])  # ensure Timestamp
            holdings = res.get("holdings") or []
            try:
                u_counts = {
                    'candidates': int(res.get('candidates', 0)),
                    'fail_trading_days': int(res.get('fail_trading_days', 0)),
                    'fail_adtv': int(res.get('fail_adtv', 0)),
                }
            except Exception:
                u_counts = {'candidates': 0, 'fail_trading_days': 0, 'fail_adtv': 0}
            universe_diag_rows.append({
                'date': d,
                **u_counts,
                'selected_count': int(len(holdings)),
            })
            if not holdings:
                if res.get("universe_size", 0) == 0:
                    empty_universe_count += 1
                elif not res.get("had_scores", False):
                    empty_scores_count += 1
                else:
                    empty_holdings_count += 1
            else:
                all_holdings[d] = holdings
            current_set = set(holdings)
            overlap = len(prev_holdings_set.intersection(current_set)) if prev_holdings_set and current_set else 0
            turnover = 1.0 - (overlap / float(len(current_set))) if current_set else 0.0
            prev_holdings_set = current_set if current_set else prev_holdings_set
            diag_rows.append({
                "date": d,
                "universe_size": int(res.get("universe_size", 0)),
                "had_scores": bool(res.get("had_scores", False)),
                "holdings_size": len(holdings),
                "elapsed_ms_universe": res.get("elapsed_ms_universe"),
                "elapsed_ms_factors": res.get("elapsed_ms_factors"),
                "turnover": round(float(turnover), 4),
                "factor_coverage_rate": res.get("factor_coverage_rate"),
                "sql_queries": int(res.get("sql_queries", 0)),
                "sql_rows": int(res.get("sql_rows", 0)),
            })
            timings_rows.append({
                "date": d,
                "elapsed_ms_universe": res.get("elapsed_ms_universe"),
                "elapsed_ms_factors": res.get("elapsed_ms_factors"),
                "elapsed_ms_quality": res.get("elapsed_ms_quality"),
                "elapsed_ms_value": res.get("elapsed_ms_value"),
                "elapsed_ms_momentum": res.get("elapsed_ms_momentum"),
                "elapsed_ms_lowvol": res.get("elapsed_ms_lowvol"),
                "elapsed_ms_fscore": res.get("elapsed_ms_fscore"),
                "elapsed_ms_fcf": res.get("elapsed_ms_fcf"),
                "sql_queries": int(res.get("sql_queries", 0)),
                "sql_rows": int(res.get("sql_rows", 0)),
            })

    # (moved) Save holdings artifact after calendar alignment below
    if not all_holdings:
        raise RuntimeError("No holdings generated in the window")

    # Phase 4: Prices + benchmark
    unique_tickers = sorted({t for lst in all_holdings.values() for t in lst})
    try:
        price_matrix = build_daily_price_matrix(db_engine, unique_tickers, start_date, end_date, logger)
    except Exception as e:
        logger.error(f"Price matrix build failed: {e}")
        raise

    # Align calendar to available prices and prune holdings to calendar dates
    cal = first_trading_day_calendar(price_matrix)
    logger.info(f"Price calendar generated {len(cal)} first-trading-day dates")
    logger.info(f"Holdings available on {len(all_holdings)} dates")
    
    # Keep only rebalance dates that exist in both holdings and the price calendar
    cal = [d for d in cal if d in all_holdings]
    logger.info(f"After alignment: {len(cal)} dates remain")
    
    if not cal:
        # Fallback: use holdings dates directly if price calendar alignment dropped all
        logger.warning("Price calendar alignment failed, using holdings dates directly")
        cal = sorted(all_holdings.keys())
        logger.info(f"Fallback calendar: {len(cal)} dates")
    
    if not cal:
        logger.error("No valid rebalance dates after alignment")
        # Persist diagnostics to ease debugging alignment failures
        pd.DataFrame(diag_rows or [{ 'error': 'alignment_failure' }]).to_csv(artifacts_dir / 'diagnostics.csv', index=False)
        raise RuntimeError("No valid rebalance dates after alignment")
    
    all_holdings = {d: all_holdings[d] for d in cal}

    # Save holdings artifact (calendar-aligned). Prefer enriched export if available.
    if 'enriched_rows' in locals() and enriched_rows:
        holdings_df = pd.DataFrame(enriched_rows)
        holdings_df = holdings_df[holdings_df['date'].isin(cal)] if not holdings_df.empty else holdings_df
        if not holdings_df.empty:
            holdings_df = holdings_df.sort_values(["date", "ticker"], kind="mergesort").reset_index(drop=True)
        holdings_df.to_csv(artifacts_dir / 'monthly_holdings.csv', index=False)
    else:
        h_rows = []
        for d, lst in all_holdings.items():
            # Enforce deterministic ticker ordering within a date as a stable tiebreaker
            for t in sorted(lst):
                h_rows.append({"date": pd.Timestamp(d), "ticker": t})
        holdings_df = pd.DataFrame(h_rows)
        # Align to price calendar dates as in enriched branch to keep outputs consistent
        if not holdings_df.empty:
            if isinstance(cal, (list, tuple, pd.Index, pd.Series)) and len(cal) > 0:
                holdings_df = holdings_df[holdings_df['date'].isin(cal)]
            holdings_df = holdings_df.sort_values(["date", "ticker"], kind="mergesort").reset_index(drop=True)
        holdings_df.to_csv(artifacts_dir / 'monthly_holdings.csv', index=False)

    # Emit per-date portfolio hash file for golden-window comparisons
    try:
        from hashlib import sha256 as _sha256
        lines = []
        for d, group in holdings_df.groupby('date', sort=True):
            tickers = list(group['ticker'].astype(str))
            payload = (str(pd.Timestamp(d).date()) + '|' + ','.join(tickers)).encode('utf-8')
            h = _sha256(payload).hexdigest()
            lines.append(f"{pd.Timestamp(d).date()},{h}")
        (artifacts_dir / 'portfolio_hashes.txt').write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')
    except Exception as _e:
        logger.warning(f"Failed writing per-date portfolio hashes: {_e}")

    # Save diagnostics (including counters and feature flags)
    if diag_rows:
        diag_df = pd.DataFrame(diag_rows)
        diag_df['empty_universe_count'] = empty_universe_count
        diag_df['empty_scores_count'] = empty_scores_count
        diag_df['empty_holdings_count'] = empty_holdings_count
        # Feature flag telemetry
        diag_df['use_vectorized_fscore_221'] = bool(strategy_cfg.get("f_score", {}).get("use_vectorized_fscore_221", True))
        # Add benchmark loader latency to each per-date row
        diag_df['elapsed_ms_benchmark'] = round(float(elapsed_ms_benchmark), 3)
        diag_df.to_csv(artifacts_dir / 'diagnostics.csv', index=False)

    # Compute per-date factor vector checksums by category for divergence localization
    try:
        factor_hash_rows = []
        # Attempt to replay factor computation hashes deterministically by reusing engine and calendars
        # For performance, only compute hashes over selected holdings tickers if full scores are not cached
        engine_hash = QVMEngineV221Flat()
        try:
            use_vec = bool(strategy_cfg.get("f_score", {}).get("use_vectorized_fscore_221", True))
            engine_hash.use_vectorized_fscore_221 = use_vec
            if use_vec:
                install_vectorized_fscore_221(engine_hash)
        except Exception:
            pass
        for d in sorted(all_holdings.keys()):
            try:
                # Prefer hashing over universe-derived scores to avoid selection bias
                from production.universe.constructors import get_liquid_universe
                universe_d = get_liquid_universe(d, db_engine)
                if not universe_d:
                    universe_d = all_holdings.get(d, [])
                scores_d = engine_hash.calculate_qvm_composite_fixed(d, universe_d)
                if not scores_d:
                    continue
                # Build stable vectors ordered by ticker for each factor family
                df_s = pd.DataFrame.from_dict(scores_d, orient='index')
                df_s.index.name = 'ticker'
                df_s = df_s.reset_index().sort_values(['ticker'], kind='mergesort')
                def _h(vals) -> str:
                    import hashlib as _hashlib
                    arr = np.asarray(list(vals), dtype='float64')
                    payload = arr.tobytes()
                    return _hashlib.sha256(payload).hexdigest()
                row = {
                    'date': pd.Timestamp(d),
                    'Q_hash': _h(df_s.get('Quality_Composite', pd.Series(dtype='float64')).fillna(np.nan).values),
                    'V_hash': _h(df_s.get('Value_Composite', pd.Series(dtype='float64')).fillna(np.nan).values),
                    'M_hash': _h(df_s.get('Momentum_Composite', pd.Series(dtype='float64')).fillna(np.nan).values),
                    'QVM_hash': _h(df_s.get('QVM_Composite', pd.Series(dtype='float64')).fillna(np.nan).values),
                }
                factor_hash_rows.append(row)
            except Exception:
                continue
        if factor_hash_rows:
            pd.DataFrame(factor_hash_rows).sort_values('date').to_csv(artifacts_dir / 'factor_hashes.csv', index=False)
    except Exception as _e:
        logger.warning(f"Failed writing factor_hashes.csv: {_e}")
    # Save timings CSV ordered by date
    if timings_rows:
        timings_df = pd.DataFrame(timings_rows)
        timings_df = timings_df.sort_values('date')
        # Ensure required schema columns exist even if None
        for col in [
            'elapsed_ms_universe','elapsed_ms_factors','elapsed_ms_quality','elapsed_ms_value',
            'elapsed_ms_momentum','elapsed_ms_lowvol','elapsed_ms_fscore','elapsed_ms_fcf',
            'sql_queries','sql_rows']:
            if col not in timings_df.columns:
                timings_df[col] = None
        timings_df.to_csv(artifacts_dir / 'timings.csv', index=False)

    # Persist universe diagnostics always if available
    try:
        if universe_diag_rows:
            u_df = pd.DataFrame(universe_diag_rows)
            # Backfill selected_count where not set (e.g., score failure path)
            if 'selected_count' in u_df.columns:
                u_df['selected_count'] = u_df['selected_count'].fillna(0).astype(int)
            u_df = u_df[['date','candidates','fail_trading_days','fail_adtv','selected_count']]
            u_df.to_csv(artifacts_dir / 'universe_diagnostics.csv', index=False)
    except Exception as _e:
        logger.warning(f"Failed writing universe_diagnostics.csv: {_e}")

    # Phase 5: Daily PnL with and without risk overlay
    # Slippage model (basis points) optional
    slippage_bps = float(backtest_cfg.get('slippage_bps', 0.0))
    bt_cfg = BacktestConfig(transaction_cost_bps=tc_bps, portfolio_size=portfolio_size, slippage_bps=slippage_bps)

    # Risk overlay selection (default: legacy drawdown-based)
    overlay_fn = None
    if risk_overlay_mode == 'vol_targeting':
        # Volatility/beta targeting exposure schedule → cash overlay
        try:
            vol_target_ann = float(backtest_cfg.get('risk_overlay', {}).get('volatility_target', 0.11))
        except Exception:
            vol_target_ann = 0.11
        exposure_schedule = _compute_exposure_schedule(
            price_matrix=price_matrix,
            benchmark_returns=benchmark_returns,
            monthly_holdings=all_holdings,
            monthly_calendar=cal,
            vol_target_ann=vol_target_ann,
        )
        def overlay_fn(_bench_prices: pd.Series, current_date: pd.Timestamp) -> float:
            try:
                exp = float(exposure_schedule.reindex([current_date]).iloc[0])
                if not np.isfinite(exp):
                    exp = 1.0
            except Exception:
                exp = 1.0
            # cash = 1 - exposure
            return float(np.clip(1.0 - exp, 0.0, 0.99))
    else:
        # Legacy drawdown-based cash overlay
        dd_rules = {}
        try:
            dd_rules = strategy_cfg.get('risk_management', {}).get('cash_allocation', {}) or {}
        except Exception:
            dd_rules = {}
        def overlay_fn(_bench_prices: pd.Series, current_date: pd.Timestamp) -> float:
            try:
                return float(drawdown_to_cash_allocation(benchmark_prices, current_date, dd_rules))
            except Exception:
                return 0.0

    # No-risk path
    no_risk_returns, no_risk_equity, _ = run_daily_pnl(
        daily_prices=price_matrix,
        monthly_rebalance_dates=cal,
        monthly_holdings=all_holdings,
        backtest_config=bt_cfg,
        risk_overlay_fn=None,
        benchmark_prices=benchmark_prices,
        logger=logger,
    )
    # With-risk path
    with_risk_returns, with_risk_equity, cash_df = run_daily_pnl(
        daily_prices=price_matrix,
        monthly_rebalance_dates=cal,
        monthly_holdings=all_holdings,
        backtest_config=bt_cfg,
        risk_overlay_fn=overlay_fn,
        benchmark_prices=benchmark_prices,
        logger=logger,
    )

    # Phase 6: Tearsheet(s)
    artifacts = {
        "no_risk_returns.csv": no_risk_returns,
        "with_risk_returns.csv": with_risk_returns,
        "with_risk_cash.csv": cash_df["cash_allocation"],
        "benchmark_returns.csv": benchmark_returns,
    }
    for name, series in artifacts.items():
        series.to_csv(artifacts_dir / name, header=True)
    # Optional Parquet outputs
    try:
        if backtest_cfg.get('output', {}).get('export_parquet', False):
            for name, series in artifacts.items():
                series.to_frame(name=name.replace('.csv','')).to_parquet(artifacts_dir / (name.replace('.csv','.parquet')))
    except Exception as e:
        logger.warning(f"Failed writing Parquet artifacts: {e}")

    # Prepare cash allocations for tearsheet consumption
    cash_allocations_df = cash_df.copy()
    cash_allocations_df = cash_allocations_df.reset_index().rename(columns={cash_allocations_df.index.name or 'index': 'date'})
    cash_allocations_df['date'] = pd.to_datetime(cash_allocations_df['date'])
    if 'cash_allocation' in cash_allocations_df.columns:
        cash_allocations_df['cash_percentage'] = cash_allocations_df['cash_allocation'] * 100.0

    # Generate comparison tearsheet (with-risk vs no-risk vs benchmark)
    generate_comparison_tearsheet(
        with_risk_returns,
        no_risk_returns,
        benchmark_returns,
        cash_allocations_df,
        strategy_cfg,
    )

    # Save static tearsheet images to artifacts directory for reproducibility
    try:
        # WITH risk management vs benchmark
        title_with = f"{strategy_cfg['strategy']['name']}: WITH Risk Management vs Benchmark"
        generate_comprehensive_tearsheet(
            with_risk_returns,
            benchmark_returns,
            title_with,
            cash_allocations_df,
        )
        fig = plt.gcf()
        fig.savefig(artifacts_dir / 'tearsheet_with_risk.png', dpi=160, bbox_inches='tight')
        plt.close(fig)

        # WITHOUT risk management vs benchmark (explicit 0% cash for control)
        title_without = f"{strategy_cfg['strategy']['name']}: WITHOUT Risk Management vs Benchmark"
        no_risk_cash_df = pd.DataFrame({
            'date': no_risk_returns.index,
            'cash_percentage': 0.0,
        })
        generate_comprehensive_tearsheet(
            no_risk_returns,
            benchmark_returns,
            title_without,
            no_risk_cash_df,
        )
        fig = plt.gcf()
        fig.savefig(artifacts_dir / 'tearsheet_without_risk.png', dpi=160, bbox_inches='tight')
        plt.close(fig)

        # Comparison grid (with risk, without risk, benchmark, cash)
        create_comparison_plots(
            with_risk_returns,
            no_risk_returns,
            benchmark_returns,
            cash_allocations_df,
            strategy_cfg,
        )
        fig = plt.gcf()
        fig.savefig(artifacts_dir / 'tearsheet_comparison.png', dpi=160, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"Tearsheet images saved under {artifacts_dir}")
    except Exception as e:
        logger.warning(f"Failed to generate or save tearsheet images: {e}")

    # Write run metadata
    try:
        import subprocess
        git_sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(project_root)).decode().strip()
        try:
            git_branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(project_root)).decode().strip()
        except Exception:
            git_branch = "unknown"
    except Exception:
        git_sha = "unknown"
        git_branch = "unknown"
    env_manifest = {
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "git_sha": git_sha,
        "git_branch": git_branch,
    }
    with open(artifacts_dir / 'environment_manifest.json', 'w', encoding='utf-8') as f:
        json.dump(env_manifest, f, indent=2, sort_keys=True)

    # Run info for baseline comparisons
    try:
        run_info = {
            "jobs": int(jobs),
            "force_parallel": bool(getattr(args, 'force_parallel', False)),
            "num_rebalance_dates": int(len(all_holdings)),
            "wall_clock_ms": round(float((_time.perf_counter() - _run_t0) * 1000.0), 3),
            "git_sha": env_manifest.get("git_sha"),
            "git_branch": env_manifest.get("git_branch"),
            "python": env_manifest.get("python"),
            "numpy": env_manifest.get("numpy"),
            "pandas": env_manifest.get("pandas"),
            "blas_threads": {
                "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            },
            "blas_vendor": None,
            "dtype": "float64",
            "cpu_count": int(cpu_count() or 1),
            "platform": sys.platform,
            "data_snapshot": os.environ.get("DATA_SNAPSHOT_ID"),
            "seed": int(os.environ.get("QVM_SEED", "42")),
        }
        # Try to detect BLAS vendor
        try:
            import numpy as _np
            from numpy import __config__ as _npconf
            info = {}
            for key in ("blas_mkl_info","openblas_info","blas_opt_info"):
                try:
                    d = getattr(_npconf, key)
                except Exception:
                    d = None
                if d:
                    info[key] = d
            run_info["blas_vendor"] = sorted(list(info.keys()))[0] if info else None
        except Exception:
            pass
        with open(artifacts_dir / 'run_info.json', 'w', encoding='utf-8') as f:
            json.dump(run_info, f, indent=2, sort_keys=True)
    except Exception:
        pass

    # Integrity manifest
    try:
        import hashlib
        manifest = {}
        for fname in [
            'strategy_config.json','backtest_config.json','monthly_holdings.csv','diagnostics.csv',
            'timings.csv','universe_diagnostics.csv','portfolio_hashes.txt','factor_hashes.csv',
            'no_risk_returns.csv','with_risk_returns.csv','with_risk_cash.csv','benchmark_returns.csv',
            'environment_manifest.json','run_info.json'
        ]:
            fpath = artifacts_dir / fname
            if fpath.exists():
                h = hashlib.sha256()
                with open(fpath, 'rb') as fh:
                    while True:
                        chunk = fh.read(8192)
                        if not chunk:
                            break
                        h.update(chunk)
                manifest[fname] = { 'sha256': h.hexdigest(), 'bytes': fpath.stat().st_size }
        # Integrity manifest (sha256 + bytes)
        with open(artifacts_dir / 'integrity_manifest.json', 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        # Artifact manifest alias for CI harnesses expecting this name
        with open(artifacts_dir / 'artifact_manifest.json', 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
    except Exception as e:
        logger.warning(f"Failed to write integrity manifest: {e}")

    logger.info(f"Artifacts saved under {artifacts_dir}")


if __name__ == "__main__":
    # Avoid passing IPython/Jupyter argv (e.g., --f=...) into argparse when running in a notebook
    if 'ipykernel' in sys.modules:
        _argv_backup = sys.argv[:]
        sys.argv = [sys.argv[0]]
        try:
            main()
        finally:
            sys.argv = _argv_backup
    else:
        main()


