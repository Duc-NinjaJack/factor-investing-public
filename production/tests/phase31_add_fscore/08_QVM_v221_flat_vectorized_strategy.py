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
from typing import Dict, List, Tuple

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
    """Select top-N tickers with stable tie-break on ticker ascending.

    Uses mergesort for stability to guarantee deterministic ordering when
    `QVM_Composite` ties occur.
    """
    df = pd.DataFrame.from_dict(scores, orient="index")
    if df.empty or "QVM_Composite" not in df.columns:
        return []
    df = df.copy()
    df["ticker"] = df.index.astype(str)
    ranked = df.sort_values(by=["QVM_Composite", "ticker"], ascending=[False, True], kind="mergesort")
    return ranked["ticker"].tolist()[:n]


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
    from production.universe.constructors import get_liquid_universe as _get_univ
    from production.engine.qvm_engine_v2_2_1_flat import QVMEngineV221Flat as _Eng
    from production.database.connection import DatabaseManager as _DBM
    from production.engine.qvm_engine_v2_2_1_flat_vectorized import install_vectorized_fscore_221 as _install_vec
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

    # Universe
    t_u0 = _time.perf_counter()
    try:
        from production.universe.constructors import get_liquid_universe_and_counts as _get_univ_counts
        univ, u_counts = _get_univ_counts(reb_date, _db_engine)
    except Exception:
        univ = _get_univ(reb_date, _db_engine)
        u_counts = {'candidates': 0, 'fail_trading_days': 0, 'fail_adtv': 0, 'selected_count': int(len(univ) or 0)}
    elapsed_universe_ms = (_time.perf_counter() - t_u0) * 1000.0
    if not univ:
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
    scores = _engine.calculate_qvm_composite_fixed(reb_date, univ)
    elapsed_factors_ms = (_time.perf_counter() - t_f0) * 1000.0
    _et = getattr(_engine, '_last_timings', {}) or {}
    if not scores:
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

    coverage = _compute_factor_coverage(scores)
    holdings = _select_top_n_holdings(scores, portfolio_size)
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
    portfolio_size = strategy_cfg.get("strategy", {}).get("portfolio", {}).get("portfolio_size", 20)
    tc_bps = backtest_cfg.get("transaction_cost_bps", 10.0)
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

            holdings = _select_top_n_holdings(scores, portfolio_size)
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

    # Save holdings artifact (calendar-aligned) - deterministically sorted
    h_rows = []
    for d, lst in all_holdings.items():
        # Enforce deterministic ticker ordering within a date as a stable tiebreaker
        for t in sorted(lst):
            h_rows.append({"date": pd.Timestamp(d), "ticker": t})
    holdings_df = pd.DataFrame(h_rows)
    if not holdings_df.empty:
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

    def overlay_fn(bench_prices: pd.Series, current_date: pd.Timestamp) -> float:
        risk_cfg = strategy_cfg.get("risk_management", {})
        method = risk_cfg.get("method", "drawdown_based")
        rules = risk_cfg.get("cash_allocation", {})
        if method == "ewma_drawdown":
            from production.risk.overlay import ewma_drawdown_cash_allocation
            return ewma_drawdown_cash_allocation(bench_prices, current_date)
        return drawdown_to_cash_allocation(bench_prices, current_date, rules)

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

        # WITHOUT risk management vs benchmark
        title_without = f"{strategy_cfg['strategy']['name']}: WITHOUT Risk Management vs Benchmark"
        generate_comprehensive_tearsheet(
            no_risk_returns,
            benchmark_returns,
            title_without,
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
    except Exception:
        git_sha = "unknown"
    env_manifest = {
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "git_sha": git_sha,
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
        }
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
            'timings.csv','universe_diagnostics.csv','portfolio_hashes.txt',
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
        with open(artifacts_dir / 'integrity_manifest.json', 'w', encoding='utf-8') as f:
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


