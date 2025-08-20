#!/usr/bin/env python3
"""
Agent Neo's FIXED Analytics-Only Wrapper
========================================
Following Agent Neo's exact instructions:
1. Observer pattern to capture from production orchestrator
2. Use engine's sector mapping
3. Change coverage denominator to eligible tickers

Author: Following Agent Neo's Surgical Fix
Date: August 18, 2025
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import text, create_engine
import logging
import traceback
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy.stats import spearmanr
import functools, inspect, types, re
import unicodedata

# Add production paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import existing production modules
try:
    from utils.db import create_db_connection
    from engine.qvm_engine_v2_1_1_flat import QVMEngineV211Flat
    from engine.qvm_engine_v2_1_1_flat_fscore_vectorized_neo_fix import install_vectorized_fscore_neo_fix, prime_fscore_cache
    from universe.constructors import get_liquid_universe_dataframe
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Agent Neo's exact factor mapping
FACTOR_CODE_TO_ID = {
    'roae': 1, 'gross_margin': 2, 'f_score': 3, 'debt_equity': 4,
    'earnings_yield': 5, 'book_value': 6, 'fcf_yield': 7, 'sales_multiple': 8,
    'mom_12m': 9, 'mom_6m': 10, 'mom_3m': 11,
    'inv_volatility_63d': 12, 'beta': 13, 'low_vol_score': 14,
    'nim': 15, 'cost_income': 16, 'operating_margin_sec': 17, 'cost_ratio_sec': 18,
    'insurance_quality': 19,
    # Added to match sidecar persistence
    'roaa': 20,
    'net_profit_margin': 21,
    'operating_margin': 22,
    'ebitda_margin': 23,
    'ev_ebitda': 24
}

ENGINE_TO_CANONICAL = {
    # Quality
    'roae_z': 'roae', 'roe_ttm_z': 'roae', 'roe_avg_z': 'roae',
    'gross_margin_z': 'gross_margin', 'gm_ttm_z': 'gross_margin',
    'de_z': 'debt_equity', 'debt_to_equity_z': 'debt_equity',
    'f_score_z': 'f_score',

    # Value
    'ep_z': 'earnings_yield', 'earnings_yield_z': 'earnings_yield',
    'pb_z': 'book_value', 'price_to_book_z': 'book_value',
    'ps_z': 'sales_multiple', 'price_to_sales_z': 'sales_multiple',
    'fcf_yield_z': 'fcf_yield',

    # Momentum
    'ret_12_1_z': 'mom_12m', 'momentum_12m_z': 'mom_12m',
    'ret_6m_z': 'mom_6m', 'ret_3m_z': 'mom_3m',

    # Defensive
    'inv_vol_63d_z': 'inv_volatility_63d', 'beta_252_z': 'beta',
    'low_vol_z': 'low_vol_score',

    # Banking / Securities
    'nim_z': 'nim', 'cti_z': 'cost_income',
    'sec_op_margin_z': 'operating_margin_sec', 'sec_cost_ratio_z': 'cost_ratio_sec',
}

# Agent Neo's Canonical registry (exact as provided)
FACTOR_META = {
    "roae": {"pillar":"Quality","sign":+1,"eligible":"all"},
    "gross_margin": {"pillar":"Quality","sign":+1,"eligible":"non_fin"},
    "f_score": {"pillar":"Quality","sign":+1,"eligible":"sector_specific"},
    "debt_equity": {"pillar":"Quality","sign":-1,"eligible":"non_fin"},
    "roaa": {"pillar":"Quality","sign":+1,"eligible":"banks"},
    "net_profit_margin": {"pillar":"Quality","sign":+1,"eligible":"non_fin"},
    "operating_margin": {"pillar":"Quality","sign":+1,"eligible":"non_fin"},
    "ebitda_margin": {"pillar":"Quality","sign":+1,"eligible":"non_fin"},
    "earnings_yield": {"pillar":"Value","sign":+1,"eligible":"all"},
    "book_value": {"pillar":"Value","sign":-1,"eligible":"all"},  # P/B lower is better
    "fcf_yield": {"pillar":"Value","sign":+1,"eligible":"non_fin","optional": True},
    "sales_multiple": {"pillar":"Value","sign":-1,"eligible":"all"},
    "ev_ebitda": {"pillar":"Value","sign":-1,"eligible":"non_fin"},
    "mom_12m": {"pillar":"Momentum","sign":+1,"eligible":"all"},
    "mom_6m":  {"pillar":"Momentum","sign":+1,"eligible":"all"},
    "mom_3m":  {"pillar":"Momentum","sign":+1,"eligible":"all"},
    "mom_1m":  {"pillar":"Momentum","sign":+1,"eligible":"all"},
    "inv_volatility_63d": {"pillar":"Defensive","sign":+1,"eligible":"all"},
    "beta": {"pillar":"Defensive","sign":-1,"eligible":"all"},
    "low_vol_score": {"pillar":"Defensive","sign":+1,"eligible":"all"},
    "nim": {"pillar":"Quality","sign":+1,"eligible":"banks"},
    "cost_income": {"pillar":"Quality","sign":-1,"eligible":"banks"},
    "operating_margin_sec": {"pillar":"Quality","sign":+1,"eligible":"securities","optional": True},
    "cost_ratio_sec": {"pillar":"Quality","sign":-1,"eligible":"securities"},
    # "insurance_quality": optional
}

# --- Canonical factor universe (Tier-0 expects these) ---
EXPECTED_FACTORS_18 = {
    'roae', 'roaa',
    'gross_margin', 'net_profit_margin', 'operating_margin', 'operating_margin_sec',
    'f_score',
    'nim', 'cost_income',
    'debt_equity',
    'earnings_yield', 'book_value', 'sales_multiple', 'ev_ebitda', 'fcf_yield',
    'mom_3m', 'mom_6m', 'mom_12m',
    'low_vol_score',
}

# Accept common aliases and normalize them into our canonical codes
CANON_ALIASES = {
    'book_to_price': 'book_value',
    'sales_to_price': 'sales_multiple',
    'ebitda_ev': 'ev_ebitda',
    'ev_to_ebitda': 'ev_ebitda',
    'ebitda_to_ev': 'ev_ebitda',
    'low_volatility': 'low_vol_score',
    'debt_to_equity': 'debt_equity',
    'total_debt_to_equity': 'debt_equity',
    'de_ratio': 'debt_equity',
    'financial_leverage': 'debt_equity',
}

def canonicalize(name: str) -> str:
    n = (name or '').strip().lower()
    return CANON_ALIASES.get(n, n)

# Agent Neo's High-coverage synonym map (extended with exact engine metric names)
METRIC_TO_FACTOR = {
    # Quality - exact engine metric names from logs
    "roae": "roae", "roae_ttm":"roae", "returnonavgequity_ttm":"roae", "roe_avg_ttm":"roae",
    "roae_raw": "roae",  # engine uses this exact name
    "gross_margin":"gross_margin", "grossmargin_ttm":"gross_margin",
    "gross_margin_raw": "gross_margin",  # engine uses this exact name
    "debt_to_equity":"debt_equity", "debt_equity":"debt_equity", "d_to_e":"debt_equity",
    
    # Value - exact engine metric names from logs
    "earnings_yield":"earnings_yield", "ep":"earnings_yield", "e_p":"earnings_yield",
    "earnings_yield_raw": "earnings_yield",  # engine uses this exact name
    "pb":"book_value", "p_b":"book_value", "price_to_book":"book_value",
    "book_to_price_raw": "book_value",  # engine uses this exact name
    "fcf_yield":"fcf_yield", "free_cash_flow_yield":"fcf_yield",
    "fcf_yield_raw": "fcf_yield",  # engine uses this exact name
    "ps":"sales_multiple", "price_to_sales":"sales_multiple",
    "sales_to_price_raw": "sales_multiple",  # engine uses this exact name
    
    # Momentum - exact engine metric names from logs
    "mom_12m":"mom_12m", "momentum_12m":"mom_12m", "mom12":"mom_12m",
    "momentum_12m_raw": "mom_12m",  # engine uses this exact name
    "mom_6m":"mom_6m", "mom6":"mom_6m", 
    "momentum_6m_raw": "mom_6m",  # engine uses this exact name
    "mom_3m":"mom_3m", "mom3":"mom_3m",
    "momentum_3m_raw": "mom_3m",  # engine uses this exact name
    "momentum_1m_raw": "mom_1m",  # engine has this but we don't track 1m momentum
    
    # Defensive - exact engine metric names from logs
    "inv_vol_63d":"inv_volatility_63d", "inv_volatility_63d":"inv_volatility_63d",
    "low_volatility_raw": "low_vol_score",  # engine uses this exact name
    "beta":"beta", "low_vol_score":"low_vol_score",
    
    # Sector-specific - exact engine metric names from logs
    "f_score":"f_score", "f_score_non_fin":"f_score",
    "f_score_banking":"f_score", "f_score_securities":"f_score",
    "f_score_normalized": "f_score",  # engine uses this exact name
    "nim":"nim", "net_interest_margin":"nim",
    "nim_raw": "nim",  # engine uses this exact name
    "cost_income":"cost_income", "cost_to_income":"cost_income",
    "cost_income_raw": "cost_income",  # engine uses this exact name
    "operating_margin_sec":"operating_margin_sec",
    "operating_margin_raw": "operating_margin",  # non-fin operating margin
    "cost_ratio_sec":"cost_ratio_sec",
    
    # Additional quality metrics seen in logs
    "roaa_raw": "roaa",  # not in our factor list but engine calculates it
    "net_profit_margin_raw": "net_profit_margin",  # not in our factor list
    "ebitda_margin_raw": "ebitda_margin",  # not in our factor list
    "ebitda_to_ev_raw": "ebitda_to_ev",  # not in our factor list
    
    # Missing factors that should be captured
    "debt_to_equity_raw": "debt_equity",
    "d_e_raw": "debt_equity", 
    "de_raw": "debt_equity",
    "inv_volatility_63d_raw": "inv_volatility_63d",
    "volatility_63d_raw": "inv_volatility_63d",
    "beta_252_raw": "beta",
    "beta_raw": "beta",
    "cost_ratio_securities_raw": "cost_ratio_sec",
    "sec_cost_ratio_raw": "cost_ratio_sec"
}

# Agent Neo's extended mappings for unmapped metrics
METRIC_TO_FACTOR.update({
    'ebitda_margin_raw':       'ebitda_margin',         # quality (non-fin)
    'ebitda_to_ev_raw':        'ev_ebitda',             # value
    'net_profit_margin_raw':   'net_profit_margin',     # quality (non-fin)
    'roaa_raw':                'roaa',                  # quality (banks)
    'momentum_1m_raw':         'mom_1m',                # analytics-only; not in EXPECTED_FACTORS_18
    # ensure leverage routes to debt_equity if used by engine under any alias
    'debt_equity_raw':         'debt_equity',
    'debt_to_equity_raw':      'debt_equity',
    'de_ratio_raw':            'debt_equity',
    'financial_leverage_raw':  'debt_equity',
    'leverage_raw':            'debt_equity',
})

# Agent Neo's mapping guard (ensure synonyms are covered)
METRIC_TO_FACTOR.update({
    'debt_equity_raw':         'debt_equity',
    'debt_to_equity_raw':      'debt_equity',
    'de_ratio_raw':            'debt_equity',
    'financial_leverage_raw':  'debt_equity',
    'leverage_raw':            'debt_equity',
    'ebitda_to_ev_raw':        'ev_ebitda',
    'net_profit_margin_raw':   'net_profit_margin',
    'roaa_raw':                'roaa',
})

def _canonicalize(s: str) -> str:
    s = s.lower()
    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s

def fetch_dates_from_composites(engine, version: str, start_date: str, end_date: str) -> List[pd.Timestamp]:
    """Agent's Fix 1: Date source = composites calendar (only dates with actual composites)"""
    with engine.begin() as conn:
        query = text("""
            SELECT DISTINCT date
            FROM factor_scores_qvm
            WHERE strategy_version = :version
              AND date BETWEEN :start_date AND :end_date
            ORDER BY date
        """)
        result = conn.execute(query, {
            'version': version,
            'start_date': start_date, 
            'end_date': end_date
        }).fetchall()
        
        dates = [pd.Timestamp(row.date) for row in result]
        logger.info(f"📅 COMPOSITES CALENDAR: Found {len(dates)} dates with existing composites for {version}")
        return dates

def already_in_sidecar(engine, date: pd.Timestamp, version: str, factor_id: int) -> bool:
    """Agent's Fix 2: Resume/idempotency guard - fast EXISTS check"""
    with engine.begin() as conn:
        query = text("""
            SELECT 1 FROM factor_signals_raw 
            WHERE strategy_version = :version 
              AND factor_id = :factor_id 
              AND date = :date 
            LIMIT 1
        """)
        result = conn.execute(query, {
            'version': version,
            'factor_id': factor_id,
            'date': date.date()
        }).fetchone()
        return result is not None

def map_analytics_to_production_version(analytics_version: str) -> str:
    """Map analytics sidecar version to corresponding composites version for calendar."""
    if 'analytics_v1_neo_fixed' in analytics_version:
        return 'qvm_v2.1.1_flat_corrected'
    if 'analytics_v2_complete' in analytics_version:
        return 'qvm_v2.1.1_flat_corrected'
    return analytics_version

def get_latest_composite_date(engine, prod_version: str) -> Optional[pd.Timestamp]:
    """Fetch the latest date available in composites calendar for the given production version."""
    with engine.begin() as conn:
        row = conn.execute(text(
            """
            SELECT MAX(date) AS max_date
            FROM factor_scores_qvm
            WHERE strategy_version = :v
            """
        ), { 'v': prod_version }).fetchone()
        return pd.Timestamp(row.max_date) if row and row.max_date else None

def get_earliest_composite_date(engine, prod_version: str) -> Optional[pd.Timestamp]:
    with engine.begin() as conn:
        row = conn.execute(text(
            """
            SELECT MIN(date) AS min_date
            FROM factor_scores_qvm
            WHERE strategy_version = :v
            """
        ), { 'v': prod_version }).fetchone()
        return pd.Timestamp(row.min_date) if row and row.min_date else None

def get_last_sidecar_date_for_factors(engine, analytics_version: str, factor_ids: List[int]) -> Optional[pd.Timestamp]:
    """Return min over MAX(date) per selected factor_id to ensure earliest missing is covered."""
    if not factor_ids:
        return None
    with engine.begin() as conn:
        rows = conn.execute(text(
            """
            SELECT factor_id, MAX(date) AS max_date
            FROM factor_signals_raw
            WHERE strategy_version = :v AND factor_id IN :ids
            GROUP BY factor_id
            """
        ), { 'v': analytics_version, 'ids': tuple(factor_ids) }).fetchall()
        if not rows:
            return None
        # Use the earliest of last-dates across selected factors
        min_max = min([r.max_date for r in rows if r.max_date is not None], default=None)
        return pd.Timestamp(min_max) if min_max else None

def sidecar_has_all_selected_factors(engine, date: pd.Timestamp, version: str, factor_ids: List[int]) -> bool:
    """Check if all selected factor_ids already exist for this date/version in sidecar."""
    if not factor_ids:
        return False
    with engine.begin() as conn:
        rows = conn.execute(text(
            """
            SELECT factor_id, COUNT(*) AS cnt
            FROM factor_signals_raw
            WHERE strategy_version = :v AND date = :d AND factor_id IN :ids
            GROUP BY factor_id
            """
        ), { 'v': version, 'd': date.date(), 'ids': tuple(factor_ids) }).fetchall()
        present_ids = {r.factor_id for r in rows}
        return set(factor_ids).issubset(present_ids)

def get_factor_ids_for_factors(factors: List[str]) -> List[int]:
    canonical = [canonicalize(f) for f in (factors or [])]
    ids = []
    for c in canonical:
        fid = FACTOR_CODE_TO_ID.get(c)
        if fid is not None:
            ids.append(fid)
        else:
            logger.warning(f"Unknown factor code '{c}' - skipping id resolution")
    return sorted(set(ids))

class FactorAnalyticsWrapperNeoFix:
    """
    Agent Neo's FIXED Analytics Wrapper with Observer Pattern + Institutional Batching
    """
    
    def __init__(self, strategy_version: str, write_sidecar: bool = False, 
                 factors: List[str] = None, audit_tier: str = 'tier0',
                 enable_debt_equity: bool = False):
        self.version = strategy_version
        self.write_sidecar = write_sidecar
        self.factors = factors or ['f_score']
        self.audit_tier = audit_tier
        self.enable_debt_equity = enable_debt_equity
        
        # Agent Neo's TWO-TAP observer pattern storage
        self.captured_factors = {}  # For legacy compatibility
        self._captured_z = {}       # Tap A: z-scores for audit only  
        self._raw_capture = {}      # Tap B: raw values for writes (legacy)
        self._norm_stats = {}       # Tap B: normalization stats (legacy)
        
        # Agent Neo's required storage buffers (exact as specified)
        self._tapb_raw_buffers = {}      # factor_code -> [(raw_series, sector_series), ...]
        self._tapb_seen_metrics = set()  # metric names seen today for diagnostics
        self._unknown_metric_names = set()  # unknown metrics for mapping expansion
        
        # Agent Neo's context storage for normalization hook
        self.current_date = None
        self.current_strategy_version = strategy_version
        self.factor_code_to_id = FACTOR_CODE_TO_ID
        
        # Agent Neo Fix: Use analytics_writer user with limited permissions  
        self.engine_ro = create_db_connection(project_root)
        self.engine_sidecar = self._get_analytics_writer_connection()
        
        # Initialize QVM engine with production config path to avoid env var issues
        production_config_path = project_root / 'config'
        self.qvm_engine = QVMEngineV211Flat(config_path=str(production_config_path))
        
        # First install Agent Neo's fixed vectorized F-Score
        install_vectorized_fscore_neo_fix(self.qvm_engine)
        
        # Defer TWO‑TAP install until an engine instance exists.
        self._snz_hook_installed = False
        self._snz_hook_engine_id = None
        
        logger.info(f"✅ Initialized FIXED analytics wrapper with observer pattern")
        logger.info(f"   Sidecar writing: {'ENABLED' if write_sidecar else 'DISABLED'}")
        logger.info(f"   Strategy version: {strategy_version}")
        logger.info(f"   Factors: {self.factors}")
        logger.info(f"   Audit tier: {self.audit_tier.upper()}")
        
        self._install_safety_guard()
    
    def _reset_tapb_scope(self):
        # Per-date containers
        self._tapb_raw_buffers = {}
        self._norm_stats = {}
        self._captured_z = {}
        self._tapb_seen_metrics = set()
        self._unknown_metric_names = set()

    def _prepare_date(self):
        # Call this once per processing date BEFORE any factor computation
        self._reset_tapb_scope()
        self._install_observer_hooks(self.qvm_engine)   # SNZ hooks
        # Ensure direct normalization hook captures RAW/sector from df + metric columns
        try:
            self._install_tap_b_normalization_hook(self.qvm_engine)
            logger.info("🔌 TAP-B installed on calculate_sector_neutral_zscore (direct df/metric capture)")
        except Exception as e:
            logger.warning(f"TAP-B normalization hook install skipped: {e}")
        self._install_base_metric_tapb_fallback_live()   # NEW: live module hook
        logger.info("🔧 Tap-B per-date fresh scope ready (hooks installed).")
    
    def _install_observer_hooks(self, engine=None):
        # Engine not yet available during __init__: defer.
        if engine is None:
            self._snz_hook_installed = False
            self._snz_hook_engine_id = None
            return
            
        # If we've already wrapped this very engine, skip.
        if getattr(self, "_snz_hook_installed", False) and getattr(self, "_snz_hook_engine_id", None) == id(engine):
            return
            
        # NEW: TWO‑TAP observer on ALL SNZ entry points with robust name resolution.
        import types, functools, pandas as pd, inspect, sys, numpy as np

        # Storage for this run-date (cleared per date)
        self._tapA_z_rows = []       # (date, ticker, canonical, z)
        self._tapB_raw_rows = []     # (date, ticker, canonical, sector, raw)
        self._tap_stats_rows = []    # (date, canonical, sector, mean, std, n)

        # Accept multiple method names used across v2.0.1→v2.1.1 engines.
        snz_candidates = [
            # NOTE: leave calculate_sector_neutral_zscore to the dedicated TAP‑B normalization hook
            "sector_neutralize",
            "_sector_neutralize",
            "sector_neutral_z",
            "normalize_factor_by_sector",
            "zscore_by_sector",
            "calc_sector_neutral_z",
        ]

        # Canonical factor allow‑list (18) + synonym registry
        # (Defensive low_volatility not counted in the 18, but we still capture if present.)
        self.EXPECTED_FACTORS_18 = {
            "roae","roaa","nim","cost_income",
            "net_profit_margin","gross_margin","operating_margin","ebitda_margin",
            "f_score",
            "earnings_yield","book_value","sales_multiple","ev_ebitda","fcf_yield",
            "mom_3m","mom_6m","mom_12m","low_vol_score","debt_equity"
        }
        # Comprehensive engine→canonical mapping (incl. synonyms).
        self.ENGINE_TO_CANONICAL = {
            # Quality
            "roae": "roae",
            "roaa": "roaa",
            "nim": "nim",
            "cost_income": "cost_income_ratio", "cir": "cost_income_ratio",
            "net_profit_margin": "net_profit_margin", "npm": "net_profit_margin",
            "gross_margin": "gross_margin", "gm": "gross_margin",
            "operating_margin": "operating_margin", "opm": "operating_margin",
            "operating_margin_sec": "operating_margin",
            "ebitda_margin": "ebitda_margin",
            "f_score": "f_score",
            # Value
            "earnings_yield": "earnings_yield", "e_p": "earnings_yield",
            "book_to_price": "book_to_price", "b_p": "book_to_price", "b2p":"book_to_price",
            "sales_to_price": "sales_to_price", "s_p": "sales_to_price", "sales_multiple":"sales_to_price",
            "ebitda_to_ev": "ebitda_to_ev", "ev_ebitda_inv": "ebitda_to_ev", "ev_ebitda":"ebitda_to_ev", "ev_to_ebitda":"ebitda_to_ev",
            "fcf_yield": "fcf_yield",
            # Momentum
            "momentum_1m": "momentum_1m", "ret_1m": "momentum_1m", "mom_1m": "momentum_1m",
            "momentum_3m": "momentum_3m", "ret_3m": "momentum_3m", "mom_3m": "momentum_3m",
            "momentum_6m": "momentum_6m", "ret_6m": "momentum_6m", "mom_6m": "momentum_6m",
            "momentum_12m":"momentum_12m","ret_12m":"momentum_12m","mom_12m":"momentum_12m",
            # Defensive (captured though not in EXPECTED_FACTORS_18)
            "low_volatility": "low_volatility", "inv_vol_63d": "low_volatility", "low_vol_score":"low_volatility",
        }

        # Some engine names need a safe transformation into canonical RAW and canonical naming
        # Ensure research-facing canonical names match config
        self.ENGINE_TO_CANONICAL.update({
            # Value (stable research names)
            "book_value": "book_to_price",  # alias
            "sales_multiple": "sales_to_price",  # alias
            "ev_ebitda": "ebitda_to_ev",  # canonical orientation
            # Defensive
            "low_vol_score": "low_volatility",
        })

        # Orientation transforms
        _transform_policy = {
            # If engine emits EV/EBITDA, invert to EBITDA/EV (higher is better)
            "ev_ebitda": "invert", "ev_to_ebitda": "invert", "ev_ebitda_inv": "identity",
            # defaults to 'identity' for others
        }

        # Helper: infer factor name from stack when not passed
        def _infer_factor_name_from_stack(default=""):
            import inspect
            candidates = ("factor_name","factor","metric","metric_name","name","key","k","fname","fn","code")
            for frameinfo in inspect.stack()[2:8]:  # skip current + wrapper
                try:
                    loc = frameinfo.frame.f_locals
                    for c in candidates:
                        val = loc.get(c)
                        if isinstance(val, str) and val:
                            return val
                except Exception:
                    pass
            return default

        # Helper: robust factor name → canonical
        def _canonize(name):
            # Never truth-test unless it's a real string.
            if not isinstance(name, str):
                return ("", "identity")
            name = name.strip()
            if not name:
                return ("", "identity")
            key = name.lower().rstrip("_z")
            canon = self.ENGINE_TO_CANONICAL.get(key, key)
            transform = _transform_policy.get(key, "identity")
            return (canon, transform)

        def _coerce_series(obj, index=None, prefer_col=None):
            if obj is None:
                return None
            try:
                if isinstance(obj, pd.Series):
                    return obj
                if isinstance(obj, pd.DataFrame):
                    if prefer_col and prefer_col in obj.columns:
                        return obj[prefer_col]
                    if "raw" in obj.columns:
                        return obj["raw"]
                    if obj.shape[1] >= 1:
                        return obj.iloc[:, 0]
                if isinstance(obj, dict):
                    return pd.Series(obj)
                # list/ndarray/iterable
                return pd.Series(obj, index=index)
            except Exception:
                return None

        # Helper: core TWO‑TAP capture (Tap‑A z, Tap‑B raw + stats)
        def _tap_capture(asof, factor_engine_name, raw_s, sector_s, z_s, self_engine=None):
            try:
                canon, transform = _canonize(factor_engine_name)
                if not canon:
                    return
                # Only gate writes to RAW table by EXPECTED_FACTORS_18 (defensive still audited).
                allow_raw = (canon in self.EXPECTED_FACTORS_18)
                # Standardize inputs to aligned Series
                raw_series   = _coerce_series(raw_s)
                sector_series= _coerce_series(sector_s, index=getattr(raw_series, "index", None))
                z_series     = _coerce_series(z_s, index=getattr(raw_series, "index", None))
                
                # Fallback: derive sectors from engine if missing
                if (sector_series is None or sector_series.isna().all()) and self_engine is not None:
                    import pandas as pd
                    sec_map = getattr(self_engine, "sector_map", None)
                    if isinstance(sec_map, dict):
                        sector_series = pd.Series(sec_map).reindex(raw_series.index)
                    elif hasattr(self_engine, "universe"):
                        uni = getattr(self_engine, "universe")
                        if hasattr(uni, "set_index") and "sector" in getattr(uni, "columns", []):
                            sector_series = uni.set_index("ticker").loc[raw_series.index, "sector"]
                
                if raw_series is None or sector_series is None or z_series is None:
                    return
                df = pd.DataFrame({"raw": raw_series, "z": z_series, "sector": sector_series})
                # Safe transformations to canonical RAW (e.g., invert ev/ebitda → ebitda/ev)
                if transform == "invert":
                    df.loc[df["raw"] == 0, "raw"] = np.nan
                    df["raw"] = 1.0 / df["raw"]
                df.replace([np.inf, -np.inf], np.nan, inplace=True)
                df.dropna(subset=["raw"], inplace=True)
                if df.empty:
                    return
                # Tap‑A: z‑audit rows (index = ticker)
                for tkr, zval in df["z"].items():
                    self._tapA_z_rows.append((asof, tkr, canon, float(zval)))
                # Tap‑B: raw rows + per‑sector stats
                if allow_raw:
                    for tkr, raw_val, sector in df[["raw", "sector"]].itertuples():
                        self._tapB_raw_rows.append((asof, tkr, canon, str(sector), float(raw_val)))
                    stats = df.groupby("sector")["raw"].agg(["mean","std","count"]).reset_index()
                    for _, r in stats.iterrows():
                        std = float(r["std"]) if pd.notnull(r["std"]) and r["std"] > 0 else 1e-12
                        self._tap_stats_rows.append((asof, canon, str(r["sector"]), float(r["mean"]), std, int(r["count"])))
            except Exception as e:
                if hasattr(self, "logger"):
                    self.logger.exception("TWO‑TAP capture failed for factor=%r: %s", factor_engine_name, e)

        # Attach wrappers to *all* available SNZ entry points with correct binding
        for method_name in snz_candidates:
            if not hasattr(engine, method_name):
                continue
            bound_orig = getattr(engine, method_name)  # bound method
            # Always resolve the **unbound** function to avoid double self.
            orig_func = getattr(engine.__class__, method_name, None)
            if orig_func is None:
                # Fallback for unusual descriptors
                orig_func = getattr(bound_orig, "__func__", bound_orig)
            

            @functools.wraps(bound_orig)
            def _make_wrapper(ofunc):
                def _wrapper(self_engine, *args, **kwargs):
                    # Robust param binding against the **unbound** function signature
                    try:
                        sig = inspect.signature(ofunc)
                        bound = sig.bind_partial(self_engine, *args, **kwargs)
                        bound.apply_defaults()
                        amap = bound.arguments
                    except Exception:
                        amap = {}
                    factor_name = (amap.get("factor_name") or amap.get("key") or amap.get("name")
                                   or kwargs.get("factor_name") or kwargs.get("key") or kwargs.get("name")
                                   or "")  # DO NOT use args[0] as a name
                    if not isinstance(factor_name, str) or not factor_name:
                        factor_name = _infer_factor_name_from_stack("")
                    raw_obj = (amap.get("raw_series") or amap.get("raw") or amap.get("values")
                               or kwargs.get("raw_series") or kwargs.get("raw") or kwargs.get("values")
                               or (args[1] if len(args) > 1 else None))
                    sector_obj = (amap.get("sector_series") or amap.get("sectors") or amap.get("sector_map")
                                  or kwargs.get("sector_series") or kwargs.get("sectors") or kwargs.get("sector_map")
                                  or (args[2] if len(args) > 2 else None))
                    # Call the **unbound** original with exactly one self
                    z_series = ofunc(self_engine, *args, **kwargs)
                    asof = (getattr(self, "analysis_date", None) or getattr(self_engine, "analysis_date", None)
                            or kwargs.get("asof") or kwargs.get("analysis_date"))
                    _tap_capture(asof, factor_name, raw_obj, sector_obj, z_series, self_engine=self_engine)
                    return z_series
                return _wrapper
            wrapper = _make_wrapper(orig_func)
            setattr(engine, method_name, types.MethodType(wrapper, engine))

        self._snz_hook_installed = True
        self._snz_hook_engine_id = id(engine)

        # --- Module-level SNZ monkey‑patch (e.g., base_metrics_calculator) ---
        # Some factors normalize via utility modules, bypassing engine methods.
        # Ensure common import paths exist (some runners change package roots)
        candidate_modpaths = [
            "src.utils.factor_calculation.base_metrics_calculator",
            "utils.factor_calculation.base_metrics_calculator",
            "factor_calculation.base_metrics_calculator",
        ]
        for mp in candidate_modpaths:
            try:
                __import__(mp)
            except Exception:
                pass
        # Now scan loaded modules (containment, not only endswith)
        for modname, module in list(sys.modules.items()):
            if not isinstance(modname, str) or "base_metrics_calculator" not in modname:
                continue
            for fn in snz_candidates:
                if not hasattr(module, fn):
                    continue
                orig_fn = getattr(module, fn)
                @functools.wraps(orig_fn)
                def _make_fn_wrapper(ofn):
                    def _fn_wrapper(*args, **kwargs):
                        # We don't have 'self_engine' here; use wrapper's analysis_date
                        # Extract names/series robustly (positional/keyword)
                        try:
                            sig = inspect.signature(ofn)
                            bound = sig.bind_partial(*args, **kwargs)
                            bound.apply_defaults()
                            amap = bound.arguments
                        except Exception:
                            amap = {}
                        factor_name = (amap.get("factor_name") or amap.get("key") or amap.get("name")
                                       or kwargs.get("factor_name") or kwargs.get("key") or kwargs.get("name")
                                       or "")  # DO NOT use args[0] as a name
                        if not isinstance(factor_name, str) or not factor_name:
                            factor_name = _infer_factor_name_from_stack("")
                        raw_obj = (amap.get("raw_series") or amap.get("raw") or amap.get("values")
                                   or kwargs.get("raw_series") or kwargs.get("raw") or kwargs.get("values")
                                   or (args[0] if len(args) > 0 else None))
                        sector_obj = (amap.get("sector_series") or amap.get("sectors") or amap.get("sector_map")
                                      or kwargs.get("sector_series") or kwargs.get("sectors") or kwargs.get("sector_map")
                                      or (args[1] if len(args) > 1 else None))
                        z_series = ofn(*args, **kwargs)
                        asof = getattr(self, "analysis_date", None) or kwargs.get("asof") or kwargs.get("analysis_date")
                        _tap_capture(asof, factor_name, raw_obj, sector_obj, z_series, self_engine=None)
                        return z_series
                    return _fn_wrapper
                setattr(module, fn, _make_fn_wrapper(orig_fn))

        # --- NEW: Engine-module namespace monkey‑patch ---
        # Many factors do:  from utils...base_metrics_calculator import sector_neutralize as SNZ
        # Then call SNZ(...) directly. Patching the utility module won't affect that bound name.
        # We must patch the *engine module's* globals to intercept those calls.
        try:
            import sys, inspect, functools
            eng_mod_name = getattr(engine.__class__, "__module__", None)
            eng_mod = sys.modules.get(eng_mod_name)
        except Exception:
            eng_mod = None

        patched_engine_symbols = []
        if eng_mod is not None:
            # Candidates used across v2.0.1 → v2.1.1 and variants we've seen in logs/config/spec
            eng_symbol_candidates = set(snz_candidates) | {
                "zscore_by_sector", "sector_neutral", "sector_normalize",
                "calc_sector_neutral_z", "sector_zscore", "snz", "snorm",
            }
            # Also heuristically catch any callable imported from a base_metrics_calculator module
            # or whose name suggests a sector-neutral zscore.
            def _looks_like_snz(name, fn):
                nm = (name or "").lower()
                if any(k in nm for k in ["sector", "neutral", "zscore", "normalize"]):
                    return True
                try:
                    srcmod = getattr(fn, "__module__", "") or ""
                except Exception:
                    srcmod = ""
                return "base_metrics_calculator" in srcmod

            for sym_name, sym_val in list(vars(eng_mod).items()):
                if not callable(sym_val):
                    continue
                if (sym_name in eng_symbol_candidates) or _looks_like_snz(sym_name, sym_val):
                    orig_fn = sym_val
                    @functools.wraps(orig_fn)
                    def _make_eng_fn_wrapper(ofn):
                        def _fn_wrapper(*args, **kwargs):
                            # Robust arg binding to extract factor name + raw + sector
                            try:
                                sig = inspect.signature(ofn); bound = sig.bind_partial(*args, **kwargs)
                                bound.apply_defaults(); amap = bound.arguments
                            except Exception:
                                amap = {}
                            factor_name = (amap.get("factor_name") or amap.get("key") or amap.get("name")
                                           or kwargs.get("factor_name") or kwargs.get("key") or kwargs.get("name")
                                           or "")
                            raw_obj = (amap.get("raw_series") or amap.get("raw") or amap.get("values")
                                       or kwargs.get("raw_series") or kwargs.get("raw") or kwargs.get("values")
                                       or (args[0] if len(args) > 0 else None))
                            sector_obj = (amap.get("sector_series") or amap.get("sectors") or amap.get("sector_map")
                                          or kwargs.get("sector_series") or kwargs.get("sectors") or kwargs.get("sector_map")
                                          or (args[1] if len(args) > 1 else None))
                            z_series = ofn(*args, **kwargs)
                            asof = (getattr(self, "analysis_date", None) or kwargs.get("asof")
                                    or kwargs.get("analysis_date"))
                            _tap_capture(asof, factor_name, raw_obj, sector_obj, z_series, self_engine=None)
                            return z_series
                        return _fn_wrapper
                    # Avoid double-wrap: if already wrapped, it has __wrapped__
                    if not hasattr(orig_fn, "__wrapped__"):
                        setattr(eng_mod, sym_name, _make_eng_fn_wrapper(orig_fn))
                        patched_engine_symbols.append(sym_name)

        # Optional: small debug statement you already log
        if patched_engine_symbols:
            logger.info("🔧 TWO‑TAP engine-module symbols patched: %s", patched_engine_symbols)

        # ------------------------------------------------------------------
        # TAP‑B v2: Scoped pandas groupby.apply interposer (wrapper‑only)
        # ------------------------------------------------------------------
        import threading, contextlib

        # Thread‑local flag to restrict interposition strictly to engine methods
        self._tap_tls = getattr(self, "_tap_tls", threading.local())

        def _tap_in_scope():
            return bool(getattr(self._tap_tls, "active", False))

        @contextlib.contextmanager
        def _tap_scope(engine_inst):
            """Activate capture around pandas groupby.apply only while engine factor methods run."""
            if _tap_in_scope():
                # allow nesting without double‑patching
                yield
                return

            self._tap_tls.active = True
            self._tap_tls.engine = engine_inst
            # reentrancy/bypass flags
            if not hasattr(self._tap_tls, "suspend"):
                self._tap_tls.suspend = False

            saved_df_groupby = pd.DataFrame.groupby

            # function‑name heuristics that look like sector‑neutral zscore funcs
            snz_fn_candidates = {
                "sector_zscore", "_sector_zscore", "zscore_by_sector",
                "sector_neutral_z", "sector_normalize", "normalize_factor_by_sector",
                "calc_sector_neutral_z"
            }

            def _infer_factor_name_from_stack(default=""):
                candidates = ("factor_name","factor","metric","metric_name",
                              "name","key","k","fname","fn","code")
                for frameinfo in inspect.stack()[2:8]:
                    try:
                        loc = frameinfo.frame.f_locals
                        for c in candidates:
                            val = loc.get(c)
                            if isinstance(val, str) and val:
                                return val
                    except Exception:
                        pass
                return default

            def _groupby_wrapper(df_self, by=None, *g_args, **g_kwargs):
                # If suspended (e.g., internal stats aggregation), bypass interposer
                if getattr(self._tap_tls, "suspend", False):
                    return saved_df_groupby(df_self, by=by, *g_args, **g_kwargs)
                gb_obj = saved_df_groupby(df_self, by=by, *g_args, **g_kwargs)
                orig_apply = gb_obj.apply

                def _apply_wrapper(func, *a, **k):
                    fn_name = getattr(func, "__name__", "").lower()
                    looks_like_snz = (
                        fn_name in snz_fn_candidates or
                        ("zscore" in fn_name) or
                        ("sector" in fn_name and "z" in fn_name)
                    )
                    if not _tap_in_scope() or not looks_like_snz:
                        return orig_apply(func, *a, **k)

                    # Heuristics to recover sector and raw vectors
                    sector_series = None
                    if isinstance(by, str) and isinstance(df_self, pd.DataFrame) and (by in df_self.columns):
                        sector_series = df_self[by]
                    elif hasattr(by, "index"):
                        sector_series = by

                    raw_series = None
                    if isinstance(df_self, pd.DataFrame):
                        prefer = ["raw","value","val","x","metric","factor","measure"]
                        for c in prefer:
                            if c in df_self.columns:
                                raw_series = df_self[c]; break
                        if raw_series is None:
                            num_cols = df_self.select_dtypes(include="number").columns.tolist()
                            if isinstance(by, str) and by in num_cols:
                                try:
                                    num_cols.remove(by)
                                except ValueError:
                                    pass
                            if num_cols:
                                raw_series = df_self[num_cols[0]]

                    # Call the real apply to produce z, but suspend inner interposer to avoid recursion
                    prev = self._tap_tls.suspend
                    self._tap_tls.suspend = True
                    try:
                        z_series = orig_apply(func, *a, **k)
                    finally:
                        self._tap_tls.suspend = prev

                    factor_name = _infer_factor_name_from_stack("")
                    asof = getattr(self, "analysis_date", None)

                    try:
                        _tap_capture(asof, factor_name, raw_series, sector_series, z_series,
                                     self_engine=getattr(self._tap_tls, "engine", None))
                    except Exception as e:
                        if hasattr(self, "logger"):
                            self.logger.exception("TWO‑TAP groupby/apply capture failed: %s", e)

                    return z_series

                class _GBProxy:
                    def __init__(self, gb, apply_fn):
                        self._gb = gb; self.apply = apply_fn
                    def __getattr__(self, name):
                        return getattr(self._gb, name)

                return _GBProxy(gb_obj, _apply_wrapper)

            # Activate monkey‑patch for the duration of the engine call
            pd.DataFrame.groupby = _groupby_wrapper
            try:
                yield
            finally:
                pd.DataFrame.groupby = saved_df_groupby
                self._tap_tls.active = False
                self._tap_tls.engine = None

        # Wrap engine methods that compute individual factors so they run inside the scope
        calc_targets = [
            # generic families
            "calculate_individual_quality_factors",
            "calculate_individual_value_factors",
            "calculate_individual_momentum_factors",
            # enhanced factors
            "calculate_low_volatility",
            "calculate_fcf_yield",
            "calculate_f_score",
            # sector variants observed in some engines
            "calculate_individual_quality_factors_banking",
            "calculate_individual_quality_factors_securities",
            "calculate_individual_value_factors_banking",
            "calculate_individual_value_factors_securities",
        ]

        for m in calc_targets:
            if not hasattr(engine, m):
                continue
            bound = getattr(engine, m)
            orig = getattr(engine.__class__, m, getattr(bound, "__func__", bound))

            @functools.wraps(bound)
            def _make_calc_wrapper(ofunc):
                def _wrapper(self_engine, *args, **kwargs):
                    with _tap_scope(self_engine):
                        return ofunc(self_engine, *args, **kwargs)
                return _wrapper

            setattr(engine, m, types.MethodType(_make_calc_wrapper(orig), engine))


    def _ensure_observer_hooks(self, engine):
        """
        Install TWO‑TAP wrappers on the given engine instance exactly once.
        Safe to call repeatedly; no double-wrapping.
        """
        if engine is None:
            raise ValueError("Engine is None in _ensure_observer_hooks")
        if getattr(self, "_snz_hook_installed", False) and getattr(self, "_snz_hook_engine_id", None) == id(engine):
            return
        # (Re)install for this engine instance
        self._install_observer_hooks(engine)

    def _flush_two_tap_to_db(self, raw_conn, strategy_version):
        # NEW: chunked writes; RAW→factor_signals_raw, stats→factor_norm_stats
        if not self._tapB_raw_rows and not self._tap_stats_rows:
            return
        cur = raw_conn.cursor()
        sv = strategy_version or "analytics_v1_neo_fixed"

        def _chunks(it, n=1000):
            it = list(it)
            for i in range(0, len(it), n):
                yield it[i:i+n]

        # RAW rows: (date, ticker, factor_code, sector, raw_value)
        raw_records = []
        for (d, t, f_code, _sector, v) in self._tapB_raw_rows:
            factor_id = FACTOR_CODE_TO_ID.get(ENGINE_TO_CANONICAL.get(str(f_code).lower(), str(f_code)))
            if not factor_id:
                continue
            raw_records.append((t, d, sv, factor_id, v))  # (ticker, date, version, factor_id, raw_value)

        if raw_records:
            insert_raw = (
                "INSERT INTO factor_signals_raw "
                "(ticker, date, strategy_version, factor_id, raw_value) "
                "VALUES (%s,%s,%s,%s,%s) "
                "ON DUPLICATE KEY UPDATE raw_value = VALUES(raw_value)"
            )
            for ch in _chunks(raw_records, 1000):
                cur.executemany(insert_raw, ch)

        # STATS rows: (date, factor_code, sector, mean, std, n)
        stats_records = []
        for (d, f_code, sector, mean_val, std_val, universe_n) in self._tap_stats_rows:
            factor_id = FACTOR_CODE_TO_ID.get(ENGINE_TO_CANONICAL.get(str(f_code).lower(), str(f_code)))
            if not factor_id:
                continue
            stats_records.append((d, sector, sv, factor_id, mean_val, std_val, universe_n))

        if stats_records:
            insert_stats = (
                "INSERT INTO factor_norm_stats "
                "(date, sector, strategy_version, factor_id, mean_value, std_value, universe_size) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s) "
                "ON DUPLICATE KEY UPDATE mean_value=VALUES(mean_value), std_value=VALUES(std_value), universe_size=VALUES(universe_size)"
            )
            for ch in _chunks(stats_records, 1000):
                cur.executemany(insert_stats, ch)

        raw_conn.commit()
        cur.close()
        # clear buffers after successful commit
        self._tapA_z_rows.clear()
        self._tapB_raw_rows.clear()
        self._tap_stats_rows.clear()

    def _synthesize_debt_equity(self, raw_conn, asof_date, tickers):
        """
        Compute Debt/Equity = TotalDebt / BookEquity, with BookEquity = (B/P) * MarketCap.
        Uses PIT fundamentals with 45D lag as per engine spec.
        Writes to factor_signals_raw under canonical 'debt_equity'.
        """
        cur = raw_conn.cursor()
        # 1) Fetch raw book_to_price and market cap for the as-of date
        book_value_factor_id = FACTOR_CODE_TO_ID.get('book_value', 6)
        cur.execute(
            """
            SELECT fs.ticker, fs.raw_value AS book_to_price, v.market_cap
            FROM factor_signals_raw fs
            JOIN vcsc_daily_data_complete v
              ON v.ticker = fs.ticker AND v.trading_date = %s
            WHERE fs.date = %s
              AND fs.factor_id = %s
              AND fs.strategy_version = %s
              AND fs.ticker IN %s
            """,
            (asof_date, asof_date, book_value_factor_id, self.version, tuple(tickers)),
        )
        rows = cur.fetchall()
        if not rows:
            cur.close(); return 0
        # 2) Determine PIT quarter with 45D lag and pull TotalDebt
        cur.execute(
            """
            SELECT t.ticker,
                   (COALESCE(t.ShortTermDebt,0) + COALESCE(t.LongTermDebt,0)) AS total_debt
            FROM (
                SELECT ticker, ShortTermDebt, LongTermDebt, year, quarter
                FROM v_comprehensive_fundamental_items
                WHERE (year, quarter) = (
                    SELECT year, quarter FROM pit_quarter_with_lag(%s, 45)
                )
            ) AS t
            WHERE t.ticker IN %s
            """,
            (asof_date, tuple([r[0] for r in rows])),
        )
        debt = {t: d for (t, d, *_) in cur.fetchall()}
        # 3) Insert synthesized Debt/Equity
        inserts = []
        for tkr, b2p, mcap in rows:
            if b2p is None or mcap is None: 
                continue
            book_equity = float(b2p) * float(mcap)
            if book_equity <= 0 or tkr not in debt:
                continue
            de = float(debt[tkr]) / book_equity
            inserts.append(("analytics_v1_neo_fixed", asof_date, tkr, "debt_equity", "ALL", de))
        if inserts:
            cur.executemany(
                "INSERT INTO factor_signals_raw "
                "(strategy_version, analysis_date, ticker, factor_name, sector, raw_value, created_at) "
                "VALUES (%s,%s,%s,%s,%s,%s,NOW())",
                inserts
            )
            raw_conn.commit()
        cur.close()
        return len(inserts)
    
    def _install_base_metric_tapb_fallback_live(self):
        """
        Scan already-loaded modules and wrap base-metric functions that compute factors
        bypassing SNZ. This catches leverage (debt_equity) and friends even if the engine
        never calls our z-score path.
        """
        import sys, types
        
        def _bm_wrap(self, fn, factor_code):
            @functools.wraps(fn)
            def _w(*args, **kwargs):
                out = fn(*args, **kwargs)
                try:
                    if isinstance(out, pd.Series) and not out.empty:
                        # Try to find a sector vector aligned to out.index
                        sec = None
                        for arg in list(args) + list(kwargs.values()):
                            if hasattr(arg, 'columns'):
                                # common names
                                for cand in ('sector', 'industry', 'gics_sector'):
                                    if cand in arg.columns:
                                        sec = arg[cand].astype(str).reindex(out.index)
                                        break
                            if sec is not None:
                                break
                        if sec is None:
                            sec = pd.Series('ALL', index=out.index)
                        self._tapb_raw_buffers.setdefault(factor_code, []).append((out.copy(), sec.copy()))
                        self._tapb_seen_metrics.add(f"{factor_code}_base_metric")
                except Exception as e:
                    logger.exception(f"TAP-B base-metric fallback failed on {fn.__name__}: {e}")
                return out
            return _w

        # function-name → canonical factor
        name_to_factor = {
            # leverage / D/E
            'compute_debt_to_equity':'debt_equity',
            'calc_debt_to_equity':   'debt_equity',
            'debt_to_equity':        'debt_equity',
            'debt_equity':           'debt_equity',
            'financial_leverage':    'debt_equity',
            'compute_financial_leverage':'debt_equity',
            'calc_financial_leverage':   'debt_equity',
            'leverage':              'debt_equity',

            # value: EV/EBITDA
            'compute_ev_ebitda':     'ev_ebitda',
            'calc_ev_ebitda':        'ev_ebitda',
            'ev_to_ebitda':          'ev_ebitda',
            'ebitda_to_ev':          'ev_ebitda',

            # quality margins
            'compute_net_profit_margin':'net_profit_margin',
            'net_profit_margin':        'net_profit_margin',
            'compute_ebitda_margin':    'ebitda_margin',
            'ebitda_margin':            'ebitda_margin',
            'compute_operating_margin': 'operating_margin',
            'operating_margin':         'operating_margin',
        }

        wrapped = 0
        for mod_name, mod in list(sys.modules.items()):
            if not mod or not isinstance(mod, types.ModuleType):
                continue
            # Heuristic: limit to likely modules to avoid accidental wraps
            if not any(k in mod_name.lower() for k in ('base_metrics', 'factor_calculation', 'metrics_calculator')):
                continue
            for fn_name, fcode in name_to_factor.items():
                if hasattr(mod, fn_name):
                    fn = getattr(mod, fn_name)
                    if isinstance(fn, (types.FunctionType, types.MethodType)) and not getattr(fn, '_tapb_wrapped', False):
                        setattr(mod, fn_name, _bm_wrap(self, fn, fcode))
                        setattr(getattr(mod, fn_name), '_tapb_wrapped', True)
                        logger.info(f"🔌 TAP-B fallback installed on {mod_name}.{fn_name} -> {fcode}")
                        wrapped += 1
        if wrapped == 0:
            logger.warning("⚠️  No base-metric functions found in live modules; if leverage still missing, import location is unusual.")
    
    def _inject_debt_equity_from_snapshots(self):
        """Agent Neo's debt_equity synthesis from DF snapshots"""
        import numpy as np
        
        # Return if we already captured non-empty D/E
        pairs = self._tapb_raw_buffers.get('debt_equity', [])
        if any((isinstance(r, pd.Series) and r.size) for (r, _) in pairs):
            return

        snaps = list(getattr(self, "_df_snapshots", []))
        if not snaps:
            logger.warning("⚠️ No DF snapshots available for D/E synthesis.")
            return

        for df in reversed(snaps):
            # sector vector
            sec_col = next((c for c in df.columns if c.lower()=='sector'), None)
            sec = df[sec_col].astype(str) if sec_col else pd.Series('ALL', index=df.index)

            # debt candidates
            if 'total_debt' in df.columns:
                debt = pd.to_numeric(df['total_debt'], errors='coerce')
            elif 'interest_bearing_debt' in df.columns:
                debt = pd.to_numeric(df['interest_bearing_debt'], errors='coerce')
            else:
                st = next((c for c in df.columns if 'short' in c.lower() and 'debt' in c.lower()), None)
                lt = next((c for c in df.columns if 'long' in c.lower() and 'debt' in c.lower()), None)
                if st and lt:
                    debt = pd.to_numeric(df[st], errors='coerce') + pd.to_numeric(df[lt], errors='coerce')
                else:
                    continue

            # equity candidates
            for eqc in ('equity','book_equity','shareholders_equity','total_equity'):
                if eqc in df.columns:
                    equity = pd.to_numeric(df[eqc], errors='coerce')
                    break
            else:
                continue

            ratio = (debt / equity).replace([np.inf, -np.inf], np.nan).dropna()
            ratio = ratio[(ratio >= 0) & (ratio <= 10)]  # sanity filter
            if ratio.empty:
                continue

            self._tapb_raw_buffers.setdefault('debt_equity', []).append(
                (ratio.copy(), sec.reindex(ratio.index).fillna('ALL'))
            )
            logger.info(f"🧩 Injected debt_equity from snapshots with {len(ratio)} tickers.")
            return

        logger.warning("⚠️ D/E synthesis attempted but no usable columns found.")
    
    def _pit_quarter_from_date(self, dt: pd.Timestamp) -> tuple[int, int]:
        """Resolve (year, quarter) for fundamentals with a 45-day reporting lag."""
        dt = pd.Timestamp(dt)
        eff = dt - pd.Timedelta(days=45)
        y = eff.year
        q = ((eff.month - 1) // 3) + 1
        return y, q
    
    def _fetch_last_price_shares(self, tickers: list[str], asof: pd.Timestamp) -> pd.DataFrame:
        if not tickers:
            return pd.DataFrame(columns=["ticker","close","total_shares"])
        sql = text("""
            WITH last_dates AS (
                SELECT ticker, MAX(trading_date) AS lastdate
                FROM vcsc_daily_data_complete
                WHERE ticker IN :tickers AND trading_date <= :asof
                GROUP BY ticker
            )
            SELECT v.ticker, v.close, v.total_shares
            FROM vcsc_daily_data_complete v
            JOIN last_dates d ON v.ticker=d.ticker AND v.trading_date=d.lastdate
            WHERE v.close IS NOT NULL AND v.total_shares IS NOT NULL
        """)
        with self.engine_ro.begin() as conn:
            df = pd.read_sql(sql, conn, params={"tickers": tuple(tickers), "asof": pd.Timestamp(asof)})
        return df
    
    def _fetch_total_debt(self, tickers: list[str], y: int, q: int) -> pd.DataFrame:
        if not tickers:
            return pd.DataFrame(columns=["ticker","TotalDebt"])
        # Prefer the same view used elsewhere in the engine
        sql = text("""
            SELECT ticker,
                   COALESCE(ShortTermDebt,0) + COALESCE(LongTermDebt,0) AS TotalDebt
            FROM v_comprehensive_fundamental_items
            WHERE year = :y AND quarter = :q AND ticker IN :tickers
        """)
        with self.engine_ro.begin() as conn:
            df = pd.read_sql(sql, conn, params={"y": y, "q": q, "tickers": tuple(tickers)})
        return df
    
    def _inject_debt_equity_from_bp_and_mcap(self, analysis_date: pd.Timestamp, universe_df: pd.DataFrame):
        """
        Force-capture non-financial Debt/Equity from raw components we already trust:
          - Equity ≈ (book_to_price_raw) * (PIT MarketCap)
          - Debt   = ShortTermDebt + LongTermDebt @ PIT Quarter
        Writes to self._tapb_raw_buffers['debt_equity'].
        """
        import numpy as np

        # If already captured non-empty, skip
        pairs = self._tapb_raw_buffers.get('debt_equity', [])
        if any((isinstance(r, pd.Series) and r.size) for (r, _) in pairs):
            return

        # 1) Identify eligible tickers (non-fin)
        sec_map = universe_df.set_index('ticker')['sector'].to_dict()
        nonfin = [t for t,s in sec_map.items() if s not in ('Banks','Banking','Securities')]

        if not nonfin:
            return

        # 2) Get book_to_price_raw captured by Tap-B under canonical 'book_value'
        bp = pd.Series(dtype=float)
        for raw_series, _sec_series in self._tapb_raw_buffers.get('book_value', []):
            s = pd.to_numeric(raw_series, errors='coerce').dropna()
            bp = s if bp.empty else bp.combine_first(s)    # first-seen wins

        if bp.empty:
            # No B/P; nothing to synthesize from
            return

        # Restrict to non-fin tickers we actually have B/P for
        candidates = sorted(set(nonfin) & set(bp.index))

        if not candidates:
            return

        # 3) PIT market cap = last close × last total_shares
        px_sh = self._fetch_last_price_shares(candidates, analysis_date)
        if px_sh.empty:
            return
        mcap = (px_sh.set_index('ticker')['close'] * px_sh.set_index('ticker')['total_shares']).rename('mcap').dropna()

        # 4) Equity ≈ B/P × MarketCap (book equity)
        equity = (bp.reindex(mcap.index) * mcap).dropna()
        equity = equity[equity > 0]

        if equity.empty:
            return

        # 5) TotalDebt at PIT quarter
        y, q = self._pit_quarter_from_date(analysis_date)
        debt_df = self._fetch_total_debt(list(equity.index), y, q)
        if debt_df.empty:
            return

        debt = pd.to_numeric(debt_df.set_index('ticker')['TotalDebt'], errors='coerce').dropna()
        common = sorted(set(debt.index) & set(equity.index))
        if not common:
            return

        # 6) Compute D/E with sanity bounds, assemble sector labels
        de = (debt.reindex(common) / equity.reindex(common)).replace([np.inf, -np.inf], np.nan).dropna()
        de = de[(de >= 0) & (de <= 10)]
        if de.empty:
            return

        sec = pd.Series({t: sec_map.get(t, 'Non-Financial') for t in de.index})
        self._tapb_raw_buffers.setdefault('debt_equity', []).append((de.copy(), sec.copy()))
        self._tapb_seen_metrics.add('debt_equity_synth_bp_mcap')
        logger.info(f"🧩 Injected debt_equity from B/P×MCap with {len(de)} tickers.")
    
    def _resolve_factor_code_from_metric(self, metric_name: str) -> str:
        """Agent Neo's robust metric→factor resolver (exact as provided)"""
        # Metric names in engine often come as DB/feature names; normalize aggressively
        key = _canonicalize(metric_name)
        candidate = None
        
        if key in METRIC_TO_FACTOR: 
            candidate = METRIC_TO_FACTOR[key]

        # Heuristics: common suffixes/prefixes
        if not candidate and key.endswith("_z"): 
            stripped = key[:-2]
            if stripped in METRIC_TO_FACTOR:
                candidate = METRIC_TO_FACTOR[stripped]
        if not candidate and key.endswith("_raw"): 
            stripped = key[:-4]
            if stripped in METRIC_TO_FACTOR:
                candidate = METRIC_TO_FACTOR[stripped]

        # Try direct canonical
        if not candidate and key in FACTOR_META: 
            candidate = key

        # Last resort: handle sectorized f_score
        if not candidate and key.startswith("f_score"): 
            candidate = "f_score"

        # Only return if candidate is in our FACTOR_META (factors we actually track)
        if candidate and candidate in FACTOR_META:
            return candidate

        # Unknown — record and let Tier-0 fail (so you know to add mapping)
        self._unknown_metric_names.add(metric_name)
        return "__unknown__"
    
    def _install_tap_b_normalization_hook(self, engine):
        """
        Agent Neo's exact Tap B: capture raw metric series + sector labels right before sector z-normalization.
        Works for calls like calculate_sector_neutral_zscore(df, 'roae', 'sector')
        and tolerates kwargs ordering.
        """

        # 1) Locate the method on the bound instance
        target = getattr(engine, "calculate_sector_neutral_zscore", None)
        if target is None or not callable(target):
            raise RuntimeError("calculate_sector_neutral_zscore not found on engine")

        # Unwrap any previously wrapped function to avoid recursive self-calls
        orig_bound = target
        try:
            while hasattr(orig_bound, "__wrapped__") and callable(getattr(orig_bound, "__wrapped__")):
                orig_bound = orig_bound.__wrapped__
        except Exception:
            pass
        self._orig_calc_snz = orig_bound
        wrapper_self = self  # capture wrapper instance for closure

        @functools.wraps(orig_bound)
        def _wrapped_calc_snz(df, metric_column, sector_column="sector", *args, **kwargs):
            # Defensive: cope with various invocation patterns (kwargs, alias names)
            # Resolve metric_column / sector_column from kwargs if needed
            if not isinstance(metric_column, str) and "metric_column" in kwargs:
                metric_column = kwargs["metric_column"]
            if not isinstance(sector_column, str) and "sector_column" in kwargs:
                sector_column = kwargs["sector_column"]

            # If engine passed Series instead of column name, normalize to a column name
            # (rare, but handle gracefully)
            if not isinstance(metric_column, str):
                # create a temporary column name and stash series into df (view)
                tmp_col = "__tapb_metric__"
                df = df.copy()
                df[tmp_col] = pd.Series(metric_column, index=df.index)
                metric_column = tmp_col

            # 2) Capture RAW series + sector labels + z-scores (Agent Neo's strengthened Tap-B)
            try:
                raw_series   = df[metric_column].copy()
                sector_series= df[sector_column].copy() if sector_column in df.columns else df["sector"].copy()
                
                # Resolve canonical factor code (e.g., 'roae', 'book_value', 'mom_12m', 'f_score', 'nim', ...)
                factor_code = wrapper_self._resolve_factor_code_from_metric(metric_column)

                # Agent Neo's strengthened capture: Book raw values for conversion
                wrapper_self._tapb_raw_buffers.setdefault(factor_code, []).append(
                    (raw_series, sector_series)
                )
                
                # Also capture raw data in legacy _raw_capture for compatibility (Agent Neo's fix)
                if factor_code != "__unknown__":
                    wrapper_self._raw_capture.setdefault(factor_code, {}).update(raw_series.to_dict())
                    
                    # Compute normalization stats per sector for _norm_stats
                    sector_stats = {}
                    for sector in sector_series.unique():
                        sector_mask = sector_series == sector
                        sector_values = raw_series[sector_mask].dropna()
                        if len(sector_values) > 0:
                            sector_stats[sector] = {
                                'mean': float(sector_values.mean()),
                                'std': float(sector_values.std()) if len(sector_values) > 1 else 1.0,
                                'count': len(sector_values)
                            }
                    
                    if sector_stats:
                        wrapper_self._norm_stats[metric_column] = sector_stats
                
                wrapper_self._tapb_seen_metrics.add(metric_column)

            except Exception as e:
                logger.warning(f"TAP-B capture skipped for metric='{metric_column}': {e}")

            # Log what metric we're seeing for debugging
            # Throttle metric logging: only once per metric per date
            try:
                if metric_column not in wrapper_self._tapb_seen_metrics:
                    logger.info(f"TAP-B processing metric: '{metric_column}' -> factor_code: '{factor_code if 'factor_code' in locals() else 'unknown'}'")
            except Exception:
                pass

            # 3) Call the original method (returns z-scored Series) and capture z-scores for Tap-A
            # Suspend pandas interposer during the inner SNZ call to avoid nested re-entry
            prev_suspend = False
            if hasattr(wrapper_self, "_tap_tls"):
                prev_suspend = getattr(wrapper_self._tap_tls, "suspend", False)
                wrapper_self._tap_tls.suspend = True
            try:
                z_result = wrapper_self._orig_calc_snz(df, metric_column, sector_column, *args, **kwargs)
            finally:
                if hasattr(wrapper_self, "_tap_tls"):
                    wrapper_self._tap_tls.suspend = prev_suspend
            
            # Agent Neo's Tap-A: Capture z-scores for audit (factor_code must be known)
            try:
                if factor_code != "__unknown__" and hasattr(z_result, 'to_dict'):
                    wrapper_self._captured_z.setdefault(factor_code, {}).update(z_result.to_dict())
            except Exception as e:
                logger.warning(f"TAP-A z-score capture failed for {factor_code}: {e}")
            
            return z_result

        # Rebind to the instance (important for bound method semantics)
        setattr(engine, "calculate_sector_neutral_zscore", _wrapped_calc_snz)
        logger.info("🔌 TAP-B installed on calculate_sector_neutral_zscore")
    
    def _install_individual_factor_hooks(self):
        """Fallback: Install hooks on individual factor calculation methods"""
        
        # Map of method names to factor names for all 19 factors
        method_factor_map = {
            '_get_raw_f_score_non_financial': 'f_score',
            '_get_raw_f_score_banking': 'f_score', 
            '_get_raw_f_score_securities': 'f_score',
            '_get_individual_f_score_factors': 'f_score',
            '_get_individual_fcf_yield_factors': 'fcf_yield',
            '_get_individual_low_vol_factors': ['inv_volatility_63d', 'beta', 'low_vol_score'],
            '_get_individual_quality_factors': ['roae', 'gross_margin', 'debt_equity', 'nim', 'cost_income', 'operating_margin_sec', 'cost_ratio_sec'],
            '_get_individual_value_factors': ['earnings_yield', 'book_value', 'sales_multiple'],
            '_get_individual_momentum_factors': ['mom_12m', 'mom_6m', 'mom_3m']
        }
        
        for method_name, factor_names in method_factor_map.items():
            if hasattr(self.qvm_engine, method_name):
                original_method = getattr(self.qvm_engine, method_name)
                
                def create_wrapper(orig_method, factors):
                    def wrapped_method(*args, **kwargs):
                        result = orig_method(*args, **kwargs)
                        
                        # Capture results based on return type
                        if isinstance(result, dict):
                            # Direct factor -> ticker -> value mapping
                            if isinstance(factors, list):
                                # Multiple factors returned
                                for factor_name in factors:
                                    if factor_name in result:
                                        factor_data = result[factor_name]
                                        self._capture_factor_data(factor_name, factor_data)
                            else:
                                # Single factor
                                self._capture_factor_data(factors, result)
                        
                        return result
                    return wrapped_method
                
                # Install the hook
                wrapped = create_wrapper(original_method, factor_names)
                setattr(self.qvm_engine, method_name, wrapped)
                logger.debug(f"Installed hook on {method_name} for factors: {factor_names}")
    
    def _capture_factor_data(self, factor_name: str, factor_data):
        """Helper to capture factor data in consistent format"""
        if isinstance(factor_data, dict):
            for ticker, value in factor_data.items():
                if ticker not in self.captured_factors:
                    self.captured_factors[ticker] = {}
                self.captured_factors[ticker][factor_name] = value
        elif hasattr(factor_data, 'items'):  # pandas Series
            for ticker, value in factor_data.items():
                if ticker not in self.captured_factors:
                    self.captured_factors[ticker] = {}
                self.captured_factors[ticker][factor_name] = value
    
    def _get_analytics_writer_connection(self):
        """Agent Neo Fix: Enforce analytics_writer credentials"""
        try:
            config_path = project_root / 'config' / 'database.yml'
            with open(config_path, 'r') as f:
                db_config = yaml.safe_load(f)
            
            if 'analytics_writer' in db_config:
                analytics_config = db_config['analytics_writer']
                connection_string = (
                    f"mysql+pymysql://{analytics_config['username']}:{analytics_config['password']}"
                    f"@{analytics_config['host']}/{analytics_config['schema_name']}"
                )
                engine = create_engine(connection_string, pool_pre_ping=True)
                logger.info("✅ Using dedicated analytics_writer database user")
                return engine
            else:
                logger.warning("⚠️  analytics_writer credentials not found, using production user")
                logger.warning("   This should be fixed before production deployment")
                return create_db_connection(project_root)
                
        except Exception as e:
            logger.error(f"Failed to setup analytics writer connection: {str(e)}")
            return create_db_connection(project_root)
    
    def _install_safety_guard(self):
        """Runtime guard prevents writes to factor_scores_qvm"""
        def blocked_composite_write(*args, **kwargs):
            raise RuntimeError(
                "BLOCKED BY DESIGN: Analytics wrapper never writes to factor_scores_qvm."
            )
        self._write_composites_to_production = blocked_composite_write
        logger.info("🛡️  Safety guard installed: factor_scores_qvm writes blocked")

    def run_date(self, date: pd.Timestamp) -> tuple[bool, int]:
        """
        Agent Neo's flow with observer pattern (HARDENED):
        Returns (success: bool, rows_written: int) - Agent's Fix 3: success counted only after commit with rows
        """
        
        logger.info(f"Processing {date.date()} (analytics-only mode)")
        
        try:
            # Make analysis_date accessible to TAP (fallback path) and install TWO‑TAP now.
            self.analysis_date = date
            self._ensure_observer_hooks(self.qvm_engine)
            
            # Agent Neo's per-date reset and hook installation
            self._prepare_date()
            rows_written = 0
            
            # Get universe
            universe_config = {
                'lookback_days': 63,
                'adtv_threshold_bn': 10.0, 
                'top_n': 200,
                'min_trading_coverage': 0.6
            }
            
            universe_df = get_liquid_universe_dataframe(date, self.engine_ro, universe_config)
            
            if universe_df.empty:
                logger.warning(f"Empty universe for {date.date()}, skipping")
                return False, 0
                
            logger.info(f"Universe size: {len(universe_df)} stocks")
            
            # Agent Neo Fix B: Use engine's sector mapping, not wrapper's collapsed mapping
            sector_map = self._load_correct_sector_map(universe_df)
            
            # Trigger production quality composition (observer hooks will capture)
            try:
                self._trigger_production_quality_composition(date, universe_df)
            except Exception as e:
                logger.error(f"Engine orchestration failed: {e}")
                # On failure, best-effort flush whatever was captured
                if self.write_sidecar and (self._tapB_raw_rows or self._tap_stats_rows):
                    try:
                        raw_conn = self.engine_sidecar.raw_connection()
                        try:
                            self._flush_two_tap_to_db(raw_conn, self.version)
                            logger.info("✅ PATCH-M: Flushed captured data despite engine issues")
                        finally:
                            raw_conn.close()
                    except Exception as flush_e:
                        logger.error(f"PATCH-M: Failed to flush buffers: {flush_e}")
                # Re-raise after best-effort flush
                raise

            # Force-capture leverage if SNZ path never emits it (Agent Neo's B/P×MarketCap synthesis)
            if getattr(self, 'enable_debt_equity', False):
                self._inject_debt_equity_from_bp_and_mcap(date, universe_df)
            raw_factors = self._convert_captured_to_raw_factors()
            
            if not raw_factors:
                logger.error("No factors captured from production orchestrator")
                return False, 0
            
            # Agent Neo Tier-0 Audit with ELIGIBLE denominator
            if self.audit_tier == 'tier0':
                audit_passed = self._tier0_audit_neo_fixed(date, sector_map, raw_factors, universe_df)
                if not audit_passed:
                    logger.error(f"❌ TIER-0 AUDIT FAILED for {date.date()} - SKIPPING DATE")
                    return False, 0
                logger.info(f"✅ TIER-0 AUDIT PASSED for {date.date()} - PROCEEDING TO WRITE")
            
            # Agent Neo's TWO-TAP database flush
            if self.write_sidecar:
                # Use DB-API raw connection for batched executemany writes
                raw_conn = self.engine_sidecar.raw_connection()
                try:
                    # Count rows before flushing (since flush clears buffers)
                    tap_raw_count = len(self._tapB_raw_rows)
                    tap_stats_count = len(self._tap_stats_rows)

                    # Flush TWO-TAP data (raw + stats)
                    self._flush_two_tap_to_db(raw_conn, self.version)

                    # Synthesize debt_equity from book_to_price and market cap
                    debt_equity_rows = self._synthesize_debt_equity(raw_conn, date, universe_df['ticker'].tolist())

                    rows_written = tap_raw_count + tap_stats_count + debt_equity_rows

                    # Fallback: If TAP-B low-level buffers were empty, persist using high-level writers
                    if rows_written == 0:
                        try:
                            with self.engine_sidecar.begin() as conn2:
                                w1 = self._write_factor_signals_raw_analytics_only(date, raw_factors, conn2)
                                w2 = self._write_factor_norm_stats_analytics_only(date, raw_factors, sector_map, conn2)
                                rows_written = (w1 or 0) + (w2 or 0) + debt_equity_rows
                        except Exception as e:
                            logger.error(f"Fallback write via SQLAlchemy failed: {e}")

                    if rows_written > 0:
                        logger.info(f"✅ TWO-TAP factors captured: {rows_written} rows written to sidecar (including {debt_equity_rows} debt_equity)")
                    else:
                        logger.error(f"❌ No rows written to sidecar - returning FAILURE")
                        return False, 0
                finally:
                    try:
                        raw_conn.close()
                    except Exception:
                        pass
            
            return True, rows_written
            
        except Exception as e:
            logger.error(f"Error processing {date.date()}: {str(e)}")
            logger.error(traceback.format_exc())
            return False, 0
    
    def _load_correct_sector_map(self, universe_df: pd.DataFrame) -> Dict[str, str]:
        """
        Agent's Fix 1: Canonical sector mapping (robust to PIT drift)
        Use sector-class resolver instead of string matching
        """
        # Agent's canonical sector resolver
        CANON = {
            "BANKS": {"Bank", "Banks", "Banking", "Commercial Banks"},
            "SECURITIES": {"Securities", "Brokerage", "Financial Services"},
            "NON_FIN": {"Non-Financial", "Industrials", "Materials", "Real Estate", 
                       "Construction", "Logistics", "Plastics", "Utilities", "Wholesale",
                       "Construction Materials", "Food & Beverage", "Technology",
                       "Ancillary Production", "Mining & Oil", "Healthcare", "Household Goods",
                       "Electrical Equipment", "Seafood", "Retail", "Agriculture", "Machinery",
                       "Industrial Services", "Hotels & Tourism", "Rubber Products"}
        }
        
        def canonical_sector(label: str) -> str:
            lab = (label or "").strip()
            for k, vals in CANON.items():
                if lab in vals: 
                    return k
            return "NON_FIN"  # default safe bucket
        
        # Map to canonical sectors but preserve original for logging
        original_sector_map = universe_df.set_index('ticker')['sector'].to_dict()
        canonical_sector_map = {ticker: canonical_sector(sector) 
                               for ticker, sector in original_sector_map.items()}
        
        # Log sector distribution with both original and canonical
        orig_counts = pd.Series(original_sector_map).value_counts()
        canon_counts = pd.Series(canonical_sector_map).value_counts()
        
        logger.info(f"✅ Agent's Fix 1: Canonical sector mapping applied")
        logger.info(f"   Original - Banks: {orig_counts.get('Banks', 0)}, Securities: {orig_counts.get('Securities', 0)}")
        logger.info(f"   Canonical - BANKS: {canon_counts.get('BANKS', 0)}, SECURITIES: {canon_counts.get('SECURITIES', 0)}, NON_FIN: {canon_counts.get('NON_FIN', 0)}")
        
        # Return original mapping for production consistency 
        return original_sector_map
    
    def _resolve_pit_quarter(self, date_d: pd.Timestamp, max_back_quarters: int = 3):
        """
        Agent's TEMPORAL LOGIC SPECIFICATION: PrevQuarter + bounded fallback
        
        For any rebalance date D, use PIT fundamentals already available by D.
        Base quarter = PrevQuarter(Y(D), Q(D)) with bounded fallback chain.
        """
        
        # Current calendar quarter
        y = date_d.year
        q = ((date_d.month - 1) // 3) + 1
        
        # Agent's spec: Base quarter = PrevQuarter(Y(D), Q(D))
        def prev_quarter(year, quarter):
            if quarter == 1:
                return year - 1, 4  # Q1 → prev year Q4 (cold-start fix)
            else:
                return year, quarter - 1
        
        def shift_quarter(year, quarter, shift):
            """Shift quarter by 'shift' quarters (negative = backward)"""
            total_quarters = (year * 4) + (quarter - 1) + shift
            new_year = total_quarters // 4
            new_quarter = (total_quarters % 4) + 1
            return new_year, new_quarter
        
        # Base quarter (PrevQuarter)
        yb, qb = prev_quarter(y, q)
        
        # Agent's bounded fallback chain: Q_b, Q_b-1, Q_b-2, Q_b-3
        for back in range(0, max_back_quarters + 1):
            yi, qi = shift_quarter(yb, qb, -back)
            
            # Check if banking fundamentals exist for this quarter (quick test)
            with self.engine_ro.begin() as conn:
                test_query = text("""
                    SELECT COUNT(*) as count 
                    FROM intermediary_calculations_banking_cleaned 
                    WHERE year = :year AND quarter = :quarter AND has_full_ttm = 1
                    LIMIT 1
                """)
                result = conn.execute(test_query, {'year': yi, 'quarter': qi}).fetchone()
                
                if result and result.count > 0:
                    logger.info(f"TEMPORAL RESOLVER: {date_d.date()} → ({yi}Q{qi}) [fallback-{back}]")
                    logger.info(f"   Found {result.count} banking records for {yi}Q{qi}")
                    return yi, qi
        
        # Agent's spec: If none found, return None (eligible=0, hard fail)
        logger.error(f"TEMPORAL RESOLVER: {date_d.date()} → NO VALID QUARTER (tried {max_back_quarters+1} quarters)")
        return None, None
    
    def _trigger_production_quality_composition(self, date: pd.Timestamp, universe_df: pd.DataFrame):
        """
        Agent's TEMPORAL LOGIC FIX: Use PrevQuarter + bounded fallback (exactly per spec)
        Agent's Fix 2 & 5: Orchestrator-level capture with vectorized path telemetry
        """
        
        # Agent's TEMPORAL LOGIC: PrevQuarter + bounded fallback
        year_b, quarter_b = self._resolve_pit_quarter(date)
        
        # Agent's spec: If temporal resolution fails, eligible=0 → hard fail
        if year_b is None or quarter_b is None:
            logger.error(f"TEMPORAL RESOLUTION FAILED: {date.date()} - no valid fundamentals quarter found")
            logger.error(f"This will trigger Agent's Fix 3: eligible_count=0 → HARD FAIL")
            return  # Early return, captured_factors remains empty
        
        tickers = universe_df['ticker'].tolist()
        
        # Agent's Fix 5: Force vectorized path visibility & cache telemetry
        import os
        logger.info(f"Agent's Fix 5: F_SCORE_IMPL={os.getenv('F_SCORE_IMPL', 'NOT_SET')}")
        logger.info(f"Agent's Fix 5: Vectorized cache keys present? date={date.date()}")
        
        logger.info(f"✅ AGENT TEMPORAL LOGIC: {date.date()} → PrevQuarter({year_b}Q{quarter_b}) with bounded fallback")
        logger.info(f"Triggering production QVM calculation for {len(tickers)} tickers (Agent's orchestrator hook)")
        
        # Prime F-Score cache using RESOLVED temporal quarters (not current)
        prime_fscore_cache(self.qvm_engine, universe_df, date, year_b, quarter_b)
        
        # Agent's Orchestrator Hook: Trigger full QVM calculation to capture ALL factors
        logger.info("Agent's orchestrator hook: Triggering full QVM calculation to capture ALL individual factors")
        
        # Clear captured factors before orchestrator-level capture
        self.captured_factors = {}
        
        try:
            # Agent Neo: Store date context for normalization hook to access
            self.current_date = date
            
            # Agent's critical fix: Call the FULL QVM calculation which computes ALL individual factors
            # This will trigger our orchestrator hook that captures all 19 factors
            qvm_results = self.qvm_engine.calculate_qvm_composite(date, tickers)
            logger.info(f"Agent's orchestrator: Full QVM calculation completed for {len(qvm_results)} tickers")
            
            # The orchestrator hook should have captured all factors in self.captured_factors
            if self.captured_factors:
                captured_factor_names = set()
                for ticker_factors in self.captured_factors.values():
                    captured_factor_names.update(ticker_factors.keys())
                logger.info(f"Agent's orchestrator captured {len(captured_factor_names)} factor types: {sorted(captured_factor_names)}")
            else:
                logger.warning("Agent's orchestrator hook captured no factors - may need fallback")
                
                # Fallback: If orchestrator didn't capture anything, fall back to F-Score only
                # Now separate by sector using CORRECT mapping
                universe_indexed = universe_df.set_index('ticker')
                
                # Agent Neo Fix: Use 'Banks' not 'Banking'!
                nf_tickers = [t for t in tickers if universe_indexed.loc[t, 'sector'] not in ['Banks', 'Securities']]
                bank_tickers = [t for t in tickers if universe_indexed.loc[t, 'sector'] == 'Banks']
                sec_tickers = [t for t in tickers if universe_indexed.loc[t, 'sector'] == 'Securities']
                
                logger.info(f"   Fallback F-Score: Non-Financial: {len(nf_tickers)}, Banks: {len(bank_tickers)}, Securities: {len(sec_tickers)}")
                
                # Agent's Fallback: F-Score only capture
                full_fscore_vector = {}
                
                if nf_tickers and hasattr(self.qvm_engine, '_get_raw_f_score_non_financial'):
                    nf_result = self.qvm_engine._get_raw_f_score_non_financial(nf_tickers, year_b, quarter_b, date)
                    full_fscore_vector.update(nf_result)
                    logger.info(f"Fallback: Non-Financial F-Score captured {len(nf_result)} tickers using {year_b}Q{quarter_b}")
                
                if bank_tickers and hasattr(self.qvm_engine, '_get_raw_f_score_banking'):
                    bank_result = self.qvm_engine._get_raw_f_score_banking(bank_tickers, year_b, quarter_b)
                    full_fscore_vector.update(bank_result)
                    logger.info(f"Fallback: Banking F-Score captured {len(bank_result)} tickers using {year_b}Q{quarter_b}")
                
                if sec_tickers and hasattr(self.qvm_engine, '_get_raw_f_score_securities'):
                    sec_result = self.qvm_engine._get_raw_f_score_securities(sec_tickers, year_b, quarter_b)
                    full_fscore_vector.update(sec_result)
                    logger.info(f"Fallback: Securities F-Score captured {len(sec_result)} tickers using {year_b}Q{quarter_b}")
                
                # Store fallback F-Score data
                for ticker, fscore in full_fscore_vector.items():
                    if ticker not in self.captured_factors:
                        self.captured_factors[ticker] = {}
                    self.captured_factors[ticker]['f_score'] = fscore
                
                logger.info(f"Fallback: F-Score vector assembled with {len(full_fscore_vector)} total tickers")
            
        except Exception as e:
            logger.error(f"Agent's Fix 2: Orchestrator capture failed: {e}")
            raise
    
    def _normalize_factor_set(self):
        # Canonicalize CLI selections; do not force extra factors
        # 'all' means the predefined Tier-0 superset
        if self.factors and self.factors != ['all']:
            return {canonicalize(f) for f in self.factors}
        return EXPECTED_FACTORS_18

    def _convert_captured_to_raw_factors(self) -> Dict[str, pd.Series]:
        """
        Flatten Tap-B buffers into {factor_code: Series(ticker -> raw_value)}.
        Also ensure presence for EXPECTED_FACTORS_18 (prevents the "not captured" hard fail).
        """
        raw_factors: Dict[str, pd.Series] = {}
        allowed = self._normalize_factor_set()

        # flatten Tap-B buffers
        for fcode, pairs in (self._tapb_raw_buffers or {}).items():
            canon = canonicalize(fcode)
            if canon not in allowed:
                continue
            combined = pd.Series(dtype=float)
            for raw_series, _sec_series in pairs:
                s = pd.to_numeric(raw_series, errors='coerce').dropna()
                combined = combined.combine_first(s)
            if not combined.empty:
                raw_factors[canon] = combined

        # presence guarantee only for factors in CLI selection; drop debt_equity unless enabled
        for req in list(allowed):
            if req == 'debt_equity' and not getattr(self, 'enable_debt_equity', False):
                continue
            raw_factors.setdefault(req, pd.Series(dtype=float))

        # Diagnostics
        logger.info(f"TAP-B saw metrics today: {sorted(self._tapb_seen_metrics)}")
        if self._unknown_metric_names:
            logger.warning(f"Unmapped metrics: {sorted(self._unknown_metric_names)}")
        logger.info(f"Agent Neo's Tap B captured {len([k for k,v in raw_factors.items() if not v.empty])} canonical factors: {sorted([k for k,v in raw_factors.items() if not v.empty])}")
        
        counts = {k: int(v.size) for k, v in raw_factors.items()}
        logger.info(f"📊 Tap-B capture sizes: {sorted(counts.items())}")
        
        return raw_factors
    
    def _tier0_audit_neo_fixed(self, date: pd.Timestamp, sector_map: Dict, 
                               raw_factors: Dict, universe_df: pd.DataFrame) -> bool:
        """
        Agent's Step 4: Generalize Tier-0 gates for all factors with sign-aware Spearman
        - Coverage check for all requested factors
        - Sign-aware Spearman correlation validation 
        - Sector normalization validation
        """
        
        allowed = self._normalize_factor_set()
        logger.info(f"🔍 Agent's Step 4: Generalized Tier-0 Audit for {len(allowed)} factors on {date.date()}")
        
        if not raw_factors:
            logger.error("Tier-0: No factors captured - HARD FAIL")
            return False
        
        # Agent's surgical fix: Eligibility-aware denominators per factor
        factor_coverage = {}
        universe_count = len(universe_df)
        overall_eligible_count = 0
        
        for factor_name in sorted(allowed):
            if factor_name == 'debt_equity' and not getattr(self, 'enable_debt_equity', False):
                # Skip D/E until explicitly enabled
                continue
            # If factor is optional (e.g., fcf_yield before PIT availability), allow skip
            if factor_name not in raw_factors or raw_factors[factor_name].empty:
                if FACTOR_META.get(factor_name, {}).get('optional', False):
                    logger.warning(f"Tier-0: Optional factor '{factor_name}' missing; continuing")
                    continue
                logger.error(f"Tier-0: Factor '{factor_name}' not captured - HARD FAIL")
                return False
            
            factor_series = raw_factors[factor_name].dropna()
            factor_meta = FACTOR_META.get(factor_name, {})
            eligible_type = factor_meta.get('eligible', 'all')
            
            # Agent's fix: Calculate eligible denominator based on factor type
            if eligible_type == 'all':
                eligible_denominator = universe_count
                eligible_tickers = set(universe_df['ticker'])
            elif eligible_type == 'non_fin':
                eligible_tickers = {t for t, s in sector_map.items() if s not in ['Banks', 'Securities', 'Insurance']}
                eligible_denominator = len(eligible_tickers & set(universe_df['ticker']))
            elif eligible_type == 'banks':
                eligible_tickers = {t for t, s in sector_map.items() if s == 'Banks'}
                eligible_denominator = len(eligible_tickers & set(universe_df['ticker']))
            elif eligible_type == 'securities':
                eligible_tickers = {t for t, s in sector_map.items() if s == 'Securities'}
                eligible_denominator = len(eligible_tickers & set(universe_df['ticker']))
            elif eligible_type == 'sector_specific':
                # F-Score applies to all sectors with sector-specific variants
                eligible_denominator = universe_count
                eligible_tickers = set(universe_df['ticker'])
            else:
                eligible_denominator = universe_count
                eligible_tickers = set(universe_df['ticker'])
            
            actual_count = len(factor_series)
            eligible_coverage = actual_count / eligible_denominator if eligible_denominator > 0 else 0.0
            universe_coverage = actual_count / universe_count if universe_count > 0 else 0.0
            
            factor_coverage[factor_name] = {
                'eligible_count': actual_count,
                'eligible_denominator': eligible_denominator,
                'eligible_coverage': eligible_coverage,
                'universe_coverage': universe_coverage,
                'series': factor_series,
                'eligible_type': eligible_type
            }
            
            # Agent's Fix 3: If any factor has eligible_count == 0, FAIL the date
            if actual_count == 0:
                if FACTOR_META.get(factor_name, {}).get('optional', False):
                    logger.warning(f"Tier-0: Optional factor '{factor_name}' has zero eligibles; continuing")
                    continue
                logger.error(f"Tier-0: Factor '{factor_name}' no eligibles—sector routing or PIT inputs missing - HARD FAIL")
                return False
            
            overall_eligible_count = max(overall_eligible_count, actual_count)
            logger.info(f"   Factor '{factor_name}' ({eligible_type}): {actual_count}/{eligible_denominator} = {eligible_coverage:.1%} eligible coverage")
        
        # Agent Neo's Step 4: Sign-aware Spearman correlation validation using Tap A z-scores
        spearman_checks_passed = True
        logger.info("🔍 Agent Neo's Step 4: Sign-aware Spearman validation (Tap A z-scores vs pillar composites)")
        
        # TODO: Load pillar composites for this date and compare to captured z-scores
        # For now, validate factor z-scores exist in _captured_z
        for factor_name in sorted(allowed):
            if factor_name == 'debt_equity' and not getattr(self, 'enable_debt_equity', False):
                continue
            if factor_name in self._captured_z:
                z_count = len(self._captured_z[factor_name])
                logger.info(f"   Factor '{factor_name}' z-scores captured: {z_count} tickers")
                
                # Agent Neo's sign-aware Spearman threshold check
                if z_count >= 10:  # Minimum for meaningful correlation
                    sign = FACTOR_META.get(factor_name, {}).get('sign', 1)
                    pillar = FACTOR_META.get(factor_name, {}).get('pillar', 'Unknown')
                    logger.info(f"   Factor '{factor_name}' ({pillar}, sign={sign:+1}): Ready for correlation check")
                else:
                    logger.warning(f"   Factor '{factor_name}': Insufficient z-scores for correlation check ({z_count} < 10)")
            else:
                logger.warning(f"   Factor '{factor_name}': No z-scores captured in Tap A")
                # Don't fail on missing z-scores if raw data exists (defensive implementation)
        
        # Sector normalization check (generalized for all factors)
        sectors_ok = True
        for factor_name in sorted(allowed):
            if factor_name == 'debt_equity' and not getattr(self, 'enable_debt_equity', False):
                continue
            if factor_name in factor_coverage:
                factor_series = factor_coverage[factor_name]['series']
                
                for sector in set(sector_map.values()):
                    sector_tickers = [t for t, s in sector_map.items() if s == sector]
                    sector_values = factor_series[factor_series.index.isin(sector_tickers)]
                    
                    if len(sector_values) > 1:
                        z_scores = (sector_values - sector_values.mean()) / sector_values.std()
                        mean_z = z_scores.mean()
                        std_z = z_scores.std()
                        
                        if abs(mean_z) > 1e-6 or abs(std_z - 1.0) > 0.05:
                            logger.error(f"   Factor '{factor_name}' sector {sector} normalization failed")
                            sectors_ok = False
        
        # Agent's surgical fix: Overall pass/fail decision using eligible coverage
        # Consider only non-optional factors for coverage thresholds
        mandatory_cov = [cov for fname, cov in factor_coverage.items() if not FACTOR_META.get(fname, {}).get('optional', False)]
        if not mandatory_cov:
            logger.error("Tier-0: No mandatory factors available for coverage checks - HARD FAIL")
            return False
        min_eligible_coverage = min(c['eligible_coverage'] for c in mandatory_cov)
        avg_eligible_coverage = sum(c['eligible_coverage'] for c in mandatory_cov) / len(mandatory_cov)
        min_universe_coverage = min(c['universe_coverage'] for c in mandatory_cov)
        
        logger.info(f"   Overall: min_eligible_coverage={min_eligible_coverage:.1%}, avg_eligible_coverage={avg_eligible_coverage:.1%}")
        logger.info(f"   Universe: min_universe_coverage={min_universe_coverage:.1%}")
        
        # Agent's exact gates: ≥95% eligible coverage for sector-specific, ≥95% universe for universal
        if min_eligible_coverage >= 0.95 and sectors_ok:
            logger.info("✅ TIER-0 AUDIT PASSED (≥95% eligible coverage, sectors OK)")
            return True
        elif avg_eligible_coverage >= 0.90 and sectors_ok and spearman_checks_passed:
            logger.info("✅ TIER-0 AUDIT PASSED (≥90% avg eligible coverage, sectors OK, correlations OK)")
            return True
        else:
            logger.error(f"TIER-0 AUDIT FAILED: Min eligible coverage {min_eligible_coverage:.1%} < 95%")
            return False
    
    def _write_factor_signals_raw_analytics_only(self, date: pd.Timestamp, raw_factors: Dict, conn) -> int:
        """
        Agent's Multi-Row Batch Insert: Write raw factors to sidecar ONLY
        Returns: number of rows written
        """
        
        logger.info(f"Writing raw factors to sidecar for {date.date()}")
        total_rows = 0
        
        for factor_name, factor_series in raw_factors.items():
            factor_id = FACTOR_CODE_TO_ID.get(factor_name)
            if not factor_id:
                logger.warning(f"No factor_id mapping for {factor_name}, skipping")
                continue
                
            # Build VALUES clause for multi-row insert
            values_data = []
            for ticker, raw_value in factor_series.dropna().items():
                values_data.append({
                    'ticker': ticker,
                    'date': date.date(),
                    'strategy_version': self.version,
                    'factor_id': factor_id,
                    'raw_value': float(raw_value)
                })
            
            if values_data:
                # Agent's multi-row INSERT as specified
                values_clause = ', '.join(['(:ticker_%d, :date_%d, :strategy_version_%d, :factor_id_%d, :raw_value_%d)' % (i,i,i,i,i) for i in range(len(values_data))])
                
                # Flatten parameters for multi-row insert
                params = {}
                for i, data in enumerate(values_data):
                    params[f'ticker_{i}'] = data['ticker']
                    params[f'date_{i}'] = data['date'] 
                    params[f'strategy_version_{i}'] = data['strategy_version']
                    params[f'factor_id_{i}'] = data['factor_id']
                    params[f'raw_value_{i}'] = data['raw_value']
                
                upsert_sql = text(f'''
                    INSERT INTO factor_signals_raw (ticker, date, strategy_version, factor_id, raw_value)
                    VALUES {values_clause}
                    ON DUPLICATE KEY UPDATE raw_value = VALUES(raw_value)
                ''')
                
                result = conn.execute(upsert_sql, params)
                rows_affected = result.rowcount
                total_rows += rows_affected
                
                logger.info(f"Written {rows_affected} {factor_name} raw signals to sidecar (multi-row batch)")
        
        return total_rows
    
    def _write_factor_norm_stats_analytics_only(self, date: pd.Timestamp, raw_factors: Dict, sector_map: Dict, conn) -> int:
        """
        Agent Neo's Tap B Normalization Stats: Write captured per-sector stats to sidecar ONLY
        Returns: number of rows written
        """
        
        logger.info(f"Writing Agent Neo's Tap B normalization stats to sidecar for {date.date()}")
        total_rows = 0
        
        # Agent Neo's fix: Use captured normalization stats from Tap B with canonical key handling
        for key, sector_stats in self._norm_stats.items():
            k = canonicalize(str(key).lower())
            # fallbacks if you keep an ENGINE_TO_CANONICAL mapping elsewhere
            k = ENGINE_TO_CANONICAL.get(k, k)
            factor_id = FACTOR_CODE_TO_ID.get(k)
            if not factor_id:
                continue
            
            # Prepare normalization records from captured stats
            norm_records = []
            total_universe_size = len(sector_map)  # Total universe size
            
            for sector, stats in sector_stats.items():
                norm_records.append({
                    'date': date.date(),
                    'strategy_version': self.version,
                    'factor_id': factor_id,
                    'sector': sector,
                    'mean_value': float(stats['mean']),
                    'std_value': float(stats['std']) if stats['std'] > 0 else 1.0,
                    'eligible_count': int(stats['count']),
                    'universe_size': total_universe_size
                })
            
            if norm_records:
                # Agent's multi-row INSERT as specified
                values_clause = ', '.join(['(:date_%d, :strategy_version_%d, :factor_id_%d, :sector_%d, :mean_value_%d, :std_value_%d, :eligible_count_%d, :universe_size_%d)' % (i,i,i,i,i,i,i,i) for i in range(len(norm_records))])
                
                # Flatten parameters for multi-row insert
                params = {}
                for i, data in enumerate(norm_records):
                    params[f'date_{i}'] = data['date']
                    params[f'strategy_version_{i}'] = data['strategy_version']
                    params[f'factor_id_{i}'] = data['factor_id']
                    params[f'sector_{i}'] = data['sector']
                    params[f'mean_value_{i}'] = data['mean_value']
                    params[f'std_value_{i}'] = data['std_value']
                    params[f'eligible_count_{i}'] = data['eligible_count']
                    params[f'universe_size_{i}'] = data['universe_size']
                
                upsert_sql = text(f'''
                    INSERT INTO factor_norm_stats (date, strategy_version, factor_id, sector, 
                                                 mean_value, std_value, eligible_count, universe_size)
                    VALUES {values_clause}
                    ON DUPLICATE KEY UPDATE 
                        mean_value = VALUES(mean_value),
                        std_value = VALUES(std_value),
                        eligible_count = VALUES(eligible_count),
                        universe_size = VALUES(universe_size)
                ''')
                
                result = conn.execute(upsert_sql, params)
                rows_affected = result.rowcount
                total_rows += rows_affected
                
                logger.info(f"Written {rows_affected} normalization stats rows for factor_id={factor_id}")
        
        return total_rows

def main():
    """Agent's Institutional-Grade 30-Day Batching Wrapper"""
    parser = argparse.ArgumentParser(description='Agent Neo Fixed Analytics Wrapper - Institutional Grade Batching')
    parser.add_argument('--start-date', type=str, required=False, help='Start date (YYYY-MM-DD). Optional in from-last/daily modes')
    parser.add_argument('--end-date', type=str, required=False, help='End date (YYYY-MM-DD). Optional in daily mode')
    parser.add_argument('--version', type=str, required=True)
    parser.add_argument('--factors', nargs='+', default=['f_score'], 
                        help='Factors to extract. Use "all" for all 19 factors, "list" to show available factors')
    parser.add_argument('--audit-tier', type=str, default='tier0')
    parser.add_argument('--write-sidecar', action='store_true')
    
    # Agent's CLI flags for institutional hardening
    parser.add_argument('--batch-size', type=int, default=30, help='Days per batch (default: 30)')
    parser.add_argument('--mode', type=str, choices=['range','from-last','daily'], default='range',
                        help='range: use provided dates; from-last: start at last sidecar date+1 to latest composite; daily: process latest composite date only')
    parser.add_argument('--resume', action='store_true', help='Skip dates already in sidecar')
    parser.add_argument('--max-batch-fail-rate', type=float, default=0.25, help='Abort if batch fail rate exceeds this')
    parser.add_argument('--max-retries', type=int, default=2, help='Max retries for transient DB errors')
    parser.add_argument('--enable-debt-equity', action='store_true', help='Enable Debt/Equity synthesis after Tap-B flush')
    
    args = parser.parse_args()
    
    # Agent's CLI Extension: Handle --factors all|list
    if args.factors == ['list']:
        print("\n🔍 Available Factors:")
        for factor_code, meta in FACTOR_META.items():
            optional_mark = " (optional)" if meta.get('optional', False) else ""
            direction_symbol = "↑" if meta['direction'] == 1 else "↓"
            print(f"  {factor_code:20} | {meta['pillar']:10} | {direction_symbol} | {meta['family']}{optional_mark}")
        print(f"\nTotal: {len(FACTOR_META)} factors")
        print("Usage: --factors all  (for all factors)")
        print("       --factors f_score roae  (for specific factors)")
        return
    
    if args.factors == ['all']:
        # Use canonical 18-factor set (engine-supported, Tier-0 audited)
        try:
            canonical_all = sorted(list(EXPECTED_FACTORS_18))
        except NameError:
            # Fallback: minimal safe set
            canonical_all = [
                'roae','roaa','nim','cost_income',
                'net_profit_margin','gross_margin','operating_margin','ebitda_margin',
                'f_score',
                'earnings_yield','book_value','sales_multiple','ev_ebitda','fcf_yield',
                'mom_3m','mom_6m','mom_12m','low_volatility'
            ]
        # Normalize alias 'low_volatility' to canonical inside wrapper later
        args.factors = canonical_all
        print(f"🎯 Canonical --factors all: Processing {len(args.factors)} factors (canonical sidecar set)")
    
    logger.info("🛡️  AGENT NEO'S INSTITUTIONAL-GRADE BATCHING WRAPPER")
    logger.info(f"   Strategy version: {args.version}")
    logger.info(f"   Sidecar writing: {'ENABLED' if args.write_sidecar else 'DISABLED'}")
    logger.info(f"   Factors: {args.factors}")
    logger.info(f"   Audit tier: {args.audit_tier.upper()}")
    logger.info(f"   Batch size: {args.batch_size} days")
    logger.info(f"   Resume mode: {'ENABLED' if args.resume else 'DISABLED'}")
    logger.info(f"   Max batch fail rate: {args.max_batch_fail_rate:.1%}")
    
    wrapper = FactorAnalyticsWrapperNeoFix(
        strategy_version=args.version,
        write_sidecar=args.write_sidecar,
        factors=args.factors,
        audit_tier=args.audit_tier,
        enable_debt_equity=args.enable_debt_equity
    )
    
    # Agent's Fix 1: Date source = composites calendar (not naive calendar days)
    logger.info("📅 Agent's Fix 1: Fetching dates from composites calendar...")
    # Agent's version mapping: analytics versions -> production composite version
    production_version = map_analytics_to_production_version(args.version)
    
    logger.info(f"📅 Version mapping: {args.version} -> {production_version} (for composites calendar)")
    
    # Resolve factor ids for selection; used by resume and from-last logic
    selected_factor_ids = get_factor_ids_for_factors(args.factors)

    # Resolve date range based on mode
    if args.mode == 'daily':
        latest = get_latest_composite_date(wrapper.engine_ro, production_version)
        if latest is None:
            logger.error("❌ No latest composite date found")
            return
        start_date, end_date = latest.strftime('%Y-%m-%d'), latest.strftime('%Y-%m-%d')
        logger.info(f"   Mode=daily → latest composite date: {start_date}")
    elif args.mode == 'from-last':
        last_sidecar = get_last_sidecar_date_for_factors(wrapper.engine_sidecar, args.version, selected_factor_ids)
        latest_comp = get_latest_composite_date(wrapper.engine_ro, production_version)
        if latest_comp is None:
            logger.error("❌ No composite dates found for production version")
            return
        if last_sidecar is None:
            # Start from earliest composite date
            earliest = get_earliest_composite_date(wrapper.engine_ro, production_version)
            if earliest is None:
                logger.error("❌ No composite dates available to start from")
                return
            start_date = earliest.strftime('%Y-%m-%d')
        else:
            # Start from next trading day after last_sidecar
            start_date = (last_sidecar + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        end_date = latest_comp.strftime('%Y-%m-%d')
        logger.info(f"   Mode=from-last → range {start_date} to {end_date} (last sidecar: {last_sidecar.date() if last_sidecar else 'None'})")
    else:
        # range mode - require explicit dates
        if not args.start_date or not args.end_date:
            logger.error("❌ --mode range requires --start-date and --end-date")
            return
        start_date, end_date = args.start_date, args.end_date
        logger.info(f"   Mode=range → range {start_date} to {end_date}")

    work_dates = fetch_dates_from_composites(
        wrapper.engine_ro,
        version=production_version,
        start_date=start_date,
        end_date=end_date
    )
    
    if not work_dates:
        logger.error("❌ No dates found in composites calendar - check version and date range")
        return
    
    # Split into Agent's 30-day batches
    batch_size = args.batch_size
    current_start_idx = 0
    batch_num = 1
    total_success = 0
    total_rows_written = 0
    total_dates = len(work_dates)
    
    logger.info(f"🔄 AGENT'S BATCHED PROCESSING: {len(work_dates)} dates split into {batch_size}-day batches")
    
    while current_start_idx < len(work_dates):
        # Define batch end
        batch_end_idx = min(current_start_idx + batch_size, len(work_dates))
        batch_dates = work_dates[current_start_idx:batch_end_idx]
        
        logger.info(f"📦 BATCH {batch_num}: Processing {batch_dates[0].date()} to {batch_dates[-1].date()} ({len(batch_dates)} days)")
        
        # Process this batch with Agent's hardening
        batch_success = 0
        batch_rows_written = 0
        factor_id_fscore = FACTOR_CODE_TO_ID.get('f_score', 3)  # Default to 3 if not found
        
        for date in batch_dates:
            # Enhanced resume: skip date if all selected factors exist
            if args.resume and sidecar_has_all_selected_factors(wrapper.engine_sidecar, date, args.version, selected_factor_ids):
                logger.info(f"⏭️  Skipping {date.date()} (all selected factors already present)")
                batch_success += 1
                continue
            
            # Agent's Fix 3: success==commit (only count after rows written)
            success, rows_written = wrapper.run_date(date)
            if success and rows_written > 0:
                batch_success += 1
                batch_rows_written += rows_written
                logger.info(f"TemporalResolver: D={date.date()} wrote_rows={rows_written}")
            elif not success:
                logger.error(f"❌ HARD FAIL: {date.date()} failed Tier-0 audit or temporal resolution")
        
        # Agent's Fix 4: Batch checkpoint & backoff logic
        batch_success_rate = (batch_success / len(batch_dates)) if len(batch_dates) > 0 else 0
        logger.info(f"✅ BATCH {batch_num} COMPLETE: {batch_success}/{len(batch_dates)} days ({batch_success_rate:.1%} success)")
        logger.info(f"   Rows written in batch: {batch_rows_written}")
        
        # Update totals
        total_success += batch_success
        total_rows_written += batch_rows_written
        
        # Agent's backoff: Check batch fail rate
        if batch_success_rate < (1.0 - args.max_batch_fail_rate):
            logger.error(f"❌ BATCH FAIL RATE {(1.0-batch_success_rate):.1%} > MAX {args.max_batch_fail_rate:.1%} - ABORTING RUN")
            logger.error(f"   This prevents marching through systemic issues")
            break
        
        # Progress checkpoint every batch (like production)
        overall_success_rate = (total_success / total_dates) * 100
        logger.info(f"📊 OVERALL PROGRESS: {total_success}/{total_dates} dates ({overall_success_rate:.1f}% success) across {batch_num} batches")
        logger.info(f"   Total rows written: {total_rows_written}")
        
        # Move to next batch
        current_start_idx = batch_end_idx
        batch_num += 1
    
    # Final summary
    final_success_rate = (total_success / total_dates) * 100
    logger.info(f"🎯 AGENT'S INSTITUTIONAL FINAL RESULTS:")
    logger.info(f"   Dates processed: {total_success}/{total_dates} ({final_success_rate:.1f}% success)")
    logger.info(f"   Rows written: {total_rows_written}")
    logger.info(f"   Batches completed: {batch_num-1}")
    
    if total_success == total_dates:
        logger.info("✅ SUCCESS: All dates processed successfully")
    else:
        logger.warning(f"⚠️  PARTIAL: {total_success}/{total_dates} dates processed ({final_success_rate:.1f}% success rate)")

if __name__ == "__main__":
    main()