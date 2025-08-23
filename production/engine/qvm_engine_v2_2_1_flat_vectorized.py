"""
QVM v2.2.1 Flat — Vectorized F-Score service
============================================
Purpose:
- Provide fast, single-pass, per-sector vectorized F-Score calculators aligned with v2.2.1 timing
- Include a per-date in-memory cache and cache priming helpers to ensure ≤3 queries/sector/date
- Normalize sector labels for robust universe ingress

Design notes:
- Timing: Financials use lagged quarter from v2.2.1 engine; shares use analysis_date and analysis_date - 1y
- Banking fallbacks: Robust alias sequence with AvgCustomerDeposits backfill when CustomerDeposits missing
- Securities: Use cleaned intermediary table with y vs y-1 comparisons

This module is deliberately light-weight and can be installed into an engine via installers, 
or called directly from the engine's F-Score path.
"""

import types
import pandas as pd
import numpy as np
from sqlalchemy import text
from typing import Dict, List


# -----------------------
# Cache (per-engine, per-date)
# -----------------------
def _ensure_cache(engine):
    if not hasattr(engine, "_fscore_cache_221"):
        engine._fscore_cache_221 = {}
    return engine._fscore_cache_221


# -----------------------
# Sector label normalization
# -----------------------
_SECTOR_NORMALIZATION_MAP = {
    # Banks / Banking
    "bank": "Banking", "banks": "Banking", "banking": "Banking",
    # Securities
    "securities": "Securities", "brokerage": "Securities",
    # Insurance (treated as financial, excluded from NF path)
    "insurance": "Insurance",
}


def normalize_sector_labels_221(df: pd.DataFrame, sector_col: str = "sector") -> pd.DataFrame:
    """
    Normalize sector labels to canonical forms expected by the engine.
    Canonical set: {"Banking", "Securities", "Insurance", <other sectors as-is>}.
    """
    if df is None or df.empty or sector_col not in df.columns:
        return df
    out = df.copy()
    out[sector_col] = (
        out[sector_col]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(_SECTOR_NORMALIZATION_MAP)
        .fillna(df[sector_col])
    )
    return out


# -----------------------
# Bulk fetch helpers
# -----------------------
def _fetch_nonfin_intermediary(conn, tickers: List[str], y: int, q: int) -> pd.DataFrame:
    sql = text(
        """
        SELECT ticker, NetProfit_TTM, AvgTotalAssets, NetCFO_TTM, Revenue_TTM, COGS_TTM
        FROM intermediary_calculations_enhanced
        WHERE year=:y AND quarter=:q AND has_full_ttm=1 AND ticker IN :tickers
        """
    )
    return pd.read_sql(sql, conn, params={"y": y, "q": q, "tickers": tuple(tickers)})


def _fetch_nf_bal_snapshot(conn, tickers: List[str], y: int, q: int) -> pd.DataFrame:
    try:
        sql = text(
            """
            SELECT ticker, CurrentAssets, CurrentLiabilities,
                   COALESCE(ShortTermDebt,0)+COALESCE(LongTermDebt,0) AS TotalDebt
            FROM comprehensive_fundamentals_snapshot
            WHERE year=:y AND quarter=:q AND ticker IN :tickers
            """
        )
        return pd.read_sql(sql, conn, params={"y": y, "q": q, "tickers": tuple(tickers)})
    except Exception:
        sql = text(
            """
            SELECT ticker, CurrentAssets, CurrentLiabilities,
                   COALESCE(ShortTermDebt,0)+COALESCE(LongTermDebt,0) AS TotalDebt
            FROM v_comprehensive_fundamental_items
            WHERE year=:y AND quarter=:q AND ticker IN :tickers
            """
        )
        return pd.read_sql(sql, conn, params={"y": y, "q": q, "tickers": tuple(tickers)})


def _fetch_last_shares(conn, tickers: List[str], asof) -> pd.DataFrame:
    sql = text(
        """
        WITH last_dates AS (
          SELECT ticker, MAX(trading_date) AS lastdate
          FROM vcsc_daily_data_complete
          WHERE ticker IN :tickers AND trading_date <= :asof
          GROUP BY ticker
        )
        SELECT v.ticker, v.total_shares
        FROM vcsc_daily_data_complete v
        JOIN last_dates d ON v.ticker=d.ticker AND v.trading_date=d.lastdate
        WHERE v.total_shares IS NOT NULL AND v.total_shares>0
        """
    )
    return pd.read_sql(sql, conn, params={"tickers": tuple(tickers), "asof": pd.Timestamp(asof).date()})


# -----------------------
# Vectorized calculators (aligned with v2.2.1 timing)
# -----------------------
def compute_nf_vectorized_221(engine, tickers: List[str], lagged_year: int, lagged_quarter: int, analysis_date) -> Dict[str, int]:
    cache = _ensure_cache(engine)
    key = ("nf", lagged_year, lagged_quarter, pd.Timestamp(analysis_date).date(), tuple(sorted(tickers)))
    if key in cache:
        return cache[key]

    prev_year = lagged_year - 1
    with engine.engine.begin() as conn:
        cur_ice = _fetch_nonfin_intermediary(conn, tickers, lagged_year, lagged_quarter)
        prv_ice = _fetch_nonfin_intermediary(conn, tickers, prev_year, lagged_quarter)
        cur_bal = _fetch_nf_bal_snapshot(conn, tickers, lagged_year, lagged_quarter)
        prv_bal = _fetch_nf_bal_snapshot(conn, tickers, prev_year, lagged_quarter)
        cur_sh = _fetch_last_shares(conn, tickers, analysis_date)
        prv_sh = _fetch_last_shares(conn, tickers, pd.Timestamp(analysis_date) - pd.DateOffset(years=1))

    cur = cur_ice.merge(cur_bal, on="ticker", how="left").merge(cur_sh, on="ticker", how="left").rename(columns={"total_shares": "shares"})
    prv = prv_ice.merge(prv_bal, on="ticker", how="left").merge(prv_sh, on="ticker", how="left").rename(columns={"total_shares": "shares_prev"})
    df = cur.merge(prv, on="ticker", suffixes=("", "_prev"), how="inner").copy()

    assets = df["AvgTotalAssets"].replace(0, np.nan)
    assets_prev = df["AvgTotalAssets_prev"].replace(0, np.nan)

    # Profitability
    roa = df["NetProfit_TTM"] / assets
    roa_prev = df["NetProfit_TTM_prev"] / assets_prev
    cfo = df["NetCFO_TTM"]
    accrual = (df["NetCFO_TTM"] > df["NetProfit_TTM"]).astype(int)

    # Leverage/Liquidity/Funding
    lev = (df["TotalDebt"] / assets).replace([np.inf, -np.inf], np.nan)
    lev_prev = (df["TotalDebt_prev"] / assets_prev).replace([np.inf, -np.inf], np.nan)
    cr = df["CurrentAssets"] / df["CurrentLiabilities"].replace(0, np.nan)
    cr_prev = df["CurrentAssets_prev"] / df["CurrentLiabilities_prev"].replace(0, np.nan)
    shares = df["shares"].replace(0, np.nan)
    shares_prev = df["shares_prev"].replace(0, np.nan)

    # Operating efficiency (Gross Margin delta & Asset Turnover delta)
    gm = (df["Revenue_TTM"] - df["COGS_TTM"]) / df["Revenue_TTM"].replace(0, np.nan)
    gm_prev = (df["Revenue_TTM_prev"] - df["COGS_TTM_prev"]) / df["Revenue_TTM_prev"].replace(0, np.nan)
    ato = df["Revenue_TTM"] / assets
    ato_prev = df["Revenue_TTM_prev"] / assets_prev

    score = (
        (roa > 0).astype(int) +
        (cfo > 0).astype(int) +
        (roa > roa_prev).astype(int) +
        accrual +
        (lev < lev_prev).astype(int) +
        (cr > cr_prev).astype(int) +
        (shares <= shares_prev * 1.01).astype(int) +
        (gm > gm_prev).astype(int) +
        (ato > ato_prev).astype(int)
    )

    res = dict(zip(df["ticker"], score.fillna(0).astype(int)))
    # Structured observability logs
    try:
        n = len(df)
        accrual_rate = float(accrual.fillna(0).mean()) if n else 0.0
        share_tol = ((shares <= shares_prev * 1.01) & shares.notna() & shares_prev.notna()).mean() if n else 0.0
        nan_assets = float(assets.isna().mean()) if n else 0.0
        engine.logger.info(
            "FScore221 NF: rows=%d, accrual_rate=%.3f, share_tol_rate=%.3f, assets_nan=%.3f, tables={%s,%s}",
            n, accrual_rate, share_tol, nan_assets,
            "intermediary_calculations_enhanced", "v_comprehensive_fundamental_items|snapshot"
        )
    except Exception:
        pass
    cache[key] = res
    return res


def compute_bank_vectorized_221(engine, tickers: List[str], lagged_year: int, lagged_quarter: int) -> Dict[str, int]:
    cache = _ensure_cache(engine)
    key = ("bank", lagged_year, lagged_quarter, tuple(sorted(tickers)))
    if key in cache:
        return cache[key]

    prev_year = lagged_year - 1
    with engine.engine.begin() as conn:
        sql = text(
            """
            SELECT ticker, NetProfit_TTM, AvgTotalAssets,
                   NII_TTM, AvgEarningAssets,
                   TotalOperatingIncome_TTM, OperatingExpenses_TTM,
                   AvgCustomerDeposits
            FROM intermediary_calculations_banking_cleaned
            WHERE year=:y AND quarter=:q AND has_full_ttm=1 AND ticker IN :tickers
            """
        )
        cur = pd.read_sql(sql, conn, params={"y": lagged_year, "q": lagged_quarter, "tickers": tuple(tickers)})
        prv = pd.read_sql(sql, conn, params={"y": prev_year, "q": lagged_quarter, "tickers": tuple(tickers)})

        # Prefer banking-specific comprehensive view; fallback to general table with proxies
        used_banking_view = True
        try:
            sqlb = text(
                """
                SELECT ticker,
                       COALESCE(ShareholdersEquity, TotalEquity, AvgTotalEquity) AS ShareholdersEquity,
                       COALESCE(CustomerDeposits, TotalLiabilities, AvgCustomerDeposits, 0) AS CustomerDeposits
                FROM v_comprehensive_fundamental_items_banking
                WHERE year=:y AND quarter=:q AND ticker IN :tickers
                """
            )
            curb = pd.read_sql(sqlb, conn, params={"y": lagged_year, "q": lagged_quarter, "tickers": tuple(tickers)})
            prvb = pd.read_sql(sqlb, conn, params={"y": prev_year, "q": lagged_quarter, "tickers": tuple(tickers)})
        except Exception:
            used_banking_view = False
            sqlb = text(
                """
                SELECT ticker,
                       COALESCE(TotalEquity, OwnersEquity) AS ShareholdersEquity,
                       COALESCE(TotalLiabilities, CurrentLiabilities, 0) AS CustomerDeposits
                FROM v_comprehensive_fundamental_items
                WHERE year=:y AND quarter=:q AND ticker IN :tickers
                """
            )
            curb = pd.read_sql(sqlb, conn, params={"y": lagged_year, "q": lagged_quarter, "tickers": tuple(tickers)})
            prvb = pd.read_sql(sqlb, conn, params={"y": prev_year, "q": lagged_quarter, "tickers": tuple(tickers)})

    cur = cur.merge(curb, on="ticker", how="left")
    prv = prv.merge(prvb, on="ticker", how="left", suffixes=("", "_prev"))
    df = cur.merge(prv, on="ticker", how="inner", suffixes=("", "_prev"))

    # Backfill CustomerDeposits with AvgCustomerDeposits when missing
    if "CustomerDeposits" not in df.columns and "AvgCustomerDeposits" in df.columns:
        df["CustomerDeposits"] = df["AvgCustomerDeposits"]
    if "CustomerDeposits_prev" not in df.columns and "AvgCustomerDeposits_prev" in df.columns:
        df["CustomerDeposits_prev"] = df["AvgCustomerDeposits_prev"]

    df["CustomerDeposits"] = df.get("CustomerDeposits", 0).fillna(0)
    df["CustomerDeposits_prev"] = df.get("CustomerDeposits_prev", 0).fillna(0)

    assets = df["AvgTotalAssets"].replace(0, np.nan)
    assets_prev = df.get("AvgTotalAssets_prev", df["AvgTotalAssets"]).replace(0, np.nan)

    roa = df["NetProfit_TTM"] / assets
    roa_prev = df.get("NetProfit_TTM_prev", df["NetProfit_TTM"]) / assets_prev

    nim = df["NII_TTM"] / df["AvgEarningAssets"].replace(0, np.nan)
    nim_prev = df.get("NII_TTM_prev", df["NII_TTM"]) / df.get("AvgEarningAssets_prev", df["AvgEarningAssets"]).replace(0, np.nan)

    cir = df["OperatingExpenses_TTM"] / df["TotalOperatingIncome_TTM"].replace(0, np.nan)
    cir_prev = df.get("OperatingExpenses_TTM_prev", df["OperatingExpenses_TTM"]) / df.get("TotalOperatingIncome_TTM_prev", df["TotalOperatingIncome_TTM"]).replace(0, np.nan)

    lev = df["CustomerDeposits"] / df["ShareholdersEquity"].replace(0, np.nan)
    lev_prev = df["CustomerDeposits_prev"] / df.get("ShareholdersEquity_prev", df["ShareholdersEquity"]).replace(0, np.nan)

    score = (
        (roa > 0).astype(int) +
        (roa > roa_prev).astype(int) +
        (nim > nim_prev).astype(int) +
        (cir < cir_prev).astype(int) +
        (lev < lev_prev).astype(int) +
        (df["NetProfit_TTM"] > df.get("NetProfit_TTM_prev", 0)).astype(int)
    )

    res = dict(zip(df["ticker"], score.fillna(0).astype(int)))
    # Observability logs
    try:
        n = len(df)
        used_backfill_now = int((df.get("AvgCustomerDeposits").notna() & df.get("CustomerDeposits").notna() & (df["CustomerDeposits"] == df["AvgCustomerDeposits"]).fillna(False)).sum()) if "AvgCustomerDeposits" in df else 0
        used_backfill_prev = int((df.get("AvgCustomerDeposits_prev").notna() & df.get("CustomerDeposits_prev").notna() & (df["CustomerDeposits_prev"] == df["AvgCustomerDeposits_prev"]).fillna(False)).sum()) if "AvgCustomerDeposits_prev" in df else 0
        engine.logger.info(
            "FScore221 BANK: rows=%d, view=%s, backfill_now=%d, backfill_prev=%d",
            n, "banking_view" if used_banking_view else "general_view", used_backfill_now, used_backfill_prev
        )
    except Exception:
        pass
    cache[key] = res
    return res


def compute_sec_vectorized_221(engine, tickers: List[str], lagged_year: int, lagged_quarter: int) -> Dict[str, int]:
    cache = _ensure_cache(engine)
    key = ("sec", lagged_year, lagged_quarter, tuple(sorted(tickers)))
    if key in cache:
        return cache[key]

    prev_year = lagged_year - 1
    with engine.engine.begin() as conn:
        sql = text(
            """
            SELECT ticker, TotalOperatingRevenue_TTM, NetProfit_TTM,
                   AvgTotalAssets, OperatingResult_TTM, OperatingExpenses_TTM
            FROM intermediary_calculations_securities_cleaned
            WHERE year=:y AND quarter=:q AND has_full_ttm=1 AND ticker IN :tickers
            """
        )
        cur = pd.read_sql(sql, conn, params={"y": lagged_year, "q": lagged_quarter, "tickers": tuple(tickers)})
        prv = pd.read_sql(sql, conn, params={"y": prev_year, "q": lagged_quarter, "tickers": tuple(tickers)})

    df = cur.merge(prv, on="ticker", how="inner", suffixes=("", "_prev"))
    assets = df["AvgTotalAssets"].replace(0, np.nan)
    assets_prev = df["AvgTotalAssets_prev"].replace(0, np.nan)
    roa = df["NetProfit_TTM"] / assets
    roa_prev = df["NetProfit_TTM_prev"] / assets_prev
    om = df["OperatingResult_TTM"] / df["TotalOperatingRevenue_TTM"].replace(0, np.nan)
    om_prev = df["OperatingResult_TTM_prev"] / df["TotalOperatingRevenue_TTM_prev"].replace(0, np.nan)
    cost = df["OperatingExpenses_TTM"] / df["TotalOperatingRevenue_TTM"].replace(0, np.nan)
    cost_prev = df["OperatingExpenses_TTM_prev"] / df["TotalOperatingRevenue_TTM_prev"].replace(0, np.nan)

    score = (
        (roa > 0).astype(int) +
        (df["OperatingResult_TTM"] > 0).astype(int) +
        (roa > roa_prev).astype(int) +
        (om > om_prev).astype(int) +
        (cost < cost_prev).astype(int)
    )
    res = dict(zip(df["ticker"], score.fillna(0).astype(int)))
    try:
        n = len(df)
        engine.logger.info("FScore221 SEC: rows=%d, table=%s", n, "intermediary_calculations_securities_cleaned")
    except Exception:
        pass
    cache[key] = res
    return res


# -----------------------
# Public install/warmup hooks
# -----------------------
def install_vectorized_fscore_221(engine):
    """
    Install vectorized F-Score methods onto the engine instance.
    Safe to call multiple times.
    """
    engine._get_raw_f_score_non_financial_221 = types.MethodType(
        lambda self, tickers, y, q, d: compute_nf_vectorized_221(self, tickers, y, q, d), engine
    )
    engine._get_raw_f_score_banking_221 = types.MethodType(
        lambda self, tickers, y, q: compute_bank_vectorized_221(self, tickers, y, q), engine
    )
    engine._get_raw_f_score_securities_221 = types.MethodType(
        lambda self, tickers, y, q: compute_sec_vectorized_221(self, tickers, y, q), engine
    )
    _ensure_cache(engine)
    return engine


def prime_fscore_cache_221(engine, universe_df: pd.DataFrame, analysis_date, lagged_year: int, lagged_quarter: int) -> None:
    """
    Warm the per-date cache with at most one round-trip per sector group.
    universe_df: DataFrame with columns [ticker, sector]. Sector labels are normalized internally.
    """
    if universe_df is None or universe_df.empty:
        return
    _ensure_cache(engine)

    uf = universe_df.copy()
    if "ticker" not in uf.columns or "sector" not in uf.columns:
        return
    uf = normalize_sector_labels_221(uf, "sector")

    tickers = uf["ticker"].tolist()
    sec_map = uf.set_index("ticker")["sector"].to_dict()

    nf = [t for t in tickers if sec_map.get(t) not in ("Banking", "Securities", "Insurance")]
    bank = [t for t in tickers if sec_map.get(t) == "Banking"]
    sec = [t for t in tickers if sec_map.get(t) == "Securities"]

    if nf:
        compute_nf_vectorized_221(engine, nf, lagged_year, lagged_quarter, analysis_date)
    if bank:
        compute_bank_vectorized_221(engine, bank, lagged_year, lagged_quarter)
    if sec:
        compute_sec_vectorized_221(engine, sec, lagged_year, lagged_quarter)


