# production/engine/qvm_engine_v2_1_1_flat_fscore_vectorized_neo_fix.py
"""
Agent Neo's Fixed Vectorized F-Score Implementation
Following Section 3D: "Replicate the orchestrator's alias map + fallbacks"
"""

import types
import pandas as pd
import numpy as np
from sqlalchemy import text

# -----------------------
# Per-date cache structure
# -----------------------
def _ensure_cache(engine):
    if not hasattr(engine, "_fscore_cache"):
        engine._fscore_cache = {}
    return engine._fscore_cache

# -----------------------
# Helper: bulk fetches
# -----------------------
def _fetch_nonfin_intermediary(conn, tickers, y, q):
    sql = text("""
        SELECT ticker, NetProfit_TTM, AvgTotalAssets, NetCFO_TTM, Revenue_TTM, COGS_TTM
        FROM intermediary_calculations_enhanced
        WHERE year=:y AND quarter=:q AND has_full_ttm=1 AND ticker IN :tickers
    """)
    return pd.read_sql(sql, conn, params={"y": y, "q": q, "tickers": tuple(tickers)})

def _fetch_nf_bal_snapshot(conn, tickers, y, q):
    try:
        sql = text("""
            SELECT ticker, CurrentAssets, CurrentLiabilities,
                   COALESCE(ShortTermDebt,0)+COALESCE(LongTermDebt,0) AS TotalDebt
            FROM comprehensive_fundamentals_snapshot
            WHERE year=:y AND quarter=:q AND ticker IN :tickers
        """)
        return pd.read_sql(sql, conn, params={"y": y, "q": q, "tickers": tuple(tickers)})
    except Exception:
        sql = text("""
            SELECT ticker, CurrentAssets, CurrentLiabilities,
                   COALESCE(ShortTermDebt,0)+COALESCE(LongTermDebt,0) AS TotalDebt
            FROM v_comprehensive_fundamental_items
            WHERE year=:y AND quarter=:q AND ticker IN :tickers
        """)
        return pd.read_sql(sql, conn, params={"y": y, "q": q, "tickers": tuple(tickers)})

def _fetch_last_shares(conn, tickers, asof):
    sql = text("""
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
    """)
    return pd.read_sql(sql, conn, params={"tickers": tuple(tickers), "asof": pd.Timestamp(asof).date()})

# -----------------------
# Vectorized calculators
# -----------------------
def _compute_nf_vectorized(engine, tickers, y, q, analysis_date):
    cache = _ensure_cache(engine)
    key = ('nf', y, q, pd.Timestamp(analysis_date).date(), tuple(sorted(tickers)))
    if key in cache:
        return cache[key]

    py = y - 1
    with engine.engine.begin() as conn:
        cur_ice = _fetch_nonfin_intermediary(conn, tickers, y, q)
        prv_ice = _fetch_nonfin_intermediary(conn, tickers, py, q)
        cur_bal = _fetch_nf_bal_snapshot(conn, tickers, y, q)
        prv_bal = _fetch_nf_bal_snapshot(conn, tickers, py, q)
        cur_sh  = _fetch_last_shares(conn, tickers, analysis_date)
        prv_sh  = _fetch_last_shares(conn, tickers, pd.Timestamp(analysis_date) - pd.DateOffset(years=1))

    cur = cur_ice.merge(cur_bal, on="ticker", how="left").merge(cur_sh, on="ticker", how="left").rename(columns={"total_shares":"shares"})
    prv = prv_ice.merge(prv_bal, on="ticker", how="left").merge(prv_sh, on="ticker", how="left").rename(columns={"total_shares":"shares_prev"})
    df  = cur.merge(prv, on="ticker", suffixes=("", "_prev"), how="inner").copy()

    eps = 1e-12
    assets     = df["AvgTotalAssets"].replace(0, np.nan)
    assets_prev= df["AvgTotalAssets_prev"].replace(0, np.nan)

    # Profitability
    roa      = df["NetProfit_TTM"] / assets
    roa_prev = df["NetProfit_TTM_prev"] / assets_prev
    cfo      = df["NetCFO_TTM"]
    accrual  = (df["NetCFO_TTM"] > df["NetProfit_TTM"]).astype(int)

    # Leverage/Liquidity/Funding
    lev      = (df["TotalDebt"] / assets).replace([np.inf, -np.inf], np.nan)
    lev_prev = (df["TotalDebt_prev"] / assets_prev).replace([np.inf, -np.inf], np.nan)
    cr       = df["CurrentAssets"] / df["CurrentLiabilities"].replace(0, np.nan)
    cr_prev  = df["CurrentAssets_prev"] / df["CurrentLiabilities_prev"].replace(0, np.nan)
    shares   = df["shares"].replace(0, np.nan)
    shares_prev = df["shares_prev"].replace(0, np.nan)

    # Operating efficiency (Gross Margin delta & Asset Turnover delta)
    gm       = (df["Revenue_TTM"] - df["COGS_TTM"]) / df["Revenue_TTM"].replace(0, np.nan)
    gm_prev  = (df["Revenue_TTM_prev"] - df["COGS_TTM_prev"]) / df["Revenue_TTM_prev"].replace(0, np.nan)
    ato      = df["Revenue_TTM"] / assets
    ato_prev = df["Revenue_TTM_prev"] / assets_prev

    # Final 9-signal Piotroski for Non-Financial
    score = (
        (roa > 0).astype(int) +           # P1: Positive ROA
        (cfo > 0).astype(int) +           # P2: Positive CFO
        (roa > roa_prev).astype(int) +    # P3: ∆ROA
        accrual +                          # P4: Quality of Earnings (CFO > NI)
        (lev < lev_prev).astype(int) +    # P5: ∆Leverage
        (cr > cr_prev).astype(int) +      # P6: ∆Liquidity
        (shares <= shares_prev * 1.01).astype(int) + # P7: No new equity (1% tolerance)
        (gm > gm_prev).astype(int) +      # P8: ∆Gross Margin
        (ato > ato_prev).astype(int)      # P9: ∆Asset Turnover
    )

    res = dict(zip(df["ticker"], score.fillna(0).astype(int)))
    cache[key] = res
    return res

def _compute_bank_vectorized(engine, tickers, y, q):
    """
    Agent Neo Fix: Replicate orchestrator's alias map + fallbacks
    The key issue was CustomerDeposits doesn't exist in v_comprehensive_fundamental_items
    """
    cache = _ensure_cache(engine)
    key = ('bank', y, q, None, tuple(sorted(tickers)))
    if key in cache:
        return cache[key]
    
    py = y - 1
    with engine.engine.begin() as conn:
        # Banking fundamentals from cleaned table
        sql = text("""
            SELECT ticker, NetProfit_TTM, AvgTotalAssets,
                   NII_TTM, AvgEarningAssets,
                   TotalOperatingIncome_TTM, OperatingExpenses_TTM,
                   AvgCustomerDeposits
            FROM intermediary_calculations_banking_cleaned
            WHERE year=:y AND quarter=:q AND has_full_ttm=1 AND ticker IN :tickers
        """)
        cur = pd.read_sql(sql, conn, params={"y": y, "q": q, "tickers": tuple(tickers)})
        prv = pd.read_sql(sql, conn, params={"y": py, "q": q, "tickers": tuple(tickers)})
        
        # Agent Neo Fix: For balance sheet, try banking-specific table first
        try:
            # Try banking-specific comprehensive table if exists
            sqlb = text("""
                SELECT ticker, 
                       COALESCE(ShareholdersEquity, TotalEquity, AvgTotalEquity) AS ShareholdersEquity,
                       COALESCE(CustomerDeposits, TotalLiabilities, AvgCustomerDeposits, 0) AS CustomerDeposits
                FROM v_comprehensive_fundamental_items_banking
                WHERE year=:y AND quarter=:q AND ticker IN :tickers
            """)
            curb = pd.read_sql(sqlb, conn, params={"y": y, "q": q, "tickers": tuple(tickers)})
            prvb = pd.read_sql(sqlb, conn, params={"y": py, "q": q, "tickers": tuple(tickers)})
        except:
            # Agent Neo's fallback: Use available proxies from general table
            sqlb = text("""
                SELECT ticker, 
                       COALESCE(TotalEquity, OwnersEquity) AS ShareholdersEquity,
                       COALESCE(TotalLiabilities, CurrentLiabilities, 0) AS CustomerDeposits
                FROM v_comprehensive_fundamental_items
                WHERE year=:y AND quarter=:q AND ticker IN :tickers
            """)
            curb = pd.read_sql(sqlb, conn, params={"y": y, "q": q, "tickers": tuple(tickers)})
            prvb = pd.read_sql(sqlb, conn, params={"y": py, "q": q, "tickers": tuple(tickers)})
    
    # Merge data
    cur = cur.merge(curb, on="ticker", how="left")
    prv = prv.merge(prvb, on="ticker", how="left", suffixes=("", "_prev"))
    df = cur.merge(prv, on="ticker", how="inner", suffixes=("", "_prev"))
    
    # Agent Neo: Use AvgCustomerDeposits from banking table if CustomerDeposits not available
    if 'CustomerDeposits' not in df.columns and 'AvgCustomerDeposits' in df.columns:
        df['CustomerDeposits'] = df['AvgCustomerDeposits']
    if 'CustomerDeposits_prev' not in df.columns and 'AvgCustomerDeposits_prev' in df.columns:
        df['CustomerDeposits_prev'] = df['AvgCustomerDeposits_prev']
    
    # Fill any remaining NaNs with 0 (Agent Neo's fallback)
    df['CustomerDeposits'] = df.get('CustomerDeposits', 0).fillna(0)
    df['CustomerDeposits_prev'] = df.get('CustomerDeposits_prev', 0).fillna(0)
    
    assets      = df["AvgTotalAssets"].replace(0, np.nan)
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
    cache[key] = res
    return res

def _compute_sec_vectorized(engine, tickers, y, q):
    cache = _ensure_cache(engine)
    key = ('sec', y, q, None, tuple(sorted(tickers)))
    if key in cache:
        return cache[key]
    py = y - 1
    with engine.engine.begin() as conn:
        sql = text("""
            SELECT ticker, TotalOperatingRevenue_TTM, NetProfit_TTM,
                   AvgTotalAssets, OperatingResult_TTM, OperatingExpenses_TTM
            FROM intermediary_calculations_securities_cleaned
            WHERE year=:y AND quarter=:q AND has_full_ttm=1 AND ticker IN :tickers
        """)
        cur = pd.read_sql(sql, conn, params={"y": y, "q": q, "tickers": tuple(tickers)})
        prv = pd.read_sql(sql, conn, params={"y": py, "q": q, "tickers": tuple(tickers)})

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
    cache[key] = res
    return res

# -----------------------
# Public install hooks
# -----------------------
def install_vectorized_fscore_neo_fix(engine):
    """Agent Neo's fixed version with proper fallbacks"""
    engine._get_raw_f_score_non_financial = types.MethodType(
        lambda self, tickers, y, q, analysis_date:
            _compute_nf_vectorized(self, tickers, y, q, analysis_date), engine
    )
    engine._get_raw_f_score_banking = types.MethodType(
        lambda self, tickers, y, q:
            _compute_bank_vectorized(self, tickers, y, q), engine
    )
    engine._get_raw_f_score_securities = types.MethodType(
        lambda self, tickers, y, q:
            _compute_sec_vectorized(self, tickers, y, q), engine
    )
    return engine

def prime_fscore_cache(engine, universe_df, analysis_date, y, q):
    """Warm up cache once per date with a single DB pass per sector group"""
    _ensure_cache(engine)
    tick = universe_df["ticker"].tolist()
    sec_map = universe_df.set_index("ticker")["sector"].to_dict()
    
    # Agent Neo Fix: Use 'Banks' not 'Banking'
    nf   = [t for t in tick if sec_map.get(t) not in ("Banks", "Securities")]
    bank = [t for t in tick if sec_map.get(t) == "Banks"]
    sec  = [t for t in tick if sec_map.get(t) == "Securities"]

    if nf:
        _compute_nf_vectorized(engine, nf, y, q, analysis_date)
    if bank:
        _compute_bank_vectorized(engine, bank, y, q)
    if sec:
        _compute_sec_vectorized(engine, sec, y, q)