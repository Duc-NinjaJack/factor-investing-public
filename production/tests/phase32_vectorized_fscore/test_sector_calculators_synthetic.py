import sys
from pathlib import Path

import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from production.engine.qvm_engine_v2_2_1_flat_vectorized import (
    compute_nf_vectorized_221,
    compute_bank_vectorized_221,
    compute_sec_vectorized_221,
)


class DummyConn:
    pass


class DummySQLEngine:
    def __init__(self, frames):
        self.frames = frames

    def begin(self):
        # minimal context manager
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    # used by pd.read_sql via text+conn → we will monkeypatch pd.read_sql


class DummyEngine:
    def __init__(self, frames):
        self.engine = DummySQLEngine(frames)
        class L: 
            def info(self, *a, **k): pass
            def warning(self, *a, **k): pass
        self.logger = L()


def _patch_read_sql(monkeypatch, mapping):
    def fake_read_sql(sql, conn, params=None, **kwargs):
        key = (str(sql).strip(), tuple(sorted((params or {}).items())))
        if key in mapping:
            return mapping[key].copy()
        # crude fallback by table name
        s = str(sql)
        if 'intermediary_calculations_enhanced' in s:
            return mapping['nf_cur'] if 'prev' not in (params or {}) else mapping['nf_prev']
        if 'comprehensive_fundamentals' in s or 'v_comprehensive_fundamental_items' in s:
            return mapping['nf_bal_cur'] if params.get('y') == mapping.get('y_cur') else mapping['nf_bal_prev']
        if 'vcsc_daily_data_complete' in s and 'MAX(trading_date)' in s:
            return pd.DataFrame()
        if 'vcsc_daily_data_complete' in s:
            return mapping['shares_cur'] if params and 'asof' in params and 'prev' not in s else mapping['shares_prev']
        if 'intermediary_calculations_banking_cleaned' in s:
            return mapping['bank_cur'] if params.get('y') == mapping.get('y_cur') else mapping['bank_prev']
        if 'v_comprehensive_fundamental_items_banking' in s or 'v_comprehensive_fundamental_items' in s:
            return mapping['bank_bal_cur'] if params.get('y') == mapping.get('y_cur') else mapping['bank_bal_prev']
        if 'intermediary_calculations_securities_cleaned' in s:
            return mapping['sec_cur'] if params.get('y') == mapping.get('y_cur') else mapping['sec_prev']
        return pd.DataFrame()
    monkeypatch.setattr(pd, 'read_sql', fake_read_sql, raising=True)


def test_nf_roa_sign_and_shares_tolerance(monkeypatch):
    # two tickers with NI>0 and shares flat vs +2%
    nf_cur = pd.DataFrame({'ticker': ['AAA','BBB'], 'NetProfit_TTM':[10, 5], 'AvgTotalAssets':[100,50], 'NetCFO_TTM':[12,6], 'Revenue_TTM':[200,100], 'COGS_TTM':[100,50]})
    nf_prev = pd.DataFrame({'ticker': ['AAA','BBB'], 'NetProfit_TTM':[8, 4], 'AvgTotalAssets':[100,50], 'NetCFO_TTM':[10,5], 'Revenue_TTM':[180,100], 'COGS_TTM':[90,55]})
    nf_bal_cur = pd.DataFrame({'ticker':['AAA','BBB'], 'CurrentAssets':[50,25], 'CurrentLiabilities':[25,25], 'TotalDebt':[20,10]})
    nf_bal_prev= pd.DataFrame({'ticker':['AAA','BBB'], 'CurrentAssets':[40,20], 'CurrentLiabilities':[25,20], 'TotalDebt':[25,10]})
    shares_cur = pd.DataFrame({'ticker':['AAA','BBB'], 'total_shares':[100,102]})
    shares_prev= pd.DataFrame({'ticker':['AAA','BBB'], 'total_shares':[100,100]})

    mapping = {
        'nf_cur': nf_cur, 'nf_prev': nf_prev,
        'nf_bal_cur': nf_bal_cur, 'nf_bal_prev': nf_bal_prev,
        'shares_cur': shares_cur, 'shares_prev': shares_prev,
        'y_cur': 2025
    }
    _patch_read_sql(monkeypatch, mapping)
    engine = DummyEngine(mapping)
    res = compute_nf_vectorized_221(engine, ['AAA','BBB'], 2025, 2, pd.Timestamp('2025-08-18'))
    # AAA should get P7 (no issuance) = 1; BBB issuance +2% >1% tolerance → 0
    assert res['AAA'] >= res['BBB']


def test_bank_backfill_avg_customer_deposits(monkeypatch):
    bank_cur = pd.DataFrame({'ticker':['VCB'], 'NetProfit_TTM':[10], 'AvgTotalAssets':[100], 'NII_TTM':[5], 'AvgEarningAssets':[80], 'TotalOperatingIncome_TTM':[20], 'OperatingExpenses_TTM':[10], 'AvgCustomerDeposits':[60]})
    bank_prev= pd.DataFrame({'ticker':['VCB'], 'NetProfit_TTM':[9], 'AvgTotalAssets':[95], 'NII_TTM':[4.5], 'AvgEarningAssets':[75], 'TotalOperatingIncome_TTM':[19], 'OperatingExpenses_TTM':[11], 'AvgCustomerDeposits':[55]})
    bank_bal_cur = pd.DataFrame({'ticker':['VCB'], 'ShareholdersEquity':[10]})
    bank_bal_prev= pd.DataFrame({'ticker':['VCB'], 'ShareholdersEquity':[9]})
    mapping = {
        'bank_cur': bank_cur, 'bank_prev': bank_prev,
        'bank_bal_cur': bank_bal_cur, 'bank_bal_prev': bank_bal_prev,
        'y_cur': 2025
    }
    _patch_read_sql(monkeypatch, mapping)
    engine = DummyEngine(mapping)
    res = compute_bank_vectorized_221(engine, ['VCB'], 2025, 2)
    # Should compute without CustomerDeposits column by backfilling AvgCustomerDeposits
    assert 'VCB' in res


def test_sec_operating_metrics_deltas(monkeypatch):
    sec_cur = pd.DataFrame({'ticker':['SSI'], 'TotalOperatingRevenue_TTM':[100], 'NetProfit_TTM':[5], 'AvgTotalAssets':[50], 'OperatingResult_TTM':[10], 'OperatingExpenses_TTM':[40]})
    sec_prev= pd.DataFrame({'ticker':['SSI'], 'TotalOperatingRevenue_TTM':[80], 'NetProfit_TTM':[4], 'AvgTotalAssets':[48], 'OperatingResult_TTM':[8], 'OperatingExpenses_TTM':[42]})
    mapping = {'sec_cur': sec_cur, 'sec_prev': sec_prev, 'y_cur': 2025}
    _patch_read_sql(monkeypatch, mapping)
    engine = DummyEngine(mapping)
    res = compute_sec_vectorized_221(engine, ['SSI'], 2025, 2)
    assert res['SSI'] >= 3  # roa>0, opRes>0, deltas improve


