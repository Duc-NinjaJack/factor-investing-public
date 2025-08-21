import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from production.engine.qvm_engine_v2_2_1_flat_vectorized import (
    prime_fscore_cache_221,
    compute_bank_vectorized_221,
)


class DummyLogger:
    def info(self, *a, **k):
        pass
    def warning(self, *a, **k):
        pass


class DummyEngine:
    def __init__(self):
        self.logger = DummyLogger()
        self._fscore_cache_221 = {}
        self.engine = None


def test_cache_hit_behavior(monkeypatch):
    engine = DummyEngine()
    calls = {'bank': 0}

    def fake_bank(self, tickers, y, q):
        calls['bank'] += 1
        return {t: 3 for t in tickers}

    import production.engine.qvm_engine_v2_2_1_flat_vectorized as mod
    monkeypatch.setattr(mod, 'compute_bank_vectorized_221', fake_bank, raising=True)

    universe_df = pd.DataFrame({'ticker': ['VCB','TCB'], 'sector':['Banking','Banking']})
    prime_fscore_cache_221(engine, universe_df, pd.Timestamp('2025-08-18'), 2025, 2)
    # second priming same date/keys should be cache hit and not call compute again
    prime_fscore_cache_221(engine, universe_df, pd.Timestamp('2025-08-18'), 2025, 2)
    assert calls['bank'] == 1


def test_banking_fallback_when_view_absent(monkeypatch):
    # When banking view raises, fallback still returns a result
    engine = DummyEngine()

    def fake_read_sql(sql, conn, params=None, **kwargs):
        s = str(sql)
        if 'intermediary_calculations_banking_cleaned' in s:
            return pd.DataFrame({'ticker':['VCB'], 'NetProfit_TTM':[10], 'AvgTotalAssets':[100], 'NII_TTM':[5], 'AvgEarningAssets':[90], 'TotalOperatingIncome_TTM':[20], 'OperatingExpenses_TTM':[10]})
        if 'v_comprehensive_fundamental_items_banking' in s:
            raise Exception('banking view missing')
        if 'v_comprehensive_fundamental_items' in s:
            return pd.DataFrame({'ticker':['VCB'], 'TotalEquity':[10], 'TotalLiabilities':[80]})
        return pd.DataFrame()

    monkeypatch.setattr(pd, 'read_sql', fake_read_sql, raising=True)
    res = compute_bank_vectorized_221(engine, ['VCB'], 2025, 2)
    assert 'VCB' in res


