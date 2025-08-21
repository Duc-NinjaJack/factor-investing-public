import sys
from pathlib import Path

import pandas as pd


# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from production.engine.qvm_engine_v2_2_1_flat_vectorized import (
    normalize_sector_labels_221,
    prime_fscore_cache_221,
)


class DummyLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


class DummyEngine:
    def __init__(self):
        self.logger = DummyLogger()
        self._fscore_cache_221 = {}
        # No DB usage when compute_* are monkeypatched in tests
        self.engine = None


def test_normalize_sector_labels_221_basic():
    df = pd.DataFrame({
        'ticker': ['VCB', 'SSI', 'HPG', 'TCB', 'BVH'],
        'sector': ['banks', 'brokerage', 'Materials', 'Bank', 'Insurance']
    })
    out = normalize_sector_labels_221(df, 'sector')
    out_map = dict(zip(out['ticker'], out['sector']))
    assert out_map['VCB'] == 'Banking'
    assert out_map['SSI'] == 'Securities'
    assert out_map['HPG'] == 'Materials'
    assert out_map['TCB'] == 'Banking'
    assert out_map['BVH'] == 'Insurance'


def test_prime_fscore_cache_221_grouping(monkeypatch):
    # Prepare universe with mixed labels
    universe_df = pd.DataFrame({
        'ticker': ['VCB', 'TCB', 'SSI', 'VND', 'HPG', 'VIC'],
        'sector': ['banks', 'Banking', 'brokerage', 'Securities', 'Materials', 'Real Estate']
    })

    engine = DummyEngine()

    calls = {'nf': None, 'bank': None, 'sec': None}

    # Monkeypatch compute_* to capture tickers
    def fake_nf(self, tickers, y, q, d):
        calls['nf'] = list(sorted(tickers))
        return {}

    def fake_bank(self, tickers, y, q):
        calls['bank'] = list(sorted(tickers))
        return {}

    def fake_sec(self, tickers, y, q):
        calls['sec'] = list(sorted(tickers))
        return {}

    import production.engine.qvm_engine_v2_2_1_flat_vectorized as mod
    monkeypatch.setattr(mod, 'compute_nf_vectorized_221', fake_nf, raising=True)
    monkeypatch.setattr(mod, 'compute_bank_vectorized_221', fake_bank, raising=True)
    monkeypatch.setattr(mod, 'compute_sec_vectorized_221', fake_sec, raising=True)

    # Execute cache priming
    prime_fscore_cache_221(engine, universe_df, pd.Timestamp('2025-08-18'), 2025, 2)

    # Validate groupings
    assert calls['bank'] == ['TCB', 'VCB']
    assert calls['sec'] == ['SSI', 'VND']
    assert calls['nf'] == ['HPG', 'VIC']


