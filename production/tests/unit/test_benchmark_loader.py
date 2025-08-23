import pandas as pd
import numpy as np
from production.utils.benchmark_loader import load_benchmark_series


class DummyEngine:
    def connect(self):
        return self
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return False


def test_benchmark_loader_cache_hit(monkeypatch):
    calls = {'count': 0}

    def fake_read_sql(query, engine, params=None, **kwargs):  # type: ignore
        calls['count'] += 1
        dates = pd.date_range('2020-01-01', '2020-01-10', freq='D')
        if 'trading_date' in str(query):
            return pd.DataFrame({'date': dates, 'close_price': np.linspace(100, 110, len(dates))})
        else:
            return pd.DataFrame({'date': dates, 'close_price': np.linspace(100, 110, len(dates))})

    monkeypatch.setattr(pd, 'read_sql', fake_read_sql)

    eng = DummyEngine()
    close1, rets1 = load_benchmark_series(eng, pd.Timestamp('2020-01-01'), pd.Timestamp('2020-01-31'))
    assert isinstance(close1, pd.Series)
    assert not close1.empty
    # Second call should hit cache and not increment read_sql
    prev_calls = calls['count']
    close2, rets2 = load_benchmark_series(eng, pd.Timestamp('2020-01-01'), pd.Timestamp('2020-01-31'))
    assert calls['count'] == prev_calls
    pd.testing.assert_series_equal(close1, close2)
    pd.testing.assert_series_equal(rets1, rets2)


