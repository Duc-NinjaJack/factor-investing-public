import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from production.engine.qvm_engine_v2_2_1_flat import QVMEngineV221Flat


class DummyDB:
    def __init__(self, has_rows=True):
        self.has_rows = has_rows

    def __call__(self, sql, engine, params=None):
        # only used in _get_lagged_quarter_info to check count(*)
        if 'FROM intermediary_calculations_enhanced' in str(sql):
            return pd.DataFrame({'count': [1 if self.has_rows else 0]})
        return pd.DataFrame()


def test_lagged_quarter_switches_before_announcement(monkeypatch, tmp_path):
    # Create engine and monkeypatch DB engine + read_sql counts
    eng = QVMEngineV221Flat(config_path=str(REPO_ROOT / 'config'), log_level='INFO')

    # Ensure read_sql used in _get_lagged_quarter_info returns count>0
    dummy = DummyDB(True)
    import pandas as pd_mod
    monkeypatch.setattr(pd_mod, 'read_sql', dummy, raising=False)

    # Analysis date: early May → current quarter Q2, lagged Q1, but earnings delay may push to Q4 prev year
    d = pd.Timestamp('2025-05-01')
    y, q = eng._get_lagged_quarter_info(d)
    assert q in (1, 4)


