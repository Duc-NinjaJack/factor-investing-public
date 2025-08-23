import warnings
import pandas as pd
import numpy as np

from production.engine.qvm_engine_v2_0_1_flat import QVMEngineV201Flat


class _DummyEngine(QVMEngineV201Flat):
    def __init__(self):
        # Bypass heavy init; minimally set attributes used in method
        self.logger = __import__('logging').getLogger('NormalizationTest')
        self.normalization_config = {
            'min_sector_size': 3,
            'robust': 'median_mad',
            'fallback': ['sector', 'market']
        }


def _make_sample_df(n=20, sectors=('Tech', 'Bank')):
    rng = np.random.default_rng(0)
    tickers = [f"T{i:03d}" for i in range(n)]
    sector_vals = [sectors[i % len(sectors)] for i in range(n)]
    values = rng.normal(0, 1, size=n)
    df = pd.DataFrame({
        'ticker': tickers,
        'sector': sector_vals,
        'metric': values,
        # Include a date column to exercise inferred-date code paths
        'date': pd.Timestamp('2020-01-31')
    })
    # Intentionally shuffle index to mimic real joins and require reindexing
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df


def test_sector_neutral_normalization_emits_no_alignment_warnings():
    eng = _DummyEngine()
    df = _make_sample_df()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        z = eng.calculate_sector_neutral_zscore(df, 'metric', 'sector')
    # No warnings expected from alignment/misalignment paths
    assert not any('align' in str(w.message).lower() for w in caught), [str(w.message) for w in caught]
    # Output shape must match input rows exactly
    assert isinstance(z, pd.Series)
    assert len(z) == len(df)
    # Index alignment preserved to input order
    pd.testing.assert_index_equal(z.index, df.index)

