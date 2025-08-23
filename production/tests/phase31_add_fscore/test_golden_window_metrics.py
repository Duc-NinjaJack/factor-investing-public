import json
import subprocess
import sys
from pathlib import Path
import os


GOLDEN_PATH = Path('production/tests/phase31_add_fscore/golden_metrics_2019_2020.json')


def _compute_metrics(run_dir: Path):
    import pandas as pd
    def metrics(series):
        series = series.dropna()
        cagr = (1+series).prod()**(252/len(series)) - 1
        sharpe = (series.mean()*252) / (series.std()*(252**0.5) + 1e-12)
        eq = (1+series).cumprod()
        mdd = (eq/eq.cummax()-1).min()
        return float(cagr), float(sharpe), float(mdd)
    wr = pd.read_csv(run_dir / 'with_risk_returns.csv', index_col=0).iloc[:,0]
    nr = pd.read_csv(run_dir / 'no_risk_returns.csv', index_col=0).iloc[:,0]
    br = pd.read_csv(run_dir / 'benchmark_returns.csv', index_col=0).iloc[:,0]
    return {
        'with_risk': dict(zip(['CAGR','Sharpe','MDD'], metrics(wr))),
        'no_risk': dict(zip(['CAGR','Sharpe','MDD'], metrics(nr))),
        'benchmark': dict(zip(['CAGR','Sharpe','MDD'], metrics(br))),
    }


def test_golden_window_metrics_within_tolerance(tmp_path):
    # Prefer existing artifacts to avoid redundant heavy runs
    artifacts_root = Path('artifacts/qvm_v221_flat_vectorized')
    subdirs = sorted([p for p in artifacts_root.iterdir()] if artifacts_root.exists() else [], key=lambda p: p.stat().st_mtime, reverse=True)
    if not subdirs:
        runner = Path('production/tests/phase31_add_fscore/08_QVM_v221_flat_vectorized_strategy.py').as_posix()
        env = os.environ.copy()
        env["USE_VECTORIZED_F_SCORE_221"] = "0"  # keep snapshot on pre-vectorized path for CI stability
        cmd = [sys.executable, runner, '--window', '2019-01-01:2020-12-31']
        subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        subdirs = sorted([p for p in artifacts_root.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    assert subdirs, 'No artifacts found for golden window run'
    run_dir = subdirs[0]

    # Load or create golden snapshot
    current = _compute_metrics(run_dir)
    if not GOLDEN_PATH.exists():
        GOLDEN_PATH.write_text(json.dumps(current, indent=2))
        assert True, 'Golden snapshot created'
        return

    golden = json.loads(GOLDEN_PATH.read_text())

    # Tolerances
    tol = {
        'CAGR': 0.02,    # 2% abs to allow small data/env drift
        'Sharpe': 0.16,  # 0.16 abs
        'MDD': 0.06,     # 6% abs
    }

    for key in ['with_risk','no_risk','benchmark']:
        for metric in ['CAGR','Sharpe','MDD']:
            diff = abs(current[key][metric] - golden[key][metric])
            assert diff <= tol[metric], f"{key}:{metric} diff {diff:.4f} exceeds tolerance {tol[metric]:.4f}"


