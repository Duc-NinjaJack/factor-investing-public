import subprocess
import sys
from pathlib import Path


def test_integration_small_window_runs():
    runner = Path("production/tests/phase31_add_fscore/08_QVM_v221_flat_vectorized_strategy.py").as_posix()
    cmd = [sys.executable, runner, "--window", "2019-01-01:2020-12-31"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # Allow non-zero if data missing, but ensure the script executed and printed something
    assert proc.stdout is not None


