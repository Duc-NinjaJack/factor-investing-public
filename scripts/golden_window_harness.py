#!/usr/bin/env python3
"""
Golden Window Harness
=====================

Runs the QVM v2.2.1 flat vectorized strategy over a fixed golden window and
persists canonical artifacts into `golden_outputs/` for Gate 0 determinism
checks. It executes sequentially (jobs=1) and writes a manifest of file
hashes for quick equivalence testing.

Usage:
  QVM_SEED=42 python scripts/golden_window_harness.py --config config/golden_window.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict

import hashlib

PROJECT_ROOT = Path(__file__).resolve().parents[0].parent
RUNNER = PROJECT_ROOT / 'production' / 'tests' / 'phase31_add_fscore' / '08_QVM_v221_flat_vectorized_strategy.py'
ARTIFACTS_BASE = PROJECT_ROOT / 'artifacts' / 'qvm_v221_flat_vectorized'
GOLDEN_DIR = PROJECT_ROOT / 'golden_outputs'


def compute_file_hash(path: Path) -> Dict[str, str]:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        while True:
            chunk = fh.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return {'sha256': h.hexdigest(), 'bytes': path.stat().st_size}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=str(PROJECT_ROOT / 'config' / 'golden_window.yaml'))
    args = parser.parse_args()

    os.environ.setdefault('QVM_SEED', '42')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

    # Run the strategy sequentially with deterministic env
    cmd = [
        'python', str(RUNNER),
        '--config', args.config,
        '--jobs', '1'
    ]
    subprocess.check_call(cmd, cwd=str(PROJECT_ROOT))

    # Locate the latest artifacts directory
    latest = ARTIFACTS_BASE / 'latest'
    if latest.is_symlink() or latest.exists():
        run_dir = ARTIFACTS_BASE / latest.readlink()
    else:
        # Fallback: pick the most recent directory
        runs = sorted([p for p in ARTIFACTS_BASE.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
        if not runs:
            raise RuntimeError('No artifacts found after run')
        run_dir = runs[0]

    # Prepare golden output directory
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    target = GOLDEN_DIR / 'golden_window'
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(run_dir, target)

    # Compute canonical hashes for key artifacts
    manifest = {}
    for name in [
        'monthly_holdings.csv', 'diagnostics.csv', 'timings.csv', 'universe_diagnostics.csv',
        'run_info.json', 'portfolio_hashes.txt', 'strategy_config.json', 'backtest_config.json'
    ]:
        p = target / name
        if p.exists():
            manifest[name] = compute_file_hash(p)
    with open(target / 'artifact_manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    print(f"Golden window artifacts frozen at: {target}")


if __name__ == '__main__':
    main()


