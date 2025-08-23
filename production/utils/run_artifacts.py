#!/usr/bin/env python3
"""
Run Artifacts Writer
====================

Creates a reproducibility artifact run.json per run with:
- config hash
- environment snapshot
- git commit
- seeds
- calendar hash

Writes to artifacts/run.json and artifacts/runs/{timestamp}_run.json
"""

from __future__ import annotations

import json
import os
import sys
import platform
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import pandas as pd


def _safe_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _hash_json(obj: object) -> str:
    import hashlib
    try:
        payload = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    except TypeError:
        payload = str(obj).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _calendar_hash(calendar_anchors: Optional[pd.DataFrame]) -> str:
    if calendar_anchors is None or calendar_anchors.empty:
        return "none"
    cols = [c for c in ["target", "anchor", "anchor_type", "delta_days"] if c in calendar_anchors.columns]
    df = calendar_anchors[cols].copy()
    # Normalize datetime to ISO date
    for c in ["target", "anchor"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c]).dt.strftime("%Y-%m-%d")
    return _hash_json(df.to_dict(orient="list"))


def write_run_artifact(
    run_config: Dict,
    calendar_anchors: Optional[pd.DataFrame] = None,
    seeds: Optional[Dict] = None,
    extra: Optional[Dict] = None,
    base_dir: str = "artifacts",
) -> str:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    env_snapshot = {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "env_flags": {
            "DISABLE_DB_IN_CHILDREN": os.environ.get("DISABLE_DB_IN_CHILDREN", "0"),
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        },
    }

    artifact = {
        "timestamp_utc": ts,
        "config_hash": _hash_json(run_config or {}),
        "git_commit": _safe_git_commit(),
        "calendar_hash": _calendar_hash(calendar_anchors),
        "seeds": seeds or {},
        "environment": env_snapshot,
        "config": run_config or {},
    }
    if extra:
        artifact["extra"] = extra

    # Write stable and timestamped files
    stable_path = out_dir / "run.json"
    ts_path = runs_dir / f"{ts}_run.json"
    stable_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2))
    ts_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2))
    return str(ts_path.resolve())


