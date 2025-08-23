#!/usr/bin/env python3
"""
Parallelism Utilities
=====================

Goals (temporary safety phase):
- Enforce 'spawn' start method for process-based parallelism
- Provide helpers to materialize DB reads to Parquet before fan-out
- Provide a stub guard to prevent DB access in child processes
"""

from __future__ import annotations

import os
import multiprocessing as mp
import logging
from pathlib import Path
from typing import Optional

import pandas as pd


def ensure_spawn_start_method(logger: Optional[logging.Logger] = None) -> None:
    try:
        current = mp.get_start_method(allow_none=True)
        if current != 'spawn':
            mp.set_start_method('spawn', force=True)
            if logger:
                logger.info("Parallel start method set to 'spawn' (was %s)", current)
        else:
            if logger:
                logger.debug("Parallel start method already 'spawn'")
    except RuntimeError:
        # Start method likely already set by parent; ignore
        if logger:
            logger.debug("Start method already set; leaving as-is")


def materialize_dataframe(df: pd.DataFrame, output_path: str, logger: Optional[logging.Logger] = None) -> str:
    """
    Write a DataFrame to Parquet on local filesystem for safe sharing across processes.
    Returns the absolute path to the written file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    if logger:
        logger.info("Materialized DataFrame to %s (%d rows, %d cols)", str(path), len(df), len(df.columns))
    return str(path.resolve())


def disable_db_access_in_children() -> None:
    """
    Set an environment flag to signal child processes not to open DB connections.
    Runners should check this flag before constructing engines.
    """
    os.environ['DISABLE_DB_IN_CHILDREN'] = '1'


def db_access_allowed() -> bool:
    return os.environ.get('DISABLE_DB_IN_CHILDREN', '0') != '1'


