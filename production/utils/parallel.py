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
from typing import Optional, Callable, Iterable, TypeVar, List, Any

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


def disable_db_access_in_children(force: Optional[bool] = None, logger: Optional[logging.Logger] = None) -> None:
    """
    Ensure the environment flag controlling DB access in child processes is set.

    Behavior:
    - If 'force' is None: respect any pre-set value. If not set, default to disabling (set to '1').
    - If 'force' is True/False: explicitly set to '1'/'0' respectively.
    - Optionally log the resulting state when a logger is provided.
    """
    existing = os.environ.get('DISABLE_DB_IN_CHILDREN')
    if force is None:
        if existing is None:
            os.environ['DISABLE_DB_IN_CHILDREN'] = '1'
            if logger:
                logger.info("Parallel safety: set DISABLE_DB_IN_CHILDREN=1 (default)")
        else:
            if logger:
                logger.info(f"Parallel safety: keeping DISABLE_DB_IN_CHILDREN={existing} (pre-set)")
    else:
        os.environ['DISABLE_DB_IN_CHILDREN'] = '1' if force else '0'
        if logger:
            logger.info(f"Parallel safety: set DISABLE_DB_IN_CHILDREN={os.environ['DISABLE_DB_IN_CHILDREN']} (forced)")


def db_access_allowed() -> bool:
    return os.environ.get('DISABLE_DB_IN_CHILDREN', '0') != '1'


T = TypeVar('T')
U = TypeVar('U')


def _child_initializer_set_env():
    # Child process initializer to ensure DB is disabled
    os.environ['DISABLE_DB_IN_CHILDREN'] = '1'


def safe_parallel_map(
    func: Callable[[T], U],
    iterable: Iterable[T],
    *,
    max_workers: Optional[int] = None,
    use_threads: bool = False,
    logger: Optional[logging.Logger] = None,
) -> List[U]:
    """
    Execute a map operation in parallel with safety guards and clear logging.

    - Enforces 'spawn' start method for process-based execution
    - Disables DB access in child processes via DISABLE_DB_IN_CHILDREN=1
    - Emits info logs when fan-out occurs and suggests prematerialization to Parquet
    - Falls back to sequential execution on error
    """
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

    items = list(iterable)
    n = len(items)
    if logger:
        logger.info("Parallel fan-out: %d tasks | executor=%s | max_workers=%s", n, 'threads' if use_threads else 'processes', str(max_workers or 'default'))
        logger.info("Prematerialization hint: For high fan-out, materialize inputs to Parquet and pass file paths instead of DB handles.")

    if n == 0:
        return []

    # Ensure spawn and disable DB in children
    try:
        ensure_spawn_start_method(logger)
    except Exception:
        pass

    results: List[U] = []
    Executor = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    try:
        if use_threads:
            # Threads share env; still set flag in parent for cooperative checks
            disable_db_access_in_children(force=True, logger=logger)
            with Executor(max_workers=max_workers) as ex:
                futs = [ex.submit(func, item) for item in items]
                for fut in as_completed(futs):
                    results.append(fut.result())
        else:
            with Executor(max_workers=max_workers, initializer=_child_initializer_set_env) as ex:
                futs = [ex.submit(func, item) for item in items]
                for fut in as_completed(futs):
                    results.append(fut.result())
    except Exception as e:
        if logger:
            logger.error("Parallel execution failed (%s). Falling back to sequential.", e)
        # Sequential fallback
        results = [func(item) for item in items]

    return results

