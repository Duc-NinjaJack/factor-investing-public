#!/usr/bin/env python3
import pandas as pd
import importlib.util
import sys
import types
from pathlib import Path


def _load_runner_module():
    here = Path(__file__).resolve()
    runner_path = here.with_name("08_QVM_v221_flat_vectorized_strategy.py")
    # Stub validation module to avoid hard dependency on pydantic in unit test
    stub_mod = types.ModuleType('production.scripts.validation_manager')
    def _noop_validate_strategy_config(cfg, logger):
        return True
    def _pass_backtest_config(cfg, logger):
        return cfg
    stub_mod.validate_strategy_config = _noop_validate_strategy_config
    stub_mod.validate_backtest_config = _pass_backtest_config
    sys.modules['production.scripts.validation_manager'] = stub_mod
    spec = importlib.util.spec_from_file_location("runner_module", str(runner_path))
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _mk_benchmark_index(start: str = '2021-01-01', end: str = '2021-12-31') -> pd.Series:
    idx = pd.date_range(start=start, end=end, freq='B')
    # simulate holidays by dropping some dates
    idx = idx.difference(pd.DatetimeIndex([
        '2021-02-12', '2021-02-15',  # Tet
        '2021-04-30', '2021-05-03',
    ]))
    return pd.Series(range(len(idx)), index=idx, dtype='float64')


def test_anchor_first_trading_day():
    bench = _mk_benchmark_index()
    cfg = { 'rebalance': { 'anchor': 'first_trading_day' } }
    runner = _load_runner_module()
    dates = runner._build_rebalance_calendar(bench, pd.Timestamp('2021-01-01'), pd.Timestamp('2021-12-31'), cfg)
    # Should have 12 monthly dates and all are business days present in index
    assert len(dates) == 12
    assert all(d in bench.index for d in dates)


def test_anchor_mid_month_handles_holiday():
    bench = _mk_benchmark_index()
    cfg = { 'rebalance': { 'anchor': 'mid_month' } }
    runner = _load_runner_module()
    dates = runner._build_rebalance_calendar(bench, pd.Timestamp('2021-01-01'), pd.Timestamp('2021-12-31'), cfg)
    assert len(dates) == 12
    # Mid-month anchor should be >= 15th and on/after business day
    assert all(d.day >= 15 for d in dates)
    assert all(d in bench.index for d in dates)


def test_anchor_quarter_lag_with_days():
    bench = _mk_benchmark_index()
    cfg = { 'rebalance': { 'anchor': 'quarter_lag', 'lag_days': 45 } }
    runner = _load_runner_module()
    dates = runner._build_rebalance_calendar(bench, pd.Timestamp('2021-01-01'), pd.Timestamp('2021-12-31'), cfg)
    assert len(dates) == 12
    assert all(d in bench.index for d in dates)


