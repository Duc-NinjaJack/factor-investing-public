import types
import importlib.machinery
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
RUNNER_PATH = (THIS_DIR / "08_QVM_v221_flat_vectorized_strategy.py").resolve()


def _load_runner_module():
    loader = importlib.machinery.SourceFileLoader("qvm_runner_v221", str(RUNNER_PATH))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    loader.exec_module(module)  # type: ignore[arg-type]
    return module


def test_select_constrained_holdings_empty_scores():
    m = _load_runner_module()
    out = m._select_constrained_holdings(scores={}, n=10)
    assert out == []


def test_select_constrained_holdings_sector_cap_enforced():
    m = _load_runner_module()
    # 10 tickers, alternating sectors A and B; ranked by composite descending
    scores = {f"T{i:02d}": {"QVM_Composite": 100 - i} for i in range(10)}
    sector_map = {f"T{i:02d}": ("A" if i % 2 == 0 else "B") for i in range(10)}
    # sector_cap 0.2 with n=10 -> max 2 per sector
    sel = m._select_constrained_holdings(scores, n=10, sector_map=sector_map, sector_cap=0.2)
    # With only two sectors and a 2-per-sector cap, we can select at most 4
    assert len(sel) == 4
    # Count per sector must be <= 2
    from collections import Counter
    c = Counter(sector_map[t] for t in sel)
    assert max(c.values()) <= 2


def test_select_constrained_holdings_adv_cap_filters_names():
    m = _load_runner_module()
    # 5 tickers, only first three have enough ADV to pass 5% cap at equal weight
    n = 5
    scores = {f"X{i}": {"QVM_Composite": 10 - i} for i in range(n)}
    adv_map = {"X0": 100.0, "X1": 100.0, "X2": 100.0, "X3": 1.0, "X4": 0.0}  # VND units (arbitrary scale)
    portfolio_notional_vnd = 100.0  # equal_notional = 20
    # cap = 0.05 * ADV -> names with ADV < 400 will fail (20 <= 0.05*ADV)
    # Here, X0..X2: 0.05*100=5 >= 20? No, fails; flip: set higher ADV
    adv_map = {"X0": 10000.0, "X1": 10000.0, "X2": 10000.0, "X3": 100.0, "X4": 0.0}
    sel = m._select_constrained_holdings(
        scores=scores,
        n=n,
        adv_map_vnd=adv_map,
        adv_participation_cap=0.05,
        portfolio_notional_vnd=portfolio_notional_vnd,
    )
    # X3, X4 should be filtered out by ADV capacity
    assert "X3" not in sel and "X4" not in sel
    assert set(sel).issuperset({"X0", "X1", "X2"})


def test_select_constrained_holdings_churn_control_retains_prev():
    m = _load_runner_module()
    # Ranked X0>X1>...; prev holdings include a low-ranked name we want to retain due to age
    scores = {f"X{i}": {"QVM_Composite": 100 - i} for i in range(10)}
    prev = {"X9", "X8"}
    ages = {"X9": 0, "X8": 2}
    sel = m._select_constrained_holdings(scores, n=5, prev_holdings=prev, min_hold_months=1, ages=ages)
    # X9 (age 0) should be retained if possible
    assert "X9" in sel


def test_compute_exposure_schedule_edges_and_bounds():
    m = _load_runner_module()
    # Empty inputs -> empty series
    s = m._compute_exposure_schedule(
        price_matrix=pd.DataFrame(),
        benchmark_returns=pd.Series(dtype=float),
        monthly_holdings={},
        monthly_calendar=[],
    )
    assert s.empty

    # Tiny history: expect default exposure 1.0 and within [0,1]
    dates = pd.date_range("2021-01-01", periods=10, freq="B")
    prices = pd.DataFrame({"A": np.linspace(100, 110, len(dates))}, index=dates)
    bench = prices["A"].pct_change().fillna(0.0)
    sched = m._compute_exposure_schedule(
        price_matrix=prices,
        benchmark_returns=bench,
        monthly_holdings={dates[0]: ["A"]},
        monthly_calendar=[dates[0]],
        vol_target_ann=0.10,
    )
    assert not sched.empty
    assert float(sched.min()) >= 0.0 and float(sched.max()) <= 1.0


