import pandas as pd
import numpy as np

from production.backtester.core import first_trading_day_calendar, equal_weight_positions, apply_transaction_costs
from production.risk.overlay import drawdown_to_cash_allocation, volatility_targeting


def test_first_trading_day_calendar_basic():
    dates = pd.to_datetime([
        "2020-01-02", "2020-01-03", "2020-02-03", "2020-02-04", "2020-03-02"
    ])
    pm = pd.DataFrame(index=dates, data={"AAA": [1, 1, 1, 1, 1]})
    cal = first_trading_day_calendar(pm)
    assert cal[0] == pd.Timestamp("2020-01-02")
    assert cal[1] == pd.Timestamp("2020-02-03")


def test_equal_weight_positions():
    w = equal_weight_positions(["A","B","C"], 2)
    assert np.isclose(w.sum(), 1.0)
    assert len(w) == 2


def test_apply_transaction_costs():
    p = pd.Series({"A": 0.5, "B": 0.5})
    n = pd.Series({"A": 1.0})
    cost = apply_transaction_costs(p, n, 10.0)
    assert cost > 0


def test_drawdown_to_cash_allocation():
    prices = pd.Series([100, 110, 90, 85], index=pd.to_datetime(["2020-01-01","2020-01-02","2020-01-03","2020-01-06"]))
    cash = drawdown_to_cash_allocation(prices, pd.Timestamp("2020-01-06"))
    assert 0.0 <= cash <= 1.0


def test_volatility_targeting():
    rng = pd.date_range("2020-01-01", periods=200, freq="B")
    rets = pd.Series(0.001, index=rng)
    managed, exposure = volatility_targeting(rets, target_vol=0.10)
    assert not managed.empty and not exposure.empty


