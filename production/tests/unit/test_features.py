import logging
import pandas as pd
import numpy as np

from production.utils.features import (
    compute_momentum_raw,
    compute_low_volatility_raw,
    prepare_fcf_yield_raw,
    normalize_f_score_to_unit,
    compute_period_return,
    make_normalization_frame,
)


def _make_price_panel_wide(start="2020-01-01", periods=400, tickers=("AAA", "BBB", "CCC")):
    idx = pd.bdate_range(start=start, periods=periods)
    rng = np.random.default_rng(42)
    prices = {}
    for t in tickers:
        # geometric random walk, start at ~100
        rets = rng.normal(0, 0.01, size=len(idx))
        path = 100 * np.exp(np.cumsum(rets))
        prices[t] = path
    df = pd.DataFrame(prices, index=idx)
    return df


def _make_price_panel_long(wide_df: pd.DataFrame):
    df = wide_df.stack().reset_index()
    df.columns = ["date", "ticker", "price"]
    return df


def test_momentum_raw_multi_horizon_and_skip_alignment_wide_and_long():
    wide = _make_price_panel_wide()
    long = _make_price_panel_long(wide)
    universe = list(wide.columns)
    # analysis date near the end to ensure enough data
    analysis_date = pd.Timestamp(wide.index[-1])
    lookbacks = {"1m": 1, "3m": 3, "6m": 6, "12m": 12}
    skip = 1

    out_wide = compute_momentum_raw(wide, analysis_date, universe, lookbacks, skip_months=skip)
    out_long = compute_momentum_raw(long, analysis_date, universe, lookbacks, skip_months=skip)

    # Labels and index alignment
    assert set(out_wide.keys()) == set(lookbacks.keys())
    assert set(out_long.keys()) == set(lookbacks.keys())
    for k in lookbacks.keys():
        s_w = out_wide[k]
        s_l = out_long[k]
        assert s_w.name == f"momentum_{k}_raw"
        assert s_l.name == f"momentum_{k}_raw"
        # Should compute for subset or all tickers, but indices must be a subset of universe
        assert set(s_w.index).issubset(set(universe))
        assert set(s_l.index).issubset(set(universe))
        # Values finite
        assert np.isfinite(s_w.values).all()
        assert np.isfinite(s_l.values).all()
        # Wide and long path equality for common tickers
        common = s_w.index.intersection(s_l.index)
        pd.testing.assert_series_equal(s_w.loc[common], s_l.loc[common], check_names=False, check_dtype=False)


def test_low_volatility_raw_window_and_missing_data_stability():
    wide = _make_price_panel_wide(periods=120)
    long = _make_price_panel_long(wide)
    # Introduce missing data for one ticker
    mask = long["ticker"] == long["ticker"].unique()[0]
    sel = long.loc[mask, "price"]
    na_idx = sel.sample(frac=0.2, random_state=1).index
    long.loc[na_idx, "price"] = np.nan

    analysis_date = pd.Timestamp(long["date"].max())
    universe = list(wide.columns)
    series = compute_low_volatility_raw(long, analysis_date, universe, lookback_days=63)

    assert isinstance(series, pd.Series)
    assert series.name == "low_volatility_raw"
    # Should only include tickers with enough data (63 valid returns)
    assert set(series.index).issubset(set(universe))
    # All finite real numbers
    assert np.isfinite(series.values).all()


def test_prepare_fcf_yield_raw_actual_vs_imputed_and_invalids():
    fundamentals = pd.DataFrame(
        {
            "ticker": ["A", "B", "C", "D", "E"],
            "NetCFO_TTM": [100.0, 200.0, 300.0, np.nan, 100.0],
            "CapEx_TTM": [50.0, 0.0, np.nan, 10.0, np.nan],
            "NetCFI_TTM": [-30.0, -60.0, -120.0, -5.0, np.nan],
        }
    )
    market_caps = pd.DataFrame(
        {"ticker": ["A", "B", "C", "D", "E"], "market_cap": [1000.0, 2000.0, 4000.0, 0.0, 500.0]}
    )

    logger = logging.getLogger("test_fcf")
    # actual capex used for A; B uses CFI proxy (CapEx_TTM==0); C uses CFI proxy; D filtered (mcap<=0 or NaN CFO); E skipped (no proxy)
    s_default = prepare_fcf_yield_raw(fundamentals, market_caps, use_actual_capex_when_available=True, logger=logger)
    assert s_default.name == "fcf_yield_raw"
    # Expected tickers present: A, B, C. D filtered by mcap=0; E has NaNs preventing computation
    assert set(s_default.index) == {"A", "B", "C"}
    # Spot-check values
    # A: (100-50)/1000 = 0.05
    assert np.isclose(s_default.loc["A"], 0.05, rtol=0, atol=1e-12)
    # B: (200- max(0,-(-60))) / 2000 = (200-60)/2000 = 0.07
    assert np.isclose(s_default.loc["B"], 0.07, rtol=0, atol=1e-12)
    # C: (300-120)/4000 = 180/4000 = 0.045
    assert np.isclose(s_default.loc["C"], 0.045, rtol=0, atol=1e-12)

    # When not using actual capex, A also uses proxy from CFI (30)
    s_proxy = prepare_fcf_yield_raw(fundamentals, market_caps, use_actual_capex_when_available=False)
    assert np.isclose(s_proxy.loc["A"], (100.0 - 30.0) / 1000.0, rtol=0, atol=1e-12)


def test_normalize_f_score_to_unit_edge_cases():
    raw = {
        "X": (3, 9),  # 0.3333
        "Y": (0, 0),  # max=0 -> 0
        "Z": (np.nan, 9),  # non-numeric -> 0
        "W": (5, np.nan),  # non-numeric -> 0
    }
    s = normalize_f_score_to_unit(raw)
    assert s.name == "f_score_normalized"
    assert np.isclose(s.loc["X"], 3 / 9)
    assert s.loc["Y"] == 0.0
    assert s.loc["Z"] == 0.0
    assert s.loc["W"] == 0.0


# Normalization helpers and immutability

def test_make_normalization_frame_drops_nonfinite_and_maps_and_immutable():
    s = pd.Series([1.0, np.nan, 2.0, np.inf, -np.inf], index=["A", "B", "C", "D", "E"], name="val")
    s_before = s.copy()
    sector_map = {"A": "Tech", "B": "Health", "C": "Tech", "E": "Energy"}

    out = make_normalization_frame(s, sector_map, value_column_name="metric_x", sector_column_name="sector_name")

    # Columns and alignment
    assert list(out.columns) == ["ticker", "metric_x", "sector_name"]
    assert set(out["ticker"]) == {"A", "C"}  # dropped NaN/inf/-inf
    # Values and sector mapping preserved
    out_sorted = out.sort_values("ticker").reset_index(drop=True)
    expected = pd.DataFrame({
        "ticker": ["A", "C"],
        "metric_x": [1.0, 2.0],
        "sector_name": ["Tech", "Tech"],
    })
    pd.testing.assert_frame_equal(out_sorted, expected, check_dtype=False)
    # Immutability
    pd.testing.assert_series_equal(s, s_before)


def test_make_normalization_frame_empty_after_drop_returns_empty_frame():
    s = pd.Series([np.nan, np.inf, -np.inf], index=["A", "B", "C"], name="val")
    out = make_normalization_frame(s, {"A": "X"}, value_column_name="metric_x")
    assert out.empty
    assert list(out.columns) == ["ticker", "metric_x", "sector"]


def test_compute_period_return_boundaries_missing_unsorted_and_immutable():
    dates = pd.to_datetime(["2020-01-01", "2020-01-05", "2020-01-10"]) \
        .astype("datetime64[ns]")
    prices = pd.Series([100.0, 110.0, 121.0], index=dates)
    prices_unsorted = prices.sort_index(ascending=False)  # deliberately unsorted
    before = prices_unsorted.copy()

    # Exact boundaries pick on/after start, on/before end
    r_mid = compute_period_return(prices_unsorted, pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-09"))
    assert r_mid == 0.0  # both sides select 2020-01-05

    r_full = compute_period_return(prices_unsorted, pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-10"))
    assert np.isclose(r_full, 0.21, rtol=0, atol=1e-12)

    # Missing start or end price within bounds -> None
    prices_nan = pd.Series([100.0, np.nan, 121.0], index=dates)
    assert compute_period_return(prices_nan, pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-09")) is None

    # Insufficient data windows
    assert compute_period_return(prices, pd.Timestamp("2020-01-11"), pd.Timestamp("2020-01-12")) is None  # start after last
    assert compute_period_return(prices, pd.Timestamp("2019-12-01"), pd.Timestamp("2019-12-31")) is None  # end before first

    # Immutability
    pd.testing.assert_series_equal(prices_unsorted, before)


def test_momentum_raw_edge_cases_empty_universe_and_insufficient_window():
    wide = _make_price_panel_wide(periods=30)
    analysis_date = pd.Timestamp(wide.index[-1])

    # Empty universe -> empty dict
    out_empty = compute_momentum_raw(wide, analysis_date, [], {"3m": 3}, skip_months=1)
    assert out_empty == {}

    # Insufficient window: function should handle gracefully; labels present, index subset of universe; values finite if present
    out_short = compute_momentum_raw(wide, analysis_date, list(wide.columns), {"12m": 12}, skip_months=1)
    assert set(out_short.keys()) == {"12m"}
    s = out_short["12m"]
    assert isinstance(s, pd.Series)
    assert s.name == "momentum_12m_raw"
    assert set(s.index).issubset(set(wide.columns))
    if not s.empty:
        assert np.isfinite(s.values).all()


def test_low_volatility_raw_insufficient_returns_empty_series():
    long = _make_price_panel_long(_make_price_panel_wide(periods=10))
    analysis_date = pd.Timestamp(long["date"].max())
    universe = long["ticker"].unique().tolist()
    series = compute_low_volatility_raw(long, analysis_date, universe, lookback_days=63)
    assert isinstance(series, pd.Series)
    assert series.name == "low_volatility_raw"
    assert series.empty


def test_prepare_fcf_yield_raw_filters_negative_mcap_and_inner_join():
    fundamentals = pd.DataFrame(
        {
            "ticker": ["A", "B", "C", "X_only_in_f"],
            "NetCFO_TTM": [100.0, 200.0, 300.0, 10.0],
            "CapEx_TTM": [10.0, 0.0, np.nan, 1.0],
            "NetCFI_TTM": [-20.0, -50.0, -100.0, -1.0],
        }
    )
    market_caps = pd.DataFrame(
        {
            "ticker": ["A", "B", "C", "Y_only_in_m"],
            "market_cap": [1000.0, -2000.0, 4000.0, 500.0],  # B has negative mcap -> filtered
        }
    )

    s = prepare_fcf_yield_raw(fundamentals, market_caps, use_actual_capex_when_available=True)
    # Only A and C merge; B filtered due to negative mcap; X/Y excluded by inner join
    assert set(s.index) == {"A", "C"}
    # Spot-check A: (100-10)/1000 = 0.09
    assert np.isclose(s.loc["A"], 0.09, rtol=0, atol=1e-12)


def test_utilities_do_not_mutate_inputs():
    # Momentum (wide and long)
    wide = _make_price_panel_wide(periods=120)
    long = _make_price_panel_long(wide)
    wide_before = wide.copy()
    long_before = long.copy()
    analysis_date = pd.Timestamp(wide.index[-1])
    compute_momentum_raw(wide, analysis_date, list(wide.columns), {"3m": 3}, skip_months=1)
    compute_momentum_raw(long, analysis_date, list(wide.columns), {"3m": 3}, skip_months=1)
    pd.testing.assert_frame_equal(wide, wide_before)
    pd.testing.assert_frame_equal(long, long_before)

    # Low-vol
    long2 = long.copy()
    before2 = long2.copy()
    compute_low_volatility_raw(long2, analysis_date, list(wide.columns), lookback_days=63)
    pd.testing.assert_frame_equal(long2, before2)

    # FCF Yield
    fundamentals = pd.DataFrame(
        {
            "ticker": ["A", "B"],
            "NetCFO_TTM": [100.0, 200.0],
            "CapEx_TTM": [10.0, 0.0],
            "NetCFI_TTM": [-20.0, -60.0],
        }
    )
    market_caps = pd.DataFrame({"ticker": ["A", "B"], "market_cap": [1000.0, 2000.0]})
    f_before = fundamentals.copy()
    m_before = market_caps.copy()
    prepare_fcf_yield_raw(fundamentals, market_caps)
    pd.testing.assert_frame_equal(fundamentals, f_before)
    pd.testing.assert_frame_equal(market_caps, m_before)

