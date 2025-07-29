# Regime Switching Methodology Documentation

## Overview

This document summarizes the **regime switching methodology** used throughout the project to dynamically adapt investment strategies based on prevailing market conditions. The approach is central to the dynamic QVM (Quality-Value-Momentum) strategy and other adaptive models in the codebase.

## Methodology

Market regimes are identified using a combination of drawdown, volatility, and trend criteria. The main regimes are:

- **Bear:** Drawdown > 20% from peak
- **Stress:** Rolling volatility in the top quartile
- **Bull:** Price above trend moving average and not Bear/Stress
- **Sideways:** All other conditions

The canonical implementation is as follows (see references for source):

```python
def identify_market_regimes(benchmark_returns: pd.Series, 
                            bear_threshold: float = -0.20,
                            vol_window: int = 60,
                            trend_window: int = 200) -> pd.DataFrame:
    """
    Identifies market regimes using multiple criteria:
    - Bear: Drawdown > 20% from peak
    - Stress: Rolling volatility in top quartile
    - Bull: Price above trend MA and not Bear/Stress
    - Sideways: Everything else
    """
    # ... implementation ...
```

## References

The regime switching methodology is implemented and discussed in the following files:

- **Phase 7: Institutional Backtesting**
  - [`production/tests/phase7_institutional_backtesting/04_deep_dive_attribution_analysis.md`](../../phase7_institutional_backtesting/04_deep_dive_attribution_analysis.md)
    - See the function `identify_market_regimes` and its docstring for detailed logic and explanation.
- **Phase 8: Risk Management**
  - [`production/tests/phase8_risk_management/06_risk_overlay_analysis.md`](../../phase8_risk_management/06_risk_overlay_analysis.md)
    - Contains a validated and referenced copy of the regime identification logic.
- **Archive (Unrestricted Universe)**
  - [`production/tests/archive/phase7_unrestricted_universe/04_deep_dive_attribution_analysis.md`](../../archive/phase7_unrestricted_universe/04_deep_dive_attribution_analysis.md)
  - [`production/tests/archive/phase8_unrestricted_universe/06_risk_overlay_analysis.md`](../../archive/phase8_unrestricted_universe/06_risk_overlay_analysis.md)
- **Jupyter Notebooks**
  - [`production/tests/phase7_institutional_backtesting/04_deep_dive_attribution_analysis.ipynb`](../../phase7_institutional_backtesting/04_deep_dive_attribution_analysis.ipynb)
  - [`production/tests/phase8_risk_management/06_risk_overlay_analysis.ipynb`](../../phase8_risk_management/06_risk_overlay_analysis.ipynb)

## Usage in Dynamic QVM and Other Strategies

The regime classification is used to dynamically adjust factor weights and risk overlays in strategies such as the dynamic QVM model. For further details on how regime switching integrates with composite models, see:

- [`production/tests/phase15_composite_model_engineering/15b_dynamic_composite_backtest.md`](../../phase15_composite_model_engineering/15b_dynamic_composite_backtest.md)
- [`production/tests/phase16_weighted_composite_model/16_weighted_composite_engineering.md`](../../phase16_weighted_composite_model/16_weighted_composite_engineering.md)

---

_Last updated: 2024-06-11_