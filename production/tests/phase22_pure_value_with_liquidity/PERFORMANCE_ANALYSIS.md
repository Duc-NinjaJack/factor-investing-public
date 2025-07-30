# Phase 22 Performance Analysis: 10B vs 3B VND Strategy

## Executive Summary

The tearsheet comparison shows the **10B VND strategy outperforming the 3B VND strategy** with:
- **10B VND**: 17.66% annual return, 0.55 Sharpe ratio, 15.85% alpha
- **3B VND**: 10.90% annual return, 0.23 Sharpe ratio, 9.04% alpha

However, **this was a simulated tearsheet, not a real backtest**. The performance difference is artificially created and doesn't reflect actual market behavior.

## Why 10B VND Appears to Outperform 3B VND

### 1. **Simulation Parameters (Artificial)**

In the simulated tearsheet (`run_tearsheet_real_benchmark.py`), the performance difference was artificially created:

```python
# 10B VND strategy: moderate alpha, lower volatility
strategy_10b_alpha = 0.0002  # ~5% annual alpha
strategy_10b_vol = benchmark_vol * 1.15  # 15% higher volatility

# 3B VND strategy: higher alpha, higher volatility  
strategy_3b_alpha = 0.0003  # ~7.5% annual alpha
strategy_3b_vol = benchmark_vol * 1.25  # 25% higher volatility
```

**Key Issue**: The 3B VND strategy was designed with HIGHER alpha (7.5% vs 5%) but also HIGHER volatility (25% vs 15% above benchmark). The higher volatility significantly impacted the Sharpe ratio calculation.

### 2. **Market Pattern Amplification**

The simulation applied market patterns that amplified the volatility difference:

```python
# COVID crash period - 3B VND performed 15% worse vs 10% worse for 10B VND
strategy_3b_returns[covid_mask] = covid_returns * 1.15  # 15% worse
strategy_10b_returns[covid_mask] = covid_returns * 1.1  # 10% worse

# Inflation period - 3B VND performed 10% worse vs 5% worse for 10B VND  
strategy_3b_returns[inflation_mask] = inflation_returns * 1.1  # 10% worse
strategy_10b_returns[inflation_mask] = inflation_returns * 1.05  # 5% worse
```

This created a **"performance killer"** effect where the 3B VND strategy suffered more during market stress periods.

## Transaction Costs Analysis

### 1. **Simulated Tearsheet: NO Transaction Costs Applied**

The `run_tearsheet_real_benchmark.py` script:
- ✅ Mentions "Transaction Costs: 20 bps" in the summary
- ❌ **Does NOT actually apply transaction costs** to the simulated returns
- ❌ Uses purely simulated data, not real backtest results

### 2. **Real Backtesting Framework: Transaction Costs ARE Applied**

The actual backtesting framework (`22_weighted_composite_real_data_backtest.py`) **DOES apply transaction costs**:

```python
# Apply transaction costs
if i > 0:  # Not the first rebalancing
    portfolio_return.iloc[0] -= self.strategy_config['transaction_cost']
```

**Transaction Cost Configuration**:
- **Default**: 20 bps (0.002) per rebalancing
- **Applied**: Only on rebalancing dates (monthly)
- **Impact**: Reduces returns by 20 bps per rebalancing

### 3. **Expected Transaction Cost Impact**

For a monthly rebalancing strategy:
- **Annual Transaction Costs**: 12 × 20 bps = 240 bps (2.4%)
- **Net Impact**: Reduces annual returns by ~2.4%
- **Higher Impact on 3B VND**: Potentially higher turnover due to more volatile stocks

## Real vs. Simulated Performance

### **Simulated Results (Current Tearsheet)**
```
10B VND: 17.66% return, 0.55 Sharpe, 15.85% alpha
3B VND:  10.90% return, 0.23 Sharpe,  9.04% alpha
```

### **Expected Real Results (with Transaction Costs)**
```
10B VND: ~15.26% return (17.66% - 2.4% transaction costs)
3B VND:  ~8.50% return  (10.90% - 2.4% transaction costs)
```

## Why 3B VND Strategy Might Actually Underperform

### 1. **Liquidity vs. Quality Trade-off**
- **3B VND**: More stocks available, potentially higher alpha
- **10B VND**: Higher quality, more liquid stocks, lower volatility

### 2. **Market Impact**
- **3B VND**: Lower liquidity stocks → Higher market impact when trading
- **10B VND**: Higher liquidity stocks → Lower market impact

### 3. **Turnover and Transaction Costs**
- **3B VND**: Potentially higher turnover due to more volatile stocks
- **10B VND**: Lower turnover due to more stable, liquid stocks

### 4. **Risk-Adjusted Performance**
- **3B VND**: Higher volatility reduces Sharpe ratio despite higher alpha
- **10B VND**: Lower volatility improves risk-adjusted returns

## Recommendations

### 1. **Run Real Backtest**
To get accurate performance comparison, run the actual backtesting framework:

```bash
python 22_weighted_composite_real_data_backtest.py
```

### 2. **Transaction Cost Sensitivity Analysis**
Test different transaction cost levels:
- 10 bps (low cost environment)
- 20 bps (current assumption)
- 30 bps (high cost environment)

### 3. **Liquidity Threshold Optimization**
Test intermediate thresholds:
- 5B VND
- 7B VND
- 15B VND

### 4. **Market Regime Analysis**
Analyze performance across different market conditions:
- Bull markets
- Bear markets
- High volatility periods
- Low volatility periods

## Conclusion

The current tearsheet shows **artificial performance differences** based on simulated parameters. To get real insights:

1. **Run actual backtests** with real market data
2. **Apply proper transaction costs** (already implemented in framework)
3. **Test multiple liquidity thresholds** to find optimal balance
4. **Consider market impact** and slippage for lower liquidity stocks

The **10B VND strategy likely outperforms 3B VND** due to:
- Better risk-adjusted returns (lower volatility)
- Lower transaction costs (less turnover)
- Higher quality, more liquid stocks
- Lower market impact

However, the exact magnitude of outperformance requires real backtesting with actual market data. 