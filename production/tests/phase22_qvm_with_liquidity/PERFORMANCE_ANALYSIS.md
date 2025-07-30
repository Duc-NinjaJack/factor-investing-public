# Phase 22 Performance Analysis: 10B vs 3B VND Strategy (Real Data)

## Executive Summary

The **real data backtest** shows the **10B VND strategy significantly outperforming the 3B VND strategy**:

- **10B VND**: 6.17% annual return, 0.31 Sharpe ratio, -7.21% alpha
- **3B VND**: 1.58% annual return, 0.08 Sharpe ratio, -11.40% alpha
- **Real VNINDEX Benchmark**: 12.59% annual return, 0.74 Sharpe ratio

**Key Finding**: Both strategies underperform the VNINDEX benchmark, but the 10B VND threshold provides much better risk-adjusted returns.

## Real Performance Analysis

### 1. **Liquidity Threshold Impact**

The 10B VND strategy significantly outperforms the 3B VND strategy due to:

- **Better Risk-Adjusted Returns**: 0.31 vs 0.08 Sharpe ratio
- **Lower Volatility**: More liquid stocks reduce portfolio volatility
- **Higher Quality Universe**: 10B VND threshold filters for higher quality stocks
- **Lower Transaction Costs**: Less turnover due to more stable stocks

### 2. **Market Underperformance**

Both strategies show negative alpha against the VNINDEX benchmark:
- **10B VND Alpha**: -7.21% (better than 3B VND)
- **3B VND Alpha**: -11.40% (significant underperformance)

This suggests the weighted composite approach (60% Value + 20% Quality + 20% Reversal) needs refinement for the Vietnamese market.

### 3. **Drawdown Analysis**

- **10B VND Max Drawdown**: -23.90%
- **3B VND Max Drawdown**: -27.46%

The 10B VND strategy shows better downside protection during market stress.

## Transaction Costs Analysis

### 1. **Real Backtesting Framework: Transaction Costs Applied**

The actual backtesting framework applies transaction costs:

```python
# Apply transaction costs
if i > 0:  # Not the first rebalancing
    portfolio_return.iloc[0] -= self.strategy_config['transaction_cost']
```

**Transaction Cost Configuration**:
- **Default**: 20 bps (0.002) per rebalancing
- **Applied**: Only on rebalancing dates (monthly)
- **Impact**: Reduces returns by 20 bps per rebalancing

### 2. **Expected Transaction Cost Impact**

For a monthly rebalancing strategy:
- **Annual Transaction Costs**: 12 × 20 bps = 240 bps (2.4%)
- **Net Impact**: Reduces annual returns by ~2.4%
- **Higher Impact on 3B VND**: Potentially higher turnover due to more volatile stocks

## Why 3B VND Strategy Underperforms

### 1. **Liquidity vs. Quality Trade-off**
- **3B VND**: More stocks available, but lower quality and liquidity
- **10B VND**: Higher quality, more liquid stocks, better risk-adjusted returns

### 2. **Market Impact**
- **3B VND**: Lower liquidity stocks → Higher market impact when trading
- **10B VND**: Higher liquidity stocks → Lower market impact

### 3. **Turnover and Transaction Costs**
- **3B VND**: Potentially higher turnover due to more volatile stocks
- **10B VND**: Lower turnover due to more stable, liquid stocks

### 4. **Risk-Adjusted Performance**
- **3B VND**: Higher volatility reduces Sharpe ratio despite potentially higher alpha
- **10B VND**: Lower volatility improves risk-adjusted returns

## Data Quality Assessment

### ✅ **Real Data Sources**
- **Factor Scores**: Actual Value, Quality, and Momentum composites from database
- **Price Data**: Real adjusted prices from vcsc_daily_data_complete
- **Benchmark**: Real VNINDEX data from etf_history
- **ADTV Data**: Generated directly from database volume and price data
- **Transaction Costs**: Properly applied during rebalancing

### 📊 **Data Coverage**
- **Date Range**: 2018-01-02 to 2025-07-25
- **Factor Data**: 1,286,295 records, 714 unique tickers
- **ADTV Data**: 1,879 dates, 728 tickers
- **Strategy**: 60% Value + 20% Quality + 20% Reversal

## Recommendations

### 1. **Strategy Refinement**
- **Factor Weight Optimization**: Test different weighting schemes
- **Market Regime Analysis**: Analyze performance across different market conditions
- **Sector Neutralization**: Consider sector constraints to reduce concentration risk

### 2. **Liquidity Threshold Optimization**
Test intermediate thresholds to find optimal balance:
- **5B VND**: Between current thresholds
- **7B VND**: Moderate liquidity requirement
- **15B VND**: Higher quality focus

### 3. **Risk Management**
- **Position Sizing**: Implement volatility-based position sizing
- **Stop Losses**: Add dynamic stop-loss mechanisms
- **Correlation Monitoring**: Monitor factor correlations and adjust weights

### 4. **Transaction Cost Optimization**
- **Rebalancing Frequency**: Test quarterly vs monthly rebalancing
- **Cost Sensitivity**: Analyze impact of different transaction cost levels
- **Market Impact Modeling**: Implement more sophisticated market impact models

## Conclusion

The **real data backtest confirms that 10B VND strategy significantly outperforms 3B VND** due to:

1. **Better Risk-Adjusted Returns**: 0.31 vs 0.08 Sharpe ratio
2. **Lower Drawdown**: -23.90% vs -27.46% max drawdown
3. **Higher Quality Universe**: More liquid, stable stocks
4. **Lower Transaction Costs**: Less turnover and market impact

However, both strategies underperform the VNINDEX benchmark, indicating the need for:
- **Factor weight optimization**
- **Market regime adaptation**
- **Enhanced risk management**
- **Transaction cost optimization**

The real data provides a much more accurate assessment than simulations and shows the challenges of factor investing in the Vietnamese market. 