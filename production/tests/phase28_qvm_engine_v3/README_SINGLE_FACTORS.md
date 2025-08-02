# Single Factor Strategies for QVM Engine v3

This directory contains single factor strategy implementations for Quality (Q), Value (V), and Momentum (M) factors to complement the main QVM composite strategy.

## Files Overview

### Core Implementation
- **`single_factors.py`** - Main implementation of single factor strategies
- **`single_factor_cell.py`** - Execution script for running single factor analysis

### Strategy Classes

#### 1. QualityFactorEngine
- **Factor**: ROAA (Return on Average Assets)
- **Method**: Normalized ROAA scores within universe
- **Signal**: Higher ROAA = better quality

#### 2. ValueFactorEngine  
- **Factor**: P/E Ratio (Price-to-Earnings)
- **Method**: Inverse P/E scoring (lower P/E = better value)
- **Signal**: Lower P/E = better value

#### 3. MomentumFactorEngine
- **Factor**: Multi-horizon momentum (1M, 3M, 6M, 12M)
- **Method**: Equal-weighted momentum across horizons
- **Signal**: Higher momentum = better performance

## Usage Instructions

### Option 1: Run as Standalone Script
After running the main QVM strategy in the notebook, execute:

```python
# Import the execution function
from single_factor_cell import execute_single_factor_analysis

# Run single factor analysis
comparison_df = execute_single_factor_analysis(
    QVM_CONFIG=QVM_CONFIG,
    price_data_raw=price_data_raw,
    fundamental_data_raw=fundamental_data_raw,
    daily_returns_matrix=daily_returns_matrix,
    benchmark_returns=benchmark_returns,
    engine=engine,
    qvm_net_returns=qvm_net_returns
)
```

### Option 2: Manual Execution
Import and run individual strategies:

```python
from single_factors import QualityFactorEngine, ValueFactorEngine, MomentumFactorEngine

# Quality Factor
quality_engine = QualityFactorEngine(config, price_data, fundamental_data, returns_matrix, benchmark_returns, db_engine)
quality_returns, _ = quality_engine.run_backtest()

# Value Factor  
value_engine = ValueFactorEngine(config, price_data, fundamental_data, returns_matrix, benchmark_returns, db_engine)
value_returns, _ = value_engine.run_backtest()

# Momentum Factor
momentum_engine = MomentumFactorEngine(config, price_data, fundamental_data, returns_matrix, benchmark_returns, db_engine)
momentum_returns, _ = momentum_engine.run_backtest()
```

## Expected Outputs

### 1. Performance Comparison Table
Shows key metrics for all strategies:
- Annualized Return (%)
- Annualized Volatility (%)
- Sharpe Ratio
- Max Drawdown (%)

### 2. Visualization Plots
- **Cumulative Performance**: Growth of 1 VND over time
- **Annual Returns**: Year-by-year performance comparison
- **Risk-Return Profile**: Scatter plot of volatility vs return
- **Sharpe Ratio Comparison**: Bar chart of risk-adjusted returns

### 3. Factor Effectiveness Analysis
- Performance summary for each factor
- Key insights about factor behavior
- Comparison with QVM composite strategy

## Configuration

The single factor strategies use the same configuration as the main QVM strategy:
- Universe filters (liquidity, market cap)
- Rebalancing frequency (monthly)
- Transaction costs (30bps)
- Portfolio size (20 stocks)

## Key Features

### Look-ahead Bias Prevention
- 45-day lag for fundamental data
- Skip month for momentum calculations
- Proper date alignment

### Risk Management
- Regime-based allocation (same as QVM composite)
- Position size limits
- Sector concentration limits

### Factor Calculation
- **Quality**: Normalized ROAA within universe
- **Value**: Inverse P/E ratio scoring
- **Momentum**: Multi-horizon equal-weighted momentum

## Validation Benefits

Running single factor strategies provides:

1. **Factor Validation**: Verify individual factor effectiveness
2. **Overfitting Detection**: Compare single vs composite performance
3. **Benchmark Comparison**: Understand factor contributions
4. **Risk Analysis**: Assess factor-specific risk profiles

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure `single_factors.py` is in the same directory
2. **Data Issues**: Verify all required variables are available
3. **Memory Issues**: Large universes may require more memory

### Error Messages
- "Universe too small": Increase liquidity thresholds
- "No factor data": Check fundamental data availability
- "No qualified stocks": Adjust entry criteria

## Performance Expectations

Based on factor theory and historical research:

- **Quality Factor**: Moderate returns, low volatility
- **Value Factor**: Higher returns, moderate volatility  
- **Momentum Factor**: Variable returns, higher volatility
- **QVM Composite**: Balanced performance across regimes

## Next Steps

After running single factor analysis:

1. **Regime Analysis**: Examine factor performance by market regime
2. **Weight Optimization**: Adjust factor weights based on results
3. **Risk Overlay**: Implement additional risk management if needed
4. **Out-of-Sample Testing**: Validate results on different periods 