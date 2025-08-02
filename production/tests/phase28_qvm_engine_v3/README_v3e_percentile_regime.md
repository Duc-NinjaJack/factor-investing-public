# QVM Engine v3e - Percentile-based Regime Detection

## Overview

QVM Engine v3e introduces a robust percentile-based regime detection system that replaces the hard-coded thresholds used in previous versions. This approach makes the regime detection more adaptive to different market conditions and eliminates the issue of detecting only one regime throughout the backtest period.

## Key Improvements

### 1. Percentile-based Regime Detection

**Problem Solved**: The original implementation used hard-coded thresholds:
```yaml
# Old approach (problematic)
regime:
  volatility_threshold: 0.2659  # Fixed threshold
  return_threshold: 0.2588      # Fixed threshold
  low_return_threshold: 0.2131  # Fixed threshold
```

**New Solution**: Dynamic percentile-based thresholds:
```yaml
# New approach (adaptive)
regime:
  volatility_percentile_high: 75.0  # 75th percentile for high volatility
  return_percentile_high: 75.0      # 75th percentile for high return
  return_percentile_low: 25.0       # 25th percentile for low return
```

### 2. Adaptive Threshold Calculation

The regime detector now:
- Calculates rolling percentiles of volatility and returns over a lookback period
- Uses these percentiles to determine regime thresholds dynamically
- Adapts to changing market conditions automatically
- Ensures proper regime classification across different market environments

### 3. Robust Regime Classification

The regime detection logic now properly identifies three distinct regimes:

- **Momentum Regime**: High volatility (>75th percentile) + High return (>75th percentile)
- **Stress Regime**: High volatility (>75th percentile) + Low return (<25th percentile)  
- **Normal Regime**: Low volatility + Normal return

### 4. Enhanced Portfolio Allocation

Each regime now has appropriate allocation levels:
- Momentum: 80% allocation (high conviction)
- Stress: 30% allocation (risk reduction)
- Normal: 60% allocation (moderate exposure)

## Implementation Details

### RegimeDetector Class

The `RegimeDetector` class has been completely rewritten to:

1. **Maintain Historical Data**: Stores volatility and return history for percentile calculation
2. **Dynamic Threshold Calculation**: Uses rolling percentiles instead of fixed values
3. **Robust Classification**: Implements proper regime classification logic
4. **Fallback Mechanisms**: Handles insufficient data scenarios gracefully

### Key Methods

- `detect_regime()`: Main regime detection logic using percentiles
- `get_regime_allocation()`: Returns appropriate portfolio allocation for each regime
- Historical data management for percentile calculations

## Configuration

The new configuration file `config_v3e_percentile_regime.yml` demonstrates:

- Percentile-based regime parameters
- Portfolio allocation by regime
- Factor calculation parameters
- Universe construction rules
- Performance metrics settings

## Usage

### Running the Engine

```python
# Load configuration
with open('config_v3e_percentile_regime.yml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize engine
engine = QVMEngineV3AdoptedInsights(
    config=config,
    price_data=price_data,
    fundamental_data=fundamental_data,
    returns_matrix=returns_matrix,
    benchmark_returns=benchmark_returns,
    db_engine=db_engine
)

# Run backtest
strategy_returns, diagnostics = engine.run_backtest()
```

### Expected Results

With the percentile-based approach, you should now see:
- Multiple regime changes throughout the backtest period
- Proper regime distribution (not just one regime)
- Adaptive portfolio allocations based on market conditions
- Improved risk-adjusted returns

## Files

- `06_qvm_engine_v3e_fixed.py`: Main Python implementation
- `06_qvm_engine_v3e_fixed.ipynb`: Jupyter notebook version
- `config_v3e_percentile_regime.yml`: Configuration file
- `README_v3e_percentile_regime.md`: This documentation

## Validation

To validate the regime detection is working properly:

1. Check the regime distribution in the diagnostics output
2. Verify that multiple regimes are detected over time
3. Confirm that portfolio allocations change with regime shifts
4. Review the performance metrics for each regime

The regime distribution should show a reasonable mix of momentum, stress, and normal regimes rather than being dominated by a single regime. 