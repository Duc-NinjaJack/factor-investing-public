# Real Data Backtesting Engine - Documentation

## Overview

The Real Data Backtesting Engine is a clean, refactored version of the backtesting functionality that was previously embedded in the test notebooks. This engine provides a comprehensive framework for running backtests using real market data from the database.

## Features

- **Real Market Data**: Uses actual price data from `vcsc_daily_data_complete`
- **Factor Scores**: Incorporates QVM composite scores from `factor_scores_qvm`
- **Liquidity Filtering**: Applies ADTV-based liquidity filters
- **No Short-Selling**: Enforces long-only portfolio constraints
- **Transaction Costs**: Models realistic trading costs (20 bps default)
- **Performance Metrics**: Comprehensive risk and return analysis
- **Visualization**: Rich plotting and charting capabilities
- **Reporting**: Detailed performance reports and recommendations

## Files

### Core Engine
- `real_data_backtesting.py` - Main backtesting engine class
- `run_real_data_backtesting.py` - Usage examples and demonstration script

### Original Files (for reference)
- `production/tests/3b_backtesting/09_full_backtesting_real_data.py` - Original implementation

## Quick Start

### 1. Basic Usage

```python
from real_data_backtesting import RealDataBacktesting

# Initialize with default settings
backtesting = RealDataBacktesting()

# Run complete analysis
results = backtesting.run_complete_analysis(
    save_plots=True,
    save_report=True
)
```

### 2. Command Line Usage

```bash
# Basic usage
python real_data_backtesting.py

# Custom configuration file
python real_data_backtesting.py --config path/to/config.yml

# Custom ADTV data file
python real_data_backtesting.py --pickle path/to/adtv_data.pkl

# Skip saving plots and reports
python real_data_backtesting.py --no-plots --no-report
```

### 3. Interactive Examples

```bash
python run_real_data_backtesting.py
```

This will present an interactive menu with different usage examples.

## Configuration

### Default Settings

```python
# Liquidity thresholds
thresholds = {
    '10B_VND': 10_000_000_000,
    '3B_VND': 3_000_000_000
}

# Backtest configuration
backtest_config = {
    'start_date': '2018-01-01',
    'end_date': '2025-01-01',
    'rebalance_freq': 'M',  # Monthly rebalancing
    'portfolio_size': 25,
    'max_sector_weight': 0.4,
    'transaction_cost': 0.002,  # 20 bps
    'initial_capital': 100_000_000  # 100M VND
}
```

### Custom Configuration

```python
# Initialize with custom settings
backtesting = RealDataBacktesting()

# Customize thresholds
backtesting.thresholds = {
    '5B_VND': 5_000_000_000,
    '2B_VND': 2_000_000_000
}

# Customize backtest parameters
backtesting.backtest_config.update({
    'portfolio_size': 30,
    'transaction_cost': 0.003,  # 30 bps
    'rebalance_freq': 'W',  # Weekly rebalancing
    'start_date': '2020-01-01'
})
```

## Data Requirements

### Required Data Sources

1. **Price Data**: `vcsc_daily_data_complete` table
   - Columns: `trading_date`, `ticker`, `close_price_adjusted`

2. **Factor Scores**: `factor_scores_qvm` table
   - Columns: `date`, `ticker`, `QVM_Composite`

3. **Benchmark Data**: `etf_history` table
   - Columns: `date`, `close` (for VNINDEX)

4. **ADTV Data**: Pickle file (`unrestricted_universe_data.pkl`)
   - Contains ADTV data for liquidity filtering

### Database Configuration

The engine automatically looks for database configuration in:
1. `config/database.yml`
2. `config/config.ini`
3. Parent directory config files

## Backtesting Methodology

### Portfolio Construction

1. **Universe Selection**: All stocks with factor scores
2. **Liquidity Filter**: Apply ADTV threshold (3B or 10B VND)
3. **Stock Selection**: Top N stocks by QVM composite score
4. **Weighting**: Equal weight portfolio (long-only)
5. **Rebalancing**: Monthly (configurable)

### Performance Calculation

1. **Returns**: Daily portfolio returns
2. **Transaction Costs**: Applied at rebalancing
3. **Metrics**: Sharpe ratio, drawdown, alpha, beta, etc.
4. **Benchmark**: VNINDEX comparison

### Key Assumptions

- **No Short-Selling**: Long-only positions
- **Equal Weighting**: Equal allocation to selected stocks
- **Monthly Rebalancing**: Fixed rebalancing schedule
- **Transaction Costs**: 20 bps per trade (configurable)
- **No Market Impact**: Assumes no price impact from trades

## Performance Metrics

### Return Metrics
- **Annual Return**: Annualized portfolio return
- **Total Return**: Cumulative return over period
- **Benchmark Return**: VNINDEX return for comparison

### Risk Metrics
- **Annual Volatility**: Annualized portfolio volatility
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Beta**: Market sensitivity

### Risk-Adjusted Metrics
- **Sharpe Ratio**: Return per unit of risk
- **Calmar Ratio**: Return per unit of maximum drawdown
- **Information Ratio**: Alpha per unit of tracking error

### Attribution Metrics
- **Alpha**: Excess return vs benchmark
- **Beta**: Market exposure
- **Turnover**: Portfolio turnover rate

## Output Files

### Generated Files

1. **Plots**: `real_data_backtesting_plots_YYYYMMDD_HHMMSS.png`
   - Cumulative returns comparison
   - Drawdown analysis
   - Rolling Sharpe ratio
   - Performance metrics comparison
   - Monthly returns heatmap
   - Portfolio holdings evolution
   - Risk-return scatter
   - Benchmark comparison
   - Turnover analysis

2. **Report**: `real_data_backtesting_report_YYYYMMDD_HHMMSS.txt`
   - Summary statistics
   - Detailed analysis
   - Performance recommendations

## API Reference

### RealDataBacktesting Class

#### Initialization
```python
RealDataBacktesting(config_path=None, pickle_path=None)
```

#### Methods

##### `load_data()`
Load all required data from database and pickle files.

##### `prepare_data_for_backtesting(data)`
Prepare and align data for backtesting.

##### `run_backtest(threshold_name, threshold_value, prepared_data)`
Run backtest for a specific liquidity threshold.

##### `run_comparative_backtests(data)`
Run backtests for all configured thresholds.

##### `create_performance_visualizations(backtest_results, save_path=None)`
Create comprehensive performance charts.

##### `generate_comprehensive_report(backtest_results)`
Generate detailed performance report.

##### `run_complete_analysis(save_plots=True, save_report=True)`
Run complete backtesting analysis with all outputs.

## Examples

### Example 1: Basic Usage
```python
from real_data_backtesting import RealDataBacktesting

backtesting = RealDataBacktesting()
results = backtesting.run_complete_analysis()
```

### Example 2: Custom Configuration
```python
backtesting = RealDataBacktesting()

# Custom thresholds
backtesting.thresholds = {
    '5B_VND': 5_000_000_000,
    '2B_VND': 2_000_000_000
}

# Custom parameters
backtesting.backtest_config.update({
    'portfolio_size': 30,
    'transaction_cost': 0.003,
    'rebalance_freq': 'W'
})

results = backtesting.run_complete_analysis()
```

### Example 3: Step-by-Step Control
```python
backtesting = RealDataBacktesting()

# Load data
data = backtesting.load_data()

# Prepare data
prepared_data = backtesting.prepare_data_for_backtesting(data)

# Run individual backtest
result = backtesting.run_backtest('3B_VND', 3_000_000_000, prepared_data)

# Create visualizations
backtesting.create_performance_visualizations({'3B_VND': result})

# Generate report
report = backtesting.generate_comprehensive_report({'3B_VND': result})
print(report)
```

## Troubleshooting

### Common Issues

1. **Database Connection Error**
   - Check database configuration file
   - Verify database credentials
   - Ensure database is accessible

2. **Missing Pickle File**
   - Run `get_unrestricted_universe_data.py` first
   - Check pickle file path

3. **Insufficient Data**
   - Verify data exists in database tables
   - Check date ranges
   - Ensure factor scores are available

4. **Memory Issues**
   - Reduce portfolio size
   - Use shorter date ranges
   - Process data in chunks

### Error Messages

- `Database configuration file not found`: Check config file paths
- `Pickle file not found`: Generate ADTV data first
- `Insufficient data for backtesting`: Check data availability
- `No common dates found`: Verify data alignment

## Performance Considerations

### Optimization Tips

1. **Data Loading**: Use appropriate date ranges
2. **Memory Usage**: Process large datasets in chunks
3. **Computation**: Use vectorized operations where possible
4. **Storage**: Save results to avoid recomputation

### Scalability

- **Universe Size**: Handles 1000+ stocks
- **Time Period**: Supports multi-year backtests
- **Portfolio Size**: Configurable (10-100 stocks)
- **Rebalancing**: Flexible frequency options

## Future Enhancements

### Planned Features

1. **Risk Management**: Position sizing and risk controls
2. **Sector Constraints**: Sector weight limits
3. **Alternative Benchmarks**: Multiple benchmark options
4. **Factor Attribution**: Individual factor performance
5. **Monte Carlo**: Statistical significance testing
6. **Real-time Updates**: Live data integration

### Extension Points

- Custom factor models
- Alternative weighting schemes
- Advanced transaction cost models
- Multi-asset class support
- Real-time trading integration

## Support

For questions or issues:

1. Check the troubleshooting section
2. Review the examples in `run_real_data_backtesting.py`
3. Examine the original implementation for reference
4. Check database connectivity and data availability

## Version History

- **v1.0** (January 2025): Initial refactored version
  - Extracted from test notebooks
  - Clean, modular design
  - Comprehensive documentation
  - Usage examples and utilities