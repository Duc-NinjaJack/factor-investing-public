# Phase 22: Pure Value with Liquidity - Weighted Composite Real Data Backtesting

## 🎯 Overview

Phase 22 implements the successful **weighted composite strategy** from Phase 16 using **real market data** from the database with proper **liquidity filtering**. This combines the best of both worlds:

- **Phase 16 Methodology**: 60% Value + 20% Quality + 20% Reversal (Momentum inverted)
- **Real Data Framework**: Actual price data from database with transaction costs
- **Database API**: Robust data access using the production database connection manager

## 🚀 Key Features

### Strategy Components
- **Weighted Composite**: 60% Value + 20% Quality + 20% Reversal
- **Z-score Normalization**: Within universe for each factor
- **Quintile 5 Selection**: Top 20% of stocks by weighted composite score
- **Equal Weighting**: Within portfolio for diversification
- **Real Price Data**: From `vcsc_daily_data_complete` table
- **ADTV Liquidity Filtering**: Configurable thresholds (10B VND, 3B VND)
- **Transaction Cost Modeling**: 20 bps per trade
- **Monthly Rebalancing**: Consistent with institutional standards

### Technical Implementation
- **Database Integration**: Uses `production/database/connection.py` API
- **Factor Data**: Loads individual composites from `factor_scores_qvm` table
- **Performance Metrics**: Comprehensive risk-adjusted returns analysis
- **Visualization**: 9-panel performance dashboard
- **Reporting**: Detailed text and visual reports

## 📊 Methodology

### 1. Factor Construction
```python
# Individual factor scores loaded from database
Quality_Composite, Value_Composite, Momentum_Composite

# Momentum Reversal factor created
Momentum_Reversal = -1 * Momentum_Composite

# Z-score normalization within universe
for factor in [Quality, Value, Reversal]:
    factor_Z = (factor - mean) / std

# Weighted composite calculation
Weighted_Composite = 0.6 * Value_Z + 0.2 * Quality_Z + 0.2 * Reversal_Z
```

### 2. Portfolio Construction
```python
# Liquidity filtering
liquid_stocks = adtv_scores[adtv_scores >= threshold].index

# Factor filtering
available_stocks = factor_scores.intersection(liquid_stocks)

# Quintile 5 selection (top 20%)
q5_cutoff = weighted_composite.quantile(0.8)
top_stocks = weighted_composite[weighted_composite >= q5_cutoff]

# Equal weighting
weights = 1.0 / len(top_stocks)
```

### 3. Performance Calculation
```python
# Portfolio returns
portfolio_return = (returns * weights).sum(axis=1)

# Transaction costs
costs = turnover * 0.002  # 20 bps

# Net returns
net_returns = portfolio_return - costs
```

## 🛠️ Installation & Setup

### Prerequisites
```bash
# Required Python packages
pip install pandas numpy matplotlib seaborn sqlalchemy pymysql pyyaml

# Database access
# Ensure database.yml configuration is properly set up
```

### Directory Structure
```
production/tests/phase22_pure_value_with_liquidity/
├── 22_weighted_composite_real_data_backtest.py  # Main backtest script
├── README.md                                    # This file
└── results/                                     # Generated results (auto-created)
```

## 🚀 Usage

### Command Line Interface
```bash
# Basic usage
cd production/tests/phase22_pure_value_with_liquidity
python 22_weighted_composite_real_data_backtest.py

# With custom configuration
python 22_weighted_composite_real_data_backtest.py \
    --config ../../../config/database.yml \
    --pickle ../../../data/unrestricted_universe_data.pkl

# Skip saving outputs
python 22_weighted_composite_real_data_backtest.py --no-plots --no-report
```

### Programmatic Usage
```python
from 22_weighted_composite_real_data_backtest import WeightedCompositeBacktesting

# Initialize backtesting engine
backtesting = WeightedCompositeBacktesting(
    config_path='../../../config/database.yml',
    pickle_path='../../../data/unrestricted_universe_data.pkl'
)

# Run complete analysis
results = backtesting.run_complete_analysis(
    save_plots=True,
    save_report=True
)

# Access results
for threshold, result in results.items():
    print(f"{threshold}: {result['metrics']['sharpe_ratio']:.2f} Sharpe")
```

## 📈 Outputs

### Generated Files
1. **Performance Plots**: `weighted_composite_backtesting_plots_YYYYMMDD_HHMMSS.png`
   - 9-panel comprehensive visualization
   - Cumulative returns, drawdowns, rolling Sharpe ratios
   - Performance metrics comparison
   - Monthly returns heatmap
   - Portfolio evolution analysis
   - Risk-return scatter plot
   - Benchmark comparison
   - Factor weights visualization

2. **Detailed Report**: `weighted_composite_backtesting_report_YYYYMMDD_HHMMSS.txt`
   - Strategy overview and methodology
   - Summary statistics for all thresholds
   - Detailed performance analysis
   - Risk metrics and attribution
   - Recommendations and conclusions

### Performance Metrics
- **Return Metrics**: Annual return, total return, benchmark return
- **Risk Metrics**: Annual volatility, maximum drawdown, VaR
- **Risk-Adjusted Metrics**: Sharpe ratio, Calmar ratio, information ratio
- **Attribution Metrics**: Alpha, beta, tracking error
- **Transaction Metrics**: Turnover, transaction costs

## 🔧 Configuration

### Strategy Parameters
```python
# Weighting scheme (from Phase 16)
weighting_scheme = {
    'Value': 0.6,      # 60% Value factor
    'Quality': 0.2,    # 20% Quality factor  
    'Reversal': 0.2    # 20% Reversal factor (inverted momentum)
}

# Portfolio construction
strategy_config = {
    'portfolio_size': 25,                    # Target portfolio size
    'quintile_selection': 0.8,              # Top 20% selection
    'rebalance_freq': 'M',                  # Monthly rebalancing
    'transaction_cost': 0.002,              # 20 bps transaction cost
    'initial_capital': 100_000_000          # 100M VND initial capital
}

# Liquidity thresholds
thresholds = {
    '10B_VND': 10_000_000_000,  # 10B VND ADTV
    '3B_VND': 3_000_000_000     # 3B VND ADTV
}
```

### Database Configuration
```yaml
# config/database.yml
production:
  host: your_database_host
  port: 3306
  username: your_username
  password: your_password
  schema_name: your_database_name
```

## 📊 Expected Results

Based on Phase 16 analysis, the weighted composite strategy should show:

### Performance Expectations
- **Annual Return**: 10-15% (depending on liquidity threshold)
- **Sharpe Ratio**: 0.4-0.6 (risk-adjusted performance)
- **Maximum Drawdown**: 25-35% (risk management)
- **Alpha Generation**: 2-5% vs VNINDEX benchmark

### Liquidity Threshold Impact
- **10B VND**: Higher quality, lower universe, potentially better risk-adjusted returns
- **3B VND**: Larger universe, more opportunities, potentially higher returns with higher risk

## 🔍 Analysis & Interpretation

### Key Performance Indicators
1. **Sharpe Ratio > 0.5**: Good risk-adjusted performance
2. **Alpha > 2%**: Significant outperformance vs benchmark
3. **Max Drawdown < 30%**: Acceptable risk levels
4. **Information Ratio > 0.3**: Consistent alpha generation

### Strategy Validation
- **Factor Efficacy**: Value factor should dominate performance
- **Liquidity Impact**: Lower thresholds may show higher returns but higher risk
- **Transaction Costs**: Net returns should account for realistic implementation costs
- **Regime Stability**: Performance should be consistent across market conditions

## 🚨 Troubleshooting

### Common Issues
1. **Database Connection**: Ensure database.yml is properly configured
2. **Data Availability**: Check factor_scores_qvm table has sufficient data
3. **Memory Usage**: Large datasets may require optimization
4. **Path Issues**: Ensure correct relative paths for imports

### Debug Mode
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with verbose output
backtesting = WeightedCompositeBacktesting()
backtesting.run_complete_analysis()
```

## 📚 References

### Related Documentation
- **Phase 16**: Weighted composite methodology and results
- **Real Data Backtesting**: Framework documentation
- **Database API**: Connection management and data access
- **Factor Methodology**: QVM factor construction details

### Key Papers & Research
- Fama-French Three Factor Model
- Momentum Reversal Effects
- Liquidity Premium in Emerging Markets
- Transaction Cost Impact on Factor Strategies

## 🤝 Contributing

### Development Guidelines
1. **Code Style**: Follow PEP 8 standards
2. **Documentation**: Update README for any changes
3. **Testing**: Validate results against known benchmarks
4. **Performance**: Optimize for large datasets

### Future Enhancements
- **Dynamic Weighting**: Regime-based factor allocation
- **Risk Management**: Position sizing and stop-losses
- **Alternative Factors**: Additional factor combinations
- **Multi-Asset**: Extension to other asset classes

---

**Status**: ✅ Production Ready  
**Last Updated**: January 2025  
**Maintainer**: Quantitative Strategy Team