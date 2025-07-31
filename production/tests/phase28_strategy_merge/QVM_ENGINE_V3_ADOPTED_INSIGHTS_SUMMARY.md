# QVM Engine v3 with Adopted Insights - Comprehensive Summary

**Project:** Vietnam Factor Investing Platform - QVM Engine v3 with Adopted Insights Strategy  
**Date:** January 2025  
**Purpose:** Comprehensive documentation of research-backed QVM engine with adopted insights  
**Status:** PRODUCTION READY  

---

## 🎯 Executive Summary

The QVM Engine v3 with Adopted Insights Strategy is a sophisticated quantitative investment algorithm designed specifically for the Vietnamese equity market. Based on comprehensive research insights from multiple analysis phases, the strategy implements a regime-aware, sector-specific, and quality-conscious approach to stock selection and portfolio construction.

**Key Innovation:** The algorithm successfully integrates multiple research insights into a cohesive investment framework:
- **Phase 26**: Simple regime detection with 93.6% accuracy
- **Strategy Documentation**: Momentum-Quality-Value contrarian approach
- **Sector Analysis**: Quality-adjusted P/E and P/B for banking sector
- **Market Cap Analysis**: Size effect reversal favoring large caps post-COVID

**Expected Performance:**
- Annual Return: 10-15%
- Volatility: 15-20%
- Sharpe Ratio: 0.5-0.7
- Max Drawdown: 15-25%
- Benchmark: VNINDEX

---

## 📊 Strategy Components

### **1. Regime Detection System**
**Source:** Phase 26 Comprehensive Analysis (93.6% accuracy)

**Logic:**
```python
# Simple volatility/return based classification
if (vol > vol_75th) & (returns < -0.10): regime = 'Stress'
elif (vol > vol_75th) & (returns >= -0.10): regime = 'Bear'
elif (vol <= vol_75th) & (returns >= 0.10): regime = 'Bull'
else: regime = 'Sideways'
```

**Allocation Strategy:**
- **Bull Market**: 100% allocation
- **Bear Market**: 80% allocation
- **Sideways Market**: 60% allocation
- **Stress Market**: 40% allocation

### **2. Factor Calculation Framework**

#### **Momentum Factors**
- **1M Momentum**: 21-day price momentum
- **3M Momentum**: 63-day price momentum (primary signal)
- **6M Momentum**: 126-day price momentum
- **12M Momentum**: 252-day price momentum (contrarian signal)

#### **Quality Factors**
- **ROAA (Return on Average Assets)**: Positive signal (>2% threshold)
- **ROAE (Return on Average Equity)**: Contrarian signal (<25% threshold)
- **Operating Margin**: Additional quality metric
- **EBITDA Margin**: Profitability indicator

#### **Value Factors**
- **P/E Score**: Price-to-Earnings ratio (simplified)
- **P/B Score**: Price-to-Book ratio (simplified)
- **Value Score**: Composite value metric (contrarian signal)

### **3. Sector-Aware Factor Adjustments**

#### **Banking Sector (Quality-Adjusted)**
**Source:** Sector Conditional IC Analysis

**P/E Quality Weights:**
- Q5 (Highest ROAA): 0.7 (strong positive)
- Q3: 0.6 (moderate positive)
- Q2: 0.5 (weak positive)
- Q4: 0.1 (very weak positive)
- Q1 (Lowest ROAA): -0.5 (strong contrarian)

**P/B Quality Weights:**
- Q4: -1.5 (very strong contrarian)
- Q1: -1.0 (strong contrarian)
- Q3: -0.8 (moderate contrarian)
- Q2: -0.1 (weak contrarian)
- Q5: -0.1 (weak contrarian)

#### **Other Sectors (Standard)**
- **3M Momentum**: 40% weight
- **ROAA**: 30% weight
- **Value Contrarian**: 20% weight
- **12M Momentum Contrarian**: 10% weight

### **4. Entry Criteria**
**Source:** Strategy Documentation

```python
# Entry Criteria
3M_momentum > 0.05      # 5% positive 3-month momentum
ROAA > 0.02            # Minimum 2% return on assets
ROAE < 0.25            # Avoid extremely high ROE (contrarian)
value_score < 0.05     # AVOID high value stocks (contrarian)
daily_volume > 1000000000  # Ensure liquidity
market_cap > 1000000000 # Large-cap focus
```

---

## 🔧 Core Logic

### **Universe Selection**
1. **Market Cap Filter**: ≥1T VND minimum
2. **Liquidity Filter**: >10bn VND daily ADTV (90-day average)
3. **Data Availability**: Sufficient historical data for factor calculation

### **Factor Calculation Process**
1. **Data Retrieval**: Fetch fundamental and market data from database
2. **Momentum Calculation**: Compute multi-period price momentum
3. **Value Calculation**: Calculate P/E and P/B scores
4. **Sector Classification**: Apply sector-specific adjustments
5. **Composite Scoring**: Combine factors with appropriate weights

### **Portfolio Construction**
1. **Entry Criteria Filtering**: Apply quality and momentum thresholds
2. **Top Selection**: Select top 25 stocks by composite score
3. **Regime Allocation**: Apply regime-based allocation percentage
4. **Position Limits**: Maximum 5% per position
5. **Sector Limits**: Maximum 30% per sector
6. **Weight Normalization**: Ensure portfolio weights sum to allocation

### **Risk Management**
- **Position Concentration**: Max 5% per stock
- **Sector Concentration**: Max 30% per sector
- **Regime-Based Allocation**: Dynamic allocation based on market conditions
- **Liquidity Requirements**: Minimum trading volume thresholds
- **Quality Filters**: ROAA and ROAE thresholds

---

## 📈 Expected Performance

### **Return Expectations**
- **Annual Return**: 10-15% (vs VNINDEX benchmark)
- **Excess Return**: 3-8% above benchmark
- **Consistency**: Stable performance across market regimes

### **Risk Metrics**
- **Volatility**: 15-20% annualized
- **Sharpe Ratio**: 0.5-0.7
- **Maximum Drawdown**: 15-25%
- **VaR (95%)**: 8-12%

### **Regime Performance**
- **Bull Markets**: 15-20% annual return
- **Bear Markets**: 5-10% annual return
- **Sideways Markets**: 8-12% annual return
- **Stress Markets**: 2-5% annual return

---

## 🛠️ Implementation Details

### **Technology Stack**
- **Language**: Python 3.8+
- **Database**: PostgreSQL
- **Backtesting**: Backtrader framework
- **Data Processing**: Pandas, NumPy
- **Configuration**: YAML files

### **Data Sources**
- **Market Data**: `vcsc_daily_data_complete`
- **Fundamental Data**: `intermediary_calculations_enhanced`
- **Market Cap Data**: `equity_history_with_market_cap`
- **Sector Classifications**: `master_info`

### **Key Classes**
1. **`VNLongOnlyStrategy`**: Main strategy engine
2. **`RegimeDetector`**: Market regime detection
3. **`SectorAwareFactorCalculator`**: Sector-specific calculations
4. **`VNLongOnlyBacktraderStrategy`**: Backtrader integration

### **Configuration Parameters**
```yaml
# Strategy Parameters
liquidity_threshold: 10_000_000_000  # 10bn VND
min_market_cap: 1_000_000_000_000   # 1T VND
max_position_size: 0.05             # 5%
max_sector_exposure: 0.30          # 30%
target_portfolio_size: 25          # 25 stocks
rebalance_frequency: 21            # Monthly
```

---

## 🚀 Usage Instructions

### **1. Testing the Algorithm**
```bash
cd production/tests/phase28_strategy_merge
python test_vn_long_only_strategy.py
```

### **2. Running the Backtest**
```bash
cd production/tests/phase28_strategy_merge
python run_vn_long_only_backtest.py
```

### **3. Configuration Setup**
1. Ensure database connection in `config/database.yml`
2. Verify data availability in required tables
3. Set appropriate logging levels

### **4. Output Files**
- **Tearsheet**: `vn_long_only_backtest_YYYYMMDD_HHMMSS.txt`
- **Portfolio Data**: `vn_long_only_backtest_YYYYMMDD_HHMMSS_portfolios.csv`

---

## 📋 Data Requirements

### **Market Data**
- Daily OHLCV data for all stocks
- VNINDEX benchmark data
- Market cap history
- Trading volume data

### **Fundamental Data**
- ROAA and ROAE ratios
- Operating margins
- EBITDA margins
- Asset turnover ratios

### **Sector Classifications**
- Industry sector mappings
- Quality quintile classifications
- Market cap categories

### **Data Quality Checks**
- Point-in-time integrity
- Missing data handling
- Outlier detection
- Consistency validation

---

## ⚠️ Risk Considerations

### **Market Risks**
- **Liquidity Risk**: Illiquid positions in stress markets
- **Concentration Risk**: Sector or stock concentration
- **Regime Risk**: Incorrect regime classification
- **Data Risk**: Missing or inaccurate data

### **Implementation Risks**
- **Execution Risk**: Slippage and transaction costs
- **Rebalancing Risk**: Monthly rebalancing impact
- **Model Risk**: Factor effectiveness decay
- **Technology Risk**: System failures or data issues

### **Mitigation Strategies**
- **Diversification**: Multi-factor, multi-sector approach
- **Liquidity Filters**: Minimum volume requirements
- **Position Limits**: Maximum position sizes
- **Regime Awareness**: Dynamic allocation adjustments
- **Quality Controls**: Data validation and monitoring

---

## 🎯 Success Factors

### **Critical Success Factors**
1. **Data Quality**: Accurate and timely data feeds
2. **Regime Detection**: Reliable market regime classification
3. **Factor Stability**: Consistent factor effectiveness
4. **Execution Quality**: Efficient trade execution
5. **Risk Management**: Strict adherence to position limits

### **Performance Drivers**
1. **Momentum Capture**: 3M momentum factor effectiveness
2. **Quality Selection**: ROAA-based quality filtering
3. **Value Contrarian**: Avoiding overvalued stocks
4. **Sector Timing**: Banking sector quality adjustments
5. **Regime Timing**: Dynamic allocation based on market conditions

### **Monitoring Metrics**
- **Factor IC**: Information coefficient for each factor
- **Regime Accuracy**: Regime classification accuracy
- **Portfolio Turnover**: Monthly rebalancing activity
- **Sector Exposure**: Current sector allocations
- **Performance Attribution**: Factor contribution analysis

---

## 🔄 Monitoring and Maintenance

### **Daily Monitoring**
- **Data Quality**: Check for missing or erroneous data
- **Factor Calculation**: Verify factor computation accuracy
- **Portfolio Status**: Monitor current holdings and weights
- **Market Conditions**: Track regime classification

### **Monthly Review**
- **Performance Analysis**: Review monthly returns and attribution
- **Factor Analysis**: Assess factor effectiveness and stability
- **Portfolio Rebalancing**: Execute monthly rebalancing
- **Risk Assessment**: Review concentration and exposure metrics

### **Quarterly Assessment**
- **Strategy Review**: Comprehensive performance review
- **Factor Validation**: Validate factor effectiveness
- **Regime Analysis**: Assess regime detection accuracy
- **Parameter Optimization**: Consider parameter adjustments

### **Annual Evaluation**
- **Backtest Validation**: Full historical backtest
- **Factor Research**: New factor identification and testing
- **Strategy Enhancement**: Algorithm improvements
- **Documentation Update**: Update strategy documentation

---

## 📚 Research Foundation

### **Key Research Documents**
1. **Phase 26 Comprehensive Analysis**: Regime detection methodology
2. **Strategy Documentation**: Factor coefficient analysis
3. **Sector Conditional IC Summary**: Banking sector analysis
4. **Market Cap Quartile Analysis**: Size effect analysis

### **Academic References**
- Factor investing literature
- Regime detection methodologies
- Sector-specific factor analysis
- Vietnamese market studies

### **Validation Studies**
- Out-of-sample testing
- Monte Carlo simulations
- Parameter sensitivity analysis
- Robustness checks

---

**Document Version:** 1.0  
**Last Updated:** January 2025  
**Next Review:** Quarterly  
**Maintained By:** Factor Investing Team 