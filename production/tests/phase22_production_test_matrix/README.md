# Phase 22: Production Test Matrix - "Aureus Sigma Vietnam Value Concentrated"
**Created:** 2025-07-29  
**Status:** Implementation Ready  
**Objective:** Convert validated factor research into audit-ready, concentrated portfolio strategy through systematic testing

---

## 🎯 **EXECUTIVE SUMMARY**

This phase implements the **final production validation** of the Vietnam Factor Investing Platform through a comprehensive 12-cell test matrix. With factor scores pre-calculated and validated via `QVMEngineV2Enhanced`, we now test concentrated portfolio construction under realistic execution constraints to engineer "Aureus Sigma Vietnam Value Concentrated v1.0".

### **Critical Issue Addressed**
- **Concentration Gap**: Prior research used theoretical broad diversification; need practical 15-25 stock portfolios
- **Execution Reality**: Must incorporate Vietnam-specific transaction costs, sector caps, and risk overlays
- **Institutional Compliance**: Achieve Max Drawdown ≤ -35% while maintaining Sharpe ≥ 1.5 after all costs

---

## 📊 **TEST MATRIX SPECIFICATION**

### **Primary Test Dimensions**
```python
test_matrix = {
    'stock_counts': [15, 20, 25],           # Concentration vs diversification
    'rebalancing_frequency': ['M', 'Q'],     # Alpha decay vs transaction drag  
    'strategy_logic': ['Pure_Value', 'QVR_60_20_20'],  # Factor efficacy comparison
    'time_period': '2016-2025',             # Two full bear cycles
    'universe': 'ASC-VN-Liquid-150'         # Top ~150-200 liquid stocks
}
# Total test cells: 3 × 2 × 2 = 12 backtests
```

### **Fixed Parameters (All Tests)**
- **Universe Construction**: Top 200 stocks, 10B+ VND ADTV, quarterly refresh
- **Portfolio Weighting**: Equal-weighted within selected stocks
- **Sector Constraint**: Maximum 35% per sector (prevents concentration risk)
- **Risk Overlay**: Hybrid (Regime + Volatility) with min(regime_exposure, vol_target)
- **Transaction Costs**: Vietnam-specific model (tax 0.15%, commission 0.2%, ADTV-aware impact)

---

## 🏗️ **ARCHITECTURE: SINGLE NOTEBOOK APPROACH**

**Following Your Proven Pattern (Phases 14-18):**

### **Simple Structure (Consistent with Your Methodology)**
```
phase22_production_test_matrix/
├── README.md                          # This specification document
└── 22_production_test_matrix.ipynb    # SINGLE comprehensive notebook
```

### **Notebook Structure (Vector-Based Backtesting)**
Following the exact pattern from your Phase 14-18 notebooks:

```python
# ============================================================================
# SECTION 1: CONFIGURATION & SETUP
# ============================================================================
CONFIG = {
    "backtest_start": "2016-01-01",
    "backtest_end": "2025-07-28", 
    "rebalance_freq": "Q",
    "transaction_cost_bps": 30,
    "test_matrix": {
        'stock_counts': [15, 20, 25],
        'rebal_frequencies': ['M', 'Q'], 
        'strategies': ['Pure_Value', 'QVR_60_20_20']
    }
}

# Standard imports and visualization setup (like Phase 15/16)
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
sys.path.append('../../../production')
from universe.constructors import get_liquid_universe_dataframe

# ============================================================================
# SECTION 2: RAW DATA LOADING (Following Phase 15 Pattern)
# ============================================================================
def create_db_connection():
    # Identical to your existing notebooks
    config_path = Path.cwd().parent.parent.parent / 'config' / 'database.yml'
    # ... exact same pattern

# Load factor scores from database (pre-calculated)
factor_query = text("""
    SELECT date, ticker, Quality_Composite, Value_Composite, Momentum_Composite
    FROM factor_scores_qvm
    WHERE date BETWEEN :start_date AND :end_date 
    AND strategy_version = 'qvm_v2.0_enhanced'
""")

# Load price data (same as Phase 15)
price_query = text("""
    SELECT date, ticker, close FROM equity_history
    WHERE date BETWEEN :start_date AND :end_date
""")

# ============================================================================
# SECTION 3: TEST MATRIX EXECUTION (Vector-Based)
# ============================================================================
def run_concentrated_backtest(stock_count, rebal_freq, strategy_logic, 
                             factor_data, daily_returns_matrix):
    """
    Single backtest following your vector-based pattern from Phase 16
    """
    # Dynamic universe construction (quarterly)
    # Portfolio construction with concentration
    # Vector-based return calculation
    # Transaction cost application
    # Risk overlay implementation
    return net_returns

# Loop through 12-cell matrix
results_summary = []
for stock_count in [15, 20, 25]:
    for rebal_freq in ['M', 'Q']:
        for strategy in ['Pure_Value', 'QVR_60_20_20']:
            # Execute single backtest
            results = run_concentrated_backtest(...)
            results_summary.append(results)

# ============================================================================
# SECTION 4: RESULTS ANALYSIS & DECISION
# ============================================================================
# Performance metrics calculation (in-house functions like Phase 15)
# Tearsheet generation for top configurations
# Final decision logic with institutional gates
```

---

## 🧮 **ENHANCED TRANSACTION COST MODEL**

### **Vietnam Market-Specific Implementation**
```python
def calculate_vietnam_transaction_costs(current_weights, target_weights, adtv_data):
    """
    Vietnam-specific cost model with institutional edge case handling
    
    Base Cost Components:
    - Tax: 0.15% on sales only (mandatory VN requirement)
    - Commission: 0.2% round trip (realistic broker rates)
    - Market Impact: sqrt(participation_rate) scaling with ADTV
    
    Edge Case Refinements (per detailed review feedback):
    - Micro sells (<0.05% participation): 0 bps impact
    - Large caps (>100B VND ADTV): 5 bps impact floor  
    - VX threshold: Hard cap at 15% daily ADV
    - Look-ahead prevention: Use lagged ADTV (T-1)
    """
    
    # Per-stock impact calculation with non-linear scaling
    def calculate_market_impact(weight_change, adtv_lagged):
        trade_value = weight_change * portfolio_value
        participation_rate = trade_value / (adtv_lagged * 1e9 * 20)  # 20-day average
        
        # Edge case handling
        if participation_rate < 0.0005:  # <0.05% participation
            return 0.0
        
        # Non-linear impact with large cap floor
        impact_bps = 10 * np.sqrt(participation_rate * 100)
        if adtv_lagged > 100:  # >100B VND ADTV
            impact_bps = max(impact_bps, 5.0)  # 5bps floor
            
        return min(impact_bps / 10000, 0.005)  # Cap at 50bps
```

---

## 🛡️ **HYBRID RISK OVERLAY SYSTEM**

### **Two-Layer Risk Management**
```python
def apply_hybrid_risk_overlay(returns, regime_signal, vol_target=0.15):
    """
    Institutional-grade risk overlay with refined calibration
    
    Layer 1 - Regime Filter: 50% exposure during Bear/Stress periods
    Layer 2 - Volatility Target: Dynamic scaling to 15% annualized vol
    
    Final Logic: min(regime_exposure, vol_exposure)
    Refinements: Proper clipping bounds, anti-whipsaw smoothing
    """
    # Regime layer: Decisive exposure reduction
    regime_exposure = regime_signal.apply(lambda x: 1.0 if x else 0.5)
    
    # Volatility layer: 20-day EWMA (faster response than 60-day)
    realized_vol = returns.ewm(span=20).std() * np.sqrt(252)
    vol_exposure = (vol_target / realized_vol).shift(1)  # Prevent look-ahead
    
    # Proper bounds and precision (per review feedback)
    vol_exposure = vol_exposure.clip(lower=0.2, upper=1.0)  # Fixed clipping
    vol_exposure.fillna(1.0, inplace=True)
    
    # Hybrid combination: Most conservative wins
    final_exposure = pd.DataFrame({
        'regime': regime_exposure,
        'vol': vol_exposure
    }).min(axis=1)
    
    # Anti-whipsaw smoothing with broker precision
    final_exposure = final_exposure.ewm(span=3).mean()
    final_exposure = np.round(final_exposure, 2)  # 2 decimal places for broker orders
    
    return returns * final_exposure, final_exposure
```

---

## 📋 **REFINED SUCCESS CRITERIA & DECISION GATES**

### **All Gates Must Pass Same Configuration**
| Metric | Threshold | Test Source | Rationale |
|--------|-----------|-------------|-----------|
| **Sharpe Ratio** | ≥ 1.5 | `results_summary.csv` | Risk-adjusted excellence after all costs |
| **Max Drawdown** | ≥ -35% | `results_summary.csv` | Institutional risk compliance |
| **Information Ratio** | ≥ 0.8 | `results_summary.csv` | Consistent alpha vs VN-Index |
| **Annual Turnover** | ≤ 300% | `results_summary.csv` | Operational cost efficiency |
| **Transaction Drag** | ≤ 3% CAGR | `cost_breakdown.pkl` | Implementation cost control |

### **Enhanced Decision Logic**
```python
def select_optimal_configuration(results_df):
    """
    Institutional gate filtering with comprehensive criteria
    """
    # Step 1: Apply all institutional gates simultaneously
    candidates = results_df[
        (results_df['sharpe_ratio_after_costs'] >= 1.5) &
        (results_df['max_drawdown'] >= -35.0) &
        (results_df['information_ratio'] >= 0.8) &           # Added per review
        (results_df['annual_turnover'] <= 300.0) &
        (results_df['transaction_drag'] <= 3.0)
    ]
    
    if candidates.empty:
        return "FAILED: No configuration meets all institutional requirements"
    
    # Step 2: Rank survivors by risk-adjusted return
    optimal = candidates.nlargest(1, 'sharpe_ratio_after_costs')
    
    return {
        'config': optimal.iloc[0],
        'status': 'SUCCESS: Institutional-grade configuration identified',
        'next_phase': 'Phase 21 Pilot - 50bn VND paper trading'
    }
```

---

## 🗓️ **REFINED 4-WEEK IMPLEMENTATION SPRINT**

### **Week 1: Validated Infrastructure (Days 1-7)**
**Core Deliverables:**
- [x] **Data Pipeline**: Validate factor scores availability in `factor_scores_qvm` for 2016-2025
- [x] **Universe Integration**: Test `production/universe/constructors.py` across quarterly periods
- [ ] **Portfolio Constructor**: Build concentrated portfolio logic with sector caps
- [ ] **Transaction Cost Model**: Vietnam-specific model with edge case handling
- [ ] **Risk Overlay System**: Hybrid regime + volatility implementation
- [ ] **Unit Test Suite**: 90%+ coverage on new components
- [ ] **Single Cell Validation**: Dry-run 25-Q-Value baseline

**Success Criteria**: All components pass unit tests, single dry-run completes in <5 minutes

### **Week 2: Matrix Execution (Days 8-14)**
**Parallel Processing Strategy:**
- [ ] **Monday**: Launch 12 parallel backtests via AWS Batch/GNU parallel
- [ ] **Wednesday**: Mid-week sanity checks and error recovery
- [ ] **Friday**: Results consolidation and steering committee review

**Key Outputs**: 
- `test_matrix_summary.csv` with all 12 configurations
- Performance tearsheets for top 3 configurations
- Cost breakdown analysis and turnover statistics

### **Week 3: Analysis & Selection (Days 15-21)**
**Decision Framework:**
- [ ] **Gate Analysis**: Apply all 5 institutional criteria
- [ ] **Sensitivity Testing**: ±20% parameter stress testing on optimal config
- [ ] **Attribution Analysis**: Factor contribution breakdown and overlay effectiveness
- [ ] **Final Selection**: Lock optimal configuration or trigger iteration

**Decision Outcomes**:
- **Success**: Optimal config meets all gates → Proceed to Week 4
- **Iteration**: Adjust overlay parameters (12% vol target, 30% regime exposure)

### **Week 4: Production Hardening (Days 22-28)**
**Production Readiness:**
- [ ] **Parameter Freeze**: Lock optimal settings in `strategy_config_v1.yml`
- [ ] **Code Coverage**: Achieve 100% test coverage on production modules  
- [ ] **Docker Package**: Container deployment with monitoring dashboards
- [ ] **Documentation**: Investment memo and regulatory compliance materials
- [ ] **Pilot Preparation**: Phase 21 setup for 50bn VND paper trading

---

## 🎯 **RISK ASSESSMENT & CONTINGENCY PLANNING**

### **Scenario Planning**
**Base Case (75% Probability)**: 25-Q-Pure Value configuration meets all gates
- Expected: Sharpe 1.6-1.8, Max DD -30% to -33%, Turnover 200-250%

**Upside Case (15% Probability)**: Monthly rebalancing or QVR composite adds value
- Potential: Enhanced alpha with manageable cost increase

**Downside Case (10% Probability)**: No configuration meets -35% drawdown gate
- **Contingency**: Iterate overlay parameters (12% vol target, 30% regime exposure)
- **Backup**: Test 30-35 stock variants if concentration proves problematic

### **Data Quality Safeguards**
- **Universe Monitoring**: Alert if constructed universe <50 or >220 stocks
- **Factor Score Validation**: Sanity check against historical distributions
- **Cost Model Verification**: Compare predicted vs actual transaction costs

---

## 🏆 **SUCCESS DEFINITION & NEXT STEPS**

**Phase 22 Success Criteria:**
1. **≥1 configuration** passes all 5 institutional gates simultaneously
2. **Production codebase** achieves audit-ready quality standards
3. **Investment memo** documents strategy for regulatory approval
4. **Phase 21 Pilot** prepared for live validation

**Success Triggers Phase 21:**
- Paper trading with 50bn VND notional
- Real-time risk monitoring implementation  
- Final cost model calibration against actual execution

**Failure Triggers Iteration:**
- Risk overlay parameter adjustment
- Position sizing constraint relaxation
- Rebalancing frequency optimization

---

**This comprehensive framework leverages our validated factor research foundation to engineer an institutional-grade, concentrated equity strategy through systematic testing and risk management.** 🚀

---

*Last updated: 2025-07-29*  
*Architecture: Database-driven with validated factor scores*  
*Focus: Concentrated portfolio construction with realistic execution constraints*