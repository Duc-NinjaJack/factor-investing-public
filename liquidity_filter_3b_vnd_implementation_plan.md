# Liquidity Filter Implementation Plan: 3 Billion VND ADTV Threshold

**Project:** Factor Investing Platform - Liquidity Filter Modification  
**Date:** January 2025  
**Objective:** Implement 60-day average daily turnover above 3 billion VND liquidity filter  
**Current State:** 10 billion VND threshold  
**Target State:** 3 billion VND threshold  

---

## 📋 Executive Summary

This plan outlines the comprehensive modification of the factor investing platform's liquidity filtering mechanism to change from a 10 billion VND to 3 billion VND average daily turnover threshold. This change will significantly expand the investable universe while maintaining appropriate liquidity standards for institutional trading.

### Key Changes Required:
- **Configuration Updates:** Modify all config files to use 3B VND threshold
- **Backtesting Scripts:** Update portfolio construction logic
- **Engine Validation:** Ensure ADTV calculation works with new threshold
- **Testing Framework:** Comprehensive validation of changes
- **Documentation:** Update all relevant documentation

---

## 🔍 Current State Analysis

### Existing Implementation:
- **Current Threshold:** 10,000,000,000 VND (10 billion VND)
- **ADTV Calculation:** 63-day rolling average from `vcsc_daily_data_complete` table
- **Filtering Location:** Applied during portfolio construction in backtesting
- **Configuration:** Centralized in `config/strategy_config.yml`

### Current Files Affected:
```
config/
├── strategy_config.yml          # Main backtesting config
├── momentum.yml                 # Momentum strategy config
└── dashboard_config.yml         # Dashboard config

production/
├── engine/
│   ├── qvm_engine_v1_baseline.py
│   └── qvm_engine_v2_enhanced.py
└── tests/
    └── phase8_risk_management/
        ├── 09_production_strategy_backtest.ipynb
        ├── 11_production_strategy_backtest_v1.2.ipynb
        └── 12_small_cap_alpha_strategy.ipynb
```

---

## 🎯 Objectives

### Primary Objectives:
1. **Expand Investable Universe:** Increase from ~148 stocks to ~300-400 stocks
2. **Maintain Liquidity Standards:** Ensure 3B VND provides adequate liquidity
3. **Preserve Performance:** Maintain or improve strategy performance
4. **Ensure Consistency:** Update all components uniformly

### Secondary Objectives:
1. **Improve Diversification:** Reduce concentration risk
2. **Enhance Alpha Opportunities:** Access to more potential alpha sources
3. **Optimize Transaction Costs:** Balance liquidity vs. trading costs

---

## 📝 Required Changes

### 1. Configuration Files

#### `config/strategy_config.yml`
```yaml
# Current:
backtesting:
  universe_constraints:
    min_liquidity: 1e9       # 1B VND daily volume

# Change to:
backtesting:
  universe_constraints:
    min_liquidity: 3e9       # 3B VND daily volume
```

#### `config/momentum.yml`
```yaml
# Current:
data:
  min_volume: 100000  # Minimum daily volume

# Change to:
data:
  min_volume: 300000  # Adjusted for 3B VND threshold
```

### 2. Backtesting Scripts

#### `production/tests/phase8_risk_management/09_production_strategy_backtest.ipynb`
```python
# Current:
"liquidity_threshold_vnd": 10_000_000_000, # 10B VND

# Change to:
"liquidity_threshold_vnd": 3_000_000_000,  # 3B VND
```

#### `production/tests/phase8_risk_management/11_production_strategy_backtest_v1.2.ipynb`
```python
# Same change as above
```

#### `production/tests/phase8_risk_management/12_small_cap_alpha_strategy.ipynb`
```python
# Current:
"liquidity_thresholds_vnd_to_test": [500_000_000, 1_000_000_000, 2_500_000_000, 5_000_000_000]

# Change to:
"liquidity_thresholds_vnd_to_test": [1_000_000_000, 2_000_000_000, 3_000_000_000, 5_000_000_000]
```

### 3. Engine Files
- **No direct changes needed** - engines use configuration parameters
- **Verify ADTV calculation logic** - ensure 63-day rolling average is appropriate for 60-day requirement

---

## 🚀 Implementation Plan

### Phase 1: Configuration Updates (Priority: High)
**Duration:** 1-2 hours  
**Status:** 🔄 In Progress

#### Tasks:
- [ ] Update `config/strategy_config.yml`
- [ ] Update `config/momentum.yml`
- [ ] Update `config/dashboard_config.yml` if needed
- [ ] Verify all configuration files are consistent

#### Validation:
- [ ] Run configuration validation script
- [ ] Check for any hardcoded values in config files
- [ ] Ensure backward compatibility

### Phase 2: Backtesting Script Updates (Priority: High)
**Duration:** 2-3 hours  
**Status:** ⏳ Pending

#### Tasks:
- [ ] Update `09_production_strategy_backtest.ipynb`
- [ ] Update `11_production_strategy_backtest_v1.2.ipynb`
- [ ] Update `12_small_cap_alpha_strategy.ipynb`
- [ ] Update any other backtesting scripts found

#### Validation:
- [ ] Run syntax checks on all modified scripts
- [ ] Verify configuration parameters are correctly referenced
- [ ] Test script execution without running full backtests

### Phase 3: Engine Validation (Priority: Medium)
**Duration:** 1-2 hours  
**Status:** ⏳ Pending

#### Tasks:
- [ ] Review ADTV calculation logic in engines
- [ ] Verify 63-day rolling average is appropriate
- [ ] Test engine initialization with new config
- [ ] Validate factor calculation with new universe

#### Validation:
- [ ] Run engine unit tests
- [ ] Verify ADTV calculation accuracy
- [ ] Test with sample universe

### Phase 4: Testing and Validation (Priority: High)
**Duration:** 4-6 hours  
**Status:** ⏳ Pending

#### Tasks:
- [ ] Run existing validation scripts with new threshold
- [ ] Compare universe sizes (before vs after)
- [ ] Analyze sector composition changes
- [ ] Test portfolio construction logic
- [ ] Validate performance metrics

#### Validation:
- [ ] Universe size increased appropriately
- [ ] No critical stocks excluded
- [ ] Performance metrics maintained
- [ ] Sector diversification improved

### Phase 5: Documentation Updates (Priority: Low)
**Duration:** 1-2 hours  
**Status:** ⏳ Pending

#### Tasks:
- [ ] Update README files
- [ ] Update configuration documentation
- [ ] Update technical specifications
- [ ] Create change log entry

---

## 🧪 Testing Strategy

### 1. Unit Testing
```bash
# Test configuration loading
python -c "import yaml; config = yaml.safe_load(open('config/strategy_config.yml')); print(config['backtesting']['universe_constraints']['min_liquidity'])"

# Test engine initialization
python -c "from production.engine.qvm_engine_v2_enhanced import QVMEngineV2Enhanced; engine = QVMEngineV2Enhanced()"
```

### 2. Integration Testing
```bash
# Run existing validation scripts
cd production/tests
jupyter nbconvert --to notebook --execute 04_factor_calculation_deep_dive.ipynb
```

### 3. Backtesting Validation
```bash
# Run comparative backtests
cd production/tests/phase8_risk_management
jupyter nbconvert --to notebook --execute 09_production_strategy_backtest.ipynb
```

### 4. Performance Comparison
- Compare universe sizes
- Analyze sector composition changes
- Monitor performance metrics
- Check transaction costs impact

---

## ⚠️ Risk Assessment

### High Risk:
1. **Performance Degradation:** New universe may include lower-quality stocks
   - **Mitigation:** Monitor performance metrics closely
   - **Rollback Plan:** Revert to 10B VND if performance drops >5%

2. **Data Quality Issues:** Smaller stocks may have unreliable data
   - **Mitigation:** Implement additional data quality checks
   - **Monitoring:** Track data completeness metrics

### Medium Risk:
1. **Computational Load:** Larger universe increases processing time
   - **Mitigation:** Optimize algorithms if needed
   - **Monitoring:** Track execution times

2. **Transaction Costs:** More stocks may increase trading costs
   - **Mitigation:** Adjust portfolio construction logic
   - **Monitoring:** Track cost metrics

### Low Risk:
1. **Configuration Inconsistency:** Different config files may have different values
   - **Mitigation:** Comprehensive configuration validation
   - **Monitoring:** Automated config checks

---

## 📊 Success Criteria

### Quantitative Metrics:
- [ ] Universe size increases by 100-200% (from ~148 to ~300-400 stocks)
- [ ] Performance metrics maintained within 5% of current levels
- [ ] Sector diversification improves (no single sector >40%)
- [ ] Transaction costs remain manageable (<50 bps)

### Qualitative Metrics:
- [ ] All configuration files updated consistently
- [ ] No critical stocks excluded due to new threshold
- [ ] Backtesting scripts run without errors
- [ ] Documentation updated and accurate

---

## 📅 Timeline

| Phase | Duration | Start Date | End Date | Status |
|-------|----------|------------|----------|--------|
| Phase 1: Configuration | 1-2 hours | Day 1 | Day 1 | 🔄 In Progress |
| Phase 2: Backtesting Scripts | 2-3 hours | Day 1 | Day 1 | ⏳ Pending |
| Phase 3: Engine Validation | 1-2 hours | Day 1 | Day 1 | ⏳ Pending |
| Phase 4: Testing | 4-6 hours | Day 2 | Day 2 | ⏳ Pending |
| Phase 5: Documentation | 1-2 hours | Day 2 | Day 2 | ⏳ Pending |

**Total Estimated Time:** 9-15 hours  
**Critical Path:** Configuration → Backtesting → Testing  

---

## 🔄 Current Tasks

### Active Tasks:
1. **Configuration Analysis** - Reviewing all config files for liquidity references
2. **Impact Assessment** - Analyzing potential universe size changes
3. **Risk Evaluation** - Assessing performance impact of threshold change

### Next Tasks:
1. **Update Configuration Files** - Modify all config files with new threshold
2. **Update Backtesting Scripts** - Modify portfolio construction logic
3. **Run Validation Tests** - Test changes with existing validation framework

---

## 💡 Key Insights

### Technical Insights:
1. **ADTV Calculation:** Current 63-day rolling average is more conservative than 60-day requirement
2. **Configuration Centralization:** Most settings are properly centralized in config files
3. **Engine Flexibility:** Engines use configuration parameters, reducing modification needs

### Business Insights:
1. **Universe Expansion:** 3B VND threshold will significantly increase investable universe
2. **Diversification Benefits:** More stocks available for portfolio construction
3. **Performance Trade-offs:** Balance between universe size and data quality

### Implementation Insights:
1. **Low Risk Changes:** Most modifications are configuration updates
2. **Testing Critical:** Comprehensive testing needed due to universe size changes
3. **Monitoring Essential:** Performance monitoring required post-implementation

---

## 📞 Communication Plan

### Progress Tracking:
- **Daily Updates:** Status of current tasks
- **Weekly Reviews:** Overall progress and any issues
- **Milestone Reports:** Completion of major phases

### Issue Escalation:
- **Technical Issues:** Immediate escalation for blocking problems
- **Performance Issues:** Escalation if performance degrades >5%
- **Configuration Issues:** Escalation for inconsistent configurations

### Success Reporting:
- **Universe Size Changes:** Report on investable universe expansion
- **Performance Metrics:** Report on strategy performance maintenance
- **Implementation Completion:** Final status report

---

## 📚 References

### Documentation:
- [System Architecture](docs/2_technical_implementation/02_system_architecture.md)
- [QVM Engine Specification](docs/2_technical_implementation/02a_qvm_engine_v2_enhanced_specification.md)
- [Backtesting Framework](docs/4_backtesting_and_research/04_qvm_backtesting_framework.md)

### Related Files:
- `production/tests/audit_plan.md` - Current audit methodology
- `config/strategy_config.yml` - Main configuration file
- `production/engine/qvm_engine_v2_enhanced.py` - Main engine implementation

---

**Plan Version:** 1.0  
**Last Updated:** January 2025  
**Next Review:** Upon completion of Phase 1 