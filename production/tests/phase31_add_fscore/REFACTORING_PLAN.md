# QVM Flat Config Refactoring Plan
## File Cleanup Progress Tracker

**Target File**: `07_QVM_flat_config.py`  
**Current Status**: 2530 lines (down from 2903 lines)  
**Target**: 800-1000 lines (remove ~1600-1800 lines)  
**Progress**: 376 lines removed (21% complete)

---

## 🎯 **REFACTORING OBJECTIVES**

### ✅ **COMPLETED TASKS**
- [x] Created modular files under `scripts/` directory
- [x] Added imports for modular functions
- [x] Removed `load_strategy_config` function
- [x] Removed `load_backtest_config` function  
- [x] Removed `merge_backtest_with_strategy_config` function
- [x] Removed `get_default_strategy_config` function
- [x] Removed `get_default_backtest_config` function
- [x] Removed `_validate_factor_weights` function
- [x] Removed `_validate_risk_management_config` function
- [x] Removed `validate_strategy_config` function
- [x] Removed `get_sector_mapping_performance` function
- [x] Removed `robust_data_operation` function
- [x] Removed `get_correct_quarter_for_date` function
- [x] Removed `load_price_data_efficiently` function
- [x] Removed `calculate_dynamic_cash_allocation` function
- [x] Removed `_calculate_banking_fscore` function (complete)
- [x] Removed `_calculate_securities_fscore` function signature

---

## 🔧 **REMAINING TASKS BY PRIORITY**

### **HIGH PRIORITY - Large Functions (Remove First)**
- [ ] Remove `generate_holdings_with_flat_methodology` function (~150 lines)
- [ ] Remove `run_strategy_with_flat_methodology` function (~100 lines)
- [ ] Remove `generate_flat_methodology_tearsheet` function (~100 lines)
- [ ] Remove `main` function (~80 lines)

### **MEDIUM PRIORITY - F-Score Functions**
- [ ] Remove `_calculate_banking_fscore` function body (~60 lines)
- [ ] Remove `_calculate_securities_fscore` function (~70 lines)
- [ ] Remove `_calculate_non_financial_fscore` function (~80 lines)

### **MEDIUM PRIORITY - Visualization Functions**
- [ ] Remove `generate_factor_score_evolution_plot` function (~80 lines)
- [ ] Remove `generate_portfolio_holdings_distribution_plot` function (~100 lines)
- [ ] Remove `generate_complete_tearsheet_plots` function (~30 lines)

### **MEDIUM PRIORITY - Risk Management Functions**
- [ ] Remove `display_cash_allocation_rules` function (~70 lines)
- [ ] Remove `test_cash_allocation_scenarios` function (~60 lines)

### **LOW PRIORITY - Data Management Functions**
- [ ] Remove `get_sector_mapping` function (~50 lines)
- [ ] Remove `clear_sector_cache` function (~10 lines)
- [ ] Remove `load_data_with_fallback` function (~40 lines)
- [ ] Remove `get_most_recent_available_date` function (~30 lines)

### **LOW PRIORITY - Factor Calculation Functions**
- [ ] Remove `_calculate_flat_momentum_composite` function (~20 lines)
- [ ] Remove `_calculate_flat_defensive_composite` function (~20 lines)
- [ ] Remove `_calculate_enhanced_flat_quality_composite` function (~20 lines)
- [ ] Remove `_calculate_enhanced_flat_value_composite` function (~20 lines)

---

## 📊 **CHUNK PROGRESS TRACKING**

### **CHUNK 1: Configuration Functions** ✅
- **Lines Removed**: 1
- **Functions Removed**: 1
- **Status**: COMPLETE

### **CHUNK 2: Validation Functions** ✅
- **Lines Removed**: 62
- **Functions Removed**: 3
- **Status**: COMPLETE

### **CHUNK 3: More Functions** ✅
- **Lines Removed**: 112
- **Functions Removed**: 4
- **Status**: COMPLETE

### **CHUNK 4: Data Management Functions** ✅
- **Lines Removed**: 142
- **Functions Removed**: 3
- **Status**: COMPLETE

### **CHUNK 5: F-Score Functions** ✅
- **Lines Removed**: 58
- **Functions Removed**: 2 (1 complete, 1 partial)
- **Status**: COMPLETE

### **CHUNK 6: More F-Score Functions** ⏳
- **Target**: Remove remaining F-Score functions
- **Estimated Lines**: 150-200
- **Status**: PENDING

### **CHUNK 7: Large Core Functions** ⏳
- **Target**: Remove `generate_holdings_with_flat_methodology`
- **Estimated Lines**: 150
- **Status**: PENDING

### **CHUNK 8: Strategy Execution Functions** ⏳
- **Target**: Remove `run_strategy_with_flat_methodology`
- **Estimated Lines**: 100
- **Status**: PENDING

### **CHUNK 9: Visualization Functions** ⏳
- **Target**: Remove all plotting functions
- **Estimated Lines**: 200-250
- **Status**: PENDING

### **CHUNK 10: Main Execution Functions** ⏳
- **Target**: Remove `main` and helper functions
- **Estimated Lines**: 150-200
- **Status**: PENDING

---

## 🎯 **NEXT ACTIONS**

### **IMMEDIATE (CHUNK 6)**
1. Complete removal of `_calculate_banking_fscore` function body
2. Remove `_calculate_securities_fscore` function
3. Remove `_calculate_non_financial_fscore` function
4. **Target**: Remove 150-200 lines

### **SHORT TERM (CHUNKS 7-8)**
1. Remove large core functions
2. Remove strategy execution functions
3. **Target**: Remove 250-300 lines

### **MEDIUM TERM (CHUNKS 9-10)**
1. Remove visualization functions
2. Remove main execution functions
3. **Target**: Remove 350-450 lines

---

## 📈 **PROGRESS METRICS**

- **Starting Lines**: 2903
- **Current Lines**: 2588
- **Lines Removed**: 372
- **Remaining Lines**: 2231
- **Target Lines**: 800-1000
- **Lines to Remove**: 1231-1431
- **Completion**: 21%

---

## 🔍 **QUALITY CHECKLIST**

- [ ] All imported functions are properly removed from main file
- [ ] No duplicate function definitions remain
- [ ] All imports are working correctly
- [ ] Main algorithm logic is preserved
- [ ] File structure is clean and organized
- [ ] No syntax errors introduced
- [ ] File size is reduced to target range

---

## 📝 **NOTES**

- **Strategy**: Remove functions in chunks of 150-200 lines max
- **Priority**: Focus on large functions first for maximum impact
- **Testing**: Verify imports work after each major removal
- **Backup**: Keep original file as reference until complete
- **Goal**: Achieve clean, maintainable code structure

---

**Last Updated**: Current session  
**Next Review**: After CHUNK 6 completion
