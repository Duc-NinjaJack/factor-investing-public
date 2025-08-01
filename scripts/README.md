# Scripts Directory - Vietnam Factor Investing Platform
**Last Updated**: July 3, 2025  
**Status**: Organized structure with 714/728 tickers production-ready + Phase 2 Infrastructure Complete

---

## 📁 **Directory Structure**

### **🎯 Root Level (Core Production)**
- `factor_menu.py` - Main interactive menu interface
- `run_workflow.py` - CLI workflow orchestration  
- `populate_intermediary_calculations.py` - **Phase 2 Priority** - Intermediary calculations

### **📊 sector_extracts/ (Analysis Generation)**
- `banking_enhanced_extract.py` - Banking sector analysis (21 tickers)
- `securities_enhanced_extract.py` - Securities sector analysis (26 tickers)
- `insurance_enhanced_extract_final_corrected.py` - Insurance analysis (11 tickers, mathematical reconciliation complete)
- `nonfin_enhanced_extract.py` - Non-financial analysis (667 tickers)

**Usage**:
```bash
python scripts/sector_extracts/banking_enhanced_extract.py VCB
python scripts/sector_extracts/securities_enhanced_extract.py SSI
```

### **🏗️ sector_views/ (Database View Creation)**
- `create_banking_enhanced_view.py` - Creates `v_complete_banking_fundamentals`
- `create_securities_enhanced_view.py` - Creates `v_complete_securities_fundamentals`
- `create_insurance_view_final_corrected.py` - Creates `v_complete_insurance_fundamentals`
- `create_nonfin_enhanced_view.py` - Maintains `v_comprehensive_fundamental_items`

**Usage**:
```bash
python scripts/sector_views/create_banking_enhanced_view.py
python scripts/sector_views/create_securities_enhanced_view.py
```

### **🔍 investigations/ (Research & Analysis)**
- `2025-06-30_dividend_research/` - Cross-sector dividend investigations
- `2025-06-30_enhanced_infrastructure/` - Insurance reconciliation work
- Various sector-specific investigation scripts

### **🛠️ db_checks/ (Database Utilities)**
- Database structure validation scripts
- View definition checkers
- Data integrity utilities

### **📦 archive/ (Historical Scripts)**
- `insurance_iterations/` - Insurance development history
- Historical debug and fix scripts
- Legacy analysis tools

---

## 🚀 **Quick Start Guide**

### **Daily Operations**
```bash
# Main production interface
python scripts/factor_menu.py

# CLI workflow
python scripts/run_workflow.py full-daily
```

### **Phase 2 Priority**
```bash
# Populate intermediary calculations (current focus)
python scripts/populate_intermediary_calculations.py
```

### **Generate Sector Analysis**
```bash
# Banking analysis
python scripts/sector_extracts/banking_enhanced_extract.py VCB

# Securities analysis  
python scripts/sector_extracts/securities_enhanced_extract.py SSI

# Non-financial analysis
python scripts/sector_extracts/nonfin_enhanced_extract.py FPT
```

### **Database View Management**
```bash
# Create/update banking view
python scripts/sector_views/create_banking_enhanced_view.py

# Create/update securities view
python scripts/sector_views/create_securities_enhanced_view.py
```

---

## 📋 **Current Status (July 1, 2025)**

**Production-Ready**: 714/728 tickers (98.1%)
- ✅ **Banking**: 21 tickers with enhanced views
- ✅ **Securities**: 26 tickers with enhanced views  
- ✅ **Non-Financial**: 667 tickers with enhanced logic
- ⏳ **Insurance**: 11 tickers (infrastructure ready, factors pending Phase 2)
- ⏳ **Other Financial**: 3 tickers (pending Phase 2)

**Phase 2 Focus**: Intermediary calculations infrastructure for universal benefit

---

## 🔧 **Development Notes**

### **Path Structure**
All scripts use project-root relative paths:
```python
project_root = Path(__file__).parent.parent.parent  # For subdirectory scripts
config_path = project_root / 'config' / 'database.yml'
```

### **Output Locations**
- Analysis reports: `docs/4_validation_and_quality/{sector}/`
- Database views: Created in MySQL `alphabeta` database
- Logs: Console output with progress indicators

### **Dependencies**
- Python 3.8+
- pandas, numpy, pymysql, yaml
- Project config files in `/config/`
- MySQL database connection

---

**📝 Note**: All scripts are production-tested and maintain backward compatibility.