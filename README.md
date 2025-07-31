# Vietnam Factor Investing Platform 🇻🇳

[![Production Status](https://img.shields.io/badge/Status-Phase%2016%20Active-green.svg)]() [![Market Intelligence](https://img.shields.io/badge/Market%20Intel-Terminal%20Ready-blue.svg)]() [![Engine](https://img.shields.io/badge/QVM%20Engine-v2%20Enhanced-brightgreen.svg)]()

**Institutional-grade quantitative factor investing framework for Vietnamese equity markets**

A comprehensive production platform implementing Quality, Value, and Momentum (QVM) strategies with terminal-based market intelligence, designed for institutional portfolio management in frontier markets.

---

## 🚀 **Current Status: Phase 16 - Weighted Composite Model Engineering**

**Latest Achievement:** Successfully transitioned from illusion to reality - discovered genuine alpha in liquid universe with 0.91 Sharpe ratio (standalone Value factor). Currently diagnosing composite model underperformance vs standalone factors.

**Key Metrics:**
- **Value Factor (Standalone):** 0.91 Sharpe ratio, 12.4% annual return ✅
- **Liquid Universe:** Top 200 stocks, 10B+ VND ADTV, ~62% alpha preservation
- **Market Intelligence:** Terminal-based Daily Alpha Pulse operational

---

## 📊 **Market Intelligence Platform**

### **Daily Alpha Pulse (Terminal)**
Real-time quantitative market intelligence displayed directly in terminal:

```bash
python scripts/production_menu.py
# Select Option 0.1 - Daily Alpha Pulse (Terminal)
```

**Features:**
- **Market Overview:** VN-Index performance, breadth analysis, foreign flows
- **Factor Performance:** Quality, Value, Momentum long-short returns with strength indicators
- **Trading Intelligence:** Top volume/turnover stocks with change indicators
- **Real-time Data:** Market breadth, volume ratios, participation metrics
- **Clean Interface:** ASCII-based dashboard with color-coded performance

**Sample Output:**
```
┌────────────────────────────────────────────┐
│          DAILY ALPHA PULSE                 │
│         July 31, 2025 18:30 ICT            │
├────────────────────────────────────────────┤
│ MARKET OVERVIEW            │ TRADING DATA  │
│ • VN-Index: 1,234.5 +1.2% │ • Vol: 450M   │
│ • Breadth: 156/144 (1.1)  │ • T/O: 12.3B  │
│ • Active: 298 stocks       │ • FF: +2.1B   │
├────────────────────────────────────────────┤
│ FACTOR PERFORMANCE                         │
│ • Quality : +0.8% ████████░░ Strong       │
│ • Value   : +1.2% ██████████ Strong       │
│ • Momentum: -0.3% ███░░░░░░░ Weak         │
└────────────────────────────────────────────┘
```

---

## 🎛️ **Production Menu System**

### **Reorganized Interface (Side-by-Side Layout)**
Market Intelligence prominently featured as **Option 0** with efficient workflow navigation:

```
═══ 0. MARKET INTELLIGENCE ═══
0.1 - 📊 Daily Alpha Pulse (Terminal)
0.2 - 📈 Advanced Market Intelligence Dashboard (Future)

CORE WORKFLOW                     │ EXECUTION & MONITORING
──────────────────────────────────┼──────────────────────────────────
1. DAILY DATA UPDATES (CRITICAL)  │ 4. BACKTESTING & EXECUTION
1.1 - Market Data (OHLCV, ETFs)  │ 4.1 - Run Canonical Backtest
1.2 - Financial Info (Shares)     │ 4.2 - Generate Target Portfolio
...                               │ ...

FACTOR GENERATION (PRODUCTION ENGINE)
7.0 - 📚 Factor Generation Guide & Best Practices (CRITICAL)
7.1 - Generate QVM Factors (Date Range)
7.2 - Generate QVM Factors (Single Date)
7.3 - Incremental Update (Auto Gap Detection)
```

**Key Improvements:**
- **Market Intelligence First:** Option 0 for immediate access
- **Logical Grouping:** Data → Processing → Generation → Execution
- **Dynamic Status:** Quarterly urgency tracking with countdown
- **Visual Organization:** Side-by-side layout maximizes screen space

---

## 🔧 **Project Architecture**

```
factor-investing-public/
├── production/                          # 🏭 Production Platform
│   ├── market_intelligence/            # 📊 Market Intelligence Platform
│   │   ├── terminal_daily_pulse.py    #     Terminal-based Daily Alpha Pulse
│   │   ├── daily_alpha_pulse.py       #     Advanced analytics (future)
│   │   └── components/                #     Data loaders and utilities
│   ├── engine/                        # ⚙️ Core Engines
│   │   ├── qvm_engine_v2_enhanced.py  #     Enhanced QVM Engine (VALIDATED)
│   │   └── adaptive_engine.py         #     Adaptive Risk Management
│   ├── universe/                      # 🎯 Universe Construction
│   │   └── constructors.py           #     Liquid universe (Top 200, 10B+ ADTV)
│   ├── execution/                     # 🚀 Portfolio Execution
│   └── tests/                         # 🔬 Comprehensive Test Suite
│       ├── phase1-5/                  #     Core engine validation
│       ├── phase12_liquid_alpha_discovery/   # Liquid universe research
│       ├── phase15_composite_model_engineering/ # Composite strategies
│       ├── phase16_weighted_composite_model/    # Current phase work
│       └── phase27_official_baseline/ #     Official baseline models
├── scripts/                           # 🛠️ Production Scripts
│   ├── production_menu.py            #     Reorganized production interface
│   ├── market_intelligence_menu.py   #     Standalone market intel menu
│   └── intermediaries/               #     Data processing utilities
├── docs/                             # 📚 Documentation
│   ├── 2_technical_implementation/   #     QVM Engine specifications
│   └── 3_operational_framework/      #     Production playbooks
└── config/                           # ⚙️ Configuration
```

---

## 🎯 **Key Features**

### **🔍 Market Intelligence**
- **Terminal Daily Alpha Pulse:** Real-time market insights with ASCII dashboard
- **Factor Performance Monitoring:** Live Quality, Value, Momentum tracking
- **Trading Intelligence:** Volume leaders, sector rotation, foreign flows
- **Data Integrity:** Factual data only, no fallbacks or fabrications

### **⚙️ Production Engine**
- **Enhanced QVM Engine v2:** All critical fixes validated, institutional-grade
- **Adaptive Risk Management:** Regime-aware portfolio construction
- **Liquid Universe Focus:** Top 200 stocks, 10B+ VND average daily turnover
- **Version-Aware Processing:** Isolated testing with production safeguards

### **🔬 Institutional Testing Framework**
- **Phases 1-27:** Comprehensive validation from core engine to production models
- **Historical Validation:** 2016-2025 backtesting with out-of-sample testing
- **Risk Management:** Volatility targeting, drawdown controls, regime detection
- **Reality Check:** Transitioned from illusion (2.1+ Sharpe) to genuine alpha (0.91 Sharpe)

### **📊 Current Research Focus (Phase 16)**
- **Problem:** Composite models underperforming standalone Value factor
- **Investigation:** Weight optimization, normalization verification, attribution analysis
- **Goal:** Composite Sharpe > 0.91 through factor diversification benefits

---

## 🚀 **Quick Start**

### **1. Daily Market Intelligence**
```bash
# Launch production menu
python scripts/production_menu.py

# Select Option 0.1 for Daily Alpha Pulse
# Real-time terminal dashboard with market insights
```

### **2. Factor Generation**
```bash
# Access comprehensive factor guide
# Option 7.0 - Factor Generation Guide & Best Practices

# Generate factors for specific date
# Option 7.2 - Generate QVM Factors (Single Date)

# Auto-detect and fill missing dates
# Option 7.3 - Incremental Factor Update (Auto Gap Detection)
```

### **3. Standalone Market Intelligence**
```bash
# Dedicated market intelligence interface
python scripts/market_intelligence_menu.py

# Option 1.1 - Generate Daily Alpha Pulse Dashboard
```

---

## 📈 **Performance Highlights**

### **Liquid Universe Discovery (Phase 12-15)**
- **Genuine Alpha Confirmed:** 0.91 Sharpe ratio in liquid universe
- **Value Factor Dominance:** 12.4% annual return, strong consistency  
- **Momentum Reversal:** -0.37 Sharpe confirms mean reversion in Vietnam
- **Quality Diversification:** Defensive properties with portfolio benefits
- **Alpha Preservation:** ~62% from illusion to tradeable reality

### **Risk Management Evolution**
- **Volatility Targeting:** Adaptive exposure based on market regimes
- **Liquidity Constraints:** Focus on most tradeable stocks (10B+ VND ADTV)
- **Drawdown Controls:** Maximum 15% drawdown limits with regime detection
- **Transaction Costs:** Realistic implementation with impact modeling

---

## 📚 **Documentation**

### **Technical Implementation**
- **[QVM Engine v2 Enhanced Specification](docs/2_technical_implementation/02a_qvm_engine_v2_enhanced_specification.md)** - Complete engine documentation
- **[Temporal Logic & Data Availability](docs/2_technical_implementation/02b_temporal_logic_and_data_availability.md)** - Quarter processing rules
- **[Production Workflow v1.1](docs/2_technical_implementation/02g_quant_workflow_v1.1_production_ready.md)** - Complete production procedures

### **Operational Framework**
- **[Operational Playbook](docs/3_operational_framework/03_operational_playbook.md)** - Day-to-day operations
- **[Factor Validation Playbook](docs/3_operational_framework/03a_factor_validation_playbook.md)** - Quality assurance procedures

### **Research & Backtesting**
- **[QVM Backtesting Framework](docs/4_backtesting_and_research/04_qvm_backtesting_framework.md)** - Institutional backtesting methodology
- **Phase Test Results:** Comprehensive validation in `production/tests/phase*/`

---

## 🎖️ **Development Timeline**

### **Completed Phases**
- **✅ Phase 1-5:** Core QVM Engine validation and temporal logic fixes
- **✅ Phase 7-8:** Institutional backtesting framework and risk management  
- **✅ Phase 12:** Liquid Alpha Discovery - reality check and pivot
- **✅ Phase 14:** Liquid Universe Full Backtesting - 0.91 Sharpe confirmed
- **✅ Phase 15:** Composite Model Engineering - issue identified

### **Current Phase**
- **🔄 Phase 16:** Weighted Composite Model Engineering - diagnosing underperformance

### **Upcoming Priorities**
- **Phase 17:** Final validation and out-of-sample robustness testing
- **Phase 18:** Risk-managed production model with regime detection
- **Phase 20+:** Production deployment and live portfolio management

---

## ⚡ **Quick Commands**

```bash
# Production menu with Market Intelligence
python scripts/production_menu.py

# Standalone Market Intelligence
python scripts/market_intelligence_menu.py

# Factor generation (production-ready)
python production/scripts/run_factor_generation.py --start-date 2025-07-30 --end-date 2025-07-30

# System health check
python scripts/production_menu.py  # Option 5.1
```

---

## 🏛️ **Institutional Grade Features**

- **Data Integrity:** Comprehensive validation with 99%+ processing efficiency
- **Risk Management:** Multi-layer controls with regime detection
- **Performance Attribution:** Factor-level breakdown and sector analysis
- **Operational Procedures:** Complete documentation and monitoring tools
- **Compliance Ready:** Audit trails and validation frameworks

---

## 📝 **Note**

This is a **public subset** of a larger proprietary trading system. Sensitive data, credentials, database connections, and certain proprietary algorithms have been excluded. The framework demonstrates institutional-quality factor investing methodology while protecting intellectual property.

**Current Focus:** Phase 16 diagnostic work on composite model construction and weight optimization.

---

## 📄 **License**

This project is provided for **educational and research purposes**. Commercial use requires explicit permission.

**Author:** Duc Nguyen  
**Status:** Production Platform - Phase 16 Active  
**Last Updated:** July 31, 2025