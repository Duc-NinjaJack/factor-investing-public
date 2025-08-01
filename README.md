# Vietnam Factor Investing Platform

**Internal team repository for QVM factor investing system**

## Current Status

**Working on:** Composite model underperformance issue
- Value standalone: 0.91 Sharpe
- Composite models: 0.44-0.59 Sharpe  
- Need to debug weight optimization and normalization

## Quick Start

### Daily Operations
```bash
# Production menu (Market Intel is now Option 0)
python scripts/production_menu.py

# Market intelligence only  
python scripts/market_intelligence_menu.py
```

### Key Options
- **0.1** - Daily Alpha Pulse (terminal dashboard)
- **7.3** - Incremental factor update (auto gap detection)
- **5.5** - Processing pipeline status check

## Repository Structure

```
production/
├── market_intelligence/           # NEW: Terminal market dashboard
│   └── terminal_daily_pulse.py   # Factual data only
├── engine/
│   ├── qvm_engine_v2_enhanced.py # Validated production engine
│   └── adaptive_engine.py        # Risk-managed strategies
├── universe/
│   └── constructors.py           # Liquid universe (Top 200, 10B+ ADTV)
└── tests/                        # Comprehensive test suite
```

## Recent Updates

### Market Intelligence Platform
- Terminal-based Daily Alpha Pulse operational
- Shows VN-Index, market breadth, factor performance, top stocks
- Access via production menu Option 0.1

### Production Menu Redesign
- Market Intelligence moved to Option 0 (prominent placement)
- Side-by-side layout for better organization
- Factor Generation moved to Section 7

## Database

**Main Tables:**
- `factor_scores_qvm` - Factor scores (version-aware)
- `equity_history` - Adjusted OHLCV (16+ years)
- `vcsc_daily_data_complete` - Market microstructure
- `v_comprehensive_fundamental_items` - Point-in-time fundamentals

**Current Data Status:**
- Q2 2025: Available in intermediaries
- Temporal logic: Q1 2025 active until Aug 14
- Recommended version: qvm_v2.0_enhanced

## Key Commands

```bash
# Factor generation
python production/scripts/run_factor_generation.py --start-date 2025-07-30 --end-date 2025-07-30 --version qvm_v2.0_enhanced

# System health check
python scripts/production_menu.py  # Option 3.6

# Market intelligence
python production/market_intelligence/terminal_daily_pulse.py
```

## Team Notes

- **Database:** alphabeta (MySQL), requires VPN for remote access  
- **Engine Status:** v2 Enhanced validated and production-ready
- **Market Intel:** Terminal interface operational
- **Current Focus:** Debugging composite model underperformance

## Configuration

- `config/database.yml` - DB connection (not in repo)
- `production/config/backtest_config.yml` - Backtesting parameters
- `config/sector_mapping.yml` - Sector classifications

---

Author: Duc Nguyen  
Last Updated: July 31, 2025
>>>>>>> origin/master
