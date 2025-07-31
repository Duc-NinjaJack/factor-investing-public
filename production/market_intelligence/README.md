# Market Intelligence Module

## Overview
Provides daily and weekly quantitative market intelligence for the Vietnam Factor Investing Platform.

## Current Structure (After Cleanup)
```
market_intelligence/
├── __init__.py                    # Module definition
├── terminal_daily_pulse.py       # ✅ Primary terminal-based daily pulse
├── simple_daily_pulse.py         # Alternative simplified version
├── daily_alpha_pulse.py          # HTML dashboard (future development)
├── components/
│   ├── __init__.py
│   └── data_loader.py           # Shared data loading utilities
├── config/
│   ├── __init__.py
│   └── config.py                # Configuration settings
└── reports/                     # Generated reports (auto-created)
```

## Cleaned Up Files
- ❌ Removed: `terminal_daily_pulse_old.py` (deprecated)
- ❌ Removed: `reports/daily_alpha_pulse_20250731.html` (generated output)
- ❌ Removed: `templates/` (empty directory)

## Integration with Production Menu

### New Menu Section: 7. MARKET INTELLIGENCE
- **7.1** - 📊 Daily Alpha Pulse (Terminal)
- **7.2** - 📈 Market Intelligence Dashboard (Future)

## Usage

### Terminal Daily Pulse
```bash
# Run directly
python3 production/market_intelligence/terminal_daily_pulse.py

# Or via production menu
# Select option 7.1
```

## Features

### Terminal Daily Pulse
- Market overview and breadth
- VN-Index performance
- Factor performance monitoring (if available)
- Risk metrics and regime analysis
- Trading signals and opportunities

## Future Development
The market intelligence module is kept separate for continued development:
- Advanced dashboard capabilities
- Weekly strategic alpha reviews
- Enhanced visualizations
- Real-time market monitoring

## Database Dependencies
- `etf_history` - VN-Index data
- `equity_history` - Individual stock data
- `vcsc_daily_data_complete` - Market microstructure
- `factor_scores_qvm` - Factor performance (optional)

## Status
- ✅ Terminal version: Ready for daily use
- 🚧 Dashboard version: Under development
- 📊 Production menu integration: Complete