# Phase 30 HPBD - QVM Strategy Analysis

This directory contains the reorganized QVM (Quality, Value, Momentum) strategy analysis files from phase 29, properly structured for better organization and maintainability.

## Directory Structure

```
phase30_hpbd/
├── 01_tearsheet_demonstration.py      # Main analysis script (renamed from 19_*)
├── 01_tearsheet_demonstration.ipynb   # Jupyter notebook version
├── docs/                              # Generated analysis outputs
│   ├── 18b_complete_holdings.csv      # Holdings data dependency
│   ├── 19_comprehensive_tearsheet.txt # Comprehensive analysis results
│   ├── 19_equity_curve.png           # Equity curve visualization
│   ├── 19_tearsheet_daily_returns.csv # Daily returns data
│   ├── 19_tearsheet_performance_metrics.txt # Performance metrics
│   └── 19_tearsheet_portfolio_values.csv # Portfolio values over time
├── scripts/                           # Helper scripts (currently empty)
├── archived/                          # Old analysis files
│   └── insights/                      # Moved from project root
└── README.md                          # This file
```

## Key Changes from Phase 29

1. **File Renumbering**: 19_* files renamed to 01_* for better organization
2. **Path Updates**: All file paths updated to use `docs/` instead of `insights/`
3. **Dependency Management**: Required data files copied to `docs/` folder
4. **Clean Organization**: Analysis outputs in `docs/`, old files in `archived/`

## Running the Analysis

The main analysis can be run using either:
- `python 01_tearsheet_demonstration.py`
- `jupyter notebook 01_tearsheet_demonstration.ipynb`

Both files have been updated with correct paths and dependencies.

## Dependencies

- Database connection to factor investing database
- Required data file: `docs/18b_complete_holdings.csv`
- Python packages: pandas, numpy, matplotlib, seaborn

## Output Files

All analysis outputs are saved to the `docs/` folder:
- Performance metrics and statistics
- Portfolio values and daily returns
- Equity curve visualizations
- Comprehensive tearsheet analysis
