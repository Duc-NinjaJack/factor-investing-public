# Phase 31: QVM Engine v3 with Piotroski F-Score Integration

## Overview
This phase implements a comprehensive QVM (Quality, Value, Momentum) factor investing strategy with Piotroski F-Score integration and **drawdown protection risk management**. The strategy combines enhanced quality factor calculation with dynamic position sizing based on market drawdown levels.

## Key Features

### 1. Piotroski F-Score Integration
- **Sector-specific F-Score calculations**:
  - Non-Financial: 9 tests (ROA>0, CFO>0, ΔROA>0, etc.)
  - Banking: 6 tests (ROA>0, NIM>0, ΔROA>0, etc.)
  - Securities: 5 tests (ROA>0, BrokerageRatio>0, ΔROA>0, etc.)
- **Enhanced Quality Factor**: F-Score integrated with 15% weight
- **Real-time calculation** from database tables

### 2. Drawdown Protection Risk Management ⭐ **NEW**
- **Dynamic position sizing** based on benchmark (VNINDEX) drawdown levels
- **4x tighter protection** compared to standard strategies:
  - 0% to -5% drawdown: 100% allocation
  - -5% to -10% drawdown: 20% allocation (80% reduction)
  - -10% to -15% drawdown: 40% allocation (60% reduction)
  - -15% to -20% drawdown: 60% allocation (40% reduction)
  - -20% to -25% drawdown: 80% allocation (20% reduction)
  - -25% to -30% drawdown: 100% allocation (no reduction)
  - -30%+ drawdown: 100% allocation (no reduction)
- **Step-based allocation changes** to reduce transaction costs
- **Visual shading** on equity curve showing reduced allocation periods

### 3. Enhanced Quality Factor Weighting
- **Level**: 40% (ROAE, ROAA, Operating Margin, EBITDA Margin)
- **Change**: 25% (momentum)
- **Acceleration**: 20% (second derivative)
- **F-Score**: 15% (Piotroski F-Score)

### 4. Factor Weights (Matching 04c Strategy)
- **Quality**: 33.33% (with F-Score integration)
- **Value**: 33.33%
- **Momentum**: 33.34%

## Components

### Core Engine
- `QVMEngineV3FScore`: Main engine with F-Score integration
- `PiotroskiFScoreCalculator`: Sector-specific F-Score calculations
- `DrawdownProtectionStrategy`: Risk management with dynamic allocation

### Strategy Classes
- `FScoreIntegrationStrategy`: F-Score integration logic
- `DrawdownProtectionStrategy`: Drawdown-based position sizing

### Database Integration
- `intermediary_calculations_enhanced`: Non-financial sector data
- `intermediary_calculations_banking_cleaned`: Banking sector data
- `intermediary_calculations_securities_cleaned`: Securities sector data
- `vcsc_daily_data_complete`: Price and volume data
- `etf_history`: Benchmark data (VNINDEX)

## File Structure
```
phase31_add_fscore/
├── 01_tearsheet_fscore_integration.py      # Main tearsheet with F-Score + Drawdown protection
├── 01_tearsheet_fscore_integration.ipynb   # Jupyter notebook version
├── scripts/
│   └── run_fscore_analysis.py              # Execution script
├── docs/                                    # Output files
├── insights/                                # Analysis documentation
└── README.md                                # This file
```

## Configuration
```yaml
factor_weights:
  quality: 0.3333    # 33.33% Quality (with F-Score)
  value: 0.3333      # 33.33% Value
  momentum: 0.3334   # 33.34% Momentum

fscore_integration:
  quality_weight: 0.40   # Quality factor weight
  fscore_weight: 0.15    # F-Score weight within quality

drawdown_protection:
  step_size: 0.10        # 10% steps for allocation changes
  max_allocation: 1.0    # 100% allocation at peak
  min_allocation: 0.20   # 20% allocation at max drawdown
```

## Expected Performance
- **Risk Management**: Significant drawdown reduction through dynamic allocation
- **Quality Enhancement**: F-Score integration improves stock selection quality
- **Adaptive Strategy**: Automatically adjusts to market conditions
- **Transaction Efficiency**: Step-based allocation changes reduce costs

## Recent Fixes Applied
1. **Cash Allocation Bug Fix**: Corrected position size calculation from `(current_capital * quality_factor) / len(holdings)` to `current_capital / len(holdings)`
2. **Chart Cleanup**: Removed unnecessary F-Score integration shading from equity curve
3. **Drawdown Protection Integration**: Added complete risk management system matching 04c strategy
4. **Factor Weight Alignment**: Updated to match 04c strategy weights (33.33% each)

## Usage
```bash
# Run the analysis
python scripts/run_fscore_analysis.py

# Or run directly
python 01_tearsheet_fscore_integration.py
```

## Dependencies
- pandas, numpy, matplotlib, seaborn
- Database connection to factor investing database
- QVM Engine v3 with F-Score capabilities
- Piotroski F-Score calculator components


