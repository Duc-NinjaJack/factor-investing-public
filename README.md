# Vietnam Factor Investing Platform

**Production-ready QVM factor generation system with analytics sidecar**

## Current Status

**Engine Version:** QVM v2.1.1 Flat (4-pillar architecture)
- Quality (35%), Value (30%), Momentum (20%), Defensive (15%)
- Vectorized F-Score: 30x performance improvement
- Analytics sidecar: 18 individual factors captured

## Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/your-org/factor-investing-public.git
cd factor-investing-public

# Setup environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure database
cp production/config/database.yml.example production/config/database.yml
# Edit database.yml with your credentials

# Apply migrations (if needed)
mysql -u user -p alphabeta < production/database/migrations/*.sql
```

### Daily Operations
```bash
# Production menu with all options
python scripts/production_menu.py

# Generate composite factors (Option 7.3 - recommended)
python production/scripts/run_factor_generation.py \
  --start-date 2025-08-01 --end-date 2025-08-20 \
  --mode incremental --version qvm_v2.1.1_flat_corrected

# Extract individual factors (Option 7.7)
python production/scripts/run_factor_analytics_wrapper_neo_fix.py \
  --version analytics_v1_neo_fixed --factors all \
  --mode daily --audit-tier tier0 --write-sidecar
```

### Key Menu Options
- **0.1** - Daily Alpha Pulse (market intelligence)
- **7.1** - Factor generation (date range)
- **7.3** - Incremental update (auto gap detection)
- **7.7** - Analytics extraction (individual factors)
- **t** - Consolidated production status

## Repository Structure

```
production/
├── engine/
│   ├── qvm_engine_v2_1_1_flat.py           # Core 4-pillar engine
│   ├── qvm_engine_v2_1_1_flat_fscore_vectorized.py  # Vectorized F-Score
│   └── qvm_engine_v2_enhanced.py           # Legacy v2 enhanced
├── scripts/
│   ├── run_factor_generation.py            # Composite factor generation
│   └── run_factor_analytics_wrapper_neo_fix.py  # Individual factor extraction
├── universe/
│   └── constructors.py                     # Universe construction logic
├── database/migrations/
│   ├── 011_add_factor_analytics_sidecar_neo.sql
│   ├── 012_add_missing_dim_factors.sql
│   └── 013_create_view_factor_signals_raw_canonical.sql
└── config/
    ├── database.yml.example                # Database config template
    └── strategy_config.yml                 # Strategy parameters

scripts/
├── production_menu.py                      # Main production interface
└── monitoring/
    ├── check_factor_generation_status.py   # Composite monitoring
    └── check_sidecar_status.py            # Analytics monitoring

docs/
├── QuickStart.md                          # Setup and basic commands
├── Factor_Generation.md                   # Composite generation guide
└── Analytics_Sidecar.md                  # Individual factors guide
```

## Features

### Composite Factor Generation
- **Engine:** QVM v2.1.1 Flat with 4-pillar architecture
- **Performance:** 74 seconds per trading date (30x improvement)
- **Storage:** `factor_scores_qvm` table
- **Version:** `qvm_v2.1.1_flat_corrected`

### Individual Factor Analytics (Sidecar)
- **Method:** Non-intrusive wrapper at sector neutralization
- **Factors:** 18 individual factors captured
- **Storage:** `factor_signals_raw` and `factor_norm_stats`
- **Version:** `analytics_v1_neo_fixed`

## Database Schema

### Required Tables
- `factor_scores_qvm` - Composite factor scores
- `factor_signals_raw` - Individual factor values
- `factor_norm_stats` - Normalization statistics
- `dim_factor` - Factor metadata
- `equity_history` - Market data
- `vcsc_daily_data_complete` - Market microstructure
- `intermediary_calculations_*` - Fundamental metrics

### Version Management
- Composite version: `qvm_v2.1.1_flat_corrected`
- Analytics version: `analytics_v1_neo_fixed`
- Versions are isolated - no cross-contamination

## Documentation

- [Quick Start Guide](docs/QuickStart.md) - Installation and setup
- [Factor Generation](docs/Factor_Generation.md) - Composite scores
- [Analytics Sidecar](docs/Analytics_Sidecar.md) - Individual factors

## Environment Variables

```bash
# F-Score implementation (optional)
export F_SCORE_IMPL=vectorized  # Default, 30x faster
export F_SCORE_TIMEOUT_S=30     # Timeout for DB fallback
```

## Support

For issues or questions:
1. Check documentation in `docs/` folder
2. Review logs in script directories
3. Run monitoring scripts for status checks

---

**Version:** 2.1.1  
**Last Updated:** August 2025
