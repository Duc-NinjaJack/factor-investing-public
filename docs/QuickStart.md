# Quick Start Guide

## Prerequisites

1. **Python Environment**
   - Python 3.8+ required
   - Virtual environment recommended

2. **Database Access**
   - MySQL 8.0+ database
   - Access to `alphabeta` schema
   - Tables populated with market and fundamental data

3. **Configuration**
   ```bash
   # Copy database config template
   cp production/config/database.yml.example production/config/database.yml
   
   # Edit database.yml with your credentials
   vim production/config/database.yml
   ```

## Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Apply database migrations (if needed)
mysql -u your_user -p alphabeta < production/database/migrations/011_add_factor_analytics_sidecar_neo.sql
mysql -u your_user -p alphabeta < production/database/migrations/012_add_missing_dim_factors.sql
mysql -u your_user -p alphabeta < production/database/migrations/013_create_view_factor_signals_raw_canonical.sql
```

## Quick Commands

### 1. Generate Composite Factor Scores

```bash
# Incremental update (recommended for daily runs)
python production/scripts/run_factor_generation.py \
  --start-date 2025-08-01 \
  --end-date 2025-08-20 \
  --mode incremental \
  --version qvm_v2.1.1_flat_corrected

# Full refresh (use carefully - overwrites existing data)
python production/scripts/run_factor_generation.py \
  --start-date 2025-01-01 \
  --end-date 2025-08-20 \
  --mode refresh \
  --version qvm_v2.1.1_flat_corrected
```

### 2. Extract Individual Factor Values (Analytics Sidecar)

```bash
# Daily update (latest composite date)
python production/scripts/run_factor_analytics_wrapper_neo_fix.py \
  --version analytics_v1_neo_fixed \
  --factors all \
  --mode daily \
  --audit-tier tier0 \
  --write-sidecar \
  --resume \
  --batch-size 30

# Continue from last processed date
python production/scripts/run_factor_analytics_wrapper_neo_fix.py \
  --version analytics_v1_neo_fixed \
  --factors all \
  --mode from-last \
  --audit-tier tier0 \
  --write-sidecar \
  --resume \
  --batch-size 30

# Specific date range with selected factors
python production/scripts/run_factor_analytics_wrapper_neo_fix.py \
  --start-date 2025-08-01 \
  --end-date 2025-08-20 \
  --version analytics_v1_neo_fixed \
  --factors roae roaa book_to_price earnings_yield mom_3m mom_6m \
  --mode range \
  --audit-tier tier0 \
  --write-sidecar \
  --batch-size 30
```

### 3. Check Status

```bash
# Check composite factor generation status
python scripts/monitoring/check_factor_generation_status.py

# Check analytics sidecar status
python scripts/monitoring/check_sidecar_status.py

# Check data pipeline status
python scripts/monitoring/check_processing_pipeline_status.py
```

## Environment Variables

```bash
# F-Score implementation (optional)
export F_SCORE_IMPL=vectorized  # Default: vectorized (30x faster)
# export F_SCORE_IMPL=db        # Use database methods with fallback

# F-Score timeout for DB method (optional)
export F_SCORE_TIMEOUT_S=30     # Default: 30 seconds
```

## Version Tags

- **Composites**: `qvm_v2.1.1_flat_corrected` (production)
- **Analytics**: `analytics_v1_neo_fixed` (sidecar data)

## Important Notes

1. **Always use incremental mode** for daily updates to avoid overwriting data
2. **Backup before refresh mode** - it deletes and regenerates data
3. **Check pipeline status first** - ensure data prerequisites are ready
4. **Monitor performance** - typical runtime is ~74 seconds per trading date
5. **Audit tier0 is strict** - it ensures data quality but may skip some dates

## Troubleshooting

If factor generation fails:
1. Check data pipeline status: `python scripts/monitoring/check_processing_pipeline_status.py`
2. Verify intermediary calculations are complete
3. Ensure database connection is stable
4. Check available disk space

For missing factors in analytics:
- Some factors may be optional under tier0 audit
- Check dim_factor table for available factor definitions
- Review audit logs for specific failure reasons

## Support

For issues or questions:
- Review logs in `production/scripts/`
- Check documentation in `docs/`
- Verify database migrations are applied