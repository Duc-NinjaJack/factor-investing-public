# Analytics Sidecar Guide

## Overview

The Analytics Sidecar captures individual factor values (not composites) at the point of sector neutralization. This provides RAW factor exposures for attribution analysis and research.

## Architecture

### Non-Intrusive Design
- **Wrapper-based**: Uses pandas GroupBy interposer (Tap-B approach)
- **No engine modifications**: Wraps existing sector neutralization methods
- **Parallel data capture**: Doesn't affect composite generation
- **Version isolated**: Uses separate strategy_version tag

### Data Tables

1. **`factor_signals_raw`**: Individual factor values per ticker
   - Primary key: (strategy_version, date, ticker, factor_id)
   - Stores raw factor values before composite aggregation

2. **`factor_norm_stats`**: Sector normalization statistics
   - Primary key: (strategy_version, date, factor_id, sector)
   - Stores mean, std dev, and universe size per sector

3. **`dim_factor`**: Factor metadata registry
   - Maps factor_id to factor_code and descriptions

## Running Analytics Extraction

### Command Structure
```bash
python production/scripts/run_factor_analytics_wrapper_neo_fix.py \
  --version strategy_version \
  --factors [factor_list | all] \
  --mode [daily | from-last | range] \
  --audit-tier [tier0 | tier1 | none] \
  --write-sidecar \
  --resume \
  --batch-size N
```

### Parameters

- **`--version`**: Analytics version tag (default: `analytics_v1_neo_fixed`)
- **`--factors`**: Space-separated factor list or "all"
- **`--mode`**: 
  - `daily`: Process latest composite date
  - `from-last`: Continue from last processed date
  - `range`: Specify start/end dates
- **`--audit-tier`**: 
  - `tier0`: Strict validation (recommended)
  - `tier1`: Relaxed validation
  - `none`: No validation
- **`--write-sidecar`**: Enable database writes
- **`--resume`**: Skip already processed dates
- **`--batch-size`**: Dates per batch (default: 30)

## Available Factors

### Quality Factors
- `roae`: Return on Average Equity
- `roaa`: Return on Average Assets
- `nim`: Net Interest Margin (banking)
- `cost_income`: Cost-to-Income Ratio (banking)
- `f_score`: Piotroski F-Score

### Profitability Margins
- `net_profit_margin`: Net Profit Margin
- `gross_margin`: Gross Margin
- `operating_margin`: Operating Margin
- `ebitda_margin`: EBITDA Margin

### Value Factors
- `earnings_yield`: Earnings Yield
- `book_to_price`: Book-to-Price
- `sales_to_price`: Sales-to-Price
- `ebitda_to_ev`: EBITDA/EV
- `fcf_yield`: Free Cash Flow Yield

### Momentum Factors
- `mom_3m`: 3-month momentum
- `mom_6m`: 6-month momentum
- `mom_12m`: 12-month momentum

### Defensive Factors
- `low_volatility`: 63-day volatility

## Usage Examples

### Daily Production Update
```bash
# Extract all factors for latest trading date
python production/scripts/run_factor_analytics_wrapper_neo_fix.py \
  --version analytics_v1_neo_fixed \
  --factors all \
  --mode daily \
  --audit-tier tier0 \
  --write-sidecar \
  --batch-size 30
```

### Incremental Backfill
```bash
# Continue from last processed date
python production/scripts/run_factor_analytics_wrapper_neo_fix.py \
  --version analytics_v1_neo_fixed \
  --factors all \
  --mode from-last \
  --audit-tier tier0 \
  --write-sidecar \
  --resume \
  --batch-size 50
```

### Historical Range with Specific Factors
```bash
# Extract value and momentum factors for Q1 2025
python production/scripts/run_factor_analytics_wrapper_neo_fix.py \
  --start-date 2025-01-01 \
  --end-date 2025-03-31 \
  --version analytics_v1_neo_fixed \
  --factors earnings_yield book_to_price mom_3m mom_6m mom_12m \
  --mode range \
  --audit-tier tier0 \
  --write-sidecar \
  --resume
```

## Audit Tiers

### Tier 0 (Strict)
- Requires all dates to pass validation
- Writes only on successful audit
- Ensures data quality and completeness
- Recommended for production

### Tier 1 (Relaxed)
- Allows some validation failures
- Writes partial data
- Useful for research/debugging

### None
- No validation performed
- Fastest but no quality guarantees
- Use only for testing

## Monitoring and Validation

### Check Sidecar Status
```bash
python scripts/monitoring/check_sidecar_status.py
```

### SQL Monitoring Queries

#### Overall Coverage
```sql
SELECT 
    COUNT(*) as total_records,
    COUNT(DISTINCT date) as trading_days,
    COUNT(DISTINCT ticker) as unique_tickers,
    COUNT(DISTINCT factor_id) as unique_factors,
    MIN(date) as earliest_date,
    MAX(date) as latest_date
FROM factor_signals_raw
WHERE strategy_version = 'analytics_v1_neo_fixed';
```

#### Per-Factor Coverage
```sql
SELECT 
    df.factor_code,
    COUNT(DISTINCT fsr.date) as days_covered,
    COUNT(DISTINCT fsr.ticker) as tickers_covered,
    MIN(fsr.date) as earliest,
    MAX(fsr.date) as latest
FROM factor_signals_raw fsr
JOIN dim_factor df ON fsr.factor_id = df.factor_id
WHERE fsr.strategy_version = 'analytics_v1_neo_fixed'
GROUP BY df.factor_code
ORDER BY df.factor_code;
```

#### Gap Analysis
```sql
-- Find dates in composites but missing in sidecar
WITH composite_dates AS (
    SELECT DISTINCT date 
    FROM factor_scores_qvm 
    WHERE strategy_version = 'qvm_v2.1.1_flat_corrected'
),
sidecar_dates AS (
    SELECT DISTINCT date 
    FROM factor_signals_raw 
    WHERE strategy_version = 'analytics_v1_neo_fixed'
)
SELECT date FROM composite_dates 
WHERE date NOT IN (SELECT date FROM sidecar_dates)
ORDER BY date DESC
LIMIT 10;
```

## Data Quality Checks

### Canonical View
Use the canonical view for standardized factor names:
```sql
SELECT * FROM v_factor_signals_raw_canonical
WHERE date = '2025-08-19'
  AND ticker = 'VCB'
ORDER BY factor_code;
```

### Distribution Analysis
```sql
-- Check factor distributions
SELECT 
    factor_code,
    AVG(value) as mean_value,
    STDDEV(value) as std_value,
    MIN(value) as min_value,
    MAX(value) as max_value,
    COUNT(*) as observations
FROM v_factor_signals_raw_canonical
WHERE date = '2025-08-19'
GROUP BY factor_code;
```

## Troubleshooting

### Common Issues

1. **"Audit failed for date X"**
   - Check composite data exists for the date
   - Verify all required factors are calculated
   - Review audit logs for specific failures

2. **Missing factors**
   - Some factors are optional under tier0
   - Check dim_factor for available definitions
   - Verify sector-specific factors (e.g., NIM for banking)

3. **Slow performance**
   - Reduce batch size for memory optimization
   - Ensure database indexes are present
   - Check network/database latency

### Validation Checklist

1. ✓ Composite dates exist in factor_scores_qvm
2. ✓ All required factors defined in dim_factor
3. ✓ Database connection stable
4. ✓ Sufficient disk space for sidecar tables
5. ✓ Appropriate audit tier selected

## Best Practices

1. **Use tier0 audit** for production data quality
2. **Enable --resume** to avoid reprocessing
3. **Monitor first batch** before large runs
4. **Verify factor coverage** after extraction
5. **Compare with composites** for consistency
6. **Document custom factors** in dim_factor

## Integration with Composites

The sidecar data complements composite scores:
- Composites: Aggregated scores for portfolio construction
- Sidecar: Individual factors for attribution analysis

Both use the same underlying engine but different storage:
- Composites → `factor_scores_qvm`
- Individual → `factor_signals_raw` + `factor_norm_stats`

Version tags keep them separate:
- Composites: `qvm_v2.1.1_flat_corrected`
- Analytics: `analytics_v1_neo_fixed`