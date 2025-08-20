# Factor Generation Guide

## Overview

The QVM Engine v2.1.1 Flat generates composite factor scores using a 4-pillar architecture:
- **Quality (35%)**: ROAE, ROAA, Piotroski F-Score
- **Value (30%)**: Earnings Yield, Book-to-Price, Sales-to-Price, EBITDA/EV, FCF Yield
- **Momentum (20%)**: 3-month, 6-month, 12-month price momentum
- **Defensive (15%)**: Low Volatility (63-day)

## Engine Architecture

### Core Engine: `qvm_engine_v2_1_1_flat.py`
- Flat methodology: Single-step combination of all factors
- Sector neutralization: Every factor is z-scored within sectors
- Component transparency: Full factor attribution available

### Performance Enhancement: Vectorized F-Score
- `qvm_engine_v2_1_1_flat_fscore_vectorized.py`
- 30x performance improvement (74 seconds vs 5+ minutes per date)
- Per-date caching eliminates redundant calculations
- Automatic fallback to database methods if needed

## Running Factor Generation

### Basic Command Structure
```bash
python production/scripts/run_factor_generation.py \
  --start-date YYYY-MM-DD \
  --end-date YYYY-MM-DD \
  --mode [incremental|refresh] \
  --version strategy_version \
  --batch-size N
```

### Parameters Explained

- **`--start-date`, `--end-date`**: Date range to process
- **`--mode`**: 
  - `incremental`: Only process missing dates (recommended)
  - `refresh`: Delete and regenerate all dates (use carefully)
- **`--version`**: Strategy version tag (default: `qvm_v2.1.1_flat_corrected`)
- **`--batch-size`**: Dates to process per batch (default: 30)

### Examples

#### Daily Production Update
```bash
# Process last 30 days, only missing dates
python production/scripts/run_factor_generation.py \
  --start-date $(date -d "30 days ago" +%Y-%m-%d) \
  --end-date $(date -d "yesterday" +%Y-%m-%d) \
  --mode incremental \
  --version qvm_v2.1.1_flat_corrected
```

#### Historical Backfill
```bash
# Generate full year 2024
python production/scripts/run_factor_generation.py \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --mode incremental \
  --version qvm_v2.1.1_flat_corrected \
  --batch-size 50
```

#### Single Date Testing
```bash
# Test specific date
python production/scripts/run_factor_generation.py \
  --start-date 2025-08-19 \
  --end-date 2025-08-19 \
  --mode incremental \
  --version qvm_v2.1.1_flat_corrected
```

## Data Flow

1. **Universe Construction**
   - Loads tickers from `master_info` table
   - Applies liquidity filters (configurable)

2. **Data Retrieval**
   - Market data from `equity_history`
   - Fundamentals from `intermediary_calculations_*` tables
   - Market cap from `vcsc_daily_data_complete`

3. **Factor Calculation**
   - Individual factor computation
   - Sector neutralization (z-scores within sectors)
   - Winsorization at 3 standard deviations

4. **Composite Generation**
   - Weighted average of pillar scores
   - Final QVM composite score

5. **Database Storage**
   - Table: `factor_scores_qvm`
   - Columns: Quality_Composite, Value_Composite, Momentum_Composite, Defensive_Composite, QVM_Composite
   - Additional: Low_Volatility_63D, Piotroski_F_Score, FCF_Yield

## Performance Optimization

### Feature Flags
```bash
# Use vectorized F-Score (default, recommended)
export F_SCORE_IMPL=vectorized

# Use database methods with timeout fallback
export F_SCORE_IMPL=db
export F_SCORE_TIMEOUT_S=30
```

### Parallel Processing
For large historical generation, run multiple year ranges in parallel:
```bash
# Terminal 1
python production/scripts/run_factor_generation.py --start-date 2020-01-01 --end-date 2020-12-31 --mode incremental &

# Terminal 2
python production/scripts/run_factor_generation.py --start-date 2021-01-01 --end-date 2021-12-31 --mode incremental &

# Terminal 3
python production/scripts/run_factor_generation.py --start-date 2022-01-01 --end-date 2022-12-31 --mode incremental &
```

## Monitoring Progress

### Real-time SQL Monitoring
```sql
-- Check generation progress by year
SELECT 
    YEAR(date) as year,
    COUNT(*) as records,
    COUNT(DISTINCT date) as trading_days,
    COUNT(DISTINCT ticker) as unique_tickers,
    MIN(date) as start_date,
    MAX(date) as end_date
FROM factor_scores_qvm 
WHERE strategy_version = 'qvm_v2.1.1_flat_corrected'
GROUP BY YEAR(date)
ORDER BY year;
```

### Python Status Check
```bash
python scripts/monitoring/check_factor_generation_status.py
```

## Troubleshooting

### Common Issues and Solutions

1. **"No data available for date X"**
   - Check intermediary calculations are complete
   - Verify market data exists for the date
   - Run pipeline status check

2. **Performance Issues**
   - Ensure F_SCORE_IMPL=vectorized is set
   - Reduce batch size for memory constraints
   - Check database connection performance

3. **Inconsistent Factor Scores**
   - Verify consistent version tag usage
   - Check for data quality issues in source tables
   - Review sector mappings in master_info

### Validation Queries
```sql
-- Check for zero/null composite scores
SELECT date, COUNT(*) as zero_scores
FROM factor_scores_qvm
WHERE strategy_version = 'qvm_v2.1.1_flat_corrected'
  AND QVM_Composite = 0
GROUP BY date
ORDER BY date DESC
LIMIT 10;

-- Verify factor distribution
SELECT 
    date,
    AVG(QVM_Composite) as avg_qvm,
    STDDEV(QVM_Composite) as std_qvm,
    MIN(QVM_Composite) as min_qvm,
    MAX(QVM_Composite) as max_qvm
FROM factor_scores_qvm
WHERE strategy_version = 'qvm_v2.1.1_flat_corrected'
GROUP BY date
ORDER BY date DESC
LIMIT 5;
```

## Best Practices

1. **Always use incremental mode** for production updates
2. **Monitor first few dates** when running large batches
3. **Backup before refresh mode** operations
4. **Verify data prerequisites** before generation
5. **Use consistent version tags** across all dates
6. **Document any custom modifications** to the engine

## Output Schema

The `factor_scores_qvm` table contains:
```sql
CREATE TABLE factor_scores_qvm (
    ticker VARCHAR(10),
    date DATE,
    Quality_Composite DECIMAL(20,10),
    Value_Composite DECIMAL(20,10),
    Momentum_Composite DECIMAL(20,10),
    Defensive_Composite DECIMAL(20,10),
    QVM_Composite DECIMAL(20,10),
    Low_Volatility_63D DECIMAL(10,6),
    Piotroski_F_Score INT,
    FCF_Yield DECIMAL(10,6),
    calculation_timestamp TIMESTAMP,
    strategy_version VARCHAR(50),
    PRIMARY KEY (ticker, date, strategy_version)
);
```