# Database Content Analysis

**Date:** 2025-07-29
**Purpose:** Analyze actual database content without filters
**Context:** Investigation of limited database data in discrepancy analysis

## 🎯 Executive Summary

- **Factor Scores Table:** 1,567,488 records, 714 tickers
- **Price Data Table:** 2,319,796 records, 728 tickers
- **ETF History Table:** 18,832 records, 9 tickers
- **Pickle Data:** 1,567,488 records, 714 tickers

## 📊 Detailed Analysis

### Factor Scores Table (factor_scores_qvm)

| Metric | Value |
|--------|-------|
| Total Records | 1,567,488 |
| Date Range | 2016-01-04 to 2025-07-25 |
| Unique Dates | 2384 |
| Unique Tickers | 714 |

### Price Data Table (vcsc_daily_data_complete)

| Metric | Value |
|--------|-------|
| Total Records | 2,319,796 |
| Date Range | 2010-01-04 to 2025-07-25 |
| Unique Dates | 3882 |
| Unique Tickers | 728 |

### ETF History Table (etf_history)

| Metric | Value |
|--------|-------|
| Total Records | 18,832 |
| Date Range | 2010-01-04 to 2025-07-28 |
| Unique Dates | 3881 |
| Unique Tickers | 9 |
| Available Tickers | ['E1VFVN30', 'FUEMAV30', 'FUESSV30', 'FUESSV50', 'FUESSVFL', 'FUEVFVND', 'FUEVN100', 'VN30', 'VNINDEX'] |

### Pickle Data (unrestricted_universe_data.pkl)

| Metric | Value |
|--------|-------|
| Factor Data Shape | (1567488, 6) |
| ADTV Data Shape | (2389, 714) |
| Date Range | 2016-01-04 to 2025-07-25 |
| Unique Dates | 2384 |
| Unique Tickers | 714 |

## 🔍 Comparison Analysis

### Database vs Pickle Coverage

| Metric | Database | Pickle | Ratio |
|--------|----------|--------|-------|
| Factor Records | 1,567,488 | 1,567,488 | 1.00 |
| Unique Tickers | 714 | 714 | 1.00 |
| Unique Dates | 2384 | 2384 | 1.00 |

## 🎯 Key Findings

✅ **Database contains substantial data**
- Factor scores: 1,567,488 records
- Price data: 2,319,796 records
- Previous analysis was limited by sampling filters

## 📋 Recommendations

1. **Re-run Discrepancy Analysis**
   - Remove sampling filters
   - Use full database content
   - Compare with pickle data properly
2. **Investigate Data Pipeline**
   - Check if database is production or test
   - Verify data loading processes
   - Ensure data completeness
3. **Update Backtesting Framework**
   - Use full database content
   - Apply proper filters
   - Validate data quality
