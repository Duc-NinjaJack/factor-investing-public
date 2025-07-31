# Large Universe Momentum Validation Report (200 Stocks)

**Date**: July 30, 2025  
**Author**: Factor Investing Team  
**Version**: 1.0  

## Executive Summary

This report presents the results of comprehensive momentum factor validation using a larger universe of 200 stocks, including detailed data quality analysis to detect potential simulation or synthetic data issues. The analysis reveals important insights about data quality, factor performance across different market cap ranges, and regime-specific behavior.

## Key Findings

### 📊 **Data Quality Assessment**
- **✅ Real Market Data Confirmed**: No evidence of simulation or synthetic data
- **⚠️ Data Quality Issues**: 12-14 quality concerns detected across periods
- **📅 Data Freshness**: 3 days old (excellent)
- **🏢 Universe Diversity**: 22-24 unique sectors represented

### 📈 **Performance Results**
- **2016-2020**: Mixed performance with some positive IC values
- **2021-2025**: Generally poor performance across all horizons
- **Market Cap Range**: 1.4T - 530T VND (highly diverse)

### 🔍 **Data Quality Issues Identified**
1. **Low variation in fundamental data** (year/quarter fields)
2. **Few unique values** in certain fundamental metrics
3. **Sector representation** issues (some sectors with only 1-3 stocks)

## Methodology

### Universe Construction
```sql
-- 200 stocks by market cap with 5B VND minimum
SELECT 
    eh.ticker,
    eh.market_cap,
    mi.sector
FROM (
    SELECT 
        ticker,
        market_cap,
        ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) as rn
    FROM equity_history_with_market_cap
    WHERE date <= '{analysis_date}'
      AND market_cap IS NOT NULL
      AND market_cap >= 5000000000
) eh
LEFT JOIN master_info mi ON eh.ticker = mi.ticker
WHERE eh.rn = 1
ORDER BY eh.market_cap DESC
LIMIT 200
```

### Data Quality Checks
1. **Price Data Completeness**: Check for missing/zero prices
2. **Fundamental Data Variation**: Detect suspicious uniformity
3. **Correlation Analysis**: Identify potential simulation patterns
4. **Data Freshness**: Verify data recency

## Detailed Results

### Regime 1: 2016-2020 (Pre-COVID)

| Horizon | Mean IC | T-Stat | Hit Rate | Quality Issues | Gates Passed |
|---------|---------|--------|----------|----------------|--------------|
| 1M      | 0.0085  | 0.3009 | 66.7%    | 14            | 1/3         |
| 3M      | -0.0126 | -0.3632| 46.7%    | 14            | 0/3         |
| 6M      | 0.0033  | 0.0905 | 71.4%    | 14            | 1/3         |
| 12M     | 0.0306  | 0.8675 | 66.7%    | 14            | 2/3         |

**Characteristics:**
- **Best Performance**: 12M horizon (Mean IC = 0.0306)
- **Hit Rate**: Generally good (46.7% - 71.4%)
- **Quality Issues**: 14 issues detected (consistent)

### Regime 2: 2021-2025 (Post-COVID)

| Horizon | Mean IC | T-Stat | Hit Rate | Quality Issues | Gates Passed |
|---------|---------|--------|----------|----------------|--------------|
| 1M      | -0.0509 | -1.4768| 35.7%    | 12            | 0/3         |
| 3M      | -0.0115 | -0.2731| 38.5%    | 12            | 0/3         |
| 6M      | -0.0476 | -0.9501| 41.7%    | 12            | 0/3         |
| 12M     | -0.0950 | -1.9300| 20.0%    | 12            | 0/3         |

**Characteristics:**
- **Poor Performance**: All horizons show negative IC values
- **Hit Rate**: Very low (20.0% - 41.7%)
- **Quality Issues**: 12 issues detected (slightly better than 2016-2020)

## Data Quality Analysis

### ✅ **Real Market Data Indicators**

1. **Price Data Quality**:
   - ✅ 6,000+ price records per analysis period
   - ✅ No zero prices detected
   - ✅ Natural price variation observed

2. **Market Cap Distribution**:
   - ✅ Wide range: 1.4T - 530T VND
   - ✅ Natural distribution (not uniform)
   - ✅ Realistic market cap values

3. **Sector Diversity**:
   - ✅ 22-24 unique sectors represented
   - ✅ Realistic sector distribution
   - ✅ No artificial sector patterns

4. **Data Freshness**:
   - ✅ Data only 3 days old
   - ✅ Regular updates confirmed
   - ✅ No stale data issues

### ⚠️ **Data Quality Concerns**

1. **Fundamental Data Issues**:
   ```
   ⚠️ Low variation in year: std=0.00
   ⚠️ Few unique values in year: 1/48
   ⚠️ Low variation in quarter: std=0.00
   ⚠️ Few unique values in quarter: 1/48
   ```

2. **Sector Representation**:
   - Some sectors have only 1-3 stocks
   - May affect sector-neutral normalization
   - Cross-sectional fallback used in many cases

3. **Data Completeness**:
   - Fundamental data available for ~48-192 stocks
   - Price data more complete than fundamental data
   - Some stocks missing fundamental metrics

## Market Cap Analysis

### Universe Characteristics
- **Total Stocks**: 200 per analysis period
- **Market Cap Range**: 1.4T - 530T VND
- **Average Market Cap**: ~50T VND
- **Median Market Cap**: ~15T VND

### Sector Distribution
- **Banking**: ~15-20 stocks
- **Securities**: ~10-15 stocks
- **Non-Financial**: ~165-175 stocks
- **Other Sectors**: 1-10 stocks each

## Performance Analysis

### Regime Comparison

**2016-2020 vs 2021-2025**:
- **IC Performance**: 2016-2020 generally better
- **Hit Rate**: 2016-2020 significantly higher
- **Quality Issues**: Similar levels (12-14 issues)

### Horizon Analysis

**Best Performing Horizons**:
1. **2016-2020 12M**: Mean IC = 0.0306 (2/3 gates passed)
2. **2016-2020 6M**: Mean IC = 0.0033 (1/3 gates passed)
3. **2016-2020 1M**: Mean IC = 0.0085 (1/3 gates passed)

**Worst Performing Horizons**:
1. **2021-2025 12M**: Mean IC = -0.0950 (0/3 gates passed)
2. **2021-2025 1M**: Mean IC = -0.0509 (0/3 gates passed)
3. **2021-2025 6M**: Mean IC = -0.0476 (0/3 gates passed)

## Investment Implications

### 1. **Data Quality Confidence**
- ✅ **Real market data confirmed**
- ✅ **No simulation detected**
- ⚠️ **Some fundamental data quality issues**

### 2. **Factor Performance**
- **2016-2020**: Momentum factor shows some promise
- **2021-2025**: Poor performance across all horizons
- **Regime Shift**: Clear deterioration in recent period

### 3. **Universe Size Impact**
- **Larger universe**: More diverse sector representation
- **Market cap diversity**: Wide range of company sizes
- **Liquidity**: Better representation of liquid stocks

### 4. **Quality Gate Compliance**
- **Overall**: Poor compliance (0-2/3 gates passed)
- **Best**: 2016-2020 12M horizon
- **Worst**: 2021-2025 all horizons

## Recommendations

### Immediate Actions
1. **Investigate fundamental data quality issues**
2. **Optimize sector-neutral normalization for small sectors**
3. **Consider alternative momentum specifications**

### Medium-term Actions
1. **Develop regime-aware momentum strategies**
2. **Implement dynamic factor weighting**
3. **Create market cap-specific momentum factors**

### Long-term Actions
1. **Improve fundamental data quality**
2. **Develop sector-specific momentum models**
3. **Create automated quality monitoring**

## Technical Notes

### Data Sources
- **Price Data**: `equity_history_with_market_cap`
- **Fundamental Data**: `intermediary_calculations_*` tables
- **Sector Data**: `master_info` table

### Quality Checks Implemented
1. **Price completeness**: Missing/zero price detection
2. **Fundamental variation**: Statistical uniformity tests
3. **Correlation analysis**: Cross-stock correlation patterns
4. **Data freshness**: Timestamp validation

### Performance Metrics
- **Information Coefficient**: Spearman correlation
- **T-statistic**: Statistical significance
- **Hit Rate**: Percentage of positive IC values
- **Quality Gates**: Mean IC > 0.02, T-stat > 2.0, Hit Rate > 55%

## Conclusion

The large universe validation confirms that we are working with real market data, not simulation. However, the momentum factor shows poor performance in the recent period (2021-2025), with some promise in the earlier period (2016-2020). Data quality issues in fundamental data should be addressed to improve factor performance.

**Key Takeaway**: While the data is real, the momentum factor requires significant optimization to meet quality gates, particularly for the post-COVID period.

---

**Contact**: Factor Investing Team  
**Last Updated**: July 30, 2025  
**Next Review**: August 30, 2025 