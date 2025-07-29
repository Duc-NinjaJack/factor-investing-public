# Robustness Testing Report
Generated: 2025-07-27 08:34

## Executive Summary
- Strategy Grade: HIGHLY ROBUST
- Baseline Sharpe: 1.52
- Most Sensitive Parameter: Transaction Costs
- Average Sharpe Sensitivity: 0.157

## Detailed Results
|    | Parameter             | Baseline Value   |   Sharpe Min |   Sharpe Max |   Sharpe Range | Return Min   | Return Max   | Critical Level   |
|---:|:----------------------|:-----------------|-------------:|-------------:|---------------:|:-------------|:-------------|:-----------------|
|  0 | Transaction Costs     | 30               |      0.97189 |      1.52353 |       0.551639 | 12.8%        | 19.9%        | Yes              |
|  1 | Rebalancing Frequency | M                |      1.40406 |      1.52353 |       0.119464 | 18.7%        | 19.9%        | No               |
|  2 | Selection Percentile  | 0.2              |      1.43202 |      1.54371 |       0.111688 | 18.5%        | 20.4%        | No               |
|  3 | Sector Constraint     | 0.4              |      1.52353 |      1.52353 |       0        | 19.9%        | 19.9%        | No               |
|  4 | Position Limit        | 0.05             |      1.52353 |      1.52353 |       0        | 19.9%        | 19.9%        | No               |