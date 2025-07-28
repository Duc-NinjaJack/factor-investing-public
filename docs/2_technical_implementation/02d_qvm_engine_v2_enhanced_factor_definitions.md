# QVM Engine v2 Enhanced - Complete Factor Definitions

**Document Purpose:** Comprehensive mathematical specification of all factors calculated by the Enhanced QVM Engine v2 for audit validation.

**Engine Status:** Post-Critical-Fixes (July 23, 2025) - Contains corrected banking cost-income calculation and institutional "normalize-then-average" methodology.

---

## 1. QUALITY FACTORS

The Enhanced Engine v2 implements sector-specific quality frameworks using **institutional "normalize-then-average" methodology** where each base metric is individually converted to sector-neutral z-score before weighted combination.

### 1.1 Banking Sector Quality Factors

**Sector Weights:** ROAE=40%, ROAA=25%, NIM=20%, Cost_Income_Ratio=15%

#### 1.1.1 Return on Average Equity (ROAE)

$$ROAE_{\text{banking}} = \frac{\text{NetProfit}_{\text{TTM}}}{\text{AvgTotalEquity}}$$

- **Data Source:** `intermediary_calculations_banking_cleaned`
- **Expected Range:** 5-30% (per factor_metadata.yml)

#### 1.1.2 Return on Average Assets (ROAA)  

$$ROAA_{\text{banking}} = \frac{\text{NetProfit}_{\text{TTM}}}{\text{AvgTotalAssets}}$$

- **Expected Range:** 0.5-3.0% (per factor_metadata.yml)

#### 1.1.3 Net Interest Margin (NIM)

$$NIM_{\text{banking}} = \begin{cases}
\text{NIM} & \text{if available directly} \\
\frac{\text{NetInterestIncome}_{\text{TTM}}}{\text{AvgInterestEarningAssets}} & \text{if calculated}
\end{cases}$$

- **Expected Range:** 2.0-6.0% (per factor_metadata.yml)

#### 1.1.4 Cost-Income Ratio (CRITICAL FIX APPLIED)

$$\text{Cost\_Income\_Ratio}_{\text{banking}} = \begin{cases}
1 - \text{Cost\_Income\_Ratio} & \text{if available directly} \\
1 - \frac{|\text{OperatingExpenses}_{\text{TTM}}|}{\text{TotalOperatingIncome}_{\text{TTM}}} & \text{if calculated}
\end{cases}$$

- **CRITICAL FIX:** Added `abs()` function to handle Vietnamese negative expense accounting
- **Expected Range:** 20-60% cost ratio → 40-80% efficiency score
- **OCB Example:** `abs(-3,937,305,167,853) / 10,055,388,932,563 = 39.16%` → Efficiency = 60.84%

#### 1.1.5 Banking Quality Composite (Institutional Method)

$$Q_{\text{banking}}^{\text{composite}} = \frac{\sum_{i=1}^{4} w_i \cdot Z_{\text{sector}}(M_i)}{\sum_{i=1}^{4} w_i}$$

Where:
- $Z_{\text{sector}}(M_i) = \frac{M_i - \mu_{\text{sector}}(M_i)}{\sigma_{\text{sector}}(M_i)}$ = Sector-neutral z-score of metric $i$
- $w_i$ = Sector-specific weights: ROAE(0.40), ROAA(0.25), NIM(0.20), Cost_Income(0.15)

### 1.2 Securities Sector Quality Factors

**Sector Weights:** ROAE=50%, BrokerageRatio=30%, NetProfitMargin=20%

#### 1.2.1 Return on Average Equity (ROAE)

$$ROAE_{\text{securities}} = \frac{\text{NetProfit}_{\text{TTM}}}{\text{AvgTotalEquity}}$$

#### 1.2.2 Brokerage Ratio

$$\text{BrokerageRatio}_{\text{securities}} = \begin{cases}
\text{BrokerageRatio} & \text{if available directly} \\
\frac{\text{BrokerageIncome}_{\text{TTM}}}{\text{TotalOperatingRevenue}_{\text{TTM}}} & \text{if calculated}
\end{cases}$$

#### 1.2.3 Net Profit Margin

$$\text{NetProfitMargin}_{\text{securities}} = \frac{\text{NetProfit}_{\text{TTM}}}{\text{TotalOperatingRevenue}_{\text{TTM}}}$$

#### 1.2.4 Securities Quality Composite

$$Q_{\text{securities}}^{\text{composite}} = \frac{\sum_{i=1}^{3} w_i \cdot Z_{\text{sector}}(M_i)}{\sum_{i=1}^{3} w_i}$$

### 1.3 Non-Financial Sector Quality Factors

**Sector Weights:** ROAE=35%, NetProfitMargin=25%, GrossMargin=25%, OperatingMargin=15%

#### 1.3.1 Return on Average Equity (ROAE)

$$ROAE_{\text{non-financial}} = \frac{\text{NetProfit}_{\text{TTM}}}{\text{AvgTotalEquity}}$$

#### 1.3.2 Net Profit Margin

$$\text{NetProfitMargin}_{\text{non-financial}} = \frac{\text{NetProfit}_{\text{TTM}}}{\text{Revenue}_{\text{TTM}}}$$

#### 1.3.3 Gross Margin

$$\text{GrossMargin}_{\text{non-financial}} = \frac{\text{Revenue}_{\text{TTM}} - \text{COGS}_{\text{TTM}}}{\text{Revenue}_{\text{TTM}}}$$

#### 1.3.4 Operating Margin

$$\text{OperatingMargin}_{\text{non-financial}} = \frac{\text{Revenue}_{\text{TTM}} - \text{COGS}_{\text{TTM}} - \text{SellingExpenses}_{\text{TTM}} - \text{AdminExpenses}_{\text{TTM}}}{\text{Revenue}_{\text{TTM}}}$$

#### 1.3.5 Non-Financial Quality Composite

$$Q_{\text{non-financial}}^{\text{composite}} = \frac{\sum_{i=1}^{4} w_i \cdot Z_{\text{sector}}(M_i)}{\sum_{i=1}^{4} w_i}$$

### 1.4 Quality Factor Normalization (INSTITUTIONAL STANDARD)

**Critical Implementation:** "Normalize-then-average" methodology applied uniformly:

$$Z_{\text{sector}}(M_i) = \frac{M_i - \mu_{\text{sector}}(M_i)}{\sigma_{\text{sector}}(M_i)}$$

Where each metric $M_i$ is sector-neutral normalized before weighted combination.

---

## 2. VALUE FACTORS

Enhanced value calculation with sector-specific weights and point-in-time equity methodology.

### 2.1 Value Components

#### 2.1.1 Price-to-Earnings (Inverted)

$$PE_{\text{score}} = \frac{1}{PE_{\text{ratio}}} = \frac{\text{NetProfit}_{\text{TTM}}}{\text{MarketCap}}$$

#### 2.1.2 Price-to-Book (Inverted, CORRECTED)

$$PB_{\text{score}} = \frac{1}{PB_{\text{ratio}}} = \frac{\text{PointInTimeEquity}}{\text{MarketCap}}$$

- **CORRECTION:** Uses point-in-time equity with 45-day reporting lag, not TTM average
- **Fallback:** AvgTotalEquity if point-in-time unavailable

#### 2.1.3 Price-to-Sales (Inverted, Sector-Specific)

$$PS_{\text{score}} = \frac{1}{PS_{\text{ratio}}} = \frac{\text{SectorRevenue}_{\text{TTM}}}{\text{MarketCap}}$$

Where:

$$\text{SectorRevenue}_{\text{TTM}} = \begin{cases}
\text{TotalOperatingIncome}_{\text{TTM}} & \text{if Banking} \\
\text{TotalOperatingRevenue}_{\text{TTM}} & \text{if Securities} \\
\text{Revenue}_{\text{TTM}} & \text{if Non-Financial}
\end{cases}$$

#### 2.1.4 Enhanced EV/EBITDA (Inverted)

$$EVEBITDA_{\text{score}} = \frac{1}{\frac{EV}{EBITDA}} = \frac{\text{EBITDA}_{\text{TTM}}}{EV}$$

Where Enterprise Value:

$$EV = \text{MarketCap} + \text{TotalDebt} - \text{CashAndEquivalents}$$

- **Excluded Sectors:** Banking, Securities, Insurance (weight = 0)
- **Point-in-Time Logic:** Uses 45-day reporting lag for balance sheet items

### 2.2 Sector-Specific Value Weights

$$V_{\text{composite}} = w_{PE} \cdot PE_{\text{score}} + w_{PB} \cdot PB_{\text{score}} + w_{PS} \cdot PS_{\text{score}} + w_{EV} \cdot EVEBITDA_{\text{score}}$$

**Banking:** $w_{PE}=0.60, w_{PB}=0.40, w_{PS}=0.00, w_{EV}=0.00$

**Securities:** $w_{PE}=0.50, w_{PB}=0.30, w_{PS}=0.20, w_{EV}=0.00$

**Insurance:** $w_{PE}=0.50, w_{PB}=0.50, w_{PS}=0.00, w_{EV}=0.00$

**Non-Financial:** $w_{PE}=0.40, w_{PB}=0.30, w_{PS}=0.20, w_{EV}=0.10$

---

## 3. MOMENTUM FACTORS

Multi-timeframe momentum with skip-1-month convention and sophisticated weighting.

### 3.1 Momentum Components

#### 3.1.1 Return Calculation (Fixed Method)

$$R_{t_1,t_2} = \frac{P_{t_2}}{P_{t_1}} - 1$$

Where:
- $P_{t_1}$ = First available adjusted close on or after start date
- $P_{t_2}$ = Last available adjusted close on or before end date
- **Data Source:** `equity_history` table with proper adjusted prices

#### 3.1.2 Multi-Timeframe Returns

$$\begin{align}
R_{1M} &= R_{t-2M, t-1M} \quad \text{(1-month skip applied)} \\
R_{3M} &= R_{t-4M, t-1M} \quad \text{(3-month lookback + 1-month skip)} \\
R_{6M} &= R_{t-7M, t-1M} \quad \text{(6-month lookback + 1-month skip)} \\
R_{12M} &= R_{t-13M, t-1M} \quad \text{(12-month lookback + 1-month skip)}
\end{align}$$

#### 3.1.3 Momentum Composite

$$M_{\text{composite}} = w_{1M} \cdot R_{1M} + w_{3M} \cdot R_{3M} + w_{6M} \cdot R_{6M} + w_{12M} \cdot R_{12M}$$

**Timeframe Weights:** $w_{1M}=0.15, w_{3M}=0.25, w_{6M}=0.30, w_{12M}=0.30$

### 3.2 Momentum Normalization

$$M_{\text{score}} = Z_{\text{sector}}(M_{\text{composite}})$$

**INSTITUTIONAL CORRECTION:** Momentum also uses sector-neutral normalization for consistency.

---

## 4. QVM COMPOSITE METHODOLOGY

### 4.1 Final QVM Score

$$QVM_{\text{score}} = w_Q \cdot Q_{\text{score}} + w_V \cdot V_{\text{score}} + w_M \cdot M_{\text{score}}$$

**Factor Weights:** $w_Q=0.40, w_V=0.30, w_M=0.30$

### 4.2 Sector-Neutral Normalization Applied to All Factors

$$\begin{align}
Q_{\text{score}} &= Z_{\text{sector}}(Q_{\text{composite}}) \\
V_{\text{score}} &= Z_{\text{sector}}(V_{\text{composite}}) \\
M_{\text{score}} &= Z_{\text{sector}}(M_{\text{composite}})
\end{align}$$

**Critical Implementation:** All factor scores are sector-neutral z-scores before final combination.

---

## 5. DATA SOURCES AND TIMING

### 5.1 Primary Data Tables
- **Banking:** `intermediary_calculations_banking_cleaned`
- **Securities:** `intermediary_calculations_securities_cleaned`
- **Non-Financial:** `intermediary_calculations_enhanced`
- **Market Data:** `vcsc_daily_data_complete`
- **Price Returns:** `equity_history`
- **Point-in-Time Balance Sheet:** `v_comprehensive_fundamental_items`

### 5.2 Point-in-Time Logic
**Reporting Lag:** 45 days after quarter end

$$\text{Available Quarter} = \max\{q : \text{QuarterEnd}_q + 45 \text{ days} \leq \text{AnalysisDate}\}$$

### 5.3 Sector Mapping Correction
**Banking Fix:** `sector = 'Banks'` → `sector = 'Banking'` applied in `get_sector_mapping()`

---

## 6. VALIDATION RANGES (From factor_metadata.yml)

### 6.1 Banking Ratios
- **ROAE:** 5-30%
- **ROAA:** 0.5-3.0%
- **NIM:** 2.0-6.0%
- **Cost-Income Ratio:** 25-60% → Efficiency Score: 40-75%

### 6.2 Critical Validation Cases
- **OCB Cost-Income:** Should produce ~39.16% cost ratio (60.84% efficiency)
- **All Banking Cost-Income:** Must be <100% after `abs()` fix
- **Quality Composites:** Must be weighted averages of individual z-scores, not raw percentages

---

## 7. CRITICAL FIXES IMPLEMENTED (July 23, 2025)

### 7.1 Banking Cost-Income Calculation
**BEFORE:** `cost_ratio = row['OperatingExpenses_TTM'] / row['TotalOperatingIncome_TTM']`

**AFTER:** `cost_ratio = abs(row['OperatingExpenses_TTM']) / row['TotalOperatingIncome_TTM']`

### 7.2 Quality Composite Methodology
**BEFORE:** "Average-then-normalize" - Average raw percentages, then z-score

**AFTER:** "Normalize-then-average" - Individual z-scores, then weighted average

**Mathematical Difference:**

$$\text{OLD: } Z_{\text{sector}}\left(\frac{\sum_{i} w_i M_i}{\sum_{i} w_i}\right) \quad \text{vs} \quad \text{NEW: } \frac{\sum_{i} w_i Z_{\text{sector}}(M_i)}{\sum_{i} w_i}$$

---

## 8. AUDIT VALIDATION FRAMEWORK

**AUDIT PRIORITY:** Validate that corrected engine produces:

1. **OCB Cost-Income Ratio ≈ 39.16%** (not 139.16%)
2. **All Quality Composites** are weighted z-score averages
3. **Sector-Neutral Normalization** applied consistently across all factors
4. **Point-in-Time Equity** used for P/B ratios

**Success Criteria:**
- Banking cost-income ratios < 100% for all banks
- Quality composites follow institutional methodology: $\frac{\sum w_i Z(M_i)}{\sum w_i}$
- Manual calculations match engine output within 0.01 tolerance
- Audit notebook validation passes 100%

This document serves as the **mathematical specification** for comprehensive audit validation of the Enhanced QVM Engine v2.