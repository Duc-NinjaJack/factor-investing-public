# **Part 2: System Architecture & Database Schema**

**Document Name:** `02b_temporal_logic_and_data_availability.md`
**Version:** 1.0
**Date:** July 22, 2025
**Status:** ✅ **ACTIVE - SINGLE SOURCE OF TRUTH**
**Owner:** Duc Nguyen, Principal Quantitative Strategist

## **1. Executive Summary**

This document is the firm's constitution for **point-in-time (PIT) correctness**. Its sole purpose is to define the rigorous temporal rules that govern all data access, ensuring that no analysis or simulation can be contaminated by look-ahead bias. It specifies the precise relationship between calendar dates, financial reporting periods, data publication lags, and factor calculation dates. Adherence to this document is mandatory and forms the bedrock of our research integrity.

## **2. The Core Principle: The 45-Day Lag**

The central assumption of our temporal model is the 45-day reporting lag for Vietnamese companies, as mandated by regulations. This means that the financial data for a quarter is not considered publicly available until 45 calendar days after the quarter's end.

*   **Q1 (ends Mar 31):** Data is available from **May 15** onwards.
*   **Q2 (ends Jun 30):** Data is available from **August 14** onwards.
*   **Q3 (ends Sep 30):** Data is available from **November 14** onwards.
*   **Q4 (ends Dec 31):** Data is available from **February 14** of the next year onwards.

Any function retrieving fundamental data for a given `analysis_date` must only return data that was published on or before that date.

## **3. The Data Availability Timeline: A Practical Example**

Consider the factor calculations for **May 31, 2024**:

| Date | Event | Data Available for Calculation |
| :--- | :--- | :--- |
| **Mar 31, 2024** | Q1 2024 Ends | N/A |
| **Apr 1 - May 14** | Reporting Period | The latest available data is still from **Q4 2023**. |
| **May 15, 2024** | **Publication Date** | Q1 2024 financial data is now officially available. |
| **May 31, 2024** | **Analysis Date** | We can now use Q1 2024 data for our calculations. |

Therefore, for a factor calculation on `analysis_date = 2024-05-31`:
*   **Quality Factor:** Uses TTM data ending in Q1 2024.
*   **Value Factor:** Uses market cap from `2024-05-31` and fundamental denominators (e.g., `NetProfit_TTM`) from the period ending Q1 2024.
*   **Momentum Factor:** Uses adjusted prices up to `~2024-04-30` (respecting the skip-1-month rule).

## **4. Reference Implementation: Data Retrieval Logic**

The following Python logic serves as the reference implementation for determining which quarter's fundamental data is available for a given `analysis_date`. All production code must replicate this logic.

```python
from datetime import date, timedelta

def get_latest_available_quarter(analysis_date: date) -> tuple[int, int]:
    """
    Determines the most recent quarter of fundamental data available
    for a given analysis_date, respecting the 45-day lag.

    Args:
        analysis_date: The date of the analysis (e.g., portfolio rebalance date).

    Returns:
        A tuple of (year, quarter).
    """
    year = analysis_date.year
    
    # Check for Q3 data (available Nov 14)
    if analysis_date >= date(year, 11, 14):
        return (year, 3)
        
    # Check for Q2 data (available Aug 14)
    if analysis_date >= date(year, 8, 14):
        return (year, 2)
        
    # Check for Q1 data (available May 15)
    if analysis_date >= date(year, 5, 15):
        return (year, 1)
        
    # Check for previous year's Q4 data (available Feb 14)
    if analysis_date >= date(year, 2, 14):
        return (year - 1, 4)
        
    # If before Feb 14, only previous year's Q3 is available
    return (year - 1, 3)

# --- Example Usage ---
# On the last day of April 2024, only Q4 2023 data is available.
assert get_latest_available_quarter(date(2024, 4, 30)) == (2023, 4)

# On May 15, 2024, Q1 2024 data becomes available.
assert get_latest_available_quarter(date(2024, 5, 15)) == (2024, 1)
```

## **5. Critical Rules & Best Practices**

1.  **All Data Access Must Be PIT:** Every data query for backtesting or factor generation must be filtered by the `analysis_date`.
2.  **Fundamental Data:** Use the `get_latest_available_quarter` logic to determine the correct `(year, quarter)` key for querying `intermediary_calculations_*` tables.
3.  **Market Data:** Daily data (prices, market cap) from `equity_history` and `vcsc_daily_data_complete` can be used up to and including the `analysis_date`.
4.  **Momentum Data:** Price data used for return calculations must respect the "skip-1-month" convention (i.e., use prices from T-22 trading days and earlier).
5.  **No Future Peeking:** There are zero exceptions to these rules. A single violation invalidates the entire backtest.

---
---

### **IMPLEMENTATION NOTES**

1.  **Integrate into Documentation:** This document, `02b_temporal_logic_and_data_availability.md`, must be added to your master documentation suite immediately.
2.  **Mandatory Reading:** All current and future members of the quantitative development team must read and formally acknowledge their understanding of this document.
3.  **Code Audit:** Your data loading and backtesting classes must be audited to ensure they explicitly implement the `get_latest_available_quarter` logic. This is not optional.

### **RISK CONSIDERATIONS**

*   **Primary Risk Mitigated:** This document closes our most significant operational risk: inadvertent look-ahead bias leading to overly optimistic and entirely invalid backtest results.
*   **Remaining Risk:** Human error in implementation. The risk is minimized by having a single, authoritative document and a clear reference implementation in Python to test against.



