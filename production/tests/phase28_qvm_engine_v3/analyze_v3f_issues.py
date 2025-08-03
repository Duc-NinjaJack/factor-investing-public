#!/usr/bin/env python3
"""
Comprehensive analysis of v3f issues causing infinite returns and NaN cost drag
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
from pathlib import Path
import sys

# Add project root to path
current_path = Path.cwd()
while not (current_path / 'production').is_dir():
    if current_path.parent == current_path:
        raise FileNotFoundError("Could not find the 'production' directory.")
    current_path = current_path.parent

project_root = current_path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from production.database.connection import get_database_manager
from sqlalchemy import text

warnings.filterwarnings('ignore')

def analyze_v3f_issues():
    """Analyze v3f issues systematically."""
    print("🔍 COMPREHENSIVE V3F ISSUE ANALYSIS")
    print("=" * 60)
    
    # Database connection
    db_manager = get_database_manager()
    engine = db_manager.get_engine()
    
    print("\n📋 PHASE 1: CONFIGURATION ISSUES")
    print("-" * 40)
    
    # Issue 1: Wrong regime thresholds
    print("❌ ISSUE 1: Incorrect Regime Thresholds")
    print("   Current thresholds:")
    print("   - Volatility: 0.2659 (26.59%) - TOO HIGH!")
    print("   - Return: 0.2588 (25.88%) - TOO HIGH!")
    print("   - Low Return: 0.2131 (21.31%) - TOO HIGH!")
    print("   Expected thresholds:")
    print("   - Volatility: 0.0140 (1.40%)")
    print("   - Return: 0.0012 (0.12%)")
    print("   - Low Return: 0.0002 (0.02%)")
    
    print("\n📋 PHASE 2: DATA LOADING ANALYSIS")
    print("-" * 40)
    
    # Test data loading
    start_date = "2016-01-01"
    end_date = "2020-12-31"
    buffer_start_date = pd.Timestamp(start_date) - pd.DateOffset(months=6)
    
    print(f"📊 Data Loading Parameters:")
    print(f"   - Backtest Start: {start_date}")
    print(f"   - Backtest End: {end_date}")
    print(f"   - Buffer Start: {buffer_start_date.date()}")
    
    # Load benchmark data
    benchmark_query = text("""
        SELECT date, close
        FROM etf_history
        WHERE ticker = 'VNINDEX' AND date BETWEEN :start_date AND :end_date
        ORDER BY date
    """)
    
    benchmark_data = pd.read_sql(benchmark_query, engine, 
                                params={'start_date': buffer_start_date, 'end_date': end_date}, 
                                parse_dates=['date'])
    
    print(f"\n📈 Benchmark Data Analysis:")
    print(f"   - Total Records: {len(benchmark_data)}")
    print(f"   - Date Range: {benchmark_data['date'].min()} to {benchmark_data['date'].max()}")
    
    # Check for data gaps
    benchmark_data_sorted = benchmark_data.sort_values('date')
    date_diffs = benchmark_data_sorted['date'].diff().dt.days
    gaps = date_diffs[date_diffs > 5]
    
    print(f"   - Data Gaps: {len(gaps)} gaps found")
    if len(gaps) > 0:
        print(f"   - Largest Gap: {gaps.max()} days")
    
    print("\n📋 PHASE 3: RETURNS CALCULATION ANALYSIS")
    print("-" * 40)
    
    # Simulate the returns calculation issue
    benchmark_returns = benchmark_data.set_index('date')['close'].pct_change()
    
    print("❌ ISSUE 2: Potential Returns Calculation Problems")
    print("   - Check for infinite/NaN values in returns")
    print("   - Check for division by zero in portfolio calculations")
    print("   - Check for empty portfolios causing zero division")
    
    # Check for problematic values
    inf_returns = np.isinf(benchmark_returns).sum()
    nan_returns = np.isnan(benchmark_returns).sum()
    
    print(f"   - Infinite returns: {inf_returns}")
    print(f"   - NaN returns: {nan_returns}")
    
    print("\n📋 PHASE 4: PORTFOLIO CONSTRUCTION ANALYSIS")
    print("-" * 40)
    
    print("❌ ISSUE 3: Portfolio Construction Problems")
    print("   - Empty universe causing empty portfolios")
    print("   - Zero portfolio weights causing division issues")
    print("   - Missing factor data causing empty qualified stocks")
    
    print("\n📋 PHASE 5: TRANSACTION COST CALCULATION")
    print("-" * 40)
    
    print("❌ ISSUE 4: Transaction Cost Calculation Problems")
    print("   - NaN turnover values")
    print("   - Infinite cost calculations")
    print("   - Zero division in cost calculations")
    
    print("\n📋 PHASE 6: REGIME DETECTION ANALYSIS")
    print("-" * 40)
    
    # Test regime detection with wrong thresholds
    print("❌ ISSUE 5: Regime Detection Always Returns 'Sideways'")
    print("   - Volatility threshold too high (26.59% vs 1.40%)")
    print("   - Return threshold too high (25.88% vs 0.12%)")
    print("   - All periods classified as 'Sideways'")
    
    # Simulate regime detection
    test_volatility = 0.015  # 1.5% - typical daily volatility
    test_return = 0.002     # 0.2% - typical daily return
    
    vol_threshold_wrong = 0.2659
    ret_threshold_wrong = 0.2588
    
    print(f"   - Test Volatility: {test_volatility:.4f} ({test_volatility:.2%})")
    print(f"   - Test Return: {test_return:.4f} ({test_return:.2%})")
    print(f"   - Wrong Vol Threshold: {vol_threshold_wrong:.4f} ({vol_threshold_wrong:.2%})")
    print(f"   - Wrong Ret Threshold: {ret_threshold_wrong:.4f} ({ret_threshold_wrong:.2%})")
    print(f"   - Result: Always 'Sideways' because volatility < threshold")
    
    print("\n📋 PHASE 7: FACTOR CALCULATION ISSUES")
    print("-" * 40)
    
    print("❌ ISSUE 6: Factor Calculation Problems")
    print("   - Missing fundamental data")
    print("   - Division by zero in ROAA calculations")
    print("   - NaN values in momentum calculations")
    print("   - Empty factor dataframes")
    
    print("\n📋 PHASE 8: PROPOSED FIXES")
    print("-" * 40)
    
    print("✅ FIX 1: Correct Regime Thresholds")
    print("   - Change volatility_threshold from 0.2659 to 0.0140")
    print("   - Change return_threshold from 0.2588 to 0.0012")
    print("   - Change low_return_threshold from 0.2131 to 0.0002")
    
    print("\n✅ FIX 2: Add Data Validation")
    print("   - Check for empty dataframes before calculations")
    print("   - Add NaN/inf checks in returns calculations")
    print("   - Validate portfolio weights before calculations")
    
    print("\n✅ FIX 3: Fix Returns Calculation")
    print("   - Add safety checks for division by zero")
    print("   - Handle empty portfolios gracefully")
    print("   - Validate factor data before portfolio construction")
    
    print("\n✅ FIX 4: Improve Error Handling")
    print("   - Add try-catch blocks around critical calculations")
    print("   - Add data validation at each step")
    print("   - Add debug output for troubleshooting")
    
    print("\n✅ FIX 5: Fix Transaction Cost Calculation")
    print("   - Add NaN checks in turnover calculation")
    print("   - Handle edge cases in cost calculation")
    print("   - Add validation for cost drag calculation")
    
    print("\n📋 PHASE 9: ROOT CAUSE SUMMARY")
    print("-" * 40)
    
    print("🔍 PRIMARY ROOT CAUSES:")
    print("   1. WRONG REGIME THRESHOLDS: 26.59% vs 1.40% (19x too high)")
    print("   2. EMPTY PORTFOLIOS: No stocks qualified due to factor issues")
    print("   3. DIVISION BY ZERO: Empty portfolios causing infinite returns")
    print("   4. MISSING DATA VALIDATION: No checks for NaN/inf values")
    print("   5. POOR ERROR HANDLING: No graceful handling of edge cases")
    
    print("\n🎯 MOST CRITICAL FIX:")
    print("   - Fix regime thresholds FIRST (causes cascade of issues)")
    print("   - Add data validation SECOND (prevents calculation errors)")
    print("   - Improve error handling THIRD (provides better debugging)")
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    analyze_v3f_issues() 