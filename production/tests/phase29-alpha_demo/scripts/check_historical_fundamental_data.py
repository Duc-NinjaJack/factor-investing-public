#!/usr/bin/env python3

import sys
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add Project Root to Python Path
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

print("✅ Historical fundamental data check script initialized")

def check_historical_fundamental_data():
    """Check what fundamental data is available for historical dates."""
    try:
        db_manager = get_database_manager()
        engine = db_manager.get_engine()
        
        with engine.connect() as conn:
            print("📊 Historical Fundamental Data Availability Check")
            print("=" * 80)
            
            # 1. Check fundamental_values table (raw data)
            print("\n1. 📋 fundamental_values table (Raw Financial Data):")
            result = conn.execute(text("""
                SELECT 
                    MIN(year) as earliest_year,
                    MAX(year) as latest_year,
                    COUNT(DISTINCT year) as unique_years,
                    COUNT(DISTINCT ticker) as unique_tickers,
                    COUNT(*) as total_records
                FROM fundamental_values
                WHERE value IS NOT NULL
            """))
            
            row = result.fetchone()
            print(f"   📅 Date Range: {row[0]} to {row[1]} ({row[2]} years)")
            print(f"   📊 Coverage: {row[3]} tickers, {row[4]:,} records")
            
            # Check recent years
            result = conn.execute(text("""
                SELECT 
                    year,
                    COUNT(DISTINCT ticker) as tickers,
                    COUNT(*) as records
                FROM fundamental_values
                WHERE year >= 2016 AND value IS NOT NULL
                GROUP BY year
                ORDER BY year DESC
            """))
            
            print(f"   📈 Recent Years Coverage:")
            for row in result:
                print(f"      {row[0]}: {row[1]} tickers, {row[2]:,} records")
            
            # 2. Check intermediary_calculations_enhanced table
            print("\n2. 📊 intermediary_calculations_enhanced table (Pre-calculated):")
            result = conn.execute(text("""
                SELECT 
                    MIN(year) as earliest_year,
                    MAX(year) as latest_year,
                    COUNT(DISTINCT year) as unique_years,
                    COUNT(DISTINCT ticker) as unique_tickers,
                    COUNT(*) as total_records
                FROM intermediary_calculations_enhanced
                WHERE Revenue_TTM IS NOT NULL
            """))
            
            row = result.fetchone()
            print(f"   📅 Date Range: {row[0]} to {row[1]} ({row[2]} years)")
            print(f"   📊 Coverage: {row[3]} tickers, {row[4]:,} records")
            
            # 3. Check factor_scores_qvm table (existing factor scores)
            print("\n3. 🎯 factor_scores_qvm table (Existing Factor Scores):")
            result = conn.execute(text("""
                SELECT 
                    strategy_version,
                    MIN(date) as earliest_date,
                    MAX(date) as latest_date,
                    COUNT(DISTINCT date) as unique_dates,
                    COUNT(DISTINCT ticker) as unique_tickers,
                    COUNT(*) as total_records
                FROM factor_scores_qvm
                WHERE date >= '2016-01-01'
                GROUP BY strategy_version
                ORDER BY strategy_version
            """))
            
            print(f"   📈 Available Strategy Versions:")
            for row in result:
                print(f"      {row[0]}:")
                print(f"         Date Range: {row[1]} to {row[2]} ({row[3]} dates)")
                print(f"         Coverage: {row[4]} tickers, {row[5]:,} records")
            
            # 4. Check specific data for 2016-2025 period
            print("\n4. 🔍 Specific Data Check for 2016-2025:")
            
            # Check fundamental_values for 2016-2025
            result = conn.execute(text("""
                SELECT 
                    year,
                    COUNT(DISTINCT ticker) as tickers,
                    COUNT(*) as records
                FROM fundamental_values
                WHERE year BETWEEN 2016 AND 2025
                AND value IS NOT NULL
                GROUP BY year
                ORDER BY year
            """))
            
            print(f"   📋 fundamental_values (2016-2025):")
            for row in result:
                print(f"      {row[0]}: {row[1]} tickers, {row[2]:,} records")
            
            # Check intermediary_calculations for 2016-2025
            result = conn.execute(text("""
                SELECT 
                    year,
                    COUNT(DISTINCT ticker) as tickers,
                    COUNT(*) as records
                FROM intermediary_calculations_enhanced
                WHERE year BETWEEN 2016 AND 2025
                AND Revenue_TTM IS NOT NULL
                GROUP BY year
                ORDER BY year
            """))
            
            print(f"   📊 intermediary_calculations_enhanced (2016-2025):")
            for row in result:
                print(f"      {row[0]}: {row[1]} tickers, {row[2]:,} records")
            
            # 5. Check what's available for specific dates
            print("\n5. 🎯 Sample Date Check (2016-2025):")
            sample_dates = ['2016-06-30', '2018-06-30', '2020-06-30', '2022-06-30', '2024-06-30']
            
            for date in sample_dates:
                year = int(date[:4])
                quarter = (int(date[5:7]) - 1) // 3 + 1
                
                # Check fundamental_values
                result = conn.execute(text("""
                    SELECT COUNT(DISTINCT ticker) as tickers
                    FROM fundamental_values
                    WHERE year = :year AND quarter = :quarter
                    AND value IS NOT NULL
                """), {'year': year, 'quarter': quarter})
                
                fv_count = result.fetchone()[0]
                
                # Check intermediary_calculations
                result = conn.execute(text("""
                    SELECT COUNT(DISTINCT ticker) as tickers
                    FROM intermediary_calculations_enhanced
                    WHERE year = :year AND quarter = :quarter
                    AND Revenue_TTM IS NOT NULL
                """), {'year': year, 'quarter': quarter})
                
                ic_count = result.fetchone()[0]
                
                print(f"   {date} (Q{quarter} {year}):")
                print(f"      fundamental_values: {fv_count} tickers")
                print(f"      intermediary_calculations: {ic_count} tickers")
            
            # 6. Check if we can use existing factor scores
            print("\n6. ✅ Recommendation for Backtest:")
            
            # Check if we have factor scores for the period
            result = conn.execute(text("""
                SELECT 
                    COUNT(DISTINCT date) as dates,
                    COUNT(DISTINCT ticker) as tickers,
                    MIN(date) as start_date,
                    MAX(date) as end_date
                FROM factor_scores_qvm
                WHERE date BETWEEN '2016-01-01' AND '2025-12-31'
                AND strategy_version = 'qvm_v2.0_enhanced'
            """))
            
            row = result.fetchone()
            if row[0] > 0:
                print(f"   ✅ EXISTING FACTOR SCORES AVAILABLE:")
                print(f"      Date Range: {row[2]} to {row[3]}")
                print(f"      Coverage: {row[0]} dates, {row[1]} tickers")
                print(f"      Strategy: qvm_v2.0_enhanced")
                print(f"      RECOMMENDATION: Use existing factor scores for backtest")
            else:
                print(f"   ❌ No existing factor scores for 2016-2025")
                print(f"      Need to calculate factors from fundamental data")
                
    except Exception as e:
        print(f"❌ Error checking historical fundamental data: {e}")

if __name__ == "__main__":
    check_historical_fundamental_data()
