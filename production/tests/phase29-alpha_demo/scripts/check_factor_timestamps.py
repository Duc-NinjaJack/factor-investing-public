#!/usr/bin/env python3

import sys
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

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

print("✅ Factor timestamp check script initialized")

def check_factor_timestamps():
    """Check factor generation timestamps to see if factors are stale."""
    try:
        db_manager = get_database_manager()
        engine = db_manager.get_engine()
        
        with engine.connect() as conn:
            # Check latest factor generation timestamps
            result = conn.execute(text("""
                SELECT 
                    strategy_version,
                    COUNT(DISTINCT date) as unique_dates,
                    COUNT(DISTINCT ticker) as unique_tickers,
                    COUNT(*) as total_records,
                    MIN(date) as earliest_date,
                    MAX(date) as latest_date,
                    MAX(calculation_timestamp) as last_generated,
                    MIN(calculation_timestamp) as first_generated
                FROM factor_scores_qvm
                GROUP BY strategy_version
                ORDER BY last_generated DESC
            """))
            
            print("📊 Factor Generation Summary by Version:")
            print("=" * 80)
            
            for row in result:
                print(f"Strategy Version: {row[0]}")
                print(f"  Unique Dates: {row[1]}")
                print(f"  Unique Tickers: {row[2]}")
                print(f"  Total Records: {row[3]:,}")
                print(f"  Date Range: {row[4]} to {row[5]}")
                print(f"  First Generated: {row[7]}")
                print(f"  Last Generated: {row[6]}")
                print()
            
            # Check specific dates for qvm_v2.0_enhanced
            result = conn.execute(text("""
                SELECT 
                    date,
                    COUNT(DISTINCT ticker) as tickers,
                    MAX(calculation_timestamp) as generated_at
                FROM factor_scores_qvm
                WHERE strategy_version = 'qvm_v2.0_enhanced'
                AND date IN ('2025-05-30', '2025-06-05', '2025-06-10')
                GROUP BY date
                ORDER BY date
            """))
            
            print("📅 Specific Date Analysis (qvm_v2.0_enhanced):")
            print("=" * 50)
            
            for row in result:
                print(f"Date: {row[0]}")
                print(f"  Tickers: {row[1]}")
                print(f"  Generated: {row[2]}")
                print()
            
            # Check if factors are identical across dates
            result = conn.execute(text("""
                SELECT 
                    ticker,
                    Quality_Composite,
                    Value_Composite,
                    Momentum_Composite,
                    QVM_Composite,
                    date
                FROM factor_scores_qvm
                WHERE strategy_version = 'qvm_v2.0_enhanced'
                AND ticker IN ('HPG', 'VNM', 'VCB', 'TCB', 'FPT')
                AND date IN ('2025-05-30', '2025-06-05', '2025-06-10')
                ORDER BY ticker, date
            """))
            
            print("🔍 Factor Score Consistency Check:")
            print("=" * 50)
            
            # Group by ticker to check consistency
            ticker_data = {}
            for row in result:
                ticker = row[0]
                if ticker not in ticker_data:
                    ticker_data[ticker] = []
                ticker_data[ticker].append({
                    'date': row[5],
                    'quality': row[1],
                    'value': row[2],
                    'momentum': row[3],
                    'qvm': row[4]
                })
            
            for ticker, data in ticker_data.items():
                print(f"\n{ticker}:")
                for record in data:
                    print(f"  {record['date']}: Q={record['quality']:.6f}, V={record['value']:.6f}, M={record['momentum']:.6f}, QVM={record['qvm']:.6f}")
                
                # Check if all values are identical
                if len(data) > 1:
                    first = data[0]
                    identical = all(
                        record['quality'] == first['quality'] and
                        record['value'] == first['value'] and
                        record['momentum'] == first['momentum'] and
                        record['qvm'] == first['qvm']
                        for record in data[1:]
                    )
                    status = "❌ IDENTICAL (STALE)" if identical else "✅ VARYING (FRESH)"
                    print(f"  Status: {status}")
            
            # Check when factors were last regenerated
            result = conn.execute(text("""
                SELECT 
                    MAX(calculation_timestamp) as last_generated,
                    COUNT(DISTINCT calculation_timestamp) as unique_timestamps
                FROM factor_scores_qvm
                WHERE strategy_version = 'qvm_v2.0_enhanced'
            """))
            
            row = result.fetchone()
            print(f"\n🕒 Factor Generation Status:")
            print(f"  Last Generated: {row[0]}")
            print(f"  Unique Timestamps: {row[1]}")
            
            if row[0]:
                last_gen = row[0]
                if isinstance(last_gen, str):
                    last_gen = datetime.fromisoformat(last_gen.replace('Z', '+00:00'))
                
                days_ago = (datetime.now() - last_gen).days
                print(f"  Days Since Last Generation: {days_ago}")
                
                if days_ago > 7:
                    print(f"  ⚠️ WARNING: Factors may be stale (>7 days old)")
                else:
                    print(f"  ✅ Factors appear to be recent")
            
    except Exception as e:
        print(f"❌ Error checking factor timestamps: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_factor_timestamps()
