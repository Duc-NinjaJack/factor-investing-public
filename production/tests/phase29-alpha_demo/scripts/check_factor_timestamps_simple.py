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

print("✅ Simple factor timestamp check script initialized")

def check_factor_timestamps_simple():
    """Check factor generation timestamps with simple queries."""
    try:
        db_manager = get_database_manager()
        engine = db_manager.get_engine()
        
        with engine.connect() as conn:
            # Simple check for specific dates
            result = conn.execute(text("""
                SELECT 
                    ticker,
                    date,
                    Quality_Composite,
                    Value_Composite,
                    Momentum_Composite,
                    calculation_timestamp
                FROM factor_scores_qvm
                WHERE strategy_version = 'qvm_v2.0_enhanced'
                AND ticker IN ('HPG', 'VNM', 'VCB')
                AND date IN ('2025-05-30', '2025-06-05', '2025-06-10')
                ORDER BY ticker, date
                LIMIT 20
            """))
            
            print("🔍 Factor Score Consistency Check (Simple):")
            print("=" * 60)
            
            # Group by ticker to check consistency
            ticker_data = {}
            for row in result:
                ticker = row[0]
                if ticker not in ticker_data:
                    ticker_data[ticker] = []
                ticker_data[ticker].append({
                    'date': row[1],
                    'quality': row[2],
                    'value': row[3],
                    'momentum': row[4],
                    'timestamp': row[5]
                })
            
            for ticker, data in ticker_data.items():
                print(f"\n{ticker}:")
                for record in data:
                    print(f"  {record['date']}: Q={record['quality']:.6f}, V={record['value']:.6f}, M={record['momentum']:.6f}")
                    print(f"    Generated: {record['timestamp']}")
                
                # Check if all values are identical
                if len(data) > 1:
                    first = data[0]
                    identical = all(
                        record['quality'] == first['quality'] and
                        record['value'] == first['value'] and
                        record['momentum'] == first['momentum']
                        for record in data[1:]
                    )
                    status = "❌ IDENTICAL (STALE)" if identical else "✅ VARYING (FRESH)"
                    print(f"  Status: {status}")
            
            # Check latest timestamp
            result = conn.execute(text("""
                SELECT calculation_timestamp
                FROM factor_scores_qvm
                WHERE strategy_version = 'qvm_v2.0_enhanced'
                ORDER BY calculation_timestamp DESC
                LIMIT 1
            """))
            
            row = result.fetchone()
            if row:
                print(f"\n🕒 Latest Factor Generation:")
                print(f"  Timestamp: {row[0]}")
                
                if isinstance(row[0], str):
                    last_gen = datetime.fromisoformat(row[0].replace('Z', '+00:00'))
                else:
                    last_gen = row[0]
                
                days_ago = (datetime.now() - last_gen).days
                print(f"  Days Ago: {days_ago}")
                
                if days_ago > 7:
                    print(f"  ⚠️ WARNING: Factors may be stale (>7 days old)")
                else:
                    print(f"  ✅ Factors appear to be recent")
            
    except Exception as e:
        print(f"❌ Error checking factor timestamps: {e}")

if __name__ == "__main__":
    check_factor_timestamps_simple()
