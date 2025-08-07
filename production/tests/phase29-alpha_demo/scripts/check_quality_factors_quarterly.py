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

print("✅ Quality factor quarterly check script initialized")

def check_quality_factors_quarterly():
    """Check quality factors across quarters and years to verify quarterly refresh."""
    try:
        db_manager = get_database_manager()
        engine = db_manager.get_engine()
        
        with engine.connect() as conn:
            # Get quality factors across different quarters/years
            result = conn.execute(text("""
                SELECT 
                    ticker,
                    date,
                    Quality_Composite,
                    calculation_timestamp
                FROM factor_scores_qvm
                WHERE strategy_version = 'qvm_v2.0_enhanced'
                AND ticker IN ('HPG', 'VNM', 'VCB', 'TCB', 'FPT')
                AND date IN (
                    '2022-03-31', '2022-06-30', '2022-09-30', '2022-12-30',
                    '2023-03-31', '2023-06-30', '2023-09-29', '2023-12-29',
                    '2024-03-29', '2024-06-28', '2024-09-27', '2024-12-31',
                    '2025-03-31', '2025-06-30'
                )
                ORDER BY ticker, date
            """))
            
            print("📊 Quality Factor Quarterly Analysis")
            print("=" * 60)
            
            # Group by ticker to check quarterly changes
            ticker_data = {}
            for row in result:
                ticker = row[0]
                if ticker not in ticker_data:
                    ticker_data[ticker] = []
                ticker_data[ticker].append({
                    'date': row[1],
                    'quality': row[2],
                    'timestamp': row[3]
                })
            
            for ticker, data in ticker_data.items():
                print(f"\n{ticker}:")
                
                # Sort by date
                data.sort(key=lambda x: x['date'])
                
                # Check quarterly changes
                quarterly_changes = []
                for i in range(1, len(data)):
                    prev_date = data[i-1]['date']
                    curr_date = data[i]['date']
                    prev_quality = data[i-1]['quality']
                    curr_quality = data[i]['quality']
                    
                    # Check if it's a quarter change (3 months apart)
                    prev_dt = prev_date if isinstance(prev_date, datetime) else datetime.strptime(str(prev_date), '%Y-%m-%d')
                    curr_dt = curr_date if isinstance(curr_date, datetime) else datetime.strptime(str(curr_date), '%Y-%m-%d')
                    months_diff = (curr_dt.year - prev_dt.year) * 12 + curr_dt.month - prev_dt.month
                    
                    if months_diff >= 3:  # Quarterly change
                        change = curr_quality - prev_quality
                        quarterly_changes.append({
                            'from': prev_date,
                            'to': curr_date,
                            'change': change,
                            'months': months_diff
                        })
                
                # Show quarterly changes
                if quarterly_changes:
                    print(f"  📈 Quarterly Changes:")
                    for change in quarterly_changes:
                        status = "✅ CHANGED" if abs(change['change']) > 0.001 else "❌ SAME"
                        print(f"    {change['from']} → {change['to']} ({change['months']}m): {change['change']:.6f} {status}")
                else:
                    print(f"  ⚠️ No quarterly changes detected")
                
                # Show all values
                print(f"  📊 All Values:")
                for record in data:
                    print(f"    {record['date']}: {record['quality']:.6f}")
            
            # Check if quality factors are updated quarterly
            result = conn.execute(text("""
                SELECT 
                    COUNT(DISTINCT date) as unique_dates,
                    COUNT(DISTINCT calculation_timestamp) as unique_timestamps,
                    MIN(date) as earliest_date,
                    MAX(date) as latest_date
                FROM factor_scores_qvm
                WHERE strategy_version = 'qvm_v2.0_enhanced'
                AND date >= '2022-01-01'
            """))
            
            row = result.fetchone()
            print(f"\n🕒 Quality Factor Generation Summary:")
            print(f"  Unique Dates: {row[0]}")
            print(f"  Unique Timestamps: {row[1]}")
            print(f"  Date Range: {row[2]} to {row[3]}")
            
            if row[1] < row[0] * 0.5:
                print(f"  ⚠️ WARNING: Very few unique timestamps - factors may be stale!")
            elif row[1] < row[0] * 0.8:
                print(f"  ⚠️ WARNING: Fewer timestamps than dates - some factors may be stale!")
            else:
                print(f"  ✅ Factors appear to be generated properly")
                
    except Exception as e:
        print(f"❌ Error checking quality factors quarterly: {e}")

if __name__ == "__main__":
    check_quality_factors_quarterly()
