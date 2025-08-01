#!/usr/bin/env python3
"""
Find Correct VNM Revenue
========================

This script finds the correct revenue item_id for VNM by testing
different PL items against the known Net Sales value.
"""

import sys
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine, text

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

def find_correct_vnm_revenue():
    """Find the correct revenue item_id for VNM."""
    print("🔍 Finding Correct VNM Revenue Item")
    print("=" * 50)
    
    # Connect to database
    db_manager = get_database_manager()
    engine = db_manager.get_engine()
    
    # Known values from VNM financial statements (Q2 2025)
    known_net_sales = 16724.60  # bn VND
    known_net_profit = 2488.58  # bn VND
    known_total_assets = 55282.66  # bn VND
    
    print(f"📋 Known Values from VNM Financial Statements (Q2 2025):")
    print(f"   Net Sales: {known_net_sales:.2f} bn VND")
    print(f"   Net Profit: {known_net_profit:.2f} bn VND")
    print(f"   Total Assets: {known_total_assets:.2f} bn VND")
    
    # Test date
    test_date = pd.Timestamp('2025-01-29')
    lag_days = 45
    lag_date = test_date - pd.Timedelta(days=lag_days)
    lag_year = lag_date.year
    lag_quarter = ((lag_date.month - 1) // 3) + 1
    
    print(f"\n🔍 Testing different PL items for VNM:")
    print(f"   Test Date: {test_date.date()}")
    print(f"   Lag Date: {lag_date.date()}")
    print(f"   Lag Year/Quarter: {lag_year}Q{lag_quarter}")
    
    # Get all PL items for VNM
    pl_query = text("""
        SELECT DISTINCT 
            fv.item_id,
            COUNT(*) as count,
            AVG(fv.value) as avg_value,
            MAX(fv.value) as max_value,
            MIN(fv.value) as min_value
        FROM fundamental_values fv
        WHERE fv.ticker = 'VNM'
        AND fv.statement_type = 'PL'
        AND (fv.year < :lag_year OR (fv.year = :lag_year AND fv.quarter <= :lag_quarter))
        GROUP BY fv.item_id
        ORDER BY avg_value DESC
    """)
    
    pl_data = pd.read_sql(pl_query, engine, params={
        'lag_year': lag_year,
        'lag_quarter': lag_quarter
    })
    
    print(f"\n📊 Available PL Items for VNM (sorted by average value):")
    print(pl_data.to_string(index=False))
    
    # Test each PL item as potential revenue
    print(f"\n🧪 Testing each PL item as potential revenue:")
    print("-" * 80)
    
    results = []
    for _, row in pl_data.iterrows():
        item_id = row['item_id']
        avg_value = row['avg_value']
        
        # Get the latest value for this item
        latest_query = text("""
            SELECT 
                fv.year,
                fv.quarter,
                fv.value
            FROM fundamental_values fv
            WHERE fv.ticker = 'VNM'
            AND fv.item_id = :item_id
            AND fv.statement_type = 'PL'
            AND (fv.year < :lag_year OR (fv.year = :lag_year AND fv.quarter <= :lag_quarter))
            ORDER BY fv.year DESC, fv.quarter DESC
            LIMIT 1
        """)
        
        latest_result = pd.read_sql(latest_query, engine, params={
            'item_id': item_id,
            'lag_year': lag_year,
            'lag_quarter': lag_quarter
        })
        
        if not latest_result.empty:
            latest_value = latest_result.iloc[0]['value']
            latest_year = latest_result.iloc[0]['year']
            latest_quarter = latest_result.iloc[0]['quarter']
            
            # Calculate how close this value is to known Net Sales
            difference = abs(latest_value - known_net_sales)
            percentage_diff = (difference / known_net_sales) * 100
            
            # Calculate potential ratios
            potential_net_margin = known_net_profit / latest_value if latest_value > 0 else None
            potential_asset_turnover = latest_value / known_total_assets if known_total_assets > 0 else None
            
            results.append({
                'item_id': item_id,
                'latest_value': latest_value,
                'latest_period': f"{latest_year}Q{latest_quarter}",
                'avg_value': avg_value,
                'difference_from_net_sales': difference,
                'percentage_diff': percentage_diff,
                'potential_net_margin': potential_net_margin,
                'potential_asset_turnover': potential_asset_turnover
            })
            
            print(f"   Item {int(item_id):3d}: {latest_value:10.0f} bn VND ({latest_year}Q{latest_quarter})")
            print(f"           Diff from Net Sales: {difference:8.0f} bn VND ({percentage_diff:5.1f}%)")
            if potential_net_margin:
                print(f"           Potential Net Margin: {potential_net_margin*100:5.1f}%")
            if potential_asset_turnover:
                print(f"           Potential Asset Turnover: {potential_asset_turnover*100:5.1f}%")
            print()
    
    # Sort results by percentage difference
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('percentage_diff')
    
    print(f"\n🏆 Top Candidates for Net Sales (sorted by closest match):")
    print("-" * 80)
    
    for i, row in results_df.head(10).iterrows():
        print(f"#{i+1:2d} - Item {int(row['item_id']):3d}: {row['latest_value']:10.0f} bn VND")
        print(f"     Difference: {row['difference_from_net_sales']:8.0f} bn VND ({row['percentage_diff']:5.1f}%)")
        if row['potential_net_margin']:
            print(f"     Net Margin: {row['potential_net_margin']*100:5.1f}%")
        if row['potential_asset_turnover']:
            print(f"     Asset Turnover: {row['potential_asset_turnover']*100:5.1f}%")
        print()
    
    # Find the best candidate
    best_candidate = results_df.iloc[0]
    print(f"🎯 Best Candidate: Item {int(best_candidate['item_id'])}")
    print(f"   Value: {best_candidate['latest_value']:.0f} bn VND")
    print(f"   Difference from Net Sales: {best_candidate['difference_from_net_sales']:.0f} bn VND ({best_candidate['percentage_diff']:.1f}%)")
    
    if best_candidate['potential_net_margin']:
        print(f"   Net Margin: {best_candidate['potential_net_margin']*100:.1f}%")
    if best_candidate['potential_asset_turnover']:
        print(f"   Asset Turnover: {best_candidate['potential_asset_turnover']*100:.1f}%")

if __name__ == "__main__":
    find_correct_vnm_revenue() 