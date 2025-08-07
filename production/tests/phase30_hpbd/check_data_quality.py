#!/usr/bin/env python3

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add Project Root to Python Path
current_path = Path.cwd()
while not (current_path / 'production').is_dir():
    current_path = current_path.parent
project_root = current_path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from production.database.connection import DatabaseManager

print("✅ Data Quality Checker for Factor Scores")

def check_data_quality():
    """Check the quality of factor scores data."""
    print("🚀 Checking data quality...")
    print("=" * 80)
    
    try:
        # Database connection
        db_manager = DatabaseManager()
        engine = db_manager.get_engine()
        print("✅ Database connected")
        
        # Check a few specific dates to understand the data
        test_dates = ['2016-01-04', '2016-02-01', '2016-03-01', '2020-01-02', '2025-01-02']
        
        for test_date in test_dates:
            print(f"\n📅 Checking data for {test_date}:")
            
            # Query factor scores for this date
            query = f"""
            SELECT 
                ticker,
                Quality_Composite,
                Value_Composite,
                Momentum_Composite,
                QVM_Composite,
                strategy_version
            FROM factor_scores_qvm
            WHERE date = '{test_date}'
            AND strategy_version = 'qvm_v2.0_enhanced'
            ORDER BY QVM_Composite DESC
            LIMIT 10
            """
            
            df = pd.read_sql(query, engine)
            
            if not df.empty:
                print(f"   Found {len(df)} records")
                print(f"   Value_Composite stats: mean={df['Value_Composite'].mean():.3f}, std={df['Value_Composite'].std():.3f}")
                print(f"   Zero Value_Composite count: {(df['Value_Composite'] == 0).sum()}")
                print(f"   Sample data:")
                print(df[['ticker', 'Quality_Composite', 'Value_Composite', 'Momentum_Composite', 'QVM_Composite']].head(3))
            else:
                print(f"   No data found for {test_date}")
        
        # Check overall statistics
        print(f"\n📊 Overall statistics:")
        
        # Count total records
        count_query = """
        SELECT COUNT(*) as total_records
        FROM factor_scores_qvm
        WHERE strategy_version = 'qvm_v2.0_enhanced'
        """
        total_count = pd.read_sql(count_query, engine).iloc[0]['total_records']
        print(f"   Total records: {total_count}")
        
        # Count zero value scores
        zero_value_query = """
        SELECT COUNT(*) as zero_count
        FROM factor_scores_qvm
        WHERE strategy_version = 'qvm_v2.0_enhanced'
        AND Value_Composite = 0
        """
        zero_count = pd.read_sql(zero_value_query, engine).iloc[0]['zero_count']
        print(f"   Zero Value_Composite records: {zero_count} ({zero_count/total_count*100:.1f}%)")
        
        # Check date distribution of zero values
        zero_dates_query = """
        SELECT date, COUNT(*) as zero_count
        FROM factor_scores_qvm
        WHERE strategy_version = 'qvm_v2.0_enhanced'
        AND Value_Composite = 0
        GROUP BY date
        ORDER BY date
        LIMIT 10
        """
        zero_dates = pd.read_sql(zero_dates_query, engine)
        print(f"   Dates with zero Value_Composite (first 10):")
        for _, row in zero_dates.iterrows():
            print(f"     {row['date']}: {row['zero_count']} records")
        
        # Check if there are any dates with all zero values
        all_zero_dates_query = """
        SELECT date, COUNT(*) as total_count, 
               SUM(CASE WHEN Value_Composite = 0 THEN 1 ELSE 0 END) as zero_count
        FROM factor_scores_qvm
        WHERE strategy_version = 'qvm_v2.0_enhanced'
        GROUP BY date
        HAVING SUM(CASE WHEN Value_Composite = 0 THEN 1 ELSE 0 END) = COUNT(*)
        ORDER BY date
        LIMIT 10
        """
        all_zero_dates = pd.read_sql(all_zero_dates_query, engine)
        print(f"   Dates with ALL zero Value_Composite (first 10):")
        for _, row in all_zero_dates.iterrows():
            print(f"     {row['date']}: {row['zero_count']}/{row['total_count']} records")
        
        # Check when value scores start appearing
        print(f"\n🔍 Checking when Value_Composite scores start appearing:")
        first_nonzero_query = """
        SELECT MIN(date) as first_nonzero_date
        FROM factor_scores_qvm
        WHERE strategy_version = 'qvm_v2.0_enhanced'
        AND Value_Composite != 0
        """
        first_nonzero_date = pd.read_sql(first_nonzero_query, engine).iloc[0]['first_nonzero_date']
        print(f"   First date with non-zero Value_Composite: {first_nonzero_date}")
        
        # Check the transition period
        transition_query = f"""
        SELECT date, 
               COUNT(*) as total_count,
               SUM(CASE WHEN Value_Composite = 0 THEN 1 ELSE 0 END) as zero_count,
               SUM(CASE WHEN Value_Composite != 0 THEN 1 ELSE 0 END) as nonzero_count
        FROM factor_scores_qvm
        WHERE strategy_version = 'qvm_v2.0_enhanced'
        AND date BETWEEN '2016-01-01' AND '2016-02-01'
        GROUP BY date
        ORDER BY date
        """
        transition_data = pd.read_sql(transition_query, engine)
        print(f"   Transition period (Jan 2016):")
        for _, row in transition_data.iterrows():
            print(f"     {row['date']}: {row['zero_count']} zero, {row['nonzero_count']} non-zero")
        
        # Check strategy versions
        versions_query = """
        SELECT strategy_version, COUNT(*) as count
        FROM factor_scores_qvm
        GROUP BY strategy_version
        ORDER BY count DESC
        """
        versions = pd.read_sql(versions_query, engine)
        print(f"\n📋 Available strategy versions:")
        for _, row in versions.iterrows():
            print(f"   {row['strategy_version']}: {row['count']} records")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking data quality: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = check_data_quality()
    if success:
        print("\n🎉 Data quality check completed!")
    else:
        print("\n💥 Data quality check failed!")
