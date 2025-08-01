#!/usr/bin/env python3
"""
Factor Generation Status Monitor
================================
Monitors the factor_scores_qvm table to provide comprehensive status on:
1. Factor generation coverage by version and date
2. Recent factor generation activity
3. Data quality metrics and gaps
4. Recommendations for factor generation workflow

Author: Duc Nguyen
Date: July 31, 2025
Status: Production Ready
"""

import sys
import pandas as pd
import pymysql
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

# Setup project path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

class FactorGenerationMonitor:
    """Monitor factor generation status and coverage"""
    
    def __init__(self):
        self.db_connection = self._get_db_connection()
        
    def __del__(self):
        if hasattr(self, 'db_connection') and self.db_connection:
            self.db_connection.close()
    
    def _get_db_connection(self):
        """Get database connection"""
        try:
            config_path = project_root / 'config' / 'database.yml'
            with open(config_path, 'r') as f:
                db_config = yaml.safe_load(f)['production']
            
            connection = pymysql.connect(
                host=db_config['host'],
                user=db_config['username'],
                password=db_config['password'],
                database=db_config['schema_name'],
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor
            )
            return connection
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            raise
    
    def check_factor_coverage_by_version(self) -> pd.DataFrame:
        """Check factor generation coverage by strategy version"""
        query = """
        SELECT 
            strategy_version,
            COUNT(DISTINCT date) as unique_dates,
            COUNT(DISTINCT ticker) as unique_tickers,
            COUNT(*) as total_records,
            MIN(date) as earliest_date,
            MAX(date) as latest_date,
            MAX(calculation_timestamp) as last_generated
        FROM factor_scores_qvm
        GROUP BY strategy_version
        ORDER BY last_generated DESC
        """
        
        with self.db_connection.cursor() as cursor:
            cursor.execute(query)
            results = cursor.fetchall()
        
        return pd.DataFrame(results)
    
    def check_recent_factor_activity(self, days: int = 7) -> pd.DataFrame:
        """Check recent factor generation activity"""
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        query = """
        SELECT 
            date,
            strategy_version,
            COUNT(DISTINCT ticker) as tickers_processed,
            COUNT(*) as total_records,
            MIN(calculation_timestamp) as first_generated,
            MAX(calculation_timestamp) as last_generated
        FROM factor_scores_qvm
        WHERE date >= %s
        GROUP BY date, strategy_version
        ORDER BY date DESC, strategy_version
        """
        
        with self.db_connection.cursor() as cursor:
            cursor.execute(query, (cutoff_date,))
            results = cursor.fetchall()
        
        return pd.DataFrame(results)
    
    def check_data_quality_metrics(self) -> Dict[str, any]:
        """Check data quality metrics for factor scores"""
        
        # Check for NULL values
        null_check_query = """
        SELECT 
            strategy_version,
            COUNT(*) as total_records,
            SUM(CASE WHEN Quality_Composite IS NULL THEN 1 ELSE 0 END) as null_quality,
            SUM(CASE WHEN Value_Composite IS NULL THEN 1 ELSE 0 END) as null_value,
            SUM(CASE WHEN Momentum_Composite IS NULL THEN 1 ELSE 0 END) as null_momentum,
            SUM(CASE WHEN QVM_Composite IS NULL THEN 1 ELSE 0 END) as null_qvm
        FROM factor_scores_qvm
        WHERE date >= CURDATE() - INTERVAL 30 DAY
        GROUP BY strategy_version
        """
        
        # Check for extreme values (potential outliers)
        outlier_check_query = """
        SELECT 
            strategy_version,
            COUNT(*) as total_records,
            SUM(CASE WHEN ABS(QVM_Composite) > 10 THEN 1 ELSE 0 END) as extreme_qvm_values,
            MIN(QVM_Composite) as min_qvm,
            MAX(QVM_Composite) as max_qvm,
            AVG(QVM_Composite) as avg_qvm,
            STDDEV(QVM_Composite) as stddev_qvm
        FROM factor_scores_qvm
        WHERE date >= CURDATE() - INTERVAL 30 DAY
        GROUP BY strategy_version
        """
        
        # Check for duplicate records
        duplicate_check_query = """
        SELECT 
            strategy_version,
            COUNT(*) as total_records,
            COUNT(DISTINCT CONCAT(ticker, '-', date)) as unique_combinations,
            COUNT(*) - COUNT(DISTINCT CONCAT(ticker, '-', date)) as duplicate_records
        FROM factor_scores_qvm
        WHERE date >= CURDATE() - INTERVAL 30 DAY
        GROUP BY strategy_version
        """
        
        results = {}
        
        with self.db_connection.cursor() as cursor:
            # NULL checks
            cursor.execute(null_check_query)
            results['null_checks'] = pd.DataFrame(cursor.fetchall())
            
            # Outlier checks
            cursor.execute(outlier_check_query)
            results['outlier_checks'] = pd.DataFrame(cursor.fetchall())
            
            # Duplicate checks
            cursor.execute(duplicate_check_query)
            results['duplicate_checks'] = pd.DataFrame(cursor.fetchall())
        
        return results
    
    def find_missing_dates(self, version: str = 'qvm_v2.0_enhanced', days_back: int = 30) -> List[str]:
        """Find missing trading dates for a specific version"""
        end_date = (datetime.now() - timedelta(days=1)).date()
        start_date = end_date - timedelta(days=days_back)
        
        # Get all trading dates from equity_history
        trading_dates_query = """
        SELECT DISTINCT date
        FROM equity_history
        WHERE date BETWEEN %s AND %s
        AND volume > 0
        ORDER BY date
        """
        
        # Get existing dates for this version
        existing_dates_query = """
        SELECT DISTINCT date
        FROM factor_scores_qvm
        WHERE date BETWEEN %s AND %s
        AND strategy_version = %s
        ORDER BY date
        """
        
        with self.db_connection.cursor() as cursor:
            # Get all trading dates
            cursor.execute(trading_dates_query, (start_date, end_date))
            all_dates = {row['date'] for row in cursor.fetchall()}
            
            # Get existing dates
            cursor.execute(existing_dates_query, (start_date, end_date, version))
            existing_dates = {row['date'] for row in cursor.fetchall()}
        
        # Find missing dates
        missing_dates = sorted(all_dates - existing_dates)
        return [date.strftime('%Y-%m-%d') for date in missing_dates]
    
    def generate_recommendations(self, coverage_df: pd.DataFrame, missing_dates: List[str]) -> List[str]:
        """Generate actionable recommendations based on factor generation status"""
        recommendations = []
        
        if coverage_df.empty:
            recommendations.append("❌ No factor scores found in database")
            recommendations.append("🔄 Run Option 4.1 to generate initial factor scores")
            return recommendations
        
        latest_version = coverage_df.iloc[0]['strategy_version']
        latest_date = coverage_df.iloc[0]['latest_date']
        
        # Convert latest_date to date object if it's a string
        if isinstance(latest_date, str):
            latest_date = datetime.strptime(latest_date, '%Y-%m-%d').date()
        elif hasattr(latest_date, 'date'):
            latest_date = latest_date.date()
        
        # Check if data is current
        days_since_latest = (datetime.now().date() - latest_date).days
        
        if days_since_latest > 5:
            recommendations.append(f"⚠️ Factor scores are {days_since_latest} days old (latest: {latest_date})")
            recommendations.append("🔄 Run Option 4.3 (Incremental Update) to catch up")
        
        if missing_dates:
            recommendations.append(f"📊 Found {len(missing_dates)} missing trading dates")
            recommendations.append(f"🔄 Run Option 4.3 to fill gaps: {missing_dates[:3]}{'...' if len(missing_dates) > 3 else ''}")
        
        # Check for multiple versions
        if len(coverage_df) > 1:
            recommendations.append(f"💡 Multiple versions detected: {list(coverage_df['strategy_version'])}")
            recommendations.append(f"✅ Currently using: {latest_version}")
        
        if not missing_dates and days_since_latest <= 1:
            recommendations.append("✅ Factor generation is current - no action needed")
        
        return recommendations
    
    def run_complete_status_check(self):
        """Run complete factor generation status check"""
        print("=" * 80)
        print("🔍 FACTOR GENERATION STATUS MONITOR")
        print(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S ICT')}")
        print("=" * 80)
        
        # 1. Coverage by Version
        print("\n📊 1. FACTOR GENERATION COVERAGE BY VERSION")
        print("-" * 50)
        coverage_df = self.check_factor_coverage_by_version()
        
        if not coverage_df.empty:
            # Format dates for display
            coverage_df['earliest_date'] = pd.to_datetime(coverage_df['earliest_date']).dt.strftime('%Y-%m-%d')
            coverage_df['latest_date'] = pd.to_datetime(coverage_df['latest_date']).dt.strftime('%Y-%m-%d')
            coverage_df['last_generated'] = pd.to_datetime(coverage_df['last_generated']).dt.strftime('%Y-%m-%d %H:%M')
            
            print(coverage_df.to_string(index=False))
        else:
            print("❌ No factor scores found in database")
        
        # 2. Recent Activity
        print("\n⏰ 2. RECENT FACTOR GENERATION ACTIVITY (Last 7 Days)")
        print("-" * 50)
        recent_df = self.check_recent_factor_activity(7)
        
        if not recent_df.empty:
            # Format timestamps
            recent_df['first_generated'] = pd.to_datetime(recent_df['first_generated']).dt.strftime('%H:%M:%S')
            recent_df['last_generated'] = pd.to_datetime(recent_df['last_generated']).dt.strftime('%H:%M:%S')
            print(recent_df.to_string(index=False))
        else:
            print("❌ No recent factor generation activity found")
        
        # 3. Data Quality Metrics
        print("\n🔍 3. DATA QUALITY METRICS (Last 30 Days)")
        print("-" * 50)
        quality_metrics = self.check_data_quality_metrics()
        
        if not quality_metrics['null_checks'].empty:
            print("NULL Value Analysis:")
            print(quality_metrics['null_checks'].to_string(index=False))
            print()
        
        if not quality_metrics['outlier_checks'].empty:
            print("Outlier Analysis:")
            outlier_df = quality_metrics['outlier_checks'].copy()
            # Format numeric columns (handle decimal.Decimal types)
            for col in ['min_qvm', 'max_qvm', 'avg_qvm', 'stddev_qvm']:
                if col in outlier_df.columns:
                    # Convert to float first to handle decimal.Decimal types
                    outlier_df[col] = pd.to_numeric(outlier_df[col], errors='coerce').round(4)
            print(outlier_df.to_string(index=False))
            print()
        
        if not quality_metrics['duplicate_checks'].empty:
            print("Duplicate Record Analysis:")
            print(quality_metrics['duplicate_checks'].to_string(index=False))
        
        # 4. Missing Dates Analysis
        print("\n📅 4. MISSING DATES ANALYSIS")
        print("-" * 50)
        
        if not coverage_df.empty:
            latest_version = coverage_df.iloc[0]['strategy_version']
            missing_dates = self.find_missing_dates(latest_version, 30)
            
            print(f"Checking missing dates for version: {latest_version}")
            print(f"Search period: Last 30 trading days")
            
            if missing_dates:
                print(f"Missing dates found: {len(missing_dates)}")
                print(f"First 10 missing: {missing_dates[:10]}")
            else:
                print("✅ No missing dates found")
        else:
            missing_dates = []
        
        # 5. Recommendations
        print("\n🎯 5. ACTIONABLE RECOMMENDATIONS")
        print("-" * 50)
        recommendations = self.generate_recommendations(coverage_df, missing_dates)
        
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec}")
        else:
            print("✅ No specific recommendations - system appears healthy")
        
        print("\n" + "=" * 80)

def main():
    """Main execution function"""
    try:
        monitor = FactorGenerationMonitor()
        monitor.run_complete_status_check()
    except Exception as e:
        print(f"❌ Error running factor generation status check: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()