#!/usr/bin/env python3
"""
Processing Pipeline Status Monitor
===================================
Monitors the complete data processing pipeline from raw fundamentals 
through enhanced views to intermediary calculations.

This script provides comprehensive status for:
1. Raw fundamental data coverage by quarter
2. Enhanced view processing status
3. Intermediary calculation coverage by sector
4. Processing gaps and recommendations

Author: Duc Nguyen
Date: July 31, 2025
Status: Production Ready
"""

import sys
import pandas as pd
import pymysql
import yaml
from pathlib import Path
from datetime import datetime, date
from typing import Dict, List, Tuple

# Setup project path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

class ProcessingPipelineMonitor:
    """Monitor processing pipeline status across all stages"""
    
    def __init__(self):
        self.db_connection = self._get_db_connection()
        self.current_quarters = self._get_current_quarters()
        
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
    
    def _get_current_quarters(self):
        """Get current and previous quarter dynamically (same logic as QVM engine)"""
        from datetime import timedelta
        
        current_date = datetime.now()
        current_year = current_date.year
        
        # Define quarter end dates
        quarter_ends = {
            1: datetime(current_year, 3, 31),   # Q1 ends Mar 31
            2: datetime(current_year, 6, 30),   # Q2 ends Jun 30  
            3: datetime(current_year, 9, 30),   # Q3 ends Sep 30
            4: datetime(current_year, 12, 31)   # Q4 ends Dec 31
        }
        
        # Reporting lag: 45 days after quarter end
        reporting_lag = 45
        
        # Find the most recent quarter that should have data available
        available_quarters = []
        for quarter, end_date in quarter_ends.items():
            publish_date = end_date + timedelta(days=reporting_lag)
            if publish_date <= current_date:
                available_quarters.append((current_year, quarter))
        
        # Also check previous year Q4
        prev_year_q4_end = datetime(current_year - 1, 12, 31)
        prev_year_q4_publish = prev_year_q4_end + timedelta(days=reporting_lag)
        if prev_year_q4_publish <= current_date:
            available_quarters.append((current_year - 1, 4))
        
        # Sort by year, quarter and get the two most recent
        available_quarters.sort(key=lambda x: (x[0], x[1]), reverse=True)
        
        if len(available_quarters) >= 2:
            current_quarter = available_quarters[0]  # Most recent
            previous_quarter = available_quarters[1]  # Previous
        elif len(available_quarters) == 1:
            current_quarter = available_quarters[0]
            # Previous quarter fallback
            if current_quarter[1] == 1:
                previous_quarter = (current_quarter[0] - 1, 4)
            else:
                previous_quarter = (current_quarter[0], current_quarter[1] - 1)
        else:
            # Fallback to current year Q1 and previous year Q4
            current_quarter = (current_year, 1)
            previous_quarter = (current_year - 1, 4)
        
        return {
            'current': current_quarter,
            'previous': previous_quarter
        }
    
    def check_raw_data_status(self) -> pd.DataFrame:
        """Check raw fundamental data coverage by quarter"""
        query = """
        SELECT 
            year,
            quarter,
            COUNT(DISTINCT ticker) as tickers,
            COUNT(*) as total_records
        FROM fundamental_values 
        WHERE year >= 2024
        GROUP BY year, quarter
        ORDER BY year DESC, quarter DESC
        """
        
        with self.db_connection.cursor() as cursor:
            cursor.execute(query)
            results = cursor.fetchall()
        
        return pd.DataFrame(results)
    
    def check_enhanced_view_status(self) -> Dict[str, any]:
        """Check enhanced fundamental view status"""
        # Check if views exist
        view_check_query = """
        SELECT 
            TABLE_NAME as table_name,
            TABLE_TYPE as table_type
        FROM information_schema.tables
        WHERE table_schema = DATABASE()
          AND TABLE_NAME LIKE 'v_%fundamental%'
        ORDER BY TABLE_NAME
        """
        
        with self.db_connection.cursor() as cursor:
            cursor.execute(view_check_query)
            views = cursor.fetchall()
        
        # Check current quarter coverage in enhanced view
        current_year, current_quarter = self.current_quarters['current']
        if any(v['table_name'] == 'v_comprehensive_fundamental_items' for v in views):
            coverage_query = f"""
            SELECT 
                COUNT(DISTINCT ticker) as current_quarter_tickers,
                COUNT(*) as current_quarter_records,
                COUNT(DISTINCT CASE WHEN NetRevenue IS NOT NULL THEN ticker END) as tickers_with_revenue,
                COUNT(DISTINCT CASE WHEN TotalAssets IS NOT NULL THEN ticker END) as tickers_with_assets
            FROM v_comprehensive_fundamental_items
            WHERE year = {current_year} AND quarter = {current_quarter}
            """
            
            with self.db_connection.cursor() as cursor:
                cursor.execute(coverage_query)
                coverage = cursor.fetchone()
        else:
            coverage = None
        
        return {
            'views_available': views,
            'current_quarter_coverage': coverage,
            'current_quarter': f"Q{current_quarter} {current_year}"
        }
    
    def check_intermediary_status(self) -> Dict[str, pd.DataFrame]:
        """Check intermediary calculation status by sector"""
        
        # Banking intermediaries
        banking_query = """
        SELECT 
            'Banking' as sector,
            COUNT(DISTINCT ticker) as total_tickers,
            COUNT(DISTINCT CASE WHEN year = 2025 AND quarter = 1 THEN ticker END) as q1_2025,
            COUNT(DISTINCT CASE WHEN year = 2025 AND quarter = 2 THEN ticker END) as q2_2025,
            ROUND(COUNT(DISTINCT CASE WHEN year = 2025 AND quarter = 2 THEN ticker END) * 100.0 / 
                  COUNT(DISTINCT ticker), 1) as q2_coverage_pct
        FROM intermediary_calculations_banking_cleaned
        WHERE year = 2025
        """
        
        # Securities intermediaries  
        securities_query = """
        SELECT 
            'Securities' as sector,
            COUNT(DISTINCT ticker) as total_tickers,
            COUNT(DISTINCT CASE WHEN year = 2025 AND quarter = 1 THEN ticker END) as q1_2025,
            COUNT(DISTINCT CASE WHEN year = 2025 AND quarter = 2 THEN ticker END) as q2_2025,
            ROUND(COUNT(DISTINCT CASE WHEN year = 2025 AND quarter = 2 THEN ticker END) * 100.0 / 
                  COUNT(DISTINCT ticker), 1) as q2_coverage_pct
        FROM intermediary_calculations_securities_cleaned
        WHERE year = 2025
        """
        
        # Non-financial intermediaries by sector
        nonfin_query = """
        SELECT 
            m.sector,
            COUNT(DISTINCT m.ticker) as total_tickers,
            COUNT(DISTINCT CASE WHEN i.year = 2025 AND i.quarter = 1 THEN i.ticker END) as q1_2025,
            COUNT(DISTINCT CASE WHEN i.year = 2025 AND i.quarter = 2 THEN i.ticker END) as q2_2025,
            ROUND(COUNT(DISTINCT CASE WHEN i.year = 2025 AND i.quarter = 2 THEN i.ticker END) * 100.0 / 
                  COUNT(DISTINCT m.ticker), 1) as q2_coverage_pct
        FROM master_info m
        LEFT JOIN intermediary_calculations_enhanced i ON m.ticker = i.ticker AND i.year = 2025
        WHERE m.sector NOT IN ('Banks', 'Securities')
        GROUP BY m.sector
        ORDER BY q2_coverage_pct DESC, total_tickers DESC
        """
        
        results = {}
        
        # Execute queries
        for name, query in [('banking', banking_query), ('securities', securities_query), ('non_financial', nonfin_query)]:
            with self.db_connection.cursor() as cursor:
                cursor.execute(query)
                results[name] = pd.DataFrame(cursor.fetchall())
        
        return results
    
    def check_processing_gaps(self) -> Dict[str, any]:
        """Identify processing gaps and provide recommendations"""
        
        # Compare raw data vs intermediary processing
        gap_query = """
        SELECT 
            'Banking' as sector_type,
            (SELECT COUNT(DISTINCT fv.ticker) FROM fundamental_values fv 
             JOIN master_info m ON fv.ticker = m.ticker 
             WHERE m.sector = 'Banks' AND fv.year = 2025 AND fv.quarter = 2) as raw_q2_tickers,
            (SELECT COUNT(DISTINCT ticker) FROM intermediary_calculations_banking_cleaned 
             WHERE year = 2025 AND quarter = 2) as processed_q2_tickers
        
        UNION ALL
        
        SELECT 
            'Securities' as sector_type,
            (SELECT COUNT(DISTINCT fv.ticker) FROM fundamental_values fv 
             JOIN master_info m ON fv.ticker = m.ticker 
             WHERE m.sector = 'Securities' AND fv.year = 2025 AND fv.quarter = 2) as raw_q2_tickers,
            (SELECT COUNT(DISTINCT ticker) FROM intermediary_calculations_securities_cleaned 
             WHERE year = 2025 AND quarter = 2) as processed_q2_tickers
        
        UNION ALL
        
        SELECT 
            'Non-Financial' as sector_type,
            (SELECT COUNT(DISTINCT fv.ticker) FROM fundamental_values fv 
             JOIN master_info m ON fv.ticker = m.ticker 
             WHERE m.sector NOT IN ('Banks', 'Securities') AND fv.year = 2025 AND fv.quarter = 2) as raw_q2_tickers,
            (SELECT COUNT(DISTINCT ticker) FROM intermediary_calculations_enhanced 
             WHERE year = 2025 AND quarter = 2) as processed_q2_tickers
        """
        
        with self.db_connection.cursor() as cursor:
            cursor.execute(gap_query)
            gaps = pd.DataFrame(cursor.fetchall())
        
        # Calculate processing efficiency
        gaps['processing_gap'] = gaps['raw_q2_tickers'] - gaps['processed_q2_tickers']
        gaps['processing_efficiency_pct'] = round(gaps['processed_q2_tickers'] * 100.0 / gaps['raw_q2_tickers'], 1)
        
        return {
            'gap_analysis': gaps,
            'total_raw_q2': int(gaps['raw_q2_tickers'].sum()),
            'total_processed_q2': int(gaps['processed_q2_tickers'].sum()),
            'overall_efficiency': round(gaps['processed_q2_tickers'].sum() * 100.0 / gaps['raw_q2_tickers'].sum(), 1)
        }
    
    def generate_recommendations(self, gaps: Dict[str, any]) -> List[str]:
        """Generate actionable recommendations based on processing gaps"""
        recommendations = []
        
        gap_df = gaps['gap_analysis']
        
        for _, row in gap_df.iterrows():
            sector = row['sector_type']
            efficiency = row['processing_efficiency_pct']
            gap = row['processing_gap']
            
            if efficiency < 95 and gap > 0:
                if sector == 'Banking':
                    recommendations.append(f"🔄 Run: python3 scripts/intermediaries/banking_sector_intermediary_calculator.py ({gap} tickers need processing)")
                elif sector == 'Securities':
                    recommendations.append(f"🔄 Run: python3 scripts/intermediaries/securities_sector_intermediary_calculator.py ({gap} tickers need processing)")
                elif sector == 'Non-Financial':
                    recommendations.append(f"🔄 Run: python3 scripts/intermediaries/all_nonfin_sectors_intermediary_calculator.py ({gap} tickers need processing)")
        
        # Overall recommendations
        if gaps['overall_efficiency'] < 95:
            recommendations.append("📊 Consider running Production Menu option 3.5 (Full Intermediary Calculation)")
        
        if gaps['overall_efficiency'] >= 95:
            recommendations.append("✅ Processing pipeline is current - ready for factor generation")
        
        return recommendations
    
    def run_complete_status_check(self):
        """Run complete processing pipeline status check"""
        print("=" * 80)
        print("🔍 PROCESSING PIPELINE STATUS MONITOR")
        print(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S ICT')}")
        print("=" * 80)
        
        # 1. Raw Data Status
        print("\n📊 1. RAW FUNDAMENTAL DATA STATUS")
        print("-" * 50)
        raw_status = self.check_raw_data_status()
        print(raw_status.to_string(index=False))
        
        # 2. Enhanced View Status
        print("\n🔍 2. ENHANCED FUNDAMENTAL VIEW STATUS")
        print("-" * 50)
        view_status = self.check_enhanced_view_status()
        
        if view_status['views_available']:
            print("✅ Available Views:")
            for view in view_status['views_available']:
                print(f"   • {view['table_name']} ({view['table_type']})")
        else:
            print("❌ No enhanced fundamental views found")
        
        if view_status['q2_2025_coverage']:
            cov = view_status['q2_2025_coverage']
            print(f"\n📈 Q2 2025 Enhanced View Coverage:")
            print(f"   • Total tickers: {cov['q2_2025_tickers']:,}")
            print(f"   • With revenue data: {cov['tickers_with_revenue']:,}")
            print(f"   • With asset data: {cov['tickers_with_assets']:,}")
        
        # 3. Intermediary Status
        print("\n⚙️ 3. INTERMEDIARY CALCULATION STATUS")
        print("-" * 50)
        intermediary_status = self.check_intermediary_status()
        
        # Combine all sectors
        all_sectors = pd.concat([
            intermediary_status['banking'],
            intermediary_status['securities'],
            intermediary_status['non_financial']
        ], ignore_index=True)
        
        print(all_sectors.to_string(index=False))
        
        # 4. Processing Gaps
        print("\n🔍 4. PROCESSING GAP ANALYSIS")
        print("-" * 50)
        gaps = self.check_processing_gaps()
        print(gaps['gap_analysis'].to_string(index=False))
        
        print(f"\n📊 Overall Processing Efficiency: {gaps['overall_efficiency']}%")
        print(f"   • Raw Q2 2025 data: {gaps['total_raw_q2']:,} companies")
        print(f"   • Processed Q2 2025: {gaps['total_processed_q2']:,} companies")
        print(f"   • Processing gap: {gaps['total_raw_q2'] - gaps['total_processed_q2']} companies")
        
        # 5. Recommendations
        print("\n🎯 5. ACTIONABLE RECOMMENDATIONS")
        print("-" * 50)
        recommendations = self.generate_recommendations(gaps)
        
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec}")
        else:
            print("✅ No actions needed - processing pipeline is current")
        
        print("\n" + "=" * 80)

def main():
    """Main execution function"""
    try:
        monitor = ProcessingPipelineMonitor()
        monitor.run_complete_status_check()
    except Exception as e:
        print(f"❌ Error running processing pipeline status check: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()