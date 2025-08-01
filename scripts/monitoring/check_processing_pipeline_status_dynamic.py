#!/usr/bin/env python3
"""
Dynamic Processing Pipeline Status Monitor
==========================================
Monitors the complete data processing pipeline with DYNAMIC quarter detection.
No hardcoded quarters - automatically detects current and previous quarters
based on 45-day reporting lag (same logic as QVM engine).

This script provides comprehensive status for:
1. Raw fundamental data coverage by quarter
2. Enhanced view processing status  
3. Intermediary calculation coverage by sector
4. Processing gaps and recommendations

Author: Duc Nguyen
Date: July 31, 2025
Status: Production Ready - DYNAMIC VERSION
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

class DynamicProcessingPipelineMonitor:
    """Monitor processing pipeline status with dynamic quarter detection"""
    
    def __init__(self):
        self.db_connection = self._get_db_connection()
        self.quarters = self._get_dynamic_quarters()
        
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
    
    def _get_dynamic_quarters(self):
        """Get current and previous quarter based on CALENDAR TIME (for real-time monitoring)"""
        current_date = datetime.now()
        current_year = current_date.year
        current_month = current_date.month
        
        # Determine the last completed quarter based on calendar
        # This is for monitoring - we want to see what quarter is DUE, not what's available per 45-day lag
        if current_month >= 10:  # Oct, Nov, Dec - Q3 is due
            current_quarter = (current_year, 3)
            previous_quarter = (current_year, 2)
        elif current_month >= 7:  # Jul, Aug, Sep - Q2 is due
            current_quarter = (current_year, 2)
            previous_quarter = (current_year, 1)
        elif current_month >= 4:  # Apr, May, Jun - Q1 is due
            current_quarter = (current_year, 1)
            previous_quarter = (current_year - 1, 4)
        else:  # Jan, Feb, Mar - Q4 of previous year is due
            current_quarter = (current_year - 1, 4)
            previous_quarter = (current_year - 1, 3)
        
        return {
            'current': current_quarter,
            'previous': previous_quarter,
            'current_str': f"Q{current_quarter[1]} {current_quarter[0]}",
            'previous_str': f"Q{previous_quarter[1]} {previous_quarter[0]}"
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
        current_year, current_quarter = self.quarters['current']
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
            'current_quarter': self.quarters['current_str']
        }
    
    def check_intermediary_status(self) -> Dict[str, pd.DataFrame]:
        """Check intermediary calculation status by sector"""
        
        current_year, current_quarter = self.quarters['current']
        prev_year, prev_quarter = self.quarters['previous']
        
        # Banking intermediaries
        banking_query = f"""
        SELECT 
            'Banking' as sector,
            COUNT(DISTINCT ticker) as total_tickers,
            COUNT(DISTINCT CASE WHEN year = {prev_year} AND quarter = {prev_quarter} THEN ticker END) as prev_quarter,
            COUNT(DISTINCT CASE WHEN year = {current_year} AND quarter = {current_quarter} THEN ticker END) as current_quarter,
            ROUND(COUNT(DISTINCT CASE WHEN year = {current_year} AND quarter = {current_quarter} THEN ticker END) * 100.0 / 
                  COUNT(DISTINCT ticker), 1) as current_coverage_pct
        FROM intermediary_calculations_banking_cleaned
        WHERE (year = {current_year} AND quarter IN ({prev_quarter}, {current_quarter})) 
           OR (year = {prev_year} AND quarter = {prev_quarter})
        """
        
        # Securities intermediaries  
        securities_query = f"""
        SELECT 
            'Securities' as sector,
            COUNT(DISTINCT ticker) as total_tickers,
            COUNT(DISTINCT CASE WHEN year = {prev_year} AND quarter = {prev_quarter} THEN ticker END) as prev_quarter,
            COUNT(DISTINCT CASE WHEN year = {current_year} AND quarter = {current_quarter} THEN ticker END) as current_quarter,
            ROUND(COUNT(DISTINCT CASE WHEN year = {current_year} AND quarter = {current_quarter} THEN ticker END) * 100.0 / 
                  COUNT(DISTINCT ticker), 1) as current_coverage_pct
        FROM intermediary_calculations_securities_cleaned
        WHERE (year = {current_year} AND quarter IN ({prev_quarter}, {current_quarter})) 
           OR (year = {prev_year} AND quarter = {prev_quarter})
        """
        
        # Non-financial intermediaries by sector
        nonfin_query = f"""
        SELECT 
            m.sector,
            COUNT(DISTINCT m.ticker) as total_tickers,
            COUNT(DISTINCT CASE WHEN i.year = {prev_year} AND i.quarter = {prev_quarter} THEN i.ticker END) as prev_quarter,
            COUNT(DISTINCT CASE WHEN i.year = {current_year} AND i.quarter = {current_quarter} THEN i.ticker END) as current_quarter,
            ROUND(COUNT(DISTINCT CASE WHEN i.year = {current_year} AND i.quarter = {current_quarter} THEN i.ticker END) * 100.0 / 
                  COUNT(DISTINCT m.ticker), 1) as current_coverage_pct
        FROM master_info m
        LEFT JOIN intermediary_calculations_enhanced i ON m.ticker = i.ticker 
            AND ((i.year = {current_year} AND i.quarter IN ({prev_quarter}, {current_quarter})) 
                 OR (i.year = {prev_year} AND i.quarter = {prev_quarter}))
        WHERE m.sector NOT IN ('Banks', 'Securities')
        GROUP BY m.sector
        ORDER BY current_coverage_pct DESC, total_tickers DESC
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
        
        current_year, current_quarter = self.quarters['current']
        
        # Compare raw data vs intermediary processing
        gap_query = f"""
        SELECT 
            'Banking' as sector_type,
            (SELECT COUNT(DISTINCT fv.ticker) FROM fundamental_values fv 
             JOIN master_info m ON fv.ticker = m.ticker 
             WHERE m.sector = 'Banks' AND fv.year = {current_year} AND fv.quarter = {current_quarter}) as raw_current_tickers,
            (SELECT COUNT(DISTINCT ticker) FROM intermediary_calculations_banking_cleaned 
             WHERE year = {current_year} AND quarter = {current_quarter}) as processed_current_tickers
        
        UNION ALL
        
        SELECT 
            'Securities' as sector_type,
            (SELECT COUNT(DISTINCT fv.ticker) FROM fundamental_values fv 
             JOIN master_info m ON fv.ticker = m.ticker 
             WHERE m.sector = 'Securities' AND fv.year = {current_year} AND fv.quarter = {current_quarter}) as raw_current_tickers,
            (SELECT COUNT(DISTINCT ticker) FROM intermediary_calculations_securities_cleaned 
             WHERE year = {current_year} AND quarter = {current_quarter}) as processed_current_tickers
        
        UNION ALL
        
        SELECT 
            'Non-Financial' as sector_type,
            (SELECT COUNT(DISTINCT fv.ticker) FROM fundamental_values fv 
             JOIN master_info m ON fv.ticker = m.ticker 
             WHERE m.sector NOT IN ('Banks', 'Securities') AND fv.year = {current_year} AND fv.quarter = {current_quarter}) as raw_current_tickers,
            (SELECT COUNT(DISTINCT ticker) FROM intermediary_calculations_enhanced 
             WHERE year = {current_year} AND quarter = {current_quarter}) as processed_current_tickers
        """
        
        with self.db_connection.cursor() as cursor:
            cursor.execute(gap_query)
            gaps = pd.DataFrame(cursor.fetchall())
        
        # Calculate processing efficiency
        gaps['processing_gap'] = gaps['raw_current_tickers'] - gaps['processed_current_tickers']
        gaps['processing_efficiency_pct'] = round(gaps['processed_current_tickers'] * 100.0 / gaps['raw_current_tickers'], 1)
        
        return {
            'gap_analysis': gaps,
            'total_raw_current': int(gaps['raw_current_tickers'].sum()),
            'total_processed_current': int(gaps['processed_current_tickers'].sum()),
            'overall_efficiency': round(gaps['processed_current_tickers'].sum() * 100.0 / gaps['raw_current_tickers'].sum(), 1),
            'current_quarter': self.quarters['current_str']
        }
    
    def generate_recommendations(self, gaps: Dict[str, any]) -> List[str]:
        """Generate actionable recommendations based on processing gaps"""
        recommendations = []
        
        gap_df = gaps['gap_analysis']
        current_quarter = gaps['current_quarter']
        
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
            recommendations.append(f"✅ Processing pipeline is current for {current_quarter} - ready for factor generation")
        
        return recommendations
    
    def run_complete_status_check(self):
        """Run complete processing pipeline status check"""
        print("=" * 80)
        print("🔍 DYNAMIC PROCESSING PIPELINE STATUS MONITOR")
        print(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S ICT')}")
        print(f"🎯 Current Quarter: {self.quarters['current_str']} | Previous Quarter: {self.quarters['previous_str']}")
        print("=" * 80)
        
        # 1. Raw Data Status
        print("\n📊 1. RAW FUNDAMENTAL DATA STATUS (Recent Quarters)")
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
        
        if view_status['current_quarter_coverage']:
            cov = view_status['current_quarter_coverage']
            print(f"\n📈 {view_status['current_quarter']} Enhanced View Coverage:")
            print(f"   • Total tickers: {cov['current_quarter_tickers']:,}")
            print(f"   • With revenue data: {cov['tickers_with_revenue']:,}")
            print(f"   • With asset data: {cov['tickers_with_assets']:,}")
        
        # 3. Intermediary Status
        print("\n⚙️ 3. INTERMEDIARY CALCULATION STATUS")
        print("-" * 50)
        intermediary_status = self.check_intermediary_status()
        
        # Combine all sectors and rename columns for clarity
        all_sectors = pd.concat([
            intermediary_status['banking'],
            intermediary_status['securities'],
            intermediary_status['non_financial']
        ], ignore_index=True)
        
        # Rename columns to be more descriptive
        all_sectors = all_sectors.rename(columns={
            'prev_quarter': f"{self.quarters['previous_str']}",
            'current_quarter': f"{self.quarters['current_str']}",
            'current_coverage_pct': f"{self.quarters['current_str']} Coverage %"
        })
        
        print(all_sectors.to_string(index=False))
        
        # 4. Processing Gaps
        print("\n🔍 4. PROCESSING GAP ANALYSIS")
        print("-" * 50)
        gaps = self.check_processing_gaps()
        gap_df = gaps['gap_analysis'].copy()
        
        # Rename columns for clarity
        gap_df = gap_df.rename(columns={
            'raw_current_tickers': f"Raw {gaps['current_quarter']}",
            'processed_current_tickers': f"Processed {gaps['current_quarter']}",
            'processing_efficiency_pct': f"{gaps['current_quarter']} Efficiency %"
        })
        
        print(gap_df.to_string(index=False))
        
        print(f"\n📊 Overall Processing Efficiency: {gaps['overall_efficiency']}%")
        print(f"   • Raw {gaps['current_quarter']} data: {gaps['total_raw_current']:,} companies")
        print(f"   • Processed {gaps['current_quarter']}: {gaps['total_processed_current']:,} companies")
        print(f"   • Processing gap: {gaps['total_raw_current'] - gaps['total_processed_current']} companies")
        
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
        monitor = DynamicProcessingPipelineMonitor()
        monitor.run_complete_status_check()
    except Exception as e:
        print(f"❌ Error running dynamic processing pipeline status check: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()