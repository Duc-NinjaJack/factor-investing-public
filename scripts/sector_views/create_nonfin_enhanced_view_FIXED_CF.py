#!/usr/bin/env python3
"""
CASH FLOW FIX - Test Script
==========================
Tests corrected CF mappings using Vietnamese aggregated totals instead of section headers
"""

import sys
import pandas as pd
from sqlalchemy import create_engine, text
import yaml
from datetime import datetime

sys.path.append('src')

def test_cf_fix(engine):
    """Test cash flow fix by creating temporary view"""
    
    print("🧪 TESTING CASH FLOW FIX")
    print("=" * 50)
    
    # Test query - just get CF values for FPT before and after
    print("🔍 BEFORE FIX - Current CF values for FPT Q1 2025:")
    
    current_cf_query = """
    SELECT NetCFO, NetCFI, NetCFF, CapEx, DividendsPaid
    FROM v_comprehensive_fundamental_items
    WHERE ticker = 'FPT' AND year = 2025 AND quarter = 1
    """
    
    df_before = pd.read_sql(current_cf_query, engine)
    if not df_before.empty:
        print(f"   - NetCFO: {df_before['NetCFO'].iloc[0] if pd.notna(df_before['NetCFO'].iloc[0]) else 'NULL'}")
        print(f"   - NetCFI: {df_before['NetCFI'].iloc[0] if pd.notna(df_before['NetCFI'].iloc[0]) else 'NULL'}")
        print(f"   - NetCFF: {df_before['NetCFF'].iloc[0] if pd.notna(df_before['NetCFF'].iloc[0]) else 'NULL'}")
    print()
    
    # Create test view with fixed mappings
    print("🔧 Creating test view with FIXED CF mappings...")
    
    create_test_view_query = """
    CREATE OR REPLACE VIEW v_test_cf_fix AS
    SELECT 
        fv.ticker,
        fv.year,
        fv.quarter,
        
        -- Test the FIXED cash flow mappings
        MAX(CASE WHEN fm.vn_name = 'Lưu chuyển tiền thuần từ hoạt động kinh doanh' THEN fv.value END) as NetCFO_Fixed,
        MAX(CASE WHEN fm.vn_name = 'Lưu chuyển tiền thuần từ hoạt động đầu tư' THEN fv.value END) as NetCFI_Fixed,
        MAX(CASE WHEN fm.vn_name = 'Lưu chuyển tiền thuần từ hoạt động tài chính' THEN fv.value END) as NetCFF_Fixed,
        
        -- Compare with current (broken) mappings
        MAX(CASE WHEN fm.en_name = 'I. Net Cash Flow from Operating' THEN fv.value END) as NetCFO_Current,
        MAX(CASE WHEN fm.en_name = 'II. Net Cash Flow from Investing' THEN fv.value END) as NetCFI_Current,
        MAX(CASE WHEN fm.en_name = 'III. Net Cash Flow from Financing' THEN fv.value END) as NetCFF_Current,
        
        -- Working items (should be same)
        MAX(CASE WHEN fm.en_name = '1. Acquisition of Fixed/Long-term Assets' THEN fv.value END) as CapEx,
        MAX(CASE WHEN fm.en_name = '8. Dividends, Profit Paid to Owners' THEN fv.value END) as DividendsPaid
        
    FROM fundamental_values fv
    JOIN fs_mappings fm ON fv.item_id = fm.item_id 
        AND fv.statement_type = fm.statement_type
    JOIN master_info mi ON fv.ticker = mi.ticker
    JOIN sector_display_to_fs sdf ON mi.sector = sdf.display_sector
    WHERE fm.sector = sdf.fs_sector
    GROUP BY fv.ticker, fv.year, fv.quarter;
    """
    
    with engine.connect() as conn:
        conn.execute(text(create_test_view_query))
        conn.commit()
    
    print("✅ Test view created")
    print()
    
    # Test the fix
    print("🔍 TESTING RESULTS - FPT Q1 2025:")
    
    test_query = """
    SELECT 
        ticker, year, quarter,
        NetCFO_Fixed, NetCFI_Fixed, NetCFF_Fixed,
        NetCFO_Current, NetCFI_Current, NetCFF_Current,
        CapEx, DividendsPaid
    FROM v_test_cf_fix
    WHERE ticker = 'FPT' AND year = 2025 AND quarter = 1
    """
    
    df_test = pd.read_sql(test_query, engine)
    
    if not df_test.empty:
        row = df_test.iloc[0]
        print("   📊 FIXED MAPPINGS (Vietnamese aggregated totals):")
        def safe_format(value):
            return f"{value:,.0f}" if pd.notna(value) and value != 0 else str(value) if pd.notna(value) else 'NULL'
        
        print(f"      - NetCFO_Fixed: {safe_format(row['NetCFO_Fixed'])}")
        print(f"      - NetCFI_Fixed: {safe_format(row['NetCFI_Fixed'])}")
        print(f"      - NetCFF_Fixed: {safe_format(row['NetCFF_Fixed'])}")
        print()
        print("   ❌ CURRENT MAPPINGS (English section headers):")
        print(f"      - NetCFO_Current: {safe_format(row['NetCFO_Current'])}")
        print(f"      - NetCFI_Current: {safe_format(row['NetCFI_Current'])}")
        print(f"      - NetCFF_Current: {safe_format(row['NetCFF_Current'])}")
        print()
        print("   ✅ WORKING ITEMS (Should be same):")
        print(f"      - CapEx: {safe_format(row['CapEx'])}")
        print(f"      - DividendsPaid: {safe_format(row['DividendsPaid'])}")
        print()
        
        # Determine if fix worked
        cf_fixed_has_data = any(pd.notna(row[col]) and row[col] != 0 for col in ['NetCFO_Fixed', 'NetCFI_Fixed', 'NetCFF_Fixed'])
        cf_current_has_data = any(pd.notna(row[col]) and row[col] != 0 for col in ['NetCFO_Current', 'NetCFI_Current', 'NetCFF_Current'])
        
        print("=" * 50)
        if cf_fixed_has_data and not cf_current_has_data:
            print("✅ FIX SUCCESSFUL! Vietnamese mappings have data, English headers don't")
            print("🎯 Ready to apply fix to production view")
        elif cf_fixed_has_data and cf_current_has_data:
            print("⚠️ Both mappings have data - need to investigate")
        else:
            print("❌ Fix not working - Vietnamese mappings still empty")
        print("=" * 50)
    
    # Clean up test view
    with engine.connect() as conn:
        conn.execute(text("DROP VIEW IF EXISTS v_test_cf_fix"))
        conn.commit()
    
    print("🧹 Cleaned up test view")

def load_config():
    """Load database configuration"""
    with open('config/database.yml', 'r') as f:
        config = yaml.safe_load(f)
    return config['production']

def main():
    """Main execution"""
    print("CASH FLOW FIX - TEST SCRIPT")
    print("=" * 40)
    
    # Load config and create engine
    config = load_config()
    engine = create_engine(f"mysql+pymysql://{config['username']}:{config['password']}@{config['host']}/{config['schema_name']}")
    
    # Test the cash flow fix
    test_cf_fix(engine)

if __name__ == "__main__":
    main()