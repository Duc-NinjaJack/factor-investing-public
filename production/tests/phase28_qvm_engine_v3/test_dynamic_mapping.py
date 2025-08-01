#!/usr/bin/env python3
"""
Test Dynamic Mapping System
===========================

This script tests the dynamic mapping system integration with the QVM strategy.
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
from production.database.mappings.financial_mapping_manager import FinancialMappingManager

def test_dynamic_mapping():
    """Test the dynamic mapping system."""
    print("🔍 Testing Dynamic Mapping System")
    print("=" * 50)
    
    # Initialize mapping manager
    mapping_manager = FinancialMappingManager()
    
    # Test mapping manager
    print("\n📋 Mapping Manager Test:")
    print("-" * 30)
    mapping_manager.print_mappings_summary()
    
    # Test validation
    validation_results = mapping_manager.validate_mappings()
    print(f"\n🔍 Validation Results:")
    print(f"Errors: {validation_results['errors']}")
    print(f"Warnings: {validation_results['warnings']}")
    
    # Test getting mappings for different sectors
    print("\n🧪 Sector-Specific Mapping Tests:")
    print("-" * 40)
    
    test_sectors = [
        ('VCB', 'Banks'),
        ('VNM', 'Food & Beverage'),
        ('TCB', 'Banks'),
        ('HPG', 'Materials')
    ]
    
    for ticker, sector in test_sectors:
        print(f"\n📊 {ticker} ({sector}):")
        
        try:
            netprofit_id, netprofit_stmt = mapping_manager.get_net_profit_mapping(sector)
            revenue_id, revenue_stmt = mapping_manager.get_revenue_mapping(sector)
            totalassets_id, totalassets_stmt = mapping_manager.get_total_assets_mapping(sector)
            
            print(f"   NetProfit: Item {netprofit_id} ({netprofit_stmt})")
            print(f"   Revenue: Item {revenue_id} ({revenue_stmt})")
            print(f"   TotalAssets: Item {totalassets_id} ({totalassets_stmt})")
            
            # Test the actual SQL query with these mappings
            test_sql_query(ticker, sector, mapping_manager)
            
        except Exception as e:
            print(f"   ❌ Error: {e}")

def test_sql_query(ticker: str, sector: str, mapping_manager: FinancialMappingManager):
    """Test the SQL query with dynamic mappings."""
    try:
        # Connect to database
        db_manager = get_database_manager()
        engine = db_manager.get_engine()
        
        # Get dynamic mappings
        netprofit_id, netprofit_stmt = mapping_manager.get_net_profit_mapping(sector)
        revenue_corp_id, revenue_corp_stmt = mapping_manager.get_revenue_mapping('Corporate')
        revenue_bank_id, revenue_bank_stmt = mapping_manager.get_revenue_mapping('Banks')
        totalassets_id, totalassets_stmt = mapping_manager.get_total_assets_mapping(sector)
        
        # Test date
        test_date = pd.Timestamp('2025-01-29')
        lag_days = 45
        lag_date = test_date - pd.Timedelta(days=lag_days)
        lag_year = lag_date.year
        lag_quarter = ((lag_date.month - 1) // 3) + 1
        
        # Build the SQL query (similar to the one in the strategy)
        fundamental_query = text(f"""
            WITH quarterly_data AS (
                SELECT 
                    fv.ticker,
                    fv.year,
                    fv.quarter,
                    fv.value,
                    fv.item_id,
                    fv.statement_type,
                    mi.sector
                FROM fundamental_values fv
                LEFT JOIN master_info mi ON fv.ticker = mi.ticker
                WHERE (fv.item_id = :netprofit_id AND fv.statement_type = :netprofit_stmt)
                   OR (fv.item_id IN (:revenue_corp_id, :revenue_bank_id) AND fv.statement_type = 'PL')
                   OR (fv.item_id = :totalassets_id AND fv.statement_type = :totalassets_stmt)
                AND fv.ticker = :ticker
                AND (fv.year < :lag_year OR (fv.year = :lag_year AND fv.quarter <= :lag_quarter))
            ),
            ttm_calculations AS (
                SELECT 
                    ticker,
                    year,
                    quarter,
                    sector,
                    SUM(CASE WHEN item_id = :netprofit_id THEN value ELSE 0 END) as netprofit_ttm,
                    SUM(CASE WHEN item_id = :revenue_corp_id THEN value ELSE 0 END) as revenue_net_sales_ttm,
                    SUM(CASE WHEN item_id = :revenue_bank_id THEN value ELSE 0 END) as revenue_total_income_ttm,
                    SUM(CASE WHEN item_id = :totalassets_id THEN value ELSE 0 END) as totalassets_ttm
                FROM (
                    SELECT 
                        ticker,
                        year,
                        quarter,
                        item_id,
                        value,
                        sector,
                        ROW_NUMBER() OVER (PARTITION BY ticker, item_id ORDER BY year DESC, quarter DESC) as rn
                    FROM quarterly_data
                ) ranked
                WHERE rn <= 4
                GROUP BY ticker, year, quarter, sector
            ),
            latest_data AS (
                SELECT 
                    ttm.ticker,
                    ttm.sector,
                    ttm.year,
                    ttm.quarter,
                    ttm.netprofit_ttm,
                    CASE 
                        WHEN ttm.sector IS NOT NULL AND LOWER(ttm.sector) LIKE '%bank%' THEN ttm.revenue_total_income_ttm
                        ELSE ttm.revenue_net_sales_ttm
                    END as revenue_ttm,
                    ttm.totalassets_ttm,
                    CASE 
                        WHEN ttm.totalassets_ttm > 0 THEN ttm.netprofit_ttm / ttm.totalassets_ttm 
                        ELSE NULL 
                    END as roaa,
                    CASE 
                        WHEN (CASE 
                            WHEN ttm.sector IS NOT NULL AND LOWER(ttm.sector) LIKE '%bank%' THEN ttm.revenue_total_income_ttm
                            ELSE ttm.revenue_net_sales_ttm
                        END) > 0 THEN ttm.netprofit_ttm / (CASE 
                            WHEN ttm.sector IS NOT NULL AND LOWER(ttm.sector) LIKE '%bank%' THEN ttm.revenue_total_income_ttm
                            ELSE ttm.revenue_net_sales_ttm
                        END)
                        ELSE NULL 
                    END as net_margin,
                    CASE 
                        WHEN ttm.totalassets_ttm > 0 THEN (CASE 
                            WHEN ttm.sector IS NOT NULL AND LOWER(ttm.sector) LIKE '%bank%' THEN ttm.revenue_total_income_ttm
                            ELSE ttm.revenue_net_sales_ttm
                        END) / ttm.totalassets_ttm 
                        ELSE NULL 
                    END as asset_turnover,
                    ROW_NUMBER() OVER (PARTITION BY ttm.ticker ORDER BY ttm.year DESC, ttm.quarter DESC) as rn
                FROM ttm_calculations ttm
                WHERE ttm.netprofit_ttm > 0 AND ttm.totalassets_ttm > 0
                AND (
                    (ttm.sector IS NOT NULL AND LOWER(ttm.sector) LIKE '%bank%' AND ttm.revenue_total_income_ttm > 0)
                    OR (ttm.sector IS NULL OR LOWER(ttm.sector) NOT LIKE '%bank%') AND ttm.revenue_net_sales_ttm > 0
                )
                AND (ttm.year < :lag_year OR (ttm.year = :lag_year AND ttm.quarter <= :lag_quarter))
            )
            SELECT 
                ticker,
                sector,
                roaa,
                net_margin,
                asset_turnover
            FROM latest_data
            WHERE rn = 1
        """)
        
        # Create parameter dictionary
        params_dict = {
            'netprofit_id': netprofit_id,
            'netprofit_stmt': netprofit_stmt,
            'revenue_corp_id': revenue_corp_id,
            'revenue_bank_id': revenue_bank_id,
            'totalassets_id': totalassets_id,
            'totalassets_stmt': totalassets_stmt,
            'lag_year': lag_year,
            'lag_quarter': lag_quarter,
            'ticker': ticker
        }
        
        # Execute query
        result = pd.read_sql(fundamental_query, engine, params=params_dict)
        
        if not result.empty:
            row = result.iloc[0]
            print(f"   ✅ Query successful!")
            print(f"   ROAA: {row['roaa']:.4f} ({row['roaa']*100:.2f}%)")
            print(f"   Net Margin: {row['net_margin']:.4f} ({row['net_margin']*100:.2f}%)")
            print(f"   Asset Turnover: {row['asset_turnover']:.4f} ({row['asset_turnover']*100:.2f}%)")
        else:
            print(f"   ⚠️ No data found")
            
    except Exception as e:
        print(f"   ❌ SQL Error: {e}")

if __name__ == "__main__":
    test_dynamic_mapping() 