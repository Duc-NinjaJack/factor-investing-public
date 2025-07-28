#!/usr/bin/env python3
"""
Test script for the universe constructor module.
This script validates that the batch processing approach works correctly.
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Now import our modules
from production.universe import get_liquid_universe_dataframe
from production.universe.constructors import validate_universe_construction
import yaml
from sqlalchemy import create_engine


def create_db_connection():
    """Create database connection using config file."""
    config_path = project_root / 'config' / 'database.yml'
    
    with open(config_path, 'r') as f:
        db_config = yaml.safe_load(f)
    
    conn_params = db_config['production']
    connection_string = (
        f"mysql+pymysql://{conn_params['username']}:{conn_params['password']}"
        f"@{conn_params['host']}/{conn_params['schema_name']}"
    )
    
    engine = create_engine(connection_string, pool_pre_ping=True)
    return engine


def main():
    print("🧪 Testing Universe Constructor Module")
    print("=" * 50)
    
    # Create database connection
    engine = create_db_connection()
    
    # Test with Q1 2024 date
    test_date = pd.Timestamp('2024-03-29')
    
    try:
        # Test with lower threshold first to see if we get any results
        print(f"\n1️⃣ Testing universe construction for {test_date.date()}")
        print("    First testing with lower ADTV threshold (1B VND)...")
        
        test_config = {
            'adtv_threshold_bn': 1.0,     # Lower threshold for testing
            'top_n': 50,                   # Smaller universe for testing
            'min_trading_coverage': 0.6    # Lower coverage requirement (60% instead of 80%)
        }
        
        universe_df = get_liquid_universe_dataframe(test_date, engine, test_config)
        
        print(f"\n2️⃣ Universe construction results:")
        print(f"   Total stocks: {len(universe_df)}")
        
        if len(universe_df) > 0:
            print(f"   ADTV range: {universe_df['adtv_bn_vnd'].min():.1f}B - {universe_df['adtv_bn_vnd'].max():.1f}B VND")
            print(f"   Sectors: {universe_df['sector'].nunique()}")
            
            print(f"\n📊 Top 10 most liquid stocks:")
            display_cols = ['ticker', 'sector', 'adtv_bn_vnd', 'avg_market_cap_bn', 'trading_days']
            print(universe_df[display_cols].head(10).to_string(index=False))
            
            print(f"\n🏢 Sector composition:")
            sector_summary = universe_df.groupby('sector').agg({
                'ticker': 'count',
                'adtv_bn_vnd': 'sum'
            }).round(1).sort_values('ticker', ascending=False)
            sector_summary.columns = ['Count', 'Total_ADTV_Bn']
            print(sector_summary.head(10).to_string())
        
        # Test validation
        print(f"\n3️⃣ Testing universe validation:")
        tickers = universe_df['ticker'].tolist()
        validation = validate_universe_construction(tickers, test_date)
        
        print(f"   Overall validation: {'✅ PASS' if validation['is_valid'] else '❌ FAIL'}")
        for check_name, check_result in validation['checks'].items():
            print(f"   {check_name}: {'✅' if check_result['pass'] else '❌'} ({check_result})")
        
        if validation['errors']:
            for error in validation['errors']:
                print(f"   ❌ {error}")
        if validation['warnings']:
            for warning in validation['warnings']:
                print(f"   ⚠️  {warning}")
        
        print(f"\n✅ Universe constructor test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()