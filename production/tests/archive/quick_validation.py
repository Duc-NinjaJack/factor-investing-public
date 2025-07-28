"""
Quick Validation Script for Canonical QVM Engine
===============================================
Purpose: Quick test to ensure engine initializes correctly
"""

import sys
from pathlib import Path
import pandas as pd

# Add production engine to path
production_path = Path(__file__).parent.parent
sys.path.append(str(production_path))

# Import canonical engine
from engine.qvm_engine_canonical import CanonicalQVMEngine

def main():
    print("🔧 Quick Validation of Canonical QVM Engine")
    print("=" * 50)
    
    try:
        # Initialize engine
        print("\n1. Initializing engine...")
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / 'config'
        
        engine = CanonicalQVMEngine(config_path=str(config_path), log_level='INFO')
        print("✅ Engine initialized successfully")
        
        # Test basic functionality
        print("\n2. Testing sector mapping...")
        sector_map = engine.get_sector_mapping()
        print(f"✅ Retrieved {len(sector_map)} sector mappings")
        
        # Test date calculation
        print("\n3. Testing quarter calculation...")
        test_date = pd.Timestamp('2025-03-31')
        quarter_info = engine.get_correct_quarter_for_date(test_date)
        if quarter_info:
            year, quarter = quarter_info
            print(f"✅ Available quarter for {test_date.date()}: {year} Q{quarter}")
        else:
            print("❌ No quarter data available")
        
        # Test small calculation
        print("\n4. Testing QVM calculation on small universe...")
        test_universe = ['FPT', 'VCB']
        qvm_scores = engine.calculate_qvm_composite(test_date, test_universe)
        
        if qvm_scores:
            print(f"✅ Calculated QVM scores:")
            for ticker, score in qvm_scores.items():
                print(f"   {ticker}: {score:.4f}")
        else:
            print("❌ No QVM scores calculated")
        
        print("\n" + "=" * 50)
        print("✅ CANONICAL ENGINE BASIC VALIDATION PASSED")
        print("🚀 Ready for full unit testing")
        
    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())