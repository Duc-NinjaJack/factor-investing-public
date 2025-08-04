#!/usr/bin/env python3
"""
Verification script to check if you're using the fixed QVM Engine file.
"""

import os

def check_file_status():
    """Check which QVM Engine files exist and their status."""
    
    base_dir = "production/tests/phase29-alpha_demo"
    
    files_to_check = [
        "08_integrated_strategy_with_validated_factors.py",
        "08_integrated_strategy_with_validated_factors.ipynb", 
        "08_integrated_strategy_with_validated_factors_fixed.py",
        "08_integrated_strategy_with_validated_factors_fixed.ipynb"
    ]
    
    print("🔍 QVM Engine File Status Check")
    print("=" * 50)
    
    for file in files_to_check:
        filepath = os.path.join(base_dir, file)
        if os.path.exists(filepath):
            # Check if it's the fixed version
            if "fixed" in file:
                print(f"✅ {file} - FIXED VERSION (Use this!)")
            else:
                print(f"❌ {file} - ORIGINAL VERSION (Has shape mismatch error)")
        else:
            print(f"⚠️  {file} - NOT FOUND")
    
    print("\n" + "=" * 50)
    print("📋 RECOMMENDATION:")
    print("   Use the files with '_fixed' in the name to avoid the shape mismatch error.")
    print("   The original files will give you the ValueError you encountered.")
    
    # Check for the specific fix in the fixed file
    fixed_py = os.path.join(base_dir, "08_integrated_strategy_with_validated_factors_fixed.py")
    if os.path.exists(fixed_py):
        with open(fixed_py, 'r') as f:
            content = f.read()
            if "weights_df = pd.DataFrame" in content:
                print("✅ Fixed file contains the shape mismatch fix!")
            else:
                print("❌ Fixed file may not contain the fix - please check!")

if __name__ == "__main__":
    check_file_status() 