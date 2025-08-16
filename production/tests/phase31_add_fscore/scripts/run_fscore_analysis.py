#!/usr/bin/env python3
"""
F-Score Integration Analysis Runner
==================================
Simple script to run the F-Score integration analysis with proper setup.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

def main():
    """Run the F-Score integration analysis."""
    print("🚀 Starting F-Score Integration Analysis...")
    print("=" * 60)
    
    try:
        # Change to the phase31_add_fscore directory
        phase_dir = Path(__file__).parent.parent
        os.chdir(phase_dir)
        
        print(f"📁 Working directory: {os.getcwd()}")
        
        # Run the main analysis file
        analysis_file = "01_tearsheet_fscore_integration.py"
        
        if not Path(analysis_file).exists():
            print(f"❌ Analysis file not found: {analysis_file}")
            return 1
        
        print(f"📊 Running analysis: {analysis_file}")
        
        # Execute the analysis file
        with open(analysis_file, 'r') as f:
            exec(f.read())
        
        print("✅ Analysis completed successfully!")
        print("📊 Check the docs/ folder for output files")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        print("   Check the error details above")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
