#!/usr/bin/env python3
"""
File Organization Script for Phase 28 QVM Engine v3
==================================================

This script organizes the phase28_qvm_engine_v3 directory by:
1. Creating logical subdirectories
2. Moving files to appropriate locations
3. Creating symbolic links for easy access
4. Generating documentation
"""

import os
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

class Phase28Organizer:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def create_directory_structure(self):
        """Create the new directory structure."""
        
        directories = {
            'core_engine': [
                '28_qvm_engine_v3c.ipynb',
                'qvm_engine_v3_adopted_insights.py',
                'qvm_engine_v3_fixed.py',
                'run_qvm_engine_v3_adopted_insights_backtest.py'
            ],
            
            'validation_testing': [
                '01_walkforward_validation_2016.py',
                '01_walkforward_validation_2016.ipynb',
                '02_lag_sensitivity_analysis.py',
                '02_lag_sensitivity_analysis.ipynb',
                '03_min_adtv_10b_vnd.py',
                '03_min_adtv_10b_vnd.ipynb',
                '04_composite_vs_single_factors.py',
                '04_composite_vs_single_factors.ipynb',
                'test_qvm_engine_v3_adopted_insights.py'
            ],
            
            'regime_detection': [
                'regime_detection_diagnostic.py',
                'regime_detection_fix.py',
                'test_regime_detection.py',
                'test_regime_thresholds.py',
                'test_optimal_thresholds.py',
                'debug_regime_detection.py'
            ],
            
            'data_quality': [
                'debug_fundamental_data.py',
                'investigate_fundamental_data_quality.py',
                'check_fundamental_values_structure.py',
                'comprehensive_financial_analysis.py',
                'investigate_ttm_calculation.py'
            ],
            
            'factor_analysis': [
                'single_factor_strategies.py',
                'single_factors.py',
                'single_factor_cell.py',
                'add_single_factors.py',
                'test_momentum_signals.py',
                'README_SINGLE_FACTORS.md'
            ],
            
            'database_mapping': [
                'map_vcsc_items_to_database.py',
                'test_dynamic_mapping.py',
                'find_correct_item_ids.py',
                'find_correct_item_ids_with_mappings.py',
                'test_corrected_item_ids.py'
            ],
            
            'debugging_tools': [
                'debug_balance_sheet.py',
                'debug_factor_data.py',
                'test_fixed_query.py',
                'test_simple_ttm.py',
                'final_ratio_test.py',
                'simple_strategy_test.py'
            ],
            
            'financial_analysis': [
                'find_vnm_sales_components.py',
                'find_vcb_conversion_factor.py',
                'find_conversion_factor.py',
                'find_correct_vnm_revenue.py',
                'test_asset_turnover_fix.py',
                'find_correct_total_assets.py',
                'find_correct_vcb_mapping.py',
                'test_correct_revenue_item.py',
                'check_item_302_availability.py',
                'test_final_fundamental_fix.py',
                'debug_fundamental_query.py',
                'test_corrected_fundamental_data.py',
                'test_fundamental_query_fix.py',
                'find_correct_vnm_mapping.py',
                'check_item_ids.py',
                'test_corrected_item_ids.py',
                'check_volume_data.py',
                'test_universe_construction.py',
                'test_updated_strategy.py',
                'identify_fundamental_item_ids.py'
            ],
            
            'documentation': [
                'BACKTEST_SUMMARY.md',
                'QVM_ENGINE_V3_ADOPTED_INSIGHTS_SUMMARY.md',
                'notebook_templates.md'
            ],
            
            'legacy': [
                'lag_sensitivity_analysis.ipynb',
                'walkforward_validation_2016.ipynb',
                'fix_adtv_threshold.py',
                'fix_notebook.py'
            ]
        }
        
        # Create directories and move files
        for dir_name, files in directories.items():
            dir_path = self.base_dir / dir_name
            dir_path.mkdir(exist_ok=True)
            
            print(f"📁 Created directory: {dir_name}")
            
            for file_name in files:
                source_path = self.base_dir / file_name
                dest_path = dir_path / file_name
                
                if source_path.exists():
                    try:
                        shutil.move(str(source_path), str(dest_path))
                        print(f"   ✅ Moved: {file_name}")
                    except Exception as e:
                        print(f"   ❌ Error moving {file_name}: {e}")
                else:
                    print(f"   ⚠️  File not found: {file_name}")
    
    def create_index_files(self):
        """Create index files for each directory."""
        
        index_content = {
            'core_engine': """# Core Engine Files

This directory contains the main QVM Engine v3 implementation files.

## Files
- `28_qvm_engine_v3c.ipynb` - Main production notebook
- `qvm_engine_v3_adopted_insights.py` - Core engine implementation
- `qvm_engine_v3_fixed.py` - Fixed version with improvements
- `run_qvm_engine_v3_adopted_insights_backtest.py` - Execution script

## Usage
Run the main notebook for production backtesting:
```bash
jupyter notebook 28_qvm_engine_v3c.ipynb
```
""",
            
            'validation_testing': """# Validation & Testing

This directory contains validation and testing frameworks for the QVM Engine.

## Files
- `01_walkforward_validation_2016.py` - Walk-forward analysis
- `02_lag_sensitivity_analysis.py` - Lag parameter sensitivity
- `03_min_adtv_10b_vnd.py` - Liquidity threshold testing
- `04_composite_vs_single_factors.py` - Factor combination analysis
- `test_qvm_engine_v3_adopted_insights.py` - Comprehensive testing

## Usage
Run validation tests to ensure system robustness:
```bash
python 01_walkforward_validation_2016.py
```
""",
            
            'regime_detection': """# Regime Detection

This directory contains regime detection implementations and analysis.

## Files
- `regime_detection_diagnostic.py` - Regime detection analysis
- `regime_detection_fix.py` - Fixed regime detection
- `test_regime_detection.py` - Regime testing
- `test_regime_thresholds.py` - Threshold optimization
- `test_optimal_thresholds.py` - Optimal parameter search

## Key Insights
- Current approach uses hard-coded thresholds
- Robust alternatives include percentile-based and ensemble methods
- See `insights/robust_regime_detection.md` for detailed analysis
""",
            
            'data_quality': """# Data Quality & Analysis

This directory contains data quality analysis and debugging tools.

## Files
- `debug_fundamental_data.py` - Fundamental data debugging
- `investigate_fundamental_data_quality.py` - Data quality analysis
- `check_fundamental_values_structure.py` - Database structure validation
- `comprehensive_financial_analysis.py` - Financial data analysis
- `investigate_ttm_calculation.py` - TTM calculation analysis

## Purpose
Ensure data quality and identify potential issues in the data pipeline.
""",
            
            'factor_analysis': """# Factor Analysis

This directory contains factor analysis and testing frameworks.

## Files
- `single_factor_strategies.py` - Individual factor testing
- `single_factors.py` - Factor implementation
- `test_momentum_signals.py` - Momentum factor testing
- `README_SINGLE_FACTORS.md` - Factor documentation

## Key Factors
- Value: ROAA, P/E ratios
- Quality: ROAA quintiles by sector
- Momentum: Multi-horizon (1M, 3M, 6M, 12M)
""",
            
            'database_mapping': """# Database & Mapping

This directory contains database mapping and item identification tools.

## Files
- `map_vcsc_items_to_database.py` - Item mapping
- `test_dynamic_mapping.py` - Dynamic mapping testing
- `find_correct_item_ids.py` - Item ID identification
- `find_correct_item_ids_with_mappings.py` - Enhanced item ID search

## Purpose
Map financial statement items to database structure and validate mappings.
""",
            
            'debugging_tools': """# Debugging Tools

This directory contains various debugging and testing utilities.

## Files
- `debug_balance_sheet.py` - Balance sheet debugging
- `debug_factor_data.py` - Factor data debugging
- `test_fixed_query.py` - Query testing
- `test_simple_ttm.py` - TTM testing
- `final_ratio_test.py` - Ratio testing
- `simple_strategy_test.py` - Strategy testing

## Usage
Use these tools to debug specific components of the system.
""",
            
            'financial_analysis': """# Financial Analysis

This directory contains financial analysis and data investigation tools.

## Files
- Various `find_*` scripts for investigating specific financial data
- `test_*` scripts for validating financial calculations
- `check_*` scripts for data quality checks

## Purpose
Investigate and validate financial data quality and calculations.
""",
            
            'documentation': """# Documentation

This directory contains documentation and summary files.

## Files
- `BACKTEST_SUMMARY.md` - Backtest results summary
- `QVM_ENGINE_V3_ADOPTED_INSIGHTS_SUMMARY.md` - Engine summary
- `notebook_templates.md` - Notebook templates

## Usage
Reference these documents for system understanding and implementation.
""",
            
            'legacy': """# Legacy Files

This directory contains legacy and temporary files.

## Files
- Various legacy notebooks and scripts
- Temporary debugging files
- Fix scripts

## Note
These files are kept for reference but may not be actively maintained.
"""
        }
        
        for dir_name, content in index_content.items():
            index_path = self.base_dir / dir_name / 'README.md'
            with open(index_path, 'w') as f:
                f.write(content)
            print(f"📝 Created index: {dir_name}/README.md")
    
    def create_symlinks(self):
        """Create symbolic links for easy access to key files."""
        
        key_files = {
            'main_notebook': 'core_engine/28_qvm_engine_v3c.ipynb',
            'core_engine': 'core_engine/qvm_engine_v3_adopted_insights.py',
            'regime_analysis': 'insights/robust_regime_detection.md',
            'backtest_summary': 'documentation/BACKTEST_SUMMARY.md'
        }
        
        for link_name, target_path in key_files.items():
            target = self.base_dir / target_path
            link_path = self.base_dir / f"{link_name}.md"
            
            if target.exists():
                try:
                    # Create relative symlink
                    relative_target = os.path.relpath(target, self.base_dir)
                    os.symlink(relative_target, link_path)
                    print(f"🔗 Created symlink: {link_name} -> {target_path}")
                except Exception as e:
                    print(f"❌ Error creating symlink {link_name}: {e}")
    
    def generate_file_index(self):
        """Generate a comprehensive file index."""
        
        index_content = """# Phase 28 File Index

## Quick Navigation

### 🔧 Core Engine
- [Main Notebook](core_engine/28_qvm_engine_v3c.ipynb) - Production backtesting
- [Core Engine](core_engine/qvm_engine_v3_adopted_insights.py) - Main implementation
- [Fixed Version](core_engine/qvm_engine_v3_fixed.py) - Improved version

### 🧪 Validation & Testing
- [Walk-forward Analysis](validation_testing/01_walkforward_validation_2016.py)
- [Lag Sensitivity](validation_testing/02_lag_sensitivity_analysis.py)
- [Liquidity Testing](validation_testing/03_min_adtv_10b_vnd.py)
- [Factor Analysis](validation_testing/04_composite_vs_single_factors.py)

### 📊 Regime Detection
- [Regime Analysis](regime_detection/regime_detection_diagnostic.py)
- [Regime Testing](regime_detection/test_regime_detection.py)
- [Threshold Optimization](regime_detection/test_regime_thresholds.py)

### 📈 Factor Analysis
- [Single Factors](factor_analysis/single_factor_strategies.py)
- [Factor Implementation](factor_analysis/single_factors.py)
- [Momentum Testing](factor_analysis/test_momentum_signals.py)

### 🔍 Data Quality
- [Data Debugging](data_quality/debug_fundamental_data.py)
- [Quality Analysis](data_quality/investigate_fundamental_data_quality.py)
- [Structure Validation](data_quality/check_fundamental_values_structure.py)

### 📚 Documentation
- [Backtest Summary](documentation/BACKTEST_SUMMARY.md)
- [Engine Summary](documentation/QVM_ENGINE_V3_ADOPTED_INSIGHTS_SUMMARY.md)
- [Robust Regime Detection](insights/robust_regime_detection.md)

## Directory Structure

```
phase28_qvm_engine_v3/
├── core_engine/           # Main engine files
├── validation_testing/    # Testing frameworks
├── regime_detection/      # Regime analysis
├── data_quality/         # Data quality tools
├── factor_analysis/      # Factor testing
├── database_mapping/     # Database tools
├── debugging_tools/      # Debug utilities
├── financial_analysis/   # Financial analysis
├── documentation/        # Documentation
├── legacy/              # Legacy files
├── insights/            # Research insights
├── scripts/             # Utility scripts
└── archive/             # Previous versions
```

## Key Insights

### Robust Regime Detection
- Current approach uses hard-coded thresholds
- Robust alternatives: percentile-based, ensemble, HMM
- See [robust_regime_detection.md](insights/robust_regime_detection.md)

### Performance Characteristics
- Expected Returns: 10-15% annual (regime-dependent)
- Volatility: 15-20% annual
- Sharpe Ratio: 0.5-0.7
- Max Drawdown: 15-25%

### Implementation Status
- ✅ Core engine implemented
- ✅ Basic regime detection working
- ✅ Factor framework established
- 🔄 Robust regime detection (in progress)
- 🔄 Comprehensive validation (in progress)

## Next Steps

1. **Implement robust regime detection** using percentile-based thresholds
2. **Add comprehensive validation** with walk-forward analysis
3. **Optimize factor weights** using regime-specific analysis
4. **Enhance risk management** with dynamic position sizing
5. **Deploy production monitoring** with real-time regime tracking

---
*Generated on: {timestamp}*
""".format(timestamp=self.timestamp)
        
        index_path = self.base_dir / 'FILE_INDEX.md'
        with open(index_path, 'w') as f:
            f.write(index_content)
        print(f"📋 Created comprehensive file index: FILE_INDEX.md")
    
    def run_organization(self):
        """Run the complete organization process."""
        
        print("🚀 Starting Phase 28 file organization...")
        print("=" * 50)
        
        # Create directory structure
        print("\n📁 Creating directory structure...")
        self.create_directory_structure()
        
        # Create index files
        print("\n📝 Creating index files...")
        self.create_index_files()
        
        # Create symlinks
        print("\n🔗 Creating symbolic links...")
        self.create_symlinks()
        
        # Generate file index
        print("\n📋 Generating file index...")
        self.generate_file_index()
        
        print("\n✅ Organization complete!")
        print(f"📊 Summary: Files organized into logical directories")
        print(f"📚 Documentation: Index files created for each directory")
        print(f"🔗 Quick Access: Symbolic links created for key files")

if __name__ == "__main__":
    # Get the current directory
    current_dir = Path.cwd()
    
    # Check if we're in the right directory
    if not (current_dir / "28_qvm_engine_v3c.ipynb").exists():
        print("❌ Error: Please run this script from the phase28_qvm_engine_v3 directory")
        exit(1)
    
    # Create organizer and run
    organizer = Phase28Organizer(current_dir)
    organizer.run_organization() 