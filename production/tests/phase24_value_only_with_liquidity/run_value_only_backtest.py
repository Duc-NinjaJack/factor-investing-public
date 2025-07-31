#!/usr/bin/env python3
"""
Phase 24: Value-Only Factor Backtesting
Simple script to run the value-only backtest and save results.
"""

import sys
from pathlib import Path
import logging

# Add scripts directory to path
scripts_dir = Path(__file__).parent.parent.parent / 'scripts'
sys.path.append(str(scripts_dir))

from value_only_backtesting import ValueOnlyBacktesting

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Run value-only backtest."""
    print("🚀 PHASE 24: VALUE-ONLY FACTOR BACKTESTING")
    print("🎯 Strategy: Pure Value Factor (Value_Composite only)")
    
    try:
        # Initialize value-only backtesting engine
        value_backtesting = ValueOnlyBacktesting()
        
        # Customize configuration
        value_backtesting.backtest_config.update({
            'start_date': '2017-12-01',
            'end_date': '2025-07-28',
            'rebalance_freq': 'M',
            'portfolio_size': 25,
            'transaction_cost': 0.002  # 20 bps
        })
        
        # Run complete analysis
        results = value_backtesting.run_complete_analysis(
            save_plots=True,
            save_report=True
        )
        
        print("✅ Value-only backtest completed successfully!")
        print("📊 Results saved to:")
        print("   - value_only_performance_plots.png")
        print("   - value_only_backtest_report.txt")
        
        return results
        
    except Exception as e:
        print(f"❌ Value-only backtest failed: {e}")
        raise

if __name__ == "__main__":
    main() 