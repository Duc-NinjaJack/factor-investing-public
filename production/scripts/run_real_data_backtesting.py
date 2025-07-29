#!/usr/bin/env python3
"""
================================================================================
Real Data Backtesting Runner - Usage Example
================================================================================
Purpose:
    Demonstrate how to use the refactored RealDataBacktesting engine.
    This script shows different ways to run backtesting analysis.

Usage Examples:
    1. Basic usage with default settings
    2. Custom configuration
    3. Command line arguments
    4. Programmatic usage

Author: Quantitative Strategy Team
Date: January 2025
"""

import sys
from pathlib import Path
import logging

# Add the scripts directory to path
scripts_dir = Path(__file__).parent
sys.path.append(str(scripts_dir))

from real_data_backtesting import RealDataBacktesting

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def example_basic_usage():
    """Example 1: Basic usage with default settings."""
    logger.info("🚀 Example 1: Basic usage with default settings")
    
    try:
        # Initialize with default settings
        backtesting = RealDataBacktesting()
        
        # Run complete analysis
        results = backtesting.run_complete_analysis(
            save_plots=True,
            save_report=True
        )
        
        logger.info("✅ Basic usage completed successfully")
        return results
        
    except Exception as e:
        logger.error(f"❌ Basic usage failed: {e}")
        return None

def example_custom_configuration():
    """Example 2: Custom configuration."""
    logger.info("🚀 Example 2: Custom configuration")
    
    try:
        # Initialize with custom settings
        backtesting = RealDataBacktesting()
        
        # Customize thresholds
        backtesting.thresholds = {
            '5B_VND': 5_000_000_000,
            '2B_VND': 2_000_000_000
        }
        
        # Customize backtest configuration
        backtesting.backtest_config.update({
            'portfolio_size': 30,
            'transaction_cost': 0.003,  # 30 bps
            'rebalance_freq': 'W',  # Weekly rebalancing
            'start_date': '2020-01-01'  # Start from 2020
        })
        
        # Run analysis
        results = backtesting.run_complete_analysis(
            save_plots=True,
            save_report=True
        )
        
        logger.info("✅ Custom configuration completed successfully")
        return results
        
    except Exception as e:
        logger.error(f"❌ Custom configuration failed: {e}")
        return None

def example_programmatic_usage():
    """Example 3: Programmatic usage with step-by-step control."""
    logger.info("🚀 Example 3: Programmatic usage")
    
    try:
        # Initialize
        backtesting = RealDataBacktesting()
        
        # Step 1: Load data
        data = backtesting.load_data()
        logger.info("✅ Data loaded")
        
        # Step 2: Prepare data
        prepared_data = backtesting.prepare_data_for_backtesting(data)
        logger.info("✅ Data prepared")
        
        # Step 3: Run individual backtests
        results = {}
        for threshold_name, threshold_value in backtesting.thresholds.items():
            result = backtesting.run_backtest(threshold_name, threshold_value, prepared_data)
            results[threshold_name] = result
            logger.info(f"✅ {threshold_name} backtest completed")
        
        # Step 4: Create visualizations
        backtesting.create_performance_visualizations(results)
        logger.info("✅ Visualizations created")
        
        # Step 5: Generate report
        report = backtesting.generate_comprehensive_report(results)
        print("\n" + "=" * 80)
        print("PROGRAMMATIC USAGE REPORT")
        print("=" * 80)
        print(report)
        
        logger.info("✅ Programmatic usage completed successfully")
        return results
        
    except Exception as e:
        logger.error(f"❌ Programmatic usage failed: {e}")
        return None

def example_single_threshold_analysis():
    """Example 4: Single threshold analysis."""
    logger.info("🚀 Example 4: Single threshold analysis")
    
    try:
        # Initialize
        backtesting = RealDataBacktesting()
        
        # Load and prepare data
        data = backtesting.load_data()
        prepared_data = backtesting.prepare_data_for_backtesting(data)
        
        # Run single backtest
        result = backtesting.run_backtest('3B_VND', 3_000_000_000, prepared_data)
        
        # Print results
        metrics = result['metrics']
        print("\n" + "=" * 60)
        print("SINGLE THRESHOLD ANALYSIS RESULTS")
        print("=" * 60)
        print(f"Threshold: 3B VND")
        print(f"Annual Return: {metrics['annual_return']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Alpha: {metrics['alpha']:.2%}")
        print(f"Beta: {metrics['beta']:.2f}")
        print(f"Total Return: {metrics['total_return']:.2%}")
        print("=" * 60)
        
        logger.info("✅ Single threshold analysis completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"❌ Single threshold analysis failed: {e}")
        return None

def main():
    """Main function to run examples."""
    print("=" * 80)
    print("REAL DATA BACKTESTING EXAMPLES")
    print("=" * 80)
    print("This script demonstrates different ways to use the RealDataBacktesting engine.")
    print("Choose an example to run:")
    print("1. Basic usage with default settings")
    print("2. Custom configuration")
    print("3. Programmatic usage")
    print("4. Single threshold analysis")
    print("5. Run all examples")
    print("0. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (0-5): ").strip()
            
            if choice == '0':
                print("👋 Goodbye!")
                break
            elif choice == '1':
                example_basic_usage()
            elif choice == '2':
                example_custom_configuration()
            elif choice == '3':
                example_programmatic_usage()
            elif choice == '4':
                example_single_threshold_analysis()
            elif choice == '5':
                print("\n🔄 Running all examples...")
                example_basic_usage()
                example_custom_configuration()
                example_programmatic_usage()
                example_single_threshold_analysis()
                print("\n✅ All examples completed!")
            else:
                print("❌ Invalid choice. Please enter a number between 0 and 5.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            logger.error(f"❌ Error: {e}")

if __name__ == "__main__":
    main()