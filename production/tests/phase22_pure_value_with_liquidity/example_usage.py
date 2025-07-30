#!/usr/bin/env python3
"""
Example Usage: Phase 22 Weighted Composite Real Data Backtesting

This script demonstrates how to use the weighted composite backtesting
implementation with different configurations and analysis options.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

# Add paths for imports
sys.path.append('../../../production/database')
sys.path.append('../../../production/scripts')

try:
    # Import the main backtesting class
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "weighted_composite_backtest", 
        "22_weighted_composite_real_data_backtest.py"
    )
    weighted_composite_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(weighted_composite_module)
    WeightedCompositeBacktesting = weighted_composite_module.WeightedCompositeBacktesting
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all dependencies are installed and paths are correct.")
    sys.exit(1)

def example_basic_usage():
    """Example 1: Basic usage with default settings."""
    print("=" * 60)
    print("📊 EXAMPLE 1: Basic Usage with Default Settings")
    print("=" * 60)
    
    try:
        # Initialize backtesting engine with default settings
        backtesting = WeightedCompositeBacktesting()
        
        print("✅ Backtesting engine initialized successfully")
        print(f"   - Weighting scheme: {backtesting.weighting_scheme}")
        print(f"   - Portfolio size: {backtesting.strategy_config['portfolio_size']}")
        print(f"   - Liquidity thresholds: {list(backtesting.thresholds.keys())}")
        
        # Run complete analysis
        print("\n🚀 Running complete analysis...")
        results = backtesting.run_complete_analysis(
            save_plots=True,
            save_report=True
        )
        
        print("\n✅ Analysis completed successfully!")
        print("Generated files:")
        print("   - Performance plots: weighted_composite_backtesting_plots_*.png")
        print("   - Detailed report: weighted_composite_backtesting_report_*.txt")
        
        return results
        
    except Exception as e:
        print(f"❌ Basic usage example failed: {e}")
        return None

def example_custom_configuration():
    """Example 2: Custom configuration and analysis."""
    print("\n" + "=" * 60)
    print("📊 EXAMPLE 2: Custom Configuration")
    print("=" * 60)
    
    try:
        # Initialize with custom configuration
        backtesting = WeightedCompositeBacktesting(
            config_path='../../../config/database.yml',
            pickle_path='../../../data/unrestricted_universe_data.pkl'
        )
        
        # Modify strategy configuration
        backtesting.strategy_config.update({
            'portfolio_size': 30,           # Larger portfolio
            'quintile_selection': 0.85,     # Top 15% selection
            'transaction_cost': 0.003,      # 30 bps transaction cost
        })
        
        print("✅ Custom configuration applied:")
        print(f"   - Portfolio size: {backtesting.strategy_config['portfolio_size']}")
        print(f"   - Selection threshold: Top {backtesting.strategy_config['quintile_selection']*100:.0f}%")
        print(f"   - Transaction cost: {backtesting.strategy_config['transaction_cost']*100:.1f}%")
        
        # Load data
        print("\n📊 Loading factor data...")
        data = backtesting.load_factor_data()
        
        # Run backtest for specific threshold
        print("\n🚀 Running backtest for 10B VND threshold...")
        results_10b = backtesting.run_weighted_composite_backtest(
            '10B_VND', 
            backtesting.thresholds['10B_VND'], 
            data
        )
        
        print(f"✅ 10B VND backtest completed:")
        print(f"   - Annual Return: {results_10b['metrics']['annual_return']:.2%}")
        print(f"   - Sharpe Ratio: {results_10b['metrics']['sharpe_ratio']:.2f}")
        print(f"   - Max Drawdown: {results_10b['metrics']['max_drawdown']:.2%}")
        
        return results_10b
        
    except Exception as e:
        print(f"❌ Custom configuration example failed: {e}")
        return None

def example_performance_analysis():
    """Example 3: Detailed performance analysis."""
    print("\n" + "=" * 60)
    print("📊 EXAMPLE 3: Detailed Performance Analysis")
    print("=" * 60)
    
    try:
        # Initialize backtesting engine
        backtesting = WeightedCompositeBacktesting()
        
        # Load data
        data = backtesting.load_factor_data()
        
        # Run comparative backtests
        print("🔄 Running comparative backtests...")
        all_results = backtesting.run_comparative_backtests(data)
        
        # Analyze results
        print("\n📈 Performance Analysis:")
        print("-" * 40)
        
        for threshold, results in all_results.items():
            metrics = results['metrics']
            print(f"\n{threshold}:")
            print(f"  Annual Return: {metrics['annual_return']:.2%}")
            print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
            print(f"  Alpha: {metrics['alpha']:.2%}")
            print(f"  Beta: {metrics['beta']:.2f}")
            print(f"  Information Ratio: {metrics['information_ratio']:.2f}")
        
        # Find best performing threshold
        best_threshold = max(all_results.keys(), 
                           key=lambda x: all_results[x]['metrics']['sharpe_ratio'])
        best_metrics = all_results[best_threshold]['metrics']
        
        print(f"\n🏆 Best Performing Threshold: {best_threshold}")
        print(f"   - Sharpe Ratio: {best_metrics['sharpe_ratio']:.2f}")
        print(f"   - Annual Return: {best_metrics['annual_return']:.2%}")
        print(f"   - Max Drawdown: {best_metrics['max_drawdown']:.2%}")
        
        # Create visualizations
        print("\n📊 Creating performance visualizations...")
        backtesting.create_weighted_composite_visualizations(all_results)
        
        return all_results
        
    except Exception as e:
        print(f"❌ Performance analysis example failed: {e}")
        return None

def example_factor_analysis():
    """Example 4: Factor-level analysis."""
    print("\n" + "=" * 60)
    print("📊 EXAMPLE 4: Factor-Level Analysis")
    print("=" * 60)
    
    try:
        # Initialize backtesting engine
        backtesting = WeightedCompositeBacktesting()
        
        # Load factor data
        data = backtesting.load_factor_data()
        factor_data = data['factor_data']
        
        # Analyze factor distributions
        print("📊 Factor Distribution Analysis:")
        print("-" * 40)
        
        for factor in ['Quality_Composite', 'Value_Composite', 'Momentum_Composite']:
            values = factor_data[factor].dropna()
            print(f"\n{factor}:")
            print(f"  Mean: {values.mean():.4f}")
            print(f"  Std: {values.std():.4f}")
            print(f"  Min: {values.min():.4f}")
            print(f"  Max: {values.max():.4f}")
            print(f"  Count: {len(values):,}")
        
        # Analyze factor correlations
        print("\n📊 Factor Correlations:")
        print("-" * 40)
        
        correlation_matrix = factor_data[['Quality_Composite', 'Value_Composite', 'Momentum_Composite']].corr()
        print(correlation_matrix.round(3))
        
        # Analyze weighted composite distribution
        print("\n📊 Weighted Composite Analysis:")
        print("-" * 40)
        
        # Calculate weighted composite for a sample date
        sample_date = factor_data['date'].max()
        sample_data = factor_data[factor_data['date'] == sample_date]
        
        if len(sample_data) > 0:
            # Create momentum reversal
            sample_data['Momentum_Reversal'] = -1 * sample_data['Momentum_Composite']
            
            # Z-score normalization
            for factor in ['Quality_Composite', 'Value_Composite', 'Momentum_Reversal']:
                mean_val = sample_data[factor].mean()
                std_val = sample_data[factor].std()
                if std_val > 0:
                    sample_data[f'{factor}_Z'] = (sample_data[factor] - mean_val) / std_val
                else:
                    sample_data[f'{factor}_Z'] = 0.0
            
            # Calculate weighted composite
            sample_data['Weighted_Composite'] = (
                0.6 * sample_data['Value_Composite_Z'] +
                0.2 * sample_data['Quality_Composite_Z'] +
                0.2 * sample_data['Momentum_Reversal_Z']
            )
            
            print(f"Sample date: {sample_date}")
            print(f"Number of stocks: {len(sample_data)}")
            print(f"Weighted Composite - Mean: {sample_data['Weighted_Composite'].mean():.4f}")
            print(f"Weighted Composite - Std: {sample_data['Weighted_Composite'].std():.4f}")
            print(f"Weighted Composite - Range: [{sample_data['Weighted_Composite'].min():.4f}, {sample_data['Weighted_Composite'].max():.4f}]")
        
        return factor_data
        
    except Exception as e:
        print(f"❌ Factor analysis example failed: {e}")
        return None

def example_portfolio_analysis():
    """Example 5: Portfolio holdings analysis."""
    print("\n" + "=" * 60)
    print("📊 EXAMPLE 5: Portfolio Holdings Analysis")
    print("=" * 60)
    
    try:
        # Initialize backtesting engine
        backtesting = WeightedCompositeBacktesting()
        
        # Load data
        data = backtesting.load_factor_data()
        
        # Run backtest to get holdings
        print("🚀 Running backtest to analyze holdings...")
        results = backtesting.run_weighted_composite_backtest(
            '10B_VND', 
            backtesting.thresholds['10B_VND'], 
            data
        )
        
        holdings = results['holdings']
        
        print(f"\n📊 Portfolio Holdings Analysis:")
        print("-" * 40)
        print(f"Total rebalancing periods: {len(holdings)}")
        
        if len(holdings) > 0:
            # Analyze portfolio size evolution
            portfolio_sizes = [h['portfolio_size'] for h in holdings]
            universe_sizes = [h['universe_size'] for h in holdings]
            
            print(f"Average portfolio size: {sum(portfolio_sizes)/len(portfolio_sizes):.1f}")
            print(f"Average universe size: {sum(universe_sizes)/len(universe_sizes):.1f}")
            print(f"Portfolio size range: [{min(portfolio_sizes)}, {max(portfolio_sizes)}]")
            
            # Analyze recent holdings
            recent_holdings = holdings[-1] if holdings else None
            if recent_holdings:
                print(f"\n📊 Most Recent Portfolio ({recent_holdings['date']}):")
                print(f"  Portfolio size: {recent_holdings['portfolio_size']}")
                print(f"  Universe size: {recent_holdings['universe_size']}")
                print(f"  Stocks: {', '.join(recent_holdings['stocks'][:10])}{'...' if len(recent_holdings['stocks']) > 10 else ''}")
                
                # Show top 5 stocks by weighted composite score
                scores = recent_holdings['weighted_composite_scores']
                top_stocks = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
                print(f"\n  Top 5 stocks by weighted composite score:")
                for ticker, score in top_stocks:
                    print(f"    {ticker}: {score:.4f}")
        
        return holdings
        
    except Exception as e:
        print(f"❌ Portfolio analysis example failed: {e}")
        return None

def main():
    """Run all examples."""
    print("🎯 PHASE 22 WEIGHTED COMPOSITE BACKTESTING - EXAMPLE USAGE")
    print("=" * 80)
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run examples
    examples = [
        ("Basic Usage", example_basic_usage),
        ("Custom Configuration", example_custom_configuration),
        ("Performance Analysis", example_performance_analysis),
        ("Factor Analysis", example_factor_analysis),
        ("Portfolio Analysis", example_portfolio_analysis),
    ]
    
    results = {}
    
    for example_name, example_func in examples:
        print(f"\n{'='*20} {example_name} {'='*20}")
        try:
            result = example_func()
            results[example_name] = result
            print(f"✅ {example_name} completed successfully")
        except Exception as e:
            print(f"❌ {example_name} failed: {e}")
            results[example_name] = None
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 EXAMPLE USAGE SUMMARY")
    print("=" * 80)
    
    successful_examples = sum(1 for result in results.values() if result is not None)
    total_examples = len(examples)
    
    print(f"Successful examples: {successful_examples}/{total_examples}")
    
    if successful_examples == total_examples:
        print("🎉 All examples completed successfully!")
        print("\nNext steps:")
        print("1. Review the generated outputs")
        print("2. Analyze the performance results")
        print("3. Customize the strategy parameters")
        print("4. Run the full backtest with your preferred settings")
    else:
        print("⚠️ Some examples failed. Please check the error messages above.")
        print("\nCommon issues:")
        print("1. Database connection problems")
        print("2. Missing data files")
        print("3. Incorrect file paths")
        print("4. Insufficient data for analysis")
    
    return results

if __name__ == "__main__":
    main()