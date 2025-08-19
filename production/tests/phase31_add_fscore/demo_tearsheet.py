#!/usr/bin/env python3
"""
Tearsheet Demonstration Script
==============================

This script demonstrates how to generate and display the QVM strategy tearsheet
with all visualizations including factor score evolution and portfolio holdings distribution.
"""

import sys
import os
sys.path.append('/home/raymond/Documents/Projects/factor-investing-public')

# Import required modules
from scripts.configuration_manager import load_strategy_config, load_backtest_config
from scripts.tearsheet_generator import generate_comprehensive_tearsheet
from scripts.visualization_manager import generate_factor_score_evolution_plot, generate_portfolio_holdings_distribution_plot

def main():
    print("🎯 TEARSHEET DEMONSTRATION")
    print("=" * 50)
    
    # Load configurations
    print("Loading configurations...")
    try:
        strategy_config = load_strategy_config()
        backtest_config = load_backtest_config()
        print("✅ Configurations loaded successfully!")
        print(f"Strategy: {strategy_config['strategy']['name']}")
        print(f"Backtest: {backtest_config['active_window']}")
    except Exception as e:
        print(f"❌ Error loading configurations: {e}")
        return
    
    # Create sample data for demonstration
    print("\n📊 Creating sample data...")
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    # Create sample holdings data
    dates = pd.date_range('2022-01-01', '2024-12-31', freq='ME')
    tickers = ['VNM', 'HPG', 'VIC', 'TCB', 'MBB', 'ACV', 'FPT', 'VHM', 'GAS', 'PLX']
    holdings_data = []
    
    for date in dates:
        for ticker in tickers:
            holdings_data.append({
                'date': date,
                'ticker': ticker,
                'Quality_Composite': np.random.normal(0, 1),
                'Value_Composite': np.random.normal(0, 1),
                'Momentum_Composite': np.random.normal(0, 1),
                'Defensive_Composite': np.random.normal(0, 1),
                'QVM_Composite': np.random.normal(0, 1)
            })
    
    holdings_df = pd.DataFrame(holdings_data)
    
    # Create sample portfolio returns
    portfolio_returns = pd.Series(
        np.random.normal(0.001, 0.02, len(dates)),
        index=dates
    )
    
    # Create sample benchmark returns
    benchmark_returns = pd.Series(
        np.random.normal(0.0008, 0.018, len(dates)),
        index=dates
    )
    
    print(f"✅ Sample data created: {len(holdings_df)} holdings records, {len(portfolio_returns)} return periods")
    
    # Generate the comprehensive tearsheet
    print("\n🎨 GENERATING COMPREHENSIVE TEARSHEET")
    print("=" * 60)
    
    try:
        tearsheet_result = generate_comprehensive_tearsheet(
            strategy_returns=portfolio_returns,
            benchmark_returns=benchmark_returns,
            title='QVM 4-Pillar Strategy vs VN-Index'
        )
        print("✅ Main tearsheet generated successfully!")
    except Exception as e:
        print(f"❌ Error generating main tearsheet: {e}")
        import traceback
        traceback.print_exc()
    
    # Generate factor score evolution plot
    print("\n📊 Generating Factor Score Evolution Plot...")
    try:
        generate_factor_score_evolution_plot(holdings_df)
        print("✅ Factor Score Evolution Plot generated!")
    except Exception as e:
        print(f"❌ Error generating factor score evolution plot: {e}")
    
    # Generate portfolio holdings distribution plot
    print("\n📊 Generating Portfolio Holdings Distribution Plot...")
    try:
        generate_portfolio_holdings_distribution_plot(holdings_df)
        print("✅ Portfolio Holdings Distribution Plot generated!")
    except Exception as e:
        print(f"❌ Error generating portfolio holdings distribution plot: {e}")
    
    print("\n🎯 Tearsheet demonstration completed!")
    print("📊 All visualizations should now be displayed above")
    print("\n💡 To use this in your notebook, copy the code from this script")

if __name__ == "__main__":
    main()
