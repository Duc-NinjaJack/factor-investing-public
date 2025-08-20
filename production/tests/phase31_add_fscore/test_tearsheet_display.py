#!/usr/bin/env python3
"""
Test script to ensure tearsheet displays properly
"""

import sys
import os
sys.path.append('.')

# Set matplotlib backend for display
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for display
import matplotlib.pyplot as plt

# Import tearsheet functions
from scripts.tearsheet_generator import generate_comprehensive_tearsheet
from scripts.visualization_manager import generate_factor_score_evolution_plot, generate_portfolio_holdings_distribution_plot

def test_tearsheet_display():
    """Test that tearsheet displays properly"""
    print("🎯 TESTING TEARSHEET DISPLAY")
    print("=" * 50)
    
    # Create sample data
    import pandas as pd
    import numpy as np
    
    dates = pd.date_range('2022-01-01', '2024-12-31', freq='ME')
    portfolio_returns = pd.Series(np.random.normal(0.001, 0.02, len(dates)), index=dates)
    benchmark_returns = pd.Series(np.random.normal(0.0008, 0.018, len(dates)), index=dates)
    
    # Create sample holdings data
    tickers = ['VNM', 'HPG', 'VIC', 'TCB', 'MBB']
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
    
    print("📊 Sample data created")
    print("🎨 Generating main tearsheet...")
    
    # Generate main tearsheet
    try:
        generate_comprehensive_tearsheet(
            strategy_returns=portfolio_returns,
            benchmark_returns=benchmark_returns,
            title='QVM Strategy Test Display'
        )
        print("✅ Main tearsheet generated and should be displayed!")
    except Exception as e:
        print(f"❌ Error in main tearsheet: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n📊 Generating factor score evolution plot...")
    
    # Generate factor score evolution plot
    try:
        generate_factor_score_evolution_plot(holdings_df)
        print("✅ Factor score evolution plot generated and should be displayed!")
    except Exception as e:
        print(f"❌ Error in factor score evolution plot: {e}")
    
    print("\n📊 Generating portfolio holdings distribution plot...")
    
    # Generate portfolio holdings distribution plot
    try:
        generate_portfolio_holdings_distribution_plot(holdings_df)
        print("✅ Portfolio holdings distribution plot generated and should be displayed!")
    except Exception as e:
        print(f"❌ Error in portfolio holdings distribution plot: {e}")
    
    print("\n🎯 All plots should now be visible!")
    print("💡 If you don't see plots, try running this in a Jupyter notebook")
    
    # Keep plots open
    plt.show()

if __name__ == "__main__":
    test_tearsheet_display()



