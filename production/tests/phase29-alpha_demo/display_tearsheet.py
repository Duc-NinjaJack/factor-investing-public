#!/usr/bin/env python3
"""
Display Tearsheet for Beta-Optimized Strategy
============================================

This script displays the tearsheet image for the beta-optimized strategy.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from pathlib import Path

def display_tearsheet():
    """Display the tearsheet image."""
    print("📊 DISPLAYING TEARSHEET - BETA-OPTIMIZED STRATEGY")
    print("=" * 50)
    
    # Find the most recent tearsheet
    tearsheet_files = [
        "tearsheet_optimized_20250804_230022.png",
        "tearsheet_optimized_20250804_231213.png"
    ]
    
    tearsheet_path = None
    for file in tearsheet_files:
        if os.path.exists(file):
            tearsheet_path = file
            break
    
    if tearsheet_path is None:
        print("❌ No tearsheet file found!")
        return
    
    print(f"📈 Loading tearsheet: {tearsheet_path}")
    
    try:
        # Load and display the image
        img = mpimg.imread(tearsheet_path)
        
        # Create figure with larger size
        plt.figure(figsize=(16, 12))
        plt.imshow(img)
        plt.axis('off')
        plt.title('QVM Engine v3j Beta-Optimized Strategy - Performance Tearsheet', 
                 fontsize=16, fontweight='bold', pad=20)
        
        # Add performance summary as text overlay
        summary_text = """
        📊 PERFORMANCE SUMMARY (2024-2025)
        
        Strategy Annualized Return: 25.94%
        Benchmark Annualized Return: 19.88%
        Strategy Sharpe Ratio: 0.75
        Benchmark Sharpe Ratio: 1.08
        Strategy Max Drawdown: -37.84%
        Benchmark Max Drawdown: -18.11%
        Information Ratio: 0.52
        Beta: 1.84 (reduced from 3.37)
        
        🎯 KEY IMPROVEMENTS:
        • Beta reduced by 45% (3.37 → 1.84)
        • Sharpe ratio improved by 168% (0.28 → 0.75)
        • Max drawdown improved by 49% (-74.60% → -37.84%)
        • Annual return improved by 33% (19.52% → 25.94%)
        """
        
        plt.figtext(0.02, 0.02, summary_text, fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        print("✅ Tearsheet displayed successfully!")
        print(f"📁 File: {tearsheet_path}")
        print(f"📏 Size: {os.path.getsize(tearsheet_path) / 1024:.1f} KB")
        
    except Exception as e:
        print(f"❌ Error displaying tearsheet: {e}")
        
        # Try to open with system default viewer
        try:
            import subprocess
            import platform
            
            system = platform.system()
            if system == "Darwin":  # macOS
                subprocess.run(["open", tearsheet_path])
            elif system == "Windows":
                subprocess.run(["start", tearsheet_path], shell=True)
            else:  # Linux
                subprocess.run(["xdg-open", tearsheet_path])
                
            print(f"✅ Opened tearsheet with system viewer: {tearsheet_path}")
            
        except Exception as e2:
            print(f"❌ Could not open tearsheet: {e2}")
            print(f"📁 Manual path: {os.path.abspath(tearsheet_path)}")

def show_tearsheet_info():
    """Show information about the tearsheet."""
    print("\n📋 TEARSHEET CONTENTS")
    print("=" * 30)
    
    print("The tearsheet includes:")
    print("1. 📈 Equity Curve Comparison")
    print("   - Strategy vs Benchmark performance")
    print("   - Cumulative returns over time")
    
    print("\n2. 📊 Drawdown Analysis")
    print("   - Maximum drawdown periods")
    print("   - Recovery periods")
    
    print("\n3. 📅 Annual Returns")
    print("   - Year-by-year performance")
    print("   - Strategy vs benchmark comparison")
    
    print("\n4. 📈 Rolling Sharpe Ratio")
    print("   - Risk-adjusted performance over time")
    print("   - Rolling 12-month Sharpe ratios")
    
    print("\n5. 🔄 Portfolio Turnover")
    print("   - Trading activity over time")
    print("   - Transaction costs impact")
    
    print("\n6. 📊 Portfolio Size Evolution")
    print("   - Number of stocks over time")
    print("   - Diversification metrics")
    
    print("\n7. 📋 Performance Metrics Table")
    print("   - Comprehensive statistics")
    print("   - Risk and return metrics")

def main():
    """Main function."""
    print("🔍 TEARSHEET DISPLAY FOR BETA-OPTIMIZED STRATEGY")
    print("=" * 60)
    
    # Show tearsheet info
    show_tearsheet_info()
    
    # Display the tearsheet
    display_tearsheet()
    
    print("\n✅ Tearsheet display complete!")

if __name__ == "__main__":
    main() 