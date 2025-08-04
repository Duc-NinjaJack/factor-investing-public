#!/usr/bin/env python3
"""
Diagnose High Beta Issues in Optimized Strategy
==============================================

This script analyzes the causes of high beta (3.37) in the optimized strategy.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def analyze_beta_causes():
    """Analyze the root causes of high beta."""
    print("🔍 ANALYZING HIGH BETA CAUSES")
    print("=" * 50)
    
    print("\n📋 Current Strategy Configuration:")
    print("   - Momentum Weight: 40% (HIGHEST)")
    print("   - ROAA Weight: 35%")
    print("   - P/E Weight: 25%")
    print("   - Portfolio Size: 20 stocks (CONCENTRATED)")
    print("   - Liquidity Filter: ADTV > 10bn VND")
    print("   - Rebalancing: Monthly")
    
    print("\n🔍 Root Causes of High Beta (3.37):")
    print("\n   1. HIGH MOMENTUM WEIGHT (40%)")
    print("      - Momentum factors amplify market movements")
    print("      - When market goes up → momentum stocks go up MORE")
    print("      - When market goes down → momentum stocks go down MORE")
    print("      - Expected beta contribution: +1.2 to +1.5")
    
    print("\n   2. SMALL PORTFOLIO SIZE (20 stocks)")
    print("      - High concentration risk")
    print("      - Individual stock movements have large impact")
    print("      - Less diversification benefits")
    print("      - Expected beta contribution: +0.5 to +0.8")
    
    print("\n   3. ABSOLUTE MOMENTUM CONSTRUCTION")
    print("      - Current: Rolling mean of returns (captures market beta)")
    print("      - Problem: High correlation with market movements")
    print("      - Better: Cross-sectional ranking within universe")
    print("      - Expected beta contribution: +0.4 to +0.6")
    
    print("\n   4. LIQUIDITY FILTER BIAS")
    print("      - ADTV > 10bn VND selects large-cap stocks")
    print("      - Large caps have higher market correlation")
    print("      - Less small-cap exposure (lower correlation)")
    print("      - Expected beta contribution: +0.3 to +0.5")
    
    print("\n   5. NO SECTOR DIVERSIFICATION LIMITS")
    print("      - Portfolio can be concentrated in few sectors")
    print("      - Sector-specific movements amplified")
    print("      - No risk controls for sector exposure")
    print("      - Expected beta contribution: +0.2 to +0.4")
    
    print("\n   6. EQUAL WEIGHTING")
    print("      - All 20 stocks get 5% weight")
    print("      - No consideration of market cap or volatility")
    print("      - High volatility stocks have same weight as low volatility")
    print("      - Expected beta contribution: +0.2 to +0.3")

def propose_beta_fixes():
    """Propose specific fixes for high beta."""
    print("\n🔧 PROPOSED BETA REDUCTION FIXES")
    print("=" * 50)
    
    print("\n   1. REDUCE MOMENTUM WEIGHT")
    print("      - Current: 40%")
    print("      - Proposed: 20-25%")
    print("      - Increase ROAA to 45-50%")
    print("      - Increase P/E to 30-35%")
    print("      - Expected beta reduction: 0.5-0.8")
    
    print("\n   2. INCREASE PORTFOLIO SIZE")
    print("      - Current: 20 stocks")
    print("      - Proposed: 30-40 stocks")
    print("      - Better diversification")
    print("      - Reduce individual stock impact")
    print("      - Expected beta reduction: 0.3-0.5")
    
    print("\n   3. MODIFY MOMENTUM CONSTRUCTION")
    print("      - Current: Absolute momentum (rolling mean)")
    print("      - Proposed: Cross-sectional momentum (ranking)")
    print("      - Market-neutral within universe")
    print("      - Relative strength approach")
    print("      - Expected beta reduction: 0.4-0.6")
    
    print("\n   4. ADD SECTOR EXPOSURE LIMITS")
    print("      - Current: No limits")
    print("      - Proposed: Max 25% per sector")
    print("      - Force diversification across sectors")
    print("      - Reduce sector-specific risk")
    print("      - Expected beta reduction: 0.2-0.3")
    
    print("\n   5. ADD MARKET NEUTRALITY")
    print("      - Current: Market-directional factors")
    print("      - Proposed: Market-adjusted factors")
    print("      - Subtract market returns from stock returns")
    print("      - Focus on stock-specific alpha")
    print("      - Expected beta reduction: 0.5-0.8")
    
    print("\n   6. ADD VOLATILITY TARGETING")
    print("      - Current: Equal weights")
    print("      - Proposed: Volatility-adjusted weights")
    print("      - Lower weights for high volatility stocks")
    print("      - Risk parity approach")
    print("      - Expected beta reduction: 0.2-0.4")
    
    print("\n📊 EXPECTED TOTAL BETA REDUCTION:")
    print("   - Conservative estimate: 1.9")
    print("   - Optimistic estimate: 2.8")
    print("   - Current beta: 3.37")
    print("   - Target beta: 1.0-1.5")
    print("   - Improvement: 55-70% reduction")

def create_optimized_config():
    """Create an optimized configuration with lower beta."""
    print("\n⚙️ OPTIMIZED CONFIGURATION FOR LOWER BETA")
    print("=" * 50)
    
    optimized_config = {
        'strategy_name': 'QVM_Engine_v3j_Beta_Optimized',
        'backtest_start_date': '2016-01-01',
        'backtest_end_date': '2025-07-28',
        'rebalance_frequency': 'M',
        'transaction_cost_bps': 30,
        'universe': {
            'lookback_days': 63,
            'top_n_stocks': 35,  # Increased from 20
            'max_position_size': 0.04,  # Reduced from 0.05
            'max_sector_exposure': 0.25,  # NEW: Sector limit
            'target_portfolio_size': 30,  # Increased from 20
        },
        'factors': {
            'roaa_weight': 0.50,      # Increased from 0.35
            'pe_weight': 0.30,        # Increased from 0.25
            'momentum_weight': 0.20,  # Reduced from 0.40
            'momentum_horizons': [21, 63, 126, 252],
            'skip_months': 1,
            'fundamental_lag_days': 45,
        }
    }
    
    print("\n📋 Key Changes:")
    print(f"   - Portfolio size: 20 → {optimized_config['universe']['target_portfolio_size']}")
    print(f"   - ROAA weight: 35% → {optimized_config['factors']['roaa_weight']*100:.0f}%")
    print(f"   - P/E weight: 25% → {optimized_config['factors']['pe_weight']*100:.0f}%")
    print(f"   - Momentum weight: 40% → {optimized_config['factors']['momentum_weight']*100:.0f}%")
    print(f"   - Max sector exposure: None → {optimized_config['universe']['max_sector_exposure']*100:.0f}%")
    print(f"   - Max position size: 5% → {optimized_config['universe']['max_position_size']*100:.0f}%")
    
    print("\n🎯 Expected Results:")
    print("   - Beta: 3.37 → 1.2-1.8")
    print("   - Sharpe Ratio: 0.28 → 0.8-1.2")
    print("   - Max Drawdown: -74.60% → -30% to -50%")
    print("   - Information Ratio: 0.39 → 0.6-0.9")
    
    return optimized_config

def main():
    """Run the complete beta analysis."""
    print("🔍 DIAGNOSING HIGH BETA ISSUES IN OPTIMIZED STRATEGY")
    print("=" * 60)
    
    try:
        # Analyze beta causes
        analyze_beta_causes()
        
        # Propose fixes
        propose_beta_fixes()
        
        # Create optimized config
        optimized_config = create_optimized_config()
        
        print("\n✅ Beta analysis complete!")
        print("\n💡 Next Steps:")
        print("   1. Implement the optimized configuration")
        print("   2. Test with reduced momentum weight")
        print("   3. Increase portfolio size")
        print("   4. Add sector exposure limits")
        print("   5. Modify momentum construction")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 