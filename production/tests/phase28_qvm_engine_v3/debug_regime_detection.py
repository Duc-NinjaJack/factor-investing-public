#!/usr/bin/env python3
"""
Debug script to understand why regime detection is only showing "Sideways" regime.
"""

import pandas as pd
import numpy as np
from sqlalchemy import text
import matplotlib.pyplot as plt

def debug_regime_detection():
    """Debug the regime detection logic."""
    
    print("🔍 DEBUGGING REGIME DETECTION")
    print("="*50)
    
    # Configuration from the notebook
    config = {
        "regime": {
            "lookback_period": 60,
            "volatility_threshold": 0.2659,  # This seems very high!
            "return_threshold": 0.002,
        }
    }
    
    print(f"📊 Configuration:")
    print(f"   - Lookback Period: {config['regime']['lookback_period']} days")
    print(f"   - Volatility Threshold: {config['regime']['volatility_threshold']:.4f}")
    print(f"   - Return Threshold: {config['regime']['return_threshold']:.4f}")
    
    # The issue: volatility_threshold is 0.2659 which is extremely high!
    # Typical daily volatility is around 0.01-0.02 (1-2%)
    # 0.2659 = 26.59% daily volatility, which is unrealistic
    
    print(f"\n🚨 PROBLEM IDENTIFIED:")
    print(f"   The volatility threshold of {config['regime']['volatility_threshold']:.4f} is too high!")
    print(f"   - Typical daily volatility: 0.01-0.02 (1-2%)")
    print(f"   - Current threshold: {config['regime']['volatility_threshold']:.4f} ({config['regime']['volatility_threshold']*100:.1f}%)")
    print(f"   - This means volatility will NEVER exceed the threshold")
    print(f"   - Result: All periods classified as 'Sideways'")
    
    # Let's check what typical volatility looks like
    print(f"\n📈 TYPICAL VOLATILITY ANALYSIS:")
    print(f"   - Low volatility: < 0.01 (1%)")
    print(f"   - Medium volatility: 0.01-0.02 (1-2%)")
    print(f"   - High volatility: 0.02-0.03 (2-3%)")
    print(f"   - Extreme volatility: > 0.03 (3%+)")
    
    # Suggested fixes
    print(f"\n🔧 SUGGESTED FIXES:")
    print(f"   1. Reduce volatility threshold to 0.012-0.015")
    print(f"   2. Use percentile-based thresholds")
    print(f"   3. Implement adaptive thresholds")
    
    # Show the regime detection logic
    print(f"\n🧮 REGIME DETECTION LOGIC:")
    print(f"   if volatility > {config['regime']['volatility_threshold']:.4f}:")
    print(f"       if avg_return > {config['regime']['return_threshold']:.4f}:")
    print(f"           return 'Bull'")
    print(f"       else:")
    print(f"           return 'Bear'")
    print(f"   else:")
    print(f"       if abs(avg_return) < 0.001:")
    print(f"           return 'Sideways'")
    print(f"       else:")
    print(f"           return 'Stress'")
    
    print(f"\n❌ CURRENT ISSUE:")
    print(f"   Since volatility is never > {config['regime']['volatility_threshold']:.4f},")
    print(f"   the first condition is never met, so it always goes to 'else'")
    print(f"   and checks if abs(avg_return) < 0.001")
    print(f"   If true: 'Sideways' (most common)")
    print(f"   If false: 'Stress'")
    
    return config

def suggest_fixed_config():
    """Suggest a fixed configuration."""
    
    print(f"\n✅ SUGGESTED FIXED CONFIGURATION:")
    
    fixed_config = {
        "regime": {
            "lookback_period": 60,
            "volatility_threshold": 0.012,  # 1.2% daily volatility
            "return_threshold": 0.002,      # 0.2% daily return
            "low_return_threshold": 0.001   # 0.1% for sideways detection
        }
    }
    
    print(f"   - Volatility Threshold: {fixed_config['regime']['volatility_threshold']:.4f} (1.2%)")
    print(f"   - Return Threshold: {fixed_config['regime']['return_threshold']:.4f} (0.2%)")
    print(f"   - Low Return Threshold: {fixed_config['regime']['low_return_threshold']:.4f} (0.1%)")
    
    print(f"\n📊 EXPECTED REGIME DISTRIBUTION WITH FIXED CONFIG:")
    print(f"   - Bull: High volatility + positive returns")
    print(f"   - Bear: High volatility + negative returns")
    print(f"   - Sideways: Low volatility + low returns")
    print(f"   - Stress: Low volatility + high returns (either direction)")
    
    return fixed_config

def create_regime_detector_fix():
    """Create a fixed RegimeDetector class."""
    
    print(f"\n🔧 FIXED REGIMEDETECTOR CLASS:")
    
    fix_code = '''
class RegimeDetector:
    """
    Fixed regime detection based on volatility and return thresholds.
    """
    def __init__(self, lookback_period: int = 60):
        self.lookback_period = lookback_period
        # Fixed thresholds
        self.volatility_threshold = 0.012  # 1.2% daily volatility
        self.return_threshold = 0.002      # 0.2% daily return
        self.low_return_threshold = 0.001  # 0.1% for sideways detection
    
    def detect_regime(self, price_data: pd.DataFrame) -> str:
        """Detect market regime based on volatility and return."""
        if len(price_data) < self.lookback_period:
            return 'Sideways'
        
        recent_data = price_data.tail(self.lookback_period)
        returns = recent_data['close'].pct_change().dropna()
        
        volatility = returns.std()
        avg_return = returns.mean()
        
        # Fixed regime detection logic
        if volatility > self.volatility_threshold:
            if avg_return > self.return_threshold:
                return 'Bull'
            else:
                return 'Bear'
        else:
            if abs(avg_return) < self.low_return_threshold:
                return 'Sideways'
            else:
                return 'Stress'
    
    def get_regime_allocation(self, regime: str) -> float:
        """Get target allocation based on regime."""
        regime_allocations = {
            'Bull': 1.0,      # Fully invested
            'Bear': 0.8,      # 80% invested
            'Sideways': 0.6,  # 60% invested
            'Stress': 0.4     # 40% invested
        }
        return regime_allocations.get(regime, 0.6)
'''
    
    print(fix_code)
    return fix_code

if __name__ == "__main__":
    # Run diagnostics
    current_config = debug_regime_detection()
    fixed_config = suggest_fixed_config()
    fix_code = create_regime_detector_fix()
    
    print(f"\n🎯 SUMMARY:")
    print(f"   The regime detection is only showing 'Sideways' because")
    print(f"   the volatility threshold (0.2659) is unrealistically high.")
    print(f"   This needs to be reduced to around 0.012 (1.2%) for proper")
    print(f"   regime detection to work.") 