# Regime Detector Component
"""
Regime detection component for QVM strategy variants.
This module contains the regime detection logic that can be used by different strategies.
"""

import pandas as pd
import numpy as np

class RegimeDetector:
    """
    Simple regime detection based on volatility and return thresholds.
    FIXED: Now properly accepts and uses threshold parameters.
    """
    def __init__(self, lookback_period: int = 90, volatility_threshold: float = 0.0140, 
                 return_threshold: float = 0.0012, low_return_threshold: float = 0.0002):
        self.lookback_period = lookback_period
        self.volatility_threshold = volatility_threshold
        self.return_threshold = return_threshold
        self.low_return_threshold = low_return_threshold
        print(f"✅ RegimeDetector initialized with thresholds:")
        print(f"   - Volatility: {self.volatility_threshold:.2%}")
        print(f"   - Return: {self.return_threshold:.2%}")
        print(f"   - Low Return: {self.low_return_threshold:.2%}")
    
    def detect_regime(self, price_data: pd.DataFrame) -> str:
        """Detect market regime based on volatility and return."""
        # Use available data, but require at least 60 days
        min_required_days = 60
        if len(price_data) < min_required_days:
            print(f"   ⚠️  Insufficient data: {len(price_data)} < {min_required_days}")
            return 'Sideways'
        
        # Use all available data up to lookback_period
        available_days = min(len(price_data), self.lookback_period)
        recent_data = price_data.tail(available_days)
        returns = recent_data['close'].pct_change().dropna()
        
        volatility = returns.std()
        avg_return = returns.mean()
        
        # Debug output
        print(f"   🔍 Regime Debug: Vol={volatility:.4f} ({volatility:.2%}), AvgRet={avg_return:.4f} ({avg_return:.2%})")
        print(f"   🔍 Thresholds: VolThresh={self.volatility_threshold:.4f}, RetThresh={self.return_threshold:.4f}, LowRetThresh={self.low_return_threshold:.4f}")
        
        if volatility > self.volatility_threshold:
            if avg_return > self.return_threshold:
                print(f"   📈 Detected: Bull (Vol={volatility:.2%} > {self.volatility_threshold:.2%}, Ret={avg_return:.2%} > {self.return_threshold:.2%})")
                return 'Bull'
            else:
                print(f"   📉 Detected: Bear (Vol={volatility:.2%} > {self.volatility_threshold:.2%}, Ret={avg_return:.2%} <= {self.return_threshold:.2%})")
                return 'Bear'
        else:
            if abs(avg_return) < self.low_return_threshold:
                print(f"   ↔️  Detected: Sideways (Vol={volatility:.2%} <= {self.volatility_threshold:.2%}, |Ret|={abs(avg_return):.2%} < {self.low_return_threshold:.2%})")
                return 'Sideways'
            else:
                print(f"   ⚠️  Detected: Stress (Vol={volatility:.2%} <= {self.volatility_threshold:.2%}, |Ret|={abs(avg_return):.2%} >= {self.low_return_threshold:.2%})")
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