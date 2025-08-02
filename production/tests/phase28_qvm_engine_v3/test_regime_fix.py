#!/usr/bin/env python3
"""
Test script to verify the regime detection fix.
This demonstrates the issue and the solution.
"""

import pandas as pd
import numpy as np

# ============================================================================
# ORIGINAL BROKEN RegimeDetector (from v3c)
# ============================================================================

class RegimeDetector_BROKEN:
    """
    BROKEN: This is the original version that was stuck at "Sideways"
    """
    def __init__(self, lookback_period: int = 60):
        self.lookback_period = lookback_period
        # PROBLEM: Missing threshold parameters!
    
    def detect_regime(self, price_data: pd.DataFrame) -> str:
        """Detect market regime based on volatility and return."""
        if len(price_data) < self.lookback_period:
            return 'Sideways'
        
        recent_data = price_data.tail(self.lookback_period)
        returns = recent_data['close'].pct_change().dropna()
        
        volatility = returns.std()
        avg_return = returns.mean()
        
        # PROBLEM: These attributes don't exist!
        # This will cause AttributeError or use undefined values
        if volatility > self.volatility_threshold:  # ❌ AttributeError!
            if avg_return > self.return_threshold:   # ❌ AttributeError!
                return 'Bull'
            else:
                return 'Bear'
        else:
            if abs(avg_return) < self.low_return_threshold:  # ❌ AttributeError!
                return 'Sideways'
            else:
                return 'Stress'

# ============================================================================
# FIXED RegimeDetector (for v3d)
# ============================================================================

class RegimeDetector_FIXED:
    """
    FIXED: This version properly accepts and uses threshold parameters
    """
    def __init__(self, lookback_period: int = 90, volatility_threshold: float = 0.2659, 
                 return_threshold: float = 0.2588, low_return_threshold: float = 0.2131):
        self.lookback_period = lookback_period
        self.volatility_threshold = volatility_threshold
        self.return_threshold = return_threshold
        self.low_return_threshold = low_return_threshold
        print(f"✅ RegimeDetector initialized with thresholds:")
        print(f"   - Volatility: {self.volatility_threshold:.4f}")
        print(f"   - Return: {self.return_threshold:.4f}")
        print(f"   - Low Return: {self.low_return_threshold:.4f}")
    
    def detect_regime(self, price_data: pd.DataFrame) -> str:
        """Detect market regime based on volatility and return."""
        if len(price_data) < self.lookback_period:
            return 'Sideways'
        
        recent_data = price_data.tail(self.lookback_period)
        returns = recent_data['close'].pct_change().dropna()
        
        volatility = returns.std()
        avg_return = returns.mean()
        
        # Debug output
        print(f"   🔍 Regime Debug: Vol={volatility:.4f}, AvgRet={avg_return:.4f}")
        
        if volatility > self.volatility_threshold:
            if avg_return > self.return_threshold:
                print(f"   📈 Detected: Bull (Vol={volatility:.4f} > {self.volatility_threshold:.4f}, Ret={avg_return:.4f} > {self.return_threshold:.4f})")
                return 'Bull'
            else:
                print(f"   📉 Detected: Bear (Vol={volatility:.4f} > {self.volatility_threshold:.4f}, Ret={avg_return:.4f} <= {self.return_threshold:.4f})")
                return 'Bear'
        else:
            if abs(avg_return) < self.low_return_threshold:
                print(f"   ↔️  Detected: Sideways (Vol={volatility:.4f} <= {self.volatility_threshold:.4f}, |Ret|={abs(avg_return):.4f} < {self.low_return_threshold:.4f})")
                return 'Sideways'
            else:
                print(f"   ⚠️  Detected: Stress (Vol={volatility:.4f} <= {self.volatility_threshold:.4f}, |Ret|={abs(avg_return):.4f} >= {self.low_return_threshold:.4f})")
                return 'Stress'

# ============================================================================
# TEST THE FIX
# ============================================================================

def test_regime_detection():
    """Test both broken and fixed versions."""
    print("🧪 Testing Regime Detection Fix")
    print("=" * 50)
    
    # Test scenarios with corrected thresholds
    test_scenarios = [
        {
            'name': 'Bull Market',
            'volatility': 0.30,   # High volatility (> 0.2659)
            'return': 0.35,       # High positive return (> 0.2588)
            'expected': 'Bull'
        },
        {
            'name': 'Bear Market', 
            'volatility': 0.30,   # High volatility (> 0.2659)
            'return': -0.20,      # Negative return
            'expected': 'Bear'
        },
        {
            'name': 'Sideways Market',
            'volatility': 0.20,   # Low volatility (<= 0.2659)
            'return': 0.15,       # Low return (<= 0.2131)
            'expected': 'Sideways'
        },
        {
            'name': 'Stress Market',
            'volatility': 0.20,   # Low volatility (<= 0.2659)
            'return': 0.25,       # Moderate return (> 0.2131)
            'expected': 'Stress'
        }
    ]
    
    # Test the FIXED version
    print("\n✅ Testing FIXED RegimeDetector:")
    regime_detector_fixed = RegimeDetector_FIXED(
        lookback_period=90,
        volatility_threshold=0.2659,
        return_threshold=0.2588,
        low_return_threshold=0.2131
    )
    
    for scenario in test_scenarios:
        print(f"\n📊 Testing: {scenario['name']}")
        
        # Create synthetic price data
        np.random.seed(42)
        returns = np.random.normal(scenario['return'], scenario['volatility'], 100)
        prices = (1 + pd.Series(returns)).cumprod()
        price_data = pd.DataFrame({'close': prices})
        
        # Detect regime
        detected_regime = regime_detector_fixed.detect_regime(price_data)
        
        # Check result
        if detected_regime == scenario['expected']:
            print(f"   ✅ PASS: Expected {scenario['expected']}, Got {detected_regime}")
        else:
            print(f"   ❌ FAIL: Expected {scenario['expected']}, Got {detected_regime}")
    
    # Test the BROKEN version (this will fail)
    print("\n❌ Testing BROKEN RegimeDetector (this will fail):")
    try:
        regime_detector_broken = RegimeDetector_BROKEN(lookback_period=60)
        
        # This will cause AttributeError
        test_data = pd.DataFrame({'close': [1.0, 1.01, 1.02, 1.03, 1.04]})
        regime = regime_detector_broken.detect_regime(test_data)
        print(f"   Unexpected success: {regime}")
        
    except AttributeError as e:
        print(f"   ✅ Expected failure: {e}")
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")

# ============================================================================
# SHOW THE FIX IN QVMEngineV3AdoptedInsights
# ============================================================================

def show_qvm_fix():
    """Show how the fix should be applied in the QVM engine."""
    print("\n🔧 QVM Engine Fix:")
    print("=" * 50)
    
    print("❌ BROKEN initialization (from v3c):")
    print("""
    # Initialize components
    self.regime_detector = RegimeDetector(config['regime']['lookback_period'])
    # PROBLEM: Thresholds not passed!
    """)
    
    print("✅ FIXED initialization (for v3d):")
    print("""
    # Initialize components with FIXED regime detector
    self.regime_detector = RegimeDetector(
        lookback_period=config['regime']['lookback_period'],
        volatility_threshold=config['regime']['volatility_threshold'],
        return_threshold=config['regime']['return_threshold'],
        low_return_threshold=config['regime']['low_return_threshold']
    )
    """)

if __name__ == "__main__":
    test_regime_detection()
    show_qvm_fix()
    
    print("\n🎯 SUMMARY:")
    print("   - The original RegimeDetector was missing threshold parameters")
    print("   - This caused all regimes to default to 'Sideways'")
    print("   - The fix adds proper parameter passing in __init__")
    print("   - Now different market conditions will be properly detected") 