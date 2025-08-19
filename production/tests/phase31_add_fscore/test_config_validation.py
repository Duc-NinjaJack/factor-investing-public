#!/usr/bin/env python3
"""
Test script to verify configuration validation works correctly.
"""

import sys
import os
import yaml
import logging

# Add the project root to the path
sys.path.append('/home/raymond/Documents/Projects/factor-investing-public')

# Import the engine
import importlib.util
spec = importlib.util.spec_from_file_location("QVM_flat_config", "07_QVM_flat_config.py")
QVM_flat_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(QVM_flat_config)
QVMFlatConfigEngine = QVM_flat_config.QVMFlatConfigEngine

def test_config_validation():
    """Test configuration validation with various scenarios."""
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    print("🧪 Testing Configuration Validation")
    print("=" * 50)
    
    # Test 1: Valid configuration
    print("\n1️⃣ Testing valid configuration...")
    try:
        config_path = "../../../config/strategy_config_v2_0_1_simple.yml"
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as file:
                strategy_config = yaml.safe_load(file)
            
            engine = QVMFlatConfigEngine(strategy_config=strategy_config)
            print("✅ Valid configuration test passed")
        else:
            print("⚠️ Config file not found, skipping valid config test")
    except Exception as e:
        print(f"❌ Valid configuration test failed: {e}")
    
    # Test 2: Missing factor_weights section
    print("\n2️⃣ Testing missing factor_weights section...")
    try:
        invalid_config = {
            'strategy': {
                'name': 'Test Strategy',
                'version': '1.0',
                'portfolio': {
                    'universe_size': 100,
                    'portfolio_size': 20,
                    'starting_capital': 1000000
                }
            },
            'risk_management': {
                'enabled': True
            }
        }
        
        engine = QVMFlatConfigEngine(strategy_config=invalid_config)
        print("❌ Should have failed - missing factor_weights")
    except ValueError as e:
        if "factor_weights" in str(e):
            print("✅ Correctly caught missing factor_weights section")
        else:
            print(f"❌ Wrong error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    # Test 3: Factor weights don't sum to 1.0
    print("\n3️⃣ Testing factor weights that don't sum to 1.0...")
    try:
        invalid_config = {
            'strategy': {
                'name': 'Test Strategy',
                'version': '1.0',
                'portfolio': {
                    'universe_size': 100,
                    'portfolio_size': 20,
                    'starting_capital': 1000000
                }
            },
            'factor_weights': {
                'quality': 0.3,
                'value': 0.3,
                'momentum': 0.3,
                'defensive': 0.3  # Sum = 1.2, should fail
            },
            'risk_management': {
                'enabled': True
            }
        }
        
        engine = QVMFlatConfigEngine(strategy_config=invalid_config)
        print("❌ Should have failed - weights don't sum to 1.0")
    except ValueError as e:
        if "sum to 1.0" in str(e):
            print("✅ Correctly caught weights not summing to 1.0")
        else:
            print(f"❌ Wrong error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    # Test 4: Missing required pillars
    print("\n4️⃣ Testing missing required pillars...")
    try:
        invalid_config = {
            'strategy': {
                'name': 'Test Strategy',
                'version': '1.0',
                'portfolio': {
                    'universe_size': 100,
                    'portfolio_size': 20,
                    'starting_capital': 1000000
                }
            },
            'factor_weights': {
                'quality': 0.5,
                'value': 0.5  # Missing momentum and defensive
            },
            'risk_management': {
                'enabled': True
            }
        }
        
        engine = QVMFlatConfigEngine(strategy_config=invalid_config)
        print("❌ Should have failed - missing required pillars")
    except ValueError as e:
        if "Missing factor weights for pillars" in str(e):
            print("✅ Correctly caught missing required pillars")
        else:
            print(f"❌ Wrong error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    print("\n🎯 Configuration validation tests completed!")

if __name__ == "__main__":
    test_config_validation()
