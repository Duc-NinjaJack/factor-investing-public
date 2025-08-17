#!/usr/bin/env python3
"""
Test script to verify configuration loading from YAML file.
"""

import sys
import os
import yaml
import importlib.util

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(__file__))

# Import the configuration loading function using importlib
def import_module_from_file(module_name, file_path):
    """Import a module from file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Import the module
risk_comparison_module = import_module_from_file("risk_comparison", "06_QVM_risk_comparison.py")
load_config = risk_comparison_module.load_config
get_default_config = risk_comparison_module.get_default_config

def test_config_loading():
    """Test configuration loading functionality."""
    print("🧪 Testing Configuration Loading")
    print("=" * 40)
    
    # Test 1: Load from YAML file
    print("\n📋 Test 1: Loading from YAML file")
    try:
        config = load_config("strategy_config_simple.yml")
        print("✅ YAML configuration loaded successfully")
        print(f"   Strategy: {config['strategy']['name']}")
        print(f"   Version: {config['strategy']['version']}")
        print(f"   Portfolio Size: {config['strategy']['portfolio']['portfolio_size']}")
        print(f"   Starting Capital: {config['strategy']['portfolio']['starting_capital']:,}")
        print(f"   Factor Weights: Q({config['factor_weights']['quality']:.1%}) V({config['factor_weights']['value']:.1%}) M({config['factor_weights']['momentum']:.1%})")
        
        # Test risk management config
        risk_config = config['risk_management']
        print(f"   Risk Management: {'Enabled' if risk_config['enabled'] else 'Disabled'}")
        print(f"   Default Cash: {risk_config['default_cash']:.1%}")
        print(f"   Cash Rules: {len(risk_config['cash_allocation'])} drawdown levels")
        
    except Exception as e:
        print(f"❌ Error loading YAML config: {e}")
    
    # Test 2: Test default config fallback
    print("\n📋 Test 2: Default configuration fallback")
    try:
        default_config = get_default_config()
        print("✅ Default configuration created successfully")
        print(f"   Portfolio Size: {default_config['strategy']['portfolio']['portfolio_size']}")
        print(f"   Starting Capital: {default_config['strategy']['portfolio']['starting_capital']:,}")
        
    except Exception as e:
        print(f"❌ Error creating default config: {e}")
    
    # Test 3: Test non-existent file
    print("\n📋 Test 3: Non-existent file handling")
    try:
        config = load_config("non_existent_file.yml")
        print("✅ Fallback to default config successful")
        
    except Exception as e:
        print(f"❌ Error handling non-existent file: {e}")
    
    print("\n🎉 Configuration loading tests completed!")

if __name__ == "__main__":
    test_config_loading()
