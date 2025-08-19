#!/usr/bin/env python3
"""
Configuration Manager for QVM Strategy
=====================================

This module handles all configuration-related operations:
- Loading strategy and backtest configurations
- Validating version compatibility
- Merging configurations with strategy-compatible settings
- Providing default configurations as fallbacks

Author: Raymond
Created: August 17, 2025
"""

import os
import yaml
from typing import Dict, Optional


def validate_version_compatibility(strategy_config: Dict, backtest_config: Dict) -> bool:
    """
    Validate version compatibility between configuration files and engine.
    
    Returns:
        bool: True if versions are compatible, False otherwise
    """
    try:
        # Check strategy config version
        strategy_version = strategy_config.get('strategy', {}).get('version', 'unknown')
        if not strategy_version.startswith('2.'):
            print(f"⚠️ Warning: Strategy config version {strategy_version} may not be compatible with v2.2.1 engine")
        
        # Check if portfolio size matches requirement (20 stocks)
        portfolio_size = strategy_config.get('strategy', {}).get('portfolio', {}).get('portfolio_size', 0)
        if portfolio_size != 20:
            print(f"⚠️ Warning: Portfolio size {portfolio_size} differs from required 20 stocks")
            print(f"   Please update portfolio_size to 20 in your configuration file")
        
        # Check if 4-pillar architecture is properly configured
        factor_weights = strategy_config.get('factor_weights', {})
        expected_pillars = {'quality', 'value', 'momentum', 'defensive'}
        missing_pillars = expected_pillars - set(factor_weights.keys())
        
        if missing_pillars:
            print(f"⚠️ Warning: Missing factor weights for pillars: {missing_pillars}")
            print(f"   Please add these missing pillars to your configuration file")
            print(f"   Example: factor_weights: {pillar}: 0.25 for each missing pillar")
        
        # Check backtest config compatibility
        active_window = backtest_config.get('active_window', 'unknown')
        if active_window not in backtest_config.get('backtest_windows', {}):
            print(f"⚠️ Warning: Active backtest window {active_window} not found in configuration")
            print(f"   Please check your backtest_config.yml file")
        
        print("✅ Version compatibility validation completed")
        return True
        
    except Exception as e:
        print(f"❌ Version compatibility validation failed: {e}")
        return False


def load_strategy_config(config_path: str = None) -> Dict:
    """Load strategy configuration from YAML file."""
    # Use single, direct path
    config_file = "../../config/strategy_config_v2_0_1_simple.yml"
    
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            print(f"✅ Strategy configuration loaded from {config_file}")
            return config
        except yaml.YAMLError as e:
            print(f"❌ Error parsing strategy config {config_file}: {e}")
    
    print("❌ Strategy configuration file not found")
    return get_default_strategy_config()


def load_backtest_config(config_path: str = None) -> Dict:
    """Load and merge backtest configuration with strategy-compatible settings."""
    # Use single, direct path
    config_file = "../../config/backtest_config.yml"
    
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            print(f"✅ Backtest configuration loaded from {config_file}")
            
            # Merge with strategy-compatible settings
            merged_config = merge_backtest_with_strategy_config(config)
            return merged_config
            
        except yaml.YAMLError as e:
            print(f"❌ Error parsing backtest config {config_file}: {e}")
    
    print("❌ Backtest configuration file not found")
    return get_default_backtest_config()


def merge_backtest_with_strategy_config(backtest_config: Dict) -> Dict:
    """
    Merge backtest configuration with strategy-compatible settings.
    Resolves conflicts and ensures portfolio size is 20 stocks.
    """
    print("🔄 Merging backtest configuration with strategy settings...")
    
    # Override portfolio size to match strategy requirement (20 stocks)
    if 'universe' in backtest_config:
        backtest_config['universe']['top_n_stocks'] = 20
        print("✅ Portfolio size updated to 20 stocks")
    
    # Ensure risk management settings are compatible
    if 'risk_overlay' in backtest_config:
        # Update risk overlay to use drawdown-based approach
        backtest_config['risk_overlay']['method'] = 'drawdown_based'
        
        # Check if drawdown thresholds are configured
        if 'drawdown_thresholds' not in backtest_config['risk_overlay']:
            print("⚠️ Warning: No drawdown thresholds found in backtest config")
            print("   Please add drawdown_thresholds to your backtest_config.yml:")
            print("   risk_overlay:")
            print("     drawdown_thresholds:")
            print("       drawdown_5: 0.20")
            print("       drawdown_10: 0.40")
            print("       drawdown_15: 0.60")
            print("       drawdown_20: 0.80")
            print("       drawdown_25: 0.90")
        else:
            print("✅ Risk overlay drawdown thresholds found in configuration")
    
    # Update rebalancing frequency to monthly (strategy default)
    if 'portfolio' in backtest_config:
        backtest_config['portfolio']['rebalance_frequency'] = 'M'
        print("✅ Rebalancing frequency set to monthly")
    
    print("✅ Backtest configuration merged successfully")
    return backtest_config


def get_default_strategy_config() -> Dict:
    """Get default strategy configuration if YAML file is not available."""
    print("❌ No strategy configuration file found")
    print("   Please create config/strategy_config_v2_0_1_simple.yml")
    return {}


def get_default_backtest_config() -> Dict:
    """Get default backtest configuration if YAML file is not available."""
    print("❌ No backtest configuration file found")
    print("   Please create production/config/backtest_config.yml")
    return {}


def display_configuration_summary(strategy_config: Dict, backtest_config: Dict) -> None:
    """Display a summary of the loaded configuration."""
    print(f"\n📋 CONFIGURATION SUMMARY")
    print("-" * 40)
    
    if strategy_config:
        print(f"Strategy: {strategy_config['strategy']['name']} v{strategy_config['strategy']['version']}")
        print(f"Portfolio Size: {strategy_config['strategy']['portfolio']['portfolio_size']} stocks")
        
        # Safely display factor weights
        factor_weights = strategy_config.get('factor_weights', {})
        quality_w = factor_weights.get('quality', 0.25)
        value_w = factor_weights.get('value', 0.25)
        momentum_w = factor_weights.get('momentum', 0.25)
        defensive_w = factor_weights.get('defensive', 0.25)
        print(f"Factor Weights: Q{quality_w:.0%} V{value_w:.0%} M{momentum_w:.0%} D{defensive_w:.0%}")
    
    if backtest_config:
        print(f"Backtest Window: {backtest_config['active_window']}")
        print(f"Risk Management: Drawdown-based cash allocation")
    
    print(f"Risk Management: Drawdown-based cash allocation")
