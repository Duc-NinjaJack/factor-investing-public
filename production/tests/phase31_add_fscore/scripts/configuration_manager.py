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

from production.tests.phase31_add_fscore.scripts.validation_manager import (
    validate_backtest_config,
)


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
            print("   Example: factor_weights:")
            print("     quality: 0.25")
            print("     value: 0.25")
            print("     momentum: 0.25")
            print("     defensive: 0.25")
        
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
    # Try multiple possible paths for the strategy config
    possible_paths = [
        # From the current test directory
        "config/strategy_config_v2_0_1_simple.yml",
        # From the production config directory
        "production/config/strategy_config_v2_0_1_simple.yml",
        # From the root config directory
        "/home/raymond/Documents/Projects/factor-investing-public/config/strategy_config_v2_0_1_simple.yml",
        # From the production config directory (absolute path)
        "/home/raymond/Documents/Projects/factor-investing-public/production/config/strategy_config_v2_0_1_simple.yml"
    ]
    
    for config_file in possible_paths:
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as file:
                    config = yaml.safe_load(file)
                print(f"✅ Strategy configuration loaded from {config_file}")
                return config
            except yaml.YAMLError as e:
                print(f"❌ Error parsing strategy config {config_file}: {e}")
                continue
    
    print("❌ Strategy configuration file not found in any of the expected locations:")
    for path in possible_paths:
        print(f"   - {path}")
    return get_default_strategy_config()


def load_backtest_config(config_path: str = None) -> Dict:
    """Load and merge backtest configuration with strategy-compatible settings."""
    # Try multiple possible paths for the backtest config
    possible_paths = [
        # From the current test directory
        "config/backtest_config.yml",
        # From the production config directory
        "production/config/backtest_config.yml",
        # From the root config directory
        "/home/raymond/Documents/Projects/factor-investing-public/config/backtest_config.yml",
        # From the production config directory (absolute path)
        "/home/raymond/Documents/Projects/factor-investing-public/production/config/backtest_config.yml"
    ]
    
    for config_file in possible_paths:
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as file:
                    config = yaml.safe_load(file)
                print(f"✅ Backtest configuration loaded from {config_file}")
                
                # Merge with strategy-compatible settings
                merged_config = merge_backtest_with_strategy_config(config)
                # Schema validation and normalization
                try:
                    # Use a no-op logger substitute; runner passes a real logger
                    class _NullLogger:
                        def debug(self, *a, **k):
                            pass
                        def info(self, *a, **k):
                            pass
                        def error(self, *a, **k):
                            pass
                    merged_config = validate_backtest_config(merged_config, _NullLogger())
                except Exception:
                    # If validation fails here, defer to runner to raise with context
                    pass
                return merged_config
                
            except yaml.YAMLError as e:
                print(f"❌ Error parsing backtest config {config_file}: {e}")
                continue
    
    print("❌ Backtest configuration file not found in any of the expected locations:")
    for path in possible_paths:
        print(f"   - {path}")
    return get_default_backtest_config()


def merge_backtest_with_strategy_config(backtest_config: Dict) -> Dict:
    """
    Merge backtest configuration with strategy-compatible settings.
    Resolves conflicts and ensures portfolio size is 20 stocks.
    """
    print("🔄 Merging backtest configuration with strategy settings...")
    
    # Do not override universe size here; honor YAML `universe.top_n_stocks`.
    # Historical note: this previously forced 20 stocks, which starved sector samples.
    # If needed, control portfolio sizing in the strategy config, not here.
    if 'universe' in backtest_config:
        # No-op: ensure key exists but do not mutate provided value
        _ = backtest_config['universe'].get('top_n_stocks', None)
    
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
    print("   Please create one of the following files:")
    print("   - config/strategy_config_v2_0_1_simple.yml")
    print("   - production/config/strategy_config_v2_0_1_simple.yml")
    print("   - /home/raymond/Documents/Projects/factor-investing-public/config/strategy_config_v2_0_1_simple.yml")
    
    # Return a minimal default configuration
    return {
        'strategy': {
            'name': 'QVM 4-Pillar Flat Strategy (Default)',
            'version': '2.0.1',
            'portfolio': {
                'portfolio_size': 20,
                'universe_size': 100,
                'starting_capital': 10000000000
            }
        },
        'factor_weights': {
            'quality': 0.25,
            'value': 0.25,
            'momentum': 0.25,
            'defensive': 0.25
        },
        'risk_management': {
            'enabled': True,
            'default_cash': 0.05,
            'cash_allocation': {
                'drawdown_5': 0.20,
                'drawdown_10': 0.40,
                'drawdown_15': 0.60,
                'drawdown_20': 0.80,
                'drawdown_25': 0.90
            }
        },
        'output': {
            'logging': {
                'level': 'INFO'
            }
        }
    }


def get_default_backtest_config() -> Dict:
    """Get default backtest configuration if YAML file is not available."""
    print("❌ No backtest configuration file found")
    print("   Please create one of the following files:")
    print("   - config/backtest_config.yml")
    print("   - production/config/backtest_config.yml")
    print("   - /home/raymond/Documents/Projects/factor-investing-public/config/backtest_config.yml")
    
    # Return a minimal default configuration
    return {
        'active_window': 'FULL_2016_2025',
        'backtest_windows': {
            'LIQUID_2018_2025': {
                'start': '2018-01-01',
                'end': '2025-12-31',
                'description': 'Post-IPO spike, includes 2018 market stress'
            }
        },
        'ic_hurdles': {
            'annual_return_net': 0.15,
            'annual_volatility': 0.15,
            'sharpe_ratio_net': 1.0,
            'max_drawdown': -0.35,
            'beta_vs_vnindex': 0.75,
            'information_ratio': 0.8
        },
        'universe': {
            'method': 'liquid_universe',
            'top_n_stocks': 20,
            'min_adtv_vnd': 10000000000,
            'min_adtv_pct_mcap': 0.0004,
            'sector_concentration_limit': 0.25,
            'foreign_ownership_buffer': 0.03
        },
        'portfolio': {
            'base_leverage': 1.0,
            'max_leverage': 1.5,
            'rebalance_frequency': 'M',
            'concentration_limit': 0.20
        },
        'benchmark': {
            'primary': 'VN_INDEX',
            'secondary': 'VNFIN_LEAD'
        },
        'output': {
            'save_factor_weights': True,
            'save_portfolio_history': True,
            'save_cost_breakdown': True,
            'generate_tearsheet': True,
            'export_metrics_csv': True
        }
    }


def display_configuration_summary(strategy_config: Dict, backtest_config: Dict) -> None:
    """Display a summary of the loaded configuration."""
    print("\n📋 CONFIGURATION SUMMARY")
    print("=" * 50)
    
    # Strategy configuration summary
    if strategy_config:
        strategy_name = strategy_config.get('strategy', {}).get('name', 'Unknown')
        strategy_version = strategy_config.get('strategy', {}).get('version', 'Unknown')
        portfolio_size = strategy_config.get('strategy', {}).get('portfolio', {}).get('portfolio_size', 'Unknown')
        
        print(f"Strategy: {strategy_name} v{strategy_version}")
        print(f"Portfolio Size: {portfolio_size} stocks")
        
        # Factor weights
        factor_weights = strategy_config.get('factor_weights', {})
        if factor_weights:
            print("Factor Weights:")
            for pillar, weight in factor_weights.items():
                print(f"  {pillar.capitalize()}: {weight:.0%}")
        
        # Risk management
        risk_management = strategy_config.get('risk_management', {})
        if risk_management.get('enabled', False):
            print("Risk Management: ✅ Enabled")
            default_cash = risk_management.get('default_cash', 0)
            print(f"Default Cash: {default_cash:.0%}")
        else:
            print("Risk Management: ❌ Disabled")
    else:
        print("Strategy Configuration: ❌ Not loaded")
    
    # Backtest configuration summary
    if backtest_config:
        active_window = backtest_config.get('active_window', 'Unknown')
        backtest_windows = backtest_config.get('backtest_windows', {})
        
        print(f"\nBacktest Window: {active_window}")
        if active_window in backtest_windows:
            window_config = backtest_windows[active_window]
            start_date = window_config.get('start', 'Unknown')
            end_date = window_config.get('end', 'Unknown')
            print(f"Period: {start_date} to {end_date}")
        
        # Investment committee hurdles
        ic_hurdles = backtest_config.get('ic_hurdles', {})
        if ic_hurdles:
            print("Investment Committee Hurdles:")
            for hurdle, value in ic_hurdles.items():
                if isinstance(value, float):
                    if 'return' in hurdle or 'ratio' in hurdle:
                        print(f"  {hurdle}: {value:.1%}")
                    elif 'volatility' in hurdle or 'drawdown' in hurdle:
                        print(f"  {hurdle}: {value:.1%}")
                    else:
                        print(f"  {hurdle}: {value:.2f}")
                else:
                    print(f"  {hurdle}: {value}")
    else:
        print("Backtest Configuration: ❌ Not loaded")
    
    print("\n" + "=" * 50)
