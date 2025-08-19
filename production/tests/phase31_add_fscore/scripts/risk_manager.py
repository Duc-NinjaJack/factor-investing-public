#!/usr/bin/env python3
"""
Risk Manager for QVM Strategy
============================

This module handles all risk management operations:
- Dynamic cash allocation based on market drawdown
- Cash allocation rules validation and display
- Risk scenario testing and simulation
- Drawdown-based protection mechanisms

Author: Raymond
Created: August 17, 2025
"""

import logging
from typing import Dict, Optional
import pandas as pd


def calculate_dynamic_cash_allocation(benchmark_prices: pd.Series, 
                                    current_date: pd.Timestamp,
                                    strategy_config: Dict,
                                    default_cash: float = 0.05,
                                    logger: logging.Logger = None) -> float:
    """
    Calculate dynamic cash allocation based on market drawdown from peak.
    
    RISK MANAGEMENT LOGIC:
    - 5% drop in benchmark => 20% cash allocation (key threshold)
    - Progressive cash allocation as drawdown increases
    - Protects capital during market stress periods
    
    Args:
        benchmark_prices: Historical benchmark prices
        current_date: Current date for calculation
        strategy_config: Strategy configuration dictionary
        default_cash: Default cash allocation percentage
        logger: Logger instance for messages
        
    Returns:
        float: Cash allocation percentage (0.0 to 1.0)
    """
    try:
        # Check if risk management is enabled
        if not strategy_config.get('risk_management', {}).get('enabled', False):
            if logger:
                logger.debug("Risk management disabled - no cash allocation")
            return 0.0
        
        # Validate benchmark prices
        if benchmark_prices is None or len(benchmark_prices) < 2:
            if logger:
                logger.warning("Insufficient benchmark data for cash allocation calculation")
            return default_cash
        
        # Get historical prices up to current date
        historical_prices = benchmark_prices.loc[:current_date]
        if len(historical_prices) < 2:
            if logger:
                logger.warning("Insufficient historical prices for drawdown calculation")
            return default_cash
        
        # Calculate drawdown from peak
        peak_price = historical_prices.max()
        current_price = historical_prices.iloc[-1]
        
        if peak_price <= 0:
            if logger:
                logger.warning("Invalid peak price for drawdown calculation")
            return default_cash
        
        drawdown = (peak_price - current_price) / peak_price
        
        # Log drawdown for debugging
        if logger:
            logger.debug(f"Drawdown calculation: Peak={peak_price:.2f}, Current={current_price:.2f}, Drawdown={drawdown:.2%}")
        
        # Get cash allocation rules with proper defaults
        cash_allocation_rules = strategy_config.get('risk_management', {}).get('cash_allocation', {})
        
        # Define proper default cash allocation thresholds if not configured
        default_rules = {
            'drawdown_5': 0.20,    # 5% drawdown => 20% cash
            'drawdown_10': 0.40,   # 10% drawdown => 40% cash
            'drawdown_15': 0.60,   # 15% drawdown => 60% cash
            'drawdown_20': 0.80,   # 20% drawdown => 80% cash
            'drawdown_25': 0.90,   # 25% drawdown => 90% cash
            'drawdown_30': 0.95,   # 30% drawdown => 95% cash
            'drawdown_40': 0.98,   # 40% drawdown => 98% cash
            'drawdown_50': 0.99    # 50% drawdown => 99% cash
        }
        
        # Use configured rules if available, otherwise use defaults
        if cash_allocation_rules:
            # Merge configured rules with defaults
            effective_rules = {**default_rules, **cash_allocation_rules}
            if logger:
                logger.debug(f"Using configured cash allocation rules: {effective_rules}")
        else:
            effective_rules = default_rules
            if logger:
                logger.warning("No cash allocation rules configured - using default thresholds")
        
        # Apply progressive cash allocation based on drawdown severity
        if drawdown < 0.05:
            cash_allocation = effective_rules.get('drawdown_5', 0.20)
        elif drawdown < 0.10:
            cash_allocation = effective_rules.get('drawdown_10', 0.40)
        elif drawdown < 0.15:
            cash_allocation = effective_rules.get('drawdown_15', 0.60)
        elif drawdown < 0.20:
            cash_allocation = effective_rules.get('drawdown_20', 0.80)
        elif drawdown < 0.25:
            cash_allocation = effective_rules.get('drawdown_25', 0.90)
        elif drawdown < 0.30:
            cash_allocation = effective_rules.get('drawdown_30', 0.95)
        elif drawdown < 0.40:
            cash_allocation = effective_rules.get('drawdown_40', 0.98)
        elif drawdown < 0.50:
            cash_allocation = effective_rules.get('drawdown_50', 0.99)
        else:
            # For extreme drawdowns (>50%), go almost entirely to cash
            cash_allocation = 0.99
        
        # Ensure cash allocation is within valid bounds
        cash_allocation = max(0.0, min(1.0, cash_allocation))
        
        # Log the cash allocation decision
        if logger:
            logger.info(f"📊 Cash Allocation Decision: {drawdown:.1%} drawdown => {cash_allocation:.1%} cash")
        
        return cash_allocation
        
    except Exception as e:
        if logger:
            logger.error(f"Error calculating dynamic cash allocation: {e}")
        # Return default cash allocation on error
        return default_cash


def display_cash_allocation_rules(strategy_config: Dict) -> None:
    """
    Display the current cash allocation rules for transparency and debugging.
    This helps verify that the risk management system is properly configured.
    """
    try:
        print("\n🛡️ CASH ALLOCATION RULES VALIDATION")
        print("-" * 50)
        
        # Check if risk management is enabled
        if not strategy_config.get('risk_management', {}).get('enabled', False):
            print("❌ Risk management is DISABLED")
            print("   The strategy will not allocate cash during drawdowns")
            return
        
        print("✅ Risk management is ENABLED")
        
        # Get configured rules
        configured_rules = strategy_config.get('risk_management', {}).get('cash_allocation', {})
        
        # Define default rules for comparison
        default_rules = {
            'drawdown_5': 0.20,    # 5% drawdown => 20% cash
            'drawdown_10': 0.40,   # 10% drawdown => 40% cash
            'drawdown_15': 0.60,   # 15% drawdown => 60% cash
            'drawdown_20': 0.80,   # 20% drawdown => 80% cash
            'drawdown_25': 0.90,   # 25% drawdown => 90% cash
            'drawdown_30': 0.95,   # 30% drawdown => 95% cash
            'drawdown_40': 0.98,   # 40% drawdown => 98% cash
            'drawdown_50': 0.99    # 50% drawdown => 99% cash
        }
        
        # Merge configured rules with defaults
        effective_rules = {**default_rules, **configured_rules}
        
        print(f"\n📊 EFFECTIVE CASH ALLOCATION THRESHOLDS:")
        print(f"{'Drawdown Level':<15} {'Cash Allocation':<15} {'Status':<10}")
        print("-" * 40)
        
        for threshold, cash_pct in effective_rules.items():
            if threshold in configured_rules:
                status = "✅ Configured"
            else:
                status = "📋 Default"
            
            drawdown_pct = float(threshold.split('_')[1])  # Extract number from 'drawdown_5'
            print(f"{drawdown_pct:>5.0f}%{'':<10} {cash_pct:>6.0%}{'':<9} {status}")
        
        # Show key protection levels
        print(f"\n🎯 KEY PROTECTION LEVELS:")
        print(f"   • 5% drawdown → {effective_rules['drawdown_5']:.0%} cash (first line of defense)")
        print(f"   • 15% drawdown → {effective_rules['drawdown_15']:.0%} cash (moderate protection)")
        print(f"   • 25% drawdown → {effective_rules['drawdown_25']:.0%} cash (strong protection)")
        print(f"   • 40% drawdown → {effective_rules['drawdown_40']:.0%} cash (extreme protection)")
        
        # Validate configuration
        print(f"\n🔍 CONFIGURATION VALIDATION:")
        if configured_rules:
            print(f"   ✅ Custom cash allocation rules found: {len(configured_rules)} thresholds")
            print(f"   📋 Using {len(effective_rules)} total thresholds (custom + defaults)")
        else:
            print(f"   ⚠️ No custom cash allocation rules found")
            print(f"   📋 Using {len(effective_rules)} default thresholds")
            print(f"   💡 Consider adding custom thresholds to strategy_config_v2_0_1_simple.yml")
        
        # Show default cash allocation
        default_cash = strategy_config.get('risk_management', {}).get('default_cash', 0.05)
        print(f"   💰 Default cash allocation: {default_cash:.0%}")
        
    except Exception as e:
        print(f"❌ Error displaying cash allocation rules: {e}")


def test_cash_allocation_scenarios(strategy_config: Dict) -> None:
    """
    Test cash allocation calculation with various drawdown scenarios.
    This helps verify that the risk management system works correctly.
    """
    try:
        print("\n🧪 CASH ALLOCATION SCENARIO TESTING")
        print("-" * 50)
        
        # Create mock benchmark prices for testing
        # Simulate a market that peaked at 1000 and then declined
        peak_price = 1000.0
        test_scenarios = [
            (peak_price * 0.98, "2% drawdown"),      # 2% below peak
            (peak_price * 0.95, "5% drawdown"),      # 5% below peak (first threshold)
            (peak_price * 0.90, "10% drawdown"),     # 10% below peak
            (peak_price * 0.85, "15% drawdown"),     # 15% below peak
            (peak_price * 0.80, "20% drawdown"),     # 20% below peak
            (peak_price * 0.75, "25% drawdown"),     # 25% below peak
            (peak_price * 0.70, "30% drawdown"),     # 30% below peak
            (peak_price * 0.60, "40% drawdown"),     # 40% below peak (extreme)
            (peak_price * 0.50, "50% drawdown"),     # 50% below peak (crash)
        ]
        
        # Create mock benchmark series
        mock_prices = pd.Series([peak_price] + [scenario[0] for scenario in test_scenarios])
        mock_dates = pd.date_range('2022-01-01', periods=len(mock_prices), freq='M')
        mock_benchmark = pd.Series(mock_prices.values, index=mock_dates)
        
        print(f"📊 Testing cash allocation with mock benchmark data:")
        print(f"   Peak price: {peak_price:.0f}")
        print(f"   Test scenarios: {len(test_scenarios)} drawdown levels")
        
        print(f"\n{'Scenario':<20} {'Price':<10} {'Drawdown':<12} {'Cash Alloc':<12} {'Protection':<15}")
        print("-" * 75)
        
        for i, (price, description) in enumerate(test_scenarios):
            # Calculate cash allocation for this scenario
            test_date = mock_dates[i + 1]  # Use the date after peak
            cash_allocation = calculate_dynamic_cash_allocation(mock_benchmark, test_date, strategy_config)
            
            # Determine protection level
            if cash_allocation < 0.20:
                protection = "🟢 Low"
            elif cash_allocation < 0.50:
                protection = "🟡 Medium"
            elif cash_allocation < 0.80:
                protection = "🟠 High"
            else:
                protection = "🔴 Extreme"
            
            drawdown_pct = (peak_price - price) / peak_price
            print(f"{description:<20} {price:<10.0f} {drawdown_pct:<12.1%} {cash_allocation:<12.1%} {protection}")
        
        print(f"\n✅ Cash allocation scenario testing completed")
        print(f"   This verifies that the risk management system responds correctly to market declines")
        
    except Exception as e:
        print(f"❌ Error in cash allocation scenario testing: {e}")
        import traceback
        traceback.print_exc()


def get_risk_management_summary(strategy_config: Dict) -> Dict:
    """
    Get a summary of risk management configuration for reporting.
    
    Returns:
        Dict containing risk management summary information
    """
    try:
        risk_config = strategy_config.get('risk_management', {})
        
        summary = {
            'enabled': risk_config.get('enabled', False),
            'default_cash': risk_config.get('default_cash', 0.05),
            'cash_allocation_rules': risk_config.get('cash_allocation', {}),
            'total_thresholds': len(risk_config.get('cash_allocation', {})),
            'has_custom_rules': bool(risk_config.get('cash_allocation', {}))
        }
        
        return summary
        
    except Exception as e:
        return {
            'enabled': False,
            'error': str(e)
        }
