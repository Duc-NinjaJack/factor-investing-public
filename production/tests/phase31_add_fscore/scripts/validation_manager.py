#!/usr/bin/env python3
"""
Validation Manager for QVM Strategy
==================================

This module handles all validation and sanity check operations:
- Strategy configuration validation
- Factor weights validation
- Risk management configuration validation
- Data validation and integrity checks

Author: Raymond
Created: August 17, 2025
"""

import logging
from typing import Dict, Tuple, Optional, Literal, Union
from datetime import datetime, date

try:
    # Prefer Pydantic v1 shim under Pydantic v2
    from pydantic.v1 import BaseModel, Field, validator
    PydanticV = 1
    ConfigDict = None  # type: ignore
except Exception:
    try:
        # Fallback to native import (works on Pydantic v2 or v1)
        from pydantic import BaseModel, Field, validator  # type: ignore
        try:
            from pydantic import ConfigDict  # type: ignore
            PydanticV = 2
        except Exception:
            ConfigDict = None  # type: ignore
            PydanticV = 1
    except Exception:  # pragma: no cover - pydantic optional at import time
        BaseModel = object  # type: ignore
        def Field(*args, **kwargs):  # type: ignore
            return None
        def validator(*args, **kwargs):  # type: ignore
            def _wrap(fn):
                return fn
            return _wrap
        ConfigDict = None  # type: ignore
        PydanticV = 1


def validate_strategy_config(strategy_config: Dict, logger: logging.Logger) -> bool:
    """
    Validate complete strategy configuration.
    
    Args:
        strategy_config: Strategy configuration dictionary
        logger: Logger instance for validation messages
    
    Returns:
        bool: True if configuration is valid
    
    Raises:
        ValueError: If required parameters are missing
    """
    try:
        logger.debug("🔍 Validating strategy configuration...")
        
        # Check if strategy_config exists and is not empty
        if not strategy_config:
            logger.error("❌ Strategy configuration is empty or missing")
            raise ValueError("Strategy configuration is empty or missing")
        
        # Check required sections
        required_sections = ['strategy', 'factor_weights', 'risk_management']
        for section in required_sections:
            if section not in strategy_config:
                logger.error(f"❌ Missing required section: {section}")
                logger.error("   Please ensure your strategy_config_v2_0_1_simple.yml contains:")
                logger.error("   - strategy:")
                logger.error("   - factor_weights:")
                logger.error("   - risk_management:")
                raise ValueError(f"Missing required section: {section}")
        
        # Check strategy parameters
        strategy_config_section = strategy_config['strategy']
        required_strategy_params = ['name', 'version', 'portfolio']
        for param in required_strategy_params:
            if param not in strategy_config_section:
                logger.error(f"❌ Missing strategy parameter: {param}")
                raise ValueError(f"Missing strategy parameter: {param}")
        
        # Check portfolio parameters
        portfolio_config = strategy_config_section['portfolio']
        required_portfolio_params = ['universe_size', 'portfolio_size', 'starting_capital']
        for param in required_portfolio_params:
            if param not in portfolio_config:
                logger.error(f"❌ Missing portfolio parameter: {param}")
                raise ValueError(f"Missing portfolio parameter: {param}")
        
        # Validate factor weights (comprehensive validation)
        _validate_factor_weights(strategy_config, logger)
        
        # Validate risk management configuration
        _validate_risk_management_config(strategy_config, logger)
        
        logger.info("✅ Strategy configuration validation completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Strategy configuration validation failed: {e}")
        raise


def _validate_factor_weights(strategy_config: Dict, logger: logging.Logger) -> None:
    """Validate that factor weights exist and sum to 1.0."""
    try:
        # Check if factor_weights section exists
        if 'factor_weights' not in strategy_config:
            logger.error("❌ No factor_weights section found in strategy configuration")
            logger.error("   Please add factor_weights section to your strategy_config_v2_0_1_simple.yml:")
            logger.error("   factor_weights:")
            logger.error("     quality: 0.25")
            logger.error("     value: 0.25")
            logger.error("     momentum: 0.25")
            logger.error("     defensive: 0.25")
            raise ValueError("Missing factor_weights section in strategy configuration")

        factor_weights = strategy_config['factor_weights']

        # Check if factor_weights is empty
        if not factor_weights:
            logger.error("❌ factor_weights section is empty")
            raise ValueError("factor_weights section is empty")

        # Check for required pillars
        required_pillars = {'quality', 'value', 'momentum', 'defensive'}
        missing_pillars = required_pillars - set(factor_weights.keys())

        if missing_pillars:
            logger.error(f"❌ Missing factor weights for pillars: {missing_pillars}")
            logger.error("   Please add these missing pillars to your configuration:")
            for pillar in missing_pillars:
                logger.error(f"     {pillar}: 0.25")
            raise ValueError(f"Missing factor weights for pillars: {missing_pillars}")

        # Validate weights are numeric and sum to 1.0
        try:
            numeric_weights = {k: float(v) for k, v in factor_weights.items()}
        except Exception:
            logger.error("❌ Factor weights must be numeric values")
            raise ValueError("Non-numeric factor weight detected")

        total_weight = sum(numeric_weights.values())

        if abs(total_weight - 1.0) > 0.001:  # Allow small floating point precision errors
            logger.error(f"❌ Factor weights do not sum to 1.0: {total_weight:.6f}")
            logger.error(f"   Current weights: {factor_weights}")
            logger.error("   Please ensure factor weights sum to 1.0")
            raise ValueError(f"Factor weights must sum to 1.0, got {total_weight:.6f}")

        # All validations passed
        logger.info(f"✅ Factor weights validation passed: {total_weight:.6f}")
        logger.info(f"   Weights: {numeric_weights}")

    except Exception as e:
        logger.error(f"❌ Factor weights validation failed: {e}")
        raise


def _validate_risk_management_config(strategy_config: Dict, logger: logging.Logger) -> None:
    """
    Validate risk management configuration completeness.
    
    Raises:
        ValueError: If required parameters are missing
    """
    try:
        # Check if risk management section exists
        if 'risk_management' not in strategy_config:
            logger.error("❌ Missing risk_management section in strategy configuration")
            raise ValueError("Risk management configuration missing")
        
        risk_config = strategy_config['risk_management']
        
        # Check if risk management is enabled
        if not risk_config.get('enabled', False):
            logger.warning("⚠️ Risk management is disabled")
            return
        
        # Check required risk management parameters
        required_params = ['cash_allocation', 'default_cash']
        for param in required_params:
            if param not in risk_config:
                logger.error(f"❌ Missing risk management parameter: {param}")
                raise ValueError(f"Missing risk management parameter: {param}")
        
        # Check if cash allocation thresholds are configured
        if 'cash_allocation' not in risk_config:
            logger.warning("⚠️ No cash_allocation thresholds found in strategy configuration")
            logger.warning("   Please add cash_allocation section to your strategy_config_v2_0_1_simple.yml:")
            logger.warning("   risk_management:")
            logger.warning("     cash_allocation:")
            logger.warning("       drawdown_5: 0.20")
            logger.warning("       drawdown_10: 0.40")
            logger.warning("       drawdown_15: 0.60")
            logger.warning("       drawdown_20: 0.80")
            logger.warning("       drawdown_25: 0.90")
        else:
            cash_config = risk_config['cash_allocation']
            required_thresholds = ['drawdown_5', 'drawdown_10', 'drawdown_15', 'drawdown_20', 'drawdown_25']
            missing_thresholds = [t for t in required_thresholds if t not in cash_config]
            if missing_thresholds:
                logger.warning(f"⚠️ Missing cash allocation thresholds: {missing_thresholds}")
                logger.warning("   Please add these thresholds to your configuration file")
            else:
                logger.debug("✅ All cash allocation thresholds found in configuration")
        
        logger.debug("✅ Risk management configuration validated successfully")
        
    except Exception as e:
        logger.error(f"❌ Risk management configuration validation failed: {e}")
        raise


def get_correct_quarter_for_date(analysis_date: datetime) -> Tuple[int, int]:
    """Get the correct year and quarter for a given date."""
    try:
        year = analysis_date.year
        quarter = (analysis_date.month - 1) // 3 + 1
        return year, quarter
    except Exception as e:
        raise ValueError(f"Failed to get quarter info: {e}")


def validate_portfolio_size(portfolio_size: int, required_size: int = 20) -> bool:
    """Validate that portfolio size matches the required size."""
    if portfolio_size != required_size:
        raise ValueError(f"Portfolio size {portfolio_size} differs from required {required_size} stocks")
    return True


def validate_factor_architecture(factor_weights: Dict[str, float]) -> bool:
    """Validate that 4-pillar architecture is properly configured."""
    expected_pillars = {'quality', 'value', 'momentum', 'defensive'}
    missing_pillars = expected_pillars - set(factor_weights.keys())
    
    if missing_pillars:
        raise ValueError(f"Missing factor weights for pillars: {missing_pillars}")
    
    return True


# -----------------------------
# Backtest config schema (Pydantic)
# -----------------------------

class RebalanceConfig(BaseModel):
    anchor: Literal['first_trading_day', 'mid_month', 'quarter_lag'] = 'first_trading_day'
    lag_days: int = Field(0, ge=0)
    # Preserve any extra keys (e.g., min_holding_months)
    if ConfigDict is not None:
        model_config = ConfigDict(extra='allow')  # type: ignore
    class Config:  # pydantic v1
        extra = 'allow'  # type: ignore

class FundamentalsConfig(BaseModel):
    reporting_lag_days: Optional[int] = Field(None, ge=0)
    if ConfigDict is not None:
        model_config = ConfigDict(extra='allow')  # type: ignore
    class Config:
        extra = 'allow'  # type: ignore

class BacktestWindow(BaseModel):
    start: date
    end: date

    @validator('end')
    def _end_not_before_start(cls, v, values):  # type: ignore
        start = values.get('start')
        if start and v < start:
            raise ValueError('end must be on/after start')
        return v

class BacktestConfigModel(BaseModel):
    backtest_windows: Dict[str, BacktestWindow]
    active_window: Union[str, Dict[str, date]]
    ic_hurdles: Optional[Dict[str, float]] = None
    rebalance: Optional[RebalanceConfig] = None
    fundamentals: Optional[FundamentalsConfig] = None
    transaction_cost_bps: float = Field(10.0, ge=0.0)
    slippage_bps: float = Field(0.0, ge=0.0)
    # Allow extra keys at the top-level (e.g., universe, cost_model, portfolio, risk_overlay, normalization, rebalance_anchor_policy)
    if ConfigDict is not None:
        model_config = ConfigDict(extra='allow')  # type: ignore
    class Config:  # pydantic v1
        extra = 'allow'  # type: ignore

    @validator('active_window')
    def _validate_active_window(cls, v):  # type: ignore
        if isinstance(v, dict):
            if 'start' not in v or 'end' not in v:
                raise ValueError("active_window dict requires 'start' and 'end'")
        elif not isinstance(v, str):
            raise ValueError('active_window must be str or dict with start/end')
        return v


def validate_backtest_config(backtest_config: Dict, logger: logging.Logger) -> Dict:
    """Validate and normalize backtest configuration using a strict schema.

    Returns the normalized config dict.
    """
    try:
        logger.debug("🔍 Validating backtest configuration (schema)…")
        # Pydantic v1/v2 compatibility: prefer model_validate/model_dump when available
        if hasattr(BacktestConfigModel, 'model_validate'):
            model = BacktestConfigModel.model_validate(backtest_config)  # type: ignore[attr-defined]
        else:
            model = BacktestConfigModel.parse_obj(backtest_config)  # type: ignore[attr-defined]

        # Normalize to JSON-serializable python-native dict
        if hasattr(model, 'model_dump'):
            # Pydantic v2: ensure JSON-friendly types (e.g., dates -> str)
            normalized: Dict = model.model_dump(mode='json')  # type: ignore[assignment]
        elif hasattr(model, 'json'):
            # Pydantic v1: serialize to JSON then parse back to dict
            import json as _json
            normalized = _json.loads(model.json())  # type: ignore[attr-defined]
        else:
            # Fallback: best-effort plain dict
            normalized = model.dict()  # type: ignore[assignment]
        logger.info("✅ Backtest configuration validation completed successfully")
        return normalized
    except Exception as e:
        logger.error(f"❌ Backtest configuration validation failed: {e}")
        raise
