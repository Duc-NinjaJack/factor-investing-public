"""
Aureus Sigma Capital - Universe Construction Module
==================================================

This module provides the core infrastructure for constructing investable
universes based on systematic, point-in-time correct rules.

Key Features:
- Liquid universe construction (ASC-VN-Liquid-150)
- Look-ahead bias prevention
- Quarterly refresh capability
- Comprehensive validation

Author: Duc Nguyen, Quantitative Finance Expert
Date Created: July 28, 2025
"""

from .constructors import get_liquid_universe, get_liquid_universe_dataframe

__version__ = "1.0.0"
__all__ = ["get_liquid_universe", "get_liquid_universe_dataframe"]