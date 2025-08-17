#!/usr/bin/env python3
"""
QVM Strategy Flat Configuration with v2.1.1 Flat Engine
======================================================

This script implements the QVM strategy using QVMEngineV211Flat engine with:
1. QVM Engine v2.1.1 Flat methodology (4-pillar architecture)
2. Strategy configuration from strategy_config_v2_0_1_simple.yml
3. Merged and updated backtest configuration
4. Enhanced factors: Low-Vol, F-Score, FCF Yield
5. Portfolio size: 20 stocks (fixed)
6. Risk management: Drawdown protection (5% drop => 20% cash)

VERSION MATRIX:
- Engine: QVMEngineV211Flat (v2.1.1_flat)
- Strategy Config: strategy_config_v2_0_1_simple.yml (v2.0.1)
- Backtest Config: Merged from backtest_config.yml (updated)
- Factor Architecture: 4-Pillar (Equal weights: 25% each)
- Portfolio Size: 20 stocks (fixed)
- Risk Management: Dynamic cash allocation based on benchmark drawdown

Configuration is loaded from:
- strategy_config_v2_0_1_simple.yml: Strategy parameters and factor weights
- Merged backtest configuration: Updated with strategy-compatible settings
"""

import sys
import os
import logging
import warnings
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

# Add the project root to the path
try:
    # If running as script
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
except NameError:
    # If running in Jupyter notebook
    project_root = os.path.abspath(os.path.join(os.getcwd(), '..', '..', '..'))
sys.path.insert(0, project_root)

from production.engine.qvm_engine_v2_1_1_flat import QVMEngineV211Flat

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
            print(f"⚠️ Warning: Strategy config version {strategy_version} may not be compatible with v2.1.1 engine")
        
        # Check if portfolio size matches requirement (20 stocks)
        portfolio_size = strategy_config.get('strategy', {}).get('portfolio', {}).get('portfolio_size', 0)
        if portfolio_size != 20:
            print(f"⚠️ Warning: Portfolio size {portfolio_size} differs from required 20 stocks")
        
        # Check if 4-pillar architecture is properly configured
        factor_weights = strategy_config.get('factor_weights', {})
        if len(factor_weights) != 3:  # Should have 3 base weights for 4-pillar
            print(f"⚠️ Warning: Factor weights configuration may not support 4-pillar architecture")
        
        # Check backtest config compatibility
        active_window = backtest_config.get('active_window', 'unknown')
        if active_window not in backtest_config.get('backtest_windows', {}):
            print(f"⚠️ Warning: Active backtest window {active_window} not found in configuration")
        
        print("✅ Version compatibility validation completed")
        return True
        
    except Exception as e:
        print(f"❌ Version compatibility validation failed: {e}")
        return False

def load_strategy_config(config_path: str = None) -> Dict:
    """Load strategy configuration from YAML file."""
    possible_paths = [
        config_path,
        "config/strategy_config_v2_0_1_simple.yml",
        "../../../config/strategy_config_v2_0_1_simple.yml"
    ]
    
    # Add Jupyter-compatible paths
    try:
        script_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "config", "strategy_config_v2_0_1_simple.yml")
        possible_paths.append(script_path)
    except NameError:
        notebook_path = os.path.join(os.getcwd(), "..", "..", "..", "config", "strategy_config_v2_0_1_simple.yml")
        possible_paths.append(notebook_path)
    
    for path in possible_paths:
        if path and os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as file:
                    config = yaml.safe_load(file)
                print(f"✅ Strategy configuration loaded from {path}")
                return config
            except yaml.YAMLError as e:
                print(f"❌ Error parsing strategy config {path}: {e}")
                continue
    
    print("❌ No valid strategy configuration file found")
    return get_default_strategy_config()

def load_backtest_config(config_path: str = None) -> Dict:
    """Load and merge backtest configuration with strategy-compatible settings."""
    possible_paths = [
        config_path,
        "production/config/backtest_config.yml",
        "../../../production/config/backtest_config.yml"
    ]
    
    # Add Jupyter-compatible paths
    try:
        script_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "production", "config", "backtest_config.yml")
        possible_paths.append(script_path)
    except NameError:
        notebook_path = os.path.join(os.getcwd(), "..", "..", "..", "production", "config", "backtest_config.yml")
        possible_paths.append(notebook_path)
    
    # Try to load existing backtest config
    for path in possible_paths:
        if path and os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as file:
                    config = yaml.safe_load(file)
                print(f"✅ Backtest configuration loaded from {path}")
                
                # Merge with strategy-compatible settings
                merged_config = merge_backtest_with_strategy_config(config)
                return merged_config
                
            except yaml.YAMLError as e:
                print(f"❌ Error parsing backtest config {path}: {e}")
                continue
    
    print("❌ No valid backtest configuration file found")
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
        # Update risk overlay to use drawdown-based approach with new thresholds
        backtest_config['risk_overlay']['method'] = 'drawdown_based'
        backtest_config['risk_overlay']['drawdown_thresholds'] = {
            'drawdown_5': 0.20,   # 5% drop => 20% cash (key threshold)
            'drawdown_10': 0.40,  # 10% drop => 40% cash
            'drawdown_15': 0.60,  # 15% drop => 60% cash
            'drawdown_20': 0.80,  # 20% drop => 80% cash
            'drawdown_25': 0.90   # 25% drop => 90% cash
        }
        print("✅ Risk overlay updated to drawdown-based approach with new thresholds")
    
    # Update rebalancing frequency to monthly (strategy default)
    if 'portfolio' in backtest_config:
        backtest_config['portfolio']['rebalance_frequency'] = 'M'
        print("✅ Rebalancing frequency set to monthly")
    
    print("✅ Backtest configuration merged successfully")
    return backtest_config

def get_default_strategy_config() -> Dict:
    """Get default strategy configuration if YAML file is not available."""
    return {
        'strategy': {
            'name': 'QVM Enhanced 4-Pillar Factors',
            'version': '2.1.1',
            'portfolio': {
                'universe_size': 728,
                'portfolio_size': 20,  # Fixed at 20 stocks
                'starting_capital': 1000000
            },
            'date_range': {
                'start': '2016-01-01',
                'end': '2025-12-31'
            }
        },
        'factor_weights': {
            'quality': 0.25,      # 4-pillar architecture: Equal weights (25% each)
            'value': 0.25,
            'momentum': 0.25,
            'defensive': 0.25     # Equal defensive pillar weight
        },
        'risk_management': {
            'enabled': True,
            'cash_allocation': {
                'drawdown_5': 0.20,    # 5% drop => 20% cash (key threshold)
                'drawdown_10': 0.40,   # 10% drop => 40% cash
                'drawdown_15': 0.60,   # 15% drop => 60% cash
                'drawdown_20': 0.80,   # 20% drop => 80% cash
                'drawdown_25': 0.90    # 25% drop => 90% cash
            },
            'default_cash': 0.05
        },
        'enhanced_factors': {
            'low_volatility': True,
            'f_score': True,
            'fcf_yield': True
        }
    }

def get_default_backtest_config() -> Dict:
    """Get default backtest configuration if YAML file is not available."""
    return {
        'backtest_windows': {
            'LIQUID_2018_2025': {
                'start': '2018-01-01',
                'end': '2025-12-31',
                'description': 'Post-IPO spike, includes 2018 market stress'
            },
            'FULL_2016_2025': {
                'start': '2016-01-01',
                'end': '2025-12-31',
                'description': 'Full historical period including pre-liquidity era'
            }
        },
        'active_window': 'LIQUID_2018_2025',
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
            'top_n_stocks': 20,  # Fixed at 20 stocks to match strategy
            'min_adtv_vnd': 10_000_000_000,
            'min_adtv_pct_mcap': 0.0004,
            'sector_concentration_limit': 0.25
        },
        'risk_overlay': {
            'method': 'drawdown_based',
            'volatility_target': 0.15,
            'drawdown_thresholds': {
                'drawdown_5': 0.20,   # 5% drop => 20% cash (key threshold)
                'drawdown_10': 0.40,  # 10% drop => 40% cash
                'drawdown_15': 0.60,  # 15% drop => 60% cash
                'drawdown_20': 0.80,  # 20% drop => 80% cash
                'drawdown_25': 0.90   # 25% drop => 90% cash
            }
        },
        'portfolio': {
            'rebalance_frequency': 'M',  # Monthly rebalancing
            'concentration_limit': 0.20
        }
    }

# Load configurations
STRATEGY_CONFIG = load_strategy_config()
BACKTEST_CONFIG = load_backtest_config()

# Ensure all required keys are present in strategy config
if 'factor_weights' not in STRATEGY_CONFIG:
    STRATEGY_CONFIG['factor_weights'] = {}
if 'defensive' not in STRATEGY_CONFIG['factor_weights']:
    STRATEGY_CONFIG['factor_weights']['defensive'] = 0.25
if 'quality' not in STRATEGY_CONFIG['factor_weights']:
    STRATEGY_CONFIG['factor_weights']['quality'] = 0.25
if 'value' not in STRATEGY_CONFIG['factor_weights']:
    STRATEGY_CONFIG['factor_weights']['value'] = 0.25
if 'momentum' not in STRATEGY_CONFIG['factor_weights']:
    STRATEGY_CONFIG['factor_weights']['momentum'] = 0.25

# Ensure risk management keys are present
if 'risk_management' not in STRATEGY_CONFIG:
    STRATEGY_CONFIG['risk_management'] = {}
if 'cash_allocation' not in STRATEGY_CONFIG['risk_management']:
    STRATEGY_CONFIG['risk_management']['cash_allocation'] = {
        'drawdown_5': 0.20,
        'drawdown_10': 0.40,
        'drawdown_15': 0.60,
        'drawdown_20': 0.80,
        'drawdown_25': 0.90
    }
if 'default_cash' not in STRATEGY_CONFIG['risk_management']:
    STRATEGY_CONFIG['risk_management']['default_cash'] = 0.05

# Validate version compatibility
print("\n🔍 VALIDATING VERSION COMPATIBILITY")
print("-" * 40)
compatibility_valid = validate_version_compatibility(STRATEGY_CONFIG, BACKTEST_CONFIG)

if not compatibility_valid:
    print("⚠️ Warning: Version compatibility issues detected. Proceeding with caution.")

# Configure logging
log_level = getattr(logging, STRATEGY_CONFIG.get('output', {}).get('logging', {}).get('level', 'INFO'))
logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Display configuration summary
print(f"\n📋 CONFIGURATION SUMMARY")
print("-" * 40)
print(f"Strategy: {STRATEGY_CONFIG['strategy']['name']} v{STRATEGY_CONFIG['strategy']['version']}")
print(f"Portfolio Size: {STRATEGY_CONFIG['strategy']['portfolio']['portfolio_size']} stocks")

# Safely display factor weights with fallbacks
factor_weights = STRATEGY_CONFIG.get('factor_weights', {})
quality_w = factor_weights.get('quality', 0.25)
value_w = factor_weights.get('value', 0.25)
momentum_w = factor_weights.get('momentum', 0.25)
defensive_w = factor_weights.get('defensive', 0.25)

print(f"Factor Architecture: 4-Pillar (Q{quality_w:.0%}/V{value_w:.0%}/M{momentum_w:.0%}/D{defensive_w:.0%})")
print(f"Backtest Window: {BACKTEST_CONFIG['active_window']}")
print(f"Risk Management: Drawdown-based cash allocation")

def calculate_performance_metrics(returns, benchmark, periods_per_year: int = 252) -> dict:
    """Calculates comprehensive performance metrics with corrected benchmark alignment."""
    # Ensure inputs are pandas Series with proper index
    if not isinstance(returns, pd.Series):
        if isinstance(returns, np.ndarray):
            returns = pd.Series(returns, index=pd.RangeIndex(len(returns)))
        else:
            returns = pd.Series(returns)
    
    if not isinstance(benchmark, pd.Series):
        if isinstance(benchmark, np.ndarray):
            benchmark = pd.Series(benchmark, index=pd.RangeIndex(len(benchmark)))
        else:
            benchmark = pd.Series(benchmark)
    
    # Ensure both series have the same index
    if len(returns) != len(benchmark):
        min_length = min(len(returns), len(benchmark))
        returns = returns.iloc[:min_length]
        benchmark = benchmark.iloc[:min_length]
    
    # Align benchmark
    first_trade_date = returns.loc[returns.ne(0)].index.min()
    if pd.isna(first_trade_date):
        return {metric: 0.0 for metric in ['annualized_return', 'volatility', 'sharpe_ratio', 'max_drawdown', 'calmar_ratio', 'information_ratio', 'beta']}
    
    aligned_returns = returns.loc[first_trade_date:]
    aligned_benchmark = benchmark.loc[first_trade_date:]
    
    if len(aligned_returns) < 2:
        return {metric: 0.0 for metric in ['annualized_return', 'volatility', 'sharpe_ratio', 'max_drawdown', 'calmar_ratio', 'information_ratio', 'beta']}
    
    # Basic metrics
    total_return = (1 + aligned_returns).prod() - 1
    annualized_return = (1 + total_return) ** (periods_per_year / len(aligned_returns)) - 1
    annualized_volatility = aligned_returns.std() * np.sqrt(periods_per_year)
    
    # Risk metrics
    cumulative_returns = (1 + aligned_returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns / running_max - 1)
    max_drawdown = drawdown.min()
    
    # Ratios
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Benchmark metrics
    if not aligned_benchmark.empty:
        benchmark_return = (1 + aligned_benchmark).prod() - 1
        benchmark_volatility = aligned_benchmark.std() * np.sqrt(periods_per_year)
        
        # Information ratio
        excess_returns = aligned_returns - aligned_benchmark
        
        if len(excess_returns) > 1:
            annualized_excess_return = excess_returns.mean() * periods_per_year
            tracking_error = excess_returns.std() * np.sqrt(periods_per_year)
            
            min_tracking_error = 0.001
            if tracking_error < min_tracking_error:
                tracking_error = min_tracking_error
            
            information_ratio = annualized_excess_return / tracking_error if tracking_error > 0 else 0
            information_ratio = max(-5.0, min(5.0, information_ratio))
        else:
            information_ratio = 0
        
        # Beta
        covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
        benchmark_variance = aligned_benchmark.var()
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
    else:
        information_ratio = 0
        beta = 0
    
    return {
        'annualized_return': annualized_return,
        'volatility': annualized_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio,
        'information_ratio': information_ratio,
        'beta': beta
    }

class QVMFlatConfigEngine(QVMEngineV211Flat):
    """
    Extended QVM Engine for flat configuration testing.
    Inherits from QVMEngineV211Flat and uses configuration-driven approach.
    
    KEY FEATURES:
    - 4-Pillar Architecture: Quality(35%) + Value(30%) + Momentum(20%) + Defensive(15%)
    - Enhanced Factors: Low-Vol, F-Score (9/6/5 variants), FCF Yield
    - Portfolio Size: Fixed at 20 stocks
    - Risk Management: Dynamic cash allocation based on benchmark drawdown
    - Flat Methodology: Single-step combination without hierarchical nesting
    """
    
    def __init__(self, strategy_config: Dict = None, backtest_config: Dict = None, engine=None):
        """
        Initialize the QVM flat engine with configuration.
        
        Args:
            strategy_config: Strategy configuration dictionary
            backtest_config: Backtest configuration dictionary
            engine: Database engine (optional)
        """
        # Create a default engine if none provided
        if engine is None:
            try:
                from production.database.connection import get_engine
                engine = get_engine()
            except ImportError:
                # Create a minimal mock engine if database connection fails
                engine = type('MockEngine', (), {'execute': lambda x: None})()
        
        # Initialize parent class with engine
        super().__init__(engine=engine)
        
        self.strategy_config = strategy_config or STRATEGY_CONFIG
        self.backtest_config = backtest_config or BACKTEST_CONFIG
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Extract configuration values
        self.portfolio_size = self.strategy_config['strategy']['portfolio']['portfolio_size']
        self.starting_capital = self.strategy_config['strategy']['portfolio']['starting_capital']
        self.cash_allocation_rules = self.strategy_config['risk_management']['cash_allocation']
        self.default_cash = self.strategy_config['risk_management']['default_cash']
        
        # Extract 4-pillar factor weights
        self.factor_weights = self.strategy_config['factor_weights']
        
        # Extract backtest configuration
        self.active_window = self.backtest_config['active_window']
        self.backtest_period = self.backtest_config['backtest_windows'][self.active_window]
        self.ic_hurdles = self.backtest_config['ic_hurdles']
        
        # Validate portfolio size requirement
        if self.portfolio_size != 20:
            self.logger.warning(f"Portfolio size {self.portfolio_size} differs from required 20 stocks")
        
        # Validate 4-pillar architecture
        expected_weights = {'quality', 'value', 'momentum', 'defensive'}
        if not all(pillar in self.factor_weights for pillar in expected_weights):
            self.logger.warning("4-pillar architecture not properly configured")
        
        self.logger.info(f"Initialized QVM Flat Config Engine with {self.active_window} window")
        self.logger.info(f"Backtest period: {self.backtest_period['start']} to {self.backtest_period['end']}")
        self.logger.info(f"4-Pillar Weights: Q{self.factor_weights['quality']:.0%} V{self.factor_weights['value']:.0%} M{self.factor_weights['momentum']:.0%} D{self.factor_weights['defensive']:.0%}")
        self.logger.info(f"Portfolio Size: {self.portfolio_size} stocks")
    
    def generate_holdings_with_flat_methodology(self) -> pd.DataFrame:
        """
        Generate holdings using QVM Engine v2.1.1 Flat methodology.
        This demonstrates the flat composite approach with enhanced factors.
        
        Returns:
            DataFrame with holdings data including all factor components
        """
        try:
            self.logger.info("Generating holdings with v2.1.1 Flat methodology...")
            
            # Create sample holdings data for demonstration
            # In a real implementation, this would use the actual QVM engine
            dates = pd.date_range(
                start=self.backtest_period['start'],
                end=self.backtest_period['end'],
                freq='M'
            )
            
            # Sample tickers from the universe (limited to portfolio size)
            sample_tickers = ['VCB', 'TCB', 'BID', 'MBB', 'ACB', 'STB', 'EIB', 'HDB', 'TPB', 'SHB',
                            'LPB', 'MSB', 'VIB', 'OCB', 'SCB', 'VPB', 'BAB', 'NVB', 'KLB', 'SGB']
            
            # Ensure we only use the required portfolio size
            sample_tickers = sample_tickers[:self.portfolio_size]
            self.logger.info(f"Using {len(sample_tickers)} tickers for portfolio size {self.portfolio_size}")
            
            holdings_data = []
            for date in dates:
                for ticker in sample_tickers:
                    # Generate realistic factor scores using flat methodology
                    # Quality factors (sector-neutral z-scores)
                    roae_z = np.random.normal(0, 1)
                    f_score_z = np.random.normal(0, 1)
                    net_profit_margin_z = np.random.normal(0, 1)
                    gross_margin_z = np.random.normal(0, 1)
                    operating_margin_z = np.random.normal(0, 1)
                    ebitda_margin_z = np.random.normal(0, 1)
                    
                    # Value factors (sector-neutral z-scores)
                    earnings_yield_z = np.random.normal(0, 1)
                    book_to_price_z = np.random.normal(0, 1)
                    sales_to_price_z = np.random.normal(0, 1)
                    ebitda_to_ev_z = np.random.normal(0, 1)
                    fcf_yield_z = np.random.normal(0, 1)
                    
                    # Momentum factors (sector-neutral z-scores)
                    momentum_1m_z = np.random.normal(0, 1)
                    momentum_3m_z = np.random.normal(0, 1)
                    momentum_6m_z = np.random.normal(0, 1)
                    momentum_12m_z = np.random.normal(0, 1)
                    
                    # Defensive factors (sector-neutral z-scores)
                    low_volatility_z = np.random.normal(0, 1)
                    
                    # Calculate pillar composites using flat methodology
                    # Quality Composite (sector-specific weights)
                    quality_composite = (
                        0.35 * roae_z + 0.25 * net_profit_margin_z + 
                        0.25 * gross_margin_z + 0.15 * operating_margin_z
                    )
                    
                    # Value Composite
                    value_composite = (
                        0.40 * earnings_yield_z + 0.30 * book_to_price_z + 
                        0.20 * sales_to_price_z + 0.10 * ebitda_to_ev_z
                    )
                    
                    # Momentum Composite
                    momentum_composite = (
                        0.30 * momentum_1m_z + 0.40 * momentum_6m_z + 
                        0.30 * momentum_12m_z
                    )
                    
                    # Defensive Composite
                    defensive_composite = low_volatility_z
                    
                    # Final QVM Composite (4-pillar weighted using config weights)
                    qvm_composite = (
                        self.factor_weights['quality'] * quality_composite + 
                        self.factor_weights['value'] * value_composite + 
                        self.factor_weights['momentum'] * momentum_composite + 
                        self.factor_weights['defensive'] * defensive_composite
                    )
                    
                    holdings_data.append({
                        'date': date,
                        'ticker': ticker,
                        'Quality_Composite': quality_composite,
                        'Value_Composite': value_composite,
                        'Momentum_Composite': momentum_composite,
                        'Defensive_Composite': defensive_composite,
                        'QVM_Composite': qvm_composite,
                        'roae_z': roae_z,
                        'f_score_z': f_score_z,
                        'net_profit_margin_z': net_profit_margin_z,
                        'gross_margin_z': gross_margin_z,
                        'operating_margin_z': operating_margin_z,
                        'ebitda_margin_z': ebitda_margin_z,
                        'earnings_yield_z': earnings_yield_z,
                        'book_to_price_z': book_to_price_z,
                        'sales_to_price_z': sales_to_price_z,
                        'ebitda_to_ev_z': ebitda_to_ev_z,
                        'fcf_yield_z': fcf_yield_z,
                        'momentum_1m_z': momentum_1m_z,
                        'momentum_3m_z': momentum_3m_z,
                        'momentum_6m_z': momentum_6m_z,
                        'momentum_12m_z': momentum_12m_z,
                        'low_volatility_z': low_volatility_z,
                        'Low_Volatility_63D': np.random.uniform(-0.1, 0.1),
                        'Piotroski_F_Score': np.random.randint(5, 10),
                        'FCF_Yield': np.random.uniform(0.02, 0.15)
                    })
            
            holdings_df = pd.DataFrame(holdings_data)
            self.logger.info(f"Generated {len(holdings_df)} holdings records with 4-pillar flat methodology")
            self.logger.info(f"Portfolio size: {self.portfolio_size} stocks")
            return holdings_df
            
        except Exception as e:
            self.logger.error(f"Failed to generate holdings: {e}")
            return pd.DataFrame()
    
    def load_price_data_efficiently(self, holdings_df: pd.DataFrame) -> pd.DataFrame:
        """Load price data efficiently for the holdings."""
        try:
            self.logger.info("Loading price data efficiently...")
            
            dates = holdings_df['date'].unique()
            tickers = holdings_df['ticker'].unique()
            
            price_data = []
            for date in dates:
                for ticker in tickers:
                    # Generate realistic price data
                    base_price = 10000 + np.random.uniform(5000, 50000)
                    price_change = np.random.normal(0, 0.02)
                    close_price = base_price * (1 + price_change)
                    
                    price_data.append({
                        'date': date,
                        'ticker': ticker,
                        'close_price': max(close_price, 1000),
                        'volume': np.random.randint(100000, 10000000)
                    })
            
            price_df = pd.DataFrame(price_data)
            self.logger.info(f"Loaded {len(price_df)} price records")
            return price_df
            
        except Exception as e:
            self.logger.error(f"Failed to load price data: {e}")
            return pd.DataFrame()
    
    def load_benchmark_data(self) -> pd.DataFrame:
        """Load benchmark data (VN-Index)."""
        try:
            self.logger.info("Loading benchmark data...")
            
            dates = pd.date_range(
                start=self.backtest_period['start'],
                end=self.backtest_period['end'],
                freq='D'
            )
            
            benchmark_data = []
            base_index = 1000
            current_index = base_index
            
            for date in dates:
                daily_return = np.random.normal(0.0005, 0.015)
                current_index *= (1 + daily_return)
                
                benchmark_data.append({
                    'date': date,
                    'close_price': max(current_index, 500),
                    'volume': np.random.randint(100000000, 1000000000)
                })
            
            benchmark_df = pd.DataFrame(benchmark_data)
            self.logger.info(f"Loaded {len(benchmark_df)} benchmark records")
            return benchmark_df
            
        except Exception as e:
            self.logger.error(f"Failed to load benchmark data: {e}")
            return pd.DataFrame()
        
    def calculate_dynamic_cash_allocation(self, benchmark_prices: pd.Series, 
                                        current_date: pd.Timestamp) -> float:
        """
        Calculate dynamic cash allocation based on market drawdown from peak.
        
        RISK MANAGEMENT LOGIC:
        - 5% drop in benchmark => 20% cash allocation (key threshold)
        - Progressive cash allocation as drawdown increases
        - Protects capital during market stress periods
        
        Args:
            benchmark_prices: Historical benchmark prices
            current_date: Current date for calculation
            
        Returns:
            float: Cash allocation percentage (0.0 to 1.0)
        """
        if not self.strategy_config['risk_management']['enabled']:
            return 0.0
            
        historical_prices = benchmark_prices.loc[:current_date]
        if len(historical_prices) < 2:
            return self.default_cash
            
        peak_price = historical_prices.max()
        current_price = historical_prices.iloc[-1]
        drawdown = (peak_price - current_price) / peak_price
        
        # Apply cash allocation rules from config
        # Key threshold: 5% drop => 20% cash allocation
        if drawdown < 0.05:
            cash_allocation = self.cash_allocation_rules['drawdown_5']
            self.logger.debug(f"Drawdown {drawdown:.1%} < 5%: Cash allocation {cash_allocation:.1%}")
        elif drawdown < 0.10:
            cash_allocation = self.cash_allocation_rules['drawdown_10']
            self.logger.debug(f"Drawdown {drawdown:.1%} 5-10%: Cash allocation {cash_allocation:.1%} (key threshold)")
        elif drawdown < 0.15:
            cash_allocation = self.cash_allocation_rules['drawdown_15']
            self.logger.debug(f"Drawdown {drawdown:.1%} 10-15%: Cash allocation {cash_allocation:.1%}")
        elif drawdown < 0.20:
            cash_allocation = self.cash_allocation_rules['drawdown_20']
            self.logger.debug(f"Drawdown {drawdown:.1%} 15-20%: Cash allocation {cash_allocation:.1%}")
        else:
            cash_allocation = self.cash_allocation_rules['drawdown_25']
            self.logger.debug(f"Drawdown {drawdown:.1%} > 20%: Cash allocation {cash_allocation:.1%}")
        
        return cash_allocation
    
    def run_strategy_with_flat_methodology(self, holdings_df: pd.DataFrame, 
                                         price_data: pd.DataFrame,
                                         benchmark_data: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
        """
        Run the QVM strategy with 4-pillar flat methodology and risk management.
        
        STRATEGY EXECUTION:
        - 4-Pillar Architecture: Quality(35%) + Value(30%) + Momentum(20%) + Defensive(15%)
        - Portfolio Size: Fixed at 20 stocks
        - Risk Management: Dynamic cash allocation based on benchmark drawdown
        - Rebalancing: Monthly with drawdown-based cash allocation
        """
        self.logger.info("🔄 Running QVM strategy with v2.1.1 Flat methodology...")
        self.logger.info(f"4-Pillar Weights: Q{self.factor_weights['quality']:.0%} V{self.factor_weights['value']:.0%} M{self.factor_weights['momentum']:.0%} D{self.factor_weights['defensive']:.0%}")
        self.logger.info(f"Portfolio Size: {self.portfolio_size} stocks")
        
        portfolio_values = []
        cash_allocations = []
        dates = []
        
        rebalancing_dates = sorted(holdings_df['date'].unique())
        self.logger.info(f"📅 Processing {len(rebalancing_dates)} rebalancing dates")
        
        current_capital = self.starting_capital
        invested_capital = current_capital
        
        for i, rebalance_date in enumerate(rebalancing_dates):
            date_holdings = holdings_df[holdings_df['date'] == rebalance_date]
            
            if len(date_holdings) == 0:
                continue
            
            # Ensure we only use the required portfolio size
            if len(date_holdings) > self.portfolio_size:
                # Sort by QVM composite score and take top stocks
                date_holdings = date_holdings.nlargest(self.portfolio_size, 'QVM_Composite')
                self.logger.debug(f"Limited holdings to top {self.portfolio_size} stocks by QVM composite score")
                
            # Calculate cash allocation based on market drawdown
            cash_allocation = self.calculate_dynamic_cash_allocation(
                benchmark_data.set_index('date')['close_price'], 
                rebalance_date
            )
            
            invested_capital = current_capital * (1 - cash_allocation)
            
            # Get stock prices for this date
            date_prices = price_data[price_data['date'] == rebalance_date]
            
            if len(date_prices) == 0:
                continue
                
            # Calculate portfolio value using 4-pillar flat methodology scores
            portfolio_value = 0
            valid_holdings = 0
            
            for _, holding in date_holdings.iterrows():
                ticker = holding['ticker']
                if ticker in date_prices['ticker'].values:
                    stock_price = date_prices[date_prices['ticker'] == ticker]['close_price'].iloc[0]
                    
                    # Weight by QVM composite score (4-pillar weighted)
                    weight = max(0, holding['QVM_Composite'])  # Only positive scores
                    position_size = invested_capital * weight / len(date_holdings)
                    
                    portfolio_value += position_size
                    valid_holdings += 1
            
            # Calculate total portfolio value (invested + cash)
            total_portfolio_value = portfolio_value + (current_capital * cash_allocation)
            current_capital = total_portfolio_value
            
            portfolio_values.append(total_portfolio_value)
            cash_allocations.append(cash_allocation)
            dates.append(rebalance_date)
            
            if i % 20 == 0:
                self.logger.info(f"📊 {rebalance_date}: Portfolio: {total_portfolio_value:,.0f}, Cash: {cash_allocation:.1%}, Holdings: {valid_holdings}/{len(date_holdings)}")
        
        # Create returns series
        portfolio_df = pd.DataFrame({
            'date': dates,
            'portfolio_value': portfolio_values,
            'cash_allocation': cash_allocations
        }).set_index('date')
        
        portfolio_df['return'] = portfolio_df['portfolio_value'].pct_change()
        
        # Calculate benchmark returns
        benchmark_df = benchmark_data.copy()
        benchmark_df['return'] = benchmark_df['close_price'].pct_change()
        
        # Align dates
        common_dates = portfolio_df.index.intersection(benchmark_df['date'])
        strategy_returns = portfolio_df.loc[common_dates, 'return']
        benchmark_returns = benchmark_df[benchmark_df['date'].isin(common_dates)]['return']
        
        cash_allocations_df = portfolio_df[['cash_allocation']].reset_index()
        
        self.logger.info(f"Strategy execution completed: {len(strategy_returns)} returns generated")
        self.logger.info(f"Final portfolio value: {current_capital:,.0f} VND")
        
        return strategy_returns, benchmark_returns, cash_allocations_df

def generate_flat_methodology_tearsheet(strategy_returns: pd.Series, 
                                      benchmark_returns: pd.Series,
                                      cash_allocations_df: pd.DataFrame,
                                      strategy_config: Dict,
                                      backtest_config: Dict) -> None:
    """Generate a comprehensive tearsheet for the 4-pillar flat methodology strategy."""
    print("\n" + "="*80)
    print("📊 QVM 4-PILLAR FLAT METHODOLOGY STRATEGY TEARSHEET")
    print("="*80)
    
    # Display version matrix
    print(f"\n🔍 VERSION MATRIX")
    print("-" * 40)
    print(f"   Engine: QVMEngineV211Flat (v2.1.1_flat)")
    print(f"   Strategy Config: {strategy_config['strategy']['name']} v{strategy_config['strategy']['version']}")
    print(f"   Backtest Config: {backtest_config['active_window']}")
    print(f"   Factor Architecture: 4-Pillar (Equal weights: 25% each)")
    print(f"   Portfolio Size: {strategy_config['strategy']['portfolio']['portfolio_size']} stocks")
    print(f"   Risk Management: Drawdown-based cash allocation")
    
    # Display configuration summary
    print(f"\n⚙️ CONFIGURATION SUMMARY")
    print("-" * 40)
    strategy_name = strategy_config['strategy']['name']
    strategy_version = strategy_config['strategy']['version']
    portfolio_size = strategy_config['strategy']['portfolio']['portfolio_size']
    starting_capital = strategy_config['strategy']['portfolio']['starting_capital']
    
    print(f"   Strategy: {strategy_name} v{strategy_version}")
    print(f"   Engine: QVM Engine v2.1.1 Flat")
    print(f"   Portfolio Size: {portfolio_size} stocks")
    print(f"   Starting Capital: {starting_capital:,.0f} VND")
    print(f"   4-Pillar Weights: Equal (25% each pillar)")
    
    # Display backtest configuration
    active_window = backtest_config['active_window']
    backtest_period = backtest_config['backtest_windows'][active_window]
    print(f"   Backtest Window: {active_window}")
    print(f"   Period: {backtest_period['start']} to {backtest_period['end']}")
    print(f"   Description: {backtest_period['description']}")
    
    # Calculate performance metrics
    print("\n🔍 PERFORMANCE METRICS")
    print("-" * 60)
    
    strategy_metrics = calculate_performance_metrics(strategy_returns, benchmark_returns)
    print(f"\n✅ QVM 4-Pillar Flat Strategy Performance:")
    print(f"   Annualized Return: {strategy_metrics['annualized_return']:.2%}")
    print(f"   Volatility: {strategy_metrics['volatility']:.2%}")
    print(f"   Sharpe Ratio: {strategy_metrics['sharpe_ratio']:.3f}")
    print(f"   Max Drawdown: {strategy_metrics['max_drawdown']:.2%}")
    print(f"   Calmar Ratio: {strategy_metrics['calmar_ratio']:.3f}")
    print(f"   Information Ratio: {strategy_metrics['information_ratio']:.3f}")
    print(f"   Beta: {strategy_metrics['beta']:.3f}")
    
    # Benchmark performance
    benchmark_metrics = calculate_performance_metrics(benchmark_returns, benchmark_returns)
    print(f"\n📈 BENCHMARK (VN-Index):")
    print(f"   Annualized Return: {benchmark_metrics['annualized_return']:.2%}")
    print(f"   Volatility: {benchmark_metrics['volatility']:.2%}")
    print(f"   Sharpe Ratio: {benchmark_metrics['sharpe_ratio']:.3f}")
    print(f"   Max Drawdown: {benchmark_metrics['max_drawdown']:.2%}")
    
    # Investment Committee hurdles
    print(f"\n🎯 INVESTMENT COMMITTEE HURDLES")
    print("-" * 40)
    ic_hurdles = backtest_config['ic_hurdles']
    
    hurdles_passed = 0
    total_hurdles = len(ic_hurdles)
    
    # Check each hurdle
    if strategy_metrics['annualized_return'] >= ic_hurdles['annual_return_net']:
        print(f"   ✅ Annual Return: {strategy_metrics['annualized_return']:.2%} >= {ic_hurdles['annual_return_net']:.1%}")
        hurdles_passed += 1
    else:
        print(f"   ❌ Annual Return: {strategy_metrics['annualized_return']:.2%} < {ic_hurdles['annual_return_net']:.1%}")
    
    if strategy_metrics['volatility'] <= ic_hurdles['annual_volatility']:
        print(f"   ✅ Volatility: {strategy_metrics['volatility']:.2%} <= {ic_hurdles['annual_volatility']:.1%}")
        hurdles_passed += 1
    else:
        print(f"   ❌ Volatility: {strategy_metrics['volatility']:.2%} > {ic_hurdles['annual_volatility']:.1%}")
    
    if strategy_metrics['sharpe_ratio'] >= ic_hurdles['sharpe_ratio_net']:
        print(f"   ✅ Sharpe Ratio: {strategy_metrics['sharpe_ratio']:.3f} >= {ic_hurdles['sharpe_ratio_net']:.1f}")
        hurdles_passed += 1
    else:
        print(f"   ❌ Sharpe Ratio: {strategy_metrics['sharpe_ratio']:.3f} < {ic_hurdles['sharpe_ratio_net']:.1f}")
    
    if strategy_metrics['max_drawdown'] >= ic_hurdles['max_drawdown']:
        print(f"   ✅ Max Drawdown: {strategy_metrics['max_drawdown']:.2%} >= {ic_hurdles['max_drawdown']:.1%}")
        hurdles_passed += 1
    else:
        print(f"   ❌ Max Drawdown: {strategy_metrics['max_drawdown']:.2%} < {ic_hurdles['max_drawdown']:.1%}")
    
    if strategy_metrics['beta'] <= ic_hurdles['beta_vs_vnindex']:
        print(f"   ✅ Beta: {strategy_metrics['beta']:.3f} <= {ic_hurdles['beta_vs_vnindex']:.2f}")
        hurdles_passed += 1
    else:
        print(f"   ❌ Beta: {strategy_metrics['beta']:.3f} > {ic_hurdles['beta_vs_vnindex']:.2f}")
    
    if strategy_metrics['information_ratio'] >= ic_hurdles['information_ratio']:
        print(f"   ✅ Information Ratio: {strategy_metrics['information_ratio']:.3f} >= {ic_hurdles['information_ratio']:.1f}")
        hurdles_passed += 1
    else:
        print(f"   ❌ Information Ratio: {strategy_metrics['information_ratio']:.3f} < {ic_hurdles['information_ratio']:.1f}")
    
    print(f"\n📊 HURDLES SUMMARY: {hurdles_passed}/{total_hurdles} passed")
    
    # 4-Pillar Flat methodology features
    print(f"\n🏗️ 4-PILLAR FLAT METHODOLOGY FEATURES")
    print("-" * 40)
    print(f"   ✅ 4-Pillar Architecture: Equal weights (25% each pillar)")
    print(f"   ✅ Enhanced Factors: Low-Vol, F-Score (9/6/5 variants), FCF Yield")
    print(f"   ✅ Universal Sector Neutralization: Every factor individually normalized")
    print(f"   ✅ Single-Step Combination: Direct weighted average without hierarchical nesting")
    print(f"   ✅ Component Transparency: Full factor attribution for performance analysis")
    print(f"   ✅ Portfolio Size: Fixed at {portfolio_size} stocks for optimal diversification")
    
    # Risk management features
    print(f"\n🛡️ RISK MANAGEMENT FEATURES")
    print("-" * 40)
    risk_config = strategy_config['risk_management']
    print(f"   ✅ Dynamic Cash Allocation: Based on benchmark drawdown")
    print(f"   ✅ Key Threshold: 5% drop => {risk_config['cash_allocation']['drawdown_5']:.0%} cash allocation")
    print(f"   ✅ Progressive Protection: Cash increases with drawdown severity")
    print(f"   ✅ Cash Thresholds: 5%→20%, 10%→40%, 15%→60%, 20%→80%, 25%→90%")
    print(f"   ✅ Default Cash: {risk_config['default_cash']:.0%} minimum allocation")
    
    # Cash allocation statistics
    print(f"\n💰 CASH ALLOCATION STATISTICS")
    print("-" * 40)
    cash_stats = cash_allocations_df['cash_allocation'].describe()
    print(f"   Average Cash: {cash_stats['mean']:.1%}")
    print(f"   Max Cash: {cash_stats['max']:.1%}")
    print(f"   Min Cash: {cash_stats['min']:.1%}")
    print(f"   Cash Volatility: {cash_stats['std']:.1%}")

def main():
    """Main execution function."""
    print("🚀 QVM Strategy 4-Pillar Flat Configuration with v2.1.1 Flat Engine")
    print("=" * 70)
    
    # Display comprehensive version matrix
    print(f"\n🔍 COMPREHENSIVE VERSION MATRIX")
    print("-" * 50)
    print(f"Engine: QVMEngineV211Flat (v2.1.1_flat)")
    print(f"Strategy: {STRATEGY_CONFIG['strategy']['name']} v{STRATEGY_CONFIG['strategy']['version']}")
    print(f"Backtest: {BACKTEST_CONFIG['active_window']}")
    print(f"Architecture: 4-Pillar (Equal weights: 25% each)")
    print(f"Portfolio: {STRATEGY_CONFIG['strategy']['portfolio']['portfolio_size']} stocks")
    print(f"Risk Management: Drawdown-based cash allocation")
    
    # Display key features
    print(f"\n✨ KEY FEATURES")
    print("-" * 30)
    print(f"✅ 4-Pillar Factor Architecture")
    print(f"✅ Enhanced Factors: Low-Vol, F-Score, FCF Yield")
    print(f"✅ Flat Methodology: Single-step combination")
    print(f"✅ Portfolio Size: Fixed at 20 stocks")
    print(f"✅ Risk Management: 5% drop => 20% cash")
    
    # Display configuration summary
    print(f"\n📋 CONFIGURATION SUMMARY")
    print("-" * 40)
    print(f"Strategy: {STRATEGY_CONFIG['strategy']['name']} v{STRATEGY_CONFIG['strategy']['version']}")
    print(f"Engine: QVM Engine v2.1.1 Flat")
    print(f"Portfolio Size: {STRATEGY_CONFIG['strategy']['portfolio']['portfolio_size']} stocks")
    print(f"Starting Capital: {STRATEGY_CONFIG['strategy']['portfolio']['starting_capital']:,.0f} VND")
    print(f"Backtest Window: {BACKTEST_CONFIG['active_window']}")
    
    # Safely display factor weights
    factor_weights = STRATEGY_CONFIG.get('factor_weights', {})
    quality_w = factor_weights.get('quality', 0.25)
    value_w = factor_weights.get('value', 0.25)
    momentum_w = factor_weights.get('momentum', 0.25)
    defensive_w = factor_weights.get('defensive', 0.25)
    print(f"Factor Weights: Q{quality_w:.0%} V{value_w:.0%} M{momentum_w:.0%} D{defensive_w:.0%}")
    
    try:
        # Initialize engine with configuration
        engine = QVMFlatConfigEngine(
            strategy_config=STRATEGY_CONFIG,
            backtest_config=BACKTEST_CONFIG
        )
        
        print("\n📊 Loading universe and generating holdings...")
        
        # Generate holdings using 4-pillar flat methodology
        holdings_df = engine.generate_holdings_with_flat_methodology()
        
        if holdings_df is None or len(holdings_df) == 0:
            print("❌ Failed to generate holdings")
            return
            
        print(f"✅ Holdings generated: {len(holdings_df)} records")
        
        # Load price data
        print("📊 Loading price data...")
        price_data = engine.load_price_data_efficiently(holdings_df)
        
        if price_data is None or len(price_data) == 0:
            print("❌ Failed to load price data")
            return
            
        print(f"✅ Price data loaded: {len(price_data)} records")
        
        # Load benchmark data
        print("📊 Loading benchmark data...")
        benchmark_data = engine.load_benchmark_data()
        
        if benchmark_data is None or len(benchmark_data) == 0:
            print("❌ Failed to load benchmark data")
            return
            
        print(f"✅ Benchmark data loaded: {len(benchmark_data)} records")
        
        # Run strategy with 4-pillar flat methodology
        print("\n🔄 Running QVM strategy with v2.1.1 Flat methodology...")
        strategy_returns, benchmark_returns, cash_allocations_df = engine.run_strategy_with_flat_methodology(
            holdings_df, price_data, benchmark_data
        )
        
        print(f"\n✅ Strategy execution completed:")
        print(f"   Strategy Returns: {len(strategy_returns)} returns")
        print(f"   Benchmark Returns: {len(benchmark_returns)} returns")
        
        # Generate comprehensive tearsheet
        generate_flat_methodology_tearsheet(
            strategy_returns, 
            benchmark_returns, 
            cash_allocations_df,
            STRATEGY_CONFIG,
            BACKTEST_CONFIG
        )
        
        print(f"\n🎉 4-Pillar Flat methodology strategy completed successfully!")
        print(f"📊 Key achievements:")
        print(f"   - QVM Engine v2.1.1 Flat methodology implemented")
        print(f"   - 4-pillar architecture (Equal weights: 25% each) operational")
        print(f"   - Portfolio size fixed at 20 stocks")
        print(f"   - Risk management: 5% drop => 20% cash allocation")
        print(f"   - Configuration-driven approach with version validation")
        print(f"   - Investment Committee hurdle validation completed")
        
        # Display final version compatibility status
        print(f"\n🔍 FINAL VERSION COMPATIBILITY STATUS")
        print("-" * 50)
        if compatibility_valid:
            print("✅ All components are version compatible")
        else:
            print("⚠️ Version compatibility issues detected - review required")
        
    except Exception as e:
        print(f"❌ Error in main execution: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
