#!/usr/bin/env python3
"""
================================================================================
Value-Only Backtesting Engine - Pure Value Factor Strategy
================================================================================
Purpose:
    Subclass of RealDataBacktesting that uses only the Value_Composite factor
    for stock selection, providing a pure value strategy backtest.

Features:
    - Uses Value_Composite instead of QVM_Composite for stock selection
    - All other functionality (liquidity, returns, benchmark) remains the same
    - Pure value factor strategy with no quality or momentum components

Author: Quantitative Strategy Team
Date: January 2025
Status: PRODUCTION READY
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
import yaml
import pickle
from sqlalchemy import create_engine
from typing import Dict

# Add the scripts directory to path to import the parent class
scripts_dir = Path(__file__).parent
sys.path.append(str(scripts_dir))

from real_data_backtesting import RealDataBacktesting

logger = logging.getLogger(__name__)

class ValueOnlyBacktesting(RealDataBacktesting):
    """
    Value-only backtesting engine using Value_Composite factor for stock selection.
    """
    
    def __init__(self, config_path: str = None, pickle_path: str = None):
        """
        Initialize the value-only backtesting engine.
        
        Args:
            config_path: Path to database configuration file
            pickle_path: Path to ADTV data pickle file
        """
        # Use the same config path pattern as the working phase20 script
        if config_path is None:
            config_path = "../../../config/database.yml"
        
        self.config_path = config_path
        self.pickle_path = pickle_path or 'unrestricted_universe_data.pkl'
        self.engine = self._create_database_engine()
        
        # Default thresholds
        self.thresholds = {
            '10B_VND': 10_000_000_000,
            '3B_VND': 3_000_000_000
        }
        
        # Default backtest configuration
        self.backtest_config = {
            'start_date': '2018-01-01',
            'end_date': '2025-01-01',
            'rebalance_freq': 'M',  # Monthly rebalancing
            'portfolio_size': 25,
            'max_sector_weight': 0.4,
            'transaction_cost': 0.002,  # 20 bps
            'initial_capital': 100_000_000  # 100M VND
        }
        
        logger.info("✅ Value-Only Backtesting Engine initialized")
        logger.info("   - Using Value_Composite factor for stock selection")
        logger.info("   - Pure value strategy (no quality or momentum components)")
    
    def _create_database_engine(self):
        """Create database engine."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            # Use production config (same as phase20)
            db_config = config['production']
            
            connection_string = (
                f"mysql+pymysql://{db_config['username']}:{db_config['password']}"
                f"@{db_config['host']}/{db_config['schema_name']}"
            )
            return create_engine(connection_string, pool_recycle=3600)
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def load_data(self):
        """
        Load all required data for backtesting, using Value_Composite instead of QVM_Composite.
        """
        logger.info("📊 Loading data for value-only backtesting...")
        
        data = {}
        
        # Load price data (same as phase20)
        price_query = """
        SELECT trading_date, ticker, close_price_adjusted
        FROM vcsc_daily_data_complete
        WHERE trading_date >= '2016-01-01'
        ORDER BY trading_date, ticker
        """
        data['price_data'] = pd.read_sql(price_query, self.engine)
        data['price_data']['trading_date'] = pd.to_datetime(data['price_data']['trading_date'])
        
        # Load factor scores from database - VALUE ONLY
        factor_query = """
        SELECT date, ticker, Value_Composite
        FROM factor_scores_qvm
        WHERE date >= '2016-01-01'
        ORDER BY date, ticker
        """
        data['factor_scores'] = pd.read_sql(factor_query, self.engine)
        data['factor_scores']['date'] = pd.to_datetime(data['factor_scores']['date'])
        
        # Load benchmark data (VNINDEX) - same as phase20
        benchmark_query = """
        SELECT date, close
        FROM etf_history
        WHERE ticker = 'VNINDEX' AND date >= '2016-01-01'
        ORDER BY date
        """
        data['benchmark'] = pd.read_sql(benchmark_query, self.engine)
        data['benchmark']['date'] = pd.to_datetime(data['benchmark']['date'])
        
        # Load ADTV data (same as phase20)
        try:
            with open(self.pickle_path, 'rb') as f:
                pickle_data = pickle.load(f)
            
            data['adtv_data'] = pickle_data['adtv']
            logger.info("✅ ADTV data loaded from pickle")
            
        except FileNotFoundError:
            logger.error(f"❌ Pickle file not found: {self.pickle_path}")
            logger.error("Please run get_unrestricted_universe_data.py first.")
            raise
        
        logger.info(f"✅ All data loaded successfully for value-only backtesting")
        logger.info(f"   - Price data: {len(data['price_data']):,} records")
        logger.info(f"   - Value factor scores: {len(data['factor_scores']):,} records")
        logger.info(f"   - Benchmark: {len(data['benchmark']):,} records")
        logger.info(f"   - ADTV data: {data['adtv_data'].shape}")
        
        return data
    
    def prepare_data_for_backtesting(self, data):
        """
        Prepare data for backtesting, using Value_Composite for factor scores.
        """
        logger.info("🔧 Preparing data for value-only backtesting...")
        
        # Prepare price data (same as parent)
        price_pivot = data['price_data'].pivot(
            index='trading_date', columns='ticker', values='close_price_adjusted'
        )
        
        # Calculate returns (same as parent)
        returns = price_pivot.pct_change().dropna()
        
        # Prepare factor data - VALUE ONLY
        factor_pivot = data['factor_scores'].pivot(
            index='date', columns='ticker', values='Value_Composite'
        )
        
        # Prepare benchmark returns (same as parent)
        benchmark_returns = data['benchmark'].set_index('date')['close'].pct_change().dropna()
        
        # Align all data (same as parent)
        common_dates = returns.index.intersection(factor_pivot.index).intersection(benchmark_returns.index)
        returns = returns.loc[common_dates]
        factor_pivot = factor_pivot.loc[common_dates]
        benchmark_returns = benchmark_returns.loc[common_dates]
        
        logger.info(f"✅ Data prepared for value-only backtesting")
        logger.info(f"   - Common dates: {len(common_dates)}")
        logger.info(f"   - Returns shape: {returns.shape}")
        logger.info(f"   - Value factor scores shape: {factor_pivot.shape}")
        
        return {
            'returns': returns,
            'factor_scores': factor_pivot,
            'benchmark_returns': benchmark_returns,
            'adtv_data': data['adtv_data']
        }
    
    def run_comparative_backtests(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """Run comparative backtests for value-only strategy."""
        logger.info("🔄 Running comparative backtests...")
        
        # Check if data is already prepared (has 'returns' key)
        if 'returns' in data:
            # Data is already prepared, use it directly
            prepared_data = data
        else:
            # Data is raw, prepare it first
            prepared_data = self.prepare_data_for_backtesting(data)
        
        # Run backtests for each threshold
        backtest_results = {}
        
        for threshold_name, threshold_value in self.thresholds.items():
            try:
                result = self.run_backtest(threshold_name, threshold_value, prepared_data)
                backtest_results[threshold_name] = result
                logger.info(f"✅ {threshold_name} backtest completed")
            except Exception as e:
                logger.error(f"❌ {threshold_name} backtest failed: {e}")
                continue
        
        return backtest_results
    
    def run_complete_analysis(self, save_plots: bool = True, save_report: bool = True):
        """
        Run complete value-only analysis with custom naming.
        """
        logger.info("🚀 Running complete value-only factor analysis...")
        
        try:
            # Load data
            data = self.load_data()
            
            # Prepare data
            prepared_data = self.prepare_data_for_backtesting(data)
            
            # Run comparative backtests
            backtest_results = self.run_comparative_backtests(prepared_data)
            
            # Create visualizations with value-specific naming
            if save_plots:
                self.create_performance_visualizations(
                    backtest_results, 
                    save_path='value_only_performance_plots.png'
                )
            
            # Generate report with value-specific content
            if save_report:
                report = self.generate_comprehensive_report(backtest_results)
                with open('value_only_backtest_report.txt', 'w') as f:
                    f.write(report)
                logger.info("📄 Value-only backtest report saved to value_only_backtest_report.txt")
            
            logger.info("✅ Value-only factor analysis completed successfully")
            return {
                'backtest_results': backtest_results,
                'prepared_data': prepared_data
            }
            
        except Exception as e:
            logger.error(f"❌ Value-only analysis failed: {e}")
            raise 