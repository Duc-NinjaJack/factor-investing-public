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
        super().__init__(config_path, pickle_path)
        logger.info("✅ Value-Only Backtesting Engine initialized")
        logger.info("   - Using Value_Composite factor for stock selection")
        logger.info("   - Pure value strategy (no quality or momentum components)")
    
    def load_data(self):
        """
        Load all required data for backtesting, using Value_Composite instead of QVM_Composite.
        """
        logger.info("📊 Loading data for value-only backtesting...")
        
        data = {}
        
        # Load price data (same as parent)
        price_query = """
        SELECT trading_date, ticker, close_price_adjusted
        FROM vcsc_daily_data_complete
        WHERE trading_date >= :start_date
        ORDER BY trading_date, ticker
        """
        data['price_data'] = pd.read_sql(
            price_query, 
            self.engine, 
            params={'start_date': self.backtest_config['start_date']}
        )
        data['price_data']['trading_date'] = pd.to_datetime(data['price_data']['trading_date'])
        
        # Load factor scores from database - VALUE ONLY
        factor_query = """
        SELECT date, ticker, Value_Composite
        FROM factor_scores_qvm
        WHERE date >= :start_date
        ORDER BY date, ticker
        """
        data['factor_scores'] = pd.read_sql(
            factor_query, 
            self.engine, 
            params={'start_date': self.backtest_config['start_date']}
        )
        data['factor_scores']['date'] = pd.to_datetime(data['factor_scores']['date'])
        
        # Load benchmark data (VNINDEX) - same as parent
        benchmark_query = """
        SELECT date, close
        FROM etf_history
        WHERE ticker = 'VNINDEX' AND date >= :start_date
        ORDER BY date
        """
        data['benchmark'] = pd.read_sql(
            benchmark_query, 
            self.engine, 
            params={'start_date': self.backtest_config['start_date']}
        )
        data['benchmark']['date'] = pd.to_datetime(data['benchmark']['date'])
        
        # Load ADTV data (same as parent)
        try:
            with open(self.pickle_path, 'rb') as f:
                data['adtv_data'] = pd.read_pickle(f)
        except FileNotFoundError:
            logger.warning(f"ADTV data file {self.pickle_path} not found. Creating empty DataFrame.")
            data['adtv_data'] = pd.DataFrame()
        
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