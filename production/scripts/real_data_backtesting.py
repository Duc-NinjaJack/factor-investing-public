#!/usr/bin/env python3
"""
================================================================================
Real Data Backtesting Engine - Simplified Backtrader
================================================================================
Purpose:
    Perform comprehensive backtesting using real market data from the database.
    This script provides a clean, reusable backtesting framework that can be
    used for strategy validation and performance analysis.

Features:
    - Real price data from vcsc_daily_data_complete
    - Factor scores from factor_scores_qvm
    - ADTV liquidity filtering
    - No short-selling constraint
    - Transaction cost modeling
    - Performance metrics calculation
    - Visualization and reporting

Author: Quantitative Strategy Team
Date: January 2025
Status: PRODUCTION READY
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, text
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
from datetime import datetime, timedelta
import logging
import pickle
import argparse
import sys

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealDataBacktesting:
    """
    Real data backtesting engine using actual market data from database.
    """
    
    def __init__(self, config_path: str = None, pickle_path: str = None):
        """
        Initialize the backtesting engine.
        
        Args:
            config_path: Path to database configuration file
            pickle_path: Path to ADTV data pickle file
        """
        self.config_path = config_path or self._find_config_path()
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
        
        logger.info("✅ Real Data Backtesting Engine initialized")
    
    def _find_config_path(self) -> str:
        """Find the database configuration file."""
        # Look for config in parent directories
        current_path = Path(__file__).parent
        possible_paths = [
            current_path.parent.parent / "config" / "database.yml",
            current_path.parent.parent / "config" / "config.ini",
            current_path.parent.parent.parent / "config" / "database.yml"
        ]
        
        for path in possible_paths:
            if path.exists():
                return str(path)
        
        raise FileNotFoundError("Database configuration file not found")
    
    def _create_database_engine(self):
        """Create database engine."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            # Use production config
            db_config = config.get('production', config)
            
            connection_string = (
                f"mysql+pymysql://{db_config['username']}:{db_config['password']}"
                f"@{db_config['host']}/{db_config['schema_name']}"
            )
            return create_engine(connection_string, pool_recycle=3600)
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load all required data for backtesting."""
        logger.info("📊 Loading data for real data backtesting...")
        
        data = {}
        
        # Load ADTV data from pickle
        try:
            with open(self.pickle_path, 'rb') as f:
                pickle_data = pickle.load(f)
            
            data['adtv_data'] = pickle_data['adtv']
            logger.info("✅ ADTV data loaded from pickle")
            
        except FileNotFoundError:
            logger.error(f"❌ Pickle file not found: {self.pickle_path}")
            logger.error("Please run get_unrestricted_universe_data.py first.")
            raise
        
        # Load price data from vcsc_daily_data_complete
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
        
        # Load factor scores from database
        factor_query = """
        SELECT date, ticker, QVM_Composite
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
        
        # Load benchmark data (VNINDEX)
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
        
        logger.info(f"✅ All data loaded successfully")
        logger.info(f"   - Price data: {len(data['price_data']):,} records")
        logger.info(f"   - Factor scores: {len(data['factor_scores']):,} records")
        logger.info(f"   - Benchmark: {len(data['benchmark']):,} records")
        logger.info(f"   - ADTV data: {data['adtv_data'].shape}")
        
        return data
    
    def prepare_data_for_backtesting(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Prepare data for backtesting."""
        logger.info("🔧 Preparing data for backtesting...")
        
        # Prepare price data
        price_pivot = data['price_data'].pivot(
            index='trading_date', columns='ticker', values='close_price_adjusted'
        )
        
        # Calculate returns
        returns = price_pivot.pct_change().dropna()
        
        # Prepare factor data
        factor_pivot = data['factor_scores'].pivot(
            index='date', columns='ticker', values='QVM_Composite'
        )
        
        # Prepare benchmark returns
        benchmark_returns = data['benchmark'].set_index('date')['close'].pct_change().dropna()
        
        # Align all data
        common_dates = returns.index.intersection(factor_pivot.index).intersection(benchmark_returns.index)
        returns = returns.loc[common_dates]
        factor_pivot = factor_pivot.loc[common_dates]
        benchmark_returns = benchmark_returns.loc[common_dates]
        
        logger.info(f"✅ Data prepared for backtesting")
        logger.info(f"   - Common dates: {len(common_dates)}")
        logger.info(f"   - Returns shape: {returns.shape}")
        logger.info(f"   - Factor scores shape: {factor_pivot.shape}")
        
        return {
            'returns': returns,
            'factor_scores': factor_pivot,
            'benchmark_returns': benchmark_returns,
            'adtv_data': data['adtv_data']
        }
    
    def run_backtest(self, threshold_name: str, threshold_value: int, 
                    prepared_data: Dict[str, pd.DataFrame]) -> Dict:
        """Run backtest for a specific threshold with NO SHORT SELLING."""
        logger.info(f"🚀 Running backtest for {threshold_name}...")
        
        returns = prepared_data['returns']
        factor_scores = prepared_data['factor_scores']
        adtv_data = prepared_data['adtv_data']
        
        # Rebalancing dates (monthly)
        rebalance_dates = pd.date_range(
            start=returns.index.min(),
            end=returns.index.max(),
            freq=self.backtest_config['rebalance_freq']
        )
        
        # Filter to dates with data
        rebalance_dates = rebalance_dates[rebalance_dates.isin(returns.index)]
        
        portfolio_returns_dict = {}
        portfolio_holdings = []
        portfolio_values = [self.backtest_config['initial_capital']]
        
        for i, rebalance_date in enumerate(rebalance_dates[:-1]):
            next_rebalance = rebalance_dates[i + 1]
            
            # Get factor scores as of rebalance date
            if rebalance_date in factor_scores.index:
                factor_scores_date = factor_scores.loc[rebalance_date].dropna()
            else:
                # Use forward fill
                factor_scores_date = factor_scores.loc[:rebalance_date].iloc[-1].dropna()
            
            # Get ADTV as of rebalance date
            if rebalance_date in adtv_data.index:
                adtv_scores = adtv_data.loc[rebalance_date].dropna()
            else:
                # Use forward fill
                adtv_scores = adtv_data.loc[:rebalance_date].iloc[-1].dropna()
            
            # Apply liquidity filter
            liquid_stocks = adtv_scores[adtv_scores >= threshold_value].index
            available_stocks = factor_scores_date.index.intersection(liquid_stocks)
            
            if len(available_stocks) < self.backtest_config['portfolio_size']:
                # Skip this rebalancing if not enough stocks
                continue
            
            # Select top stocks by QVM score (NO SHORT SELLING - only long positions)
            top_stocks = factor_scores_date[available_stocks].nlargest(
                self.backtest_config['portfolio_size']
            ).index
            
            # Equal weight portfolio (long only)
            weights = pd.Series(1.0 / len(top_stocks), index=top_stocks)
            
            # Calculate portfolio returns for this period
            period_returns = returns.loc[rebalance_date:next_rebalance, top_stocks]
            portfolio_return = (period_returns * weights).sum(axis=1)
            
            # Apply transaction costs (simplified)
            if i > 0:  # Not the first rebalancing
                portfolio_return.iloc[0] -= self.backtest_config['transaction_cost']
            
            # Store returns for this period
            for date, ret in portfolio_return.items():
                portfolio_returns_dict[date] = ret
            
            portfolio_holdings.append({
                'date': rebalance_date,
                'stocks': list(top_stocks),
                'weights': weights.to_dict(),
                'universe_size': len(available_stocks),
                'portfolio_value': portfolio_values[-1]
            })
            
            # Update portfolio value
            period_return_series = pd.Series(portfolio_return.values, index=period_returns.index)
            cumulative_return = (1 + period_return_series).prod()
            portfolio_values.append(portfolio_values[-1] * cumulative_return)
        
        # Create portfolio returns series
        portfolio_returns_series = pd.Series(portfolio_returns_dict)
        
        # Align with benchmark
        aligned_data = pd.DataFrame({
            'portfolio': portfolio_returns_series,
            'benchmark': prepared_data['benchmark_returns']
        }).dropna()
        
        # Calculate performance metrics
        metrics = self._calculate_performance_metrics(aligned_data)
        
        logger.info(f"✅ {threshold_name} backtest complete")
        logger.info(f"   - Annual Return: {metrics['annual_return']:.2%}")
        logger.info(f"   - Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"   - Max Drawdown: {metrics['max_drawdown']:.2%}")
        logger.info(f"   - Alpha: {metrics['alpha']:.2%}")
        
        return {
            'returns': portfolio_returns_series,
            'benchmark_returns': prepared_data['benchmark_returns'],
            'metrics': metrics,
            'holdings': portfolio_holdings,
            'portfolio_values': portfolio_values
        }
    
    def _calculate_performance_metrics(self, aligned_data: pd.DataFrame) -> Dict:
        """Calculate comprehensive performance metrics."""
        # Basic metrics
        annual_return = aligned_data['portfolio'].mean() * 252
        annual_vol = aligned_data['portfolio'].std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
        
        # Drawdown calculation
        cumulative_returns = (1 + aligned_data['portfolio']).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Alpha and beta calculation
        covariance = np.cov(aligned_data['portfolio'], aligned_data['benchmark'])[0, 1]
        benchmark_var = aligned_data['benchmark'].var()
        beta = covariance / benchmark_var if benchmark_var > 0 else 0
        
        benchmark_return = aligned_data['benchmark'].mean() * 252
        alpha = annual_return - (beta * benchmark_return)
        
        # Additional metrics
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        information_ratio = alpha / (aligned_data['portfolio'].std() * np.sqrt(252)) if aligned_data['portfolio'].std() > 0 else 0
        
        # Turnover (simplified)
        turnover = len(aligned_data) * self.backtest_config['transaction_cost'] * 2  # Approximate
        
        return {
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'alpha': alpha,
            'beta': beta,
            'calmar_ratio': calmar_ratio,
            'information_ratio': information_ratio,
            'turnover': turnover,
            'total_return': (1 + aligned_data['portfolio']).prod() - 1,
            'benchmark_return': benchmark_return
        }
    
    def run_comparative_backtests(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """Run backtests for all thresholds."""
        logger.info("🔄 Running comparative backtests...")
        
        # Prepare data
        prepared_data = self.prepare_data_for_backtesting(data)
        
        # Run backtests
        backtest_results = {}
        
        for threshold_name, threshold_value in self.thresholds.items():
            results = self.run_backtest(threshold_name, threshold_value, prepared_data)
            backtest_results[threshold_name] = results
        
        return backtest_results
    
    def create_performance_visualizations(self, backtest_results: Dict[str, Dict], 
                                        save_path: str = None):
        """Create comprehensive performance visualizations."""
        logger.info("📈 Creating performance visualizations...")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Cumulative Returns Comparison
        ax1 = plt.subplot(3, 3, 1)
        for threshold, results in backtest_results.items():
            cumulative_returns = (1 + results['returns']).cumprod()
            ax1.plot(range(len(cumulative_returns)), cumulative_returns.values, 
                    label=threshold, linewidth=2)
        ax1.set_title('Cumulative Returns Comparison', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Cumulative Return')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Drawdown Analysis
        ax2 = plt.subplot(3, 3, 2)
        for threshold, results in backtest_results.items():
            cumulative_returns = (1 + results['returns']).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            ax2.fill_between(range(len(drawdown)), drawdown.values, 0, alpha=0.3, label=threshold)
        ax2.set_title('Drawdown Analysis', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Drawdown')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Rolling Sharpe Ratio
        ax3 = plt.subplot(3, 3, 3)
        for threshold, results in backtest_results.items():
            rolling_sharpe = results['returns'].rolling(window=252).mean() / results['returns'].rolling(window=252).std() * np.sqrt(252)
            ax3.plot(range(len(rolling_sharpe)), rolling_sharpe.values, label=threshold, linewidth=2)
        ax3.set_title('Rolling Sharpe Ratio (1-Year)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Sharpe Ratio')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Performance Metrics Comparison
        ax4 = plt.subplot(3, 3, 4)
        metrics_df = pd.DataFrame({
            threshold: results['metrics'] 
            for threshold, results in backtest_results.items()
        }).T
        
        # Plot key metrics
        metrics_to_plot = ['annual_return', 'sharpe_ratio', 'max_drawdown', 'alpha']
        x = np.arange(len(metrics_to_plot))
        width = 0.35
        
        for i, threshold in enumerate(backtest_results.keys()):
            values = [metrics_df.loc[threshold, metric] for metric in metrics_to_plot]
            ax4.bar(x + i * width, values, width, label=threshold, alpha=0.8)
        
        ax4.set_title('Performance Metrics Comparison', fontsize=12, fontweight='bold')
        ax4.set_xticks(x + width / 2)
        ax4.set_xticklabels([m.replace('_', ' ').title() for m in metrics_to_plot])
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Monthly Returns Heatmap
        ax5 = plt.subplot(3, 3, 5)
        # Use the first threshold for monthly returns
        first_threshold = list(backtest_results.keys())[0]
        monthly_returns = backtest_results[first_threshold]['returns'].resample('M').apply(lambda x: (1 + x).prod() - 1)
        monthly_returns_pivot = monthly_returns.groupby([monthly_returns.index.year, monthly_returns.index.month]).first().unstack()
        
        sns.heatmap(monthly_returns_pivot, annot=True, fmt='.2%', cmap='RdYlGn', center=0, ax=ax5)
        ax5.set_title(f'Monthly Returns Heatmap ({first_threshold})', fontsize=12, fontweight='bold')
        
        # 6. Portfolio Holdings Evolution
        ax6 = plt.subplot(3, 3, 6)
        # Plot portfolio size evolution
        for threshold, results in backtest_results.items():
            holdings_dates = [h['date'] for h in results['holdings']]
            portfolio_sizes = [len(h['stocks']) for h in results['holdings']]
            ax6.plot(holdings_dates, portfolio_sizes, label=threshold, marker='o', markersize=4)
        ax6.set_title('Portfolio Size Evolution', fontsize=12, fontweight='bold')
        ax6.set_ylabel('Number of Stocks')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Risk-Return Scatter
        ax7 = plt.subplot(3, 3, 7)
        for threshold, results in backtest_results.items():
            metrics = results['metrics']
            ax7.scatter(metrics['annual_volatility'], metrics['annual_return'], 
                       label=threshold, s=100, alpha=0.7)
        ax7.set_title('Risk-Return Profile', fontsize=12, fontweight='bold')
        ax7.set_xlabel('Annual Volatility')
        ax7.set_ylabel('Annual Return')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # 8. Benchmark Comparison
        ax8 = plt.subplot(3, 3, 8)
        for threshold, results in backtest_results.items():
            cumulative_portfolio = (1 + results['returns']).cumprod()
            cumulative_benchmark = (1 + results['benchmark_returns']).cumprod()
            ax8.plot(cumulative_portfolio.index, cumulative_portfolio.values, 
                    label=f'{threshold} Portfolio', linewidth=2)
        ax8.plot(cumulative_benchmark.index, cumulative_benchmark.values, 
                label='VNINDEX Benchmark', linewidth=2, linestyle='--', color='black')
        ax8.set_title('Portfolio vs Benchmark', fontsize=12, fontweight='bold')
        ax8.set_ylabel('Cumulative Return')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        # 9. Turnover Analysis
        ax9 = plt.subplot(3, 3, 9)
        turnover_data = []
        for threshold, results in backtest_results.items():
            turnover_data.append({
                'Threshold': threshold,
                'Turnover': results['metrics']['turnover']
            })
        turnover_df = pd.DataFrame(turnover_data)
        turnover_df.plot(x='Threshold', y='Turnover', kind='bar', ax=ax9)
        ax9.set_title('Portfolio Turnover', fontsize=12, fontweight='bold')
        ax9.set_ylabel('Turnover Rate')
        ax9.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✅ Visualizations saved to {save_path}")
        
        plt.show()
    
    def generate_comprehensive_report(self, backtest_results: Dict[str, Dict]) -> str:
        """Generate a comprehensive backtesting report."""
        logger.info("📋 Generating comprehensive report...")
        
        report = []
        report.append("=" * 80)
        report.append("REAL DATA BACKTESTING REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary statistics
        report.append("SUMMARY STATISTICS")
        report.append("-" * 40)
        
        summary_data = []
        for threshold, results in backtest_results.items():
            metrics = results['metrics']
            summary_data.append({
                'Threshold': threshold,
                'Annual Return': f"{metrics['annual_return']:.2%}",
                'Sharpe Ratio': f"{metrics['sharpe_ratio']:.2f}",
                'Max Drawdown': f"{metrics['max_drawdown']:.2%}",
                'Alpha': f"{metrics['alpha']:.2%}",
                'Beta': f"{metrics['beta']:.2f}",
                'Total Return': f"{metrics['total_return']:.2%}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        report.append(summary_df.to_string(index=False))
        report.append("")
        
        # Detailed analysis
        report.append("DETAILED ANALYSIS")
        report.append("-" * 40)
        
        for threshold, results in backtest_results.items():
            metrics = results['metrics']
            report.append(f"\n{threshold} THRESHOLD:")
            report.append(f"  Annual Return: {metrics['annual_return']:.2%}")
            report.append(f"  Annual Volatility: {metrics['annual_volatility']:.2%}")
            report.append(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            report.append(f"  Maximum Drawdown: {metrics['max_drawdown']:.2%}")
            report.append(f"  Alpha: {metrics['alpha']:.2%}")
            report.append(f"  Beta: {metrics['beta']:.2f}")
            report.append(f"  Calmar Ratio: {metrics['calmar_ratio']:.2f}")
            report.append(f"  Information Ratio: {metrics['information_ratio']:.2f}")
            report.append(f"  Total Return: {metrics['total_return']:.2%}")
            report.append(f"  Benchmark Return: {metrics['benchmark_return']:.2%}")
            report.append(f"  Turnover: {metrics['turnover']:.2%}")
        
        # Recommendations
        report.append("\nRECOMMENDATIONS")
        report.append("-" * 40)
        
        # Find best performing threshold
        best_threshold = max(backtest_results.keys(), 
                           key=lambda x: backtest_results[x]['metrics']['sharpe_ratio'])
        best_metrics = backtest_results[best_threshold]['metrics']
        
        report.append(f"Best Performing Threshold: {best_threshold}")
        report.append(f"  Sharpe Ratio: {best_metrics['sharpe_ratio']:.2f}")
        report.append(f"  Annual Return: {best_metrics['annual_return']:.2%}")
        report.append(f"  Max Drawdown: {best_metrics['max_drawdown']:.2%}")
        
        if best_metrics['sharpe_ratio'] > 1.0:
            report.append("✅ Strategy shows strong risk-adjusted performance")
        elif best_metrics['sharpe_ratio'] > 0.5:
            report.append("⚠️ Strategy shows moderate risk-adjusted performance")
        else:
            report.append("❌ Strategy shows poor risk-adjusted performance")
        
        if best_metrics['alpha'] > 0.05:
            report.append("✅ Strategy generates significant alpha")
        elif best_metrics['alpha'] > 0.02:
            report.append("⚠️ Strategy generates moderate alpha")
        else:
            report.append("❌ Strategy generates minimal alpha")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)
    
    def run_complete_analysis(self, save_plots: bool = True, save_report: bool = True):
        """Run complete backtesting analysis."""
        logger.info("🎯 Starting complete real data backtesting analysis...")
        
        try:
            # Load data
            data = self.load_data()
            
            # Run backtests
            backtest_results = self.run_comparative_backtests(data)
            
            # Create visualizations
            if save_plots:
                plot_path = f"real_data_backtesting_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                self.create_performance_visualizations(backtest_results, plot_path)
            else:
                self.create_performance_visualizations(backtest_results)
            
            # Generate report
            report = self.generate_comprehensive_report(backtest_results)
            
            if save_report:
                report_path = f"real_data_backtesting_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                with open(report_path, 'w') as f:
                    f.write(report)
                logger.info(f"✅ Report saved to {report_path}")
            
            print("\n" + "=" * 80)
            print("REAL DATA BACKTESTING COMPLETE")
            print("=" * 80)
            print(report)
            
            return backtest_results
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
            raise

def main():
    """Main function to run the backtesting analysis."""
    parser = argparse.ArgumentParser(description='Real Data Backtesting Engine')
    parser.add_argument('--config', type=str, help='Path to database configuration file')
    parser.add_argument('--pickle', type=str, help='Path to ADTV data pickle file')
    parser.add_argument('--no-plots', action='store_true', help='Skip saving plots')
    parser.add_argument('--no-report', action='store_true', help='Skip saving report')
    
    args = parser.parse_args()
    
    try:
        # Initialize backtesting engine
        backtesting = RealDataBacktesting(
            config_path=args.config,
            pickle_path=args.pickle
        )
        
        # Run complete analysis
        results = backtesting.run_complete_analysis(
            save_plots=not args.no_plots,
            save_report=not args.no_report
        )
        
        logger.info("✅ Real data backtesting completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Backtesting failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()