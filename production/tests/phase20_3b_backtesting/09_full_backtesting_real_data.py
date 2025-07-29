#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full Backtesting with Real Price Data: 10B vs 3B VND Thresholds
===============================================================
Component: Real Performance Validation
Purpose: Run full backtests with actual price data from database
Author: Duc Nguyen, Principal Quantitative Strategist
Date Created: January 2025
Status: PRODUCTION VALIDATION

This script performs comprehensive backtesting comparison using REAL price data:
- Loads factor data from database with correct column names
- Loads price data from vcsc_daily_data_complete (close_price_adjusted)
- Runs backtests with both 10B and 3B VND thresholds
- Enforces no-short-selling constraint
- Compares performance metrics (returns, Sharpe, drawdown, etc.)
- Generates detailed analysis and recommendations

Data Sources:
- vcsc_daily_data_complete (price data with close_price_adjusted)
- factor_scores_qvm (factor scores with correct column names)
- unrestricted_universe_data.pkl (ADTV data)

Dependencies:
- pandas >= 1.3.0
- numpy >= 1.21.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- sqlalchemy >= 1.4.0
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

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FullBacktestingRealData:
    """
    Full backtesting comparison using real price data from database.
    """
    
    def __init__(self, config_path: str = "../../../config/database.yml"):
        """Initialize the backtesting comparison."""
        self.config_path = config_path
        self.engine = self._create_database_engine()
        
        # Analysis parameters
        self.thresholds = {
            '10B_VND': 10_000_000_000,
            '3B_VND': 3_000_000_000
        }
        
        # Backtest parameters
        self.backtest_config = {
            'start_date': '2018-01-01',
            'end_date': '2025-01-01',
            'rebalance_freq': 'M',  # Monthly rebalancing
            'portfolio_size': 25,
            'max_sector_weight': 0.4,
            'transaction_cost': 0.002,  # 20 bps
            'initial_capital': 100_000_000  # 100M VND
        }
        
        logger.info("Full Backtesting with Real Data initialized")
    
    def _create_database_engine(self):
        """Create database engine."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            # Use production config
            db_config = config['production']
            
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
        logger.info("Loading data for full backtesting with real price data...")
        
        data = {}
        
        # Load ADTV data from pickle
        try:
            with open('unrestricted_universe_data.pkl', 'rb') as f:
                pickle_data = pickle.load(f)
            
            data['adtv_data'] = pickle_data['adtv']
            logger.info("✅ ADTV data loaded from pickle")
            
        except FileNotFoundError:
            logger.error("❌ Pickle file not found. Please run get_unrestricted_universe_data.py first.")
            raise
        
        # Load price data from vcsc_daily_data_complete
        price_query = """
        SELECT trading_date, ticker, close_price_adjusted
        FROM vcsc_daily_data_complete
        WHERE trading_date >= '2018-01-01'
        ORDER BY trading_date, ticker
        """
        data['price_data'] = pd.read_sql(price_query, self.engine)
        data['price_data']['trading_date'] = pd.to_datetime(data['price_data']['trading_date'])
        
        # Load factor scores from database
        factor_query = """
        SELECT date, ticker, QVM_Composite
        FROM factor_scores_qvm
        WHERE date >= '2018-01-01'
        ORDER BY date, ticker
        """
        data['factor_scores'] = pd.read_sql(factor_query, self.engine)
        data['factor_scores']['date'] = pd.to_datetime(data['factor_scores']['date'])
        
        # Load benchmark data (VNINDEX)
        benchmark_query = """
        SELECT date, close
        FROM etf_history
        WHERE ticker = 'VNINDEX' AND date >= '2018-01-01'
        ORDER BY date
        """
        data['benchmark'] = pd.read_sql(benchmark_query, self.engine)
        data['benchmark']['date'] = pd.to_datetime(data['benchmark']['date'])
        
        logger.info(f"✅ All data loaded successfully")
        logger.info(f"   - Price data: {len(data['price_data']):,} records")
        logger.info(f"   - Factor scores: {len(data['factor_scores']):,} records")
        logger.info(f"   - Benchmark: {len(data['benchmark']):,} records")
        logger.info(f"   - ADTV data: {data['adtv_data'].shape}")
        
        return data
    
    def prepare_data_for_backtesting(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Prepare data for backtesting."""
        logger.info("Preparing data for backtesting...")
        
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
        logger.info(f"Running backtest for {threshold_name}...")
        
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
        
        portfolio_returns = []
        portfolio_holdings = []
        portfolio_values = [self.backtest_config['initial_capital']]
        
        # Calculate performance metrics
        # Create a proper portfolio returns series that matches the returns index
        portfolio_returns_dict = {}
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
        
        # Calculate metrics
        annual_return = aligned_data['portfolio'].mean() * 252
        annual_vol = aligned_data['portfolio'].std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
        
        # Calculate drawdown
        cumulative_returns = (1 + aligned_data['portfolio']).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calculate alpha and beta
        covariance = np.cov(aligned_data['portfolio'], aligned_data['benchmark'])[0, 1]
        benchmark_var = aligned_data['benchmark'].var()
        beta = covariance / benchmark_var if benchmark_var > 0 else 0
        
        benchmark_return = aligned_data['benchmark'].mean() * 252
        alpha = annual_return - (beta * benchmark_return)
        
        # Calculate additional metrics
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        information_ratio = alpha / (aligned_data['portfolio'].std() * np.sqrt(252)) if aligned_data['portfolio'].std() > 0 else 0
        
        # Calculate turnover (simplified)
        turnover = len(portfolio_holdings) * self.backtest_config['transaction_cost'] * 2  # Approximate
        
        logger.info(f"✅ {threshold_name} backtest complete")
        logger.info(f"   - Annual Return: {annual_return:.2%}")
        logger.info(f"   - Sharpe Ratio: {sharpe_ratio:.2f}")
        logger.info(f"   - Max Drawdown: {max_drawdown:.2%}")
        logger.info(f"   - Alpha: {alpha:.2%}")
        
        return {
            'returns': aligned_data['portfolio'],
            'benchmark_returns': aligned_data['benchmark'],
            'metrics': {
                'annual_return': annual_return,
                'annual_volatility': annual_vol,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'alpha': alpha,
                'beta': beta,
                'calmar_ratio': calmar_ratio,
                'information_ratio': information_ratio,
                'turnover': turnover
            },
            'holdings': portfolio_holdings,
            'portfolio_values': portfolio_values
        }
    
    def run_comparative_backtests(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """Run backtests for both thresholds."""
        logger.info("Running comparative backtests with real data...")
        
        # Prepare data
        prepared_data = self.prepare_data_for_backtesting(data)
        
        # Run backtests
        backtest_results = {}
        
        for threshold_name, threshold_value in self.thresholds.items():
            results = self.run_backtest(threshold_name, threshold_value, prepared_data)
            backtest_results[threshold_name] = results
        
        return backtest_results
    
    def create_performance_visualizations(self, backtest_results: Dict[str, Dict]):
        """Create comprehensive performance visualizations."""
        logger.info("Creating performance visualizations...")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Cumulative Returns Comparison
        ax1 = plt.subplot(3, 3, 1)
        for threshold, results in backtest_results.items():
            cumulative_returns = (1 + results['returns']).cumprod()
            ax1.plot(cumulative_returns.index, cumulative_returns.values, 
                    label=threshold, linewidth=2)
        ax1.set_title('Cumulative Returns Comparison (Real Data)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Cumulative Return')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Drawdown Analysis
        ax2 = plt.subplot(3, 3, 2)
        for threshold, results in backtest_results.items():
            cumulative_returns = (1 + results['returns']).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            ax2.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, label=threshold)
        ax2.set_title('Drawdown Analysis (Real Data)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Drawdown')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Rolling Sharpe Ratio
        ax3 = plt.subplot(3, 3, 3)
        for threshold, results in backtest_results.items():
            rolling_sharpe = results['returns'].rolling(window=252).mean() / results['returns'].rolling(window=252).std() * np.sqrt(252)
            ax3.plot(rolling_sharpe.index, rolling_sharpe.values, label=threshold, linewidth=2)
        ax3.set_title('Rolling Sharpe Ratio (1-Year)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Sharpe Ratio')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Performance Metrics Comparison
        ax4 = plt.subplot(3, 3, 4)
        metrics_data = []
        for threshold, results in backtest_results.items():
            metrics = results['metrics']
            metrics_data.append({
                'Metric': 'Annual Return',
                'Value': metrics['annual_return'],
                'Threshold': threshold
            })
            metrics_data.append({
                'Metric': 'Sharpe Ratio',
                'Value': metrics['sharpe_ratio'],
                'Threshold': threshold
            })
            metrics_data.append({
                'Metric': 'Max Drawdown',
                'Value': metrics['max_drawdown'],
                'Threshold': threshold
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        metrics_pivot = metrics_df.pivot(index='Metric', columns='Threshold', values='Value')
        metrics_pivot.plot(kind='bar', ax=ax4)
        ax4.set_title('Performance Metrics Comparison (Real Data)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Value')
        ax4.legend()
        plt.xticks(rotation=45)
        
        # 5. Risk-Return Scatter
        ax5 = plt.subplot(3, 3, 5)
        for threshold, results in backtest_results.items():
            metrics = results['metrics']
            ax5.scatter(metrics['annual_volatility'], metrics['annual_return'], 
                       s=100, label=threshold, alpha=0.7)
        ax5.set_title('Risk-Return Profile (Real Data)', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Annual Volatility')
        ax5.set_ylabel('Annual Return')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Alpha vs Beta
        ax6 = plt.subplot(3, 3, 6)
        for threshold, results in backtest_results.items():
            metrics = results['metrics']
            ax6.scatter(metrics['beta'], metrics['alpha'], 
                       s=100, label=threshold, alpha=0.7)
        ax6.set_title('Alpha vs Beta (Real Data)', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Beta')
        ax6.set_ylabel('Alpha')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Returns Distribution
        ax7 = plt.subplot(3, 3, 7)
        for threshold, results in backtest_results.items():
            ax7.hist(results['returns'], bins=50, alpha=0.7, label=threshold)
        ax7.set_title('Returns Distribution (Real Data)', fontsize=12, fontweight='bold')
        ax7.set_xlabel('Daily Return')
        ax7.set_ylabel('Frequency')
        ax7.legend()
        
        # 8. Portfolio Value Evolution
        ax8 = plt.subplot(3, 3, 8)
        for threshold, results in backtest_results.items():
            portfolio_values = results['portfolio_values']
            ax8.plot(range(len(portfolio_values)), portfolio_values, label=threshold, linewidth=2)
        ax8.set_title('Portfolio Value Evolution (Real Data)', fontsize=12, fontweight='bold')
        ax8.set_ylabel('Portfolio Value (VND)')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        # 9. Performance Summary Table
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('tight')
        ax9.axis('off')
        
        # Create summary table
        summary_data = []
        for threshold, results in backtest_results.items():
            metrics = results['metrics']
            summary_data.append([
                threshold,
                f"{metrics['annual_return']:.2%}",
                f"{metrics['sharpe_ratio']:.2f}",
                f"{metrics['max_drawdown']:.2%}",
                f"{metrics['alpha']:.2%}",
                f"{metrics['beta']:.2f}"
            ])
        
        table = ax9.table(cellText=summary_data,
                         colLabels=['Threshold', 'Return', 'Sharpe', 'MaxDD', 'Alpha', 'Beta'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        ax9.set_title('Performance Summary (Real Data)', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('img/full_backtesting_real_data_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("✅ Visualizations saved to img/full_backtesting_real_data_comparison.png")
    
    def generate_comprehensive_report(self, backtest_results: Dict[str, Dict]) -> str:
        """Generate comprehensive backtesting report."""
        logger.info("Generating comprehensive backtesting report...")
        
        report = []
        report.append("# Full Backtesting with Real Data: 10B vs 3B VND Thresholds")
        report.append("")
        report.append("**Date:** " + datetime.now().strftime("%Y-%m-%d"))
        report.append("**Purpose:** Real performance validation of 3B VND liquidity threshold")
        report.append("**Note:** This analysis uses REAL price data from database")
        report.append("")
        
        # Executive Summary
        report.append("## 🎯 Executive Summary")
        report.append("")
        
        v10b = backtest_results['10B_VND']['metrics']
        v3b = backtest_results['3B_VND']['metrics']
        
        # Calculate improvements
        return_improvement = v3b['annual_return'] - v10b['annual_return']
        sharpe_improvement = v3b['sharpe_ratio'] - v10b['sharpe_ratio']
        drawdown_change = v3b['max_drawdown'] - v10b['max_drawdown']
        alpha_improvement = v3b['alpha'] - v10b['alpha']
        
        report.append(f"- **Annual Return:** {v3b['annual_return']:.2%} vs {v10b['annual_return']:.2%} ({return_improvement:+.2%})")
        report.append(f"- **Sharpe Ratio:** {v3b['sharpe_ratio']:.2f} vs {v10b['sharpe_ratio']:.2f} ({sharpe_improvement:+.2f})")
        report.append(f"- **Max Drawdown:** {v3b['max_drawdown']:.2%} vs {v10b['max_drawdown']:.2%} ({drawdown_change:+.2%})")
        report.append(f"- **Alpha:** {v3b['alpha']:.2%} vs {v10b['alpha']:.2%} ({alpha_improvement:+.2%})")
        report.append("")
        
        # Detailed Performance Analysis
        report.append("## 📊 Detailed Performance Analysis")
        report.append("")
        
        report.append("### Performance Metrics Comparison")
        report.append("")
        report.append("| Metric | 10B VND | 3B VND | Change |")
        report.append("|--------|---------|--------|--------|")
        report.append(f"| Annual Return | {v10b['annual_return']:.2%} | {v3b['annual_return']:.2%} | {return_improvement:+.2%} |")
        report.append(f"| Annual Volatility | {v10b['annual_volatility']:.2%} | {v3b['annual_volatility']:.2%} | {v3b['annual_volatility'] - v10b['annual_volatility']:+.2%} |")
        report.append(f"| Sharpe Ratio | {v10b['sharpe_ratio']:.2f} | {v3b['sharpe_ratio']:.2f} | {sharpe_improvement:+.2f} |")
        report.append(f"| Max Drawdown | {v10b['max_drawdown']:.2%} | {v3b['max_drawdown']:.2%} | {drawdown_change:+.2%} |")
        report.append(f"| Alpha | {v10b['alpha']:.2%} | {v3b['alpha']:.2%} | {alpha_improvement:+.2%} |")
        report.append(f"| Beta | {v10b['beta']:.2f} | {v3b['beta']:.2f} | {v3b['beta'] - v10b['beta']:+.2f} |")
        report.append(f"| Calmar Ratio | {v10b['calmar_ratio']:.2f} | {v3b['calmar_ratio']:.2f} | {v3b['calmar_ratio'] - v10b['calmar_ratio']:+.2f} |")
        report.append(f"| Information Ratio | {v10b['information_ratio']:.2f} | {v3b['information_ratio']:.2f} | {v3b['information_ratio'] - v10b['information_ratio']:+.2f} |")
        report.append("")
        
        # Implementation Decision
        report.append("## 🎯 Implementation Decision")
        report.append("")
        
        # Decision criteria
        performance_improved = v3b['annual_return'] >= v10b['annual_return']
        risk_acceptable = v3b['max_drawdown'] <= v10b['max_drawdown'] * 1.1  # Allow 10% worse drawdown
        sharpe_acceptable = v3b['sharpe_ratio'] >= v10b['sharpe_ratio'] * 0.95  # Allow 5% worse Sharpe
        
        if performance_improved and risk_acceptable and sharpe_acceptable:
            report.append("✅ **IMPLEMENTATION APPROVED**")
            report.append("- Performance maintained or improved")
            report.append("- Risk metrics within acceptable range")
            report.append("- Ready for production deployment")
        elif performance_improved and risk_acceptable:
            report.append("✅ **CONDITIONAL APPROVAL**")
            report.append("- Performance improved")
            report.append("- Risk metrics acceptable")
            report.append("- Monitor Sharpe ratio closely")
        else:
            report.append("❌ **IMPLEMENTATION REJECTED**")
            report.append("- Performance or risk metrics below acceptable thresholds")
            report.append("- Further analysis required")
            report.append("- Consider alternative thresholds")
        
        report.append("")
        
        # Recommendations
        report.append("## 📋 Recommendations")
        report.append("")
        
        if performance_improved:
            report.append("1. **Proceed with 3B VND implementation**")
            report.append("2. **Monitor performance closely** for first 3 months")
            report.append("3. **Set up alerts** for performance degradation")
            report.append("4. **Document the change** in production logs")
        else:
            report.append("1. **Maintain current 10B VND threshold**")
            report.append("2. **Investigate alternative thresholds** (5B VND, 7B VND)")
            report.append("3. **Conduct additional analysis** on universe composition")
            report.append("4. **Review factor calculation methodology**")
        
        report.append("")
        
        report_text = "\n".join(report)
        
        # Save report
        with open('full_backtesting_real_data_report.md', 'w') as f:
            f.write(report_text)
        
        logger.info("✅ Comprehensive report saved to full_backtesting_real_data_report.md")
        return report_text
    
    def run_complete_analysis(self):
        """Run the complete backtesting analysis with real data."""
        logger.info("🚀 Starting full backtesting analysis with real data...")
        
        try:
            # Load data
            data = self.load_data()
            
            # Run comparative backtests
            backtest_results = self.run_comparative_backtests(data)
            
            # Create visualizations
            self.create_performance_visualizations(backtest_results)
            
            # Generate report
            report = self.generate_comprehensive_report(backtest_results)
            
            # Save results
            results = {
                'backtest_results': backtest_results,
                'report': report
            }
            
            # Save to pickle for further analysis
            with open('full_backtesting_real_data_results.pkl', 'wb') as f:
                pickle.dump(results, f)
            
            logger.info("✅ Complete backtesting analysis with real data finished successfully!")
            logger.info("📊 Results saved to:")
            logger.info("   - img/full_backtesting_real_data_comparison.png")
            logger.info("   - full_backtesting_real_data_report.md")
            logger.info("   - full_backtesting_real_data_results.pkl")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
            raise


def main():
    """Main execution function."""
    print("🔬 Full Backtesting with Real Data: 10B vs 3B VND Thresholds")
    print("=" * 65)
    
    # Initialize analyzer
    analyzer = FullBacktestingRealData()
    
    # Run complete analysis
    results = analyzer.run_complete_analysis()
    
    print("\n✅ Full backtesting analysis with real data completed successfully!")
    print("📊 Check the generated files for detailed results.")
    
    # Print key results
    backtest_results = results['backtest_results']
    v10b = backtest_results['10B_VND']['metrics']
    v3b = backtest_results['3B_VND']['metrics']
    
    print(f"\n📈 Key Results (Real Data):")
    print(f"   10B VND: {v10b['annual_return']:.2%} return, {v10b['sharpe_ratio']:.2f} Sharpe, {v10b['max_drawdown']:.2%} drawdown")
    print(f"   3B VND:  {v3b['annual_return']:.2%} return, {v3b['sharpe_ratio']:.2f} Sharpe, {v3b['max_drawdown']:.2%} drawdown")
    print(f"   Change:  {v3b['annual_return'] - v10b['annual_return']:+.2%} return, {v3b['sharpe_ratio'] - v10b['sharpe_ratio']:+.2f} Sharpe, {v3b['max_drawdown'] - v10b['max_drawdown']:+.2%} drawdown")


if __name__ == "__main__":
    main()