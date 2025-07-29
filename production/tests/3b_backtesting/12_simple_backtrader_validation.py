#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Backtrader Validation: 10B vs 3B VND Thresholds
======================================================
Component: Simplified Backtrader Validation
Purpose: Validate results using simplified backtrader framework
Author: Duc Nguyen, Principal Quantitative Strategist
Date Created: January 2025
Status: SIMPLIFIED VALIDATION

This script uses a simplified backtrader approach to validate the backtesting results:
- Uses real price data from database
- Implements basic strategy with liquidity filtering
- Compares 10B vs 3B VND thresholds
- Provides performance metrics and visualizations

Dependencies:
- backtrader >= 1.9.76.123
- pandas >= 1.3.0
- numpy >= 1.21.0
- matplotlib >= 3.5.0
- sqlalchemy >= 1.4.0
"""

import backtrader as bt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import yaml
from datetime import datetime, timedelta
import warnings
import logging
import pickle
from typing import Dict, List, Tuple, Optional

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleLiquidityStrategy(bt.Strategy):
    """
    Simple strategy with liquidity filtering.
    """
    
    params = (
        ('liquidity_threshold', 10_000_000_000),  # Default 10B VND
        ('portfolio_size', 25),
        ('rebalance_freq', 30),  # Rebalance every 30 days
    )
    
    def __init__(self):
        """Initialize the strategy."""
        self.order_list = []
        self.portfolio_weights = {}
        self.portfolio_values = []
        
        # Store external data
        self.factor_scores = {}
        self.adtv_data = {}
        
        logger.info(f"Simple strategy initialized with {self.params.liquidity_threshold:,} VND threshold")
    
    def log(self, txt, dt=None):
        """Logging function."""
        dt = dt or self.datas[0].datetime.date(0)
        logger.info(f'{dt.isoformat()} {txt}')
    
    def notify_order(self, order):
        """Handle order notifications."""
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
            else:
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
        
        self.order_list = []
    
    def notify_trade(self, trade):
        """Handle trade notifications."""
        if not trade.isclosed:
            return
        
        self.log(f'OPERATION PROFIT, GROSS: {trade.pnl:.2f}, NET: {trade.pnlcomm:.2f}')
    
    def next(self):
        """Main strategy logic."""
        # Check if it's time to rebalance
        if len(self) % self.params.rebalance_freq != 0:
            return
        
        # Get current date
        current_date = self.datas[0].datetime.date(0)
        
        # Get factor scores and ADTV for current date
        if current_date not in self.factor_scores or current_date not in self.adtv_data:
            return
        
        factor_scores = self.factor_scores[current_date]
        adtv_scores = self.adtv_data[current_date]
        
        # Apply liquidity filter
        liquid_stocks = {ticker: adtv for ticker, adtv in adtv_scores.items() 
                        if adtv >= self.params.liquidity_threshold}
        
        # Get available stocks with factor scores
        available_stocks = {ticker: score for ticker, score in factor_scores.items() 
                           if ticker in liquid_stocks}
        
        if len(available_stocks) < self.params.portfolio_size:
            self.log(f'Not enough stocks: {len(available_stocks)} < {self.params.portfolio_size}')
            return
        
        # Select top stocks by QVM score
        sorted_stocks = sorted(available_stocks.items(), key=lambda x: x[1], reverse=True)
        top_stocks = [ticker for ticker, score in sorted_stocks[:self.params.portfolio_size]]
        
        # Calculate equal weights
        weight = 1.0 / len(top_stocks)
        
        # Close all existing positions
        for data in self.datas:
            if self.getposition(data).size > 0:
                self.close(data)
        
        # Open new positions
        for ticker in top_stocks:
            # Find the data feed for this ticker
            for data in self.datas:
                if data._name == ticker:
                    # Calculate position size
                    value = self.broker.getvalue() * weight
                    size = int(value / data.close[0])
                    
                    if size > 0:
                        self.buy(data=data, size=size)
                        self.log(f'BUY {ticker}, Size: {size}, Price: {data.close[0]:.2f}')
                    break
        
        # Store portfolio weights for tracking
        self.portfolio_weights[current_date] = {ticker: weight for ticker in top_stocks}
        
        # Track portfolio value
        self.portfolio_values.append(self.broker.getvalue())
        
        self.log(f'Rebalanced portfolio with {len(top_stocks)} stocks')


class SimpleBacktraderValidator:
    """
    Simple backtrader validation framework.
    """
    
    def __init__(self, config_path: str = "../../../config/database.yml"):
        """Initialize the validator."""
        self.config_path = config_path
        self.engine = self._create_database_engine()
        
        # Analysis parameters
        self.thresholds = {
            '10B_VND': 10_000_000_000,
            '3B_VND': 3_000_000_000
        }
        
        logger.info("Simple Backtrader Validator initialized")
    
    def _create_database_engine(self):
        """Create database engine."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            db_config = config['production']
            connection_string = (
                f"mysql+pymysql://{db_config['username']}:{db_config['password']}"
                f"@{db_config['host']}/{db_config['schema_name']}"
            )
            return create_engine(connection_string, pool_recycle=3600)
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def load_data_for_backtrader(self) -> Dict:
        """Load and prepare data for backtrader."""
        logger.info("Loading data for simple backtrader validation...")
        
        # Load ADTV data from pickle
        with open('unrestricted_universe_data.pkl', 'rb') as f:
            pickle_data = pickle.load(f)
        
        adtv_data = pickle_data['adtv']
        
        # Load price data for top stocks (limit to 20 for simplicity)
        top_stocks = adtv_data.columns.tolist()[:20]
        
        price_query = f"""
        SELECT trading_date, ticker, close_price_adjusted
        FROM vcsc_daily_data_complete
        WHERE ticker IN ({','.join([f"'{ticker}'" for ticker in top_stocks])})
        AND trading_date >= '2018-01-01'
        ORDER BY trading_date, ticker
        """
        
        price_data = pd.read_sql(price_query, self.engine)
        price_data['trading_date'] = pd.to_datetime(price_data['trading_date'])
        
        # Load factor scores
        factor_query = f"""
        SELECT date, ticker, QVM_Composite
        FROM factor_scores_qvm
        WHERE ticker IN ({','.join([f"'{ticker}'" for ticker in top_stocks])})
        AND date >= '2018-01-01'
        ORDER BY date, ticker
        """
        
        factor_data = pd.read_sql(factor_query, self.engine)
        factor_data['date'] = pd.to_datetime(factor_data['date'])
        
        # Prepare data for backtrader
        cerebro_data = {}
        
        for ticker in top_stocks:
            # Get price data for this ticker
            ticker_prices = price_data[price_data['ticker'] == ticker].copy()
            if len(ticker_prices) == 0:
                continue
            
            ticker_prices = ticker_prices.set_index('trading_date')['close_price_adjusted']
            ticker_prices = ticker_prices.sort_index()
            
            # Create backtrader data feed
            data = bt.feeds.PandasData(
                dataname=ticker_prices.to_frame(),
                datetime=None,
                open=None,
                high=None,
                low=None,
                close=0,
                volume=None,
                openinterest=None,
                name=ticker
            )
            
            cerebro_data[ticker] = data
        
        # Prepare factor scores and ADTV data by date
        factor_scores_by_date = {}
        adtv_by_date = {}
        
        for date in factor_data['date'].unique():
            date_factors = factor_data[factor_data['date'] == date]
            factor_scores_by_date[date] = dict(zip(date_factors['ticker'], date_factors['QVM_Composite']))
        
        for date in adtv_data.index:
            adtv_by_date[date] = adtv_data.loc[date].to_dict()
        
        logger.info(f"✅ Data loaded for simple backtrader")
        logger.info(f"   - Stocks: {len(cerebro_data)}")
        logger.info(f"   - Factor dates: {len(factor_scores_by_date)}")
        logger.info(f"   - ADTV dates: {len(adtv_by_date)}")
        
        return {
            'cerebro_data': cerebro_data,
            'factor_scores': factor_scores_by_date,
            'adtv_data': adtv_by_date
        }
    
    def run_simple_backtest(self, threshold_name: str, threshold_value: int, 
                           data_dict: Dict) -> Dict:
        """Run simple backtrader backtest."""
        logger.info(f"Running simple backtest for {threshold_name}...")
        
        # Create cerebro instance
        cerebro = bt.Cerebro()
        
        # Add data feeds
        for ticker, data in data_dict['cerebro_data'].items():
            cerebro.adddata(data, name=ticker)
        
        # Add strategy
        cerebro.addstrategy(SimpleLiquidityStrategy, 
                           liquidity_threshold=threshold_value,
                           portfolio_size=15,  # Reduced for simplicity
                           rebalance_freq=30)
        
        # Set initial cash
        cerebro.broker.setcash(100_000_000)  # 100M VND
        
        # Set commission
        cerebro.broker.setcommission(commission=0.002)  # 20 bps
        
        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        
        # Store external data in strategy
        strategy = cerebro.strats[0][0]
        strategy.factor_scores = data_dict['factor_scores']
        strategy.adtv_data = data_dict['adtv_data']
        
        # Run backtest
        results = cerebro.run()
        
        # Extract results
        strat = results[0]
        
        # Get analyzer results
        sharpe_ratio = strat.analyzers.sharpe.get_analysis()['sharperatio']
        drawdown = strat.analyzers.drawdown.get_analysis()
        returns = strat.analyzers.returns.get_analysis()
        
        # Calculate metrics
        total_return = returns.get('rtot', 0)
        annual_return = returns.get('rnorm100', 0) / 100
        max_drawdown = drawdown.get('max', {}).get('drawdown', 0) / 100
        
        # Get portfolio values
        portfolio_values = strat.portfolio_values if hasattr(strat, 'portfolio_values') else []
        
        logger.info(f"✅ {threshold_name} simple backtest complete")
        logger.info(f"   - Total Return: {total_return:.2%}")
        logger.info(f"   - Annual Return: {annual_return:.2%}")
        logger.info(f"   - Sharpe Ratio: {sharpe_ratio:.2f}")
        logger.info(f"   - Max Drawdown: {max_drawdown:.2%}")
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'portfolio_values': portfolio_values
        }
    
    def run_comparative_tests(self, data_dict: Dict) -> Dict:
        """Run comparative backtests."""
        logger.info("Running comparative simple backtests...")
        
        backtest_results = {}
        
        for threshold_name, threshold_value in self.thresholds.items():
            results = self.run_simple_backtest(threshold_name, threshold_value, data_dict)
            backtest_results[threshold_name] = results
        
        return backtest_results
    
    def create_visualizations(self, backtest_results: Dict):
        """Create visualizations."""
        logger.info("Creating simple backtrader visualizations...")
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Portfolio Value Comparison
        ax1 = axes[0, 0]
        for threshold, results in backtest_results.items():
            portfolio_values = results['portfolio_values']
            if portfolio_values:
                ax1.plot(portfolio_values, label=threshold, linewidth=2)
        ax1.set_title('Portfolio Value Evolution (Simple Backtrader)', fontweight='bold')
        ax1.set_ylabel('Portfolio Value (VND)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Performance Metrics Comparison
        ax2 = axes[0, 1]
        metrics_data = []
        for threshold, results in backtest_results.items():
            metrics_data.append({
                'Metric': 'Annual Return',
                'Value': results['annual_return'],
                'Threshold': threshold
            })
            metrics_data.append({
                'Metric': 'Sharpe Ratio',
                'Value': results['sharpe_ratio'],
                'Threshold': threshold
            })
            metrics_data.append({
                'Metric': 'Max Drawdown',
                'Value': results['max_drawdown'],
                'Threshold': threshold
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        metrics_pivot = metrics_df.pivot(index='Metric', columns='Threshold', values='Value')
        metrics_pivot.plot(kind='bar', ax=ax2)
        ax2.set_title('Performance Metrics (Simple Backtrader)', fontweight='bold')
        ax2.set_ylabel('Value')
        ax2.legend()
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # 3. Risk-Return Scatter
        ax3 = axes[1, 0]
        for threshold, results in backtest_results.items():
            ax3.scatter(abs(results['max_drawdown']), results['annual_return'], 
                       s=100, label=threshold, alpha=0.7)
        ax3.set_title('Risk-Return Profile (Simple Backtrader)', fontweight='bold')
        ax3.set_xlabel('Max Drawdown (Absolute)')
        ax3.set_ylabel('Annual Return')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Performance Summary Table
        ax4 = axes[1, 1]
        ax4.axis('tight')
        ax4.axis('off')
        
        summary_data = []
        for threshold, results in backtest_results.items():
            summary_data.append([
                threshold,
                f"{results['annual_return']:.2%}",
                f"{results['sharpe_ratio']:.2f}",
                f"{results['max_drawdown']:.2%}",
                f"{results['total_return']:.2%}"
            ])
        
        table = ax4.table(cellText=summary_data,
                         colLabels=['Threshold', 'Annual Return', 'Sharpe', 'MaxDD', 'Total Return'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax4.set_title('Simple Backtrader Performance Summary', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('simple_backtrader_validation_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("✅ Simple backtrader visualizations saved to simple_backtrader_validation_comparison.png")
    
    def generate_report(self, backtest_results: Dict) -> str:
        """Generate validation report."""
        logger.info("Generating simple backtrader validation report...")
        
        report = []
        report.append("# Simple Backtrader Validation: 10B vs 3B VND Thresholds")
        report.append("")
        report.append("**Date:** " + datetime.now().strftime("%Y-%m-%d"))
        report.append("**Purpose:** Simplified backtrader validation")
        report.append("**Framework:** Backtrader with simplified liquidity filtering strategy")
        report.append("")
        
        # Executive Summary
        report.append("## 🎯 Executive Summary")
        report.append("")
        
        v10b = backtest_results['10B_VND']
        v3b = backtest_results['3B_VND']
        
        return_improvement = v3b['annual_return'] - v10b['annual_return']
        sharpe_improvement = v3b['sharpe_ratio'] - v10b['sharpe_ratio']
        drawdown_change = v3b['max_drawdown'] - v10b['max_drawdown']
        
        report.append(f"- **Annual Return:** {v3b['annual_return']:.2%} vs {v10b['annual_return']:.2%} ({return_improvement:+.2%})")
        report.append(f"- **Sharpe Ratio:** {v3b['sharpe_ratio']:.2f} vs {v10b['sharpe_ratio']:.2f} ({sharpe_improvement:+.2f})")
        report.append(f"- **Max Drawdown:** {v3b['max_drawdown']:.2%} vs {v10b['max_drawdown']:.2%} ({drawdown_change:+.2%})")
        report.append(f"- **Total Return:** {v3b['total_return']:.2%} vs {v10b['total_return']:.2%}")
        report.append("")
        
        # Detailed Analysis
        report.append("## 📊 Detailed Performance Analysis")
        report.append("")
        
        report.append("### Performance Metrics Comparison")
        report.append("")
        report.append("| Metric | 10B VND | 3B VND | Change |")
        report.append("|--------|---------|--------|--------|")
        report.append(f"| Annual Return | {v10b['annual_return']:.2%} | {v3b['annual_return']:.2%} | {return_improvement:+.2%} |")
        report.append(f"| Sharpe Ratio | {v10b['sharpe_ratio']:.2f} | {v3b['sharpe_ratio']:.2f} | {sharpe_improvement:+.2f} |")
        report.append(f"| Max Drawdown | {v10b['max_drawdown']:.2%} | {v3b['max_drawdown']:.2%} | {drawdown_change:+.2%} |")
        report.append(f"| Total Return | {v10b['total_return']:.2%} | {v3b['total_return']:.2%} | {v3b['total_return'] - v10b['total_return']:+.2%} |")
        report.append("")
        
        # Implementation Decision
        report.append("## 🎯 Implementation Decision")
        report.append("")
        
        performance_improved = v3b['annual_return'] >= v10b['annual_return']
        risk_acceptable = v3b['max_drawdown'] <= v10b['max_drawdown'] * 1.1
        sharpe_acceptable = v3b['sharpe_ratio'] >= v10b['sharpe_ratio'] * 0.95
        
        if performance_improved and risk_acceptable and sharpe_acceptable:
            report.append("✅ **IMPLEMENTATION APPROVED**")
            report.append("- Performance maintained or improved")
            report.append("- Risk metrics within acceptable range")
            report.append("- Simple backtrader validation confirms results")
        elif performance_improved and risk_acceptable:
            report.append("✅ **CONDITIONAL APPROVAL**")
            report.append("- Performance improved")
            report.append("- Risk metrics acceptable")
            report.append("- Monitor Sharpe ratio closely")
        else:
            report.append("❌ **IMPLEMENTATION REJECTED**")
            report.append("- Performance or risk metrics below acceptable thresholds")
            report.append("- Simple backtrader validation confirms rejection")
            report.append("- Consider alternative thresholds")
        
        report.append("")
        
        # Recommendations
        report.append("## 📋 Recommendations")
        report.append("")
        
        if performance_improved:
            report.append("1. **Proceed with 3B VND implementation**")
            report.append("2. **Simple backtrader validation confirms positive results**")
            report.append("3. **Monitor performance closely** for first 3 months")
            report.append("4. **Set up alerts** for performance degradation")
        else:
            report.append("1. **Maintain current 10B VND threshold**")
            report.append("2. **Simple backtrader validation confirms negative results**")
            report.append("3. **Investigate alternative thresholds** (5B VND, 7B VND)")
            report.append("4. **Conduct additional analysis** on universe composition")
        
        report.append("")
        
        report_text = "\n".join(report)
        
        # Save report
        with open('simple_backtrader_validation_report.md', 'w') as f:
            f.write(report_text)
        
        logger.info("✅ Simple backtrader validation report saved to simple_backtrader_validation_report.md")
        return report_text
    
    def run_complete_validation(self):
        """Run the complete simple backtrader validation."""
        logger.info("🚀 Starting simple backtrader validation...")
        
        try:
            # Load data
            data_dict = self.load_data_for_backtrader()
            
            # Run comparative backtests
            backtest_results = self.run_comparative_tests(data_dict)
            
            # Create visualizations
            self.create_visualizations(backtest_results)
            
            # Generate report
            report = self.generate_report(backtest_results)
            
            # Save results
            results = {
                'backtest_results': backtest_results,
                'report': report
            }
            
            # Save to pickle
            with open('simple_backtrader_validation_results.pkl', 'wb') as f:
                pickle.dump(results, f)
            
            logger.info("✅ Complete simple backtrader validation finished successfully!")
            logger.info("📊 Results saved to:")
            logger.info("   - simple_backtrader_validation_comparison.png")
            logger.info("   - simple_backtrader_validation_report.md")
            logger.info("   - simple_backtrader_validation_results.pkl")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Simple backtrader validation failed: {e}")
            raise


def main():
    """Main execution function."""
    print("🔬 Simple Backtrader Validation: 10B vs 3B VND Thresholds")
    print("=" * 60)
    
    # Initialize validator
    validator = SimpleBacktraderValidator()
    
    # Run complete validation
    results = validator.run_complete_validation()
    
    print("\n✅ Simple backtrader validation completed successfully!")
    print("📊 Check the generated files for detailed results.")
    
    # Print key results
    backtest_results = results['backtest_results']
    v10b = backtest_results['10B_VND']
    v3b = backtest_results['3B_VND']
    
    print(f"\n📈 Key Results (Simple Backtrader):")
    print(f"   10B VND: {v10b['annual_return']:.2%} return, {v10b['sharpe_ratio']:.2f} Sharpe, {v10b['max_drawdown']:.2%} drawdown")
    print(f"   3B VND:  {v3b['annual_return']:.2%} return, {v3b['sharpe_ratio']:.2f} Sharpe, {v3b['max_drawdown']:.2%} drawdown")
    print(f"   Change:  {v3b['annual_return'] - v10b['annual_return']:+.2%} return, {v3b['sharpe_ratio'] - v10b['sharpe_ratio']:+.2f} Sharpe, {v3b['max_drawdown'] - v10b['max_drawdown']:+.2%} drawdown")


if __name__ == "__main__":
    main()