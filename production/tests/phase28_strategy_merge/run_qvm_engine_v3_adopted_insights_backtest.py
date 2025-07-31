"""
Vietnam Factor Investing Platform - QVM Engine v3 Adopted Insights Backtesting
=============================================================================
Component: Backtesting script for QVM Engine v3 with Adopted Insights Strategy
Purpose: Run backtest from 2020 to present with VNINDEX benchmark
Author: Factor Investing Team, Quantitative Research
Date Created: January 2025
Status: PRODUCTION READY

BACKTESTING SPECIFICATIONS:
- Period: 2020-01-01 to present
- Benchmark: VNINDEX
- Strategy: QVM Engine v3 with Adopted Insights Strategy
- Liquidity Filter: >10bn daily ADTV
- Rebalancing: Monthly
- Output: Vanilla tearsheet (backtrader format)

STRATEGY FEATURES:
1. Regime Detection: Simple volatility/return based (93.6% accuracy)
2. Sector-Aware Factors: Quality-adjusted P/E across sectors
3. Quality Awareness: ROAA positive only (dropped ROAE)
4. Value Contrarian: P/E only (dropped P/B)
5. Momentum Score: Multi-horizon with skip month
6. Risk Management: Position and sector limits

EXPECTED PERFORMANCE:
- Annual Return: 10-15%
- Volatility: 15-20%
- Sharpe Ratio: 0.5-0.7
- Max Drawdown: 15-25%

Dependencies:
- pandas >= 1.3.0
- numpy >= 1.21.0
- sqlalchemy >= 1.4.0
- backtrader >= 1.9.0
- PyYAML >= 5.4.0
"""

# Standard library imports
import pandas as pd
import numpy as np
import backtrader as bt
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
import warnings
import logging

# Import the strategy
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from qvm_engine_v3_adopted_insights import QVMEngineV3AdoptedInsights, QVMEngineV3AdoptedInsightsBacktraderStrategy

# Suppress warnings
warnings.filterwarnings('ignore')


class VNIndexData(bt.feeds.PandasData):
    """
    Custom data feed for VNINDEX benchmark.
    
    Attributes:
        params (tuple): Data feed parameters
    """
    params = (
        ('datetime', None),
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'volume'),
        ('openinterest', None),
    )


class QVMEngineV3AdoptedInsightsBacktest:
    """
    Backtesting framework for VN Long-Only Strategy.
    
    Attributes:
        start_date (pd.Timestamp): Backtest start date
        end_date (pd.Timestamp): Backtest end date
        strategy_engine (QVMEngineV3AdoptedInsights): Strategy engine instance
        logger (logging.Logger): Logger instance
        cerebro (bt.Cerebro): Backtrader cerebro instance
    """
    
    def __init__(self, start_date: str = '2020-01-01', end_date: str = None):
        """
        Initialize backtesting framework.
        
        Args:
            start_date (str): Start date for backtest (default: 2020-01-01)
            end_date (str): End date for backtest (default: today)
        """
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date) if end_date else pd.Timestamp.now()
        
        # Initialize strategy engine
        self.strategy_engine = QVMEngineV3AdoptedInsights()
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Initialize backtrader
        self.cerebro = bt.Cerebro()
        self.cerebro.broker.setcash(1000000000)  # 1B VND starting capital
        self.cerebro.broker.setcommission(commission=0.001)  # 0.1% commission
        
        self.logger.info(f"Backtesting initialized: {self.start_date} to {self.end_date}")
    
    def _setup_logging(self) -> logging.Logger:
        """
        Setup logging configuration.
        
        Returns:
            logging.Logger: Configured logger instance
        """
        logger = logging.getLogger('QVMEngineV3AdoptedInsightsBacktest')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def get_strategy_portfolios(self) -> pd.DataFrame:
        """
        Get strategy portfolios for the entire backtest period.
        
        Returns:
            pd.DataFrame: DataFrame with portfolio holdings over time
        """
        portfolios = []
        current_date = self.start_date
        
        while current_date <= self.end_date:
            try:
                # Run strategy for current date
                results = self.strategy_engine.run_strategy(current_date)
                
                if 'error' not in results:
                    portfolio = results['portfolio'].copy()
                    portfolio['date'] = current_date
                    portfolio['regime'] = results['regime']
                    portfolio['regime_allocation'] = results['regime_allocation']
                    portfolios.append(portfolio)
                
                # Move to next month
                current_date = current_date + pd.DateOffset(months=1)
                
            except Exception as e:
                self.logger.error(f"Error running strategy for {current_date}: {e}")
                current_date = current_date + pd.DateOffset(months=1)
        
        if portfolios:
            return pd.concat(portfolios, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def get_stock_data(self, portfolios_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Get historical price data for stocks in portfolios.
        
        Args:
            portfolios_df (pd.DataFrame): DataFrame with portfolio holdings
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of stock data feeds
        """
        # Get unique stocks from portfolios
        unique_stocks = portfolios_df['ticker'].unique().tolist()
        
        # Add VNINDEX for benchmark
        if 'VNINDEX' not in unique_stocks:
            unique_stocks.append('VNINDEX')
        
        stock_data = {}
        
        for ticker in unique_stocks:
            try:
                # Get historical data
                query = """
                SELECT 
                    date,
                    open,
                    high,
                    low,
                    close,
                    volume
                FROM vcsc_daily_data_complete
                WHERE ticker = :ticker
                  AND date >= :start_date
                  AND date <= :end_date
                ORDER BY date
                """
                
                params = {
                    'ticker': ticker,
                    'start_date': self.start_date,
                    'end_date': self.end_date
                }
                
                df = pd.read_sql(query, self.strategy_engine.engine, params=params)
                
                if not df.empty:
                    # Convert date to datetime
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                    
                    # Create data feed
                    data_feed = VNIndexData(dataname=df, name=ticker)
                    stock_data[ticker] = data_feed
                    
            except Exception as e:
                self.logger.error(f"Error getting data for {ticker}: {e}")
        
        return stock_data
    
    def run_backtest(self) -> Dict:
        """
        Run the complete backtest.
        
        Returns:
            Dict: Backtest results dictionary
        """
        self.logger.info("Starting backtest...")
        
        try:
            # Step 1: Get strategy portfolios
            self.logger.info("Getting strategy portfolios...")
            portfolios_df = self.get_strategy_portfolios()
            
            if portfolios_df.empty:
                return {'error': 'No portfolio data available'}
            
            # Step 2: Get stock data
            self.logger.info("Getting stock data...")
            stock_data = self.get_stock_data(portfolios_df)
            
            if not stock_data:
                return {'error': 'No stock data available'}
            
            # Step 3: Setup backtrader
            self.logger.info("Setting up backtrader...")
            self._setup_backtrader(portfolios_df, stock_data)
            
            # Step 4: Run backtest
            self.logger.info("Running backtest...")
            initial_value = self.cerebro.broker.getvalue()
            results = self.cerebro.run()
            final_value = self.cerebro.broker.getvalue()
            
            # Step 5: Calculate performance metrics
            self.logger.info("Calculating performance metrics...")
            performance = self._calculate_performance(initial_value, final_value, results)
            
            # Step 6: Generate tearsheet
            self.logger.info("Generating tearsheet...")
            tearsheet = self._generate_tearsheet(performance, portfolios_df)
            
            return {
                'performance': performance,
                'tearsheet': tearsheet,
                'portfolios': portfolios_df,
                'stock_data': list(stock_data.keys())
            }
            
        except Exception as e:
            self.logger.error(f"Backtest failed: {e}")
            return {'error': str(e)}
    
    def _setup_backtrader(self, portfolios_df: pd.DataFrame, stock_data: Dict[str, pd.DataFrame]):
        """
        Setup backtrader with data feeds and strategy.
        
        Args:
            portfolios_df (pd.DataFrame): Portfolio data
            stock_data (Dict[str, pd.DataFrame]): Stock data feeds
        """
        # Add data feeds
        for ticker, data_feed in stock_data.items():
            self.cerebro.adddata(data_feed)
        
        # Create strategy instance
        strategy = VNLongOnlyBacktraderStrategy()
        strategy.strategy_engine = self.strategy_engine
        
        # Add strategy
        self.cerebro.addstrategy(QVMEngineV3AdoptedInsightsBacktraderStrategy)
        
        # Add analyzers
        self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        self.cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    
    def _calculate_performance(self, initial_value: float, final_value: float, results) -> Dict:
        """
        Calculate performance metrics.
        
        Args:
            initial_value (float): Initial portfolio value
            final_value (float): Final portfolio value
            results: Backtrader results
            
        Returns:
            Dict: Performance metrics dictionary
        """
        if not results:
            return {}
        
        strategy_result = results[0]
        
        # Basic metrics
        total_return = (final_value / initial_value) - 1
        annual_return = (1 + total_return) ** (252 / len(strategy_result)) - 1
        
        # Get analyzer results
        sharpe_ratio = strategy_result.analyzers.sharpe.get_analysis().get('sharperatio', 0)
        max_drawdown = strategy_result.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0)
        returns_analysis = strategy_result.analyzers.returns.get_analysis()
        trades_analysis = strategy_result.analyzers.trades.get_analysis()
        
        return {
            'initial_value': initial_value,
            'final_value': final_value,
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'returns_analysis': returns_analysis,
            'trades_analysis': trades_analysis
        }
    
    def _generate_tearsheet(self, performance: Dict, portfolios_df: pd.DataFrame) -> str:
        """
        Generate vanilla tearsheet in text format.
        
        Args:
            performance (Dict): Performance metrics
            portfolios_df (pd.DataFrame): Portfolio data
            
        Returns:
            str: Formatted tearsheet
        """
        tearsheet = []
        tearsheet.append("=" * 80)
        tearsheet.append("VN LONG-ONLY FUND STRATEGY - BACKTEST TEARSHEET")
        tearsheet.append("=" * 80)
        tearsheet.append("")
        
        # Strategy Overview
        tearsheet.append("STRATEGY OVERVIEW:")
        tearsheet.append("-" * 40)
        tearsheet.append(f"Strategy: VN Long-Only Fund Strategy")
        tearsheet.append(f"Period: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}")
        tearsheet.append(f"Benchmark: VNINDEX")
        tearsheet.append(f"Liquidity Filter: >10bn daily ADTV")
        tearsheet.append(f"Rebalancing: Monthly")
        tearsheet.append("")
        
        # Performance Metrics
        tearsheet.append("PERFORMANCE METRICS:")
        tearsheet.append("-" * 40)
        tearsheet.append(f"Initial Value: {performance.get('initial_value', 0):,.0f} VND")
        tearsheet.append(f"Final Value: {performance.get('final_value', 0):,.0f} VND")
        tearsheet.append(f"Total Return: {performance.get('total_return', 0):.2%}")
        tearsheet.append(f"Annual Return: {performance.get('annual_return', 0):.2%}")
        tearsheet.append(f"Sharpe Ratio: {performance.get('sharpe_ratio', 0):.3f}")
        tearsheet.append(f"Max Drawdown: {performance.get('max_drawdown', 0):.2%}")
        tearsheet.append("")
        
        # Portfolio Statistics
        tearsheet.append("PORTFOLIO STATISTICS:")
        tearsheet.append("-" * 40)
        tearsheet.append(f"Average Portfolio Size: {portfolios_df['portfolio_size'].mean():.1f} stocks")
        tearsheet.append(f"Average Regime Allocation: {portfolios_df['regime_allocation'].mean():.1%}")
        tearsheet.append("")
        
        # Regime Analysis
        tearsheet.append("REGIME ANALYSIS:")
        tearsheet.append("-" * 40)
        regime_counts = portfolios_df['regime'].value_counts()
        for regime, count in regime_counts.items():
            percentage = count / len(portfolios_df) * 100
            tearsheet.append(f"{regime}: {count} periods ({percentage:.1f}%)")
        tearsheet.append("")
        
        # Sector Analysis
        tearsheet.append("SECTOR ANALYSIS (Latest Portfolio):")
        tearsheet.append("-" * 40)
        latest_portfolio = portfolios_df[portfolios_df['date'] == portfolios_df['date'].max()]
        if not latest_portfolio.empty:
            sector_weights = latest_portfolio.groupby('sector')['weight'].sum().sort_values(ascending=False)
            for sector, weight in sector_weights.head(10).items():
                tearsheet.append(f"{sector}: {weight:.1%}")
        tearsheet.append("")
        
        # Top Holdings (Latest Portfolio)
        tearsheet.append("TOP 10 HOLDINGS (Latest Portfolio):")
        tearsheet.append("-" * 40)
        if not latest_portfolio.empty:
            top_holdings = latest_portfolio.nlargest(10, 'weight')[['ticker', 'sector', 'weight', 'composite_score']]
            for _, row in top_holdings.iterrows():
                tearsheet.append(f"{row['ticker']} ({row['sector']}): {row['weight']:.1%} (Score: {row['composite_score']:.3f})")
        tearsheet.append("")
        
        # Strategy Features
        tearsheet.append("STRATEGY FEATURES:")
        tearsheet.append("-" * 40)
        tearsheet.append("✓ Regime Detection: Simple volatility/return based (93.6% accuracy)")
        tearsheet.append("✓ Sector-Aware Factors: Quality-adjusted P/E across sectors")
        tearsheet.append("✓ Quality Awareness: ROAA positive only (dropped ROAE)")
        tearsheet.append("✓ Value Contrarian: P/E only (dropped P/B)")
        tearsheet.append("✓ Momentum Score: Multi-horizon with skip month")
        tearsheet.append("✓ Risk Management: Position and sector limits")
        tearsheet.append("")
        
        # Risk Management
        tearsheet.append("RISK MANAGEMENT:")
        tearsheet.append("-" * 40)
        tearsheet.append("• Maximum Position Size: 5%")
        tearsheet.append("• Maximum Sector Exposure: 30%")
        tearsheet.append("• Target Portfolio Size: 25 stocks")
        tearsheet.append("• Liquidity Filter: >10bn daily ADTV")
        tearsheet.append("• Regime-Based Allocation: 40-100% based on market conditions")
        tearsheet.append("")
        
        tearsheet.append("=" * 80)
        tearsheet.append("END OF TEARSHEET")
        tearsheet.append("=" * 80)
        
        return "\n".join(tearsheet)
    
    def save_results(self, results: Dict, output_path: str = None):
        """
        Save backtest results to file.
        
        Args:
            results (Dict): Backtest results
            output_path (str, optional): Output file path
        """
        if output_path is None:
            output_path = f"qvm_engine_v3_adopted_insights_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        try:
            with open(output_path, 'w') as f:
                f.write(results['tearsheet'])
            
            self.logger.info(f"Results saved to: {output_path}")
            
            # Also save portfolio data
            portfolio_path = output_path.replace('.txt', '_portfolios.csv')
            results['portfolios'].to_csv(portfolio_path, index=False)
            self.logger.info(f"Portfolio data saved to: {portfolio_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")


def main():
    """Main function to run the backtest."""
    print("QVM Engine v3 with Adopted Insights Strategy Backtesting")
    print("=" * 60)
    
    # Initialize backtest
    backtest = QVMEngineV3AdoptedInsightsBacktest(
        start_date='2020-01-01',
        end_date='2024-12-31'  # Adjust as needed
    )
    
    # Run backtest
    print("Running backtest...")
    results = backtest.run_backtest()
    
    if 'error' in results:
        print(f"Backtest failed: {results['error']}")
        return
    
    # Display results
    print("\n" + results['tearsheet'])
    
    # Save results
    backtest.save_results(results)
    
    print("\nBacktest completed successfully!")


if __name__ == "__main__":
    main() 