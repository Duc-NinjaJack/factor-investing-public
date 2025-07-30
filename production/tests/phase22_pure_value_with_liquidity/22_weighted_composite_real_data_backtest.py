#!/usr/bin/env python3
"""
================================================================================
Phase 22: Weighted Composite Real Data Backtesting
================================================================================
Purpose:
    Implement the weighted composite strategy (60% Value, 20% Quality, 20% Reversal)
    using real market data from the database with proper liquidity filtering.
    
    This combines the successful weighted composite approach from Phase 16 with
    the robust real data backtesting framework and database API.

Strategy:
    - Weighted Composite: 60% Value + 20% Quality + 20% Reversal (Momentum inverted)
    - Z-score normalization within universe
    - Quintile 5 selection (top 20%)
    - Equal weighting within portfolio
    - Real price data from database
    - ADTV liquidity filtering
    - Transaction cost modeling

Author: Quantitative Strategy Team
Date: January 2025
Status: PRODUCTION READY
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
from datetime import datetime, timedelta
import logging
import pickle
import argparse
import sys

# Import database API
sys.path.append('../../../production/database')
from connection import DatabaseManager, get_database_manager

# Import real data backtesting framework
sys.path.append('../../../production/scripts')
from real_data_backtesting import RealDataBacktesting

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WeightedCompositeBacktesting(RealDataBacktesting):
    """
    Enhanced backtesting engine that implements the weighted composite strategy
    from Phase 16 using real market data and database API.
    """
    
    def __init__(self, config_path: str = None, pickle_path: str = None):
        """
        Initialize the weighted composite backtesting engine.
        
        Args:
            config_path: Path to database configuration file
            pickle_path: Path to ADTV data pickle file
        """
        super().__init__(config_path, pickle_path)
        
        # Weighted composite configuration (from Phase 16)
        self.weighting_scheme = {
            'Value': 0.6,      # 60% Value factor
            'Quality': 0.2,    # 20% Quality factor  
            'Reversal': 0.2    # 20% Reversal factor (inverted momentum)
        }
        
        # Strategy configuration
        self.strategy_config = {
            'portfolio_size': 25,
            'quintile_selection': 0.8,  # Top 20% (Quintile 5)
            'rebalance_freq': 'M',      # Monthly rebalancing
            'transaction_cost': 0.002,  # 20 bps
            'initial_capital': 100_000_000  # 100M VND
        }
        
        logger.info("✅ Weighted Composite Backtesting Engine initialized")
        logger.info(f"   - Weighting Scheme: {self.weighting_scheme}")
        logger.info(f"   - Portfolio Size: {self.strategy_config['portfolio_size']}")
        logger.info(f"   - Quintile Selection: Top {self.strategy_config['quintile_selection']*100:.0f}%")
    
    def load_factor_data(self) -> Dict[str, pd.DataFrame]:
        """Load individual factor scores from database."""
        logger.info("📊 Loading individual factor scores from database...")
        
        # Database manager
        db_manager = get_database_manager()
        
        # Load individual factor scores
        start_date = self.backtest_config['start_date']
        factor_query = f"""
        SELECT date, ticker, Quality_Composite, Value_Composite, Momentum_Composite
        FROM factor_scores_qvm
        WHERE date >= '{start_date}'
        ORDER BY date, ticker
        """
        
        factor_data = pd.read_sql(
            factor_query,
            db_manager.get_engine()
        )
        factor_data['date'] = pd.to_datetime(factor_data['date'])
        
        logger.info(f"✅ Factor data loaded: {len(factor_data):,} records")
        logger.info(f"   - Date range: {factor_data['date'].min()} to {factor_data['date'].max()}")
        logger.info(f"   - Unique tickers: {factor_data['ticker'].nunique()}")
        
        return {
            'factor_data': factor_data,
            'adtv_data': self.load_data()['adtv_data']
        }
    
    def calculate_weighted_composite(self, factor_data: pd.DataFrame, 
                                   rebalance_date: pd.Timestamp) -> pd.DataFrame:
        """
        Calculate weighted composite scores using Phase 16 methodology.
        
        Methodology:
        1. Filter to rebalance date
        2. Create Momentum Reversal factor (-1 * Momentum_Composite)
        3. Z-score normalize all factors within universe
        4. Apply weighted combination: 60% Value + 20% Quality + 20% Reversal
        """
        # Filter to rebalance date
        factors_on_date = factor_data[factor_data['date'] == rebalance_date].copy()
        
        if len(factors_on_date) < 50:
            logger.warning(f"Insufficient data for {rebalance_date}: {len(factors_on_date)} stocks")
            return pd.DataFrame()
        
        # Step 1: Create Momentum Reversal factor
        factors_on_date['Momentum_Reversal'] = -1 * factors_on_date['Momentum_Composite']
        
        # Step 2: Z-score normalization within universe
        for factor in ['Quality_Composite', 'Value_Composite', 'Momentum_Reversal']:
            mean_val = factors_on_date[factor].mean()
            std_val = factors_on_date[factor].std()
            
            if std_val > 0:
                factors_on_date[f'{factor}_Z'] = (factors_on_date[factor] - mean_val) / std_val
            else:
                factors_on_date[f'{factor}_Z'] = 0.0
        
        # Step 3: Calculate weighted composite
        factors_on_date['Weighted_Composite'] = (
            self.weighting_scheme['Quality'] * factors_on_date['Quality_Composite_Z'] +
            self.weighting_scheme['Value'] * factors_on_date['Value_Composite_Z'] +
            self.weighting_scheme['Reversal'] * factors_on_date['Momentum_Reversal_Z']
        )
        
        logger.debug(f"Calculated weighted composite for {len(factors_on_date)} stocks on {rebalance_date}")
        
        return factors_on_date
    
    def run_weighted_composite_backtest(self, threshold_name: str, threshold_value: int,
                                      data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Run backtest using weighted composite strategy.
        
        Args:
            threshold_name: Name of liquidity threshold (e.g., '10B_VND')
            threshold_value: ADTV threshold value
            data: Dictionary containing factor_data and adtv_data
            
        Returns:
            Dictionary with backtest results
        """
        logger.info(f"🚀 Running weighted composite backtest for {threshold_name}...")
        
        factor_data = data['factor_data']
        adtv_data = data['adtv_data']
        
        # Get price data
        price_data = self.load_data()['price_data']
        price_pivot = price_data.pivot(
            index='trading_date', columns='ticker', values='close_price_adjusted'
        )
        returns = price_pivot.pct_change().dropna()
        
        # Get benchmark data
        benchmark_data = self.load_data()['benchmark']
        benchmark_returns = benchmark_data.set_index('date')['close'].pct_change().dropna()
        
        # Rebalancing dates (monthly)
        rebalance_dates = pd.date_range(
            start=returns.index.min(),
            end=returns.index.max(),
            freq=self.strategy_config['rebalance_freq']
        )
        rebalance_dates = rebalance_dates[rebalance_dates.isin(returns.index)]
        
        # Initialize tracking variables
        portfolio_returns_dict = {}
        portfolio_holdings = []
        portfolio_values = [self.strategy_config['initial_capital']]
        
        # Run backtest
        for i, rebalance_date in enumerate(rebalance_dates[:-1]):
            next_rebalance = rebalance_dates[i + 1]
            
            # Calculate weighted composite scores
            weighted_factors = self.calculate_weighted_composite(factor_data, rebalance_date)
            
            if weighted_factors.empty:
                continue
            
            # Get ADTV data for liquidity filtering
            if rebalance_date in adtv_data.index:
                adtv_scores = adtv_data.loc[rebalance_date].dropna()
            else:
                # Use forward fill
                adtv_scores = adtv_data.loc[:rebalance_date].iloc[-1].dropna()
            
            # Apply liquidity filter
            liquid_stocks = adtv_scores[adtv_scores >= threshold_value].index
            available_stocks = weighted_factors['ticker'].intersection(liquid_stocks)
            
            if len(available_stocks) < self.strategy_config['portfolio_size']:
                logger.warning(f"Insufficient liquid stocks on {rebalance_date}: {len(available_stocks)}")
                continue
            
            # Filter to available stocks
            available_factors = weighted_factors[weighted_factors['ticker'].isin(available_stocks)]
            
            # Select top stocks by weighted composite score (Quintile 5)
            q5_cutoff = available_factors['Weighted_Composite'].quantile(self.strategy_config['quintile_selection'])
            top_stocks = available_factors[available_factors['Weighted_Composite'] >= q5_cutoff]['ticker']
            
            if len(top_stocks) == 0:
                continue
            
            # Equal weight portfolio
            weights = pd.Series(1.0 / len(top_stocks), index=top_stocks)
            
            # Calculate portfolio returns for this period
            period_returns = returns.loc[rebalance_date:next_rebalance, top_stocks]
            portfolio_return = (period_returns * weights).sum(axis=1)
            
            # Apply transaction costs
            if i > 0:  # Not the first rebalancing
                portfolio_return.iloc[0] -= self.strategy_config['transaction_cost']
            
            # Store returns for this period
            for date, ret in portfolio_return.items():
                portfolio_returns_dict[date] = ret
            
            # Store portfolio holdings
            portfolio_holdings.append({
                'date': rebalance_date,
                'stocks': list(top_stocks),
                'weights': weights.to_dict(),
                'universe_size': len(available_stocks),
                'portfolio_size': len(top_stocks),
                'weighted_composite_scores': available_factors.set_index('ticker')['Weighted_Composite'].to_dict(),
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
            'benchmark': benchmark_returns
        }).dropna()
        
        # Calculate performance metrics
        metrics = self._calculate_performance_metrics(aligned_data)
        
        logger.info(f"✅ {threshold_name} weighted composite backtest complete")
        logger.info(f"   - Annual Return: {metrics['annual_return']:.2%}")
        logger.info(f"   - Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"   - Max Drawdown: {metrics['max_drawdown']:.2%}")
        logger.info(f"   - Alpha: {metrics['alpha']:.2%}")
        
        return {
            'returns': portfolio_returns_series,
            'benchmark_returns': benchmark_returns,
            'metrics': metrics,
            'holdings': portfolio_holdings,
            'portfolio_values': portfolio_values,
            'weighting_scheme': self.weighting_scheme,
            'strategy_config': self.strategy_config
        }
    
    def run_comparative_backtests(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """Run backtests for all liquidity thresholds."""
        logger.info("🔄 Running comparative weighted composite backtests...")
        
        # Run backtests
        backtest_results = {}
        
        for threshold_name, threshold_value in self.thresholds.items():
            results = self.run_weighted_composite_backtest(threshold_name, threshold_value, data)
            backtest_results[threshold_name] = results
        
        return backtest_results
    
    def create_weighted_composite_visualizations(self, backtest_results: Dict[str, Dict], 
                                               save_path: str = None):
        """Create comprehensive visualizations for weighted composite strategy."""
        logger.info("📈 Creating weighted composite performance visualizations...")
        
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
        ax1.set_title('Weighted Composite: Cumulative Returns', fontsize=12, fontweight='bold')
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
        ax2.set_title('Weighted Composite: Drawdown Analysis', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Drawdown')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Rolling Sharpe Ratio
        ax3 = plt.subplot(3, 3, 3)
        for threshold, results in backtest_results.items():
            rolling_sharpe = results['returns'].rolling(window=252).mean() / results['returns'].rolling(window=252).std() * np.sqrt(252)
            ax3.plot(range(len(rolling_sharpe)), rolling_sharpe.values, label=threshold, linewidth=2)
        ax3.set_title('Weighted Composite: Rolling Sharpe Ratio (1-Year)', fontsize=12, fontweight='bold')
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
        
        ax4.set_title('Weighted Composite: Performance Metrics', fontsize=12, fontweight='bold')
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
        ax5.set_title(f'Weighted Composite: Monthly Returns ({first_threshold})', fontsize=12, fontweight='bold')
        
        # 6. Portfolio Holdings Evolution
        ax6 = plt.subplot(3, 3, 6)
        # Plot portfolio size evolution
        for threshold, results in backtest_results.items():
            holdings_dates = [h['date'] for h in results['holdings']]
            portfolio_sizes = [h['portfolio_size'] for h in results['holdings']]
            ax6.plot(holdings_dates, portfolio_sizes, label=threshold, marker='o', markersize=4)
        ax6.set_title('Weighted Composite: Portfolio Size Evolution', fontsize=12, fontweight='bold')
        ax6.set_ylabel('Number of Stocks')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Risk-Return Scatter
        ax7 = plt.subplot(3, 3, 7)
        for threshold, results in backtest_results.items():
            metrics = results['metrics']
            ax7.scatter(metrics['annual_volatility'], metrics['annual_return'], 
                       label=threshold, s=100, alpha=0.7)
        ax7.set_title('Weighted Composite: Risk-Return Profile', fontsize=12, fontweight='bold')
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
        ax8.set_title('Weighted Composite: Portfolio vs Benchmark', fontsize=12, fontweight='bold')
        ax8.set_ylabel('Cumulative Return')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        # 9. Weighting Scheme Visualization
        ax9 = plt.subplot(3, 3, 9)
        weights = list(self.weighting_scheme.values())
        labels = list(self.weighting_scheme.keys())
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        ax9.pie(weights, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax9.set_title('Weighted Composite: Factor Weights', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✅ Visualizations saved to {save_path}")
        
        plt.show()
    
    def generate_weighted_composite_report(self, backtest_results: Dict[str, Dict]) -> str:
        """Generate a comprehensive weighted composite backtesting report."""
        logger.info("📋 Generating weighted composite report...")
        
        report = []
        report.append("=" * 80)
        report.append("WEIGHTED COMPOSITE REAL DATA BACKTESTING REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Strategy Overview
        report.append("STRATEGY OVERVIEW")
        report.append("-" * 40)
        report.append(f"Weighting Scheme: {self.weighting_scheme}")
        report.append(f"Portfolio Size: {self.strategy_config['portfolio_size']} stocks")
        report.append(f"Selection Method: Top {self.strategy_config['quintile_selection']*100:.0f}% (Quintile 5)")
        report.append(f"Rebalancing: {self.strategy_config['rebalance_freq']}")
        report.append(f"Transaction Cost: {self.strategy_config['transaction_cost']*100:.1f}%")
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
        """Run complete weighted composite backtesting analysis."""
        logger.info("🎯 Starting complete weighted composite backtesting analysis...")
        
        try:
            # Load factor data
            data = self.load_factor_data()
            
            # Run backtests
            backtest_results = self.run_comparative_backtests(data)
            
            # Create visualizations
            if save_plots:
                plot_path = f"weighted_composite_backtesting_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                self.create_weighted_composite_visualizations(backtest_results, plot_path)
            else:
                self.create_weighted_composite_visualizations(backtest_results)
            
            # Generate report
            report = self.generate_weighted_composite_report(backtest_results)
            
            if save_report:
                report_path = f"weighted_composite_backtesting_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                with open(report_path, 'w') as f:
                    f.write(report)
                logger.info(f"✅ Report saved to {report_path}")
            
            print("\n" + "=" * 80)
            print("WEIGHTED COMPOSITE BACKTESTING COMPLETE")
            print("=" * 80)
            print(report)
            
            return backtest_results
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
            raise

def main():
    """Main function to run the weighted composite backtesting analysis."""
    parser = argparse.ArgumentParser(description='Weighted Composite Real Data Backtesting')
    parser.add_argument('--config', type=str, help='Path to database configuration file')
    parser.add_argument('--pickle', type=str, help='Path to ADTV data pickle file')
    parser.add_argument('--no-plots', action='store_true', help='Skip saving plots')
    parser.add_argument('--no-report', action='store_true', help='Skip saving report')
    
    args = parser.parse_args()
    
    try:
        # Initialize weighted composite backtesting engine
        backtesting = WeightedCompositeBacktesting(
            config_path=args.config,
            pickle_path=args.pickle
        )
        
        # Run complete analysis
        results = backtesting.run_complete_analysis(
            save_plots=not args.no_plots,
            save_report=not args.no_report
        )
        
        logger.info("✅ Weighted composite backtesting completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Backtesting failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()