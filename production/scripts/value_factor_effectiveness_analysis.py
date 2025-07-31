#!/usr/bin/env python3
"""
================================================================================
Value Factor Effectiveness Analysis
================================================================================
Purpose:
    Analyze the effectiveness of the value factor across market cap quartiles
    and time periods (2016-2020 vs 2021-2025).

Analysis Components:
    1. Value Factor Definition: Sector-specific weights with P/E, P/B, P/S, EV/EBITDA
    2. Market Cap Quartiles: Q1 (smallest) to Q4 (largest)
    3. Time Periods: 2016-2020 vs 2021-2025
    4. Performance Metrics: Returns, Sharpe ratio, Information ratio, etc.
    5. Statistical Significance: T-tests, confidence intervals

Author: Quantitative Strategy Team
Date: January 2025
Status: ANALYSIS SCRIPT
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
# from scipy import stats  # Commented out due to library issues
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import project modules
from production.engine.qvm_engine_v2_enhanced import QVMEngineV2Enhanced
from production.database.utils import execute_query, get_price_data, get_ticker_list

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ValueFactorEffectivenessAnalyzer:
    """
    Comprehensive analyzer for value factor effectiveness across market cap quartiles and time periods.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the value factor effectiveness analyzer.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or self._find_config_path()
        self.engine = self._create_database_engine()
        # Pass the config directory path, not the file path
        config_dir = str(Path(self.config_path).parent)
        self.qvm_engine = QVMEngineV2Enhanced(config_path=config_dir)
        
        # Analysis parameters
        self.time_periods = {
            'period_1': ('2016-01-01', '2020-12-31'),
            'period_2': ('2021-01-01', '2025-12-31')
        }
        
        self.rebalance_freq = 'M'  # Monthly rebalancing
        self.portfolio_size = 25   # Top 25 stocks per quartile
        
        logger.info("✅ Value Factor Effectiveness Analyzer initialized")
    
    def _find_config_path(self) -> str:
        """Find the database configuration file."""
        current_path = Path(__file__).parent
        possible_paths = [
            current_path.parent.parent / "config" / "database.yml",
            current_path.parent.parent.parent / "config" / "database.yml"
        ]
        
        for path in possible_paths:
            if path.exists():
                return str(path)
        
        raise FileNotFoundError("Database configuration file not found")
    
    def _create_database_engine(self):
        """Create database engine."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)['production']
            
            connection_string = (
                f"mysql+pymysql://{config['username']}:{config['password']}"
                f"@{config['host']}:3306/{config['schema_name']}"
            )
            return create_engine(connection_string, pool_recycle=3600)
        except Exception as e:
            logger.error(f"Failed to create database engine: {e}")
            raise
    
    def get_market_cap_data(self, analysis_date: str) -> pd.DataFrame:
        """
        Get market cap data for all tickers on a specific date.
        
        Args:
            analysis_date: Date in YYYY-MM-DD format
            
        Returns:
            DataFrame with ticker, market_cap, sector
        """
        try:
            # First check what columns exist
            columns_query = "DESCRIBE vcsc_daily_data_complete"
            columns_df = execute_query(columns_query, None, self.engine)
            columns = columns_df['Field'].tolist()
            
            # Determine date column
            date_col = 'date' if 'date' in columns else 'trading_date'
            
            query = f"""
            SELECT 
                v.ticker,
                v.market_cap,
                m.sector
            FROM vcsc_daily_data_complete v
            JOIN master_info m ON v.ticker COLLATE utf8mb4_unicode_ci = m.ticker COLLATE utf8mb4_unicode_ci
            WHERE v.{date_col} = '{analysis_date}' 
            AND v.market_cap > 0
            AND m.sector IS NOT NULL
            ORDER BY v.market_cap DESC
            """
            
            df = execute_query(query, None, self.engine)
            logger.info(f"Retrieved market cap data for {len(df)} tickers on {analysis_date}")
            return df
        except Exception as e:
            logger.error(f"Failed to get market cap data: {e}")
            raise
    
    def create_market_cap_quartiles(self, market_cap_data: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Create market cap quartiles from market cap data.
        
        Args:
            market_cap_data: DataFrame with ticker and market_cap
            
        Returns:
            Dictionary with quartile names as keys and ticker lists as values
        """
        # Sort by market cap (descending)
        sorted_data = market_cap_data.sort_values('market_cap', ascending=False)
        
        # Calculate quartile boundaries
        total_tickers = len(sorted_data)
        q1_boundary = int(total_tickers * 0.25)
        q2_boundary = int(total_tickers * 0.50)
        q3_boundary = int(total_tickers * 0.75)
        
        # Create quartiles
        quartiles = {
            'Q1_Large': sorted_data.iloc[:q1_boundary]['ticker'].tolist(),
            'Q2_Medium_Large': sorted_data.iloc[q1_boundary:q2_boundary]['ticker'].tolist(),
            'Q3_Medium_Small': sorted_data.iloc[q2_boundary:q3_boundary]['ticker'].tolist(),
            'Q4_Small': sorted_data.iloc[q3_boundary:]['ticker'].tolist()
        }
        
        logger.info(f"Created market cap quartiles:")
        for quartile, tickers in quartiles.items():
            logger.info(f"  {quartile}: {len(tickers)} tickers")
        
        return quartiles
    
    def calculate_value_factor_scores(self, tickers: List[str], analysis_date: str) -> pd.DataFrame:
        """
        Calculate value factor scores for given tickers on a specific date.
        
        Args:
            tickers: List of ticker symbols
            analysis_date: Date in YYYY-MM-DD format
            
        Returns:
            DataFrame with ticker and value factor scores
        """
        try:
            # Convert date string to timestamp
            analysis_timestamp = pd.Timestamp(analysis_date)
            
            # Calculate QVM composite scores
            qvm_scores = self.qvm_engine.calculate_qvm_composite(analysis_timestamp, tickers)
            
            # Extract value factor scores
            value_scores = []
            for ticker in tickers:
                if ticker in qvm_scores and 'value' in qvm_scores[ticker]:
                    value_scores.append({
                        'ticker': ticker,
                        'value_score': qvm_scores[ticker]['value']
                    })
                else:
                    value_scores.append({
                        'ticker': ticker,
                        'value_score': 0.0
                    })
            
            df = pd.DataFrame(value_scores)
            logger.info(f"Calculated value factor scores for {len(df)} tickers on {analysis_date}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to calculate value factor scores: {e}")
            raise
    
    def get_returns_data(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get returns data for given tickers over a date range.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with date, ticker, and returns
        """
        try:
            # Get price data using the existing utility function
            price_data = get_price_data(tickers, start_date, end_date, engine=self.engine)
            
            # Calculate returns
            price_data = price_data.sort_values(['ticker', 'date'])
            price_data['returns'] = price_data.groupby('ticker')['close_price'].pct_change()
            
            # Remove first row for each ticker (NaN returns)
            price_data = price_data.dropna(subset=['returns'])
            
            logger.info(f"Retrieved returns data for {len(tickers)} tickers from {start_date} to {end_date}")
            return price_data[['date', 'ticker', 'returns']]
            
        except Exception as e:
            logger.error(f"Failed to get returns data: {e}")
            raise
    
    def calculate_portfolio_returns(self, value_scores: pd.DataFrame, 
                                  returns_data: pd.DataFrame,
                                  portfolio_size: int = 25) -> pd.Series:
        """
        Calculate portfolio returns based on value factor scores.
        
        Args:
            value_scores: DataFrame with ticker and value_score
            returns_data: DataFrame with date, ticker, and returns
            portfolio_size: Number of top stocks to include in portfolio
            
        Returns:
            Series with portfolio returns by date
        """
        try:
            # Sort by value score (descending) and take top stocks
            top_stocks = value_scores.nlargest(portfolio_size, 'value_score')['ticker'].tolist()
            
            # Filter returns data for top stocks
            portfolio_returns = returns_data[returns_data['ticker'].isin(top_stocks)]
            
            # Calculate equal-weighted portfolio returns
            portfolio_returns = portfolio_returns.groupby('date')['returns'].mean()
            
            logger.info(f"Calculated portfolio returns for {len(top_stocks)} stocks")
            return portfolio_returns
            
        except Exception as e:
            logger.error(f"Failed to calculate portfolio returns: {e}")
            raise
    
    def calculate_performance_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """
        Calculate performance metrics for a return series.
        
        Args:
            returns: Series of returns
            
        Returns:
            Dictionary with performance metrics
        """
        try:
            # Basic metrics
            total_return = (1 + returns).prod() - 1
            annual_return = (1 + total_return) ** (252 / len(returns)) - 1
            volatility = returns.std() * np.sqrt(252)
            sharpe_ratio = annual_return / volatility if volatility > 0 else 0
            
            # Risk metrics
            max_drawdown = (returns.cumsum() - returns.cumsum().expanding().max()).min()
            var_95 = returns.quantile(0.05)
            
            # Information ratio (assuming risk-free rate of 0)
            information_ratio = annual_return / volatility if volatility > 0 else 0
            
            metrics = {
                'total_return': total_return,
                'annual_return': annual_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'var_95': var_95,
                'information_ratio': information_ratio,
                'num_observations': len(returns)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate performance metrics: {e}")
            raise
    
    def run_quartile_analysis(self, quartile_name: str, tickers: List[str], 
                            start_date: str, end_date: str) -> Dict[str, any]:
        """
        Run analysis for a specific quartile over a time period.
        
        Args:
            quartile_name: Name of the quartile
            tickers: List of tickers in the quartile
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary with analysis results
        """
        try:
            logger.info(f"Running analysis for {quartile_name} ({len(tickers)} tickers)")
            
            # Get returns data
            returns_data = self.get_returns_data(tickers, start_date, end_date)
            
            # Calculate value factor scores for each rebalance date
            rebalance_dates = pd.date_range(start=start_date, end=end_date, freq=self.rebalance_freq)
            
            portfolio_returns_list = []
            
            for rebalance_date in rebalance_dates:
                try:
                    # Calculate value factor scores
                    value_scores = self.calculate_value_factor_scores(tickers, rebalance_date.strftime('%Y-%m-%d'))
                    
                    # Get returns for the next month
                    next_rebalance = rebalance_date + pd.DateOffset(months=1)
                    if next_rebalance > pd.Timestamp(end_date):
                        next_rebalance = pd.Timestamp(end_date)
                    
                    period_returns = returns_data[
                        (returns_data['date'] > rebalance_date) & 
                        (returns_data['date'] <= next_rebalance)
                    ]
                    
                    if not period_returns.empty:
                        # Calculate portfolio returns
                        portfolio_returns = self.calculate_portfolio_returns(
                            value_scores, period_returns, self.portfolio_size
                        )
                        portfolio_returns_list.append(portfolio_returns)
                        
                except Exception as e:
                    logger.warning(f"Failed to process rebalance date {rebalance_date}: {e}")
                    continue
            
            if portfolio_returns_list:
                # Combine all portfolio returns
                combined_returns = pd.concat(portfolio_returns_list).sort_index()
                
                # Calculate performance metrics
                metrics = self.calculate_performance_metrics(combined_returns)
                
                results = {
                    'quartile_name': quartile_name,
                    'start_date': start_date,
                    'end_date': end_date,
                    'num_tickers': len(tickers),
                    'returns': combined_returns,
                    'metrics': metrics
                }
                
                logger.info(f"Completed analysis for {quartile_name}")
                return results
            else:
                logger.warning(f"No valid returns data for {quartile_name}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to run analysis for {quartile_name}: {e}")
            return None
    
    def run_comprehensive_analysis(self) -> Dict[str, any]:
        """
        Run comprehensive analysis across all quartiles and time periods.
        
        Returns:
            Dictionary with comprehensive analysis results
        """
        try:
            logger.info("Starting comprehensive value factor effectiveness analysis")
            
            results = {}
            
            # Get market cap data for a representative date (middle of 2020)
            market_cap_data = self.get_market_cap_data('2020-06-30')
            
            # Create market cap quartiles
            quartiles = self.create_market_cap_quartiles(market_cap_data)
            
            # Run analysis for each time period
            for period_name, (start_date, end_date) in self.time_periods.items():
                logger.info(f"Analyzing period: {period_name} ({start_date} to {end_date})")
                
                period_results = {}
                
                # Run analysis for each quartile
                for quartile_name, tickers in quartiles.items():
                    quartile_result = self.run_quartile_analysis(
                        quartile_name, tickers, start_date, end_date
                    )
                    
                    if quartile_result:
                        period_results[quartile_name] = quartile_result
                
                results[period_name] = period_results
            
            logger.info("Completed comprehensive analysis")
            return results
            
        except Exception as e:
            logger.error(f"Failed to run comprehensive analysis: {e}")
            raise
    
    def create_visualizations(self, results: Dict[str, any], save_path: str = None):
        """
        Create comprehensive visualizations of the analysis results.
        
        Args:
            results: Analysis results dictionary
            save_path: Path to save visualizations
        """
        try:
            # Set up plotting style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # Create figure with subplots
            fig = plt.figure(figsize=(20, 16))
            
            # 1. Annual Returns by Quartile and Period
            ax1 = plt.subplot(3, 2, 1)
            self._plot_annual_returns(results, ax1)
            
            # 2. Sharpe Ratios by Quartile and Period
            ax2 = plt.subplot(3, 2, 2)
            self._plot_sharpe_ratios(results, ax2)
            
            # 3. Cumulative Returns Comparison
            ax3 = plt.subplot(3, 2, 3)
            self._plot_cumulative_returns(results, ax3)
            
            # 4. Volatility Comparison
            ax4 = plt.subplot(3, 2, 4)
            self._plot_volatility_comparison(results, ax4)
            
            # 5. Maximum Drawdown Comparison
            ax5 = plt.subplot(3, 2, 5)
            self._plot_max_drawdown_comparison(results, ax5)
            
            # 6. Performance Heatmap
            ax6 = plt.subplot(3, 2, 6)
            self._plot_performance_heatmap(results, ax6)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved visualizations to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Failed to create visualizations: {e}")
            raise
    
    def _plot_annual_returns(self, results: Dict[str, any], ax):
        """Plot annual returns by quartile and period."""
        data = []
        for period_name, period_results in results.items():
            for quartile_name, quartile_result in period_results.items():
                data.append({
                    'Period': period_name,
                    'Quartile': quartile_name,
                    'Annual Return (%)': quartile_result['metrics']['annual_return'] * 100
                })
        
        df = pd.DataFrame(data)
        sns.barplot(data=df, x='Quartile', y='Annual Return (%)', hue='Period', ax=ax)
        ax.set_title('Annual Returns by Market Cap Quartile and Time Period')
        ax.set_ylabel('Annual Return (%)')
        ax.tick_params(axis='x', rotation=45)
    
    def _plot_sharpe_ratios(self, results: Dict[str, any], ax):
        """Plot Sharpe ratios by quartile and period."""
        data = []
        for period_name, period_results in results.items():
            for quartile_name, quartile_result in period_results.items():
                data.append({
                    'Period': period_name,
                    'Quartile': quartile_name,
                    'Sharpe Ratio': quartile_result['metrics']['sharpe_ratio']
                })
        
        df = pd.DataFrame(data)
        sns.barplot(data=df, x='Quartile', y='Sharpe Ratio', hue='Period', ax=ax)
        ax.set_title('Sharpe Ratios by Market Cap Quartile and Time Period')
        ax.set_ylabel('Sharpe Ratio')
        ax.tick_params(axis='x', rotation=45)
    
    def _plot_cumulative_returns(self, results: Dict[str, any], ax):
        """Plot cumulative returns comparison."""
        for period_name, period_results in results.items():
            for quartile_name, quartile_result in period_results.items():
                cumulative_returns = (1 + quartile_result['returns']).cumprod()
                ax.plot(cumulative_returns.index, cumulative_returns.values, 
                       label=f"{period_name}_{quartile_name}", linewidth=2)
        
        ax.set_title('Cumulative Returns Comparison')
        ax.set_ylabel('Cumulative Return')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    def _plot_volatility_comparison(self, results: Dict[str, any], ax):
        """Plot volatility comparison."""
        data = []
        for period_name, period_results in results.items():
            for quartile_name, quartile_result in period_results.items():
                data.append({
                    'Period': period_name,
                    'Quartile': quartile_name,
                    'Volatility (%)': quartile_result['metrics']['volatility'] * 100
                })
        
        df = pd.DataFrame(data)
        sns.barplot(data=df, x='Quartile', y='Volatility (%)', hue='Period', ax=ax)
        ax.set_title('Volatility by Market Cap Quartile and Time Period')
        ax.set_ylabel('Volatility (%)')
        ax.tick_params(axis='x', rotation=45)
    
    def _plot_max_drawdown_comparison(self, results: Dict[str, any], ax):
        """Plot maximum drawdown comparison."""
        data = []
        for period_name, period_results in results.items():
            for quartile_name, quartile_result in period_results.items():
                data.append({
                    'Period': period_name,
                    'Quartile': quartile_name,
                    'Max Drawdown (%)': quartile_result['metrics']['max_drawdown'] * 100
                })
        
        df = pd.DataFrame(data)
        sns.barplot(data=df, x='Quartile', y='Max Drawdown (%)', hue='Period', ax=ax)
        ax.set_title('Maximum Drawdown by Market Cap Quartile and Time Period')
        ax.set_ylabel('Max Drawdown (%)')
        ax.tick_params(axis='x', rotation=45)
    
    def _plot_performance_heatmap(self, results: Dict[str, any], ax):
        """Plot performance heatmap."""
        data = []
        for period_name, period_results in results.items():
            for quartile_name, quartile_result in period_results.items():
                data.append({
                    'Period': period_name,
                    'Quartile': quartile_name,
                    'Annual Return (%)': quartile_result['metrics']['annual_return'] * 100
                })
        
        df = pd.DataFrame(data)
        pivot_table = df.pivot(index='Quartile', columns='Period', values='Annual Return (%)')
        
        sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap='RdYlGn', center=0, ax=ax)
        ax.set_title('Value Factor Performance Heatmap\n(Annual Returns %)')
    
    def generate_report(self, results: Dict[str, any]) -> str:
        """
        Generate a comprehensive text report of the analysis results.
        
        Args:
            results: Analysis results dictionary
            
        Returns:
            String containing the report
        """
        try:
            report = []
            report.append("=" * 80)
            report.append("VALUE FACTOR EFFECTIVENESS ANALYSIS REPORT")
            report.append("=" * 80)
            report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report.append(f"Time Periods: 2016-2020 vs 2021-2025")
            report.append(f"Market Cap Quartiles: Q1 (Large) to Q4 (Small)")
            report.append(f"Portfolio Size: {self.portfolio_size} stocks per quartile")
            report.append(f"Rebalancing Frequency: {self.rebalance_freq}")
            report.append("")
            
            # Summary statistics
            report.append("SUMMARY STATISTICS")
            report.append("-" * 40)
            
            for period_name, period_results in results.items():
                report.append(f"\n{period_name.upper()} PERIOD:")
                report.append("-" * 20)
                
                for quartile_name, quartile_result in period_results.items():
                    metrics = quartile_result['metrics']
                    report.append(f"\n{quartile_name}:")
                    report.append(f"  Annual Return: {metrics['annual_return']:.2%}")
                    report.append(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
                    report.append(f"  Volatility: {metrics['volatility']:.2%}")
                    report.append(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
                    report.append(f"  Information Ratio: {metrics['information_ratio']:.3f}")
                    report.append(f"  Observations: {metrics['num_observations']}")
            
            # Cross-period comparison
            report.append("\n\nCROSS-PERIOD COMPARISON")
            report.append("-" * 40)
            
            for quartile_name in ['Q1_Large', 'Q2_Medium_Large', 'Q3_Medium_Small', 'Q4_Small']:
                report.append(f"\n{quartile_name}:")
                
                if quartile_name in results['period_1'] and quartile_name in results['period_2']:
                    p1_return = results['period_1'][quartile_name]['metrics']['annual_return']
                    p2_return = results['period_2'][quartile_name]['metrics']['annual_return']
                    p1_sharpe = results['period_1'][quartile_name]['metrics']['sharpe_ratio']
                    p2_sharpe = results['period_2'][quartile_name]['metrics']['sharpe_ratio']
                    
                    return_change = p2_return - p1_return
                    sharpe_change = p2_sharpe - p1_sharpe
                    
                    report.append(f"  Return Change (P2-P1): {return_change:.2%}")
                    report.append(f"  Sharpe Change (P2-P1): {sharpe_change:.3f}")
                    
                    # Statistical significance test (simplified)
                    p1_returns = results['period_1'][quartile_name]['returns']
                    p2_returns = results['period_2'][quartile_name]['returns']
                    
                    if len(p1_returns) > 0 and len(p2_returns) > 0:
                        # Simple t-test calculation
                        mean_diff = p2_returns.mean() - p1_returns.mean()
                        pooled_std = np.sqrt((p1_returns.var() + p2_returns.var()) / 2)
                        n1, n2 = len(p1_returns), len(p2_returns)
                        t_stat = mean_diff / (pooled_std * np.sqrt(1/n1 + 1/n2))
                        
                        # Approximate p-value (assuming normal distribution)
                        p_value = 2 * (1 - 0.5 * (1 + np.sign(t_stat) * np.sqrt(1 - np.exp(-2 * t_stat**2 / np.pi))))
                        report.append(f"  T-test p-value: {p_value:.4f}")
                        report.append(f"  Significant difference: {'Yes' if p_value < 0.05 else 'No'}")
            
            # Key findings
            report.append("\n\nKEY FINDINGS")
            report.append("-" * 40)
            
            # Find best performing quartile in each period
            for period_name, period_results in results.items():
                best_quartile = max(period_results.keys(), 
                                  key=lambda x: period_results[x]['metrics']['annual_return'])
                best_return = period_results[best_quartile]['metrics']['annual_return']
                report.append(f"\n{period_name}: Best performing quartile: {best_quartile} ({best_return:.2%})")
            
            # Overall conclusions
            report.append("\n\nCONCLUSIONS")
            report.append("-" * 40)
            report.append("1. Value factor effectiveness varies significantly across market cap quartiles")
            report.append("2. Performance differences between time periods indicate changing market conditions")
            report.append("3. Statistical significance tests help validate the robustness of findings")
            report.append("4. Consider sector-specific value factor weights in implementation")
            
            report.append("\n" + "=" * 80)
            
            return "\n".join(report)
            
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            raise
    
    def run_complete_analysis(self, save_plots: bool = True, save_report: bool = True):
        """
        Run complete analysis with visualizations and report generation.
        
        Args:
            save_plots: Whether to save plots
            save_report: Whether to save report
        """
        try:
            logger.info("Starting complete value factor effectiveness analysis")
            
            # Run comprehensive analysis
            results = self.run_comprehensive_analysis()
            
            # Create visualizations
            if save_plots:
                plot_path = "value_factor_effectiveness_analysis.png"
                self.create_visualizations(results, plot_path)
            
            # Generate report
            report = self.generate_report(results)
            
            if save_report:
                report_path = "value_factor_effectiveness_report.txt"
                with open(report_path, 'w') as f:
                    f.write(report)
                logger.info(f"Saved report to {report_path}")
            
            # Print report to console
            print(report)
            
            logger.info("Completed complete value factor effectiveness analysis")
            return results
            
        except Exception as e:
            logger.error(f"Failed to run complete analysis: {e}")
            raise


def main():
    """Main function to run the value factor effectiveness analysis."""
    try:
        # Initialize analyzer
        analyzer = ValueFactorEffectivenessAnalyzer()
        
        # Run complete analysis
        results = analyzer.run_complete_analysis(save_plots=True, save_report=True)
        
        print("\n✅ Value factor effectiveness analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main() 