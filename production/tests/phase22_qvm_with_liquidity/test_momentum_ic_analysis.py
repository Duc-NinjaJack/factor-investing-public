#!/usr/bin/env python3
"""
Momentum Factor Information Coefficient (IC) Analysis
====================================================

This script tests the Information Coefficient (IC) of the momentum factor
for two periods: 2016-2020 and 2021-2025.

IC is calculated as the correlation between factor values and forward returns.
This analysis helps validate the predictive power of the momentum factor.

Author: Factor Investing Team
Date: 2025-07-30
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add paths for imports
sys.path.append('../../../production/database')
sys.path.append('../../../production/engine')
sys.path.append('../../../production/scripts')

try:
    from database.connection import get_engine
    from engine.qvm_engine_v2_enhanced import QVMEngineV2Enhanced
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all dependencies are installed and paths are correct.")
    sys.exit(1)

class MomentumICAnalysis:
    """
    Comprehensive analysis of momentum factor Information Coefficient (IC).
    
    This class calculates momentum factors, forward returns, and IC statistics
    for validation of the momentum factor's predictive power.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the momentum IC analysis.
        
        Args:
            config_path: Path to configuration files
        """
        self.logger = self._setup_logging()
        self.logger.info("Initializing Momentum IC Analysis")
        
        # Initialize QVM engine
        try:
            self.engine = QVMEngineV2Enhanced(config_path)
            self.logger.info("QVM Engine initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize QVM Engine: {e}")
            raise
        
        # Define analysis periods
        self.periods = {
            '2016-2020': {
                'start_date': '2016-01-01',
                'end_date': '2020-12-31',
                'description': 'Pre-COVID period'
            },
            '2021-2025': {
                'start_date': '2021-01-01', 
                'end_date': '2025-12-31',
                'description': 'Post-COVID period'
            }
        }
        
        # IC analysis parameters
        self.forward_return_horizons = [1, 3, 6, 12]  # months
        self.rebalance_frequency = 'monthly'  # or 'quarterly'
        
        self.logger.info("Momentum IC Analysis initialized successfully")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger('MomentumICAnalysis')
    
    def get_universe(self, analysis_date: pd.Timestamp) -> List[str]:
        """
        Get the universe of stocks for a given date.
        
        Args:
            analysis_date: Date for universe construction
            
        Returns:
            List of ticker symbols
        """
        try:
            # Query for stocks with sufficient data
            query = f"""
            SELECT DISTINCT ticker
            FROM equity_history
            WHERE date <= '{analysis_date.date()}'
              AND close > 5000  -- Minimum price filter
            GROUP BY ticker
            HAVING COUNT(*) >= 252  -- At least 1 year of data
            ORDER BY ticker
            """
            
            with self.engine.engine.connect() as conn:
                result = pd.read_sql(query, conn)
            
            universe = result['ticker'].tolist()
            self.logger.info(f"Universe constructed: {len(universe)} stocks for {analysis_date.date()}")
            return universe
            
        except Exception as e:
            self.logger.error(f"Failed to get universe: {e}")
            return []
    
    def calculate_momentum_factors(self, analysis_date: pd.Timestamp, 
                                 universe: List[str]) -> Dict[str, float]:
        """
        Calculate momentum factors for a given date and universe.
        
        Args:
            analysis_date: Date for factor calculation
            universe: List of ticker symbols
            
        Returns:
            Dictionary mapping tickers to momentum scores
        """
        try:
            # Get fundamental data (needed for sector mapping)
            fundamental_data = self.engine.get_fundamentals_correct_timing(
                analysis_date, universe
            )
            
            if fundamental_data.empty:
                self.logger.warning(f"No fundamental data available for {analysis_date.date()}")
                return {}
            
            # Calculate momentum composite
            momentum_scores = self.engine._calculate_enhanced_momentum_composite(
                fundamental_data, analysis_date, universe
            )
            
            self.logger.info(f"Calculated momentum factors for {len(momentum_scores)} stocks")
            return momentum_scores
            
        except Exception as e:
            self.logger.error(f"Failed to calculate momentum factors: {e}")
            return {}
    
    def calculate_forward_returns(self, analysis_date: pd.Timestamp, 
                                universe: List[str], 
                                horizon_months: int) -> Dict[str, float]:
        """
        Calculate forward returns for a given horizon.
        
        Args:
            analysis_date: Starting date
            universe: List of ticker symbols
            horizon_months: Forward return horizon in months
            
        Returns:
            Dictionary mapping tickers to forward returns
        """
        try:
            # Calculate end date
            end_date = analysis_date + pd.DateOffset(months=horizon_months)
            
            # Get price data
            ticker_str = "', '".join(universe)
            query = f"""
            SELECT 
                ticker,
                date,
                close as adj_close
            FROM equity_history
            WHERE ticker IN ('{ticker_str}')
              AND date BETWEEN '{analysis_date.date()}' AND '{end_date.date()}'
            ORDER BY ticker, date
            """
            
            with self.engine.engine.connect() as conn:
                price_data = pd.read_sql(query, conn, parse_dates=['date'])
            
            if price_data.empty:
                self.logger.warning(f"No price data available for forward return calculation")
                return {}
            
            # Calculate returns
            forward_returns = {}
            for ticker in universe:
                ticker_data = price_data[price_data['ticker'] == ticker].sort_values('date')
                
                if len(ticker_data) >= 2:
                    start_price = ticker_data.iloc[0]['adj_close']
                    end_price = ticker_data.iloc[-1]['adj_close']
                    
                    if start_price > 0:
                        forward_returns[ticker] = (end_price / start_price) - 1
                    else:
                        forward_returns[ticker] = np.nan
                else:
                    forward_returns[ticker] = np.nan
            
            # Remove NaN values
            forward_returns = {k: v for k, v in forward_returns.items() if not np.isnan(v)}
            
            self.logger.info(f"Calculated {horizon_months}M forward returns for {len(forward_returns)} stocks")
            return forward_returns
            
        except Exception as e:
            self.logger.error(f"Failed to calculate forward returns: {e}")
            return {}
    
    def calculate_ic(self, factor_scores: Dict[str, float], 
                    forward_returns: Dict[str, float]) -> float:
        """
        Calculate Information Coefficient (IC) as correlation between factor and returns.
        
        Args:
            factor_scores: Dictionary mapping tickers to factor scores
            forward_returns: Dictionary mapping tickers to forward returns
            
        Returns:
            IC value (correlation coefficient)
        """
        try:
            # Find common tickers
            common_tickers = set(factor_scores.keys()) & set(forward_returns.keys())
            
            if len(common_tickers) < 10:  # Need minimum sample size
                self.logger.warning(f"Insufficient data for IC calculation: {len(common_tickers)} stocks")
                return np.nan
            
            # Create series
            factor_series = pd.Series([factor_scores[t] for t in common_tickers], 
                                    index=list(common_tickers))
            return_series = pd.Series([forward_returns[t] for t in common_tickers], 
                                    index=list(common_tickers))
            
            # Calculate Spearman correlation (rank correlation)
            ic = factor_series.corr(return_series, method='spearman')
            
            return ic
            
        except Exception as e:
            self.logger.error(f"Failed to calculate IC: {e}")
            return np.nan
    
    def run_ic_analysis_for_period(self, period_name: str, 
                                 period_config: Dict) -> Dict:
        """
        Run complete IC analysis for a specific period.
        
        Args:
            period_name: Name of the period (e.g., '2016-2020')
            period_config: Period configuration dictionary
            
        Returns:
            Dictionary containing IC analysis results
        """
        self.logger.info(f"Starting IC analysis for period: {period_name}")
        
        start_date = pd.to_datetime(period_config['start_date'])
        end_date = pd.to_datetime(period_config['end_date'])
        
        # Generate rebalance dates (monthly)
        rebalance_dates = pd.date_range(
            start=start_date + pd.DateOffset(months=12),  # Need 12 months for momentum
            end=end_date - pd.DateOffset(months=12),      # Need 12 months for forward returns
            freq='M'
        )
        
        self.logger.info(f"Generated {len(rebalance_dates)} rebalance dates")
        
        # Store results
        ic_results = {
            'period': period_name,
            'description': period_config['description'],
            'rebalance_dates': [],
            'ic_by_horizon': {horizon: [] for horizon in self.forward_return_horizons},
            'factor_coverage': [],
            'return_coverage': []
        }
        
        # Calculate IC for each rebalance date
        for i, rebalance_date in enumerate(rebalance_dates):
            self.logger.info(f"Processing {rebalance_date.date()} ({i+1}/{len(rebalance_dates)})")
            
            # Get universe
            universe = self.get_universe(rebalance_date)
            if len(universe) < 20:  # Need minimum universe size
                self.logger.warning(f"Universe too small for {rebalance_date.date()}: {len(universe)} stocks")
                continue
            
            # Calculate momentum factors
            momentum_scores = self.calculate_momentum_factors(rebalance_date, universe)
            if not momentum_scores:
                continue
            
            ic_results['factor_coverage'].append(len(momentum_scores))
            ic_results['rebalance_dates'].append(rebalance_date)
            
            # Calculate IC for each horizon
            for horizon in self.forward_return_horizons:
                forward_returns = self.calculate_forward_returns(
                    rebalance_date, universe, horizon
                )
                
                if forward_returns:
                    ic_results['return_coverage'].append(len(forward_returns))
                    ic = self.calculate_ic(momentum_scores, forward_returns)
                    ic_results['ic_by_horizon'][horizon].append(ic)
                else:
                    ic_results['ic_by_horizon'][horizon].append(np.nan)
        
        # Calculate summary statistics
        ic_results['summary_stats'] = self._calculate_ic_summary_stats(ic_results)
        
        self.logger.info(f"Completed IC analysis for {period_name}")
        return ic_results
    
    def _calculate_ic_summary_stats(self, ic_results: Dict) -> Dict:
        """
        Calculate summary statistics for IC results.
        
        Args:
            ic_results: IC analysis results
            
        Returns:
            Dictionary containing summary statistics
        """
        summary_stats = {}
        
        for horizon in self.forward_return_horizons:
            ic_series = pd.Series(ic_results['ic_by_horizon'][horizon])
            ic_series = ic_series.dropna()
            
            if len(ic_series) > 0:
                mean_ic = ic_series.mean()
                std_ic = ic_series.std()
                t_stat = mean_ic / (std_ic / np.sqrt(len(ic_series))) if std_ic > 0 else 0
                hit_rate = (ic_series > 0).mean()
                
                summary_stats[f'{horizon}M'] = {
                    'mean_ic': mean_ic,
                    'std_ic': std_ic,
                    't_stat': t_stat,
                    'hit_rate': hit_rate,
                    'n_observations': len(ic_series),
                    'min_ic': ic_series.min(),
                    'max_ic': ic_series.max()
                }
            else:
                summary_stats[f'{horizon}M'] = {
                    'mean_ic': np.nan,
                    'std_ic': np.nan,
                    't_stat': np.nan,
                    'hit_rate': np.nan,
                    'n_observations': 0,
                    'min_ic': np.nan,
                    'max_ic': np.nan
                }
        
        return summary_stats
    
    def run_complete_analysis(self) -> Dict:
        """
        Run IC analysis for both periods.
        
        Returns:
            Dictionary containing results for both periods
        """
        self.logger.info("Starting complete IC analysis for both periods")
        
        results = {}
        
        for period_name, period_config in self.periods.items():
            self.logger.info(f"Analyzing period: {period_name}")
            results[period_name] = self.run_ic_analysis_for_period(period_name, period_config)
        
        # Add comparative analysis
        results['comparison'] = self._compare_periods(results)
        
        self.logger.info("Completed complete IC analysis")
        return results
    
    def _compare_periods(self, results: Dict) -> Dict:
        """
        Compare IC results between periods.
        
        Args:
            results: Results from both periods
            
        Returns:
            Dictionary containing comparative analysis
        """
        comparison = {}
        
        for horizon in self.forward_return_horizons:
            horizon_key = f'{horizon}M'
            
            ic_2016 = pd.Series(results['2016-2020']['ic_by_horizon'][horizon]).dropna()
            ic_2021 = pd.Series(results['2021-2025']['ic_by_horizon'][horizon]).dropna()
            
            if len(ic_2016) > 0 and len(ic_2021) > 0:
                # Statistical test for difference in means
                from scipy import stats
                t_stat, p_value = stats.ttest_ind(ic_2016, ic_2021)
                
                comparison[horizon_key] = {
                    'period_2016_mean': ic_2016.mean(),
                    'period_2021_mean': ic_2021.mean(),
                    'difference': ic_2021.mean() - ic_2016.mean(),
                    't_stat_diff': t_stat,
                    'p_value_diff': p_value,
                    'significant_diff': p_value < 0.05
                }
            else:
                comparison[horizon_key] = {
                    'period_2016_mean': np.nan,
                    'period_2021_mean': np.nan,
                    'difference': np.nan,
                    't_stat_diff': np.nan,
                    'p_value_diff': np.nan,
                    'significant_diff': False
                }
        
        return comparison
    
    def generate_report(self, results: Dict, save_path: str = None) -> str:
        """
        Generate a comprehensive report of IC analysis results.
        
        Args:
            results: IC analysis results
            save_path: Path to save the report
            
        Returns:
            Report text
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("MOMENTUM FACTOR INFORMATION COEFFICIENT (IC) ANALYSIS")
        report_lines.append("=" * 80)
        report_lines.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Summary for each period
        for period_name in ['2016-2020', '2021-2025']:
            period_results = results[period_name]
            summary_stats = period_results['summary_stats']
            
            report_lines.append(f"PERIOD: {period_name}")
            report_lines.append(f"Description: {period_results['description']}")
            report_lines.append(f"Rebalance Dates: {len(period_results['rebalance_dates'])}")
            report_lines.append("")
            
            report_lines.append("IC Summary Statistics by Horizon:")
            report_lines.append("-" * 50)
            
            for horizon_key, stats in summary_stats.items():
                if stats['n_observations'] > 0:
                    report_lines.append(f"{horizon_key} Forward Returns:")
                    report_lines.append(f"  Mean IC: {stats['mean_ic']:.4f}")
                    report_lines.append(f"  Std IC: {stats['std_ic']:.4f}")
                    report_lines.append(f"  T-Stat: {stats['t_stat']:.3f}")
                    report_lines.append(f"  Hit Rate: {stats['hit_rate']:.1%}")
                    report_lines.append(f"  Observations: {stats['n_observations']}")
                    report_lines.append(f"  Range: [{stats['min_ic']:.4f}, {stats['max_ic']:.4f}]")
                    report_lines.append("")
            
            # Quality assessment
            report_lines.append("Quality Assessment:")
            report_lines.append("-" * 20)
            
            for horizon_key, stats in summary_stats.items():
                if stats['n_observations'] > 0:
                    ic_quality = "✅ GOOD" if stats['mean_ic'] > 0.02 and stats['t_stat'] > 2.0 else "❌ POOR"
                    hit_quality = "✅ GOOD" if stats['hit_rate'] > 0.55 else "❌ POOR"
                    
                    report_lines.append(f"{horizon_key}: IC={ic_quality}, Hit Rate={hit_quality}")
            
            report_lines.append("")
            report_lines.append("=" * 80)
            report_lines.append("")
        
        # Comparative analysis
        comparison = results['comparison']
        report_lines.append("COMPARATIVE ANALYSIS")
        report_lines.append("=" * 50)
        
        for horizon_key, comp_stats in comparison.items():
            if not np.isnan(comp_stats['difference']):
                report_lines.append(f"{horizon_key} Forward Returns:")
                report_lines.append(f"  2016-2020 Mean IC: {comp_stats['period_2016_mean']:.4f}")
                report_lines.append(f"  2021-2025 Mean IC: {comp_stats['period_2021_mean']:.4f}")
                report_lines.append(f"  Difference: {comp_stats['difference']:.4f}")
                report_lines.append(f"  T-Stat: {comp_stats['t_stat_diff']:.3f}")
                report_lines.append(f"  P-Value: {comp_stats['p_value_diff']:.4f}")
                report_lines.append(f"  Significant: {'Yes' if comp_stats['significant_diff'] else 'No'}")
                report_lines.append("")
        
        report_text = "\n".join(report_lines)
        
        # Save report if path provided
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            self.logger.info(f"Report saved to: {save_path}")
        
        return report_text
    
    def create_visualizations(self, results: Dict, save_path: str = None):
        """
        Create visualizations for IC analysis results.
        
        Args:
            results: IC analysis results
            save_path: Path to save the plots
        """
        try:
            # Set up the plotting style
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Momentum Factor IC Analysis Results', fontsize=16, fontweight='bold')
            
            # Plot 1: IC Time Series
            ax1 = axes[0, 0]
            for period_name in ['2016-2020', '2021-2025']:
                period_results = results[period_name]
                dates = period_results['rebalance_dates']
                
                for horizon in [1, 3, 6, 12]:
                    ic_series = pd.Series(period_results['ic_by_horizon'][horizon], index=dates)
                    ic_series = ic_series.dropna()
                    if len(ic_series) > 0:
                        ax1.plot(ic_series.index, ic_series.values, 
                               label=f'{period_name} {horizon}M', alpha=0.7)
            
            ax1.set_title('IC Time Series by Period and Horizon')
            ax1.set_ylabel('Information Coefficient')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: IC Distribution
            ax2 = axes[0, 1]
            for period_name in ['2016-2020', '2021-2025']:
                period_results = results[period_name]
                for horizon in [1, 3, 6, 12]:
                    ic_series = pd.Series(period_results['ic_by_horizon'][horizon])
                    ic_series = ic_series.dropna()
                    if len(ic_series) > 0:
                        ax2.hist(ic_series.values, alpha=0.5, 
                               label=f'{period_name} {horizon}M', bins=20)
            
            ax2.set_title('IC Distribution by Period and Horizon')
            ax2.set_xlabel('Information Coefficient')
            ax2.set_ylabel('Frequency')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Mean IC Comparison
            ax3 = axes[1, 0]
            horizons = [1, 3, 6, 12]
            x = np.arange(len(horizons))
            width = 0.35
            
            for i, period_name in enumerate(['2016-2020', '2021-2025']):
                period_results = results[period_name]
                summary_stats = period_results['summary_stats']
                
                means = [summary_stats[f'{h}M']['mean_ic'] for h in horizons]
                stds = [summary_stats[f'{h}M']['std_ic'] for h in horizons]
                
                ax3.bar(x + i*width, means, width, 
                       label=period_name, alpha=0.8,
                       yerr=stds, capsize=5)
            
            ax3.set_title('Mean IC by Period and Horizon')
            ax3.set_xlabel('Forward Return Horizon (Months)')
            ax3.set_ylabel('Mean Information Coefficient')
            ax3.set_xticks(x + width/2)
            ax3.set_xticklabels(horizons)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Hit Rate Comparison
            ax4 = axes[1, 1]
            for i, period_name in enumerate(['2016-2020', '2021-2025']):
                period_results = results[period_name]
                summary_stats = period_results['summary_stats']
                
                hit_rates = [summary_stats[f'{h}M']['hit_rate'] for h in horizons]
                
                ax4.bar(x + i*width, hit_rates, width, 
                       label=period_name, alpha=0.8)
            
            ax4.set_title('Hit Rate by Period and Horizon')
            ax4.set_xlabel('Forward Return Horizon (Months)')
            ax4.set_ylabel('Hit Rate')
            ax4.set_xticks(x + width/2)
            ax4.set_xticklabels(horizons)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Visualizations saved to: {save_path}")
            
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Failed to create visualizations: {e}")


def main():
    """Main execution function."""
    print("🚀 Starting Momentum Factor IC Analysis")
    print("=" * 60)
    
    try:
        # Initialize analysis
        ic_analysis = MomentumICAnalysis()
        
        # Run complete analysis
        print("📊 Running IC analysis for both periods...")
        results = ic_analysis.run_complete_analysis()
        
        # Generate report
        print("📝 Generating comprehensive report...")
        report = ic_analysis.generate_report(results)
        print(report)
        
        # Create visualizations
        print("📈 Creating visualizations...")
        ic_analysis.create_visualizations(results)
        
        print("\n✅ Momentum Factor IC Analysis completed successfully!")
        
        return results
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return None


if __name__ == "__main__":
    main() 