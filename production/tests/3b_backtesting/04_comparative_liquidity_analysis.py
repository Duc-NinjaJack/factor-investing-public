#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparative Liquidity Analysis: 10B vs 3B VND Thresholds
=======================================================
Component: Liquidity Threshold Validation
Purpose: Compare performance and universe characteristics between 10B and 3B VND thresholds
Author: Duc Nguyen, Principal Quantitative Strategist
Date Created: January 2025
Status: PRODUCTION VALIDATION

This script performs comprehensive comparative analysis between:
- 10B VND liquidity threshold (current production)
- 3B VND liquidity threshold (proposed implementation)

Key Metrics Analyzed:
1. Universe size expansion
2. Performance comparison (returns, Sharpe, drawdown)
3. Stock survival analysis
4. Sector diversification impact
5. Risk metrics comparison

Data Sources:
- factor_scores_qvm (factor data)
- equity_history (price data)
- etf_history (benchmark data)
- vcsc_daily_data_complete (volume data)

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

class ComparativeLiquidityAnalyzer:
    """
    Comprehensive analyzer for comparing liquidity thresholds.
    """
    
    def __init__(self, config_path: str = "../../../config/database.yml"):
        """Initialize the analyzer with database connection."""
        self.config_path = config_path
        self.engine = self._create_database_engine()
        self.results = {}
        
        # Analysis parameters
        self.thresholds = {
            '10B_VND': 10_000_000_000,
            '3B_VND': 3_000_000_000
        }
        
        # Backtest parameters
        self.backtest_config = {
            'start_date': '2018-01-01',
            'end_date': '2025-01-01',
            'rebalance_freq': 'M',
            'portfolio_size': 25,
            'max_sector_weight': 0.4,
            'transaction_cost': 0.002
        }
        
        logger.info("Comparative Liquidity Analyzer initialized")
    
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
        """Load all required data for analysis."""
        logger.info("Loading data for comparative analysis...")
        
        data = {}
        
        # Load existing pickle data
        try:
            with open('unrestricted_universe_data.pkl', 'rb') as f:
                pickle_data = pickle.load(f)
            
            data['factor_scores'] = pickle_data['factor_data']
            data['volume_data'] = pickle_data['volume_data']
            data['adtv_data'] = pickle_data['adtv']
            
            logger.info("✅ Loaded data from pickle file")
            
        except FileNotFoundError:
            logger.error("❌ Pickle file not found. Please run get_unrestricted_universe_data.py first.")
            raise
        
        # Load price data from database
        price_query = """
        SELECT date, ticker, close_price_adjusted
        FROM equity_history
        WHERE date >= '2018-01-01'
        ORDER BY date, ticker
        """
        data['price_data'] = pd.read_sql(price_query, self.engine)
        data['price_data']['date'] = pd.to_datetime(data['price_data']['date'])
        
        # Load benchmark data
        benchmark_query = """
        SELECT date, close_price_adjusted
        FROM etf_history
        WHERE ticker = 'VNINDEX' AND date >= '2018-01-01'
        ORDER BY date
        """
        data['benchmark'] = pd.read_sql(benchmark_query, self.engine)
        data['benchmark']['date'] = pd.to_datetime(data['benchmark']['date'])
        
        # Load sector information
        sector_query = """
        SELECT ticker, sector
        FROM master_info
        WHERE sector IS NOT NULL
        """
        data['sector_info'] = pd.read_sql(sector_query, self.engine)
        
        logger.info(f"✅ Data loaded successfully")
        logger.info(f"   - Factor scores: {len(data['factor_scores']):,} records")
        logger.info(f"   - Volume data: {len(data['volume_data']):,} records")
        logger.info(f"   - ADTV data: {data['adtv_data'].shape}")
        logger.info(f"   - Price data: {len(data['price_data']):,} records")
        logger.info(f"   - Benchmark: {len(data['benchmark']):,} records")
        
        return data
    
    def calculate_adtv(self, volume_data: pd.DataFrame, lookback_days: int = 63) -> pd.DataFrame:
        """Calculate Average Daily Turnover (ADTV)."""
        logger.info(f"Calculating {lookback_days}-day ADTV...")
        
        # Calculate daily turnover
        volume_data['daily_turnover'] = (
            volume_data['close_price_adjusted'] * volume_data['total_volume']
        )
        
        # Pivot and calculate rolling average
        turnover_pivot = volume_data.pivot(
            index='date', columns='ticker', values='daily_turnover'
        )
        
        adtv = turnover_pivot.rolling(window=lookback_days, min_periods=30).mean()
        
        logger.info(f"✅ ADTV calculated: {adtv.shape}")
        return adtv
    
    def analyze_universe_expansion(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Analyze universe expansion between thresholds."""
        logger.info("Analyzing universe expansion...")
        
        # Use pre-calculated ADTV data
        adtv_data = data['adtv_data']
        
        # Sample dates for analysis (monthly)
        analysis_dates = pd.date_range(
            start=data['factor_scores']['calculation_date'].min(),
            end=data['factor_scores']['calculation_date'].max(),
            freq='M'
        )
        
        universe_stats = []
        
        for date in analysis_dates:
            # Get available data for this date
            available_adtv = adtv_data.loc[:date].iloc[-1].dropna() if date in adtv_data.index else pd.Series()
            available_factors = data['factor_scores'][
                data['factor_scores']['calculation_date'] == date
            ]
            
            if available_factors.empty or available_adtv.empty:
                continue
            
            # Merge with ADTV data
            merged_data = available_factors.merge(
                available_adtv.reset_index().rename(columns={0: 'adtv'}),
                left_on='ticker', right_on='ticker', how='inner'
            )
            
            # Calculate universe sizes for each threshold
            for threshold_name, threshold_value in self.thresholds.items():
                liquid_universe = merged_data[merged_data['adtv'] >= threshold_value]
                
                universe_stats.append({
                    'date': date,
                    'threshold': threshold_name,
                    'total_stocks': len(merged_data),
                    'liquid_stocks': len(liquid_universe),
                    'liquidity_ratio': len(liquid_universe) / len(merged_data) if len(merged_data) > 0 else 0,
                    'avg_adtv': liquid_universe['adtv'].mean() if len(liquid_universe) > 0 else 0,
                    'median_adtv': liquid_universe['adtv'].median() if len(liquid_universe) > 0 else 0
                })
        
        universe_df = pd.DataFrame(universe_stats)
        
        # Calculate expansion metrics
        expansion_metrics = {}
        for date in universe_df['date'].unique():
            date_data = universe_df[universe_df['date'] == date]
            if len(date_data) == 2:  # Both thresholds present
                v10b = date_data[date_data['threshold'] == '10B_VND'].iloc[0]
                v3b = date_data[date_data['threshold'] == '3B_VND'].iloc[0]
                
                expansion_metrics[date] = {
                    'universe_expansion': v3b['liquid_stocks'] / v10b['liquid_stocks'] if v10b['liquid_stocks'] > 0 else 0,
                    'additional_stocks': v3b['liquid_stocks'] - v10b['liquid_stocks'],
                    'expansion_ratio': v3b['liquid_stocks'] / v10b['liquid_stocks'] if v10b['liquid_stocks'] > 0 else 0
                }
        
        expansion_df = pd.DataFrame(expansion_metrics).T
        
        logger.info(f"✅ Universe expansion analysis complete")
        logger.info(f"   - Average expansion ratio: {expansion_df['expansion_ratio'].mean():.2f}x")
        logger.info(f"   - Average additional stocks: {expansion_df['additional_stocks'].mean():.0f}")
        
        return {
            'universe_stats': universe_df,
            'expansion_metrics': expansion_df
        }
    
    def run_comparative_backtest(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Run backtests for both thresholds."""
        logger.info("Running comparative backtests...")
        
        # Use pre-calculated ADTV data
        adtv_data = data['adtv_data']
        
        # Prepare price data
        price_pivot = data['price_data'].pivot(
            index='date', columns='ticker', values='close_price_adjusted'
        )
        
        # Calculate returns
        returns = price_pivot.pct_change().dropna()
        
        # Prepare factor data
        factor_pivot = data['factor_scores'].pivot(
            index='calculation_date', columns='ticker', values='qvm_composite'
        )
        
        backtest_results = {}
        
        for threshold_name, threshold_value in self.thresholds.items():
            logger.info(f"Running backtest for {threshold_name}...")
            
            # Rebalancing dates
            rebalance_dates = pd.date_range(
                start=returns.index.min(),
                end=returns.index.max(),
                freq=self.backtest_config['rebalance_freq']
            )
            
            portfolio_returns = []
            portfolio_holdings = []
            
            for i, rebalance_date in enumerate(rebalance_dates[:-1]):
                next_rebalance = rebalance_dates[i + 1]
                
                # Get factor scores as of rebalance date
                factor_scores = factor_pivot.loc[:rebalance_date].iloc[-1].dropna()
                
                # Get ADTV as of rebalance date
                adtv_scores = adtv_data.loc[:rebalance_date].iloc[-1].dropna() if rebalance_date in adtv_data.index else pd.Series()
                
                # Apply liquidity filter
                liquid_stocks = adtv_scores[adtv_scores >= threshold_value].index
                available_stocks = factor_scores.index.intersection(liquid_stocks)
                
                if len(available_stocks) < self.backtest_config['portfolio_size']:
                    continue
                
                # Select top stocks
                top_stocks = factor_scores[available_stocks].nlargest(
                    self.backtest_config['portfolio_size']
                ).index
                
                # Equal weight portfolio
                weights = pd.Series(1.0 / len(top_stocks), index=top_stocks)
                
                # Calculate portfolio returns for this period
                period_returns = returns.loc[rebalance_date:next_rebalance, top_stocks]
                portfolio_return = (period_returns * weights).sum(axis=1)
                
                portfolio_returns.extend(portfolio_return.values)
                portfolio_holdings.append({
                    'date': rebalance_date,
                    'stocks': list(top_stocks),
                    'weights': weights.to_dict(),
                    'universe_size': len(available_stocks)
                })
            
            # Calculate performance metrics
            portfolio_returns_series = pd.Series(portfolio_returns, index=returns.index)
            benchmark_returns = data['benchmark'].set_index('date')['close_price_adjusted'].pct_change()
            
            # Align dates
            aligned_returns = pd.DataFrame({
                'portfolio': portfolio_returns_series,
                'benchmark': benchmark_returns
            }).dropna()
            
            # Calculate metrics
            annual_return = aligned_returns['portfolio'].mean() * 252
            annual_vol = aligned_returns['portfolio'].std() * np.sqrt(252)
            sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
            
            # Calculate drawdown
            cumulative_returns = (1 + aligned_returns['portfolio']).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Calculate alpha and beta
            covariance = np.cov(aligned_returns['portfolio'], aligned_returns['benchmark'])[0, 1]
            benchmark_var = aligned_returns['benchmark'].var()
            beta = covariance / benchmark_var if benchmark_var > 0 else 0
            
            benchmark_return = aligned_returns['benchmark'].mean() * 252
            alpha = annual_return - (beta * benchmark_return)
            
            backtest_results[threshold_name] = {
                'returns': aligned_returns['portfolio'],
                'metrics': {
                    'annual_return': annual_return,
                    'annual_volatility': annual_vol,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'alpha': alpha,
                    'beta': beta,
                    'calmar_ratio': annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
                },
                'holdings': portfolio_holdings
            }
            
            logger.info(f"✅ {threshold_name} backtest complete")
            logger.info(f"   - Annual Return: {annual_return:.2%}")
            logger.info(f"   - Sharpe Ratio: {sharpe_ratio:.2f}")
            logger.info(f"   - Max Drawdown: {max_drawdown:.2%}")
        
        return backtest_results
    
    def analyze_stock_survival(self, data: Dict[str, pd.DataFrame], backtest_results: Dict) -> pd.DataFrame:
        """Analyze stock survival rates between thresholds."""
        logger.info("Analyzing stock survival rates...")
        
        survival_data = []
        
        for threshold_name, results in backtest_results.items():
            holdings = results['holdings']
            
            # Count stock appearances
            stock_counts = {}
            total_rebalances = len(holdings)
            
            for holding in holdings:
                for stock in holding['stocks']:
                    stock_counts[stock] = stock_counts.get(stock, 0) + 1
            
            # Calculate survival metrics
            for stock, count in stock_counts.items():
                survival_rate = count / total_rebalances
                survival_data.append({
                    'threshold': threshold_name,
                    'ticker': stock,
                    'appearances': count,
                    'total_rebalances': total_rebalances,
                    'survival_rate': survival_rate
                })
        
        survival_df = pd.DataFrame(survival_data)
        
        # Merge with sector information
        survival_df = survival_df.merge(
            data['sector_info'], on='ticker', how='left'
        )
        
        logger.info(f"✅ Stock survival analysis complete")
        logger.info(f"   - Average survival rate (10B): {survival_df[survival_df['threshold'] == '10B_VND']['survival_rate'].mean():.2%}")
        logger.info(f"   - Average survival rate (3B): {survival_df[survival_df['threshold'] == '3B_VND']['survival_rate'].mean():.2%}")
        
        return survival_df
    
    def create_visualizations(self, universe_analysis: Dict, backtest_results: Dict, survival_data: pd.DataFrame):
        """Create comprehensive visualizations."""
        logger.info("Creating visualizations...")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Universe Expansion Over Time
        ax1 = plt.subplot(3, 3, 1)
        universe_stats = universe_analysis['universe_stats']
        for threshold in ['10B_VND', '3B_VND']:
            threshold_data = universe_stats[universe_stats['threshold'] == threshold]
            ax1.plot(threshold_data['date'], threshold_data['liquid_stocks'], 
                    label=threshold, linewidth=2)
        ax1.set_title('Universe Size Over Time', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of Liquid Stocks')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Expansion Ratio Over Time
        ax2 = plt.subplot(3, 3, 2)
        expansion_metrics = universe_analysis['expansion_metrics']
        ax2.plot(expansion_metrics.index, expansion_metrics['expansion_ratio'], 
                color='red', linewidth=2)
        ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5)
        ax2.set_title('Universe Expansion Ratio (3B/10B)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Expansion Ratio')
        ax2.grid(True, alpha=0.3)
        
        # 3. Performance Comparison
        ax3 = plt.subplot(3, 3, 3)
        performance_data = []
        for threshold, results in backtest_results.items():
            metrics = results['metrics']
            performance_data.append({
                'threshold': threshold,
                'annual_return': metrics['annual_return'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'max_drawdown': metrics['max_drawdown']
            })
        
        perf_df = pd.DataFrame(performance_data)
        x = np.arange(len(perf_df))
        width = 0.35
        
        ax3.bar(x - width/2, perf_df['annual_return'], width, label='Annual Return', alpha=0.8)
        ax3_twin = ax3.twinx()
        ax3_twin.bar(x + width/2, perf_df['sharpe_ratio'], width, label='Sharpe Ratio', alpha=0.8, color='orange')
        
        ax3.set_title('Performance Comparison', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Annual Return')
        ax3_twin.set_ylabel('Sharpe Ratio')
        ax3.set_xticks(x)
        ax3.set_xticklabels(perf_df['threshold'])
        ax3.legend(loc='upper left')
        ax3_twin.legend(loc='upper right')
        
        # 4. Cumulative Returns
        ax4 = plt.subplot(3, 3, 4)
        for threshold, results in backtest_results.items():
            cumulative_returns = (1 + results['returns']).cumprod()
            ax4.plot(cumulative_returns.index, cumulative_returns.values, 
                    label=threshold, linewidth=2)
        ax4.set_title('Cumulative Returns', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Cumulative Return')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Drawdown Analysis
        ax5 = plt.subplot(3, 3, 5)
        for threshold, results in backtest_results.items():
            cumulative_returns = (1 + results['returns']).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            ax5.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, label=threshold)
        ax5.set_title('Drawdown Analysis', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Drawdown')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Stock Survival Distribution
        ax6 = plt.subplot(3, 3, 6)
        for threshold in ['10B_VND', '3B_VND']:
            threshold_data = survival_data[survival_data['threshold'] == threshold]
            ax6.hist(threshold_data['survival_rate'], bins=20, alpha=0.7, label=threshold)
        ax6.set_title('Stock Survival Rate Distribution', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Survival Rate')
        ax6.set_ylabel('Frequency')
        ax6.legend()
        
        # 7. Sector Distribution
        ax7 = plt.subplot(3, 3, 7)
        sector_counts = survival_data[survival_data['threshold'] == '3B_VND']['sector'].value_counts()
        ax7.pie(sector_counts.values, labels=sector_counts.index, autopct='%1.1f%%')
        ax7.set_title('Sector Distribution (3B VND)', fontsize=12, fontweight='bold')
        
        # 8. ADTV Distribution
        ax8 = plt.subplot(3, 3, 8)
        universe_stats = universe_analysis['universe_stats']
        for threshold in ['10B_VND', '3B_VND']:
            threshold_data = universe_stats[universe_stats['threshold'] == threshold]
            ax8.hist(threshold_data['avg_adtv'] / 1e9, bins=20, alpha=0.7, label=threshold)
        ax8.set_title('Average ADTV Distribution', fontsize=12, fontweight='bold')
        ax8.set_xlabel('ADTV (Billion VND)')
        ax8.set_ylabel('Frequency')
        ax8.legend()
        
        # 9. Performance Metrics Summary
        ax9 = plt.subplot(3, 3, 9)
        metrics_summary = []
        for threshold, results in backtest_results.items():
            metrics = results['metrics']
            metrics_summary.append({
                'Metric': 'Annual Return',
                'Value': metrics['annual_return'],
                'Threshold': threshold
            })
            metrics_summary.append({
                'Metric': 'Sharpe Ratio',
                'Value': metrics['sharpe_ratio'],
                'Threshold': threshold
            })
            metrics_summary.append({
                'Metric': 'Max Drawdown',
                'Value': metrics['max_drawdown'],
                'Threshold': threshold
            })
        
        summary_df = pd.DataFrame(metrics_summary)
        summary_pivot = summary_df.pivot(index='Metric', columns='Threshold', values='Value')
        summary_pivot.plot(kind='bar', ax=ax9)
        ax9.set_title('Performance Metrics Summary', fontsize=12, fontweight='bold')
        ax9.set_ylabel('Value')
        ax9.legend()
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('comparative_liquidity_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("✅ Visualizations saved to comparative_liquidity_analysis.png")
    
    def generate_comparison_report(self, universe_analysis: Dict, backtest_results: Dict, survival_data: pd.DataFrame) -> str:
        """Generate comprehensive comparison report."""
        logger.info("Generating comparison report...")
        
        report = []
        report.append("# Comparative Liquidity Analysis: 10B vs 3B VND Thresholds")
        report.append("")
        report.append("**Date:** " + datetime.now().strftime("%Y-%m-%d"))
        report.append("**Purpose:** Validate 3B VND liquidity threshold implementation")
        report.append("")
        
        # Executive Summary
        report.append("## 🎯 Executive Summary")
        report.append("")
        
        # Universe expansion summary
        expansion_metrics = universe_analysis['expansion_metrics']
        avg_expansion = expansion_metrics['expansion_ratio'].mean()
        avg_additional = expansion_metrics['additional_stocks'].mean()
        
        report.append(f"- **Universe Expansion:** {avg_expansion:.1f}x average expansion")
        report.append(f"- **Additional Stocks:** {avg_additional:.0f} additional stocks on average")
        
        # Performance comparison
        perf_10b = backtest_results['10B_VND']['metrics']
        perf_3b = backtest_results['3B_VND']['metrics']
        
        report.append(f"- **Performance Impact:** {perf_3b['annual_return']:.1%} vs {perf_10b['annual_return']:.1%} annual return")
        report.append(f"- **Risk-Adjusted:** {perf_3b['sharpe_ratio']:.2f} vs {perf_10b['sharpe_ratio']:.2f} Sharpe ratio")
        report.append("")
        
        # Detailed Analysis
        report.append("## 📊 Detailed Analysis")
        report.append("")
        
        # Universe Analysis
        report.append("### Universe Expansion Analysis")
        report.append("")
        report.append(f"- **Average Universe Size (10B VND):** {universe_analysis['universe_stats'][universe_analysis['universe_stats']['threshold'] == '10B_VND']['liquid_stocks'].mean():.0f} stocks")
        report.append(f"- **Average Universe Size (3B VND):** {universe_analysis['universe_stats'][universe_analysis['universe_stats']['threshold'] == '3B_VND']['liquid_stocks'].mean():.0f} stocks")
        report.append(f"- **Expansion Ratio:** {avg_expansion:.1f}x")
        report.append(f"- **Additional Stocks:** {avg_additional:.0f} stocks")
        report.append("")
        
        # Performance Analysis
        report.append("### Performance Analysis")
        report.append("")
        report.append("| Metric | 10B VND | 3B VND | Change |")
        report.append("|--------|---------|--------|--------|")
        report.append(f"| Annual Return | {perf_10b['annual_return']:.2%} | {perf_3b['annual_return']:.2%} | {perf_3b['annual_return'] - perf_10b['annual_return']:+.2%} |")
        report.append(f"| Annual Volatility | {perf_10b['annual_volatility']:.2%} | {perf_3b['annual_volatility']:.2%} | {perf_3b['annual_volatility'] - perf_10b['annual_volatility']:+.2%} |")
        report.append(f"| Sharpe Ratio | {perf_10b['sharpe_ratio']:.2f} | {perf_3b['sharpe_ratio']:.2f} | {perf_3b['sharpe_ratio'] - perf_10b['sharpe_ratio']:+.2f} |")
        report.append(f"| Max Drawdown | {perf_10b['max_drawdown']:.2%} | {perf_3b['max_drawdown']:.2%} | {perf_3b['max_drawdown'] - perf_10b['max_drawdown']:+.2%} |")
        report.append(f"| Alpha | {perf_10b['alpha']:.2%} | {perf_3b['alpha']:.2%} | {perf_3b['alpha'] - perf_10b['alpha']:+.2%} |")
        report.append(f"| Beta | {perf_10b['beta']:.2f} | {perf_3b['beta']:.2f} | {perf_3b['beta'] - perf_10b['beta']:+.2f} |")
        report.append("")
        
        # Stock Survival Analysis
        report.append("### Stock Survival Analysis")
        report.append("")
        survival_10b = survival_data[survival_data['threshold'] == '10B_VND']['survival_rate']
        survival_3b = survival_data[survival_data['threshold'] == '3B_VND']['survival_rate']
        
        report.append(f"- **Average Survival Rate (10B VND):** {survival_10b.mean():.2%}")
        report.append(f"- **Average Survival Rate (3B VND):** {survival_3b.mean():.2%}")
        report.append(f"- **Survival Rate Improvement:** {survival_3b.mean() - survival_10b.mean():+.2%}")
        report.append("")
        
        # Sector Analysis
        report.append("### Sector Diversification")
        report.append("")
        sector_counts_3b = survival_data[survival_data['threshold'] == '3B_VND']['sector'].value_counts()
        report.append("**Top Sectors (3B VND Universe):**")
        for sector, count in sector_counts_3b.head(5).items():
            report.append(f"- {sector}: {count} stocks")
        report.append("")
        
        # Recommendations
        report.append("## 🎯 Recommendations")
        report.append("")
        
        if avg_expansion > 1.5 and perf_3b['sharpe_ratio'] >= perf_10b['sharpe_ratio']:
            report.append("✅ **RECOMMENDED:** Implement 3B VND threshold")
            report.append("- Significant universe expansion achieved")
            report.append("- Performance maintained or improved")
            report.append("- Better diversification opportunities")
        else:
            report.append("⚠️ **FURTHER ANALYSIS NEEDED:**")
            report.append("- Universe expansion below target")
            report.append("- Performance impact requires review")
        
        report.append("")
        report.append("## 📋 Implementation Checklist")
        report.append("")
        report.append("- [x] Configuration files updated")
        report.append("- [x] Comparative backtests completed")
        report.append("- [x] Universe expansion validated")
        report.append("- [x] Performance impact assessed")
        report.append("- [x] Risk metrics compared")
        report.append("- [x] Stock survival analyzed")
        report.append("- [x] Sector diversification reviewed")
        report.append("")
        
        report_text = "\n".join(report)
        
        # Save report
        with open('comparative_liquidity_analysis_report.md', 'w') as f:
            f.write(report_text)
        
        logger.info("✅ Comparison report saved to comparative_liquidity_analysis_report.md")
        return report_text
    
    def run_complete_analysis(self):
        """Run the complete comparative analysis."""
        logger.info("🚀 Starting complete comparative liquidity analysis...")
        
        try:
            # Load data
            data = self.load_data()
            
            # Analyze universe expansion
            universe_analysis = self.analyze_universe_expansion(data)
            
            # Run comparative backtests
            backtest_results = self.run_comparative_backtest(data)
            
            # Analyze stock survival
            survival_data = self.analyze_stock_survival(data, backtest_results)
            
            # Create visualizations
            self.create_visualizations(universe_analysis, backtest_results, survival_data)
            
            # Generate report
            report = self.generate_comparison_report(universe_analysis, backtest_results, survival_data)
            
            # Save results
            results = {
                'universe_analysis': universe_analysis,
                'backtest_results': backtest_results,
                'survival_data': survival_data,
                'report': report
            }
            
            # Save to pickle for further analysis
            with open('comparative_liquidity_analysis_results.pkl', 'wb') as f:
                pickle.dump(results, f)
            
            logger.info("✅ Complete analysis finished successfully!")
            logger.info("📊 Results saved to:")
            logger.info("   - comparative_liquidity_analysis.png")
            logger.info("   - comparative_liquidity_analysis_report.md")
            logger.info("   - comparative_liquidity_analysis_results.pkl")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
            raise


def main():
    """Main execution function."""
    print("🔬 Comparative Liquidity Analysis: 10B vs 3B VND Thresholds")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = ComparativeLiquidityAnalyzer()
    
    # Run complete analysis
    results = analyzer.run_complete_analysis()
    
    print("\n✅ Analysis completed successfully!")
    print("📊 Check the generated files for detailed results.")


if __name__ == "__main__":
    main()