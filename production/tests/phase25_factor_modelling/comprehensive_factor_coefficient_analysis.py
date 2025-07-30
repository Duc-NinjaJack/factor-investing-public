"""
Comprehensive Factor Coefficient Analysis
========================================
Analyzes factor coefficients for valuation ratios, momentum, quality metrics, and other factors
across different sectors and time periods.

This script provides detailed coefficient analysis for:
- Valuation ratios (PE, PB, PS, EV/EBITDA)
- Momentum factors (1M, 3M, 6M, 12M returns)
- Quality metrics (ROAA, ROAE, Operating Margin, EBITDA Margin)
- Financial ratios (Debt/Equity, Current Ratio, etc.)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from datetime import datetime, timedelta
import warnings

# Add the parent directory to the path to import database modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'database'))
from connection import get_engine

warnings.filterwarnings('ignore')

class ComprehensiveFactorCoefficientAnalyzer:
    def __init__(self):
        self.engine = get_engine()
        self.logger = self._setup_logging()
        
    def _setup_logging(self):
        import logging
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
        
    def get_comprehensive_factors(self, analysis_date: datetime, universe: list) -> pd.DataFrame:
        """Extract comprehensive factors from all sectors."""
        self.logger.info(f"Extracting comprehensive factors for {analysis_date}")
        
        # Banking factors
        banking_query = """
        SELECT ticker, 
               NetProfit_TTM / AvgTotalAssets as roaa,
               NetProfit_TTM / AvgTotalEquity as roae,
               OperatingProfit_TTM / TotalOperatingIncome_TTM as operating_margin,
               NULL as ebitda_margin,
               NULL as pe_ratio,
               NULL as pb_ratio,
               NULL as ps_ratio,
               NULL as ev_ebitda,
               NULL as debt_to_equity,
               NULL as current_ratio,
               NULL as quick_ratio,
               NULL as inventory_turnover,
               TotalOperatingIncome_TTM / AvgTotalAssets as asset_turnover,
               NULL as equity_turnover
        FROM intermediary_calculations_banking_cleaned
        WHERE quarter = (
            SELECT MAX(quarter) 
            FROM intermediary_calculations_banking_cleaned 
            WHERE quarter <= %s
        ) AND ticker IN ({})
        """.format(','.join(['%s'] * len(universe)))
        
        # Securities factors
        securities_query = """
        SELECT ticker, 
               ROAA,
               ROAE,
               OperatingMargin,
               NULL as ebitda_margin,
               NULL as pe_ratio,
               NULL as pb_ratio,
               NULL as ps_ratio,
               NULL as ev_ebitda,
               NULL as debt_to_equity,
               NULL as current_ratio,
               NULL as quick_ratio,
               NULL as inventory_turnover,
               AssetTurnover,
               NULL as equity_turnover
        FROM intermediary_calculations_securities_cleaned
        WHERE quarter = (
            SELECT MAX(quarter) 
            FROM intermediary_calculations_securities_cleaned 
            WHERE quarter <= %s
        ) AND ticker IN ({})
        """.format(','.join(['%s'] * len(universe)))
        
        # Enhanced (non-financial) factors
        enhanced_query = """
        SELECT ticker, 
               NetProfit_TTM / AvgTotalAssets as roaa,
               NetProfit_TTM / AvgTotalEquity as roae,
               (EBIT_TTM + DepreciationAmortization_TTM) / Revenue_TTM as operating_margin,
               EBITDA_TTM / Revenue_TTM as ebitda_margin,
               NULL as pe_ratio,
               NULL as pb_ratio,
               NULL as ps_ratio,
               NULL as ev_ebitda,
               AvgTotalDebt / AvgTotalEquity as debt_to_equity,
               AvgCurrentAssets / AvgCurrentLiabilities as current_ratio,
               (AvgCurrentAssets - AvgInventory) / AvgCurrentLiabilities as quick_ratio,
               Revenue_TTM / AvgInventory as inventory_turnover,
               Revenue_TTM / AvgTotalAssets as asset_turnover,
               Revenue_TTM / AvgTotalEquity as equity_turnover
        FROM intermediary_calculations_enhanced
        WHERE quarter = (
            SELECT MAX(quarter) 
            FROM intermediary_calculations_enhanced 
            WHERE quarter <= %s
        ) AND ticker IN ({})
        """.format(','.join(['%s'] * len(universe)))
        
        params = [analysis_date] + universe
        
        try:
            banking_df = pd.read_sql(banking_query, self.engine, params=tuple(params))
            securities_df = pd.read_sql(securities_query, self.engine, params=tuple(params))
            enhanced_df = pd.read_sql(enhanced_query, self.engine, params=tuple(params))
            
            # Add sector information
            banking_df['sector'] = 'Banking'
            securities_df['sector'] = 'Securities'
            enhanced_df['sector'] = 'Non-Financial'
            
            # Combine all dataframes
            fundamentals = pd.concat([banking_df, securities_df, enhanced_df], ignore_index=True)
            
        except Exception as e:
            self.logger.error(f"Error extracting fundamentals: {e}")
            return pd.DataFrame()
        
        # Get momentum data
        momentum_query = """
        SELECT ticker, 
               CASE 
                   WHEN period = '1M' THEN return_1m
                   WHEN period = '3M' THEN return_3m
                   WHEN period = '6M' THEN return_6m
                   WHEN period = '12M' THEN return_12m
               END as momentum_return
        FROM (
            SELECT ticker, '1M' as period, return_1m FROM equity_history 
            WHERE date = %s AND ticker IN ({})
            UNION ALL
            SELECT ticker, '3M' as period, return_3m FROM equity_history 
            WHERE date = %s AND ticker IN ({})
            UNION ALL
            SELECT ticker, '6M' as period, return_6m FROM equity_history 
            WHERE date = %s AND ticker IN ({})
            UNION ALL
            SELECT ticker, '12M' as period, return_12m FROM equity_history 
            WHERE date = %s AND ticker IN ({})
        ) momentum_data
        """.format(','.join(['%s'] * len(universe)), 
                   ','.join(['%s'] * len(universe)), 
                   ','.join(['%s'] * len(universe)), 
                   ','.join(['%s'] * len(universe)))
        
        momentum_params = [analysis_date] + universe + [analysis_date] + universe + [analysis_date] + universe + [analysis_date] + universe
        
        try:
            momentum = pd.read_sql(momentum_query, self.engine, params=tuple(momentum_params))
            
            # Pivot momentum data
            momentum_wide = momentum.pivot(index='ticker', columns='period', values='momentum_return').reset_index()
            momentum_wide.columns.name = None
            
            # Merge fundamentals and momentum
            factors = fundamentals.merge(momentum_wide, on='ticker', how='inner')
            
        except Exception as e:
            self.logger.error(f"Error extracting momentum: {e}")
            factors = fundamentals
        
        return factors
        
    def get_forward_returns(self, analysis_date: datetime, universe: list) -> pd.DataFrame:
        """Get 1-month forward returns."""
        forward_date = analysis_date + timedelta(days=30)
        
        query = """
        WITH current_prices AS (
            SELECT ticker, close_price as current_price
            FROM vcsc_daily_data_complete
            WHERE date = %s AND ticker IN ({})
        ),
        future_prices AS (
            SELECT ticker, close_price as future_price
            FROM vcsc_daily_data_complete
            WHERE date = %s AND ticker IN ({})
        )
        SELECT c.ticker, 
               (f.future_price - c.current_price) / c.current_price as forward_return
        FROM current_prices c
        JOIN future_prices f ON c.ticker = f.ticker
        """.format(','.join(['%s'] * len(universe)), ','.join(['%s'] * len(universe)))
        
        params = [analysis_date] + universe + [forward_date] + universe
        
        return pd.read_sql(query, self.engine, params=tuple(params))
        
    def normalize_factors(self, factors_df: pd.DataFrame) -> pd.DataFrame:
        """Normalize factors using sector-neutral z-scores."""
        self.logger.info("Normalizing factors using sector-neutral z-scores")
        
        factor_columns = [
            'roaa', 'roae', 'operating_margin', 'ebitda_margin',
            'pe_ratio', 'pb_ratio', 'ps_ratio', 'ev_ebitda',
            'debt_to_equity', 'current_ratio', 'quick_ratio',
            'inventory_turnover', 'asset_turnover', 'equity_turnover',
            '1M', '3M', '6M', '12M'
        ]
        
        normalized_factors = factors_df.copy()
        
        for factor in factor_columns:
            if factor in normalized_factors.columns:
                normalized_factors[f'{factor}_normalized'] = normalized_factors.groupby('sector')[factor].transform(
                    lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
                )
        
        return normalized_factors
        
    def run_factor_regression(self, normalized_factors: pd.DataFrame, forward_returns: pd.DataFrame) -> dict:
        """Run regression and return detailed coefficient analysis."""
        self.logger.info("Running comprehensive factor regression analysis")
        
        regression_data = normalized_factors.merge(forward_returns, on='ticker', how='inner')
        
        normalized_factor_cols = [col for col in regression_data.columns if col.endswith('_normalized')]
        
        X = regression_data[normalized_factor_cols].fillna(0)
        y = regression_data['forward_return'].fillna(0)
        
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X = X[mask]
        y = y[mask]
        
        if len(X) == 0:
            return {}
        
        # Manual linear regression
        X_with_intercept = np.column_stack([np.ones(len(X)), X])
        coefficients = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
        
        y_pred = X_with_intercept @ coefficients
        r_squared = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
        
        # Calculate t-statistics and p-values
        residuals = y - y_pred
        mse = np.sum(residuals ** 2) / (len(X) - len(coefficients))
        var_b = mse * np.linalg.inv(np.dot(X_with_intercept.T, X_with_intercept)).diagonal()
        sd_b = np.sqrt(var_b)
        t_stats = coefficients / sd_b
        
        # Calculate p-values (approximate)
        from scipy import stats
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), len(X) - len(coefficients)))
        
        results = {
            'coefficients': dict(zip(normalized_factor_cols, coefficients[1:])),
            'intercept': coefficients[0],
            't_statistics': dict(zip(normalized_factor_cols, t_stats[1:])),
            'p_values': dict(zip(normalized_factor_cols, p_values[1:])),
            'r_squared': r_squared,
            'n_observations': len(X),
            'factor_importance': dict(zip(normalized_factor_cols, np.abs(coefficients[1:])))
        }
        
        return results
        
    def categorize_factors(self, results: dict) -> dict:
        """Categorize factors by type and analyze coefficients."""
        factor_categories = {
            'Valuation Ratios': ['pe_ratio_normalized', 'pb_ratio_normalized', 'ps_ratio_normalized', 'ev_ebitda_normalized'],
            'Momentum Factors': ['1M_normalized', '3M_normalized', '6M_normalized', '12M_normalized'],
            'Quality Metrics': ['roaa_normalized', 'roae_normalized', 'operating_margin_normalized', 'ebitda_margin_normalized'],
            'Financial Ratios': ['debt_to_equity_normalized', 'current_ratio_normalized', 'quick_ratio_normalized'],
            'Efficiency Ratios': ['inventory_turnover_normalized', 'asset_turnover_normalized', 'equity_turnover_normalized']
        }
        
        categorized_results = {}
        
        for category, factors in factor_categories.items():
            category_data = {}
            for factor in factors:
                if factor in results['coefficients']:
                    category_data[factor.replace('_normalized', '')] = {
                        'coefficient': results['coefficients'][factor],
                        't_statistic': results['t_statistics'][factor],
                        'p_value': results['p_values'][factor],
                        'importance': results['factor_importance'][factor]
                    }
            categorized_results[category] = category_data
        
        return categorized_results
        
    def print_coefficient_analysis(self, categorized_results: dict, overall_results: dict):
        """Print detailed coefficient analysis."""
        print("\n" + "="*100)
        print("COMPREHENSIVE FACTOR COEFFICIENT ANALYSIS")
        print("="*100)
        
        print(f"\nOverall Model Performance:")
        print(f"R-squared: {overall_results['r_squared']:.4f}")
        print(f"Number of observations: {overall_results['n_observations']}")
        print(f"Intercept: {overall_results['intercept']:.4f}")
        
        print(f"\n" + "="*100)
        print("DETAILED FACTOR COEFFICIENT ANALYSIS")
        print("="*100)
        
        for category, factors in categorized_results.items():
            if factors:
                print(f"\n{category.upper()}:")
                print("-" * 50)
                
                # Sort by absolute coefficient value
                sorted_factors = sorted(factors.items(), key=lambda x: abs(x[1]['coefficient']), reverse=True)
                
                for factor_name, stats in sorted_factors:
                    significance = "***" if stats['p_value'] < 0.01 else "**" if stats['p_value'] < 0.05 else "*" if stats['p_value'] < 0.1 else ""
                    print(f"{factor_name:20} | Coef: {stats['coefficient']:8.4f} | t-stat: {stats['t_statistic']:6.2f} | p-val: {stats['p_value']:6.3f} {significance}")
        
        # Top factors by importance
        print(f"\n" + "="*100)
        print("TOP FACTORS BY IMPORTANCE (Absolute Coefficient)")
        print("="*100)
        
        all_factors = []
        for category, factors in categorized_results.items():
            for factor_name, stats in factors.items():
                all_factors.append((factor_name, stats['coefficient'], stats['t_statistic'], stats['p_value'], category))
        
        # Sort by absolute coefficient
        all_factors.sort(key=lambda x: abs(x[1]), reverse=True)
        
        for i, (factor_name, coef, t_stat, p_val, category) in enumerate(all_factors[:10]):
            significance = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
            print(f"{i+1:2}. {factor_name:20} | Coef: {coef:8.4f} | t-stat: {t_stat:6.2f} | p-val: {p_val:6.3f} | {category:15} {significance}")
        
        print("\n" + "="*100)
        
    def create_coefficient_visualizations(self, categorized_results: dict, overall_results: dict):
        """Create comprehensive coefficient visualizations."""
        self.logger.info("Creating coefficient visualizations")
        
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Comprehensive Factor Coefficient Analysis', fontsize=16, fontweight='bold')
        
        # 1. Coefficient values by category
        ax1 = axes[0, 0]
        categories = []
        coefficients = []
        colors = []
        
        for category, factors in categorized_results.items():
            if factors:
                for factor_name, stats in factors.items():
                    categories.append(f"{category}\n{factor_name}")
                    coefficients.append(stats['coefficient'])
                    colors.append('red' if stats['coefficient'] < 0 else 'blue')
        
        bars = ax1.barh(range(len(coefficients)), coefficients, color=colors, alpha=0.7)
        ax1.set_yticks(range(len(categories)))
        ax1.set_yticklabels(categories, fontsize=8)
        ax1.set_xlabel('Coefficient Value')
        ax1.set_title('Factor Coefficients by Category')
        ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        # 2. T-statistics heatmap
        ax2 = axes[0, 1]
        t_stat_data = []
        factor_names = []
        
        for category, factors in categorized_results.items():
            if factors:
                for factor_name, stats in factors.items():
                    t_stat_data.append(stats['t_statistic'])
                    factor_names.append(f"{category}: {factor_name}")
        
        if t_stat_data:
            t_stat_matrix = np.array(t_stat_data).reshape(-1, 1)
            im = ax2.imshow(t_stat_matrix, cmap='RdYlBu_r', aspect='auto')
            ax2.set_yticks(range(len(factor_names)))
            ax2.set_yticklabels(factor_names, fontsize=8)
            ax2.set_title('T-Statistics Heatmap')
            plt.colorbar(im, ax=ax2, label='T-statistic')
        
        # 3. P-values by category
        ax3 = axes[0, 2]
        categories_p = []
        p_values = []
        
        for category, factors in categorized_results.items():
            if factors:
                for factor_name, stats in factors.items():
                    categories_p.append(f"{category}\n{factor_name}")
                    p_values.append(stats['p_value'])
        
        bars = ax3.barh(range(len(p_values)), p_values, alpha=0.7)
        ax3.set_yticks(range(len(categories_p)))
        ax3.set_yticklabels(categories_p, fontsize=8)
        ax3.set_xlabel('P-value')
        ax3.set_title('P-values by Factor')
        ax3.axvline(x=0.05, color='red', linestyle='--', alpha=0.7, label='5% significance')
        ax3.axvline(x=0.01, color='darkred', linestyle='--', alpha=0.7, label='1% significance')
        ax3.legend()
        
        # 4. Factor importance by category
        ax4 = axes[1, 0]
        importance_data = []
        importance_names = []
        
        for category, factors in categorized_results.items():
            if factors:
                for factor_name, stats in factors.items():
                    importance_data.append(stats['importance'])
                    importance_names.append(f"{category}\n{factor_name}")
        
        bars = ax4.barh(range(len(importance_data)), importance_data, alpha=0.7)
        ax4.set_yticks(range(len(importance_names)))
        ax4.set_yticklabels(importance_names, fontsize=8)
        ax4.set_xlabel('Absolute Coefficient Value')
        ax4.set_title('Factor Importance by Category')
        
        # 5. Category summary
        ax5 = axes[1, 1]
        category_summary = {}
        
        for category, factors in categorized_results.items():
            if factors:
                avg_coef = np.mean([abs(stats['coefficient']) for stats in factors.values()])
                significant_count = sum(1 for stats in factors.values() if stats['p_value'] < 0.05)
                category_summary[category] = {'avg_coef': avg_coef, 'significant': significant_count}
        
        if category_summary:
            categories_summary = list(category_summary.keys())
            avg_coefs = [category_summary[cat]['avg_coef'] for cat in categories_summary]
            significant_counts = [category_summary[cat]['significant'] for cat in categories_summary]
            
            x = np.arange(len(categories_summary))
            width = 0.35
            
            bars1 = ax5.bar(x - width/2, avg_coefs, width, label='Avg |Coefficient|', alpha=0.7)
            ax5_twin = ax5.twinx()
            bars2 = ax5_twin.bar(x + width/2, significant_counts, width, label='Significant Factors', alpha=0.7, color='orange')
            
            ax5.set_xlabel('Factor Categories')
            ax5.set_ylabel('Average |Coefficient|')
            ax5_twin.set_ylabel('Number of Significant Factors')
            ax5.set_title('Category Summary')
            ax5.set_xticks(x)
            ax5.set_xticklabels(categories_summary, rotation=45, ha='right')
            
            ax5.legend(loc='upper left')
            ax5_twin.legend(loc='upper right')
        
        # 6. Model performance
        ax6 = axes[1, 2]
        performance_metrics = ['R-squared', 'Observations']
        performance_values = [overall_results['r_squared'], overall_results['n_observations']]
        
        bars = ax6.bar(performance_metrics, performance_values, alpha=0.7)
        ax6.set_title('Model Performance')
        ax6.set_ylabel('Value')
        
        # Add value labels on bars
        for bar, value in zip(bars, performance_values):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.3f}' if value < 1 else f'{value:.0f}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('comprehensive_factor_coefficient_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def run_comprehensive_analysis(self, analysis_date: datetime = None):
        """Run comprehensive factor coefficient analysis."""
        if analysis_date is None:
            analysis_date = datetime(2024, 12, 31)
        
        self.logger.info(f"Running comprehensive analysis for {analysis_date}")
        
        # Get universe
        universe_query = """
        SELECT DISTINCT ticker FROM (
            SELECT ticker FROM intermediary_calculations_banking_cleaned
            UNION ALL
            SELECT ticker FROM intermediary_calculations_securities_cleaned
            UNION ALL
            SELECT ticker FROM intermediary_calculations_enhanced
        ) combined
        LIMIT 50
        """
        
        universe_df = pd.read_sql(universe_query, self.engine)
        universe = universe_df['ticker'].tolist()
        
        self.logger.info(f"Universe size: {len(universe)} tickers")
        
        # Extract factors
        factors = self.get_comprehensive_factors(analysis_date, universe)
        
        if len(factors) == 0:
            self.logger.error("No factors extracted")
            return
        
        # Normalize factors
        normalized_factors = self.normalize_factors(factors)
        
        # Get forward returns
        forward_returns = self.get_forward_returns(analysis_date, universe)
        
        if len(forward_returns) == 0:
            self.logger.error("No forward returns available")
            return
        
        # Run regression
        results = self.run_factor_regression(normalized_factors, forward_returns)
        
        if not results:
            self.logger.error("Regression failed")
            return
        
        # Categorize factors
        categorized_results = self.categorize_factors(results)
        
        # Print analysis
        self.print_coefficient_analysis(categorized_results, results)
        
        # Create visualizations
        self.create_coefficient_visualizations(categorized_results, results)
        
        return {
            'results': results,
            'categorized_results': categorized_results,
            'factors': factors,
            'normalized_factors': normalized_factors,
            'forward_returns': forward_returns
        }


def main():
    """Main execution function."""
    analyzer = ComprehensiveFactorCoefficientAnalyzer()
    
    # Run analysis for recent date
    analysis_date = datetime(2024, 12, 31)
    
    results = analyzer.run_comprehensive_analysis(analysis_date)
    
    if results:
        print("\nAnalysis completed successfully!")
        print("Results saved to: comprehensive_factor_coefficient_analysis.png")
    else:
        print("Analysis failed")


if __name__ == "__main__":
    main() 