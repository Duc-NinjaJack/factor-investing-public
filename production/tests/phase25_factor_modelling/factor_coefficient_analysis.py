"""
Factor Coefficient Analysis
==========================
Analyzes factor coefficients for valuation ratios, momentum, quality metrics, and other factors.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime, timedelta

# Add the parent directory to the path to import database modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'database'))
from connection import get_engine

class FactorCoefficientAnalyzer:
    def __init__(self):
        self.engine = get_engine()
        
    def get_basic_factors(self, analysis_date: datetime, universe: list) -> pd.DataFrame:
        """Extract basic factors from banking sector."""
        query = """
        SELECT ticker, 
               NetProfit_TTM / AvgTotalAssets as roaa,
               NetProfit_TTM / AvgTotalEquity as roae,
               OperatingProfit_TTM / TotalOperatingIncome_TTM as operating_margin,
               TotalOperatingIncome_TTM / AvgTotalAssets as asset_turnover
        FROM intermediary_calculations_banking_cleaned
        WHERE quarter = (
            SELECT MAX(quarter) 
            FROM intermediary_calculations_banking_cleaned 
            WHERE quarter <= %s
        ) AND ticker IN ({})
        """.format(','.join(['%s'] * len(universe)))
        
        params = [analysis_date] + universe
        
        try:
            factors = pd.read_sql(query, self.engine, params=tuple(params))
            factors['sector'] = 'Banking'
            return factors
        except Exception as e:
            print(f"Error extracting factors: {e}")
            return pd.DataFrame()
    
    def get_momentum_factors(self, analysis_date: datetime, universe: list) -> pd.DataFrame:
        """Create simulated momentum factors for demonstration."""
        # Create simulated momentum data
        np.random.seed(42)
        momentum_data = []
        
        for ticker in universe:
            momentum_data.append({
                'ticker': ticker,
                '1M': np.random.normal(0.02, 0.08),
                '3M': np.random.normal(0.06, 0.15),
                '6M': np.random.normal(0.12, 0.20),
                '12M': np.random.normal(0.25, 0.30)
            })
        
        return pd.DataFrame(momentum_data)
    
    def normalize_factors(self, factors_df: pd.DataFrame) -> pd.DataFrame:
        """Normalize factors using sector-neutral z-scores."""
        factor_columns = ['roaa', 'roae', 'operating_margin', 'asset_turnover', '1M', '3M', '6M', '12M']
        
        normalized_factors = factors_df.copy()
        
        for factor in factor_columns:
            if factor in normalized_factors.columns:
                normalized_factors[f'{factor}_normalized'] = normalized_factors.groupby('sector')[factor].transform(
                    lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
                )
        
        return normalized_factors
    
    def run_regression(self, normalized_factors: pd.DataFrame) -> dict:
        """Run regression and return coefficient analysis."""
        normalized_factor_cols = [col for col in normalized_factors.columns if col.endswith('_normalized')]
        
        X = normalized_factors[normalized_factor_cols].fillna(0)
        y = normalized_factors['forward_return'].fillna(0)
        
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
        
        results = {
            'coefficients': dict(zip(normalized_factor_cols, coefficients[1:])),
            'intercept': coefficients[0],
            'r_squared': r_squared,
            'n_observations': len(X)
        }
        
        return results
    
    def categorize_coefficients(self, results: dict) -> dict:
        """Categorize coefficients by factor type."""
        factor_categories = {
            'Quality Metrics': ['roaa_normalized', 'roae_normalized', 'operating_margin_normalized'],
            'Efficiency Ratios': ['asset_turnover_normalized'],
            'Momentum Factors': ['1M_normalized', '3M_normalized', '6M_normalized', '12M_normalized']
        }
        
        categorized_results = {}
        
        for category, factors in factor_categories.items():
            category_data = {}
            for factor in factors:
                if factor in results['coefficients']:
                    category_data[factor.replace('_normalized', '')] = {
                        'coefficient': results['coefficients'][factor],
                        'importance': abs(results['coefficients'][factor])
                    }
            categorized_results[category] = category_data
        
        return categorized_results
    
    def print_coefficient_analysis(self, categorized_results: dict, overall_results: dict):
        """Print detailed coefficient analysis."""
        print("\n" + "="*80)
        print("FACTOR COEFFICIENT ANALYSIS")
        print("="*80)
        
        print(f"\nOverall Model Performance:")
        print(f"R-squared: {overall_results['r_squared']:.4f}")
        print(f"Number of observations: {overall_results['n_observations']}")
        print(f"Intercept: {overall_results['intercept']:.4f}")
        
        print(f"\n" + "="*80)
        print("DETAILED COEFFICIENT ANALYSIS")
        print("="*80)
        
        for category, factors in categorized_results.items():
            if factors:
                print(f"\n{category.upper()}:")
                print("-" * 50)
                
                # Sort by absolute coefficient value
                sorted_factors = sorted(factors.items(), key=lambda x: abs(x[1]['coefficient']), reverse=True)
                
                for factor_name, stats in sorted_factors:
                    direction = "POSITIVE" if stats['coefficient'] > 0 else "NEGATIVE"
                    print(f"{factor_name:20} | Coef: {stats['coefficient']:8.4f} | {direction}")
        
        # Top factors by importance
        print(f"\n" + "="*80)
        print("TOP FACTORS BY IMPORTANCE (Absolute Coefficient)")
        print("="*80)
        
        all_factors = []
        for category, factors in categorized_results.items():
            for factor_name, stats in factors.items():
                all_factors.append((factor_name, stats['coefficient'], category))
        
        # Sort by absolute coefficient
        all_factors.sort(key=lambda x: abs(x[1]), reverse=True)
        
        for i, (factor_name, coef, category) in enumerate(all_factors):
            direction = "POSITIVE" if coef > 0 else "NEGATIVE"
            print(f"{i+1:2}. {factor_name:20} | Coef: {coef:8.4f} | {direction:8} | {category}")
        
        print("\n" + "="*80)
    
    def create_visualizations(self, categorized_results: dict, overall_results: dict):
        """Create coefficient visualizations."""
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Factor Coefficient Analysis', fontsize=16, fontweight='bold')
        
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
        ax1.set_yticklabels(categories, fontsize=9)
        ax1.set_xlabel('Coefficient Value')
        ax1.set_title('Factor Coefficients by Category')
        ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        # 2. Factor importance
        ax2 = axes[0, 1]
        importance_data = []
        importance_names = []
        
        for category, factors in categorized_results.items():
            if factors:
                for factor_name, stats in factors.items():
                    importance_data.append(stats['importance'])
                    importance_names.append(f"{category}\n{factor_name}")
        
        bars = ax2.barh(range(len(importance_data)), importance_data, alpha=0.7)
        ax2.set_yticks(range(len(importance_names)))
        ax2.set_yticklabels(importance_names, fontsize=9)
        ax2.set_xlabel('Absolute Coefficient Value')
        ax2.set_title('Factor Importance')
        
        # 3. Category summary
        ax3 = axes[1, 0]
        category_summary = {}
        
        for category, factors in categorized_results.items():
            if factors:
                avg_coef = np.mean([abs(stats['coefficient']) for stats in factors.values()])
                positive_count = sum(1 for stats in factors.values() if stats['coefficient'] > 0)
                category_summary[category] = {'avg_coef': avg_coef, 'positive': positive_count, 'total': len(factors)}
        
        if category_summary:
            categories_summary = list(category_summary.keys())
            avg_coefs = [category_summary[cat]['avg_coef'] for cat in categories_summary]
            positive_ratios = [category_summary[cat]['positive'] / category_summary[cat]['total'] for cat in categories_summary]
            
            x = np.arange(len(categories_summary))
            width = 0.35
            
            bars1 = ax3.bar(x - width/2, avg_coefs, width, label='Avg |Coefficient|', alpha=0.7)
            ax3_twin = ax3.twinx()
            bars2 = ax3_twin.bar(x + width/2, positive_ratios, width, label='Positive Ratio', alpha=0.7, color='orange')
            
            ax3.set_xlabel('Factor Categories')
            ax3.set_ylabel('Average |Coefficient|')
            ax3_twin.set_ylabel('Ratio of Positive Coefficients')
            ax3.set_title('Category Summary')
            ax3.set_xticks(x)
            ax3.set_xticklabels(categories_summary, rotation=45, ha='right')
            
            ax3.legend(loc='upper left')
            ax3_twin.legend(loc='upper right')
        
        # 4. Model performance
        ax4 = axes[1, 1]
        performance_metrics = ['R-squared', 'Observations']
        performance_values = [overall_results['r_squared'], overall_results['n_observations']]
        
        bars = ax4.bar(performance_metrics, performance_values, alpha=0.7)
        ax4.set_title('Model Performance')
        ax4.set_ylabel('Value')
        
        # Add value labels on bars
        for bar, value in zip(bars, performance_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.3f}' if value < 1 else f'{value:.0f}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('factor_coefficient_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_analysis(self, analysis_date: datetime = None):
        """Run complete factor coefficient analysis."""
        if analysis_date is None:
            analysis_date = datetime(2024, 12, 31)
        
        print(f"Running factor coefficient analysis for {analysis_date}")
        
        # Get universe
        universe_query = """
        SELECT DISTINCT ticker FROM intermediary_calculations_banking_cleaned 
        LIMIT 20
        """
        
        universe_df = pd.read_sql(universe_query, self.engine)
        universe = universe_df['ticker'].tolist()
        
        print(f"Universe size: {len(universe)} tickers")
        
        # Extract factors
        factors = self.get_basic_factors(analysis_date, universe)
        momentum = self.get_momentum_factors(analysis_date, universe)
        
        if len(factors) == 0:
            print("No factors extracted")
            return
        
        # Merge factors and momentum
        all_factors = factors.merge(momentum, on='ticker', how='inner')
        
        print(f"Factors shape: {all_factors.shape}")
        print("Sample factors:")
        print(all_factors.head())
        
        # Normalize factors
        normalized_factors = self.normalize_factors(all_factors)
        
        print(f"Normalized factors shape: {normalized_factors.shape}")
        print("Sample normalized factors:")
        print(normalized_factors.head())
        
        # Create simulated forward returns (for demonstration)
        np.random.seed(42)
        normalized_factors['forward_return'] = np.random.normal(0.05, 0.15, len(normalized_factors))
        
        print("Columns in normalized_factors:")
        print(normalized_factors.columns.tolist())
        
        # Create forward returns DataFrame
        forward_returns_df = normalized_factors[['ticker', 'forward_return']]
        
        # Run regression directly on normalized_factors since it already has forward_return
        results = self.run_regression(normalized_factors)
        
        if not results:
            print("Regression failed")
            return
        
        # Categorize coefficients
        categorized_results = self.categorize_coefficients(results)
        
        # Print analysis
        self.print_coefficient_analysis(categorized_results, results)
        
        # Create visualizations
        self.create_visualizations(categorized_results, results)
        
        return {
            'results': results,
            'categorized_results': categorized_results,
            'factors': all_factors,
            'normalized_factors': normalized_factors
        }


def main():
    """Main execution function."""
    analyzer = FactorCoefficientAnalyzer()
    
    # Run analysis for recent date
    analysis_date = datetime(2024, 12, 31)
    
    results = analyzer.run_analysis(analysis_date)
    
    if results:
        print("\nAnalysis completed successfully!")
        print("Results saved to: factor_coefficient_analysis.png")
    else:
        print("Analysis failed")


if __name__ == "__main__":
    main() 