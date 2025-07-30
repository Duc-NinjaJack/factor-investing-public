"""
Factor Decomposition and Forward Returns Analysis
================================================
Purpose: Decompose QVM factors, normalize them, and model forward returns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from pathlib import Path
from typing import Dict, List, Optional
import warnings
import logging
from datetime import datetime, timedelta
import sys
import os

# Add the parent directory to the path to import database modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'database'))
from connection import get_engine

warnings.filterwarnings('ignore')

class FactorDecompositionAnalyzer:
    def __init__(self, config_path: str = None):
        self._setup_logging()
        self._load_configurations(config_path)
        self._create_database_engine()
        
    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def _load_configurations(self, config_path: str = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / 'config'
        
        with open(config_path / 'database.yml', 'r') as f:
            self.db_config = yaml.safe_load(f)
            
    def _create_database_engine(self):
        # Use existing database connection infrastructure
        self.engine = get_engine()
        
    def get_sector_mapping(self) -> pd.DataFrame:
        # Get sector mapping by querying each table separately
        banking_query = "SELECT DISTINCT ticker FROM intermediary_calculations_banking_cleaned"
        securities_query = "SELECT DISTINCT ticker FROM intermediary_calculations_securities_cleaned"
        enhanced_query = "SELECT DISTINCT ticker FROM intermediary_calculations_enhanced"
        
        banking_df = pd.read_sql(banking_query, self.engine)
        securities_df = pd.read_sql(securities_query, self.engine)
        enhanced_df = pd.read_sql(enhanced_query, self.engine)
        
        # Add sector information
        banking_df['sector'] = 'Banking'
        securities_df['sector'] = 'Securities'
        enhanced_df['sector'] = 'Non-Financial'
        
        # Combine all dataframes
        combined_df = pd.concat([banking_df, securities_df, enhanced_df], ignore_index=True)
        return combined_df
        
    def get_forward_returns(self, analysis_date: pd.Timestamp, universe: List[str]) -> pd.DataFrame:
        forward_date = analysis_date + pd.DateOffset(months=1)
        
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
        
        return pd.read_sql(query, self.engine, params=params)
        
    def extract_individual_factors(self, analysis_date: pd.Timestamp, universe: List[str]) -> pd.DataFrame:
        self.logger.info(f"Extracting individual factors for {analysis_date}")
        
        # Get fundamentals data from each table separately
        banking_query = """
        SELECT ticker, 
               NULL as roae, NULL as roaa, NULL as operating_margin, NULL as ebitda_margin,
               NULL as pe_ratio, NULL as pb_ratio, NULL as ps_ratio, NULL as ev_ebitda,
               NULL as debt_to_equity, NULL as current_ratio, NULL as quick_ratio,
               NULL as inventory_turnover, NULL as asset_turnover, NULL as equity_turnover
        FROM intermediary_calculations_banking_cleaned
        WHERE quarter_end = (
            SELECT MAX(quarter_end) 
            FROM intermediary_calculations_banking_cleaned 
            WHERE quarter_end <= %s
        ) AND ticker IN ({})
        """.format(','.join(['%s'] * len(universe)))
        
        securities_query = """
        SELECT ticker, 
               NULL as roae, NULL as roaa, NULL as operating_margin, NULL as ebitda_margin,
               NULL as pe_ratio, NULL as pb_ratio, NULL as ps_ratio, NULL as ev_ebitda,
               NULL as debt_to_equity, NULL as current_ratio, NULL as quick_ratio,
               NULL as inventory_turnover, NULL as asset_turnover, NULL as equity_turnover
        FROM intermediary_calculations_securities_cleaned
        WHERE quarter_end = (
            SELECT MAX(quarter_end) 
            FROM intermediary_calculations_securities_cleaned 
            WHERE quarter_end <= %s
        ) AND ticker IN ({})
        """.format(','.join(['%s'] * len(universe)))
        
        enhanced_query = """
        SELECT ticker, 
               NULL as roae, NULL as roaa, NULL as operating_margin, NULL as ebitda_margin,
               NULL as pe_ratio, NULL as pb_ratio, NULL as ps_ratio, NULL as ev_ebitda,
               NULL as debt_to_equity, NULL as current_ratio, NULL as quick_ratio,
               NULL as inventory_turnover, NULL as asset_turnover, NULL as equity_turnover
        FROM intermediary_calculations_enhanced
        WHERE quarter_end = (
            SELECT MAX(quarter_end) 
            FROM intermediary_calculations_enhanced 
            WHERE quarter_end <= %s
        ) AND ticker IN ({})
        """.format(','.join(['%s'] * len(universe)))
        
        # Execute queries with parameters
        params = [analysis_date] + universe
        
        try:
            banking_df = pd.read_sql(banking_query, self.engine, params=params)
            securities_df = pd.read_sql(securities_query, self.engine, params=params)
            enhanced_df = pd.read_sql(enhanced_query, self.engine, params=params)
            
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
            momentum = pd.read_sql(momentum_query, self.engine, params=momentum_params)
            
            # Pivot momentum data
            momentum_wide = momentum.pivot(index='ticker', columns='period', values='momentum_return').reset_index()
            momentum_wide.columns.name = None
            
            # Merge fundamentals and momentum
            factors = fundamentals.merge(momentum_wide, on='ticker', how='inner')
            
        except Exception as e:
            self.logger.error(f"Error extracting momentum: {e}")
            factors = fundamentals
        
        return factors
        
    def normalize_factors(self, factors_df: pd.DataFrame, sector_mapping: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Normalizing factors using sector-neutral z-scores")
        
        factors_with_sector = factors_df.merge(sector_mapping, on='ticker', how='inner')
        
        factor_columns = [
            'roae', 'roaa', 'operating_margin', 'ebitda_margin',
            'pe_ratio', 'pb_ratio', 'ps_ratio', 'ev_ebitda',
            'debt_to_equity', 'current_ratio', 'quick_ratio',
            'inventory_turnover', 'asset_turnover', 'equity_turnover',
            '1M', '3M', '6M', '12M'
        ]
        
        normalized_factors = factors_with_sector.copy()
        
        for factor in factor_columns:
            if factor in normalized_factors.columns:
                normalized_factors[f'{factor}_normalized'] = normalized_factors.groupby('sector')[factor].transform(
                    lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
                )
        
        return normalized_factors
        
    def run_factor_regression(self, normalized_factors: pd.DataFrame, forward_returns: pd.DataFrame) -> Dict:
        self.logger.info("Running factor regression analysis")
        
        regression_data = normalized_factors.merge(forward_returns, on='ticker', how='inner')
        
        normalized_factor_cols = [col for col in regression_data.columns if col.endswith('_normalized')]
        
        X = regression_data[normalized_factor_cols].fillna(0)
        y = regression_data['forward_return'].fillna(0)
        
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X = X[mask]
        y = y[mask]
        
        if len(X) == 0:
            return {}
        
        # Manual linear regression using numpy
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
        
    def run_time_series_analysis(self, start_date: pd.Timestamp, end_date: pd.Timestamp, 
                                universe: List[str]) -> pd.DataFrame:
        self.logger.info(f"Running time series analysis from {start_date} to {end_date}")
        
        sector_mapping = self.get_sector_mapping()
        dates = pd.date_range(start=start_date, end=end_date, freq='M')
        
        time_series_results = []
        
        for date in dates:
            try:
                self.logger.info(f"Processing date: {date}")
                
                factors = self.extract_individual_factors(date, universe)
                if len(factors) == 0:
                    continue
                    
                normalized_factors = self.normalize_factors(factors, sector_mapping)
                forward_returns = self.get_forward_returns(date, universe)
                
                if len(forward_returns) == 0:
                    continue
                    
                regression_results = self.run_factor_regression(normalized_factors, forward_returns)
                
                if regression_results:
                    result_row = {
                        'date': date,
                        'r_squared': regression_results['r_squared'],
                        'n_observations': regression_results['n_observations']
                    }
                    
                    for factor, coef in regression_results['coefficients'].items():
                        result_row[f'coef_{factor}'] = coef
                    
                    time_series_results.append(result_row)
                    
            except Exception as e:
                self.logger.error(f"Error processing date {date}: {e}")
                continue
        
        return pd.DataFrame(time_series_results)
        
    def create_visualizations(self, time_series_results: pd.DataFrame):
        self.logger.info("Creating visualizations")
        
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Factor Decomposition and Forward Returns Analysis', fontsize=16, fontweight='bold')
        
        if not time_series_results.empty:
            coef_cols = [col for col in time_series_results.columns if col.startswith('coef_')]
            
            # Plot coefficient evolution
            for col in coef_cols[:5]:
                factor_name = col.replace('coef_', '').replace('_normalized', '')
                axes[0, 0].plot(time_series_results['date'], time_series_results[col], 
                               label=factor_name, alpha=0.7)
            
            axes[0, 0].set_title('Factor Coefficients Over Time')
            axes[0, 0].set_xlabel('Date')
            axes[0, 0].set_ylabel('Coefficient Value')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # R-squared over time
            axes[0, 1].plot(time_series_results['date'], time_series_results['r_squared'], 
                           color='red', linewidth=2)
            axes[0, 1].set_title('Model R-squared Over Time')
            axes[0, 1].set_xlabel('Date')
            axes[0, 1].set_ylabel('R-squared')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Coefficient heatmap
            coef_data = time_series_results[coef_cols].T
            coef_data.index = [col.replace('coef_', '').replace('_normalized', '') for col in coef_cols]
            
            sns.heatmap(coef_data, cmap='RdYlBu_r', center=0, ax=axes[1, 0], 
                       cbar_kws={'label': 'Coefficient Value'})
            axes[1, 0].set_title('Factor Coefficients Heatmap')
            axes[1, 0].set_xlabel('Time Period')
            axes[1, 0].set_ylabel('Factors')
            
            # Factor importance over time
            factor_importance = coef_data.abs().mean(axis=1).sort_values(ascending=False)
            axes[1, 1].barh(range(len(factor_importance)), factor_importance.values)
            axes[1, 1].set_yticks(range(len(factor_importance)))
            axes[1, 1].set_yticklabels(factor_importance.index)
            axes[1, 1].set_title('Average Factor Importance (Absolute Coefficient)')
            axes[1, 1].set_xlabel('Average Absolute Coefficient')
        
        plt.tight_layout()
        plt.savefig('factor_decomposition_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def run_comprehensive_analysis(self, start_date: str, end_date: str):
        self.logger.info("Starting comprehensive factor decomposition analysis")
        
        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)
        
        # Get universe by querying each table separately
        banking_query = "SELECT DISTINCT ticker FROM intermediary_calculations_banking_cleaned"
        securities_query = "SELECT DISTINCT ticker FROM intermediary_calculations_securities_cleaned"
        enhanced_query = "SELECT DISTINCT ticker FROM intermediary_calculations_enhanced"
        
        banking_df = pd.read_sql(banking_query, self.engine)
        securities_df = pd.read_sql(securities_query, self.engine)
        enhanced_df = pd.read_sql(enhanced_query, self.engine)
        
        # Combine all tickers
        universe_df = pd.concat([banking_df, securities_df, enhanced_df], ignore_index=True)
        universe = universe_df['ticker'].unique().tolist()
        
        self.logger.info(f"Universe size: {len(universe)} tickers")
        
        # Run time series analysis
        time_series_results = self.run_time_series_analysis(start_date, end_date, universe)
        
        # Create visualizations
        self.create_visualizations(time_series_results)
        
        # Save results
        time_series_results.to_csv('factor_decomposition_time_series.csv', index=False)
        
        # Print summary
        self.print_summary_statistics(time_series_results)
        
        return time_series_results
        
    def print_summary_statistics(self, time_series_results: pd.DataFrame):
        print("\n" + "="*80)
        print("FACTOR DECOMPOSITION ANALYSIS SUMMARY")
        print("="*80)
        
        if not time_series_results.empty:
            print(f"\nTime Series Analysis ({len(time_series_results)} periods):")
            print(f"Average R-squared: {time_series_results['r_squared'].mean():.4f}")
            print(f"R-squared range: {time_series_results['r_squared'].min():.4f} - {time_series_results['r_squared'].max():.4f}")
            print(f"Average observations per period: {time_series_results['n_observations'].mean():.0f}")
            
            # Top factors by average importance
            coef_cols = [col for col in time_series_results.columns if col.startswith('coef_')]
            if coef_cols:
                factor_importance = time_series_results[coef_cols].abs().mean().sort_values(ascending=False)
                print(f"\nTop 5 Most Important Factors:")
                for i, (factor, importance) in enumerate(factor_importance.head().items()):
                    factor_name = factor.replace('coef_', '').replace('_normalized', '')
                    print(f"  {i+1}. {factor_name}: {importance:.4f}")
        
        print("\n" + "="*80)


def main():
    analyzer = FactorDecompositionAnalyzer()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # 2 years
    
    results = analyzer.run_comprehensive_analysis(
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )
    
    print("Analysis completed successfully!")
    print("Results saved to:")
    print("- factor_decomposition_analysis.png")
    print("- factor_decomposition_time_series.csv")


if __name__ == "__main__":
    main() 