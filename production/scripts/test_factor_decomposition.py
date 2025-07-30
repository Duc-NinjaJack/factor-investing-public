"""
Test Factor Decomposition Analysis
=================================
Simplified version to test the factor decomposition functionality
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime, timedelta

# Add the parent directory to the path to import database modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'database'))
from connection import get_engine

def test_factor_decomposition():
    """Test the factor decomposition functionality with a simple example."""
    
    print("Testing Factor Decomposition Analysis...")
    
    # Get database connection
    engine = get_engine()
    
    # Test with a small subset of data
    test_date = datetime(2024, 12, 31)
    
    # Get a small universe of tickers
    universe_query = """
    SELECT DISTINCT ticker FROM intermediary_calculations_banking_cleaned 
    LIMIT 5
    """
    
    try:
        universe_df = pd.read_sql(universe_query, engine)
        universe = universe_df['ticker'].tolist()
        print(f"Test universe: {universe}")
        
        # Get sector mapping
        sector_query = """
        SELECT DISTINCT ticker, 'Banking' as sector 
        FROM intermediary_calculations_banking_cleaned 
        WHERE ticker IN ({})
        """.format(','.join(['%s'] * len(universe)))
        
        sector_mapping = pd.read_sql(sector_query, engine, params=tuple(universe))
        print(f"Sector mapping shape: {sector_mapping.shape}")
        
        # Get some basic factor data
        factor_query = """
        SELECT ticker, 
               NetProfit_TTM / AvgTotalAssets as roaa,
               NetProfit_TTM / AvgTotalEquity as roae
        FROM intermediary_calculations_banking_cleaned
        WHERE quarter = (
            SELECT MAX(quarter) 
            FROM intermediary_calculations_banking_cleaned 
            WHERE quarter <= %s
        ) AND ticker IN ({})
        """.format(','.join(['%s'] * len(universe)))
        
        factors = pd.read_sql(factor_query, engine, params=tuple([test_date] + universe))
        print(f"Factors shape: {factors.shape}")
        print("Sample factors:")
        print(factors.head())
        
        # Normalize factors
        factors_with_sector = factors.merge(sector_mapping, on='ticker', how='inner')
        
        for factor in ['roaa', 'roae']:
            if factor in factors_with_sector.columns:
                factors_with_sector[f'{factor}_normalized'] = factors_with_sector.groupby('sector')[factor].transform(
                    lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
                )
        
        print("\nNormalized factors:")
        print(factors_with_sector[['ticker', 'sector', 'roaa', 'roaa_normalized', 'roae', 'roae_normalized']].head())
        
        # Create a simple forward returns simulation
        np.random.seed(42)
        factors_with_sector['forward_return'] = np.random.normal(0.05, 0.15, len(factors_with_sector))
        
        # Run simple regression
        normalized_cols = [col for col in factors_with_sector.columns if col.endswith('_normalized')]
        
        if normalized_cols:
            X = factors_with_sector[normalized_cols].fillna(0)
            y = factors_with_sector['forward_return'].fillna(0)
            
            # Manual linear regression
            X_with_intercept = np.column_stack([np.ones(len(X)), X])
            coefficients = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
            
            y_pred = X_with_intercept @ coefficients
            r_squared = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
            
            print(f"\nRegression Results:")
            print(f"Intercept: {coefficients[0]:.4f}")
            for i, col in enumerate(normalized_cols):
                print(f"{col}: {coefficients[i+1]:.4f}")
            print(f"R-squared: {r_squared:.4f}")
            
            # Create simple visualization
            plt.figure(figsize=(12, 8))
            
            # Plot factor coefficients
            plt.subplot(2, 2, 1)
            factor_names = [col.replace('_normalized', '') for col in normalized_cols]
            plt.bar(factor_names, coefficients[1:])
            plt.title('Factor Coefficients')
            plt.ylabel('Coefficient Value')
            plt.xticks(rotation=45)
            
            # Plot actual vs predicted returns
            plt.subplot(2, 2, 2)
            plt.scatter(y, y_pred, alpha=0.6)
            plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
            plt.xlabel('Actual Returns')
            plt.ylabel('Predicted Returns')
            plt.title(f'Actual vs Predicted (R² = {r_squared:.3f})')
            
            # Plot factor distributions
            plt.subplot(2, 2, 3)
            for col in normalized_cols:
                plt.hist(factors_with_sector[col].dropna(), alpha=0.5, label=col.replace('_normalized', ''))
            plt.xlabel('Normalized Factor Value')
            plt.ylabel('Frequency')
            plt.title('Factor Distributions')
            plt.legend()
            
            # Plot returns by sector
            plt.subplot(2, 2, 4)
            sector_returns = factors_with_sector.groupby('sector')['forward_return'].mean()
            plt.bar(sector_returns.index, sector_returns.values)
            plt.title('Average Returns by Sector')
            plt.ylabel('Average Return')
            
            plt.tight_layout()
            plt.savefig('test_factor_decomposition.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"\nTest completed successfully!")
            print(f"Results saved to: test_factor_decomposition.png")
            
        else:
            print("No normalized factors found for regression")
            
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_factor_decomposition() 