"""
Debug Value Factor Analysis
===========================
Investigates why the value factor coefficient is exactly 0.0000
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Add the parent directory to the path to import database modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'database'))
from connection import get_engine

class ValueFactorDebugger:
    def __init__(self):
        self.engine = get_engine()
        
    def check_banking_table_schema(self):
        """Check the schema of the banking table to understand available columns."""
        print("="*80)
        print("CHECKING BANKING TABLE SCHEMA")
        print("="*80)
        
        schema_query = """
        DESCRIBE intermediary_calculations_banking_cleaned
        """
        
        schema = pd.read_sql(schema_query, self.engine)
        print("Available columns in intermediary_calculations_banking_cleaned:")
        print(schema)
        
        # Check if market_cap column exists
        market_cap_exists = 'market_cap' in schema['Field'].values
        print(f"\nMarket cap column exists: {market_cap_exists}")
        
        if not market_cap_exists:
            print("Looking for alternative market cap columns...")
            cap_columns = [col for col in schema['Field'].values if 'cap' in col.lower() or 'market' in col.lower()]
            print(f"Potential market cap columns: {cap_columns}")
        
        return schema
    
    def get_sample_banking_data(self):
        """Get sample data from banking table to understand the structure."""
        print("\n" + "="*80)
        print("SAMPLE BANKING DATA")
        print("="*80)
        
        sample_query = """
        SELECT * FROM intermediary_calculations_banking_cleaned 
        LIMIT 5
        """
        
        sample_data = pd.read_sql(sample_query, self.engine)
        print("Sample data columns:")
        print(sample_data.columns.tolist())
        print("\nSample data:")
        print(sample_data)
        
        return sample_data
    
    def get_market_cap_from_daily_data(self, ticker: str, analysis_date: datetime):
        """Get market cap from daily data table."""
        try:
            # Get latest available data for the ticker
            market_cap_query = """
            SELECT trading_date, close_price, volume, shares_outstanding
            FROM vcsc_daily_data_complete
            WHERE ticker = %s AND trading_date <= %s
            ORDER BY trading_date DESC
            LIMIT 1
            """
            
            market_data = pd.read_sql(market_cap_query, self.engine, params=(ticker, analysis_date))
            
            if len(market_data) > 0:
                close_price = market_data.iloc[0]['close_price']
                shares_outstanding = market_data.iloc[0].get('shares_outstanding', 0)
                
                if shares_outstanding > 0:
                    market_cap = close_price * shares_outstanding
                else:
                    # Try to get shares from another source or estimate
                    market_cap = 0
                
                return market_cap, close_price, shares_outstanding
            else:
                return 0, 0, 0
                
        except Exception as e:
            print(f"Error getting market cap for {ticker}: {e}")
            return 0, 0, 0
    
    def calculate_value_factors_debug(self, analysis_date: datetime, universe: list):
        """Calculate value factors with detailed debugging."""
        print("\n" + "="*80)
        print("VALUE FACTOR CALCULATION DEBUG")
        print("="*80)
        
        value_data = []
        
        for ticker in universe[:5]:  # Test with first 5 tickers
            print(f"\n--- Processing {ticker} ---")
            
            try:
                # Get fundamental data
                fundamental_query = """
                SELECT 
                    NetProfit_TTM,
                    AvgTotalEquity,
                    TotalOperatingIncome_TTM
                FROM intermediary_calculations_banking_cleaned
                WHERE ticker = %s AND quarter = (
                    SELECT MAX(quarter) 
                    FROM intermediary_calculations_banking_cleaned 
                    WHERE quarter <= %s
                )
                """
                
                fundamental_data = pd.read_sql(fundamental_query, self.engine, params=(ticker, analysis_date))
                
                if len(fundamental_data) > 0:
                    row = fundamental_data.iloc[0]
                    print(f"Fundamental data: {row.to_dict()}")
                    
                    # Get market cap from daily data
                    market_cap, close_price, shares = self.get_market_cap_from_daily_data(ticker, analysis_date)
                    print(f"Market data - Close: {close_price}, Shares: {shares}, Market Cap: {market_cap}")
                    
                    # Calculate value ratios
                    pe_score = 0
                    pb_score = 0
                    ps_score = 0
                    
                    # P/E ratio
                    if pd.notna(row['NetProfit_TTM']) and row['NetProfit_TTM'] > 0 and market_cap > 0:
                        pe_ratio = market_cap / row['NetProfit_TTM']
                        pe_score = 1 / pe_ratio if pe_ratio > 0 else 0
                        print(f"P/E ratio: {pe_ratio:.2f}, P/E score: {pe_score:.6f}")
                    else:
                        print(f"P/E calculation failed - NetProfit: {row['NetProfit_TTM']}, Market Cap: {market_cap}")
                    
                    # P/B ratio
                    if pd.notna(row['AvgTotalEquity']) and row['AvgTotalEquity'] > 0 and market_cap > 0:
                        pb_ratio = market_cap / row['AvgTotalEquity']
                        pb_score = 1 / pb_ratio if pb_ratio > 0 else 0
                        print(f"P/B ratio: {pb_ratio:.2f}, P/B score: {pb_score:.6f}")
                    else:
                        print(f"P/B calculation failed - Equity: {row['AvgTotalEquity']}, Market Cap: {market_cap}")
                    
                    # P/S ratio
                    if pd.notna(row['TotalOperatingIncome_TTM']) and row['TotalOperatingIncome_TTM'] > 0 and market_cap > 0:
                        ps_ratio = market_cap / row['TotalOperatingIncome_TTM']
                        ps_score = 1 / ps_ratio if ps_ratio > 0 else 0
                        print(f"P/S ratio: {ps_ratio:.2f}, P/S score: {ps_score:.6f}")
                    else:
                        print(f"P/S calculation failed - Revenue: {row['TotalOperatingIncome_TTM']}, Market Cap: {market_cap}")
                    
                    # Banking sector weights: PE=60%, PB=40%
                    value_score = 0.6 * pe_score + 0.4 * pb_score
                    print(f"Final value score: {value_score:.6f}")
                    
                else:
                    print("No fundamental data found")
                    value_score = 0
                    
                value_data.append({
                    'ticker': ticker,
                    'value_score': value_score,
                    'pe_score': pe_score,
                    'pb_score': pb_score,
                    'ps_score': ps_score,
                    'market_cap': market_cap,
                    'close_price': close_price,
                    'shares_outstanding': shares
                })
                
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
                value_data.append({
                    'ticker': ticker,
                    'value_score': 0,
                    'pe_score': 0,
                    'pb_score': 0,
                    'ps_score': 0,
                    'market_cap': 0,
                    'close_price': 0,
                    'shares_outstanding': 0
                })
        
        return pd.DataFrame(value_data)
    
    def check_value_factor_distribution(self, value_factors_df: pd.DataFrame):
        """Check the distribution of value factors."""
        print("\n" + "="*80)
        print("VALUE FACTOR DISTRIBUTION ANALYSIS")
        print("="*80)
        
        print("Value factors summary:")
        print(value_factors_df.describe())
        
        print("\nValue factors by ticker:")
        print(value_factors_df[['ticker', 'value_score', 'pe_score', 'pb_score', 'ps_score']])
        
        # Check for zero values
        zero_count = (value_factors_df['value_score'] == 0).sum()
        total_count = len(value_factors_df)
        print(f"\nZero value scores: {zero_count}/{total_count} ({zero_count/total_count*100:.1f}%)")
        
        # Check for non-zero values
        non_zero = value_factors_df[value_factors_df['value_score'] != 0]
        if len(non_zero) > 0:
            print(f"\nNon-zero value scores:")
            print(non_zero[['ticker', 'value_score', 'pe_score', 'pb_score', 'ps_score']])
        else:
            print("\nALL VALUE SCORES ARE ZERO!")
        
        return value_factors_df
    
    def run_debug_analysis(self, analysis_date: datetime = None):
        """Run comprehensive debug analysis."""
        if analysis_date is None:
            analysis_date = datetime(2024, 12, 18)
        
        print(f"Debugging value factor analysis for {analysis_date}")
        
        # Check table schema
        schema = self.check_banking_table_schema()
        
        # Check daily data schema
        daily_schema = self.check_daily_data_schema()
        
        # Get sample data
        sample_data = self.get_sample_banking_data()
        
        # Get sample daily data
        sample_daily = self.get_sample_daily_data('ACB')
        
        # Get universe
        universe_query = """
        SELECT DISTINCT ticker FROM intermediary_calculations_banking_cleaned 
        LIMIT 10
        """
        
        universe_df = pd.read_sql(universe_query, self.engine)
        universe = universe_df['ticker'].tolist()
        
        print(f"\nUniverse: {universe}")
        
        # Calculate value factors with debugging
        value_factors = self.calculate_value_factors_debug(analysis_date, universe)
        
        # Check distribution
        self.check_value_factor_distribution(value_factors)
        
        return value_factors

    def check_daily_data_schema(self):
        """Check the schema of the daily data table to understand available columns."""
        print("\n" + "="*80)
        print("CHECKING DAILY DATA TABLE SCHEMA")
        print("="*80)
        
        schema_query = """
        DESCRIBE vcsc_daily_data_complete
        """
        
        schema = pd.read_sql(schema_query, self.engine)
        print("Available columns in vcsc_daily_data_complete:")
        print(schema)
        
        # Check for potential market cap related columns
        cap_columns = [col for col in schema['Field'].values if 'cap' in col.lower() or 'market' in col.lower()]
        volume_columns = [col for col in schema['Field'].values if 'volume' in col.lower()]
        share_columns = [col for col in schema['Field'].values if 'share' in col.lower()]
        
        print(f"\nPotential market cap columns: {cap_columns}")
        print(f"Volume columns: {volume_columns}")
        print(f"Share columns: {share_columns}")
        
        return schema
    
    def get_sample_daily_data(self, ticker: str):
        """Get sample daily data for a ticker."""
        print(f"\n--- Sample daily data for {ticker} ---")
        
        sample_query = """
        SELECT * FROM vcsc_daily_data_complete 
        WHERE ticker = %s 
        ORDER BY trading_date DESC 
        LIMIT 3
        """
        
        sample_data = pd.read_sql(sample_query, self.engine, params=(ticker,))
        print("Sample daily data columns:")
        print(sample_data.columns.tolist())
        print("\nSample daily data:")
        print(sample_data)
        
        return sample_data


def main():
    """Main execution function."""
    debugger = ValueFactorDebugger()
    
    # Run debug analysis
    analysis_date = datetime(2024, 12, 18)
    
    results = debugger.run_debug_analysis(analysis_date)
    
    print("\n" + "="*80)
    print("DEBUG ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main() 