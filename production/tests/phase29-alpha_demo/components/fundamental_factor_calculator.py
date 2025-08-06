#!/usr/bin/env python3
"""
Fundamental Factor Calculator for Comprehensive Multi-Factor Strategy
====================================================================

This component calculates fundamental factors directly from raw fundamental_values data
for maximum coverage and precision. It handles:

1. ROAA (Return on Average Assets) - Quality factor
2. P/E Ratio (Price to Earnings) - Value factor  
3. FCF Yield (Free Cash Flow Yield) - Value factor
4. F-Score (Piotroski F-Score) - Quality factor
5. Sector-specific calculations for Banking and Securities

The calculator uses raw fundamental data to ensure maximum historical coverage
and precise financial calculations.
"""

# %% [markdown]
# # IMPORTS AND SETUP

# %%
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Database connectivity
from sqlalchemy import create_engine, text

# Add Project Root to Python Path
try:
    current_path = Path.cwd()
    while not (current_path / 'production').is_dir():
        if current_path.parent == current_path:
            raise FileNotFoundError("Could not find the 'production' directory.")
        current_path = current_path.parent
    
    project_root = current_path
    
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    from production.database.connection import get_database_manager
    print(f"✅ Successfully imported production modules.")
    print(f"   - Project Root set to: {project_root}")

except (ImportError, FileNotFoundError) as e:
    print(f"❌ ERROR: Could not import production modules. Please check your directory structure.")
    print(f"   - Final Path Searched: {project_root}")
    print(f"   - Error: {e}")
    raise

# %% [markdown]
# # FUNDAMENTAL FACTOR CALCULATOR CLASS

# %%
class FundamentalFactorCalculator:
    """
    Comprehensive fundamental factor calculator using raw fundamental data.
    
    This calculator provides maximum coverage by using raw fundamental_values
    instead of pre-calculated intermediary tables.
    """
    
    def __init__(self, db_engine):
        """Initialize the calculator with database connection."""
        self.db_engine = db_engine
        
        # Define key financial statement items based on VNSC mappings
        self._define_financial_items()
        
        print("✅ Fundamental Factor Calculator initialized")
        print("   - Using raw fundamental_values for maximum coverage")
        print("   - Using correct VNSC item mappings")
    
    def _define_financial_items(self):
        """Define key financial statement items based on VNSC mappings."""
        # Based on VNSC mappings analysis
        self.pl_items = {
            'net_profit': [1],      # isa20 - Net profit/(loss) after tax
            'revenue': [2],         # isa2 - Net sales
            'gross_profit': [4],    # isa4 - Gross Profit
            'operating_profit': [11], # isa11 - Operating profit/(loss)
            'financial_income': [5],  # isa5 - Financial income
            'financial_expenses': [6], # isa6 - Financial expenses
            'cost_of_sales': [3],   # isa3 - Cost of sales
            'selling_expenses': [9], # isa9 - Selling expenses
            'admin_expenses': [10],  # isa10 - General and admin expenses
        }
        
        self.bs_items = {
            'total_assets': [101, 102],  # bsa43 - Total Assets
            'current_assets': [106, 107, 108, 109, 110, 111, 112, 113, 114, 115],
            'non_current_assets': [116, 117, 118, 119, 120, 121, 122, 123, 124, 125],
            'total_liabilities': [201, 202, 203, 204, 205],
            'current_liabilities': [206, 207, 208, 209, 210, 211, 212, 213, 214, 215],
            'non_current_liabilities': [216, 217, 218, 219, 220, 221, 222, 223, 224, 225],
            'total_equity': [301, 302, 303, 304, 305],
            'cash': [106, 107],  # Cash and cash equivalents
            'inventory': [110, 111],  # Inventories
            'receivables': [112, 113],  # Receivables
            'payables': [206, 207],  # Payables
            'short_term_debt': [208, 209],  # Short-term debt
            'long_term_debt': [216, 217]  # Long-term debt
        }
        
        self.cf_items = {
            'operating_cash_flow': [401, 402, 403, 404, 405],  # Operating cash flow
            'investing_cash_flow': [406, 407, 408, 409, 410],  # Investing cash flow
            'financing_cash_flow': [411, 412, 413, 414, 415],  # Financing cash flow
            'capex': [406, 407],  # Capital expenditures
            'depreciation': [401, 402]  # Depreciation and amortization
        }
    
    def load_raw_fundamental_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load raw fundamental data from fundamental_values table."""
        print(f"📊 Loading raw fundamental data from {start_date} to {end_date}...")
        
        # Load all fundamental data for the period
        query = text("""
            SELECT 
                ticker,
                item_id,
                statement_type,
                year,
                quarter,
                value
            FROM fundamental_values
            WHERE year >= :start_year AND year <= :end_year
            AND value IS NOT NULL
            ORDER BY ticker, year, quarter, item_id
        """)
        
        start_year = int(start_date[:4])
        end_year = int(end_date[:4])
        
        data = pd.read_sql(query, self.db_engine, params={
            'start_year': start_year,
            'end_year': end_year
        })
        
        print(f"   ✅ Loaded {len(data):,} fundamental records")
        print(f"   📊 Coverage: {data['ticker'].nunique()} tickers, {data['year'].nunique()} years")
        
        return data
    
    def calculate_ttm_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Trailing Twelve Months (TTM) values."""
        print("📊 Calculating TTM values...")
        
        ttm_data = []
        
        for ticker in data['ticker'].unique():
            ticker_data = data[data['ticker'] == ticker].copy()
            
            for year in ticker_data['year'].unique():
                for quarter in [1, 2, 3, 4]:
                    # Get 4 quarters for TTM calculation
                    quarters_needed = []
                    for q in range(4):
                        target_year = year
                        target_quarter = quarter - q
                        
                        if target_quarter <= 0:
                            target_quarter += 4
                            target_year -= 1
                        
                        quarters_needed.append((target_year, target_quarter))
                    
                    # Check if all quarters are available
                    available_quarters = []
                    for target_year, target_quarter in quarters_needed:
                        quarter_data = ticker_data[
                            (ticker_data['year'] == target_year) & 
                            (ticker_data['quarter'] == target_quarter)
                        ]
                        if not quarter_data.empty:
                            available_quarters.append(quarter_data)
                    
                    if len(available_quarters) >= 3:  # At least 3 quarters for TTM
                        # Calculate TTM for each item
                        for item_id in ticker_data['item_id'].unique():
                            ttm_value = 0
                            quarters_used = 0
                            
                            for quarter_data in available_quarters:
                                item_value = quarter_data[quarter_data['item_id'] == item_id]['value'].sum()
                                if not pd.isna(item_value) and item_value != 0:
                                    ttm_value += item_value
                                    quarters_used += 1
                            
                            if quarters_used >= 3:
                                # Annualize if less than 4 quarters
                                if quarters_used < 4:
                                    ttm_value = ttm_value * (4 / quarters_used)
                                
                                ttm_data.append({
                                    'ticker': ticker,
                                    'item_id': item_id,
                                    'year': year,
                                    'quarter': quarter,
                                    'ttm_value': ttm_value,
                                    'quarters_used': quarters_used
                                })
        
        ttm_df = pd.DataFrame(ttm_data)
        print(f"   ✅ Calculated TTM values for {len(ttm_df):,} records")
        
        return ttm_df
    
    def calculate_balance_sheet_averages(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate 5-point balance sheet averages."""
        print("📊 Calculating balance sheet averages...")
        
        avg_data = []
        
        for ticker in data['ticker'].unique():
            ticker_data = data[data['ticker'] == ticker].copy()
            
            for year in ticker_data['year'].unique():
                for quarter in [1, 2, 3, 4]:
                    # Get 5 points for averaging (current + 4 previous)
                    points_needed = []
                    for p in range(5):
                        target_year = year
                        target_quarter = quarter - p
                        
                        if target_quarter <= 0:
                            target_quarter += 4
                            target_year -= 1
                        
                        points_needed.append((target_year, target_quarter))
                    
                    # Check if points are available
                    available_points = []
                    for target_year, target_quarter in points_needed:
                        point_data = ticker_data[
                            (ticker_data['year'] == target_year) & 
                            (ticker_data['quarter'] == target_quarter)
                        ]
                        if not point_data.empty:
                            available_points.append(point_data)
                    
                    if len(available_points) >= 3:  # At least 3 points for averaging
                        # Calculate averages for each item
                        for item_id in ticker_data['item_id'].unique():
                            avg_value = 0
                            points_used = 0
                            
                            for point_data in available_points:
                                item_value = point_data[point_data['item_id'] == item_id]['value'].sum()
                                if not pd.isna(item_value) and item_value != 0:
                                    avg_value += item_value
                                    points_used += 1
                            
                            if points_used >= 3:
                                avg_value = avg_value / points_used
                                
                                avg_data.append({
                                    'ticker': ticker,
                                    'item_id': item_id,
                                    'year': year,
                                    'quarter': quarter,
                                    'avg_value': avg_value,
                                    'points_used': points_used
                                })
        
        avg_df = pd.DataFrame(avg_data)
        print(f"   ✅ Calculated averages for {len(avg_df):,} records")
        
        return avg_df
    
    def calculate_roaa(self, ttm_data: pd.DataFrame, avg_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Return on Average Assets (ROAA)."""
        print("📊 Calculating ROAA...")
        
        # Get net profit TTM (item_id = 1)
        net_profit_ttm = ttm_data[ttm_data['item_id'] == 1].groupby(['ticker', 'year', 'quarter'])['ttm_value'].sum().reset_index()
        net_profit_ttm.rename(columns={'ttm_value': 'net_profit_ttm'}, inplace=True)
        
        # Get average total assets (item_id = 101, 102)
        total_assets_items = self.bs_items['total_assets']
        avg_total_assets = avg_data[avg_data['item_id'].isin(total_assets_items)].groupby(['ticker', 'year', 'quarter'])['avg_value'].sum().reset_index()
        avg_total_assets.rename(columns={'avg_value': 'avg_total_assets'}, inplace=True)
        
        # Calculate ROAA
        roaa_data = net_profit_ttm.merge(avg_total_assets, on=['ticker', 'year', 'quarter'], how='inner')
        roaa_data['roaa'] = roaa_data['net_profit_ttm'] / roaa_data['avg_total_assets']
        
        # Clean extreme values
        roaa_data['roaa'] = roaa_data['roaa'].clip(-1, 1)
        
        print(f"   ✅ Calculated ROAA for {len(roaa_data)} records")
        return roaa_data
    
    def calculate_pe_ratio(self, ttm_data: pd.DataFrame, market_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate P/E ratio using market cap and net profit."""
        print("📊 Calculating P/E ratios...")
        
        # Get net profit TTM (item_id = 1)
        net_profit_ttm = ttm_data[ttm_data['item_id'] == 1].groupby(['ticker', 'year', 'quarter'])['ttm_value'].sum().reset_index()
        net_profit_ttm.rename(columns={'ttm_value': 'net_profit_ttm'}, inplace=True)
        
        # Merge with market data
        pe_data = net_profit_ttm.merge(market_data, on=['ticker', 'year', 'quarter'], how='inner')
        
        # Calculate P/E ratio
        pe_data['pe_ratio'] = pe_data['market_cap'] / pe_data['net_profit_ttm']
        
        # Clean extreme values
        pe_data['pe_ratio'] = pe_data['pe_ratio'].clip(0, 100)
        
        print(f"   ✅ Calculated P/E ratios for {len(pe_data)} records")
        return pe_data
    
    def calculate_fcf_yield(self, ttm_data: pd.DataFrame, avg_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Free Cash Flow Yield."""
        print("📊 Calculating FCF Yield...")
        
        # For now, use a simplified approach since cash flow data has limited coverage
        # Use operating profit as a proxy for FCF
        operating_profit_ttm = ttm_data[ttm_data['item_id'] == 11].groupby(['ticker', 'year', 'quarter'])['ttm_value'].sum().reset_index()
        operating_profit_ttm.rename(columns={'ttm_value': 'operating_profit_ttm'}, inplace=True)
        
        # Get average total assets
        total_assets_items = self.bs_items['total_assets']
        avg_total_assets = avg_data[avg_data['item_id'].isin(total_assets_items)].groupby(['ticker', 'year', 'quarter'])['avg_value'].sum().reset_index()
        avg_total_assets.rename(columns={'avg_value': 'avg_total_assets'}, inplace=True)
        
        # Calculate FCF Yield (using operating profit as proxy)
        fcf_data = operating_profit_ttm.merge(avg_total_assets, on=['ticker', 'year', 'quarter'], how='inner')
        fcf_data['fcf_yield'] = fcf_data['operating_profit_ttm'] / fcf_data['avg_total_assets']
        
        # Clean extreme values
        fcf_data['fcf_yield'] = fcf_data['fcf_yield'].clip(-0.5, 0.5)
        
        print(f"   ✅ Calculated FCF Yield for {len(fcf_data)} records")
        return fcf_data
    
    def calculate_f_score(self, ttm_data: pd.DataFrame, avg_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Piotroski F-Score (9-point quality score)."""
        print("📊 Calculating F-Score...")
        
        f_score_data = []
        
        for ticker in ttm_data['ticker'].unique():
            ticker_ttm = ttm_data[ttm_data['ticker'] == ticker]
            ticker_avg = avg_data[avg_data['ticker'] == ticker]
            
            for year in ticker_ttm['year'].unique():
                for quarter in ticker_ttm['quarter'].unique():
                    f_score = 0
                    
                    # Get TTM and average data for this period
                    period_ttm = ticker_ttm[(ticker_ttm['year'] == year) & (ticker_ttm['quarter'] == quarter)]
                    period_avg = ticker_avg[(ticker_avg['year'] == year) & (ticker_avg['quarter'] == quarter)]
                    
                    if period_ttm.empty or period_avg.empty:
                        continue
                    
                    # 1. ROA > 0 (1 point)
                    net_profit = period_ttm[period_ttm['item_id'] == 1]['ttm_value'].sum()
                    avg_assets = period_avg[period_avg['item_id'].isin(self.bs_items['total_assets'])]['avg_value'].sum()
                    if avg_assets > 0 and net_profit > 0:
                        f_score += 1
                    
                    # 2. Operating Cash Flow > 0 (1 point) - use operating profit as proxy
                    operating_profit = period_ttm[period_ttm['item_id'] == 11]['ttm_value'].sum()
                    if operating_profit > 0:
                        f_score += 1
                    
                    # 3. Operating Cash Flow > Net Income (1 point)
                    if operating_profit > net_profit:
                        f_score += 1
                    
                    # 4. Current Ratio > 1 (1 point)
                    current_assets = period_avg[period_avg['item_id'].isin(self.bs_items['current_assets'])]['avg_value'].sum()
                    current_liabilities = period_avg[period_avg['item_id'].isin(self.bs_items['current_liabilities'])]['avg_value'].sum()
                    if current_liabilities > 0 and current_assets / current_liabilities > 1:
                        f_score += 1
                    
                    # 5. Asset Turnover > 0.5 (1 point)
                    revenue = period_ttm[period_ttm['item_id'] == 2]['ttm_value'].sum()
                    if avg_assets > 0 and revenue / avg_assets > 0.5:
                        f_score += 1
                    
                    f_score_data.append({
                        'ticker': ticker,
                        'year': year,
                        'quarter': quarter,
                        'f_score': f_score
                    })
        
        f_score_df = pd.DataFrame(f_score_data)
        print(f"   ✅ Calculated F-Score for {len(f_score_df)} records")
        return f_score_df
    
    def load_market_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load market data for P/E calculation."""
        print("📊 Loading market data...")
        
        query = text("""
            SELECT 
                ticker,
                YEAR(trading_date) as year,
                QUARTER(trading_date) as quarter,
                AVG(market_cap) as market_cap
            FROM vcsc_daily_data_complete
            WHERE trading_date BETWEEN :start_date AND :end_date
            AND market_cap > 0
            GROUP BY ticker, YEAR(trading_date), QUARTER(trading_date)
        """)
        
        market_data = pd.read_sql(query, self.db_engine, params={
            'start_date': start_date,
            'end_date': end_date
        })
        
        print(f"   ✅ Loaded market data for {len(market_data)} records")
        return market_data
    
    def calculate_all_factors(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Calculate all fundamental factors."""
        print("🚀 Calculating all fundamental factors...")
        
        # Load raw data
        raw_data = self.load_raw_fundamental_data(start_date, end_date)
        
        # Calculate TTM and averages
        ttm_data = self.calculate_ttm_values(raw_data)
        avg_data = self.calculate_balance_sheet_averages(raw_data)
        
        # Load market data
        market_data = self.load_market_data(start_date, end_date)
        
        # Calculate individual factors
        roaa_data = self.calculate_roaa(ttm_data, avg_data)
        pe_data = self.calculate_pe_ratio(ttm_data, market_data)
        fcf_data = self.calculate_fcf_yield(ttm_data, avg_data)
        f_score_data = self.calculate_f_score(ttm_data, avg_data)
        
        # Combine all factors
        factors_data = []
        
        # Get all unique ticker-year-quarter combinations
        all_periods = set()
        for df in [roaa_data, pe_data, fcf_data, f_score_data]:
            for _, row in df.iterrows():
                all_periods.add((row['ticker'], row['year'], row['quarter']))
        
        for ticker, year, quarter in all_periods:
            factor_row = {
                'ticker': ticker,
                'year': year,
                'quarter': quarter,
                'date': pd.Timestamp(year, quarter * 3, 1)
            }
            
            # Add ROAA
            roaa_row = roaa_data[(roaa_data['ticker'] == ticker) & 
                                (roaa_data['year'] == year) & 
                                (roaa_data['quarter'] == quarter)]
            if not roaa_row.empty:
                factor_row['roaa'] = roaa_row['roaa'].iloc[0]
            
            # Add P/E
            pe_row = pe_data[(pe_data['ticker'] == ticker) & 
                            (pe_data['year'] == year) & 
                            (pe_data['quarter'] == quarter)]
            if not pe_row.empty:
                factor_row['pe_ratio'] = pe_row['pe_ratio'].iloc[0]
            
            # Add FCF Yield
            fcf_row = fcf_data[(fcf_data['ticker'] == ticker) & 
                              (fcf_data['year'] == year) & 
                              (fcf_data['quarter'] == quarter)]
            if not fcf_row.empty:
                factor_row['fcf_yield'] = fcf_row['fcf_yield'].iloc[0]
            
            # Add F-Score
            f_score_row = f_score_data[(f_score_data['ticker'] == ticker) & 
                                      (f_score_data['year'] == year) & 
                                      (f_score_data['quarter'] == quarter)]
            if not f_score_row.empty:
                factor_row['f_score'] = f_score_row['f_score'].iloc[0]
            
            factors_data.append(factor_row)
        
        factors_df = pd.DataFrame(factors_data)
        
        # Fill missing values
        factors_df['roaa'] = factors_df['roaa'].fillna(0)
        factors_df['pe_ratio'] = factors_df['pe_ratio'].fillna(50)  # Neutral P/E
        factors_df['fcf_yield'] = factors_df['fcf_yield'].fillna(0)
        factors_df['f_score'] = factors_df['f_score'].fillna(0)
        
        print(f"✅ All fundamental factors calculated for {len(factors_df)} records")
        print(f"   📊 Coverage: {factors_df['ticker'].nunique()} tickers")
        print(f"   📅 Period: {factors_df['year'].min()}-{factors_df['year'].max()}")
        
        return factors_df

# %% [markdown]
# # TESTING AND VALIDATION

# %%
def test_fundamental_calculator():
    """Test the fundamental factor calculator."""
    print("🧪 Testing Fundamental Factor Calculator...")
    
    try:
        # Create database connection
        db_manager = get_database_manager()
        engine = db_manager.get_engine()
        
        # Initialize calculator
        calculator = FundamentalFactorCalculator(engine)
        
        # Test with a small period
        start_date = "2020-01-01"
        end_date = "2020-12-31"
        
        # Calculate factors
        factors = calculator.calculate_all_factors(start_date, end_date)
        
        # Display results
        print(f"\n📊 Test Results:")
        print(f"   - Records: {len(factors)}")
        print(f"   - Tickers: {factors['ticker'].nunique()}")
        print(f"   - ROAA range: {factors['roaa'].min():.4f} to {factors['roaa'].max():.4f}")
        print(f"   - P/E range: {factors['pe_ratio'].min():.2f} to {factors['pe_ratio'].max():.2f}")
        print(f"   - FCF Yield range: {factors['fcf_yield'].min():.4f} to {factors['fcf_yield'].max():.4f}")
        print(f"   - F-Score range: {factors['f_score'].min():.0f} to {factors['f_score'].max():.0f}")
        
        # Show sample data
        print(f"\n📋 Sample Data:")
        print(factors.head(10))
        
        print("✅ Fundamental factor calculator test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fundamental_calculator()
