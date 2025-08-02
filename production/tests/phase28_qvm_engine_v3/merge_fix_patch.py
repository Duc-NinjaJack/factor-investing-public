# Fix for MergeError in calculate_factors_for_date method
# Replace the existing method in 06_qvm_engine_v3e_optimized.py

def calculate_factors_for_date(self, date: pd.Timestamp, 
                             fundamental_data: pd.DataFrame,
                             momentum_data: dict,
                             universe_mask: pd.DataFrame) -> pd.DataFrame:
    """Calculate all factors for a specific date (optimized) - FIXED VERSION"""
    
    # Get fundamental data for the date
    date_fundamentals = fundamental_data[fundamental_data['date'] == date].copy()
    
    if date_fundamentals.empty:
        return pd.DataFrame()
    
    # Start with fundamental factors
    factors_df = date_fundamentals[['ticker', 'pe_ratio', 'pb_ratio', 'roe', 
                                  'debt_to_equity', 'current_ratio', 'quick_ratio', 
                                  'gross_margin', 'net_margin']].copy()
    
    # Add momentum factors (FIXED: avoid duplicate column conflicts)
    for key, momentum_series in momentum_data.items():
        # Get momentum data for the specific date
        momentum_subset = momentum_series.loc[date]
        
        # Reset index and select only ticker and momentum value columns
        momentum_df = momentum_subset.reset_index()
        # Select only ticker and the momentum value column, drop any date column
        momentum_df = momentum_df[['ticker', key]]
        
        # Merge on ticker only
        factors_df = factors_df.merge(momentum_df, on='ticker', how='left')
    
    # Calculate quality-adjusted P/E (vectorized)
    factors_df = self._calculate_quality_adjusted_pe(factors_df)
    
    # Calculate composite score (vectorized)
    factors_df = self._calculate_composite_score_vectorized(factors_df)
    
    # Apply universe filter
    if date in universe_mask.index:
        universe_subset = universe_mask.loc[date]
        universe_tickers = universe_subset[universe_subset].index.tolist()
        factors_df = factors_df[factors_df['ticker'].isin(universe_tickers)]
    
    return factors_df 