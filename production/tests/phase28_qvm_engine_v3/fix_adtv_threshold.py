#!/usr/bin/env python3
"""
Fix ADTV threshold to use Vietnam Dong (VND) instead of shares.
"""

def fix_adtv_threshold():
    """Fix the ADTV threshold configuration and SQL query."""
    
    print("🔧 FIXING ADTV THRESHOLD TO USE VND")
    print("="*50)
    
    print("📊 CURRENT ISSUE:")
    print("   - ADTV threshold is currently set to 1,000,000 shares")
    print("   - This should be in Vietnam Dong (VND) for proper liquidity filtering")
    print("   - ADTV = Average Daily Trading Volume in currency terms")
    
    print("\n✅ FIXED CONFIGURATION:")
    print("   Change from:")
    print("   'adtv_threshold_shares': 1000000,  # 1 million shares")
    print("   To:")
    print("   'adtv_threshold_vnd': 10000000000,  # 10 billion VND")
    
    print("\n🔧 FIXED SQL QUERY:")
    print("   Change from:")
    print("   AVG(total_volume) as avg_volume")
    print("   To:")
    print("   AVG(total_volume * close_price_adjusted) as avg_adtv_vnd")
    
    print("\n📈 REASONABLE VND THRESHOLDS FOR VIETNAMESE STOCKS:")
    print("   - Small cap: 5 billion VND/day")
    print("   - Mid cap: 10 billion VND/day") 
    print("   - Large cap: 20 billion VND/day")
    print("   - Very liquid: 50 billion VND/day")
    
    return {
        "old_config": {
            "adtv_threshold_shares": 1000000
        },
        "new_config": {
            "adtv_threshold_vnd": 10000000000  # 10 billion VND
        }
    }

def create_fixed_code():
    """Create the fixed code snippets."""
    
    print("\n🔧 FIXED CODE SNIPPETS:")
    print("="*50)
    
    # Fixed configuration
    config_fix = '''
# --- FIXED CONFIGURATION ---
"universe": {
    "lookback_days": 60,
    "adtv_threshold_vnd": 10000000000,  # 10 billion VND (was 1M shares)
    "min_market_cap_bn": 1.0,
    "target_portfolio_size": 25
},
'''
    
    # Fixed SQL query
    sql_fix = '''
# --- FIXED SQL QUERY ---
universe_query = text("""
    SELECT 
        ticker,
        AVG(total_volume * close_price_adjusted) as avg_adtv_vnd,
        AVG(market_cap) as avg_market_cap
    FROM vcsc_daily_data_complete
    WHERE trading_date <= :analysis_date
      AND trading_date >= DATE_SUB(:analysis_date, INTERVAL :lookback_days DAY)
    GROUP BY ticker
    HAVING avg_adtv_vnd >= :adtv_threshold AND avg_market_cap >= :min_market_cap
""")
'''
    
    # Fixed method
    method_fix = '''
# --- FIXED _get_universe METHOD ---
def _get_universe(self, analysis_date: pd.Timestamp) -> list:
    """Get liquid universe based on ADTV and market cap filters."""
    lookback_days = self.config['universe']['lookback_days']
    adtv_threshold = self.config['universe']['adtv_threshold_vnd']  # Now in VND
    min_market_cap = self.config['universe']['min_market_cap_bn'] * 1e9
    
    # Get universe data with ADTV in VND
    universe_query = text("""
        SELECT 
            ticker,
            AVG(total_volume * close_price_adjusted) as avg_adtv_vnd,
            AVG(market_cap) as avg_market_cap
        FROM vcsc_daily_data_complete
        WHERE trading_date <= :analysis_date
          AND trading_date >= DATE_SUB(:analysis_date, INTERVAL :lookback_days DAY)
        GROUP BY ticker
        HAVING avg_adtv_vnd >= :adtv_threshold AND avg_market_cap >= :min_market_cap
    """)
    
    universe_df = pd.read_sql(universe_query, self.engine, 
                             params={'analysis_date': analysis_date, 'lookback_days': lookback_days, 'adtv_threshold': adtv_threshold, 'min_market_cap': min_market_cap})
    
    return universe_df['ticker'].tolist()
'''
    
    print(config_fix)
    print(sql_fix)
    print(method_fix)
    
    return {
        "config_fix": config_fix,
        "sql_fix": sql_fix,
        "method_fix": method_fix
    }

def explain_benefits():
    """Explain the benefits of using VND for ADTV."""
    
    print("\n🎯 BENEFITS OF USING VND FOR ADTV:")
    print("="*50)
    
    benefits = [
        "1. **Proper Liquidity Measurement**: ADTV in VND measures actual trading value, not just share count",
        "2. **Price-Aware Filtering**: Accounts for both volume and price, filtering out low-value high-volume stocks",
        "3. **Market Standard**: Most institutional investors use currency-based liquidity measures",
        "4. **Better Risk Management**: Prevents investing in stocks with low trading value despite high share volume",
        "5. **Consistent with Market Cap**: Both ADTV and market cap are in currency terms"
    ]
    
    for benefit in benefits:
        print(f"   {benefit}")
    
    print("\n📊 EXAMPLE:")
    print("   Stock A: 1M shares traded at 10,000 VND = 10B VND ADTV")
    print("   Stock B: 1M shares traded at 1,000 VND = 1B VND ADTV")
    print("   With 10B VND threshold: Stock A passes, Stock B fails")
    print("   This is more meaningful than just counting shares!")

def suggest_threshold_values():
    """Suggest different threshold values for different strategies."""
    
    print("\n💡 SUGGESTED THRESHOLD VALUES:")
    print("="*50)
    
    thresholds = {
        "Conservative": 20000000000,  # 20B VND - very liquid stocks only
        "Balanced": 10000000000,      # 10B VND - good liquidity (recommended)
        "Aggressive": 5000000000,     # 5B VND - more stocks included
        "Very Aggressive": 2000000000 # 2B VND - maximum universe
    }
    
    for strategy, threshold in thresholds.items():
        print(f"   {strategy}: {threshold:,} VND ({threshold/1e9:.1f}B VND)")
    
    print("\n📈 RECOMMENDATION:")
    print("   Start with 10B VND (Balanced) and adjust based on:")
    print("   - Desired universe size")
    print("   - Liquidity requirements")
    print("   - Market conditions")

if __name__ == "__main__":
    # Run the fix analysis
    configs = fix_adtv_threshold()
    code_snippets = create_fixed_code()
    explain_benefits()
    suggest_threshold_values()
    
    print("\n🎯 SUMMARY:")
    print("   The ADTV threshold should be changed from 1M shares to 10B VND")
    print("   This provides better liquidity filtering and is more meaningful")
    print("   for institutional investment strategies.") 