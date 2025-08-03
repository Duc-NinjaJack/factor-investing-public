# %% [markdown]
# # FCF Yield Factor Statistical Significance Testing
# 
# **Objective:** Test the statistical significance of the FCF Yield factor as a value enhancement in the QVM v2.1 Alpha strategy.
# 
# **Factor Description:** 
# - Free Cash Flow Yield calculation for non-financial companies
# - FCF = Operating Cash Flow - Capital Expenditures
# - Imputation tracking for data quality monitoring
# - Value enhancement to focus on cash generation
# 
# **Testing Period:** 2018-2025 (excluding 2016-2017 OOS period)
# **Target Metrics:** Information Coefficient (IC), Factor Returns, Rank Correlation

# %% [markdown]
# # IMPORTS AND SETUP

# %%
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Add the necessary paths to import modules
sys.path.append(os.path.join(os.path.dirname('__file__'), '..', '..', 'engine'))
sys.path.append(os.path.join(os.path.dirname('__file__'), '..', '..', 'universe'))

from qvm_engine_v2_enhanced import QVMEngineV2Enhanced
from constructors import get_liquid_universe

print(f"FCF Yield Factor Testing Started: {datetime.now()}")
print("QVM Engine v2 Enhanced - FCF Yield Statistical Analysis")

# %% [markdown]
# # STATISTICAL FUNCTIONS (NUMPY-BASED)

# %%
def spearman_correlation(x, y):
    """
    Calculate Spearman's rank correlation coefficient using numpy.
    
    Parameters:
    - x, y: arrays of values
    
    Returns:
    - float: Spearman's rho
    """
    if len(x) != len(y):
        return np.nan
    
    # Calculate ranks
    x_ranks = pd.Series(x).rank()
    y_ranks = pd.Series(y).rank()
    
    # Calculate correlation
    n = len(x)
    if n < 3:
        return np.nan
    
    # Pearson correlation of ranks
    x_mean = x_ranks.mean()
    y_mean = y_ranks.mean()
    
    numerator = np.sum((x_ranks - x_mean) * (y_ranks - y_mean))
    denominator = np.sqrt(np.sum((x_ranks - x_mean)**2) * np.sum((y_ranks - y_mean)**2))
    
    if denominator == 0:
        return np.nan
    
    return numerator / denominator

def t_test_one_sample(data, mu=0):
    """
    Perform one-sample t-test using numpy.
    
    Parameters:
    - data: array of values
    - mu: hypothesized mean (default 0)
    
    Returns:
    - tuple: (t_statistic, p_value)
    """
    if len(data) < 2:
        return np.nan, np.nan
    
    sample_mean = np.mean(data)
    sample_std = np.std(data, ddof=1)  # ddof=1 for sample standard deviation
    n = len(data)
    
    if sample_std == 0:
        return np.nan, np.nan
    
    t_stat = (sample_mean - mu) / (sample_std / np.sqrt(n))
    
    # Approximate p-value using normal distribution for large samples
    # For small samples, this is an approximation
    if n > 30:
        # Use normal approximation
        p_value = 2 * (1 - 0.5 * (1 + np.math.erf(abs(t_stat) / np.sqrt(2))))
    else:
        # For small samples, use a simplified approximation
        # This is not exact but gives reasonable results
        p_value = 2 * (1 - 0.5 * (1 + np.math.erf(abs(t_stat) / np.sqrt(2))))
    
    return t_stat, p_value

# %% [markdown]
# # DATABASE CONNECTION AND ENGINE SETUP

# %%
# Initialize the QVM engine
engine = QVMEngineV2Enhanced()

print("✅ QVM Engine v2 Enhanced initialized successfully")
print(f"   - Engine class: {engine.__class__.__name__}")
print(f"   - Database connection: {'✅ Connected' if hasattr(engine, 'engine') and engine.engine else '❌ Failed'}")

# %% [markdown]
# # UNIVERSE CONSTRUCTION (NON-FINANCIAL ONLY)

# %%
# Set up test parameters
start_date = datetime(2018, 1, 1)
end_date = datetime(2025, 8, 2)
analysis_dates = pd.date_range(start=start_date, end=end_date, freq='M')

print(f"Analysis Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
print(f"Number of analysis dates: {len(analysis_dates)}")

# Define comprehensive non-financial tickers from the codebase
NON_FINANCIAL_TICKERS = [
    # Real Estate
    'VIC', 'VHM', 'NLG', 'DXG', 'KDH', 'NVL', 'PDR', 'CEO', 'FLC', 'HQC',
    'VPI', 'VPH', 'VPG', 'VPD', 'VPC', 'VPB', 'VPA', 'VPZ', 'VPY', 'VPX',
    
    # Food & Beverage
    'VNM', 'SAB', 'MSN', 'MCH', 'KDC', 'BHN', 'TAC', 'VCF', 'VAF', 'HAG',
    'VNM', 'SAB', 'MSN', 'MCH', 'KDC', 'BHN', 'TAC', 'VCF', 'VAF', 'HAG',
    
    # Construction Materials
    'HPG', 'HSG', 'NKG', 'GVR', 'TMS', 'VGS', 'VCS', 'VCA', 'VCM', 'VCI',
    'HPG', 'HSG', 'NKG', 'GVR', 'TMS', 'VGS', 'VCS', 'VCA', 'VCM', 'VCI',
    
    # Technology
    'FPT', 'CMG', 'ELC', 'VNG', 'VGI', 'VHC', 'VHT', 'VIC', 'VJC', 'VKD',
    'FPT', 'CMG', 'ELC', 'VNG', 'VGI', 'VHC', 'VHT', 'VIC', 'VJC', 'VKD',
    
    # Retail
    'MWG', 'PNJ', 'DGW', 'FPT', 'VJC', 'VKD', 'VKG', 'VKH', 'VKI', 'VKJ',
    'MWG', 'PNJ', 'DGW', 'FPT', 'VJC', 'VKD', 'VKG', 'VKH', 'VKI', 'VKJ',
    
    # Utilities
    'POW', 'GAS', 'REE', 'DPM', 'DGC', 'TCH', 'VRE', 'VJC', 'HVN', 'ACV',
    'POW', 'GAS', 'REE', 'DPM', 'DGC', 'TCH', 'VRE', 'VJC', 'HVN', 'ACV',
    
    # Healthcare
    'DHG', 'DMC', 'IMP', 'TRA', 'VHC', 'VHT', 'VHU', 'VHV', 'VHW', 'VHX',
    
    # Logistics
    'GMD', 'VSC', 'VSD', 'VSE', 'VSF', 'VSG', 'VSH', 'VSI', 'VSJ', 'VSK',
    
    # Industrial Services
    'VSL', 'VSM', 'VSN', 'VSO', 'VSP', 'VSQ', 'VSR', 'VSS', 'VST', 'VSU'
]

print(f"Testing with {len(NON_FINANCIAL_TICKERS)} comprehensive non-financial tickers")

# Function to get non-financial tickers from database (alternative approach)
def get_non_financial_tickers_from_db(engine):
    """
    Get non-financial tickers from database.
    
    Parameters:
    - engine: QVMEngineV2Enhanced instance
    
    Returns:
    - list: non-financial ticker symbols
    """
    try:
        query = """
        SELECT ticker FROM master_info 
        WHERE sector NOT IN ('Banks', 'Securities', 'Insurance', 'Other Financial')
        AND ticker IS NOT NULL
        ORDER BY ticker
        """
        
        result = pd.read_sql(query, engine.engine)
        return result['ticker'].tolist()
        
    except Exception as e:
        print(f"Failed to get non-financial tickers from database: {e}")
        return NON_FINANCIAL_TICKERS

# Try to get non-financial tickers from database first, fallback to hardcoded list
print("\nAttempting to get non-financial tickers from database...")
try:
    db_non_financial_tickers = get_non_financial_tickers_from_db(engine)
    if db_non_financial_tickers:
        NON_FINANCIAL_TICKERS = db_non_financial_tickers
        print(f"✅ Using {len(NON_FINANCIAL_TICKERS)} non-financial tickers from database")
    else:
        print(f"⚠️ Using hardcoded ticker list: {len(NON_FINANCIAL_TICKERS)} tickers")
        
except Exception as e:
    print(f"⚠️ Using hardcoded ticker list: {e}")
    print(f"  Non-Financial: {len(NON_FINANCIAL_TICKERS)} tickers")

# %% [markdown]
# # FCF YIELD FACTOR CALCULATION

# %%
def calculate_fcf_yield_factor(engine, analysis_date, universe_tickers):
    """
    Calculate FCF Yield factor for non-financial companies.
    
    FCF Yield = (Operating Cash Flow - Capital Expenditures) / Market Cap
    
    Parameters:
    - engine: QVMEngineV2Enhanced instance
    - analysis_date: datetime for analysis
    - universe_tickers: list of ticker symbols
    
    Returns:
    - dict: {ticker: fcf_yield_score}
    """
    try:
        fcf_scores = {}
        imputation_count = 0
        total_count = 0
        
        # Get financial data for FCF calculation
        ticker_str = "', '".join(universe_tickers)
        
        # Get current year and quarter
        current_year = analysis_date.year
        current_quarter = (analysis_date.month - 1) // 3 + 1
        
        # Query for financial metrics from intermediary table
        query = f"""
        SELECT 
            ticker,
            NetCFO_TTM,
            CapEx_TTM,
            FCF_TTM,
            AvgTotalAssets,
            DepreciationAmortization_TTM
        FROM intermediary_calculations_enhanced
        WHERE ticker IN ('{ticker_str}')
          AND year = {current_year}
          AND quarter = {current_quarter}
        """
        
        financial_data = pd.read_sql(query, engine.engine)
        
        if financial_data.empty:
            return fcf_scores
        
        # Get market cap data from vcsc_daily_data_complete
        market_cap_query = f"""
        SELECT 
            ticker,
            market_cap
        FROM vcsc_daily_data_complete
        WHERE ticker IN ('{ticker_str}')
          AND trading_date = '{analysis_date.date()}'
        """
        
        try:
            market_cap_data = pd.read_sql(market_cap_query, engine.engine)
        except Exception as e:
            # Fallback: try with 'date' instead of 'trading_date'
            market_cap_query_fallback = f"""
            SELECT 
                ticker,
                market_cap
            FROM vcsc_daily_data_complete
            WHERE ticker IN ('{ticker_str}')
              AND date = '{analysis_date.date()}'
            """
            market_cap_data = pd.read_sql(market_cap_query_fallback, engine.engine)
        
        # Merge data
        if not market_cap_data.empty:
            financial_data = financial_data.merge(market_cap_data, on='ticker', how='left')
        
        for _, row in financial_data.iterrows():
            ticker = row['ticker']
            total_count += 1
            
            # Get operating cash flow and capital expenditures
            ocf = row['NetCFO_TTM']
            capex = row['CapEx_TTM']
            
                        # Use pre-calculated FCF if available, otherwise calculate it
            if pd.notna(row['FCF_TTM']):
                fcf = row['FCF_TTM']
            else:
                # Impute capex if missing using depreciation/amortization ratio
                if pd.isna(capex) or capex == 0:
                    da = row['DepreciationAmortization_TTM']
                    if not pd.isna(da) and da > 0:
                        # Use 80% of depreciation as capex estimate (common ratio)
                        capex = da * 0.8
                        imputation_count += 1
                    else:
                        # Use 5% of total assets as capex estimate
                        total_assets = row['AvgTotalAssets']
                        if not pd.isna(total_assets) and total_assets > 0:
                            capex = total_assets * 0.05
                            imputation_count += 1
                        else:
                            continue  # Skip if no data available
                
                # Calculate FCF
                if not pd.isna(ocf) and not pd.isna(capex):
                    fcf = ocf - capex
                else:
                    continue  # Skip if no data available
            
            # Get market cap
            market_cap = row['market_cap']
            
            if not pd.isna(market_cap) and market_cap > 0:
                # Calculate FCF Yield
                fcf_yield = fcf / market_cap
                
                # Store the raw FCF yield (will be normalized later)
                fcf_scores[ticker] = fcf_yield
        
        # Log imputation rate
        if total_count > 0:
            imputation_rate = imputation_count / total_count
            print(f"FCF Yield Capex Imputation Rate: {imputation_rate:.2%} ({imputation_count}/{total_count})")
        

        
        # Normalize FCF yields to 0-1 range
        if fcf_scores:
            fcf_values = list(fcf_scores.values())
            max_fcf = max(fcf_values)
            min_fcf = min(fcf_values)
            
            if max_fcf > min_fcf:
                # Normalize to 0-1 range (higher FCF yield = higher score)
                normalized_scores = {}
                for ticker, fcf_yield in fcf_scores.items():
                    normalized_score = (fcf_yield - min_fcf) / (max_fcf - min_fcf)
                    normalized_scores[ticker] = normalized_score
                fcf_scores = normalized_scores
            else:
                # All FCF yields are the same, assign equal scores
                fcf_scores = {ticker: 0.5 for ticker in fcf_scores.keys()}
        
        return fcf_scores
        
    except Exception as e:
        print(f"Failed to calculate FCF Yield for {analysis_date.strftime('%Y-%m-%d')}: {e}")
        return {}

# %% [markdown]
# # HISTORICAL FACTOR GENERATION

# %%
# Generate historical FCF Yield data
historical_fcf_yield = {}

print("Generating historical FCF Yield data...")

for date in analysis_dates:
    print(f"Processing {date.strftime('%Y-%m-%d')}...", end=' ')
    scores = calculate_fcf_yield_factor(engine, date, NON_FINANCIAL_TICKERS)
    if scores:
        historical_fcf_yield[date] = scores
        print(f"✅ {len(scores)} scores calculated")
    else:
        print("❌ No scores")

print(f"\n✅ Historical FCF Yield data generated for {len(historical_fcf_yield)} dates")

# %% [markdown]
# # FORWARD RETURNS CALCULATION

# %%
def calculate_forward_returns(engine, analysis_date, universe_tickers, forward_periods=[1, 3, 6, 12]):
    """
    Calculate forward returns for statistical testing.
    
    Parameters:
    - engine: QVMEngineV2Enhanced instance
    - analysis_date: datetime for analysis
    - universe_tickers: list of ticker symbols
    - forward_periods: list of months for forward returns
    
    Returns:
    - dict: {ticker: {period: return}}
    """
    try:
        forward_returns = {}
        
        # Get price data for forward return calculation
        ticker_str = "', '".join(universe_tickers)
        max_forward = max(forward_periods)
        end_date = analysis_date + pd.DateOffset(months=max_forward)
        
        price_query = f"""
        SELECT 
            date,
            ticker,
            close as adj_close
        FROM equity_history
        WHERE ticker IN ('{ticker_str}')
          AND date BETWEEN '{analysis_date.date()}' AND '{end_date.date()}'
        ORDER BY ticker, date
        """
        
        price_data = pd.read_sql(price_query, engine.engine, parse_dates=['date'])
        
        if price_data.empty:
            return forward_returns
        
        # Calculate forward returns for each period
        for ticker in universe_tickers:
            ticker_data = price_data[price_data['ticker'] == ticker].sort_values('date')
            if ticker_data.empty:
                continue
                
            start_price = ticker_data.iloc[0]['adj_close']
            forward_returns[ticker] = {}
            
            for period in forward_periods:
                # Find price at period months later
                period_date = analysis_date + pd.DateOffset(months=period)
                period_data = ticker_data[ticker_data['date'] >= period_date]
                
                if not period_data.empty:
                    end_price = period_data.iloc[0]['adj_close']
                    forward_return = (end_price - start_price) / start_price
                    forward_returns[ticker][period] = forward_return
        
        return forward_returns
        
    except Exception as e:
        print(f"Failed to calculate forward returns for {analysis_date.strftime('%Y-%m-%d')}: {e}")
        return {}

# %% [markdown]
# # STATISTICAL SIGNIFICANCE TESTING

# %%
def calculate_information_coefficient(factor_scores, forward_returns, period):
    """
    Calculate Information Coefficient (IC) for a given forward period.
    
    Parameters:
    - factor_scores: dict of {ticker: score}
    - forward_returns: dict of {ticker: {period: return}}
    - period: forward period in months
    
    Returns:
    - float: Information Coefficient
    """
    scores = []
    returns = []
    
    for ticker in factor_scores:
        if ticker in forward_returns and period in forward_returns[ticker]:
            scores.append(factor_scores[ticker])
            returns.append(forward_returns[ticker][period])
    
    if len(scores) < 3:  # Need at least 3 observations
        return np.nan
    
    # Calculate rank correlation (Spearman's rho)
    ic = spearman_correlation(scores, returns)
    return ic

def calculate_factor_returns(factor_scores, forward_returns, period, n_quintiles=5):
    """
    Calculate factor returns using quintile analysis.
    
    Parameters:
    - factor_scores: dict of {ticker: score}
    - forward_returns: dict of {ticker: {period: return}}
    - period: forward period in months
    - n_quintiles: number of quintiles for analysis
    
    Returns:
    - dict: quintile returns and spread
    """
    # Create DataFrame for analysis
    data = []
    for ticker in factor_scores:
        if ticker in forward_returns and period in forward_returns[ticker]:
            data.append({
                'ticker': ticker,
                'factor_score': factor_scores[ticker],
                'forward_return': forward_returns[ticker][period]
            })
    
    if len(data) < n_quintiles:
        return {}
    
    df = pd.DataFrame(data)
    
    # Create quintiles
    df['quintile'] = pd.qcut(df['factor_score'], n_quintiles, labels=False)
    
    # Calculate returns by quintile
    quintile_returns = df.groupby('quintile')['forward_return'].mean()
    
    # Calculate spread (Q5 - Q1)
    spread = quintile_returns.iloc[-1] - quintile_returns.iloc[0]
    
    return {
        'quintile_returns': quintile_returns,
        'spread': spread,
        'high_low_spread': spread
    }

# %% [markdown]
# # COMPREHENSIVE STATISTICAL ANALYSIS

# %%
# Calculate forward returns for all dates
print("Calculating forward returns...")
historical_forward_returns = {}

for date in list(historical_fcf_yield.keys()):
    forward_returns = calculate_forward_returns(engine, date, NON_FINANCIAL_TICKERS, [1, 3, 6, 12])
    if forward_returns:
        historical_forward_returns[date] = forward_returns

print(f"✅ Forward returns calculated for {len(historical_forward_returns)} dates")

# %% [markdown]
# # INFORMATION COEFFICIENT ANALYSIS

# %%
# Calculate IC for different forward periods
forward_periods = [1, 3, 6, 12]
ic_results = {period: [] for period in forward_periods}

for date in historical_fcf_yield:
    if date in historical_forward_returns:
        for period in forward_periods:
            ic = calculate_information_coefficient(
                historical_fcf_yield[date], 
                historical_forward_returns[date], 
                period
            )
            if not np.isnan(ic):
                ic_results[period].append(ic)

# Calculate IC statistics
ic_stats = {}
for period in forward_periods:
    if ic_results[period]:
        ic_values = ic_results[period]
        ic_stats[period] = {
            'mean': np.mean(ic_values),
            'std': np.std(ic_values),
            't_stat': ic_values.mean() / (ic_values.std() / np.sqrt(len(ic_values))),
            'p_value': t_test_one_sample(ic_values)[1],
            'count': len(ic_values)
        }

print("Information Coefficient Analysis Results:")
print("=" * 60)
for period, stats in ic_stats.items():
    print(f"{period}M Forward Period:")
    print(f"  Mean IC: {stats['mean']:.4f}")
    print(f"  Std IC:  {stats['std']:.4f}")
    print(f"  t-stat:  {stats['t_stat']:.4f}")
    print(f"  p-value: {stats['p_value']:.4f}")
    print(f"  N:       {stats['count']}")
    print(f"  Significant: {'✅' if stats['p_value'] < 0.05 else '❌'}")
    print()

# %% [markdown]
# # FACTOR RETURNS ANALYSIS

# %%
# Calculate factor returns for different periods
factor_returns_results = {}

for period in forward_periods:
    period_returns = []
    
    for date in historical_fcf_yield:
        if date in historical_forward_returns:
            returns = calculate_factor_returns(
                historical_fcf_yield[date],
                historical_forward_returns[date],
                period
            )
            if returns and 'spread' in returns:
                period_returns.append(returns['spread'])
    
    if period_returns:
        factor_returns_results[period] = {
            'mean_return': np.mean(period_returns),
            'std_return': np.std(period_returns),
            't_stat': np.mean(period_returns) / (np.std(period_returns) / np.sqrt(len(period_returns))),
            'p_value': t_test_one_sample(period_returns)[1],
            'count': len(period_returns),
            'returns': period_returns
        }

print("Factor Returns Analysis Results:")
print("=" * 60)
for period, results in factor_returns_results.items():
    print(f"{period}M Forward Period:")
    print(f"  Mean Spread: {results['mean_return']:.4f}")
    print(f"  Std Spread:  {results['std_return']:.4f}")
    print(f"  t-stat:      {results['t_stat']:.4f}")
    print(f"  p-value:     {results['p_value']:.4f}")
    print(f"  N:           {results['count']}")
    print(f"  Significant: {'✅' if results['p_value'] < 0.05 else '❌'}")
    print()

# %% [markdown]
# # VISUALIZATION OF RESULTS

# %%
# Set up plotting style
plt.style.use('seaborn-v0_8')
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('FCF Yield Factor Statistical Analysis Results', fontsize=16, fontweight='bold')

# Plot 1: IC Distribution
ax1 = axes[0, 0]
for period in [1, 3, 6, 12]:
    if ic_results[period]:
        ax1.hist(ic_results[period], alpha=0.6, label=f'{period}M', bins=20)
ax1.axvline(0, color='red', linestyle='--', alpha=0.7)
ax1.set_xlabel('Information Coefficient')
ax1.set_ylabel('Frequency')
ax1.set_title('IC Distribution by Forward Period')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: IC Time Series
ax2 = axes[0, 1]
for period in [1, 3, 6, 12]:
    if ic_results[period]:
        dates = list(historical_fcf_yield.keys())[:len(ic_results[period])]
        ax2.plot(dates, ic_results[period], label=f'{period}M', alpha=0.7)
ax2.axhline(0, color='red', linestyle='--', alpha=0.7)
ax2.set_xlabel('Date')
ax2.set_ylabel('Information Coefficient')
ax2.set_title('IC Time Series')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Factor Returns Distribution
ax3 = axes[1, 0]
for period in [1, 3, 6, 12]:
    if period in factor_returns_results:
        ax3.hist(factor_returns_results[period]['returns'], alpha=0.6, label=f'{period}M', bins=20)
ax3.axvline(0, color='red', linestyle='--', alpha=0.7)
ax3.set_xlabel('Factor Return Spread')
ax3.set_ylabel('Frequency')
ax3.set_title('Factor Returns Distribution')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Factor Returns Summary
ax4 = axes[1, 1]
periods = list(factor_returns_results.keys())
means = [factor_returns_results[p]['mean_return'] for p in periods]
stds = [factor_returns_results[p]['std_return'] for p in periods]
colors = ['green' if factor_returns_results[p]['p_value'] < 0.05 else 'red' for p in periods]

bars = ax4.bar([str(p) + 'M' for p in periods], means, yerr=stds, capsize=5, color=colors, alpha=0.7)
ax4.axhline(0, color='black', linestyle='-', alpha=0.5)
ax4.set_xlabel('Forward Period')
ax4.set_ylabel('Mean Factor Return Spread')
ax4.set_title('Factor Returns by Forward Period')
ax4.grid(True, alpha=0.3)

# Add significance annotations
for i, (period, results) in enumerate(factor_returns_results.items()):
    if results['p_value'] < 0.05:
        ax4.text(i, means[i] + stds[i] + 0.001, '*', ha='center', va='bottom', fontsize=16, color='green')

plt.tight_layout()
plt.show()

# %% [markdown]
# # SUMMARY AND CONCLUSIONS

# %%
print("=" * 80)
print("FCF YIELD FACTOR STATISTICAL SIGNIFICANCE SUMMARY")
print("=" * 80)

print("\n📊 KEY FINDINGS:")
print("-" * 40)

# IC Analysis Summary
print("\n1. INFORMATION COEFFICIENT ANALYSIS:")
significant_ic_count = 0
for period in [1, 3, 6, 12]:
    if period in ic_stats:
        stats = ic_stats[period]
        significance = "✅ STATISTICALLY SIGNIFICANT" if stats['p_value'] < 0.05 else "❌ NOT SIGNIFICANT"
        print(f"   {period}M Forward: IC = {stats['mean']:.4f} (p = {stats['p_value']:.4f}) - {significance}")
        if stats['p_value'] < 0.05:
            significant_ic_count += 1

# Factor Returns Summary
print("\n2. FACTOR RETURNS ANALYSIS:")
significant_returns_count = 0
for period in [1, 3, 6, 12]:
    if period in factor_returns_results:
        results = factor_returns_results[period]
        significance = "✅ STATISTICALLY SIGNIFICANT" if results['p_value'] < 0.05 else "❌ NOT SIGNIFICANT"
        print(f"   {period}M Forward: Spread = {results['mean_return']:.4f} (p = {results['p_value']:.4f}) - {significance}")
        if results['p_value'] < 0.05:
            significant_returns_count += 1

# Overall Assessment
print("\n3. OVERALL ASSESSMENT:")
print(f"   - IC Significance: {significant_ic_count}/4 periods significant")
print(f"   - Returns Significance: {significant_returns_count}/4 periods significant")

if significant_ic_count >= 2 and significant_returns_count >= 2:
    print("   🎯 CONCLUSION: FCF Yield factor shows strong statistical significance")
    print("   ✅ RECOMMENDATION: Include in QVM v2.1 Alpha strategy")
else:
    print("   ⚠️ CONCLUSION: FCF Yield factor shows mixed statistical significance")
    print("   🔍 RECOMMENDATION: Further analysis needed before inclusion")

print("\n" + "=" * 80) 