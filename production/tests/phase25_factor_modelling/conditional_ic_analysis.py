"""
Conditional IC Analysis
======================
Analyzes the conditional Information Coefficient (IC) of value score
across different quality quintiles based on ROAA.
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

class ConditionalICAnalyzer:
    def __init__(self):
        self.engine = get_engine()
        
    def get_comprehensive_data(self, analysis_date: datetime, universe: list) -> pd.DataFrame:
        """Get comprehensive data including all factors and forward returns."""
        # Get basic factors
        factors_query = """
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
            factors = pd.read_sql(factors_query, self.engine, params=tuple(params))
            factors['sector'] = 'Banking'
        except Exception as e:
            print(f"Error extracting factors: {e}")
            return pd.DataFrame()
        
        # Get momentum factors
        momentum_data = []
        for ticker in universe:
            try:
                price_query = """
                SELECT trading_date, close_price 
                FROM vcsc_daily_data_complete 
                WHERE ticker = %s 
                AND trading_date <= %s 
                ORDER BY trading_date DESC 
                LIMIT 252
                """
                
                price_data = pd.read_sql(price_query, self.engine, params=(ticker, analysis_date))
                
                if len(price_data) >= 30:
                    current_price = price_data.iloc[0]['close_price']
                    
                    momentum_1m = 0
                    momentum_3m = 0
                    momentum_6m = 0
                    momentum_12m = 0
                    
                    if len(price_data) >= 21:
                        price_1m = price_data.iloc[20]['close_price']
                        momentum_1m = (current_price - price_1m) / price_1m if price_1m > 0 else 0
                    
                    if len(price_data) >= 63:
                        price_3m = price_data.iloc[62]['close_price']
                        momentum_3m = (current_price - price_3m) / price_3m if price_3m > 0 else 0
                    
                    if len(price_data) >= 126:
                        price_6m = price_data.iloc[125]['close_price']
                        momentum_6m = (current_price - price_6m) / price_6m if price_6m > 0 else 0
                    
                    if len(price_data) >= 252:
                        price_12m = price_data.iloc[251]['close_price']
                        momentum_12m = (current_price - price_12m) / price_12m if price_12m > 0 else 0
                    
                    momentum_data.append({
                        'ticker': ticker,
                        '1M': momentum_1m,
                        '3M': momentum_3m,
                        '6M': momentum_6m,
                        '12M': momentum_12m
                    })
                else:
                    momentum_data.append({
                        'ticker': ticker,
                        '1M': 0, '3M': 0, '6M': 0, '12M': 0
                    })
                    
            except Exception as e:
                print(f"Error calculating momentum for {ticker}: {e}")
                momentum_data.append({
                    'ticker': ticker,
                    '1M': 0, '3M': 0, '6M': 0, '12M': 0
                })
        
        momentum = pd.DataFrame(momentum_data)
        
        # Get value factors
        value_data = []
        for ticker in universe:
            try:
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
                    
                    market_cap_query = """
                    SELECT trading_date, close_price, market_cap, total_shares
                    FROM vcsc_daily_data_complete
                    WHERE ticker = %s AND trading_date <= %s
                    ORDER BY trading_date DESC
                    LIMIT 1
                    """
                    
                    market_data = pd.read_sql(market_cap_query, self.engine, params=(ticker, analysis_date))
                    
                    if len(market_data) > 0:
                        market_cap = market_data.iloc[0]['market_cap']
                        close_price = market_data.iloc[0]['close_price']
                        total_shares = market_data.iloc[0]['total_shares']
                    else:
                        market_cap = 0
                        close_price = 0
                        total_shares = 0
                    
                    pe_score = 0
                    pb_score = 0
                    ps_score = 0
                    
                    if pd.notna(row['NetProfit_TTM']) and row['NetProfit_TTM'] > 0 and market_cap > 0:
                        pe_ratio = market_cap / row['NetProfit_TTM']
                        pe_score = 1 / pe_ratio if pe_ratio > 0 else 0
                    
                    if pd.notna(row['AvgTotalEquity']) and row['AvgTotalEquity'] > 0 and market_cap > 0:
                        pb_ratio = market_cap / row['AvgTotalEquity']
                        pb_score = 1 / pb_ratio if pb_ratio > 0 else 0
                    
                    if pd.notna(row['TotalOperatingIncome_TTM']) and row['TotalOperatingIncome_TTM'] > 0 and market_cap > 0:
                        ps_ratio = market_cap / row['TotalOperatingIncome_TTM']
                        ps_score = 1 / ps_ratio if ps_ratio > 0 else 0
                    
                    value_score = 0.6 * pe_score + 0.4 * pb_score
                    
                else:
                    value_score = 0
                    pe_score = 0
                    pb_score = 0
                    ps_score = 0
                    market_cap = 0
                    close_price = 0
                    total_shares = 0
                    
                value_data.append({
                    'ticker': ticker,
                    'value_score': value_score,
                    'pe_score': pe_score,
                    'pb_score': pb_score,
                    'ps_score': ps_score,
                    'market_cap': market_cap,
                    'close_price': close_price,
                    'total_shares': total_shares
                })
                
            except Exception as e:
                print(f"Error calculating value factors for {ticker}: {e}")
                value_data.append({
                    'ticker': ticker,
                    'value_score': 0,
                    'pe_score': 0,
                    'pb_score': 0,
                    'ps_score': 0,
                    'market_cap': 0,
                    'close_price': 0,
                    'total_shares': 0
                })
        
        value_factors = pd.DataFrame(value_data)
        
        # Get forward returns
        forward_returns_data = []
        for ticker in universe:
            try:
                current_date = analysis_date
                future_date = current_date + timedelta(days=30)
                
                current_query = """
                SELECT close_price 
                FROM vcsc_daily_data_complete 
                WHERE ticker = %s AND trading_date = %s
                """
                current_result = pd.read_sql(current_query, self.engine, params=(ticker, current_date))
                
                future_query = """
                SELECT close_price, trading_date
                FROM vcsc_daily_data_complete 
                WHERE ticker = %s AND trading_date >= %s
                ORDER BY trading_date ASC
                LIMIT 1
                """
                future_result = pd.read_sql(future_query, self.engine, params=(ticker, future_date))
                
                if len(current_result) > 0 and len(future_result) > 0:
                    current_price = current_result.iloc[0]['close_price']
                    future_price = future_result.iloc[0]['close_price']
                    
                    if current_price > 0:
                        forward_return = (future_price - current_price) / current_price
                    else:
                        forward_return = 0
                else:
                    forward_return = 0
                    
                forward_returns_data.append({
                    'ticker': ticker,
                    'forward_return': forward_return
                })
                
            except Exception as e:
                print(f"Error calculating forward returns for {ticker}: {e}")
                forward_returns_data.append({
                    'ticker': ticker,
                    'forward_return': 0
                })
        
        forward_returns = pd.DataFrame(forward_returns_data)
        
        # Merge all data
        all_data = factors.merge(momentum, on='ticker', how='inner')
        all_data = all_data.merge(value_factors, on='ticker', how='inner')
        all_data = all_data.merge(forward_returns, on='ticker', how='inner')
        
        return all_data
    
    def calculate_conditional_ic(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate conditional IC of value score across ROAA quintiles."""
        # Create ROAA quintiles
        data['roaa_quintile'] = pd.qcut(data['roaa'], q=5, labels=['Q1 (Lowest)', 'Q2', 'Q3', 'Q4', 'Q5 (Highest)'])
        
        # Calculate IC for each quintile
        ic_results = []
        
        for quintile in data['roaa_quintile'].unique():
            if pd.isna(quintile):
                continue
                
            quintile_data = data[data['roaa_quintile'] == quintile]
            
            if len(quintile_data) < 5:  # Need at least 5 observations
                continue
            
            # Calculate correlation between value_score and forward_return
            correlation = quintile_data['value_score'].corr(quintile_data['forward_return'])
            
            # Calculate additional statistics
            ic_results.append({
                'roaa_quintile': quintile,
                'ic': correlation,
                'n_observations': len(quintile_data),
                'mean_roaa': quintile_data['roaa'].mean(),
                'mean_value_score': quintile_data['value_score'].mean(),
                'mean_forward_return': quintile_data['forward_return'].mean(),
                'std_value_score': quintile_data['value_score'].std(),
                'std_forward_return': quintile_data['forward_return'].std()
            })
        
        return pd.DataFrame(ic_results)
    
    def calculate_overall_ic(self, data: pd.DataFrame) -> dict:
        """Calculate overall IC and statistics."""
        overall_ic = data['value_score'].corr(data['forward_return'])
        
        return {
            'overall_ic': overall_ic,
            'n_observations': len(data),
            'mean_roaa': data['roaa'].mean(),
            'mean_value_score': data['value_score'].mean(),
            'mean_forward_return': data['forward_return'].mean(),
            'std_value_score': data['value_score'].std(),
            'std_forward_return': data['forward_return'].std()
        }
    
    def print_conditional_ic_analysis(self, conditional_ic: pd.DataFrame, overall_stats: dict):
        """Print comprehensive conditional IC analysis."""
        print("\n" + "="*100)
        print("CONDITIONAL IC ANALYSIS: VALUE SCORE vs FORWARD RETURNS")
        print("="*100)
        
        print(f"\nOVERALL STATISTICS:")
        print(f"Overall IC: {overall_stats['overall_ic']:.4f}")
        print(f"Total Observations: {overall_stats['n_observations']}")
        print(f"Mean ROAA: {overall_stats['mean_roaa']:.4f}")
        print(f"Mean Value Score: {overall_stats['mean_value_score']:.4f}")
        print(f"Mean Forward Return: {overall_stats['mean_forward_return']:.4f}")
        
        print(f"\n" + "="*100)
        print("CONDITIONAL IC BY ROAA QUINTILES")
        print("="*100)
        
        print(f"\n{'Quintile':<15} {'IC':<10} {'N':<5} {'Mean ROAA':<12} {'Mean Value':<12} {'Mean Return':<12} {'IC vs Overall':<15}")
        print("-" * 100)
        
        for _, row in conditional_ic.iterrows():
            ic_vs_overall = row['ic'] - overall_stats['overall_ic']
            ic_vs_overall_str = f"{ic_vs_overall:+.4f}"
            
            print(f"{row['roaa_quintile']:<15} {row['ic']:<10.4f} {row['n_observations']:<5} "
                  f"{row['mean_roaa']:<12.4f} {row['mean_value_score']:<12.4f} "
                  f"{row['mean_forward_return']:<12.4f} {ic_vs_overall_str:<15}")
        
        print("\n" + "="*100)
        print("INTERPRETATION")
        print("="*100)
        
        # Find strongest and weakest IC
        strongest_ic = conditional_ic.loc[conditional_ic['ic'].idxmax()]
        weakest_ic = conditional_ic.loc[conditional_ic['ic'].idxmin()]
        
        print(f"\nStrongest IC: {strongest_ic['roaa_quintile']} (IC = {strongest_ic['ic']:.4f})")
        print(f"Weakest IC: {weakest_ic['roaa_quintile']} (IC = {weakest_ic['ic']:.4f})")
        
        # Analyze patterns
        print(f"\nPattern Analysis:")
        
        if strongest_ic['ic'] > 0 and weakest_ic['ic'] < 0:
            print("✅ Value factor shows regime-dependent behavior")
            print("   - Positive IC in some quality quintiles")
            print("   - Negative IC in other quality quintiles")
        elif strongest_ic['ic'] < 0 and weakest_ic['ic'] < 0:
            print("❌ Value factor consistently negative across all quality levels")
            print("   - Strong contrarian signal regardless of quality")
        elif strongest_ic['ic'] > 0 and weakest_ic['ic'] > 0:
            print("✅ Value factor consistently positive across all quality levels")
            print("   - Positive signal regardless of quality")
        
        # Check for quality interaction
        high_quality_ic = conditional_ic[conditional_ic['roaa_quintile'].str.contains('Q5')]['ic'].iloc[0]
        low_quality_ic = conditional_ic[conditional_ic['roaa_quintile'].str.contains('Q1')]['ic'].iloc[0]
        
        print(f"\nQuality Interaction:")
        print(f"High Quality (Q5) IC: {high_quality_ic:.4f}")
        print(f"Low Quality (Q1) IC: {low_quality_ic:.4f}")
        
        if abs(high_quality_ic - low_quality_ic) > 0.01:
            print("💡 Significant quality interaction detected")
            if high_quality_ic > low_quality_ic:
                print("   - Value factor works better in high-quality companies")
            else:
                print("   - Value factor works better in low-quality companies")
        else:
            print("⚠️  No significant quality interaction")
        
        print("\n" + "="*100)
    
    def create_conditional_ic_visualizations(self, conditional_ic: pd.DataFrame, overall_stats: dict):
        """Create visualizations for conditional IC analysis."""
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Conditional IC Analysis: Value Score vs Forward Returns by ROAA Quintiles', fontsize=16, fontweight='bold')
        
        # 1. IC by quintile
        ax1 = axes[0, 0]
        quintiles = conditional_ic['roaa_quintile']
        ics = conditional_ic['ic']
        colors = ['red' if ic < 0 else 'blue' for ic in ics]
        
        bars = ax1.bar(range(len(quintiles)), ics, color=colors, alpha=0.7)
        ax1.set_xlabel('ROAA Quintiles')
        ax1.set_ylabel('Information Coefficient (IC)')
        ax1.set_title('Value Score IC by ROAA Quintile')
        ax1.set_xticks(range(len(quintiles)))
        ax1.set_xticklabels(quintiles, rotation=45, ha='right')
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax1.axhline(y=overall_stats['overall_ic'], color='green', linestyle='--', alpha=0.7, label=f'Overall IC: {overall_stats["overall_ic"]:.4f}')
        
        # Add value labels on bars
        for bar, ic in zip(bars, ics):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height > 0 else -0.01),
                    f'{ic:.3f}', ha='center', va='bottom' if height > 0 else 'top')
        
        ax1.legend()
        
        # 2. IC vs Mean ROAA
        ax2 = axes[0, 1]
        mean_roaa = conditional_ic['mean_roaa']
        
        ax2.scatter(mean_roaa, ics, s=100, alpha=0.7, c=colors)
        ax2.set_xlabel('Mean ROAA')
        ax2.set_ylabel('Information Coefficient (IC)')
        ax2.set_title('IC vs Mean ROAA')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.axhline(y=overall_stats['overall_ic'], color='green', linestyle='--', alpha=0.7)
        
        # Add labels for each point
        for i, (roaa, ic, quintile) in enumerate(zip(mean_roaa, ics, quintiles)):
            ax2.annotate(quintile.split()[0], (roaa, ic), xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 3. Number of observations by quintile
        ax3 = axes[1, 0]
        n_obs = conditional_ic['n_observations']
        
        bars = ax3.bar(range(len(quintiles)), n_obs, alpha=0.7)
        ax3.set_xlabel('ROAA Quintiles')
        ax3.set_ylabel('Number of Observations')
        ax3.set_title('Sample Size by ROAA Quintile')
        ax3.set_xticks(range(len(quintiles)))
        ax3.set_xticklabels(quintiles, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, n in zip(bars, n_obs):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{n}', ha='center', va='bottom')
        
        # 4. Mean value score vs mean forward return by quintile
        ax4 = axes[1, 1]
        mean_value = conditional_ic['mean_value_score']
        mean_return = conditional_ic['mean_forward_return']
        
        scatter = ax4.scatter(mean_value, mean_return, s=100, alpha=0.7, c=colors)
        ax4.set_xlabel('Mean Value Score')
        ax4.set_ylabel('Mean Forward Return')
        ax4.set_title('Mean Value Score vs Mean Forward Return')
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax4.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        # Add labels for each point
        for i, (val, ret, quintile) in enumerate(zip(mean_value, mean_return, quintiles)):
            ax4.annotate(quintile.split()[0], (val, ret), xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('conditional_ic_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_conditional_ic_analysis(self, analysis_date: datetime = None):
        """Run comprehensive conditional IC analysis."""
        if analysis_date is None:
            analysis_date = datetime(2024, 12, 18)
        
        print(f"Running conditional IC analysis for {analysis_date}")
        
        # Get universe
        universe_query = """
        SELECT DISTINCT ticker FROM intermediary_calculations_banking_cleaned 
        LIMIT 30
        """
        
        universe_df = pd.read_sql(universe_query, self.engine)
        universe = universe_df['ticker'].tolist()
        
        print(f"Universe size: {len(universe)} tickers")
        
        # Get comprehensive data
        data = self.get_comprehensive_data(analysis_date, universe)
        
        if len(data) == 0:
            print("No data extracted")
            return
        
        print(f"Data shape: {data.shape}")
        print("Sample data:")
        print(data.head())
        
        # Calculate conditional IC
        conditional_ic = self.calculate_conditional_ic(data)
        
        # Calculate overall IC
        overall_stats = self.calculate_overall_ic(data)
        
        # Print analysis
        self.print_conditional_ic_analysis(conditional_ic, overall_stats)
        
        # Create visualizations
        self.create_conditional_ic_visualizations(conditional_ic, overall_stats)
        
        return {
            'conditional_ic': conditional_ic,
            'overall_stats': overall_stats,
            'data': data
        }


def main():
    """Main execution function."""
    analyzer = ConditionalICAnalyzer()
    
    # Run analysis for date with sufficient historical data
    analysis_date = datetime(2024, 12, 18)
    
    results = analyzer.run_conditional_ic_analysis(analysis_date)
    
    if results:
        print("\nConditional IC analysis completed successfully!")
        print("Results saved to: conditional_ic_analysis.png")
    else:
        print("Conditional IC analysis failed")


if __name__ == "__main__":
    main() 