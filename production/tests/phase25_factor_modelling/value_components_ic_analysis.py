"""
Value Components IC Analysis
===========================
Analyzes the individual contributions of different value measures
(P/E, P/B, P/S) to the overall value factor IC.
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

class ValueComponentsICAnalyzer:
    def __init__(self):
        self.engine = get_engine()
        
    def get_comprehensive_data(self, analysis_date: datetime, universe: list) -> pd.DataFrame:
        """Get comprehensive data including all value components and forward returns."""
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
        
        # Get value components with detailed breakdown
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
                    
                    # Calculate individual value components
                    pe_ratio = 0
                    pb_ratio = 0
                    ps_ratio = 0
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
                    
                    # Calculate composite value score (banking weights: 60% P/E, 40% P/B)
                    value_score = 0.6 * pe_score + 0.4 * pb_score
                    
                else:
                    pe_ratio = 0
                    pb_ratio = 0
                    ps_ratio = 0
                    pe_score = 0
                    pb_score = 0
                    ps_score = 0
                    value_score = 0
                    market_cap = 0
                    close_price = 0
                    total_shares = 0
                    
                value_data.append({
                    'ticker': ticker,
                    'pe_ratio': pe_ratio,
                    'pb_ratio': pb_ratio,
                    'ps_ratio': ps_ratio,
                    'pe_score': pe_score,
                    'pb_score': pb_score,
                    'ps_score': ps_score,
                    'value_score': value_score,
                    'market_cap': market_cap,
                    'close_price': close_price,
                    'total_shares': total_shares
                })
                
            except Exception as e:
                print(f"Error calculating value components for {ticker}: {e}")
                value_data.append({
                    'ticker': ticker,
                    'pe_ratio': 0, 'pb_ratio': 0, 'ps_ratio': 0,
                    'pe_score': 0, 'pb_score': 0, 'ps_score': 0,
                    'value_score': 0, 'market_cap': 0, 'close_price': 0, 'total_shares': 0
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
        all_data = factors.merge(value_factors, on='ticker', how='inner')
        all_data = all_data.merge(forward_returns, on='ticker', how='inner')
        
        return all_data
    
    def calculate_component_ics(self, data: pd.DataFrame) -> dict:
        """Calculate IC for each value component."""
        # Remove rows with zero values for meaningful analysis
        valid_data = data[
            (data['pe_score'] > 0) | 
            (data['pb_score'] > 0) | 
            (data['ps_score'] > 0)
        ].copy()
        
        if len(valid_data) == 0:
            print("No valid value component data found")
            return {}
        
        # Calculate individual ICs
        ic_results = {}
        
        # P/E Score IC
        pe_ic = valid_data['pe_score'].corr(valid_data['forward_return'])
        ic_results['pe_score'] = {
            'ic': pe_ic,
            'n_observations': len(valid_data[valid_data['pe_score'] > 0]),
            'mean_score': valid_data['pe_score'].mean(),
            'std_score': valid_data['pe_score'].std(),
            'mean_return': valid_data['forward_return'].mean(),
            'std_return': valid_data['forward_return'].std()
        }
        
        # P/B Score IC
        pb_ic = valid_data['pb_score'].corr(valid_data['forward_return'])
        ic_results['pb_score'] = {
            'ic': pb_ic,
            'n_observations': len(valid_data[valid_data['pb_score'] > 0]),
            'mean_score': valid_data['pb_score'].mean(),
            'std_score': valid_data['pb_score'].std(),
            'mean_return': valid_data['forward_return'].mean(),
            'std_return': valid_data['forward_return'].std()
        }
        
        # P/S Score IC
        ps_ic = valid_data['ps_score'].corr(valid_data['forward_return'])
        ic_results['ps_score'] = {
            'ic': ps_ic,
            'n_observations': len(valid_data[valid_data['ps_score'] > 0]),
            'mean_score': valid_data['ps_score'].mean(),
            'std_score': valid_data['ps_score'].std(),
            'mean_return': valid_data['forward_return'].mean(),
            'std_return': valid_data['forward_return'].std()
        }
        
        # Composite Value Score IC
        composite_ic = valid_data['value_score'].corr(valid_data['forward_return'])
        ic_results['composite_value'] = {
            'ic': composite_ic,
            'n_observations': len(valid_data[valid_data['value_score'] > 0]),
            'mean_score': valid_data['value_score'].mean(),
            'std_score': valid_data['value_score'].std(),
            'mean_return': valid_data['forward_return'].mean(),
            'std_return': valid_data['forward_return'].std()
        }
        
        return ic_results
    
    def calculate_weighted_contribution(self, ic_results: dict) -> dict:
        """Calculate weighted contribution of each component to composite IC."""
        # Banking sector weights: 60% P/E, 40% P/B
        weights = {
            'pe_score': 0.6,
            'pb_score': 0.4,
            'ps_score': 0.0  # Not used in banking composite
        }
        
        # Calculate theoretical composite IC from weighted components
        theoretical_composite_ic = (
            weights['pe_score'] * ic_results['pe_score']['ic'] +
            weights['pb_score'] * ic_results['pb_score']['ic']
        )
        
        # Calculate actual composite IC
        actual_composite_ic = ic_results['composite_value']['ic']
        
        # Calculate contribution analysis
        contribution_analysis = {
            'pe_contribution': weights['pe_score'] * ic_results['pe_score']['ic'],
            'pb_contribution': weights['pb_score'] * ic_results['pb_score']['ic'],
            'theoretical_composite': theoretical_composite_ic,
            'actual_composite': actual_composite_ic,
            'difference': actual_composite_ic - theoretical_composite_ic,
            'pe_weight': weights['pe_score'],
            'pb_weight': weights['pb_score']
        }
        
        return contribution_analysis
    
    def calculate_conditional_component_ics(self, data: pd.DataFrame) -> dict:
        """Calculate conditional ICs for each component across ROAA quintiles."""
        # Create ROAA quintiles
        data['roaa_quintile'] = pd.qcut(data['roaa'], q=5, labels=['Q1 (Lowest)', 'Q2', 'Q3', 'Q4', 'Q5 (Highest)'])
        
        conditional_ics = {}
        
        for quintile in data['roaa_quintile'].unique():
            if pd.isna(quintile):
                continue
                
            quintile_data = data[data['roaa_quintile'] == quintile]
            
            if len(quintile_data) < 5:  # Need at least 5 observations
                continue
            
            # Calculate IC for each component in this quintile
            quintile_ics = {}
            
            # P/E Score IC
            pe_ic = quintile_data['pe_score'].corr(quintile_data['forward_return'])
            quintile_ics['pe_score'] = pe_ic
            
            # P/B Score IC
            pb_ic = quintile_data['pb_score'].corr(quintile_data['forward_return'])
            quintile_ics['pb_score'] = pb_ic
            
            # P/S Score IC
            ps_ic = quintile_data['ps_score'].corr(quintile_data['forward_return'])
            quintile_ics['ps_score'] = ps_ic
            
            # Composite Value Score IC
            composite_ic = quintile_data['value_score'].corr(quintile_data['forward_return'])
            quintile_ics['composite_value'] = composite_ic
            
            conditional_ics[quintile] = {
                'ics': quintile_ics,
                'n_observations': len(quintile_data),
                'mean_roaa': quintile_data['roaa'].mean()
            }
        
        return conditional_ics
    
    def print_component_ic_analysis(self, ic_results: dict, contribution_analysis: dict):
        """Print comprehensive component IC analysis."""
        print("\n" + "="*100)
        print("VALUE COMPONENTS IC ANALYSIS")
        print("="*100)
        
        print(f"\nINDIVIDUAL COMPONENT ICs:")
        print(f"{'Component':<15} {'IC':<10} {'N':<5} {'Mean Score':<12} {'Std Score':<12} {'Mean Return':<12}")
        print("-" * 80)
        
        for component, stats in ic_results.items():
            if component != 'composite_value':
                print(f"{component:<15} {stats['ic']:<10.4f} {stats['n_observations']:<5} "
                      f"{stats['mean_score']:<12.4f} {stats['std_score']:<12.4f} "
                      f"{stats['mean_return']:<12.4f}")
        
        print(f"\nCOMPOSITE VALUE SCORE:")
        composite_stats = ic_results['composite_value']
        print(f"Composite IC: {composite_stats['ic']:.4f}")
        print(f"Observations: {composite_stats['n_observations']}")
        print(f"Mean Score: {composite_stats['mean_score']:.4f}")
        print(f"Mean Return: {composite_stats['mean_return']:.4f}")
        
        print(f"\n" + "="*100)
        print("WEIGHTED CONTRIBUTION ANALYSIS")
        print("="*100)
        
        print(f"\nBanking Sector Weights:")
        print(f"P/E Score Weight: {contribution_analysis['pe_weight']:.1%}")
        print(f"P/B Score Weight: {contribution_analysis['pb_weight']:.1%}")
        
        print(f"\nIndividual Contributions:")
        print(f"P/E Contribution: {contribution_analysis['pe_contribution']:.4f}")
        print(f"P/B Contribution: {contribution_analysis['pb_contribution']:.4f}")
        print(f"Theoretical Composite IC: {contribution_analysis['theoretical_composite']:.4f}")
        print(f"Actual Composite IC: {contribution_analysis['actual_composite']:.4f}")
        print(f"Difference: {contribution_analysis['difference']:.4f}")
        
        print(f"\n" + "="*100)
        print("INTERPRETATION")
        print("="*100)
        
        # Find strongest and weakest components
        component_ics = {k: v['ic'] for k, v in ic_results.items() if k != 'composite_value'}
        strongest_component = max(component_ics, key=component_ics.get)
        weakest_component = min(component_ics, key=component_ics.get)
        
        print(f"\nStrongest Component: {strongest_component} (IC = {component_ics[strongest_component]:.4f})")
        print(f"Weakest Component: {weakest_component} (IC = {component_ics[weakest_component]:.4f})")
        
        # Analyze contribution efficiency
        pe_contribution = contribution_analysis['pe_contribution']
        pb_contribution = contribution_analysis['pb_contribution']
        
        print(f"\nContribution Analysis:")
        if abs(pe_contribution) > abs(pb_contribution):
            print(f"💡 P/E Score dominates the contrarian signal")
            print(f"   - P/E Contribution: {pe_contribution:.4f}")
            print(f"   - P/B Contribution: {pb_contribution:.4f}")
        else:
            print(f"💡 P/B Score dominates the contrarian signal")
            print(f"   - P/B Contribution: {pb_contribution:.4f}")
            print(f"   - P/E Contribution: {pe_contribution:.4f}")
        
        # Check if theoretical matches actual
        difference = contribution_analysis['difference']
        if abs(difference) < 0.01:
            print(f"✅ Theoretical and actual composite ICs match closely")
        else:
            print(f"⚠️  Theoretical and actual composite ICs differ by {difference:.4f}")
            print(f"   - Possible interaction effects between components")
        
        print("\n" + "="*100)
    
    def print_conditional_component_analysis(self, conditional_ics: dict):
        """Print conditional component IC analysis."""
        print("\n" + "="*100)
        print("CONDITIONAL COMPONENT ICs BY ROAA QUINTILES")
        print("="*100)
        
        for quintile, data in conditional_ics.items():
            print(f"\n{quintile} (N={data['n_observations']}, Mean ROAA={data['mean_roaa']:.4f}):")
            print(f"{'Component':<15} {'IC':<10}")
            print("-" * 30)
            
            for component, ic in data['ics'].items():
                print(f"{component:<15} {ic:<10.4f}")
        
        print(f"\n" + "="*100)
        print("CONDITIONAL PATTERN ANALYSIS")
        print("="*100)
        
        # Analyze patterns across quintiles
        for component in ['pe_score', 'pb_score', 'ps_score', 'composite_value']:
            print(f"\n{component.upper()} Pattern:")
            component_ics = []
            quintiles = []
            
            for quintile, data in conditional_ics.items():
                if component in data['ics']:
                    component_ics.append(data['ics'][component])
                    quintiles.append(quintile)
            
            if component_ics:
                min_ic = min(component_ics)
                max_ic = max(component_ics)
                min_quintile = quintiles[component_ics.index(min_ic)]
                max_quintile = quintiles[component_ics.index(max_ic)]
                
                print(f"  Range: {min_ic:.4f} to {max_ic:.4f}")
                print(f"  Best: {max_quintile} ({max_ic:.4f})")
                print(f"  Worst: {min_quintile} ({min_ic:.4f})")
        
        print("\n" + "="*100)
    
    def create_component_ic_visualizations(self, ic_results: dict, contribution_analysis: dict, conditional_ics: dict):
        """Create visualizations for component IC analysis."""
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Value Components IC Analysis', fontsize=16, fontweight='bold')
        
        # 1. Individual Component ICs
        ax1 = axes[0, 0]
        components = ['pe_score', 'pb_score', 'ps_score']
        ics = [ic_results[comp]['ic'] for comp in components]
        colors = ['red' if ic < 0 else 'blue' for ic in ics]
        
        bars = ax1.bar(components, ics, color=colors, alpha=0.7)
        ax1.set_xlabel('Value Components')
        ax1.set_ylabel('Information Coefficient (IC)')
        ax1.set_title('Individual Component ICs')
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Add value labels on bars
        for bar, ic in zip(bars, ics):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height > 0 else -0.01),
                    f'{ic:.3f}', ha='center', va='bottom' if height > 0 else 'top')
        
        # 2. Weighted Contributions
        ax2 = axes[0, 1]
        contributions = ['P/E Contribution', 'P/B Contribution']
        values = [contribution_analysis['pe_contribution'], contribution_analysis['pb_contribution']]
        colors = ['red' if v < 0 else 'blue' for v in values]
        
        bars = ax2.bar(contributions, values, color=colors, alpha=0.7)
        ax2.set_xlabel('Component Contributions')
        ax2.set_ylabel('Weighted IC Contribution')
        ax2.set_title('Weighted Contributions to Composite IC')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height > 0 else -0.01),
                    f'{val:.3f}', ha='center', va='bottom' if height > 0 else 'top')
        
        # 3. Theoretical vs Actual Composite IC
        ax3 = axes[0, 2]
        ic_types = ['Theoretical', 'Actual']
        ic_values = [contribution_analysis['theoretical_composite'], contribution_analysis['actual_composite']]
        colors = ['red' if v < 0 else 'blue' for v in ic_values]
        
        bars = ax3.bar(ic_types, ic_values, color=colors, alpha=0.7)
        ax3.set_xlabel('Composite IC Type')
        ax3.set_ylabel('Information Coefficient (IC)')
        ax3.set_title('Theoretical vs Actual Composite IC')
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Add value labels on bars
        for bar, val in zip(bars, ic_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height > 0 else -0.01),
                    f'{val:.3f}', ha='center', va='bottom' if height > 0 else 'top')
        
        # 4. Component ICs by ROAA Quintile (P/E)
        ax4 = axes[1, 0]
        quintiles = list(conditional_ics.keys())
        pe_ics = [conditional_ics[q]['ics']['pe_score'] for q in quintiles]
        colors = ['red' if ic < 0 else 'blue' for ic in pe_ics]
        
        bars = ax4.bar(range(len(quintiles)), pe_ics, color=colors, alpha=0.7)
        ax4.set_xlabel('ROAA Quintiles')
        ax4.set_ylabel('P/E Score IC')
        ax4.set_title('P/E Score IC by ROAA Quintile')
        ax4.set_xticks(range(len(quintiles)))
        ax4.set_xticklabels([q.split()[0] for q in quintiles], rotation=45, ha='right')
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # 5. Component ICs by ROAA Quintile (P/B)
        ax5 = axes[1, 1]
        pb_ics = [conditional_ics[q]['ics']['pb_score'] for q in quintiles]
        colors = ['red' if ic < 0 else 'blue' for ic in pb_ics]
        
        bars = ax5.bar(range(len(quintiles)), pb_ics, color=colors, alpha=0.7)
        ax5.set_xlabel('ROAA Quintiles')
        ax5.set_ylabel('P/B Score IC')
        ax5.set_title('P/B Score IC by ROAA Quintile')
        ax5.set_xticks(range(len(quintiles)))
        ax5.set_xticklabels([q.split()[0] for q in quintiles], rotation=45, ha='right')
        ax5.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # 6. Component ICs by ROAA Quintile (Composite)
        ax6 = axes[1, 2]
        composite_ics = [conditional_ics[q]['ics']['composite_value'] for q in quintiles]
        colors = ['red' if ic < 0 else 'blue' for ic in composite_ics]
        
        bars = ax6.bar(range(len(quintiles)), composite_ics, color=colors, alpha=0.7)
        ax6.set_xlabel('ROAA Quintiles')
        ax6.set_ylabel('Composite Value IC')
        ax6.set_title('Composite Value IC by ROAA Quintile')
        ax6.set_xticks(range(len(quintiles)))
        ax6.set_xticklabels([q.split()[0] for q in quintiles], rotation=45, ha='right')
        ax6.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig('value_components_ic_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_component_ic_analysis(self, analysis_date: datetime = None):
        """Run comprehensive component IC analysis."""
        if analysis_date is None:
            analysis_date = datetime(2024, 12, 18)
        
        print(f"Running value components IC analysis for {analysis_date}")
        
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
        print(data[['ticker', 'pe_score', 'pb_score', 'ps_score', 'value_score', 'forward_return']].head())
        
        # Calculate component ICs
        ic_results = self.calculate_component_ics(data)
        
        if not ic_results:
            print("No valid component IC results")
            return
        
        # Calculate weighted contribution
        contribution_analysis = self.calculate_weighted_contribution(ic_results)
        
        # Calculate conditional component ICs
        conditional_ics = self.calculate_conditional_component_ics(data)
        
        # Print analysis
        self.print_component_ic_analysis(ic_results, contribution_analysis)
        self.print_conditional_component_analysis(conditional_ics)
        
        # Create visualizations
        self.create_component_ic_visualizations(ic_results, contribution_analysis, conditional_ics)
        
        return {
            'ic_results': ic_results,
            'contribution_analysis': contribution_analysis,
            'conditional_ics': conditional_ics,
            'data': data
        }


def main():
    """Main execution function."""
    analyzer = ValueComponentsICAnalyzer()
    
    # Run analysis for date with sufficient historical data
    analysis_date = datetime(2024, 12, 18)
    
    results = analyzer.run_component_ic_analysis(analysis_date)
    
    if results:
        print("\nValue components IC analysis completed successfully!")
        print("Results saved to: value_components_ic_analysis.png")
    else:
        print("Value components IC analysis failed")


if __name__ == "__main__":
    main() 