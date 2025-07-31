"""
Non-Banking Sector Analysis
===========================
Analyzes P/B and P/E ICs conditional on ROAA for non-banking sectors
using the correct column names from enhanced and securities tables.
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

class NonBankingSectorAnalyzer:
    def __init__(self):
        self.engine = get_engine()
        
    def get_non_banking_sector_data(self, analysis_date: datetime) -> pd.DataFrame:
        """Get comprehensive data for non-banking sectors."""
        # Get all sectors except Banks
        sectors_query = """
        SELECT DISTINCT sector FROM master_info
        WHERE sector IS NOT NULL AND sector != '' AND sector != 'Banks'
        ORDER BY sector
        """
        
        sectors_df = pd.read_sql(sectors_query, self.engine)
        sectors = sectors_df['sector'].tolist()
        
        print(f"Available non-banking sectors: {sectors}")
        
        all_sector_data = []
        
        for sector in sectors:
            try:
                # Get universe for this sector from master_info
                universe_query = """
                SELECT DISTINCT ticker FROM master_info 
                WHERE sector = %s
                LIMIT 20
                """
                
                universe_df = pd.read_sql(universe_query, self.engine, params=(sector,))
                universe = universe_df['ticker'].tolist()
                
                if len(universe) < 5:  # Need at least 5 stocks per sector
                    continue
                
                print(f"Processing sector: {sector} with {len(universe)} tickers")
                
                # Try to get data from enhanced table first
                sector_data = self.get_enhanced_data(sector, universe, analysis_date)
                
                # If no data in enhanced, try securities table
                if sector_data is None or len(sector_data) == 0:
                    sector_data = self.get_securities_data(sector, universe, analysis_date)
                
                if sector_data is not None and len(sector_data) > 0:
                    all_sector_data.append(sector_data)
                    print(f"  ✅ Got {len(sector_data)} records for {sector}")
                else:
                    print(f"  ❌ No data found for {sector}")
                
            except Exception as e:
                print(f"Error processing sector {sector}: {e}")
                continue
        
        if all_sector_data:
            combined_data = pd.concat(all_sector_data, ignore_index=True)
            return combined_data
        else:
            return pd.DataFrame()
    
    def get_enhanced_data(self, sector: str, universe: list, analysis_date: datetime) -> pd.DataFrame:
        """Get data from enhanced intermediary table."""
        try:
            # Get factors for this sector from enhanced table using correct column names
            factors_query = """
            SELECT ticker, 
                   NetProfit_TTM / AvgTotalAssets as roaa,
                   NetProfit_TTM / AvgTotalEquity as roae,
                   GrossProfit_TTM / Revenue_TTM as gross_margin,
                   Revenue_TTM / AvgTotalAssets as asset_turnover
            FROM intermediary_calculations_enhanced
            WHERE quarter = (
                SELECT MAX(quarter) 
                FROM intermediary_calculations_enhanced 
                WHERE quarter <= %s
            ) AND ticker IN ({})
            """.format(','.join(['%s'] * len(universe)))
            
            params = [analysis_date] + universe
            
            factors = pd.read_sql(factors_query, self.engine, params=tuple(params))
            factors['sector'] = sector
            
            if len(factors) == 0:
                return pd.DataFrame()
            
            # Get value components and forward returns
            value_factors = self.get_value_factors(universe, analysis_date)
            forward_returns = self.get_forward_returns(universe, analysis_date)
            
            # Merge data
            sector_data = factors.merge(value_factors, on='ticker', how='inner')
            sector_data = sector_data.merge(forward_returns, on='ticker', how='inner')
            
            return sector_data
            
        except Exception as e:
            print(f"Error getting enhanced data for {sector}: {e}")
            return pd.DataFrame()
    
    def get_securities_data(self, sector: str, universe: list, analysis_date: datetime) -> pd.DataFrame:
        """Get data from securities intermediary table."""
        try:
            # Get factors for this sector from securities table using correct column names
            factors_query = """
            SELECT ticker, 
                   NetProfit_TTM / AvgTotalAssets as roaa,
                   NetProfit_TTM / AvgTotalEquity as roae,
                   NetProfitMargin as net_margin,
                   AssetTurnover as asset_turnover
            FROM intermediary_calculations_securities_cleaned
            WHERE quarter = (
                SELECT MAX(quarter) 
                FROM intermediary_calculations_securities_cleaned 
                WHERE quarter <= %s
            ) AND ticker IN ({})
            """.format(','.join(['%s'] * len(universe)))
            
            params = [analysis_date] + universe
            
            factors = pd.read_sql(factors_query, self.engine, params=tuple(params))
            factors['sector'] = sector
            
            if len(factors) == 0:
                return pd.DataFrame()
            
            # Get value components and forward returns
            value_factors = self.get_value_factors(universe, analysis_date)
            forward_returns = self.get_forward_returns(universe, analysis_date)
            
            # Merge data
            sector_data = factors.merge(value_factors, on='ticker', how='inner')
            sector_data = sector_data.merge(forward_returns, on='ticker', how='inner')
            
            return sector_data
            
        except Exception as e:
            print(f"Error getting securities data for {sector}: {e}")
            return pd.DataFrame()
    
    def get_value_factors(self, universe: list, analysis_date: datetime) -> pd.DataFrame:
        """Get value factors for a universe of tickers."""
        value_data = []
        
        for ticker in universe:
            try:
                # Try enhanced table first
                fundamental_query = """
                SELECT 
                    NetProfit_TTM,
                    AvgTotalEquity,
                    Revenue_TTM
                FROM intermediary_calculations_enhanced
                WHERE ticker = %s AND quarter = (
                    SELECT MAX(quarter) 
                    FROM intermediary_calculations_enhanced 
                    WHERE quarter <= %s
                )
                """
                
                fundamental_data = pd.read_sql(fundamental_query, self.engine, params=(ticker, analysis_date))
                
                if len(fundamental_data) == 0:
                    # Try securities table
                    fundamental_query = """
                    SELECT 
                        NetProfit_TTM,
                        AvgTotalEquity,
                        TotalOperatingRevenue_TTM as Revenue_TTM
                    FROM intermediary_calculations_securities_cleaned
                    WHERE ticker = %s AND quarter = (
                        SELECT MAX(quarter) 
                        FROM intermediary_calculations_securities_cleaned 
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
                    
                    # Calculate value components
                    pe_ratio = 0
                    pb_ratio = 0
                    pe_score = 0
                    pb_score = 0
                    
                    if pd.notna(row['NetProfit_TTM']) and row['NetProfit_TTM'] > 0 and market_cap > 0:
                        pe_ratio = market_cap / row['NetProfit_TTM']
                        pe_score = 1 / pe_ratio if pe_ratio > 0 else 0
                    
                    if pd.notna(row['AvgTotalEquity']) and row['AvgTotalEquity'] > 0 and market_cap > 0:
                        pb_ratio = market_cap / row['AvgTotalEquity']
                        pb_score = 1 / pb_ratio if pb_ratio > 0 else 0
                    
                else:
                    pe_ratio = 0
                    pb_ratio = 0
                    pe_score = 0
                    pb_score = 0
                    market_cap = 0
                    close_price = 0
                    total_shares = 0
                    
                value_data.append({
                    'ticker': ticker,
                    'pe_ratio': pe_ratio,
                    'pb_ratio': pb_ratio,
                    'pe_score': pe_score,
                    'pb_score': pb_score,
                    'market_cap': market_cap,
                    'close_price': close_price,
                    'total_shares': total_shares
                })
                
            except Exception as e:
                print(f"Error calculating value factors for {ticker}: {e}")
                value_data.append({
                    'ticker': ticker,
                    'pe_ratio': 0, 'pb_ratio': 0,
                    'pe_score': 0, 'pb_score': 0,
                    'market_cap': 0, 'close_price': 0, 'total_shares': 0
                })
        
        return pd.DataFrame(value_data)
    
    def get_forward_returns(self, universe: list, analysis_date: datetime) -> pd.DataFrame:
        """Get forward returns for a universe of tickers."""
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
        
        return pd.DataFrame(forward_returns_data)
    
    def calculate_sector_conditional_ics(self, data: pd.DataFrame) -> dict:
        """Calculate conditional ICs for each sector across ROAA quintiles."""
        sector_conditional_ics = {}
        
        for sector in data['sector'].unique():
            sector_data = data[data['sector'] == sector].copy()
            
            if len(sector_data) < 10:  # Need at least 10 observations per sector
                continue
            
            # Create ROAA quintiles for this sector
            try:
                sector_data['roaa_quintile'] = pd.qcut(sector_data['roaa'], q=5, labels=['Q1 (Lowest)', 'Q2', 'Q3', 'Q4', 'Q5 (Highest)'])
            except ValueError:
                # If not enough unique values, skip this sector
                continue
            
            sector_ics = {}
            
            for quintile in sector_data['roaa_quintile'].unique():
                if pd.isna(quintile):
                    continue
                    
                quintile_data = sector_data[sector_data['roaa_quintile'] == quintile]
                
                if len(quintile_data) < 3:  # Need at least 3 observations per quintile
                    continue
                
                # Calculate IC for each component in this quintile
                quintile_ics = {}
                
                # P/E Score IC
                pe_ic = quintile_data['pe_score'].corr(quintile_data['forward_return'])
                quintile_ics['pe_score'] = pe_ic
                
                # P/B Score IC
                pb_ic = quintile_data['pb_score'].corr(quintile_data['forward_return'])
                quintile_ics['pb_score'] = pb_ic
                
                sector_ics[quintile] = {
                    'ics': quintile_ics,
                    'n_observations': len(quintile_data),
                    'mean_roaa': quintile_data['roaa'].mean(),
                    'mean_pe_score': quintile_data['pe_score'].mean(),
                    'mean_pb_score': quintile_data['pb_score'].mean(),
                    'mean_forward_return': quintile_data['forward_return'].mean()
                }
            
            if sector_ics:
                sector_conditional_ics[sector] = sector_ics
        
        return sector_conditional_ics
    
    def calculate_sector_overall_ics(self, data: pd.DataFrame) -> dict:
        """Calculate overall ICs for each sector."""
        sector_overall_ics = {}
        
        for sector in data['sector'].unique():
            sector_data = data[data['sector'] == sector].copy()
            
            if len(sector_data) < 5:  # Need at least 5 observations per sector
                continue
            
            # Calculate overall ICs for this sector
            pe_ic = sector_data['pe_score'].corr(sector_data['forward_return'])
            pb_ic = sector_data['pb_score'].corr(sector_data['forward_return'])
            
            sector_overall_ics[sector] = {
                'pe_ic': pe_ic,
                'pb_ic': pb_ic,
                'n_observations': len(sector_data),
                'mean_roaa': sector_data['roaa'].mean(),
                'mean_pe_score': sector_data['pe_score'].mean(),
                'mean_pb_score': sector_data['pb_score'].mean(),
                'mean_forward_return': sector_data['forward_return'].mean()
            }
        
        return sector_overall_ics
    
    def print_non_banking_analysis(self, sector_overall_ics: dict, sector_conditional_ics: dict):
        """Print comprehensive non-banking sector analysis."""
        print("\n" + "="*100)
        print("NON-BANKING SECTOR CONDITIONAL IC ANALYSIS: P/B AND P/E BY ROAA QUINTILES")
        print("="*100)
        
        print(f"\nOVERALL SECTOR ICs:")
        print(f"{'Sector':<25} {'P/E IC':<10} {'P/B IC':<10} {'N':<5} {'Mean ROAA':<12} {'Mean Return':<12}")
        print("-" * 90)
        
        for sector, stats in sector_overall_ics.items():
            print(f"{sector:<25} {stats['pe_ic']:<10.4f} {stats['pb_ic']:<10.4f} "
                  f"{stats['n_observations']:<5} {stats['mean_roaa']:<12.4f} "
                  f"{stats['mean_forward_return']:<12.4f}")
        
        print(f"\n" + "="*100)
        print("CONDITIONAL ICs BY SECTOR AND ROAA QUINTILE")
        print("="*100)
        
        for sector, quintile_data in sector_conditional_ics.items():
            print(f"\n{sector.upper()} SECTOR:")
            print(f"{'Quintile':<15} {'P/E IC':<10} {'P/B IC':<10} {'N':<5} {'Mean ROAA':<12}")
            print("-" * 60)
            
            for quintile, stats in quintile_data.items():
                print(f"{quintile:<15} {stats['ics']['pe_score']:<10.4f} "
                      f"{stats['ics']['pb_score']:<10.4f} {stats['n_observations']:<5} "
                      f"{stats['mean_roaa']:<12.4f}")
        
        print(f"\n" + "="*100)
        print("CROSS-SECTOR COMPARISON")
        print("="*100)
        
        # Compare sectors
        for component in ['pe_score', 'pb_score']:
            print(f"\n{component.upper()} CROSS-SECTOR COMPARISON:")
            
            sector_comparison = {}
            for sector, quintile_data in sector_conditional_ics.items():
                all_ics = []
                for quintile, stats in quintile_data.items():
                    if component in stats['ics']:
                        all_ics.append(stats['ics'][component])
                
                if all_ics:
                    sector_comparison[sector] = {
                        'mean_ic': np.mean(all_ics),
                        'std_ic': np.std(all_ics),
                        'min_ic': min(all_ics),
                        'max_ic': max(all_ics),
                        'range': max(all_ics) - min(all_ics)
                    }
            
            # Sort by mean IC
            sorted_sectors = sorted(sector_comparison.items(), key=lambda x: x[1]['mean_ic'], reverse=True)
            
            print(f"{'Sector':<25} {'Mean IC':<10} {'Std IC':<10} {'Range':<10} {'Min':<10} {'Max':<10}")
            print("-" * 90)
            
            for sector, stats in sorted_sectors:
                print(f"{sector:<25} {stats['mean_ic']:<10.4f} {stats['std_ic']:<10.4f} "
                      f"{stats['range']:<10.4f} {stats['min_ic']:<10.4f} {stats['max_ic']:<10.4f}")
        
        print("\n" + "="*100)
    
    def create_non_banking_visualizations(self, sector_overall_ics: dict, sector_conditional_ics: dict):
        """Create visualizations for non-banking sector analysis."""
        plt.style.use('seaborn-v0_8')
        
        # Create multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Non-Banking Sector Conditional IC Analysis: P/B and P/E by ROAA Quintiles', fontsize=16, fontweight='bold')
        
        # 1. Overall P/E IC by Sector
        ax1 = axes[0, 0]
        sectors = list(sector_overall_ics.keys())
        pe_ics = [sector_overall_ics[s]['pe_ic'] for s in sectors]
        colors = ['red' if ic < 0 else 'blue' for ic in pe_ics]
        
        bars = ax1.bar(range(len(sectors)), pe_ics, color=colors, alpha=0.7)
        ax1.set_xlabel('Sectors')
        ax1.set_ylabel('P/E Information Coefficient (IC)')
        ax1.set_title('Overall P/E IC by Sector')
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax1.set_xticks(range(len(sectors)))
        ax1.set_xticklabels(sectors, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, ic in zip(bars, pe_ics):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height > 0 else -0.01),
                    f'{ic:.3f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=8)
        
        # 2. Overall P/B IC by Sector
        ax2 = axes[0, 1]
        pb_ics = [sector_overall_ics[s]['pb_ic'] for s in sectors]
        colors = ['red' if ic < 0 else 'blue' for ic in pb_ics]
        
        bars = ax2.bar(range(len(sectors)), pb_ics, color=colors, alpha=0.7)
        ax2.set_xlabel('Sectors')
        ax2.set_ylabel('P/B Information Coefficient (IC)')
        ax2.set_title('Overall P/B IC by Sector')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_xticks(range(len(sectors)))
        ax2.set_xticklabels(sectors, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, ic in zip(bars, pb_ics):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height > 0 else -0.01),
                    f'{ic:.3f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=8)
        
        # 3. P/E vs P/B IC Comparison
        ax3 = axes[1, 0]
        ax3.scatter(pe_ics, pb_ics, s=100, alpha=0.7)
        ax3.set_xlabel('P/E Information Coefficient (IC)')
        ax3.set_ylabel('P/B Information Coefficient (IC)')
        ax3.set_title('P/E vs P/B IC Comparison by Sector')
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax3.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        # Add sector labels
        for i, sector in enumerate(sectors):
            ax3.annotate(sector, (pe_ics[i], pb_ics[i]), xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 4. Sector Sample Sizes
        ax4 = axes[1, 1]
        sample_sizes = [sector_overall_ics[s]['n_observations'] for s in sectors]
        
        bars = ax4.bar(range(len(sectors)), sample_sizes, alpha=0.7)
        ax4.set_xlabel('Sectors')
        ax4.set_ylabel('Number of Observations')
        ax4.set_title('Sample Size by Sector')
        ax4.set_xticks(range(len(sectors)))
        ax4.set_xticklabels(sectors, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, size in zip(bars, sample_sizes):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{size}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('non_banking_sector_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_non_banking_analysis(self, analysis_date: datetime = None):
        """Run comprehensive non-banking sector conditional IC analysis."""
        if analysis_date is None:
            analysis_date = datetime(2024, 12, 18)
        
        print(f"Running non-banking sector conditional IC analysis for {analysis_date}")
        
        # Get comprehensive non-banking sector data
        data = self.get_non_banking_sector_data(analysis_date)
        
        if len(data) == 0:
            print("No non-banking sector data extracted")
            return
        
        print(f"Total data shape: {data.shape}")
        print("Sectors in data:")
        print(data['sector'].value_counts())
        
        # Calculate sector overall ICs
        sector_overall_ics = self.calculate_sector_overall_ics(data)
        
        # Calculate sector conditional ICs
        sector_conditional_ics = self.calculate_sector_conditional_ics(data)
        
        # Print analysis
        self.print_non_banking_analysis(sector_overall_ics, sector_conditional_ics)
        
        # Create visualizations
        self.create_non_banking_visualizations(sector_overall_ics, sector_conditional_ics)
        
        return {
            'sector_overall_ics': sector_overall_ics,
            'sector_conditional_ics': sector_conditional_ics,
            'data': data
        }


def main():
    """Main execution function."""
    analyzer = NonBankingSectorAnalyzer()
    
    # Run analysis for date with sufficient historical data
    analysis_date = datetime(2024, 12, 18)
    
    results = analyzer.run_non_banking_analysis(analysis_date)
    
    if results:
        print("\nNon-banking sector conditional IC analysis completed successfully!")
        print("Results saved to: non_banking_sector_analysis.png")
    else:
        print("Non-banking sector conditional IC analysis failed")


if __name__ == "__main__":
    main() 