#!/usr/bin/env python3
"""
Momentum Transaction Cost Analysis

This script analyzes the impact of transaction costs on momentum factor
Information Coefficient (IC) performance and optimal rebalancing frequency.

Author: Factor Investing Team
Date: 2025-07-30
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add paths for imports
sys.path.append('../../../production')

try:
    from database.connection import get_engine
    from engine.qvm_engine_v2_enhanced import QVMEngineV2Enhanced
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

class MomentumTransactionCostAnalyzer:
    """
    Analyzer for transaction cost impact on momentum factor IC.
    """
    
    def __init__(self):
        """Initialize the transaction cost analyzer."""
        self.results = {}
        self.engine = None
        
    def initialize_engine(self):
        """Initialize the QVM engine."""
        try:
            self.engine = QVMEngineV2Enhanced()
            print("✅ QVM Engine initialized")
        except Exception as e:
            print(f"❌ Failed to initialize engine: {e}")
            raise
    
    def get_test_universe(self, analysis_date, limit=50):
        """Get a test universe of stocks."""
        try:
            query = f"""
            SELECT DISTINCT ticker
            FROM equity_history
            WHERE date <= '{analysis_date.date()}'
              AND close > 5000
            GROUP BY ticker
            HAVING COUNT(*) >= 252
            ORDER BY ticker
            LIMIT {limit}
            """
            with self.engine.engine.connect() as conn:
                result = pd.read_sql(query, conn)
            return result['ticker'].tolist()
        except Exception as e:
            print(f"❌ Failed to get universe: {e}")
            return []
    
    def calculate_momentum_ic_with_costs(self, analysis_date, universe, 
                                       forward_months=1, transaction_cost=0.0015):
        """
        Calculate momentum IC with transaction cost consideration.
        
        Args:
            analysis_date: Date for analysis
            universe: List of ticker symbols
            forward_months: Forward return horizon
            transaction_cost: Transaction cost as fraction (e.g., 0.0015 = 15 bps)
        """
        try:
            # Get fundamental data for sector mapping
            fundamental_data = self.engine.get_fundamentals_correct_timing(analysis_date, universe)
            if fundamental_data.empty:
                return None
            
            # Calculate momentum factors
            momentum_scores = self.engine._calculate_enhanced_momentum_composite(
                fundamental_data, analysis_date, universe
            )
            
            if not momentum_scores:
                return None
            
            # Calculate forward returns with transaction costs
            end_date = analysis_date + pd.DateOffset(months=forward_months)
            ticker_str = "', '".join(universe)
            query = f"""
            SELECT ticker, date, close as adj_close
            FROM equity_history
            WHERE ticker IN ('{ticker_str}')
              AND date BETWEEN '{analysis_date.date()}' AND '{end_date.date()}'
            ORDER BY ticker, date
            """
            
            with self.engine.engine.connect() as conn:
                price_data = pd.read_sql(query, conn, parse_dates=['date'])
            
            if price_data.empty:
                return None
            
            # Calculate forward returns with transaction costs
            forward_returns = {}
            for ticker in universe:
                ticker_data = price_data[price_data['ticker'] == ticker].sort_values('date')
                if len(ticker_data) >= 2:
                    start_price = ticker_data.iloc[0]['adj_close']
                    end_price = ticker_data.iloc[-1]['adj_close']
                    if start_price > 0:
                        # Apply transaction costs (entry and exit)
                        gross_return = (end_price / start_price) - 1
                        net_return = gross_return - (2 * transaction_cost)  # Round trip
                        forward_returns[ticker] = net_return
            
            # Calculate IC
            common_tickers = set(momentum_scores.keys()) & set(forward_returns.keys())
            if len(common_tickers) < 10:
                return None
            
            factor_series = pd.Series([momentum_scores[t] for t in common_tickers], 
                                    index=list(common_tickers))
            return_series = pd.Series([forward_returns[t] for t in common_tickers], 
                                    index=list(common_tickers))
            
            ic = factor_series.corr(return_series, method='spearman')
            
            return {
                'date': analysis_date,
                'ic': ic,
                'n_stocks': len(common_tickers),
                'transaction_cost': transaction_cost,
                'gross_ic': self._calculate_gross_ic(factor_series, return_series, transaction_cost)
            }
            
        except Exception as e:
            print(f"❌ Error calculating IC with costs: {e}")
            return None
    
    def _calculate_gross_ic(self, factor_series, return_series, transaction_cost):
        """Calculate gross IC (without transaction costs)."""
        try:
            # Reconstruct gross returns
            gross_returns = return_series + (2 * transaction_cost)
            return factor_series.corr(gross_returns, method='spearman')
        except:
            return np.nan
    
    def test_rebalancing_frequency(self, start_date='2017-01-01', end_date='2024-12-31'):
        """Test different rebalancing frequencies."""
        print("🚀 Testing Rebalancing Frequencies")
        print("=" * 60)
        
        frequencies = {
            'monthly': 'M',
            'quarterly': 'Q',
            'semi_annual': '6M',
            'annual': 'Y'
        }
        
        transaction_costs = [0.0005, 0.001, 0.0015, 0.002, 0.003]  # 5, 10, 15, 20, 30 bps
        
        results = {}
        
        for freq_name, freq_code in frequencies.items():
            print(f"\n📊 Testing {freq_name} rebalancing...")
            
            # Generate rebalance dates
            if freq_code == '6M':
                rebalance_dates = pd.date_range(
                    start=pd.to_datetime(start_date) + pd.DateOffset(months=12),
                    end=pd.to_datetime(end_date) - pd.DateOffset(months=12),
                    freq='6M'
                )
            else:
                rebalance_dates = pd.date_range(
                    start=pd.to_datetime(start_date) + pd.DateOffset(months=12),
                    end=pd.to_datetime(end_date) - pd.DateOffset(months=12),
                    freq=freq_code
                )
            
            freq_results = {}
            
            for cost in transaction_costs:
                print(f"  💰 Testing transaction cost: {cost*100:.1f} bps")
                
                ic_results = []
                for test_date in rebalance_dates:
                    universe = self.get_test_universe(test_date)
                    if len(universe) < 20:
                        continue
                    
                    ic_result = self.calculate_momentum_ic_with_costs(
                        test_date, universe, forward_months=1, transaction_cost=cost
                    )
                    
                    if ic_result:
                        ic_results.append(ic_result)
                
                if ic_results:
                    ic_values = [r['ic'] for r in ic_results if not np.isnan(r['ic'])]
                    if ic_values:
                        ic_series = pd.Series(ic_values)
                        freq_results[cost] = {
                            'mean_ic': ic_series.mean(),
                            'std_ic': ic_series.std(),
                            't_stat': ic_series.mean() / (ic_series.std() / np.sqrt(len(ic_series))) if ic_series.std() > 0 else 0,
                            'hit_rate': (ic_series > 0).mean(),
                            'n_observations': len(ic_series)
                        }
            
            results[freq_name] = freq_results
        
        self.results = results
        return results
    
    def analyze_cost_impact(self):
        """Analyze the impact of transaction costs on IC."""
        if not self.results:
            return
        
        print("\n📊 TRANSACTION COST IMPACT ANALYSIS")
        print("=" * 60)
        
        # Create summary table
        summary_data = []
        
        for freq_name, freq_results in self.results.items():
            for cost, stats in freq_results.items():
                summary_data.append({
                    'Frequency': freq_name,
                    'Transaction_Cost_bps': cost * 10000,  # Convert to bps
                    'Mean_IC': stats['mean_ic'],
                    'T_Stat': stats['t_stat'],
                    'Hit_Rate': stats['hit_rate'],
                    'N_Observations': stats['n_observations']
                })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Find optimal frequency for each cost level
        print("\n🎯 OPTIMAL FREQUENCY BY TRANSACTION COST:")
        print("-" * 50)
        
        for cost in sorted(summary_df['Transaction_Cost_bps'].unique()):
            cost_data = summary_df[summary_df['Transaction_Cost_bps'] == cost]
            best_freq = cost_data.loc[cost_data['Mean_IC'].idxmax()]
            
            print(f"  {cost:.0f} bps: {best_freq['Frequency']} (IC: {best_freq['Mean_IC']:.4f})")
        
        # Find optimal cost for each frequency
        print("\n🎯 OPTIMAL COST BY FREQUENCY:")
        print("-" * 40)
        
        for freq in summary_df['Frequency'].unique():
            freq_data = summary_df[summary_df['Frequency'] == freq]
            best_cost = freq_data.loc[freq_data['Mean_IC'].idxmax()]
            
            print(f"  {freq}: {best_cost['Transaction_Cost_bps']:.0f} bps (IC: {best_cost['Mean_IC']:.4f})")
        
        return summary_df
    
    def generate_cost_report(self):
        """Generate comprehensive transaction cost report."""
        if not self.results:
            return "No results available"
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("MOMENTUM TRANSACTION COST ANALYSIS REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Results by frequency and cost
        for freq_name, freq_results in self.results.items():
            report_lines.append(f"REBALANCING FREQUENCY: {freq_name.upper()}")
            report_lines.append("-" * 50)
            report_lines.append(f"{'Cost (bps)':<12} {'Mean IC':<10} {'T-Stat':<8} {'Hit Rate':<10}")
            report_lines.append("-" * 50)
            
            for cost, stats in sorted(freq_results.items()):
                report_lines.append(
                    f"{cost*10000:<12.0f} {stats['mean_ic']:<10.4f} {stats['t_stat']:<8.3f} {stats['hit_rate']:<10.1%}"
                )
            
            report_lines.append("")
        
        return "\n".join(report_lines)

def main():
    """Main execution function."""
    print("🚀 MOMENTUM TRANSACTION COST ANALYSIS")
    print("=" * 60)
    
    try:
        analyzer = MomentumTransactionCostAnalyzer()
        analyzer.initialize_engine()
        
        # Run analysis
        results = analyzer.test_rebalancing_frequency()
        
        # Analyze impact
        summary_df = analyzer.analyze_cost_impact()
        
        # Generate report
        report = analyzer.generate_cost_report()
        print("\n" + report)
        
        # Save report
        with open('data/momentum_transaction_cost_report.txt', 'w') as f:
            f.write(report)
        
        # Save summary data
        if summary_df is not None:
            summary_df.to_csv('data/momentum_transaction_cost_summary.csv', index=False)
        
        print("\n✅ Transaction cost analysis completed successfully!")
        print("📄 Report saved to: data/momentum_transaction_cost_report.txt")
        print("📊 Summary saved to: data/momentum_transaction_cost_summary.csv")
        
        return results
        
    except Exception as e:
        print(f"❌ Transaction cost analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main() 