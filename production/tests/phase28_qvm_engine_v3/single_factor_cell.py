#!/usr/bin/env python3
"""
Single Factor Strategies Execution Script
This script can be run after the main QVM strategy to add single factor analysis.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from single_factors import run_single_factors

def execute_single_factor_analysis(QVM_CONFIG, price_data_raw, fundamental_data_raw, 
                                  daily_returns_matrix, benchmark_returns, engine, qvm_net_returns):
    """
    Execute single factor strategies and generate comparison analysis.
    
    This function should be called after the main QVM strategy execution.
    """
    
    print("\n" + "="*80)
    print("📊 SINGLE FACTOR STRATEGIES EXECUTION")
    print("="*80)

    try:
        # Run single factor strategies
        single_factor_results = run_single_factors(
            config=QVM_CONFIG,
            price_data=price_data_raw,
            fundamental_data=fundamental_data_raw,
            returns_matrix=daily_returns_matrix,
            benchmark_returns=benchmark_returns,
            db_engine=engine,
            qvm_returns=qvm_net_returns
        )

        # Generate comparison analysis
        print("\n" + "="*80)
        print("📊 SINGLE FACTOR STRATEGIES: PERFORMANCE COMPARISON")
        print("="*80)

        # Calculate metrics for all strategies
        strategies = {
            'QVM Composite': single_factor_results['qvm_returns'],
            'Quality': single_factor_results['quality_returns'],
            'Value': single_factor_results['value_returns'],
            'Momentum': single_factor_results['momentum_returns']
        }

        # Define performance metrics calculation function
        def calculate_performance_metrics(returns, benchmark, periods_per_year=252):
            """Calculate performance metrics."""
            first_trade_date = returns.loc[returns.ne(0)].index.min()
            if pd.isna(first_trade_date):
                return {metric: 0.0 for metric in ['Annualized Return (%)', 'Annualized Volatility (%)', 'Sharpe Ratio', 'Max Drawdown (%)']}
            
            aligned_returns = returns.loc[first_trade_date:]
            aligned_benchmark = benchmark.loc[first_trade_date:]

            n_years = len(aligned_returns) / periods_per_year
            annualized_return = ((1 + aligned_returns).prod() ** (1 / n_years) - 1) if n_years > 0 else 0
            annualized_volatility = aligned_returns.std() * np.sqrt(periods_per_year)
            sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else 0.0
            
            cumulative_returns = (1 + aligned_returns).cumprod()
            max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()
            
            return {
                'Annualized Return (%)': annualized_return * 100,
                'Annualized Volatility (%)': annualized_volatility * 100,
                'Sharpe Ratio': sharpe_ratio,
                'Max Drawdown (%)': max_drawdown * 100
            }

        comparison_data = []
        for strategy_name, returns in strategies.items():
            metrics = calculate_performance_metrics(returns, benchmark_returns)
            comparison_data.append({
                'Strategy': strategy_name,
                **metrics
            })

        comparison_df = pd.DataFrame(comparison_data)
        print("\n📊 Performance Comparison Table:")
        print(comparison_df.to_string(index=False, float_format='%.2f'))

        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Single Factor Strategies vs QVM Composite', fontsize=16, fontweight='bold')

        # Cumulative performance comparison
        ax1 = axes[0, 0]
        for strategy_name, returns in strategies.items():
            (1 + returns).cumprod().plot(ax=ax1, label=strategy_name, lw=2)
        ax1.set_title('Cumulative Performance')
        ax1.set_ylabel('Growth of 1 VND')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Annual returns comparison
        ax2 = axes[0, 1]
        annual_returns = {}
        for strategy_name, returns in strategies.items():
            annual_returns[strategy_name] = returns.resample('Y').apply(lambda x: (1+x).prod()-1) * 100
        
        annual_df = pd.DataFrame(annual_returns)
        annual_df.plot(kind='bar', ax=ax2)
        ax2.set_title('Annual Returns Comparison')
        ax2.set_ylabel('Return (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Risk-return scatter
        ax3 = axes[1, 0]
        for _, row in comparison_df.iterrows():
            ax3.scatter(row['Annualized Volatility (%)'], row['Annualized Return (%)'], 
                       s=100, label=row['Strategy'], alpha=0.7)
        ax3.set_xlabel('Volatility (%)')
        ax3.set_ylabel('Return (%)')
        ax3.set_title('Risk-Return Profile')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Sharpe ratio comparison
        ax4 = axes[1, 1]
        comparison_df.plot(x='Strategy', y='Sharpe Ratio', kind='bar', ax=ax4, color='skyblue')
        ax4.set_title('Sharpe Ratio Comparison')
        ax4.set_ylabel('Sharpe Ratio')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Factor Effectiveness Analysis
        print("\n" + "="*80)
        print("🔍 FACTOR EFFECTIVENESS ANALYSIS")
        print("="*80)

        print("\n📈 Factor Performance Summary:")
        for _, row in comparison_df.iterrows():
            print(f"   {row['Strategy']:15} | Return: {row['Annualized Return (%)']:6.2f}% | "
                  f"Vol: {row['Annualized Volatility (%)']:6.2f}% | Sharpe: {row['Sharpe Ratio']:5.2f} | "
                  f"MaxDD: {row['Max Drawdown (%)']:6.2f}%")

        print("\n🎯 Key Insights:")
        print("   - Quality Factor: Focuses on profitability and efficiency")
        print("   - Value Factor: Targets undervalued stocks with low P/E ratios")
        print("   - Momentum Factor: Captures price momentum across multiple horizons")
        print("   - QVM Composite: Combines all three factors with regime detection")

        print("\n✅ Single factor strategies successfully executed and analyzed!")
        
        return comparison_df

    except Exception as e:
        print(f"❌ An error occurred during single factor strategy execution: {e}")
        raise

if __name__ == "__main__":
    print("This script should be run from within the notebook after the main QVM strategy execution.")
    print("Please ensure the following variables are available:")
    print("- QVM_CONFIG")
    print("- price_data_raw")
    print("- fundamental_data_raw")
    print("- daily_returns_matrix")
    print("- benchmark_returns")
    print("- engine")
    print("- qvm_net_returns") 