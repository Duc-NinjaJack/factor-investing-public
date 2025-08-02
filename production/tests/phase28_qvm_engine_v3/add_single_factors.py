#!/usr/bin/env python3
"""
Script to add single factor strategies to the QVM Engine v3c notebook.
This script appends the single factor execution code to the notebook.
"""

import re

def add_single_factors_to_notebook():
    """Add single factor strategies execution to the notebook."""
    
    notebook_path = "28_qvm_engine_v3c.ipynb"
    
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if single factors are already added
    if "SINGLE FACTOR STRATEGIES EXECUTION" in content:
        print("Single factor strategies already present in notebook.")
        return
    
    # Find the end of the QVM execution section
    # Look for the end of the main execution cell
    pattern = r'(except Exception as e:\s*print\(f"❌ An error occurred during the QVM Engine v3 execution: \{e\}"\)\s*raise\s*```\s*Cell \d+:\s*```\s*## CELL \d+: SUMMARY AND CONCLUSIONS)'
    
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        print("Could not find the end of QVM execution section.")
        return
    
    # Prepare the single factor code to insert
    single_factor_code = '''
    # ============================================================================
    # SINGLE FACTOR STRATEGIES EXECUTION
    # ============================================================================
    
    print("\\n" + "="*80)
    print("📊 SINGLE FACTOR STRATEGIES EXECUTION")
    print("="*80)

    # Import and run single factor strategies
    from single_factors import run_single_factors
    
    try:
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
        print("\\n" + "="*80)
        print("📊 SINGLE FACTOR STRATEGIES: PERFORMANCE COMPARISON")
        print("="*80)

        # Calculate metrics for all strategies
        strategies = {
            'QVM Composite': single_factor_results['qvm_returns'],
            'Quality': single_factor_results['quality_returns'],
            'Value': single_factor_results['value_returns'],
            'Momentum': single_factor_results['momentum_returns']
        }

        comparison_data = []
        for strategy_name, returns in strategies.items():
            metrics = calculate_performance_metrics(returns, benchmark_returns)
            comparison_data.append({
                'Strategy': strategy_name,
                **metrics
            })

        comparison_df = pd.DataFrame(comparison_data)
        print("\\n📊 Performance Comparison Table:")
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
        print("\\n" + "="*80)
        print("🔍 FACTOR EFFECTIVENESS ANALYSIS")
        print("="*80)

        print("\\n📈 Factor Performance Summary:")
        for _, row in comparison_df.iterrows():
            print(f"   {row['Strategy']:15} | Return: {row['Annualized Return (%)']:6.2f}% | "
                  f"Vol: {row['Annualized Volatility (%)']:6.2f}% | Sharpe: {row['Sharpe Ratio']:5.2f} | "
                  f"MaxDD: {row['Max Drawdown (%)']:6.2f}%")

        print("\\n🎯 Key Insights:")
        print("   - Quality Factor: Focuses on profitability and efficiency")
        print("   - Value Factor: Targets undervalued stocks with low P/E ratios")
        print("   - Momentum Factor: Captures price momentum across multiple horizons")
        print("   - QVM Composite: Combines all three factors with regime detection")

        print("\\n✅ Single factor strategies successfully executed and analyzed!")

    except Exception as e:
        print(f"❌ An error occurred during single factor strategy execution: {e}")
        raise

'''
    
    # Insert the single factor code before the summary section
    new_content = content.replace(match.group(1), single_factor_code + match.group(1))
    
    # Write the updated notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("✅ Single factor strategies successfully added to notebook!")

if __name__ == "__main__":
    add_single_factors_to_notebook() 