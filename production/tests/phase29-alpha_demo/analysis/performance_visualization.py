# %% [markdown]
# # Performance Visualization for Component Contribution Analysis
#
# This script generates comprehensive visualizations comparing the performance of different QVM Engine v3j strategy variants.
#
# **File:** analysis/performance_visualization.py

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import importlib.util
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# %%
# Load the component comparison results
results_df = pd.read_csv('component_comparison_results.csv')
print("Loaded component comparison results:")
print(results_df)

# %%
# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('QVM Engine v3j Component Contribution Analysis', fontsize=16, fontweight='bold')

# %%
# 1. Sharpe Ratio Comparison
ax1 = axes[0, 0]
strategies = results_df['Strategy']
sharpe_ratios = results_df['Sharpe Ratio']
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

bars = ax1.bar(strategies, sharpe_ratios, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
ax1.set_title('Sharpe Ratio Comparison', fontsize=14, fontweight='bold')
ax1.set_ylabel('Sharpe Ratio', fontsize=12)
ax1.grid(True, alpha=0.3)

# Add value labels on bars
for bar, value in zip(bars, sharpe_ratios):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

# %%
# 2. Risk-Return Scatter Plot
ax2 = axes[0, 1]
volatility = results_df['Annualized Volatility (%)']
returns = results_df['Annualized Return (%)']

scatter = ax2.scatter(volatility, returns, s=200, c=sharpe_ratios, cmap='viridis', 
                     alpha=0.8, edgecolors='black', linewidth=1)

# Add strategy labels
for i, strategy in enumerate(strategies):
    ax2.annotate(strategy, (volatility.iloc[i], returns.iloc[i]), 
                xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')

ax2.set_xlabel('Annualized Volatility (%)', fontsize=12)
ax2.set_ylabel('Annualized Return (%)', fontsize=12)
ax2.set_title('Risk-Return Profile', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax2)
cbar.set_label('Sharpe Ratio', fontsize=10)

# %%
# 3. Maximum Drawdown Comparison
ax3 = axes[1, 0]
max_drawdowns = results_df['Max Drawdown (%)']

bars = ax3.bar(strategies, max_drawdowns, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
ax3.set_title('Maximum Drawdown Comparison', fontsize=14, fontweight='bold')
ax3.set_ylabel('Maximum Drawdown (%)', fontsize=12)
ax3.grid(True, alpha=0.3)

# Add value labels on bars
for bar, value in zip(bars, max_drawdowns):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height - 2,
             f'{value:.1f}%', ha='center', va='top', fontweight='bold', color='white')

# %%
# 4. Component Contribution Analysis
ax4 = axes[1, 1]

# Calculate improvements over base strategy
base_sharpe = results_df[results_df['Strategy'] == 'Base']['Sharpe Ratio'].iloc[0]
improvements = []
labels = []

for strategy in strategies:
    if strategy != 'Base':
        strategy_sharpe = results_df[results_df['Strategy'] == strategy]['Sharpe Ratio'].iloc[0]
        improvement = strategy_sharpe - base_sharpe
        improvement_pct = (improvement / base_sharpe) * 100
        improvements.append(improvement_pct)
        labels.append(f'{strategy}\n(+{improvement_pct:.1f}%)')

bars = ax4.bar(labels, improvements, color=['#4ECDC4', '#45B7D1', '#96CEB4'], 
               alpha=0.8, edgecolor='black', linewidth=1)
ax4.set_title('Sharpe Ratio Improvement Over Base Strategy', fontsize=14, fontweight='bold')
ax4.set_ylabel('Improvement (%)', fontsize=12)
ax4.grid(True, alpha=0.3)

# Add value labels on bars
for bar, value in zip(bars, improvements):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 5,
             f'+{value:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('insights/component_performance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# Create detailed performance metrics table
print("\n" + "="*80)
print("DETAILED PERFORMANCE METRICS COMPARISON")
print("="*80)

# Format the results for better display
display_df = results_df.copy()
display_df['Annualized Return (%)'] = display_df['Annualized Return (%)'].round(2)
display_df['Annualized Volatility (%)'] = display_df['Annualized Volatility (%)'].round(2)
display_df['Sharpe Ratio'] = display_df['Sharpe Ratio'].round(3)
display_df['Max Drawdown (%)'] = display_df['Max Drawdown (%)'].round(2)
display_df['Calmar Ratio'] = display_df['Calmar Ratio'].round(3)
display_df['Information Ratio'] = display_df['Information Ratio'].round(3)
display_df['Beta'] = display_df['Beta'].round(3)

print(display_df.to_string(index=False))

# %%
# Component contribution summary
print("\n" + "="*80)
print("COMPONENT CONTRIBUTION SUMMARY")
print("="*80)

base_metrics = results_df[results_df['Strategy'] == 'Base'].iloc[0]

for strategy in ['Regime_Only', 'Factors_Only', 'Integrated']:
    strategy_metrics = results_df[results_df['Strategy'] == strategy].iloc[0]
    
    sharpe_improvement = strategy_metrics['Sharpe Ratio'] - base_metrics['Sharpe Ratio']
    sharpe_improvement_pct = (sharpe_improvement / base_metrics['Sharpe Ratio']) * 100
    
    vol_reduction = (base_metrics['Annualized Volatility (%)'] - strategy_metrics['Annualized Volatility (%)']) / base_metrics['Annualized Volatility (%)'] * 100
    return_improvement = (strategy_metrics['Annualized Return (%)'] - base_metrics['Annualized Return (%)']) / base_metrics['Annualized Return (%)'] * 100
    drawdown_improvement = (base_metrics['Max Drawdown (%)'] - strategy_metrics['Max Drawdown (%)']) / abs(base_metrics['Max Drawdown (%)']) * 100
    
    print(f"\n{strategy}:")
    print(f"  Sharpe Ratio: {strategy_metrics['Sharpe Ratio']:.3f} (Base: {base_metrics['Sharpe Ratio']:.3f})")
    print(f"  Sharpe Improvement: +{sharpe_improvement:.3f} (+{sharpe_improvement_pct:.1f}%)")
    print(f"  Volatility: {strategy_metrics['Annualized Volatility (%)']:.2f}% (Base: {base_metrics['Annualized Volatility (%)']:.2f}%)")
    print(f"  Volatility Reduction: {vol_reduction:.1f}%")
    print(f"  Return: {strategy_metrics['Annualized Return (%)']:.2f}% (Base: {base_metrics['Annualized Return (%)']:.2f}%)")
    print(f"  Return Improvement: {return_improvement:.1f}%")
    print(f"  Max Drawdown: {strategy_metrics['Max Drawdown (%)']:.2f}% (Base: {base_metrics['Max Drawdown (%)']:.2f}%)")
    print(f"  Drawdown Improvement: {drawdown_improvement:.1f}%")

# %%
# Create a summary insights table
print("\n" + "="*80)
print("KEY INSIGHTS SUMMARY")
print("="*80)

insights_data = {
    'Metric': [
        'Best Sharpe Ratio',
        'Best Risk-Adjusted Returns',
        'Lowest Volatility',
        'Lowest Drawdown',
        'Highest Return',
        'Lowest Beta',
        'Best Calmar Ratio'
    ],
    'Strategy': [
        'Integrated (0.393)',
        'Integrated (0.393)',
        'Integrated (13.47%)',
        'Integrated (-44.44%)',
        'Factors_Only (6.96%)',
        'Integrated (0.573)',
        'Integrated (0.119)'
    ],
    'Component Contribution': [
        'Regime + Factors synergy',
        'Optimal risk-return balance',
        'Regime detection effect',
        'Combined risk management',
        'Factor selection effect',
        'Reduced market sensitivity',
        'Superior risk management'
    ]
}

insights_df = pd.DataFrame(insights_data)
print(insights_df.to_string(index=False))

# %%
# Save insights to file
with open('insights/performance_visualization_insights.md', 'w') as f:
    f.write("# Performance Visualization Insights\n\n")
    f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write("## Key Performance Metrics\n\n")
    f.write(display_df.to_markdown(index=False))
    f.write("\n\n")
    
    f.write("## Component Contribution Summary\n\n")
    for strategy in ['Regime_Only', 'Factors_Only', 'Integrated']:
        strategy_metrics = results_df[results_df['Strategy'] == strategy].iloc[0]
        sharpe_improvement = strategy_metrics['Sharpe Ratio'] - base_metrics['Sharpe Ratio']
        sharpe_improvement_pct = (sharpe_improvement / base_metrics['Sharpe Ratio']) * 100
        
        f.write(f"### {strategy}\n")
        f.write(f"- Sharpe Ratio: {strategy_metrics['Sharpe Ratio']:.3f} (+{sharpe_improvement_pct:.1f}% over Base)\n")
        f.write(f"- Annualized Return: {strategy_metrics['Annualized Return (%)']:.2f}%\n")
        f.write(f"- Annualized Volatility: {strategy_metrics['Annualized Volatility (%)']:.2f}%\n")
        f.write(f"- Max Drawdown: {strategy_metrics['Max Drawdown (%)']:.2f}%\n\n")

print(f"\nVisualization complete! Charts saved to 'insights/component_performance_comparison.png'")
print(f"Detailed insights saved to 'insights/performance_visualization_insights.md'") 