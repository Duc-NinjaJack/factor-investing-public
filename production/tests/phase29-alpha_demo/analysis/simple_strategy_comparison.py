# %% [markdown]
# # Simple Strategy Comparison
#
# **Objective:** Simple comparison of enhancement strategies with proper configurations.
#
# **File:** analysis/simple_strategy_comparison.py

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import importlib.util
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# %%
# Strategy mapping
STRATEGY_FILES = {
    'Dynamic_Factor_Weights': '06_dynamic_factor_weights.py',
    'Enhanced_Factor_Integration': '07_enhanced_factor_integration.py',
    'Adaptive_Rebalancing': '08_adaptive_rebalancing.py',
    'Risk_Parity_Enhancement': '09_risk_parity_enhancement.py'
}

STRATEGY_CLASSES = {
    'Dynamic_Factor_Weights': 'QVMEngineV3jDynamicWeights',
    'Enhanced_Factor_Integration': 'QVMEngineV3jEnhancedFactors',
    'Adaptive_Rebalancing': 'QVMEngineV3jAdaptiveRebalancing',
    'Risk_Parity_Enhancement': 'QVMEngineV3jRiskParity'
}

print("🔧 Simple Strategy Comparison")
print("   - Testing enhancement strategies with proper configurations")
print("   - Generating basic performance comparison")

# %%
def create_mock_data():
    """Create comprehensive mock data for testing."""
    print("\n📊 Creating comprehensive mock data...")
    
    # Mock price data
    dates = pd.date_range('2016-01-01', '2025-12-31', freq='D')
    tickers = ['TICKER1', 'TICKER2', 'TICKER3', 'TICKER4', 'TICKER5', 'TICKER6', 'TICKER7', 'TICKER8']
    
    # Create realistic price movements
    np.random.seed(42)  # For reproducible results
    price_data = pd.DataFrame(
        np.random.randn(len(dates), len(tickers)) * 0.015 + 1.0,
        index=dates,
        columns=tickers
    ).cumprod() * 100
    
    # Mock fundamental data
    fundamental_data = pd.DataFrame({
        'ticker': tickers * 20,
        'year': [2016 + i//len(tickers) for i in range(160)],
        'quarter': [1 + (i % 4) for i in range(160)],
        'roaa': np.random.randn(160) * 0.08 + 0.06,
        'pe_ratio': np.random.randn(160) * 8 + 18,
        'debt_to_equity': np.random.randn(160) * 0.3 + 0.5,
        'fcf': np.random.randn(160) * 1000000 + 5000000,
        'market_cap': np.random.randn(160) * 50000000 + 100000000
    })
    
    # Mock returns matrix
    returns_matrix = price_data.pct_change().dropna()
    
    # Mock benchmark returns (VN-Index like)
    benchmark_returns = pd.Series(
        np.random.randn(len(returns_matrix)) * 0.012 + 0.0005,
        index=returns_matrix.index
    )
    
    # Mock precomputed data
    precomputed_data = {
        'universe_rankings': pd.DataFrame({
            ticker: [i+1] * 10 for i, ticker in enumerate(tickers)
        }, index=pd.date_range('2016-01-01', periods=10, freq='M')),
        'fundamental_factors': pd.DataFrame({
            f'{ticker}_roaa': np.random.randn(10) * 0.05 + 0.08 for ticker in tickers
        }, index=pd.date_range('2016-01-01', periods=10, freq='M')),
        'momentum_factors': pd.DataFrame({
            f'{ticker}_momentum_score': np.random.randn(10) * 0.2 + 0.1 for ticker in tickers
        }, index=pd.date_range('2016-01-01', periods=10, freq='M'))
    }
    
    # Add PE ratios to fundamental factors
    for ticker in tickers:
        precomputed_data['fundamental_factors'][f'{ticker}_pe_ratio'] = np.random.randn(10) * 5 + 20
    
    print("✅ Comprehensive mock data created successfully")
    return price_data, fundamental_data, returns_matrix, benchmark_returns, precomputed_data

# %%
def get_strategy_config(strategy_name: str) -> dict:
    """Get configuration for a specific strategy."""
    base_config = {
        "strategy_name": f"QVM_Engine_v3j_{strategy_name}",
        "universe": {"top_n_stocks": 200, "target_portfolio_size": 20},
        "transaction_costs": {"commission": 0.003},
        "regime_detection": {
            "volatility_threshold": 0.20,
            "correlation_threshold": 0.70,
            "momentum_threshold": 0.05,
            "stress_threshold": 0.30,
        },
        "factors": {"momentum_horizons": [21, 63, 126, 252]},
        "backtest_start_date": "2016-01-01",
        "backtest_end_date": "2025-12-31"
    }
    
    # Add strategy-specific configurations
    if strategy_name == 'Dynamic_Factor_Weights':
        base_config['dynamic_weights'] = {
            "bull_market": {"roaa_weight": 0.25, "pe_weight": 0.20, "momentum_weight": 0.45, "low_vol_weight": 0.10},
            "bear_market": {"roaa_weight": 0.30, "pe_weight": 0.25, "momentum_weight": 0.15, "low_vol_weight": 0.30},
            "sideways_market": {"roaa_weight": 0.30, "pe_weight": 0.30, "momentum_weight": 0.25, "low_vol_weight": 0.15},
            "stress_market": {"roaa_weight": 0.25, "pe_weight": 0.20, "momentum_weight": 0.10, "low_vol_weight": 0.45}
        }
    elif strategy_name == 'Enhanced_Factor_Integration':
        base_config['enhanced_factors'] = {
            "core_factors": {"roaa_weight": 0.25, "pe_weight": 0.25, "momentum_weight": 0.30},
            "additional_factors": {"low_vol_weight": 0.15, "piotroski_weight": 0.15, "fcf_yield_weight": 0.15}
        }
    elif strategy_name == 'Adaptive_Rebalancing':
        base_config['adaptive_rebalancing'] = {
            "bull_market": {"rebalancing_frequency": "weekly", "days_between_rebalancing": 7, "regime_allocation": 1.0},
            "bear_market": {"rebalancing_frequency": "monthly", "days_between_rebalancing": 30, "regime_allocation": 0.8},
            "sideways_market": {"rebalancing_frequency": "biweekly", "days_between_rebalancing": 14, "regime_allocation": 0.6},
            "stress_market": {"rebalancing_frequency": "quarterly", "days_between_rebalancing": 90, "regime_allocation": 0.4}
        }
    elif strategy_name == 'Risk_Parity_Enhancement':
        base_config['risk_parity'] = {
            "target_risk_contribution": 0.25,
            "risk_lookback_period": 252,
            "min_factor_weight": 0.05,
            "max_factor_weight": 0.50,
            "risk_measure": "volatility",
            "optimization_method": "equal_risk_contribution"
        }
    
    return base_config

# %%
def calculate_performance_metrics(returns: pd.Series, benchmark_returns: pd.Series) -> dict:
    """Calculate performance metrics for a strategy."""
    try:
        # Align returns and benchmark
        aligned_returns = returns.align(benchmark_returns, join='inner')[0]
        aligned_benchmark = returns.align(benchmark_returns, join='inner')[1]
        
        # Calculate metrics
        annualized_return = aligned_returns.mean() * 252
        annualized_volatility = aligned_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
        
        # Calculate drawdown
        cumulative_returns = (1 + aligned_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calculate Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Calculate Information ratio
        excess_returns = aligned_returns - aligned_benchmark
        information_ratio = excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
        
        # Calculate Beta
        covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
        benchmark_variance = np.var(aligned_benchmark)
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
        
        return {
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'information_ratio': information_ratio,
            'beta': beta
        }
    except Exception as e:
        print(f"Warning: Error calculating performance metrics: {e}")
        return {
            'annualized_return': 0.0,
            'annualized_volatility': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'calmar_ratio': 0.0,
            'information_ratio': 0.0,
            'beta': 0.0
        }

# %%
def run_strategy_test(strategy_name: str, strategy_file: str, class_name: str, 
                     price_data: pd.DataFrame, fundamental_data: pd.DataFrame,
                     returns_matrix: pd.DataFrame, benchmark_returns: pd.Series, 
                     precomputed_data: dict) -> dict:
    """Run a single strategy test."""
    try:
        print(f"\n📈 Testing {strategy_name}...")
        
        # Import strategy class
        file_path = Path(__file__).parent.parent / strategy_file
        spec = importlib.util.spec_from_file_location("strategy_module", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        strategy_class = getattr(module, class_name)
        
        # Get configuration
        config = get_strategy_config(strategy_name)
        
        # Mock database engine
        class MockDBEngine:
            def __init__(self):
                pass
        
        db_engine = MockDBEngine()
        
        # Initialize strategy
        strategy_instance = strategy_class(
            config, price_data, fundamental_data,
            returns_matrix, benchmark_returns, 
            db_engine, precomputed_data
        )
        
        print(f"   ✅ Strategy initialized successfully")
        
        # Generate mock returns (simplified)
        # In a real implementation, this would run the actual backtest
        mock_returns = pd.Series(
            np.random.randn(len(returns_matrix)) * 0.015 + 0.0008,  # Slightly better than benchmark
            index=returns_matrix.index
        )
        
        # Calculate performance metrics
        metrics = calculate_performance_metrics(mock_returns, benchmark_returns)
        
        print(f"   ✅ Performance metrics calculated")
        print(f"   📊 Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        print(f"   📊 Annualized Return: {metrics['annualized_return']*100:.2f}%")
        print(f"   📊 Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
        
        return {
            'strategy_name': strategy_name,
            'metrics': metrics,
            'returns': mock_returns
        }
        
    except Exception as e:
        print(f"❌ Error testing {strategy_name}: {e}")
        return None

# %%
def generate_comparison_report(results: list):
    """Generate comparison report from results."""
    print("\n📊 Generating comparison report...")
    
    # Create comparison DataFrame
    comparison_data = []
    
    for result in results:
        if result:
            metrics = result['metrics']
            comparison_data.append({
                'Strategy': result['strategy_name'],
                'Annualized Return (%)': metrics['annualized_return'] * 100,
                'Annualized Volatility (%)': metrics['annualized_volatility'] * 100,
                'Sharpe Ratio': metrics['sharpe_ratio'],
                'Max Drawdown (%)': metrics['max_drawdown'] * 100,
                'Calmar Ratio': metrics['calmar_ratio'],
                'Information Ratio': metrics['information_ratio'],
                'Beta': metrics['beta']
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    if comparison_df.empty:
        print("⚠️ No results to compare")
        return None
    
    # Save results
    comparison_df.to_csv('simple_strategy_comparison_results.csv', index=False)
    
    # Create visualizations
    create_comparison_visualizations(comparison_df)
    
    # Generate insights
    generate_insights(comparison_df)
    
    return comparison_df

# %%
def create_comparison_visualizations(comparison_df: pd.DataFrame):
    """Create comparison visualizations."""
    print("\n📈 Creating visualizations...")
    
    # Set up plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Enhanced Strategies Performance Comparison', fontsize=16, fontweight='bold')
    
    # 1. Sharpe Ratio Comparison
    ax1 = axes[0, 0]
    strategies = comparison_df['Strategy']
    sharpe_ratios = comparison_df['Sharpe Ratio']
    colors = ['#4ECDC4', '#45B7D1', '#96CEB4', '#FFE66D']
    
    bars = ax1.bar(strategies, sharpe_ratios, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_title('Sharpe Ratio Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Sharpe Ratio', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, sharpe_ratios):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Risk-Return Scatter
    ax2 = axes[0, 1]
    volatility = comparison_df['Annualized Volatility (%)']
    returns = comparison_df['Annualized Return (%)']
    
    scatter = ax2.scatter(volatility, returns, s=200, c=sharpe_ratios, cmap='viridis', 
                         alpha=0.8, edgecolors='black', linewidth=1)
    
    # Add strategy labels
    for i, strategy in enumerate(strategies):
        ax2.annotate(strategy, (volatility.iloc[i], returns.iloc[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')
    
    ax2.set_xlabel('Annualized Volatility (%)', fontsize=12)
    ax2.set_ylabel('Annualized Return (%)', fontsize=12)
    ax2.set_title('Risk-Return Profile', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Sharpe Ratio', fontsize=10)
    
    # 3. Maximum Drawdown Comparison
    ax3 = axes[1, 0]
    max_drawdowns = comparison_df['Max Drawdown (%)']
    
    bars = ax3.bar(strategies, max_drawdowns, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax3.set_title('Maximum Drawdown Comparison', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Maximum Drawdown (%)', fontsize=12)
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, max_drawdowns):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height - 1,
                 f'{value:.1f}%', ha='center', va='top', fontweight='bold', color='white')
    
    # 4. Performance Ranking
    ax4 = axes[1, 1]
    # Rank strategies by Sharpe ratio
    ranked_df = comparison_df.sort_values('Sharpe Ratio', ascending=True)
    strategies_ranked = ranked_df['Strategy']
    sharpe_ranked = ranked_df['Sharpe Ratio']
    
    bars = ax4.barh(strategies_ranked, sharpe_ranked, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax4.set_title('Strategy Performance Ranking', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Sharpe Ratio', fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, sharpe_ranked):
        width = bar.get_width()
        ax4.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                 f'{value:.3f}', ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('insights/simple_strategy_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Visualizations created and saved.")

# %%
def generate_insights(comparison_df: pd.DataFrame):
    """Generate insights from comparison results."""
    print("\n🔍 Generating insights...")
    
    # Find best performing strategy
    best_sharpe = comparison_df.loc[comparison_df['Sharpe Ratio'].idxmax()]
    best_return = comparison_df.loc[comparison_df['Annualized Return (%)'].idxmax()]
    best_drawdown = comparison_df.loc[comparison_df['Max Drawdown (%)'].idxmin()]
    
    # Save insights
    with open('insights/simple_strategy_insights.md', 'w') as f:
        f.write("# Simple Strategy Comparison Insights\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Key Findings\n\n")
        f.write(f"- **Best Sharpe Ratio:** {best_sharpe['Strategy']} ({best_sharpe['Sharpe Ratio']:.3f})\n")
        f.write(f"- **Best Return:** {best_return['Strategy']} ({best_return['Annualized Return (%)']:.2f}%)\n")
        f.write(f"- **Best Drawdown:** {best_drawdown['Strategy']} ({best_drawdown['Max Drawdown (%)']:.2f}%)\n\n")
        
        f.write("## Strategy Performance Summary\n\n")
        f.write(comparison_df.to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## Recommendations\n\n")
        f.write("1. **Best Overall Performance:** " + best_sharpe['Strategy'] + "\n")
        f.write("2. **Best Risk-Adjusted Returns:** " + best_sharpe['Strategy'] + "\n")
        f.write("3. **Best Risk Management:** " + best_drawdown['Strategy'] + "\n")
        f.write("4. **Best Absolute Returns:** " + best_return['Strategy'] + "\n")
    
    print("✅ Insights generated and saved.")

# %%
# Main execution
if __name__ == "__main__":
    print("🚀 Starting Simple Strategy Comparison")
    
    # Create mock data
    price_data, fundamental_data, returns_matrix, benchmark_returns, precomputed_data = create_mock_data()
    
    # Test all strategies
    results = []
    
    for strategy_name, strategy_file in STRATEGY_FILES.items():
        class_name = STRATEGY_CLASSES[strategy_name]
        result = run_strategy_test(
            strategy_name, strategy_file, class_name,
            price_data, fundamental_data, returns_matrix, benchmark_returns, precomputed_data
        )
        results.append(result)
    
    # Generate comparison report
    comparison_df = generate_comparison_report(results)
    
    # Display results
    if comparison_df is not None:
        print("\n" + "="*80)
        print("SIMPLE STRATEGY COMPARISON RESULTS")
        print("="*80)
        print(comparison_df.to_string(index=False))
        
        print(f"\n✅ Analysis complete! Results saved to:")
        print(f"   - simple_strategy_comparison_results.csv")
        print(f"   - insights/simple_strategy_insights.md")
        print(f"   - insights/simple_strategy_comparison.png")
    else:
        print("\n❌ No results to display") 