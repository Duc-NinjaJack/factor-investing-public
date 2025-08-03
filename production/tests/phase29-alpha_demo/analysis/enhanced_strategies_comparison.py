# %% [markdown]
# # Enhanced Strategies Comparison Analysis
#
# **Objective:** Comprehensive comparison of all QVM Engine v3j enhancement strategies.
# This script tests and compares the performance of all enhancement strategies against the original integrated strategy.
#
# **File:** analysis/enhanced_strategies_comparison.py
#
# **Strategies Tested:**
# - 04_integrated_strategy.py (Original baseline)
# - 06_dynamic_factor_weights.py (Regime-specific factor weights)
# - 07_enhanced_factor_integration.py (Additional factors)
# - 08_adaptive_rebalancing.py (Regime-aware rebalancing)
# - 09_risk_parity_enhancement.py (Risk parity allocation)

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
# Import database connection
try:
    from production.database.connection import get_database_manager
    from components.base_engine import BaseEngine
    print("✅ Database connection and components imported successfully.")
except ImportError as e:
    print(f"⚠️ Import error: {e}")
    print("   - Make sure you're running from the correct directory")
    print("   - Check that all required modules are available")

# %%
# Strategy mapping for dynamic imports
STRATEGY_FILES = {
    'Integrated_Baseline': '04_integrated_strategy.py',
    'Dynamic_Factor_Weights': '06_dynamic_factor_weights.py',
    'Enhanced_Factor_Integration': '07_enhanced_factor_integration.py',
    'Adaptive_Rebalancing': '08_adaptive_rebalancing.py',
    'Risk_Parity_Enhancement': '09_risk_parity_enhancement.py'
}

STRATEGY_CLASSES = {
    'Integrated_Baseline': 'QVMEngineV3jIntegrated',
    'Dynamic_Factor_Weights': 'QVMEngineV3jDynamicWeights',
    'Enhanced_Factor_Integration': 'QVMEngineV3jEnhancedFactors',
    'Adaptive_Rebalancing': 'QVMEngineV3jAdaptiveRebalancing',
    'Risk_Parity_Enhancement': 'QVMEngineV3jRiskParity'
}

print("🔧 Enhanced Strategies Comparison Analysis")
print("   - Strategies to test: 5 enhancement strategies")
print("   - Comparison baseline: Original integrated strategy")
print("   - Analysis focus: Performance improvement identification")

# %%
class EnhancedStrategiesComparison:
    """
    Comprehensive comparison of all QVM Engine v3j enhancement strategies.
    """
    
    def __init__(self):
        """Initialize the enhanced strategies comparison."""
        self.results = {}
        self.diagnostics = {}
        self.performance_metrics = {}
        
        # Initialize database connection
        try:
            self.db_manager = get_database_manager()
            self.db_engine = self.db_manager.get_engine()
            print("✅ Database connection established.")
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            return
        
        # Initialize data attributes
        self.price_data = pd.DataFrame()
        self.fundamental_data = pd.DataFrame()
        self.returns_matrix = pd.DataFrame()
        self.benchmark_returns = pd.Series()
        self.precomputed_data = {}
        
        # Load and precompute data
        self._load_and_precompute_data()
    
    def _load_and_precompute_data(self):
        """Load and precompute all required data."""
        try:
            print("\n📊 Loading and precomputing data...")
            
            # Initialize base engine for data loading
            base_config = {
                "universe": {"top_n_stocks": 200, "target_portfolio_size": 20},
                "factors": {"momentum_horizons": [21, 63, 126, 252]}
            }
            self.base_engine = BaseEngine(base_config, self.db_engine)
            
            # Precompute all data
            self.precomputed_data = self.base_engine.precompute_all_data()
            
            # Load additional data
            self.price_data = self._load_price_data()
            self.fundamental_data = self._load_fundamental_data()
            self.returns_matrix = self._load_returns_matrix()
            self.benchmark_returns = self._load_benchmark_returns()
            
            print("✅ Data loading and precomputation complete.")
            
        except Exception as e:
            print(f"❌ Data loading failed: {e}")
            raise
    
    def _load_price_data(self) -> pd.DataFrame:
        """Load historical price data."""
        try:
            query = """
            SELECT ticker, date, close_price 
            FROM equity_history 
            WHERE date >= '2016-01-01' AND date <= '2025-12-31'
            ORDER BY date, ticker
            """
            df = pd.read_sql(query, self.db_engine)
            return df.pivot(index='date', columns='ticker', values='close_price')
        except Exception as e:
            print(f"Warning: Error loading price data: {e}")
            return pd.DataFrame()
    
    def _load_fundamental_data(self) -> pd.DataFrame:
        """Load fundamental data."""
        try:
            query = """
            SELECT ticker, year, quarter, roaa, pe_ratio 
            FROM intermediary_calculations_enhanced 
            WHERE year >= 2016 AND year <= 2025
            ORDER BY year, quarter, ticker
            """
            df = pd.read_sql(query, self.db_engine)
            return df
        except Exception as e:
            print(f"Warning: Error loading fundamental data: {e}")
            return pd.DataFrame()
    
    def _load_returns_matrix(self) -> pd.DataFrame:
        """Load daily returns matrix."""
        try:
            # Calculate returns from price data
            if not self.price_data.empty:
                returns = self.price_data.pct_change().dropna()
                return returns
            return pd.DataFrame()
        except Exception as e:
            print(f"Warning: Error loading returns matrix: {e}")
            return pd.DataFrame()
    
    def _load_benchmark_returns(self) -> pd.Series:
        """Load benchmark returns."""
        try:
            # Use VN-Index as benchmark
            query = """
            SELECT date, close_price 
            FROM vcsc_daily_data_complete 
            WHERE ticker = 'VNINDEX' 
            AND date >= '2016-01-01' AND date <= '2025-12-31'
            ORDER BY date
            """
            df = pd.read_sql(query, self.db_engine)
            benchmark_prices = df.set_index('date')['close_price']
            benchmark_returns = benchmark_prices.pct_change().dropna()
            return benchmark_returns
        except Exception as e:
            print(f"Warning: Error loading benchmark returns: {e}")
            return pd.Series()
    
    def run_all_strategies(self):
        """Run all enhancement strategies and collect results."""
        print("\n🚀 Running all enhancement strategies...")
        
        for strategy_name, strategy_file in STRATEGY_FILES.items():
            print(f"\n📈 Testing {strategy_name}...")
            try:
                # Dynamically import strategy class
                strategy_class = self._import_strategy_class(strategy_file, STRATEGY_CLASSES[strategy_name])
                
                # Get strategy configuration
                config = self._get_strategy_config(strategy_name)
                
                # Initialize strategy
                strategy_instance = strategy_class(
                    config, self.price_data, self.fundamental_data,
                    self.returns_matrix, self.benchmark_returns, 
                    self.db_engine, self.precomputed_data
                )
                
                # Run backtest
                net_returns, diagnostics = strategy_instance.run_backtest()
                
                # Generate performance metrics
                metrics = strategy_instance.generate_comprehensive_tearsheet(net_returns, diagnostics)
                
                # Store results
                self.results[strategy_name] = net_returns
                self.diagnostics[strategy_name] = diagnostics
                self.performance_metrics[strategy_name] = metrics
                
                print(f"✅ {strategy_name} completed successfully.")
                
            except Exception as e:
                print(f"❌ Error running {strategy_name}: {e}")
                continue
        
        print(f"\n✅ All strategies completed. Results collected for {len(self.results)} strategies.")
    
    def _import_strategy_class(self, strategy_file: str, class_name: str):
        """Dynamically import strategy class from file."""
        try:
            file_path = Path(__file__).parent.parent / strategy_file
            spec = importlib.util.spec_from_file_location("strategy_module", file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return getattr(module, class_name)
        except Exception as e:
            print(f"Error importing {strategy_file}: {e}")
            raise
    
    def _get_strategy_config(self, strategy_name: str) -> dict:
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
            "factors": {"momentum_horizons": [21, 63, 126, 252]}
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
    
    def generate_comparison_report(self):
        """Generate comprehensive comparison report."""
        print("\n📊 Generating comprehensive comparison report...")
        
        # Create performance comparison DataFrame
        comparison_data = []
        
        for strategy_name, metrics in self.performance_metrics.items():
            if metrics:
                comparison_data.append({
                    'Strategy': strategy_name,
                    'Annualized Return (%)': metrics.get('annualized_return', 0) * 100,
                    'Annualized Volatility (%)': metrics.get('annualized_volatility', 0) * 100,
                    'Sharpe Ratio': metrics.get('sharpe_ratio', 0),
                    'Max Drawdown (%)': metrics.get('max_drawdown', 0) * 100,
                    'Calmar Ratio': metrics.get('calmar_ratio', 0),
                    'Information Ratio': metrics.get('information_ratio', 0),
                    'Beta': metrics.get('beta', 0)
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Check if we have any results
        if comparison_df.empty:
            print("⚠️ No strategy results available. Creating placeholder report.")
            comparison_df = pd.DataFrame({
                'Strategy': ['No Results Available'],
                'Annualized Return (%)': [0.0],
                'Annualized Volatility (%)': [0.0],
                'Sharpe Ratio': [0.0],
                'Max Drawdown (%)': [0.0],
                'Calmar Ratio': [0.0],
                'Information Ratio': [0.0],
                'Beta': [0.0]
            })
        
        # Save results
        comparison_df.to_csv('enhanced_strategies_comparison_results.csv', index=False)
        
        # Generate insights only if we have real data
        if not comparison_df.empty and 'No Results Available' not in comparison_df['Strategy'].values:
            self._generate_insights(comparison_df)
            self._create_visualizations(comparison_df)
        else:
            print("⚠️ Skipping insights generation due to insufficient data.")
        
        print("✅ Comparison report generated successfully.")
        return comparison_df
    
    def _generate_insights(self, comparison_df: pd.DataFrame):
        """Generate insights from comparison results."""
        print("\n🔍 Generating insights...")
        
        # Find best performing strategy
        best_sharpe = comparison_df.loc[comparison_df['Sharpe Ratio'].idxmax()]
        best_return = comparison_df.loc[comparison_df['Annualized Return (%)'].idxmax()]
        best_drawdown = comparison_df.loc[comparison_df['Max Drawdown (%)'].idxmin()]
        
        # Calculate improvements over baseline
        baseline = comparison_df[comparison_df['Strategy'] == 'Integrated_Baseline'].iloc[0]
        
        insights = {
            'best_sharpe_strategy': best_sharpe['Strategy'],
            'best_sharpe_ratio': best_sharpe['Sharpe Ratio'],
            'best_return_strategy': best_return['Strategy'],
            'best_return': best_return['Annualized Return (%)'],
            'best_drawdown_strategy': best_drawdown['Strategy'],
            'best_drawdown': best_drawdown['Max Drawdown (%)'],
            'improvements': {}
        }
        
        # Calculate improvements for each strategy
        for _, row in comparison_df.iterrows():
            if row['Strategy'] != 'Integrated_Baseline':
                sharpe_improvement = (row['Sharpe Ratio'] - baseline['Sharpe Ratio']) / baseline['Sharpe Ratio'] * 100
                return_improvement = (row['Annualized Return (%)'] - baseline['Annualized Return (%)']) / baseline['Annualized Return (%)'] * 100
                drawdown_improvement = (baseline['Max Drawdown (%)'] - row['Max Drawdown (%)']) / baseline['Max Drawdown (%)'] * 100
                
                insights['improvements'][row['Strategy']] = {
                    'sharpe_improvement_pct': sharpe_improvement,
                    'return_improvement_pct': return_improvement,
                    'drawdown_improvement_pct': drawdown_improvement
                }
        
        # Save insights
        with open('insights/enhanced_strategies_insights.md', 'w') as f:
            f.write("# Enhanced Strategies Comparison Insights\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Key Findings\n\n")
            f.write(f"- **Best Sharpe Ratio:** {insights['best_sharpe_strategy']} ({insights['best_sharpe_ratio']:.3f})\n")
            f.write(f"- **Best Return:** {insights['best_return_strategy']} ({insights['best_return']:.2f}%)\n")
            f.write(f"- **Best Drawdown:** {insights['best_drawdown_strategy']} ({insights['best_drawdown']:.2f}%)\n\n")
            
            f.write("## Strategy Improvements Over Baseline\n\n")
            for strategy, improvements in insights['improvements'].items():
                f.write(f"### {strategy}\n")
                f.write(f"- Sharpe Ratio: {improvements['sharpe_improvement_pct']:+.1f}%\n")
                f.write(f"- Annualized Return: {improvements['return_improvement_pct']:+.1f}%\n")
                f.write(f"- Max Drawdown: {improvements['drawdown_improvement_pct']:+.1f}%\n\n")
        
        print("✅ Insights generated and saved.")
    
    def _create_visualizations(self, comparison_df: pd.DataFrame):
        """Create performance comparison visualizations."""
        print("\n📈 Creating visualizations...")
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Enhanced Strategies Performance Comparison', fontsize=16, fontweight='bold')
        
        # 1. Sharpe Ratio Comparison
        ax1 = axes[0, 0]
        strategies = comparison_df['Strategy']
        sharpe_ratios = comparison_df['Sharpe Ratio']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFE66D']
        
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
            ax3.text(bar.get_x() + bar.get_width()/2., height - 2,
                     f'{value:.1f}%', ha='center', va='top', fontweight='bold', color='white')
        
        # 4. Improvement Over Baseline
        ax4 = axes[1, 1]
        baseline = comparison_df[comparison_df['Strategy'] == 'Integrated_Baseline']['Sharpe Ratio'].iloc[0]
        improvements = []
        labels = []
        
        for strategy in strategies:
            if strategy != 'Integrated_Baseline':
                strategy_sharpe = comparison_df[comparison_df['Strategy'] == strategy]['Sharpe Ratio'].iloc[0]
                improvement = (strategy_sharpe - baseline) / baseline * 100
                improvements.append(improvement)
                labels.append(f'{strategy}\n({improvement:+.1f}%)')
        
        bars = ax4.bar(labels, improvements, color=['#4ECDC4', '#45B7D1', '#96CEB4', '#FFE66D'], 
                       alpha=0.8, edgecolor='black', linewidth=1)
        ax4.set_title('Sharpe Ratio Improvement Over Baseline', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Improvement (%)', fontsize=12)
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, improvements):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + (1 if value > 0 else -1),
                     f'{value:+.1f}%', ha='center', va='bottom' if value > 0 else 'top', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('insights/enhanced_strategies_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualizations created and saved.")

# %%
# Main execution
if __name__ == "__main__":
    print("🚀 Starting Enhanced Strategies Comparison Analysis")
    
    # Initialize comparison
    comparison = EnhancedStrategiesComparison()
    
    # Run all strategies
    comparison.run_all_strategies()
    
    # Generate comparison report
    results_df = comparison.generate_comparison_report()
    
    # Display results
    print("\n" + "="*80)
    print("ENHANCED STRATEGIES COMPARISON RESULTS")
    print("="*80)
    print(results_df.to_string(index=False))
    
    print(f"\n✅ Analysis complete! Results saved to:")
    print(f"   - enhanced_strategies_comparison_results.csv")
    print(f"   - insights/enhanced_strategies_insights.md")
    print(f"   - insights/enhanced_strategies_comparison.png") 