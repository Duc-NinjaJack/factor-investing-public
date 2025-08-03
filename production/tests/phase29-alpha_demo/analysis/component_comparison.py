# Component Comparison Analysis
"""
Component comparison analysis for QVM strategy variants.
This script runs all strategy variants and compares their performance to understand
the contribution of each component (regime detection, factors) to overall performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import warnings

# Add project root to path
try:
    current_path = Path.cwd()
    while not (current_path / 'production').is_dir():
        if current_path.parent == current_path:
            raise FileNotFoundError("Could not find the 'production' directory.")
        current_path = current_path.parent
    
    project_root = current_path
    
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    from production.database.connection import get_database_manager
    from components.base_engine import BaseEngine

except (ImportError, FileNotFoundError) as e:
    print(f"❌ ERROR: Could not import production modules: {e}")
    raise

warnings.filterwarnings('ignore')

class ComponentComparison:
    """
    Component comparison analysis for QVM strategy variants.
    """
    
    def __init__(self):
        self.base_engine = None
        self.engine = None
        self.results = {}
        
    def setup_database(self):
        """Setup database connection."""
        try:
            db_manager = get_database_manager()
            self.engine = db_manager.get_engine()
            
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            print(f"✅ Database connection established successfully.")
            
            self.base_engine = BaseEngine({}, self.engine)
            
        except Exception as e:
            print(f"❌ FAILED to connect to the database: {e}")
            raise
    
    def run_base_strategy(self, config: dict):
        """Run base strategy (equal weight, no factors, no regime)."""
        print("\n" + "="*80)
        print("🚀 RUNNING BASE STRATEGY")
        print("="*80)
        
        # Load data
        price_data_raw, fundamental_data_raw, daily_returns_matrix, benchmark_returns = self.base_engine.load_all_data_for_backtest(config, self.engine)
        precomputed_data = self.base_engine.precompute_all_data(config, self.engine)
        
        # Import and run base strategy
        from components.base_strategy import QVMEngineV3jBase
        
        qvm_engine = QVMEngineV3jBase(
            config=config,
            price_data=price_data_raw,
            fundamental_data=fundamental_data_raw,
            returns_matrix=daily_returns_matrix,
            benchmark_returns=benchmark_returns,
            db_engine=self.engine,
            precomputed_data=precomputed_data
        )
        
        net_returns, diagnostics = qvm_engine.run_backtest()
        
        # Calculate performance metrics
        metrics = self.base_engine.calculate_performance_metrics(net_returns, benchmark_returns)
        
        self.results['Base'] = {
            'returns': net_returns,
            'diagnostics': diagnostics,
            'metrics': metrics,
            'benchmark': benchmark_returns
        }
        
        print(f"✅ Base strategy completed. Sharpe Ratio: {metrics['Sharpe Ratio']:.3f}")
    
    def run_regime_only_strategy(self, config: dict):
        """Run regime-only strategy (regime detection, no factors)."""
        print("\n" + "="*80)
        print("🚀 RUNNING REGIME-ONLY STRATEGY")
        print("="*80)
        
        # Modify config for regime-only
        regime_config = config.copy()
        regime_config['strategy_name'] = 'QVM_Engine_v3j_Regime_Only'
        regime_config['factors']['roaa_weight'] = 0.0
        regime_config['factors']['pe_weight'] = 0.0
        regime_config['factors']['momentum_weight'] = 0.0
        
        # Load data
        price_data_raw, fundamental_data_raw, daily_returns_matrix, benchmark_returns = self.base_engine.load_all_data_for_backtest(regime_config, self.engine)
        precomputed_data = self.base_engine.precompute_all_data(regime_config, self.engine)
        
        # Import and run regime-only strategy
        from components.regime_only_strategy import QVMEngineV3jRegimeOnly
        
        qvm_engine = QVMEngineV3jRegimeOnly(
            config=regime_config,
            price_data=price_data_raw,
            fundamental_data=fundamental_data_raw,
            returns_matrix=daily_returns_matrix,
            benchmark_returns=benchmark_returns,
            db_engine=self.engine,
            precomputed_data=precomputed_data
        )
        
        net_returns, diagnostics = qvm_engine.run_backtest()
        
        # Calculate performance metrics
        metrics = self.base_engine.calculate_performance_metrics(net_returns, benchmark_returns)
        
        self.results['Regime_Only'] = {
            'returns': net_returns,
            'diagnostics': diagnostics,
            'metrics': metrics,
            'benchmark': benchmark_returns
        }
        
        print(f"✅ Regime-only strategy completed. Sharpe Ratio: {metrics['Sharpe Ratio']:.3f}")
    
    def run_factors_only_strategy(self, config: dict):
        """Run factors-only strategy (factors, no regime detection)."""
        print("\n" + "="*80)
        print("🚀 RUNNING FACTORS-ONLY STRATEGY")
        print("="*80)
        
        # Modify config for factors-only
        factors_config = config.copy()
        factors_config['strategy_name'] = 'QVM_Engine_v3j_Factors_Only'
        factors_config['factors']['roaa_weight'] = 0.3
        factors_config['factors']['pe_weight'] = 0.3
        factors_config['factors']['momentum_weight'] = 0.4
        
        # Load data
        price_data_raw, fundamental_data_raw, daily_returns_matrix, benchmark_returns = self.base_engine.load_all_data_for_backtest(factors_config, self.engine)
        precomputed_data = self.base_engine.precompute_all_data(factors_config, self.engine)
        
        # Import and run factors-only strategy
        from components.factors_only_strategy import QVMEngineV3jFactorsOnly
        
        qvm_engine = QVMEngineV3jFactorsOnly(
            config=factors_config,
            price_data=price_data_raw,
            fundamental_data=fundamental_data_raw,
            returns_matrix=daily_returns_matrix,
            benchmark_returns=benchmark_returns,
            db_engine=self.engine,
            precomputed_data=precomputed_data
        )
        
        net_returns, diagnostics = qvm_engine.run_backtest()
        
        # Calculate performance metrics
        metrics = self.base_engine.calculate_performance_metrics(net_returns, benchmark_returns)
        
        self.results['Factors_Only'] = {
            'returns': net_returns,
            'diagnostics': diagnostics,
            'metrics': metrics,
            'benchmark': benchmark_returns
        }
        
        print(f"✅ Factors-only strategy completed. Sharpe Ratio: {metrics['Sharpe Ratio']:.3f}")
    
    def run_integrated_strategy(self, config: dict):
        """Run integrated strategy (regime detection + factors)."""
        print("\n" + "="*80)
        print("🚀 RUNNING INTEGRATED STRATEGY")
        print("="*80)
        
        # Load data
        price_data_raw, fundamental_data_raw, daily_returns_matrix, benchmark_returns = self.base_engine.load_all_data_for_backtest(config, self.engine)
        precomputed_data = self.base_engine.precompute_all_data(config, self.engine)
        
        # Import and run integrated strategy
        from components.integrated_strategy import QVMEngineV3jIntegrated
        
        qvm_engine = QVMEngineV3jIntegrated(
            config=config,
            price_data=price_data_raw,
            fundamental_data=fundamental_data_raw,
            returns_matrix=daily_returns_matrix,
            benchmark_returns=benchmark_returns,
            db_engine=self.engine,
            precomputed_data=precomputed_data
        )
        
        net_returns, diagnostics = qvm_engine.run_backtest()
        
        # Calculate performance metrics
        metrics = self.base_engine.calculate_performance_metrics(net_returns, benchmark_returns)
        
        self.results['Integrated'] = {
            'returns': net_returns,
            'diagnostics': diagnostics,
            'metrics': metrics,
            'benchmark': benchmark_returns
        }
        
        print(f"✅ Integrated strategy completed. Sharpe Ratio: {metrics['Sharpe Ratio']:.3f}")
    
    def generate_comparison_report(self):
        """Generate comprehensive comparison report."""
        print("\n" + "="*80)
        print("📊 COMPONENT COMPARISON REPORT")
        print("="*80)
        
        # Create comparison DataFrame
        comparison_data = []
        for strategy_name, result in self.results.items():
            metrics = result['metrics']
            comparison_data.append({
                'Strategy': strategy_name,
                'Annualized Return (%)': metrics['Annualized Return (%)'],
                'Annualized Volatility (%)': metrics['Annualized Volatility (%)'],
                'Sharpe Ratio': metrics['Sharpe Ratio'],
                'Max Drawdown (%)': metrics['Max Drawdown (%)'],
                'Calmar Ratio': metrics['Calmar Ratio'],
                'Information Ratio': metrics['Information Ratio'],
                'Beta': metrics['Beta']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Display comparison table
        print("\n📈 Performance Comparison:")
        print(comparison_df.to_string(index=False, float_format='%.3f'))
        
        # Calculate component contributions
        print("\n🔍 Component Contribution Analysis:")
        base_sharpe = comparison_df[comparison_df['Strategy'] == 'Base']['Sharpe Ratio'].iloc[0]
        
        for strategy_name in ['Regime_Only', 'Factors_Only', 'Integrated']:
            if strategy_name in comparison_df['Strategy'].values:
                strategy_sharpe = comparison_df[comparison_df['Strategy'] == strategy_name]['Sharpe Ratio'].iloc[0]
                improvement = strategy_sharpe - base_sharpe
                improvement_pct = (improvement / base_sharpe) * 100 if base_sharpe != 0 else 0
                
                print(f"   - {strategy_name}: {improvement:+.3f} ({improvement_pct:+.1f}%)")
        
        # Generate comparison plots
        self._generate_comparison_plots()
        
        return comparison_df
    
    def _generate_comparison_plots(self):
        """Generate comparison plots."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('QVM Engine v3j - Component Comparison Analysis', fontsize=16, fontweight='bold')
        
        # 1. Cumulative Performance
        ax1 = axes[0, 0]
        for strategy_name, result in self.results.items():
            returns = result['returns']
            benchmark = result['benchmark']
            
            # Align data
            first_trade_date = returns.loc[returns.ne(0)].index.min()
            if pd.notna(first_trade_date):
                aligned_returns = returns.loc[first_trade_date:]
                (1 + aligned_returns).cumprod().plot(ax=ax1, label=strategy_name, alpha=0.8)
        
        ax1.set_title('Cumulative Performance')
        ax1.set_ylabel('Growth of 1 VND')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Sharpe Ratio Comparison
        ax2 = axes[0, 1]
        strategies = list(self.results.keys())
        sharpe_ratios = [self.results[s]['metrics']['Sharpe Ratio'] for s in strategies]
        
        bars = ax2.bar(strategies, sharpe_ratios, color=['#16A085', '#3498DB', '#E67E22', '#9B59B6'])
        ax2.set_title('Sharpe Ratio Comparison')
        ax2.set_ylabel('Sharpe Ratio')
        ax2.grid(True, axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, sharpe_ratios):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom')
        
        # 3. Maximum Drawdown Comparison
        ax3 = axes[1, 0]
        max_drawdowns = [self.results[s]['metrics']['Max Drawdown (%)'] for s in strategies]
        
        bars = ax3.bar(strategies, max_drawdowns, color=['#16A085', '#3498DB', '#E67E22', '#9B59B6'])
        ax3.set_title('Maximum Drawdown Comparison')
        ax3.set_ylabel('Max Drawdown (%)')
        ax3.grid(True, axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, max_drawdowns):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 1, 
                    f'{value:.1f}%', ha='center', va='top')
        
        # 4. Information Ratio Comparison
        ax4 = axes[1, 1]
        info_ratios = [self.results[s]['metrics']['Information Ratio'] for s in strategies]
        
        bars = ax4.bar(strategies, info_ratios, color=['#16A085', '#3498DB', '#E67E22', '#9B59B6'])
        ax4.set_title('Information Ratio Comparison')
        ax4.set_ylabel('Information Ratio')
        ax4.grid(True, axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, info_ratios):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def save_results(self, output_path: str = "component_comparison_results.csv"):
        """Save comparison results to CSV."""
        comparison_df = self.generate_comparison_report()
        comparison_df.to_csv(output_path, index=False)
        print(f"\n✅ Results saved to: {output_path}")

def main():
    """Main execution function."""
    
    # Configuration
    QVM_CONFIG = {
        "strategy_name": "QVM_Engine_v3j_Comparison",
        "backtest_start_date": "2016-01-01",
        "backtest_end_date": "2025-07-28",
        "rebalance_frequency": "M",
        "transaction_cost_bps": 30,
        
        "universe": {
            "lookback_days": 63,
            "top_n_stocks": 200,
            "max_position_size": 0.05,
            "max_sector_exposure": 0.30,
            "target_portfolio_size": 20,
        },
        
        "factors": {
            "roaa_weight": 0.3,
            "pe_weight": 0.3,
            "momentum_weight": 0.4,
            "momentum_horizons": [21, 63, 126, 252],
            "skip_months": 1,
            "fundamental_lag_days": 45,
        },
        
        "regime": {
            "lookback_period": 90,
            "volatility_threshold": 0.0140,
            "return_threshold": 0.0012,
            "low_return_threshold": 0.0002
        }
    }
    
    # Initialize comparison
    comparison = ComponentComparison()
    comparison.setup_database()
    
    # Run all strategies
    comparison.run_base_strategy(QVM_CONFIG)
    comparison.run_regime_only_strategy(QVM_CONFIG)
    comparison.run_factors_only_strategy(QVM_CONFIG)
    comparison.run_integrated_strategy(QVM_CONFIG)
    
    # Generate comparison report
    comparison.generate_comparison_report()
    
    # Save results
    comparison.save_results()
    
    print("\n✅ Component comparison analysis complete!")

if __name__ == "__main__":
    main() 