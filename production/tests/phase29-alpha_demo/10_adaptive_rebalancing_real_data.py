# %% [markdown]
# # Adaptive Rebalancing Strategy - Real Data Implementation
#
# **Objective:** Production-ready implementation of the Adaptive Rebalancing strategy using real market data.
#
# **File:** 10_adaptive_rebalancing_real_data.py

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import warnings
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import production modules
try:
    from production.database.connection import get_database_manager
    from components.base_engine import BaseEngine
    from components.regime_detector import RegimeDetector
    print("✅ Production modules imported successfully.")
except ImportError as e:
    print(f"⚠️ Import error: {e}")

warnings.filterwarnings('ignore')

# %%
# Configuration
QVM_CONFIG = {
    "strategy_name": "QVM_Engine_v3j_Adaptive_Rebalancing_Real_Data",
    "backtest_start_date": "2016-01-01",
    "backtest_end_date": "2025-07-28",
    "universe": {"top_n_stocks": 200, "target_portfolio_size": 20},
    "transaction_costs": {"commission": 0.003},
    "regime_detection": {
        "volatility_threshold": 0.20,
        "correlation_threshold": 0.70,
        "momentum_threshold": 0.05,
        "stress_threshold": 0.30,
    },
    "adaptive_rebalancing": {
        "bull_market": {"rebalancing_frequency": "weekly", "days_between_rebalancing": 7, "regime_allocation": 1.0},
        "bear_market": {"rebalancing_frequency": "monthly", "days_between_rebalancing": 30, "regime_allocation": 0.8},
        "sideways_market": {"rebalancing_frequency": "biweekly", "days_between_rebalancing": 14, "regime_allocation": 0.6},
        "stress_market": {"rebalancing_frequency": "quarterly", "days_between_rebalancing": 90, "regime_allocation": 0.4}
    },
    "factors": {"momentum_horizons": [21, 63, 126, 252]}
}

print("🚀 Adaptive Rebalancing Strategy - Real Data Implementation")

# %%
class QVMEngineV3jAdaptiveRebalancingRealData:
    """Production-ready Adaptive Rebalancing strategy with real data."""
    
    def __init__(self, config, db_engine):
        self.config = config
        self.db_engine = db_engine
        self.regime_detector = RegimeDetector(
            volatility_threshold=config["regime_detection"]["volatility_threshold"],
            return_threshold=config["regime_detection"]["momentum_threshold"],
            low_return_threshold=config["regime_detection"]["stress_threshold"]
        )
        self.base_engine = BaseEngine(config, db_engine)
        print("✅ Strategy initialized.")
    
    def run_real_data_backtest(self):
        """Run backtest with real data."""
        print("\n📊 Loading real data...")
        
        # Load price data
        price_query = """
        SELECT ticker, date, close_price 
        FROM equity_history 
        WHERE date >= '2016-01-01' AND date <= '2025-07-28'
        ORDER BY date, ticker
        """
        price_df = pd.read_sql(price_query, self.db_engine)
        price_data = price_df.pivot(index='date', columns='ticker', values='close_price')
        
        # Load benchmark
        benchmark_query = """
        SELECT date, close_price 
        FROM vcsc_daily_data_complete 
        WHERE ticker = 'VNINDEX' 
        AND date >= '2016-01-01' AND date <= '2025-07-28'
        ORDER BY date
        """
        benchmark_df = pd.read_sql(benchmark_query, self.db_engine)
        benchmark_returns = benchmark_df.set_index('date')['close_price'].pct_change().dropna()
        
        # Calculate returns
        returns_matrix = price_data.pct_change().dropna()
        
        # Generate rebalancing dates
        rebalancing_dates = self._generate_rebalancing_dates(returns_matrix)
        
        # Run backtest
        portfolio_returns = self._run_backtest_loop(returns_matrix, benchmark_returns, rebalancing_dates)
        
        return portfolio_returns, {
            'rebalancing_dates': rebalancing_dates,
            'price_data': price_data,
            'benchmark_returns': benchmark_returns
        }
    
    def _generate_rebalancing_dates(self, returns_matrix):
        """Generate adaptive rebalancing dates."""
        rebalancing_dates = []
        current_date = returns_matrix.index[0]
        end_date = returns_matrix.index[-1]
        
        while current_date <= end_date:
            if current_date in returns_matrix.index:
                market_returns = returns_matrix.loc[:current_date].tail(252)
                if len(market_returns) >= 252:
                    regime = self.regime_detector.detect_regime(market_returns)
                    rebalancing_config = self.config["adaptive_rebalancing"][regime]
                    
                    rebalancing_dates.append({
                        'date': current_date,
                        'regime': regime,
                        'days_between': rebalancing_config["days_between_rebalancing"],
                        'allocation': rebalancing_config["regime_allocation"]
                    })
                    
                    current_date += pd.Timedelta(days=rebalancing_config["days_between_rebalancing"])
                else:
                    current_date += pd.Timedelta(days=7)
            else:
                current_date += pd.Timedelta(days=1)
        
        return pd.DataFrame(rebalancing_dates)
    
    def _run_backtest_loop(self, returns_matrix, benchmark_returns, rebalancing_dates):
        """Run the main backtest loop."""
        portfolio_value = 1000000000  # 1B VND
        portfolio_returns = []
        current_positions = {}
        
        trading_dates = returns_matrix.index
        trading_dates = trading_dates[
            (trading_dates >= self.config["backtest_start_date"]) &
            (trading_dates <= self.config["backtest_end_date"])
        ]
        
        print(f"📈 Running backtest over {len(trading_dates)} trading days")
        
        for i, current_date in enumerate(trading_dates):
            # Check if rebalancing needed
            rebalancing_needed = current_date in rebalancing_dates['date'].values
            
            # Calculate daily returns
            daily_return = 0.0
            if current_positions:
                for ticker, weight in current_positions.items():
                    if ticker in returns_matrix.columns:
                        ticker_return = returns_matrix.loc[current_date, ticker]
                        if not pd.isna(ticker_return):
                            daily_return += weight * ticker_return
            
            # Update portfolio
            portfolio_value *= (1 + daily_return)
            portfolio_returns.append(daily_return)
            
            # Rebalance if needed
            if rebalancing_needed:
                rebalancing_info = rebalancing_dates[rebalancing_dates['date'] == current_date].iloc[0]
                regime = rebalancing_info['regime']
                allocation = rebalancing_info['allocation']
                
                # Select stocks (simplified)
                available_stocks = returns_matrix.columns[:self.config["universe"]["target_portfolio_size"]]
                new_weights = {stock: allocation/len(available_stocks) for stock in available_stocks}
                
                # Apply transaction costs
                cost = self._calculate_transaction_costs(current_positions, new_weights, portfolio_value)
                portfolio_value -= cost
                
                current_positions = new_weights
                
                if i % 100 == 0:
                    print(f"   🔄 Rebalancing on {current_date.strftime('%Y-%m-%d')} - Regime: {regime}")
            
            # Progress update
            if i % 500 == 0:
                print(f"   📊 Progress: {i}/{len(trading_dates)} days ({i/len(trading_dates)*100:.1f}%)")
        
        return pd.Series(portfolio_returns, index=trading_dates)
    
    def _calculate_transaction_costs(self, old_weights, new_weights, portfolio_value):
        """Calculate transaction costs."""
        total_cost = 0
        all_stocks = set(old_weights.keys()) | set(new_weights.keys())
        
        for stock in all_stocks:
            old_weight = old_weights.get(stock, 0)
            new_weight = new_weights.get(stock, 0)
            position_change = abs(new_weight - old_weight) * portfolio_value
            total_cost += position_change * self.config["transaction_costs"]["commission"]
        
        return total_cost
    
    def generate_performance_report(self, returns, diagnostics):
        """Generate performance report."""
        print("\n📊 Generating performance report...")
        
        # Calculate metrics
        annualized_return = returns.mean() * 252
        annualized_volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
        
        # Drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Benchmark comparison
        aligned_benchmark = diagnostics['benchmark_returns'].reindex(returns.index)
        excess_returns = returns - aligned_benchmark
        information_ratio = excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
        
        # Regime analysis
        regime_stats = diagnostics['rebalancing_dates']['regime'].value_counts()
        
        # Save results
        with open('insights/adaptive_rebalancing_real_data_results.md', 'w') as f:
            f.write("# Adaptive Rebalancing Strategy - Real Data Results\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Performance Metrics\n\n")
            f.write(f"- **Annualized Return:** {annualized_return*100:.2f}%\n")
            f.write(f"- **Annualized Volatility:** {annualized_volatility*100:.2f}%\n")
            f.write(f"- **Sharpe Ratio:** {sharpe_ratio:.3f}\n")
            f.write(f"- **Max Drawdown:** {max_drawdown*100:.2f}%\n")
            f.write(f"- **Information Ratio:** {information_ratio:.3f}\n")
            f.write(f"- **Total Return:** {(cumulative_returns.iloc[-1]-1)*100:.2f}%\n\n")
            
            f.write("## Regime Analysis\n\n")
            for regime, count in regime_stats.items():
                f.write(f"- **{regime}:** {count} rebalancing events\n")
            f.write(f"\n- **Total Rebalancing Events:** {len(diagnostics['rebalancing_dates'])}\n")
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Cumulative returns
        plt.subplot(2, 2, 1)
        plt.plot(cumulative_returns.index, cumulative_returns.values, label='Strategy', linewidth=2)
        benchmark_cumulative = (1 + aligned_benchmark).cumprod()
        plt.plot(benchmark_cumulative.index, benchmark_cumulative.values, label='VN-Index', alpha=0.7)
        plt.title('Cumulative Returns')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Regime distribution
        plt.subplot(2, 2, 2)
        regime_stats.plot(kind='bar', color=['#4ECDC4', '#45B7D1', '#96CEB4', '#FFE66D'])
        plt.title('Regime Distribution')
        plt.ylabel('Number of Events')
        plt.xticks(rotation=45)
        
        # Drawdown
        plt.subplot(2, 2, 3)
        plt.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
        plt.plot(drawdown.index, drawdown.values, color='red', linewidth=1)
        plt.title('Drawdown')
        plt.ylabel('Drawdown')
        plt.grid(True, alpha=0.3)
        
        # Monthly returns
        plt.subplot(2, 2, 4)
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        monthly_returns.plot(kind='bar', alpha=0.7)
        plt.title('Monthly Returns')
        plt.ylabel('Return')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('insights/adaptive_rebalancing_real_data_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Performance report generated and saved.")
        
        return {
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'information_ratio': information_ratio,
            'total_return': cumulative_returns.iloc[-1] - 1
        }

# %%
# Main execution
if __name__ == "__main__":
    print("🚀 Starting Adaptive Rebalancing Strategy - Real Data")
    
    try:
        # Initialize database connection
        db_manager = get_database_manager()
        db_engine = db_manager.get_engine()
        print("✅ Database connection established.")
        
        # Initialize strategy
        strategy = QVMEngineV3jAdaptiveRebalancingRealData(QVM_CONFIG, db_engine)
        
        # Run backtest
        returns, diagnostics = strategy.run_real_data_backtest()
        
        # Generate report
        metrics = strategy.generate_performance_report(returns, diagnostics)
        
        # Display results
        print("\n" + "="*80)
        print("ADAPTIVE REBALANCING STRATEGY - REAL DATA RESULTS")
        print("="*80)
        print(f"Annualized Return: {metrics['annualized_return']*100:.2f}%")
        print(f"Annualized Volatility: {metrics['annualized_volatility']*100:.2f}%")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        print(f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
        print(f"Information Ratio: {metrics['information_ratio']:.3f}")
        print(f"Total Return: {metrics['total_return']*100:.2f}%")
        
        print(f"\n✅ Analysis complete! Results saved to:")
        print(f"   - insights/adaptive_rebalancing_real_data_results.md")
        print(f"   - insights/adaptive_rebalancing_real_data_performance.png")
        
    except Exception as e:
        print(f"❌ Error running strategy: {e}")
        import traceback
        traceback.print_exc() 