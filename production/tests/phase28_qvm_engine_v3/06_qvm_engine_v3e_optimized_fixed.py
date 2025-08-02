# Execute the optimized data loading and backtest

import pandas as pd
import numpy as np
import yaml
import warnings
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List
import time

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Load configuration
def load_config():
    """Load QVM configuration from YAML file"""
    with open('config/qvm_engine_v3e_config.yml', 'r') as file:
        return yaml.safe_load(file)

# Database connection
def create_db_connection():
    """Create database connection"""
    try:
        engine = create_engine('postgresql://postgres:postgres@localhost:5432/factor_investing')
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("✅ Database connection established successfully")
        return engine
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return None

class OptimizedDataPreloader:
    """Optimized data preloader for QVM Engine v3e"""
    
    def __init__(self, config: dict, db_engine):
        self.config = config
        self.db_engine = db_engine
        self.start_date = pd.to_datetime(config['data']['start_date'])
        self.end_date = pd.to_datetime(config['data']['end_date'])
        
    def load_all_data(self):
        """Load all required data in optimized manner"""
        print("🔄 Loading all data...")
        
        # Load price data
        price_data = self._load_price_data()
        
        # Load fundamental data
        fundamental_data = self._load_fundamental_data()
        
        # Load benchmark data
        benchmark_data = self._load_benchmark_data()
        
        # Pre-calculate momentum data
        momentum_data = self._pre_calculate_momentum(price_data)
        
        # Pre-calculate universe data
        universe_data = self._pre_calculate_universe(price_data)
        
        # Pre-calculate regime data
        regime_data = self._pre_calculate_regime(benchmark_data)
        
        return {
            'price_data': price_data,
            'fundamental_data': fundamental_data,
            'benchmark_data': benchmark_data,
            'momentum_data': momentum_data,
            'universe_data': universe_data,
            'regime_data': regime_data
        }
    
    def _load_price_data(self):
        """Load price data efficiently"""
        print("  📈 Loading price data...")
        query = """
        SELECT ticker, date, close_price, volume
        FROM price_data 
        WHERE date BETWEEN :start_date AND :end_date
        ORDER BY ticker, date
        """
        
        with self.db_engine.connect() as conn:
            df = pd.read_sql(
                query, 
                conn, 
                params={'start_date': self.start_date, 'end_date': self.end_date},
                parse_dates=['date']
            )
        
        # Pivot for efficient access
        price_pivot = df.pivot(index='date', columns='ticker', values='close_price')
        volume_pivot = df.pivot(index='date', columns='ticker', values='volume')
        
        return {
            'prices': price_pivot,
            'volumes': volume_pivot,
            'raw_data': df
        }
    
    def _load_fundamental_data(self):
        """Load fundamental data efficiently"""
        print("  📊 Loading fundamental data...")
        query = """
        SELECT ticker, date, pe_ratio, pb_ratio, roe, debt_to_equity, 
               current_ratio, quick_ratio, gross_margin, net_margin
        FROM fundamental_data 
        WHERE date BETWEEN :start_date AND :end_date
        ORDER BY ticker, date
        """
        
        with self.db_engine.connect() as conn:
            df = pd.read_sql(
                query, 
                conn, 
                params={'start_date': self.start_date, 'end_date': self.end_date},
                parse_dates=['date']
            )
        
        return df
    
    def _load_benchmark_data(self):
        """Load benchmark data"""
        print("  🎯 Loading benchmark data...")
        query = """
        SELECT date, close_price
        FROM benchmark_data 
        WHERE ticker = 'VNINDEX' 
        AND date BETWEEN :start_date AND :end_date
        ORDER BY date
        """
        
        with self.db_engine.connect() as conn:
            df = pd.read_sql(
                query, 
                conn, 
                params={'start_date': self.start_date, 'end_date': self.end_date},
                parse_dates=['date']
            )
        
        return df.set_index('date')['close_price']
    
    def _pre_calculate_momentum(self, price_data):
        """Pre-calculate momentum factors"""
        print("  ⚡ Pre-calculating momentum factors...")
        prices = price_data['prices']
        
        momentum_data = {}
        
        # Calculate different momentum periods
        for period in [20, 60, 120]:
            momentum = prices.pct_change(period).shift(1)
            momentum_data[f'momentum_{period}d'] = momentum
        
        return momentum_data
    
    def _pre_calculate_universe(self, price_data):
        """Pre-calculate universe filters"""
        print("  🌍 Pre-calculating universe filters...")
        prices = price_data['prices']
        volumes = price_data['volumes']
        
        # Calculate rolling averages for liquidity filter
        avg_volume = volumes.rolling(window=20).mean()
        avg_price = prices.rolling(window=20).mean()
        
        # Liquidity filter: volume > 100k and price > 10k
        liquidity_mask = (avg_volume > 100000) & (avg_price > 10000)
        
        return liquidity_mask
    
    def _pre_calculate_regime(self, benchmark_returns):
        """Pre-calculate regime detection"""
        print("  🔄 Pre-calculating regime detection...")
        
        # Calculate rolling volatility
        rolling_vol = benchmark_returns.rolling(window=60).std() * np.sqrt(252)
        
        # Regime classification
        regime_allocation = pd.Series(index=benchmark_returns.index, dtype=float)
        
        # High volatility regime (defensive)
        high_vol_mask = rolling_vol > rolling_vol.quantile(0.75)
        regime_allocation[high_vol_mask] = 0.3
        
        # Low volatility regime (aggressive)
        low_vol_mask = rolling_vol < rolling_vol.quantile(0.25)
        regime_allocation[low_vol_mask] = 0.8
        
        # Normal regime
        normal_mask = ~(high_vol_mask | low_vol_mask)
        regime_allocation[normal_mask] = 0.6
        
        return {
            'regime_allocation_series': regime_allocation,
            'volatility_series': rolling_vol
        }

class OptimizedFactorCalculator:
    """Optimized factor calculator for QVM Engine v3e"""
    
    def __init__(self, config: dict):
        self.config = config
    
    def calculate_factors_for_date(self, date: pd.Timestamp, 
                                 fundamental_data: pd.DataFrame,
                                 momentum_data: dict,
                                 universe_mask: pd.DataFrame) -> pd.DataFrame:
        """Calculate all factors for a specific date (optimized) - FIXED VERSION"""
        
        # Get fundamental data for the date
        date_fundamentals = fundamental_data[fundamental_data['date'] == date].copy()
        
        if date_fundamentals.empty:
            return pd.DataFrame()
        
        # Start with fundamental factors
        factors_df = date_fundamentals[['ticker', 'pe_ratio', 'pb_ratio', 'roe', 
                                      'debt_to_equity', 'current_ratio', 'quick_ratio', 
                                      'gross_margin', 'net_margin']].copy()
        
        # Add momentum factors (FIXED: avoid duplicate column conflicts)
        for key, momentum_series in momentum_data.items():
            # Get momentum data for the specific date
            momentum_subset = momentum_series.loc[date]
            
            # Reset index and select only ticker and momentum value columns
            momentum_df = momentum_subset.reset_index()
            # Select only ticker and the momentum value column, drop any date column
            momentum_df = momentum_df[['ticker', key]]
            
            # Merge on ticker only
            factors_df = factors_df.merge(momentum_df, on='ticker', how='left')
        
        # Calculate quality-adjusted P/E (vectorized)
        factors_df = self._calculate_quality_adjusted_pe(factors_df)
        
        # Calculate composite score (vectorized)
        factors_df = self._calculate_composite_score_vectorized(factors_df)
        
        # Apply universe filter
        if date in universe_mask.index:
            universe_subset = universe_mask.loc[date]
            universe_tickers = universe_subset[universe_subset].index.tolist()
            factors_df = factors_df[factors_df['ticker'].isin(universe_tickers)]
        
        return factors_df
    
    def _calculate_quality_adjusted_pe(self, factors_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate quality-adjusted P/E ratio (vectorized)"""
        
        def safe_qcut(x):
            try:
                return pd.qcut(x, q=5, labels=False, duplicates='drop')
            except:
                return pd.Series([2] * len(x), index=x.index)  # Neutral rank
        
        # Calculate quality score (vectorized)
        quality_factors = ['roe', 'current_ratio', 'quick_ratio', 'gross_margin', 'net_margin']
        
        # Normalize quality factors
        for factor in quality_factors:
            if factor in factors_df.columns:
                factors_df[f'{factor}_rank'] = factors_df.groupby(factors_df.index)[factor].transform(safe_qcut)
        
        # Calculate composite quality score
        quality_columns = [col for col in factors_df.columns if col.endswith('_rank')]
        if quality_columns:
            factors_df['quality_score'] = factors_df[quality_columns].mean(axis=1)
        else:
            factors_df['quality_score'] = 2  # Neutral score
        
        # Calculate quality-adjusted P/E
        factors_df['quality_adjusted_pe'] = factors_df['pe_ratio'] / (factors_df['quality_score'] + 1)
        
        return factors_df
    
    def _calculate_composite_score_vectorized(self, factors_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate composite QVM score (vectorized)"""
        
        def safe_qcut(x):
            try:
                return pd.qcut(x, q=5, labels=False, duplicates='drop')
            except:
                return pd.Series([2] * len(x), index=x.index)  # Neutral rank
        
        # Value factors (lower is better)
        value_factors = ['quality_adjusted_pe', 'pb_ratio']
        for factor in value_factors:
            if factor in factors_df.columns:
                factors_df[f'{factor}_rank'] = factors_df.groupby(factors_df.index)[factor].transform(safe_qcut)
        
        # Quality factors (higher is better)
        quality_factors = ['roe', 'current_ratio', 'gross_margin']
        for factor in quality_factors:
            if factor in factors_df.columns:
                factors_df[f'{factor}_rank'] = factors_df.groupby(factors_df.index)[factor].transform(safe_qcut)
        
        # Momentum factors (higher is better)
        momentum_factors = [col for col in factors_df.columns if col.startswith('momentum_')]
        for factor in momentum_factors:
            factors_df[f'{factor}_rank'] = factors_df.groupby(factors_df.index)[factor].transform(safe_qcut)
        
        # Calculate composite scores
        rank_columns = [col for col in factors_df.columns if col.endswith('_rank')]
        
        if rank_columns:
            factors_df['value_score'] = factors_df[[col for col in rank_columns if any(vf in col for vf in value_factors)]].mean(axis=1)
            factors_df['quality_score'] = factors_df[[col for col in rank_columns if any(qf in col for qf in quality_factors)]].mean(axis=1)
            factors_df['momentum_score'] = factors_df[[col for col in rank_columns if 'momentum' in col]].mean(axis=1)
            
            # Composite QVM score
            factors_df['qvm_score'] = (
                factors_df['value_score'] * 0.4 +
                factors_df['quality_score'] * 0.3 +
                factors_df['momentum_score'] * 0.3
            )
        else:
            factors_df['qvm_score'] = 2  # Neutral score
        
        return factors_df

class OptimizedQVMEngineV3e:
    """Optimized QVM Engine v3e for enhanced performance"""
    
    def __init__(self, config: dict, preloaded_data: dict, db_engine):
        self.config = config
        self.preloaded_data = preloaded_data
        self.db_engine = db_engine
        
        # Extract preloaded data
        self.price_data = preloaded_data['price_data']
        self.fundamental_data = preloaded_data['fundamental_data']
        self.benchmark_data = preloaded_data['benchmark_data']
        self.momentum_data = preloaded_data['momentum_data']
        self.universe_data = preloaded_data['universe_data']
        self.regime_data = preloaded_data['regime_data']
        
        # Initialize factor calculator
        self.factor_calculator = OptimizedFactorCalculator(config)
        
        # Portfolio parameters
        self.max_positions = config['portfolio']['max_positions']
        self.position_size = config['portfolio']['position_size']
    
    def run_backtest(self) -> (pd.Series, pd.DataFrame):
        """Run the optimized backtest"""
        print("\n🚀 Starting optimized QVM Engine v3e backtest execution...")
        
        rebalance_dates = self._generate_rebalance_dates()
        daily_holdings, diagnostics = self._run_optimized_backtesting_loop(rebalance_dates)
        net_returns = self._calculate_net_returns(daily_holdings)
        
        print("✅ Optimized QVM Engine v3e backtest execution complete.")
        return net_returns, diagnostics
    
    def _generate_rebalance_dates(self) -> list:
        """Generate rebalancing dates"""
        start_date = self.config['data']['start_date']
        end_date = self.config['data']['end_date']
        
        # Monthly rebalancing
        rebalance_dates = pd.date_range(
            start=start_date, 
            end=end_date, 
            freq='M'
        )
        
        return rebalance_dates.tolist()
    
    def _run_optimized_backtesting_loop(self, rebalance_dates: list) -> (pd.DataFrame, pd.DataFrame):
        """Run the optimized backtesting loop"""
        print(f"  📅 Processing {len(rebalance_dates)} rebalancing dates...")
        
        # Initialize tracking DataFrames
        all_holdings = []
        all_diagnostics = []
        
        current_portfolio = pd.Series(dtype=float)
        
        for i, rebal_date in enumerate(rebalance_dates):
            if i % 10 == 0:
                print(f"    Processing date {i+1}/{len(rebalance_dates)}: {rebal_date.strftime('%Y-%m-%d')}")
            
            # Get regime allocation
            regime_allocation = self.regime_data['regime_allocation_series'].loc[rebal_date]
            
            # Calculate factors (optimized)
            factors_df = self.factor_calculator.calculate_factors_for_date(
                rebal_date, self.fundamental_data, self.momentum_data, self.universe_data
            )
            
            if factors_df.empty:
                print(" ⚠️ No factor data. Skipping.")
                continue
            
            # Apply entry criteria
            qualified_df = self._apply_entry_criteria(factors_df)
            
            if qualified_df.empty:
                print(" ⚠️ No qualified stocks. Skipping.")
                continue
            
            # Construct portfolio
            new_portfolio = self._construct_portfolio(qualified_df, regime_allocation)
            
            # Track holdings for the period
            period_holdings = self._track_period_holdings(
                current_portfolio, new_portfolio, rebal_date, rebalance_dates, i
            )
            all_holdings.append(period_holdings)
            
            # Track diagnostics
            diagnostics = {
                'date': rebal_date,
                'regime_allocation': regime_allocation,
                'num_qualified': len(qualified_df),
                'portfolio_size': len(new_portfolio),
                'avg_qvm_score': qualified_df['qvm_score'].mean(),
                'avg_pe': qualified_df['pe_ratio'].mean(),
                'avg_roe': qualified_df['roe'].mean()
            }
            all_diagnostics.append(diagnostics)
            
            current_portfolio = new_portfolio
        
        # Combine all holdings
        daily_holdings = pd.concat(all_holdings, ignore_index=True)
        diagnostics_df = pd.DataFrame(all_diagnostics)
        
        return daily_holdings, diagnostics_df
    
    def _apply_entry_criteria(self, factors_df: pd.DataFrame) -> pd.DataFrame:
        """Apply entry criteria to filter stocks"""
        # Basic filters
        qualified = factors_df[
            (factors_df['pe_ratio'] > 0) &  # Positive P/E
            (factors_df['pe_ratio'] < 50) &  # Reasonable P/E
            (factors_df['pb_ratio'] > 0) &   # Positive P/B
            (factors_df['pb_ratio'] < 10) &  # Reasonable P/B
            (factors_df['roe'] > 0)          # Positive ROE
        ].copy()
        
        # Sort by QVM score (higher is better)
        qualified = qualified.sort_values('qvm_score', ascending=False)
        
        return qualified
    
    def _construct_portfolio(self, qualified_df: pd.DataFrame, regime_allocation: float) -> pd.Series:
        """Construct portfolio based on regime allocation"""
        # Select top stocks
        top_stocks = qualified_df.head(self.max_positions)
        
        # Calculate position sizes
        num_positions = len(top_stocks)
        if num_positions == 0:
            return pd.Series(dtype=float)
        
        # Equal weight within regime allocation
        position_weight = regime_allocation / num_positions
        
        portfolio = pd.Series(position_weight, index=top_stocks['ticker'])
        
        return portfolio
    
    def _track_period_holdings(self, current_portfolio: pd.Series, new_portfolio: pd.Series, 
                             rebal_date: pd.Timestamp, rebalance_dates: list, current_idx: int) -> pd.DataFrame:
        """Track holdings for the period between rebalancing dates"""
        # Determine period end date
        if current_idx < len(rebalance_dates) - 1:
            period_end = rebalance_dates[current_idx + 1]
        else:
            period_end = self.config['data']['end_date']
        
        # Generate daily dates for the period
        period_dates = pd.date_range(start=rebal_date, end=period_end, freq='D')
        
        # Create holdings DataFrame for the period
        holdings_list = []
        for date in period_dates:
            holdings = {
                'date': date,
                'portfolio': new_portfolio.copy()
            }
            holdings_list.append(holdings)
        
        return pd.DataFrame(holdings_list)
    
    def _calculate_net_returns(self, daily_holdings: pd.DataFrame) -> pd.Series:
        """Calculate net returns from daily holdings"""
        print("  📊 Calculating net returns...")
        
        # Get price data
        prices = self.price_data['prices']
        
        # Calculate daily returns
        returns_list = []
        
        for _, row in daily_holdings.iterrows():
            date = row['date']
            portfolio = row['portfolio']
            
            if portfolio.empty:
                returns_list.append(0.0)
                continue
            
            # Get current prices for portfolio stocks
            if date in prices.index:
                current_prices = prices.loc[date, portfolio.index]
                
                # Calculate portfolio return
                if len(current_prices) > 0:
                    portfolio_return = (portfolio * current_prices.pct_change()).sum()
                    returns_list.append(portfolio_return)
                else:
                    returns_list.append(0.0)
            else:
                returns_list.append(0.0)
        
        # Create returns series
        returns_series = pd.Series(returns_list, index=daily_holdings['date'])
        
        # Calculate cumulative returns
        net_returns = (1 + returns_series).cumprod() - 1
        
        return net_returns

def calculate_performance_metrics(returns: pd.Series, benchmark: pd.Series, periods_per_year: int = 252) -> dict:
    """Calculate comprehensive performance metrics"""
    
    # Basic metrics
    total_return = returns.iloc[-1]
    annualized_return = (1 + total_return) ** (periods_per_year / len(returns)) - 1
    
    # Volatility
    daily_returns = returns.pct_change().dropna()
    volatility = daily_returns.std() * np.sqrt(periods_per_year)
    
    # Sharpe ratio
    risk_free_rate = 0.02  # 2% annual risk-free rate
    sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
    
    # Maximum drawdown
    cumulative = (1 + returns.pct_change()).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Beta and alpha
    benchmark_returns = benchmark.pct_change().dropna()
    aligned_returns = daily_returns.reindex(benchmark_returns.index).dropna()
    aligned_benchmark = benchmark_returns.reindex(aligned_returns.index).dropna()
    
    if len(aligned_returns) > 0 and len(aligned_benchmark) > 0:
        # Ensure same length
        min_len = min(len(aligned_returns), len(aligned_benchmark))
        aligned_returns = aligned_returns.iloc[:min_len]
        aligned_benchmark = aligned_benchmark.iloc[:min_len]
        
        beta = np.cov(aligned_returns, aligned_benchmark)[0, 1] / np.var(aligned_benchmark)
        alpha = aligned_returns.mean() - beta * aligned_benchmark.mean()
        alpha = alpha * periods_per_year  # Annualize alpha
    else:
        beta = 1.0
        alpha = 0.0
    
    # Information ratio
    excess_returns = aligned_returns - aligned_benchmark
    information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(periods_per_year) if excess_returns.std() > 0 else 0
    
    return {
        'Total Return': f"{total_return:.2%}",
        'Annualized Return': f"{annualized_return:.2%}",
        'Volatility': f"{volatility:.2%}",
        'Sharpe Ratio': f"{sharpe_ratio:.2f}",
        'Max Drawdown': f"{max_drawdown:.2%}",
        'Beta': f"{beta:.2f}",
        'Alpha': f"{alpha:.2%}",
        'Information Ratio': f"{information_ratio:.2f}"
    }

def generate_comprehensive_tearsheet(strategy_returns: pd.Series, benchmark_returns: pd.Series, diagnostics: pd.DataFrame, title: str):
    """Generate comprehensive performance tearsheet"""
    
    # Calculate performance metrics
    metrics = calculate_performance_metrics(strategy_returns, benchmark_returns)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{title} - Performance Analysis', fontsize=16, fontweight='bold')
    
    # 1. Equity curve
    axes[0, 0].plot(strategy_returns.index, (1 + strategy_returns) * 100, label='Strategy', linewidth=2)
    axes[0, 0].plot(benchmark_returns.index, (1 + benchmark_returns) * 100, label='Benchmark', linewidth=2, alpha=0.7)
    axes[0, 0].set_title('Cumulative Returns')
    axes[0, 0].set_ylabel('Cumulative Return (%)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Drawdown
    cumulative = (1 + strategy_returns.pct_change()).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max * 100
    axes[0, 1].fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
    axes[0, 1].plot(drawdown.index, drawdown, color='red', linewidth=1)
    axes[0, 1].set_title('Drawdown')
    axes[0, 1].set_ylabel('Drawdown (%)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Rolling Sharpe ratio
    rolling_sharpe = strategy_returns.pct_change().rolling(window=252).mean() / strategy_returns.pct_change().rolling(window=252).std() * np.sqrt(252)
    axes[1, 0].plot(rolling_sharpe.index, rolling_sharpe, linewidth=2)
    axes[1, 0].set_title('Rolling Sharpe Ratio (1-year)')
    axes[1, 0].set_ylabel('Sharpe Ratio')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Regime allocation over time
    if 'regime_allocation' in diagnostics.columns:
        axes[1, 1].plot(diagnostics['date'], diagnostics['regime_allocation'], linewidth=2)
        axes[1, 1].set_title('Regime Allocation Over Time')
        axes[1, 1].set_ylabel('Allocation')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print metrics
    print(f"\n📊 {title} - Performance Metrics:")
    print("=" * 50)
    for metric, value in metrics.items():
        print(f"{metric:20}: {value}")
    
    # Print diagnostics summary
    if not diagnostics.empty:
        print(f"\n📈 Strategy Diagnostics:")
        print("=" * 30)
        print(f"Average Portfolio Size: {diagnostics['portfolio_size'].mean():.1f}")
        print(f"Average Qualified Stocks: {diagnostics['num_qualified'].mean():.1f}")
        print(f"Average QVM Score: {diagnostics['avg_qvm_score'].mean():.2f}")
        print(f"Average P/E Ratio: {diagnostics['avg_pe'].mean():.2f}")
        print(f"Average ROE: {diagnostics['avg_roe'].mean():.2%}")

# Main execution
if __name__ == "__main__":
    print("🚀 Starting Optimized QVM Engine v3e Execution")
    print("=" * 60)
    
    # Load configuration
    QVM_CONFIG = load_config()
    
    # Create database connection
    engine = create_db_connection()
    if engine is None:
        print("❌ Cannot proceed without database connection")
        exit(1)
    
    # Step 1: Preload all data
    print("\n📊 Step 1: Preloading all data...")
    preloader = OptimizedDataPreloader(QVM_CONFIG, engine)
    preloaded_data = preloader.load_all_data()
    
    # Step 2: Run optimized backtest
    print("\n📊 Step 2: Running optimized backtest...")
    qvm_engine = OptimizedQVMEngineV3e(
        config=QVM_CONFIG,
        preloaded_data=preloaded_data,
        db_engine=engine
    )
    
    qvm_net_returns, qvm_diagnostics = qvm_engine.run_backtest()
    
    # Step 3: Generate comprehensive performance report
    print("\n" + "="*80)
    print("📊 GENERATING COMPREHENSIVE PERFORMANCE REPORT")
    print("="*80)
    
    # Get benchmark returns for comparison
    benchmark_returns = preloaded_data['benchmark_data'].pct_change().cumsum()
    
    # Generate tearsheet
    generate_comprehensive_tearsheet(
        qvm_net_returns, 
        benchmark_returns, 
        qvm_diagnostics, 
        "Optimized QVM Engine v3e"
    )
    
    print("\n✅ Optimized QVM Engine v3e execution completed successfully!") 