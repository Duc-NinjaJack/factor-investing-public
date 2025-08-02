import pandas as pd
import numpy as np
import yaml
import warnings
from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
import jupytext

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def create_db_connection():
    """
    Create database connection for the QVM engine
    """
    try:
        from production.database.connection import get_engine
        engine = get_engine()
        return engine
    except ImportError:
        print("Database connection module not found. Using mock data.")
        return None

class RegimeDetector:
    """
    Percentile-based regime detector that adapts to market conditions
    """
    
    def __init__(self, lookback_period: int = 90, 
                 volatility_percentile_high: float = 75.0,
                 return_percentile_high: float = 75.0,
                 return_percentile_low: float = 25.0):
        """
        Initialize regime detector with percentile-based thresholds
        
        Args:
            lookback_period: Days to look back for percentile calculation
            volatility_percentile_high: Percentile for high volatility threshold
            return_percentile_high: Percentile for high return threshold  
            return_percentile_low: Percentile for low return threshold
        """
        self.lookback_period = lookback_period
        self.volatility_percentile_high = volatility_percentile_high
        self.return_percentile_high = return_percentile_high
        self.return_percentile_low = return_percentile_low
        
        # Store historical data for percentile calculation
        self.volatility_history = []
        self.return_history = []
        
    def detect_regime(self, price_data: pd.DataFrame) -> str:
        """
        Detect market regime using percentile-based thresholds
        
        Args:
            price_data: DataFrame with 'close' column
            
        Returns:
            Regime classification: 'momentum', 'stress', or 'normal'
        """
        if len(price_data) < self.lookback_period:
            return 'normal'  # Default regime for insufficient data
            
        # Calculate rolling volatility and returns
        returns = price_data['close'].pct_change().dropna()
        volatility = returns.rolling(window=20).std().dropna()
        
        # Update historical data
        if len(volatility) > 0:
            self.volatility_history.append(volatility.iloc[-1])
        if len(returns) > 0:
            self.return_history.append(returns.iloc[-1])
            
        # Keep only recent history for percentile calculation
        if len(self.volatility_history) > self.lookback_period:
            self.volatility_history = self.volatility_history[-self.lookback_period:]
        if len(self.return_history) > self.lookback_period:
            self.return_history = self.return_history[-self.lookback_period:]
            
        # Calculate dynamic thresholds using percentiles
        if len(self.volatility_history) >= 10:  # Minimum data requirement
            vol_threshold = np.percentile(self.volatility_history, self.volatility_percentile_high)
            return_threshold_high = np.percentile(self.return_history, self.return_percentile_high)
            return_threshold_low = np.percentile(self.return_history, self.return_percentile_low)
        else:
            # Fallback to reasonable defaults if insufficient data
            vol_threshold = 0.02  # 2% daily volatility
            return_threshold_high = 0.01  # 1% daily return
            return_threshold_low = -0.01  # -1% daily return
            
        # Get current values
        current_vol = volatility.iloc[-1] if len(volatility) > 0 else 0
        current_return = returns.iloc[-1] if len(returns) > 0 else 0
        
        # Regime classification logic
        if current_vol > vol_threshold:
            if current_return > return_threshold_high:
                return 'momentum'
            elif current_return < return_threshold_low:
                return 'stress'
            else:
                return 'normal'
        else:
            return 'normal'
    
    def get_regime_allocation(self, regime: str) -> float:
        """
        Get portfolio allocation based on detected regime
        
        Args:
            regime: Detected regime ('momentum', 'stress', 'normal')
            
        Returns:
            Portfolio allocation percentage
        """
        allocation_map = {
            'momentum': 0.8,  # High allocation in momentum regime
            'stress': 0.3,    # Low allocation in stress regime
            'normal': 0.6     # Moderate allocation in normal regime
        }
        return allocation_map.get(regime, 0.6)

class SectorAwareFactorCalculator:
    """
    Enhanced factor calculator with sector-aware calculations
    """
    
    def __init__(self, engine):
        self.engine = engine
        
    def calculate_sector_aware_pe(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate sector-aware PE ratios
        """
        def safe_qcut(x):
            try:
                if len(x.dropna()) >= 4:
                    return pd.qcut(x, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
                else:
                    return pd.Series(['Q2'] * len(x), index=x.index)
            except:
                return pd.Series(['Q2'] * len(x), index=x.index)
        
        # Calculate PE ratios
        data['pe_ratio'] = data['market_cap'] / data['net_income']
        
        # Sector-aware PE scoring
        data['pe_score'] = data.groupby('sector')['pe_ratio'].transform(safe_qcut)
        
        # Convert quartiles to numerical scores
        pe_score_map = {'Q1': 4, 'Q2': 3, 'Q3': 2, 'Q4': 1}
        data['pe_score'] = data['pe_score'].map(pe_score_map).fillna(2)
        
        return data
    
    def calculate_momentum_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate momentum scores with sector adjustment
        """
        # Calculate returns for different periods
        for period in [20, 60, 120]:
            data[f'return_{period}d'] = data['close'].pct_change(period)
        
        # Calculate momentum score as weighted average
        data['momentum_score'] = (
            data['return_20d'] * 0.5 + 
            data['return_60d'] * 0.3 + 
            data['return_120d'] * 0.2
        )
        
        # Sector-aware momentum scoring
        def safe_qcut(x):
            try:
                if len(x.dropna()) >= 4:
                    return pd.qcut(x, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
                else:
                    return pd.Series(['Q2'] * len(x), index=x.index)
            except:
                return pd.Series(['Q2'] * len(x), index=x.index)
        
        data['momentum_quartile'] = data.groupby('sector')['momentum_score'].transform(safe_qcut)
        
        # Convert quartiles to numerical scores
        momentum_score_map = {'Q1': 1, 'Q2': 2, 'Q3': 3, 'Q4': 4}
        data['momentum_score'] = data['momentum_quartile'].map(momentum_score_map).fillna(2)
        
        return data

class QVMEngineV3AdoptedInsights:
    """
    QVM Engine v3e with percentile-based regime detection and adopted insights
    """
    
    def __init__(self, config: dict, price_data: pd.DataFrame, fundamental_data: pd.DataFrame,
                 returns_matrix: pd.DataFrame, benchmark_returns: pd.Series, db_engine):
        """
        Initialize QVM Engine v3e
        
        Args:
            config: Configuration dictionary
            price_data: Historical price data
            fundamental_data: Fundamental data
            returns_matrix: Returns matrix
            benchmark_returns: Benchmark returns series
            db_engine: Database engine
        """
        self.config = config
        self.price_data = price_data
        self.fundamental_data = fundamental_data
        self.returns_matrix = returns_matrix
        self.benchmark_returns = benchmark_returns
        self.db_engine = db_engine
        
        # Initialize regime detector with percentile-based approach
        regime_config = config.get('regime', {})
        self.regime_detector = RegimeDetector(
            lookback_period=regime_config.get('lookback_period', 90),
            volatility_percentile_high=regime_config.get('volatility_percentile_high', 75.0),
            return_percentile_high=regime_config.get('return_percentile_high', 75.0),
            return_percentile_low=regime_config.get('return_percentile_low', 25.0)
        )
        
        # Initialize factor calculator
        self.factor_calculator = SectorAwareFactorCalculator(self)
        
        # Performance tracking
        self.portfolio_returns = []
        self.regime_history = []
        self.allocation_history = []
        
    def run_backtest(self) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Run the backtest with percentile-based regime detection
        
        Returns:
            Tuple of (strategy_returns, diagnostics_dataframe)
        """
        print("Starting QVM Engine v3e backtest with percentile-based regime detection...")
        
        # Generate rebalance dates
        rebalance_dates = self._generate_rebalance_dates()
        
        # Run backtesting loop
        holdings_df, diagnostics_df = self._run_backtesting_loop(rebalance_dates)
        
        # Calculate net returns
        strategy_returns = self._calculate_net_returns(holdings_df)
        
        print(f"Backtest completed. Regime distribution: {pd.Series(self.regime_history).value_counts().to_dict()}")
        
        return strategy_returns, diagnostics_df
    
    def _generate_rebalance_dates(self) -> List[pd.Timestamp]:
        """Generate rebalance dates"""
        start_date = self.price_data.index.min()
        end_date = self.price_data.index.max()
        
        rebalance_dates = []
        current_date = start_date
        
        while current_date <= end_date:
            if current_date in self.price_data.index:
                rebalance_dates.append(current_date)
            current_date += pd.Timedelta(days=1)
            
        return rebalance_dates
    
    def _run_backtesting_loop(self, rebalance_dates: List[pd.Timestamp]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run the main backtesting loop"""
        holdings_data = []
        diagnostics_data = []
        
        for i, analysis_date in enumerate(rebalance_dates):
            if i % 100 == 0:
                print(f"Processing date {analysis_date.strftime('%Y-%m-%d')} ({i+1}/{len(rebalance_dates)})")
            
            try:
                # Get universe
                universe = self._get_universe(analysis_date)
                if not universe:
                    continue
                
                # Detect current regime
                regime = self._detect_current_regime(analysis_date)
                self.regime_history.append(regime)
                
                # Get regime allocation
                regime_allocation = self.regime_detector.get_regime_allocation(regime)
                self.allocation_history.append(regime_allocation)
                
                # Calculate factors
                factors_df = self._calculate_factors(universe, analysis_date)
                
                # Apply entry criteria
                qualified_df = self._apply_entry_criteria(factors_df)
                
                # Construct portfolio
                portfolio_weights = self._construct_portfolio(qualified_df, regime_allocation)
                
                # Store holdings
                for ticker, weight in portfolio_weights.items():
                    holdings_data.append({
                        'date': analysis_date,
                        'ticker': ticker,
                        'weight': weight,
                        'regime': regime,
                        'allocation': regime_allocation
                    })
                
                # Store diagnostics
                diagnostics_data.append({
                    'date': analysis_date,
                    'regime': regime,
                    'allocation': regime_allocation,
                    'universe_size': len(universe),
                    'qualified_size': len(qualified_df),
                    'portfolio_size': len(portfolio_weights)
                })
                
            except Exception as e:
                print(f"Error processing {analysis_date}: {str(e)}")
                continue
        
        holdings_df = pd.DataFrame(holdings_data)
        diagnostics_df = pd.DataFrame(diagnostics_data)
        
        return holdings_df, diagnostics_df
    
    def _get_universe(self, analysis_date: pd.Timestamp) -> List[str]:
        """Get investable universe for the given date"""
        # Get available tickers at the analysis date
        available_tickers = self.price_data.loc[:analysis_date].columns.tolist()
        
        # Filter for minimum data requirements
        min_data_required = 120  # At least 120 days of data
        qualified_tickers = []
        
        for ticker in available_tickers:
            ticker_data = self.price_data.loc[:analysis_date, ticker].dropna()
            if len(ticker_data) >= min_data_required:
                qualified_tickers.append(ticker)
        
        return qualified_tickers
    
    def _detect_current_regime(self, analysis_date: pd.Timestamp) -> str:
        """Detect current market regime using percentile-based approach"""
        # Get historical price data up to analysis date
        historical_data = self.price_data.loc[:analysis_date]
        
        # Use VNINDEX as proxy for market data if available, otherwise use first ticker
        if 'VNINDEX' in historical_data.columns:
            market_data = historical_data[['VNINDEX']].copy()
        else:
            market_data = historical_data.iloc[:, :1].copy()
            market_data.columns = ['close']
        
        # Detect regime
        regime = self.regime_detector.detect_regime(market_data)
        
        return regime
    
    def _calculate_factors(self, universe: List[str], analysis_date: pd.Timestamp) -> pd.DataFrame:
        """Calculate factors for the universe"""
        factors_data = []
        
        for ticker in universe:
            try:
                # Get price data
                ticker_prices = self.price_data.loc[:analysis_date, ticker].dropna()
                if len(ticker_prices) < 120:
                    continue
                
                # Get fundamental data
                ticker_fundamentals = self.fundamental_data[
                    (self.fundamental_data['ticker'] == ticker) & 
                    (self.fundamental_data['date'] <= analysis_date)
                ].iloc[-1] if len(self.fundamental_data[
                    (self.fundamental_data['ticker'] == ticker) & 
                    (self.fundamental_data['date'] <= analysis_date)
                ]) > 0 else None
                
                if ticker_fundamentals is None:
                    continue
                
                # Create data row
                data_row = {
                    'ticker': ticker,
                    'close': ticker_prices.iloc[-1],
                    'sector': ticker_fundamentals.get('sector', 'Unknown'),
                    'market_cap': ticker_fundamentals.get('market_cap', 0),
                    'net_income': ticker_fundamentals.get('net_income', 0)
                }
                
                # Add price data for momentum calculation
                for i, price in enumerate(ticker_prices.tail(120)):
                    data_row[f'price_{i}'] = price
                
                factors_data.append(data_row)
                
            except Exception as e:
                print(f"Error calculating factors for {ticker}: {str(e)}")
                continue
        
        if not factors_data:
            return pd.DataFrame()
        
        factors_df = pd.DataFrame(factors_data)
        
        # Calculate momentum factors
        factors_df = self._calculate_momentum_factors(factors_df, analysis_date)
        
        # Calculate PE factors
        factors_df = self._calculate_pe_factors(factors_df, factors_df)
        
        # Calculate composite score
        factors_df = self._calculate_composite_score(factors_df)
        
        return factors_df
    
    def _calculate_momentum_factors(self, market_df: pd.DataFrame, analysis_date: pd.Timestamp) -> pd.DataFrame:
        """Calculate momentum factors"""
        for ticker in market_df['ticker'].unique():
            ticker_data = market_df[market_df['ticker'] == ticker].iloc[0]
            
            # Extract price series
            prices = []
            for i in range(120):
                price_key = f'price_{i}'
                if price_key in ticker_data:
                    prices.append(ticker_data[price_key])
            
            if len(prices) < 120:
                continue
            
            prices = prices[::-1]  # Reverse to get chronological order
            price_series = pd.Series(prices)
            
            # Calculate returns
            returns_20d = price_series.pct_change(20).iloc[-1]
            returns_60d = price_series.pct_change(60).iloc[-1]
            returns_120d = price_series.pct_change(120).iloc[-1]
            
            # Update momentum scores
            idx = market_df[market_df['ticker'] == ticker].index[0]
            market_df.loc[idx, 'momentum_20d'] = returns_20d
            market_df.loc[idx, 'momentum_60d'] = returns_60d
            market_df.loc[idx, 'momentum_120d'] = returns_120d
        
        return market_df
    
    def _calculate_pe_factors(self, market_df: pd.DataFrame, fundamental_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate PE factors"""
        # Calculate PE ratio
        market_df['pe_ratio'] = market_df['market_cap'] / market_df['net_income'].replace(0, np.nan)
        
        # Handle infinite and NaN values
        market_df['pe_ratio'] = market_df['pe_ratio'].replace([np.inf, -np.inf], np.nan)
        
        # Calculate PE score (lower PE = higher score)
        pe_ratio_valid = market_df['pe_ratio'].dropna()
        if len(pe_ratio_valid) > 0:
            pe_quantiles = pd.qcut(pe_ratio_valid, q=4, labels=[4, 3, 2, 1], duplicates='drop')
            market_df.loc[pe_ratio_valid.index, 'pe_score'] = pe_quantiles
        
        return market_df
    
    def _calculate_composite_score(self, factors_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate composite factor score"""
        # Normalize momentum factors
        for col in ['momentum_20d', 'momentum_60d', 'momentum_120d']:
            if col in factors_df.columns:
                factors_df[col] = (factors_df[col] - factors_df[col].mean()) / factors_df[col].std()
        
        # Calculate composite score
        momentum_score = (
            factors_df.get('momentum_20d', 0) * 0.5 +
            factors_df.get('momentum_60d', 0) * 0.3 +
            factors_df.get('momentum_120d', 0) * 0.2
        )
        
        pe_score = factors_df.get('pe_score', 2)  # Default to neutral score
        
        # Combine scores (momentum + value)
        factors_df['composite_score'] = momentum_score * 0.6 + pe_score * 0.4
        
        return factors_df
    
    def _apply_entry_criteria(self, factors_df: pd.DataFrame) -> pd.DataFrame:
        """Apply entry criteria to filter stocks"""
        if len(factors_df) == 0:
            return factors_df
        
        # Filter by composite score (top 50%)
        score_threshold = factors_df['composite_score'].quantile(0.5)
        qualified_df = factors_df[factors_df['composite_score'] >= score_threshold].copy()
        
        return qualified_df
    
    def _construct_portfolio(self, qualified_df: pd.DataFrame, regime_allocation: float) -> pd.Series:
        """Construct portfolio weights"""
        if len(qualified_df) == 0:
            return pd.Series()
        
        # Equal weight allocation
        n_stocks = len(qualified_df)
        weight_per_stock = regime_allocation / n_stocks
        
        portfolio_weights = pd.Series(weight_per_stock, index=qualified_df['ticker'])
        
        return portfolio_weights
    
    def _calculate_net_returns(self, daily_holdings: pd.DataFrame) -> pd.Series:
        """Calculate net strategy returns"""
        if len(daily_holdings) == 0:
            return pd.Series()
        
        # Group by date and calculate portfolio returns
        portfolio_returns = []
        
        for date in daily_holdings['date'].unique():
            date_holdings = daily_holdings[daily_holdings['date'] == date]
            
            if len(date_holdings) == 0:
                continue
            
            # Calculate weighted return for this date
            date_return = 0
            for _, holding in date_holdings.iterrows():
                ticker = holding['ticker']
                weight = holding['weight']
                
                if ticker in self.returns_matrix.columns and date in self.returns_matrix.index:
                    ticker_return = self.returns_matrix.loc[date, ticker]
                    if pd.notna(ticker_return):
                        date_return += weight * ticker_return
            
            portfolio_returns.append({
                'date': date,
                'return': date_return
            })
        
        returns_df = pd.DataFrame(portfolio_returns)
        returns_df.set_index('date', inplace=True)
        
        return returns_df['return']

def load_all_data_for_backtest(config: dict, db_engine):
    """
    Load all required data for backtesting
    """
    print("Loading data for backtest...")
    
    # Load configuration
    start_date = config.get('start_date', '2016-01-01')
    end_date = config.get('end_date', '2025-01-01')
    
    # Mock data generation for demonstration
    # In production, this would load from database
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate mock price data
    n_stocks = 100
    tickers = [f'STOCK_{i:03d}' for i in range(n_stocks)]
    
    # Create price data with realistic patterns
    np.random.seed(42)
    price_data = pd.DataFrame(index=date_range, columns=tickers)
    
    for ticker in tickers:
        # Generate random walk with trend
        returns = np.random.normal(0.0005, 0.02, len(date_range))
        prices = 100 * np.exp(np.cumsum(returns))
        price_data[ticker] = prices
    
    # Add VNINDEX as market proxy
    market_returns = np.random.normal(0.0003, 0.015, len(date_range))
    market_prices = 1000 * np.exp(np.cumsum(market_returns))
    price_data['VNINDEX'] = market_prices
    
    # Generate fundamental data
    fundamental_data = []
    for ticker in tickers:
        for date in date_range[::90]:  # Quarterly data
            fundamental_data.append({
                'ticker': ticker,
                'date': date,
                'sector': np.random.choice(['Banking', 'Technology', 'Consumer', 'Energy']),
                'market_cap': np.random.uniform(1000, 100000),
                'net_income': np.random.uniform(10, 1000)
            })
    
    fundamental_df = pd.DataFrame(fundamental_data)
    
    # Generate returns matrix
    returns_matrix = price_data.pct_change().dropna()
    
    # Generate benchmark returns (VNINDEX)
    benchmark_returns = returns_matrix['VNINDEX']
    
    print(f"Data loaded: {len(price_data)} days, {len(tickers)} stocks")
    
    return price_data, fundamental_df, returns_matrix, benchmark_returns

def calculate_performance_metrics(returns: pd.Series, benchmark: pd.Series, periods_per_year: int = 252) -> dict:
    """
    Calculate comprehensive performance metrics
    """
    if len(returns) == 0:
        return {}
    
    # Basic metrics
    total_return = (1 + returns).prod() - 1
    annualized_return = (1 + total_return) ** (periods_per_year / len(returns)) - 1
    volatility = returns.std() * np.sqrt(periods_per_year)
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
    
    # Drawdown analysis
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Benchmark comparison
    if len(benchmark) > 0:
        benchmark_total_return = (1 + benchmark).prod() - 1
        benchmark_annualized = (1 + benchmark_total_return) ** (periods_per_year / len(benchmark)) - 1
        benchmark_volatility = benchmark.std() * np.sqrt(periods_per_year)
        
        excess_return = annualized_return - benchmark_annualized
        information_ratio = excess_return / (returns - benchmark).std() * np.sqrt(periods_per_year) if (returns - benchmark).std() > 0 else 0
    else:
        benchmark_annualized = 0
        benchmark_volatility = 0
        excess_return = annualized_return
        information_ratio = sharpe_ratio
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'benchmark_return': benchmark_annualized,
        'benchmark_volatility': benchmark_volatility,
        'excess_return': excess_return,
        'information_ratio': information_ratio
    }

def generate_comprehensive_tearsheet(strategy_returns: pd.Series, benchmark_returns: pd.Series, 
                                   diagnostics: pd.DataFrame, title: str):
    """
    Generate comprehensive performance tearsheet
    """
    if len(strategy_returns) == 0:
        print("No strategy returns to analyze")
        return
    
    # Calculate performance metrics
    metrics = calculate_performance_metrics(strategy_returns, benchmark_returns)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{title} - Performance Analysis', fontsize=16, fontweight='bold')
    
    # 1. Cumulative returns
    cumulative_strategy = (1 + strategy_returns).cumprod()
    cumulative_benchmark = (1 + benchmark_returns).cumprod()
    
    axes[0, 0].plot(cumulative_strategy.index, cumulative_strategy.values, 
                   label='Strategy', linewidth=2, color='blue')
    axes[0, 0].plot(cumulative_benchmark.index, cumulative_benchmark.values, 
                   label='Benchmark', linewidth=2, color='red', alpha=0.7)
    axes[0, 0].set_title('Cumulative Returns')
    axes[0, 0].set_ylabel('Cumulative Return')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Regime distribution
    if len(diagnostics) > 0 and 'regime' in diagnostics.columns:
        regime_counts = diagnostics['regime'].value_counts()
        axes[0, 1].pie(regime_counts.values, labels=regime_counts.index, autopct='%1.1f%%')
        axes[0, 1].set_title('Regime Distribution')
    
    # 3. Allocation over time
    if len(diagnostics) > 0 and 'allocation' in diagnostics.columns:
        axes[1, 0].plot(diagnostics['date'], diagnostics['allocation'], 
                       linewidth=2, color='green')
        axes[1, 0].set_title('Portfolio Allocation Over Time')
        axes[1, 0].set_ylabel('Allocation %')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Performance metrics table
    metrics_text = '\n'.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
    axes[1, 1].text(0.1, 0.9, metrics_text, transform=axes[1, 1].transAxes, 
                   fontsize=10, verticalalignment='top', fontfamily='monospace')
    axes[1, 1].set_title('Performance Metrics')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print metrics
    print(f"\n{title} - Performance Summary:")
    print("=" * 50)
    for metric, value in metrics.items():
        print(f"{metric.replace('_', ' ').title()}: {value:.4f}")

def main():
    """
    Main execution function
    """
    print("QVM Engine v3e - Percentile-based Regime Detection")
    print("=" * 60)
    
    # Load configuration from config file
    try:
        with open('config/config_v3e_percentile_regime.yml', 'r') as f:
            config = yaml.safe_load(f)
        print("Configuration loaded from config/config_v3e_percentile_regime.yml")
    except FileNotFoundError:
        print("Config file not found, using default configuration")
        # Fallback configuration with percentile-based regime detection
        config = {
            "start_date": "2016-01-01",
            "end_date": "2025-01-01",
            "regime": {
                "lookback_period": 90,
                "volatility_percentile_high": 75.0,  # 75th percentile for high volatility
                "return_percentile_high": 75.0,      # 75th percentile for high return
                "return_percentile_low": 25.0        # 25th percentile for low return
            }
        }
    
    # Create database connection
    db_engine = create_db_connection()
    
    # Load data
    price_data, fundamental_data, returns_matrix, benchmark_returns = load_all_data_for_backtest(config, db_engine)
    
    # Initialize and run engine
    engine = QVMEngineV3AdoptedInsights(
        config=config,
        price_data=price_data,
        fundamental_data=fundamental_data,
        returns_matrix=returns_matrix,
        benchmark_returns=benchmark_returns,
        db_engine=db_engine
    )
    
    # Run backtest
    strategy_returns, diagnostics = engine.run_backtest()
    
    # Generate tearsheet
    generate_comprehensive_tearsheet(strategy_returns, benchmark_returns, diagnostics, 
                                   "QVM Engine v3e - Percentile-based Regime Detection")
    
    # Print regime statistics
    if len(engine.regime_history) > 0:
        regime_distribution = pd.Series(engine.regime_history).value_counts()
        print(f"\nRegime Distribution:")
        print("=" * 30)
        for regime, count in regime_distribution.items():
            percentage = (count / len(engine.regime_history)) * 100
            print(f"{regime}: {count} ({percentage:.1f}%)")

if __name__ == "__main__":
    main() 