# ============================================================================
# Single Factor Strategies for QVM Engine v3
# ============================================================================
# 
# This module provides single factor strategy implementations for:
# - Quality Factor (Q): Using ROAA as the primary quality metric
# - Value Factor (V): Using P/E ratio as the value metric  
# - Momentum Factor (M): Using multi-horizon momentum
#
# These strategies can be used to validate the composite QVM strategy
# and provide benchmarks for factor effectiveness analysis.
# ============================================================================

import pandas as pd
import numpy as np
from sqlalchemy import text
from datetime import datetime

class SingleFactorEngine:
    """
    Base class for single factor strategies (Quality, Value, Momentum).
    """
    def __init__(self, config: dict, price_data: pd.DataFrame, fundamental_data: pd.DataFrame,
                 returns_matrix: pd.DataFrame, benchmark_returns: pd.Series, db_engine, factor_name: str):
        
        self.config = config
        self.engine = db_engine
        self.factor_name = factor_name
        
        # Slice data to the exact backtest window
        start = pd.Timestamp(config['backtest_start_date'])
        end = pd.Timestamp(config['backtest_end_date'])
        
        self.price_data_raw = price_data[price_data['date'].between(start, end)].copy()
        self.fundamental_data_raw = fundamental_data[fundamental_data['date'].between(start, end)].copy()
        self.daily_returns_matrix = returns_matrix.loc[start:end].copy()
        self.benchmark_returns = benchmark_returns.loc[start:end].copy()
        
        # Initialize components
        self.regime_detector = None  # Will be set by subclasses
        self.sector_calculator = None  # Will be set by subclasses
        
        print(f"✅ {factor_name} Single Factor Engine initialized.")
        print(f"   - Strategy: {factor_name}_Single_Factor")
        print(f"   - Period: {self.daily_returns_matrix.index.min().date()} to {self.daily_returns_matrix.index.max().date()}")

    def run_backtest(self) -> (pd.Series, pd.DataFrame):
        """Executes the full backtesting pipeline for single factor."""
        print(f"\n🚀 Starting {self.factor_name} Single Factor backtest execution...")
        
        rebalance_dates = self._generate_rebalance_dates()
        daily_holdings, diagnostics = self._run_backtesting_loop(rebalance_dates)
        net_returns = self._calculate_net_returns(daily_holdings)
        
        print(f"✅ {self.factor_name} Single Factor backtest execution complete.")
        return net_returns, diagnostics

    def _generate_rebalance_dates(self) -> list:
        """Generates monthly rebalance dates based on actual trading days."""
        all_trading_dates = self.daily_returns_matrix.index
        rebal_dates_calendar = pd.date_range(
            start=self.config['backtest_start_date'],
            end=self.config['backtest_end_date'],
            freq=self.config['rebalance_frequency']
        )
        actual_rebal_dates = [all_trading_dates[all_trading_dates.searchsorted(d, side='left')-1] for d in rebal_dates_calendar if d >= all_trading_dates.min()]
        return sorted(list(set(actual_rebal_dates)))

    def _run_backtesting_loop(self, rebalance_dates: list) -> (pd.DataFrame, pd.DataFrame):
        """The core loop for portfolio construction at each rebalance date."""
        daily_holdings = pd.DataFrame(0.0, index=self.daily_returns_matrix.index, columns=self.daily_returns_matrix.columns)
        diagnostics_log = []
        
        for i, rebal_date in enumerate(rebalance_dates):
            print(f"   - Processing rebalance {i+1}/{len(rebalance_dates)}: {rebal_date.date()}...", end="")
            
            # Get universe
            universe = self._get_universe(rebal_date)
            if len(universe) < 5:
                print(" ⚠️ Universe too small. Skipping.")
                continue
            
            # Detect regime
            regime = self._detect_current_regime(rebal_date)
            regime_allocation = self._get_regime_allocation(regime)
            
            # Calculate single factor
            factors_df = self._calculate_single_factor(universe, rebal_date)
            if factors_df.empty:
                print(" ⚠️ No factor data. Skipping.")
                continue
            
            # Apply entry criteria
            qualified_df = self._apply_entry_criteria(factors_df)
            if qualified_df.empty:
                print(" ⚠️ No qualified stocks. Skipping.")
                continue
            
            # Construct portfolio
            target_portfolio = self._construct_portfolio(qualified_df, regime_allocation)
            if target_portfolio.empty:
                print(" ⚠️ Portfolio empty. Skipping.")
                continue
            
            # Apply holdings
            start_period = rebal_date + pd.Timedelta(days=1)
            end_period = rebalance_dates[i+1] if i + 1 < len(rebalance_dates) else self.daily_returns_matrix.index.max()
            holding_dates = self.daily_returns_matrix.index[(self.daily_returns_matrix.index >= start_period) & (self.daily_returns_matrix.index <= end_period)]
            
            daily_holdings.loc[holding_dates] = 0.0
            valid_tickers = target_portfolio.index.intersection(daily_holdings.columns)
            daily_holdings.loc[holding_dates, valid_tickers] = target_portfolio[valid_tickers].values
            
            # Calculate turnover
            if i > 0:
                try:
                    prev_holdings_idx = self.daily_returns_matrix.index.get_loc(rebal_date) - 1
                except KeyError:
                    prev_dates = self.daily_returns_matrix.index[self.daily_returns_matrix.index < rebal_date]
                    if len(prev_dates) > 0:
                        prev_holdings_idx = self.daily_returns_matrix.index.get_loc(prev_dates[-1])
                    else:
                        prev_holdings_idx = -1
                
                prev_holdings = daily_holdings.iloc[prev_holdings_idx] if prev_holdings_idx >= 0 else pd.Series(dtype='float64')
            else:
                prev_holdings = pd.Series(dtype='float64')

            turnover = (target_portfolio - prev_holdings.reindex(target_portfolio.index).fillna(0)).abs().sum() / 2.0
            
            diagnostics_log.append({
                'date': rebal_date,
                'universe_size': len(universe),
                'portfolio_size': len(target_portfolio),
                'regime': regime,
                'regime_allocation': regime_allocation,
                'turnover': turnover
            })
            print(f" ✅ Universe: {len(universe)}, Portfolio: {len(target_portfolio)}, Regime: {regime}, Turnover: {turnover:.1%}")

        if diagnostics_log:
            return daily_holdings, pd.DataFrame(diagnostics_log).set_index('date')
        else:
            return daily_holdings, pd.DataFrame()

    def _get_universe(self, analysis_date: pd.Timestamp) -> list:
        """Get liquid universe based on ADTV and market cap filters."""
        lookback_days = self.config['universe']['lookback_days']
        adtv_threshold = self.config['universe']['adtv_threshold_shares']
        min_market_cap = self.config['universe']['min_market_cap_bn'] * 1e9
        
        universe_query = text("""
            SELECT 
                ticker,
                AVG(total_volume) as avg_volume,
                AVG(market_cap) as avg_market_cap
            FROM vcsc_daily_data_complete
            WHERE trading_date <= :analysis_date
              AND trading_date >= DATE_SUB(:analysis_date, INTERVAL :lookback_days DAY)
            GROUP BY ticker
            HAVING avg_volume >= :adtv_threshold AND avg_market_cap >= :min_market_cap
        """)
        
        universe_df = pd.read_sql(universe_query, self.engine, 
                                 params={'analysis_date': analysis_date, 'lookback_days': lookback_days, 'adtv_threshold': adtv_threshold, 'min_market_cap': min_market_cap})
        
        return universe_df['ticker'].tolist()

    def _detect_current_regime(self, analysis_date: pd.Timestamp) -> str:
        """Detect current market regime."""
        lookback_days = self.config['regime']['lookback_period']
        start_date = analysis_date - pd.Timedelta(days=lookback_days)
        
        benchmark_data = self.benchmark_returns.loc[start_date:analysis_date]
        if len(benchmark_data) < lookback_days // 2:
            return 'Sideways'
        
        price_series = (1 + benchmark_data).cumprod()
        price_data = pd.DataFrame({'close': price_series})
        
        return self._detect_regime(price_data)

    def _detect_regime(self, price_data: pd.DataFrame) -> str:
        """Detect market regime based on volatility and return."""
        lookback_period = self.config['regime']['lookback_period']
        
        if len(price_data) < lookback_period:
            return 'Sideways'
        
        recent_data = price_data.tail(lookback_period)
        returns = recent_data['close'].pct_change().dropna()
        
        volatility = returns.std()
        avg_return = returns.mean()
        
        # Optimal thresholds from comprehensive testing
        if volatility > self.config['regime']['volatility_threshold']:
            if avg_return > self.config['regime']['return_threshold']:
                return 'Bull'
            else:
                return 'Bear'
        else:
            if abs(avg_return) < 0.001:
                return 'Sideways'
            else:
                return 'Stress'

    def _get_regime_allocation(self, regime: str) -> float:
        """Get target allocation based on regime."""
        regime_allocations = {
            'Bull': 1.0,      # Fully invested
            'Bear': 0.8,      # 80% invested
            'Sideways': 0.6,  # 60% invested
            'Stress': 0.4     # 40% invested
        }
        return regime_allocations.get(regime, 0.6)

    def _calculate_single_factor(self, universe: list, analysis_date: pd.Timestamp) -> pd.DataFrame:
        """Calculate single factor - to be implemented by subclasses."""
        raise NotImplementedError

    def _apply_entry_criteria(self, factors_df: pd.DataFrame) -> pd.DataFrame:
        """Apply entry criteria to filter stocks."""
        qualified = factors_df.copy()
        
        if 'roaa' in qualified.columns:
            qualified = qualified[qualified['roaa'] > 0]
        
        if 'net_margin' in qualified.columns:
            qualified = qualified[qualified['net_margin'] > 0]
        
        return qualified

    def _construct_portfolio(self, qualified_df: pd.DataFrame, regime_allocation: float) -> pd.Series:
        """Construct the portfolio using the qualified stocks."""
        if qualified_df.empty:
            return pd.Series(dtype='float64')
        
        # Sort by factor score
        factor_score_col = f'{self.factor_name.lower()}_score'
        if factor_score_col not in qualified_df.columns:
            return pd.Series(dtype='float64')
        
        qualified_df = qualified_df.sort_values(factor_score_col, ascending=False)
        
        target_size = self.config['universe']['target_portfolio_size']
        selected_stocks = qualified_df.head(target_size)
        
        if selected_stocks.empty:
            return pd.Series(dtype='float64')
        
        portfolio = pd.Series(regime_allocation / len(selected_stocks), index=selected_stocks['ticker'])
        
        return portfolio

    def _calculate_net_returns(self, daily_holdings: pd.DataFrame) -> pd.Series:
        """Calculate net returns with transaction costs."""
        holdings_shifted = daily_holdings.shift(1).fillna(0.0)
        gross_returns = (holdings_shifted * self.daily_returns_matrix).sum(axis=1)
        
        turnover = (holdings_shifted - holdings_shifted.shift(1)).abs().sum(axis=1) / 2.0
        costs = turnover * (self.config['transaction_cost_bps'] / 10000)
        net_returns = (gross_returns - costs).rename(f'{self.factor_name}_Single_Factor')
        
        return net_returns

class QualityFactorEngine(SingleFactorEngine):
    """Quality factor engine using ROAA as the primary quality metric."""
    
    def __init__(self, config: dict, price_data: pd.DataFrame, fundamental_data: pd.DataFrame,
                 returns_matrix: pd.DataFrame, benchmark_returns: pd.Series, db_engine):
        super().__init__(config, price_data, fundamental_data, returns_matrix, benchmark_returns, db_engine, "Quality")

    def _calculate_single_factor(self, universe: list, analysis_date: pd.Timestamp) -> pd.DataFrame:
        """Calculate quality factor using ROAA."""
        try:
            lag_days = self.config['factors']['fundamental_lag_days']
            lag_date = analysis_date - pd.Timedelta(days=lag_days)
            lag_year = lag_date.year
            
            ticker_list = "','".join(universe)
            
            quality_query = text(f"""
                WITH netprofit_ttm AS (
                    SELECT 
                        fv.ticker,
                        SUM(fv.value / 1e9) as netprofit_ttm
                    FROM fundamental_values fv
                    WHERE fv.ticker IN ('{ticker_list}')
                    AND fv.item_id = 1
                    AND fv.statement_type = 'PL'
                    AND fv.year <= {lag_year}
                    AND fv.year >= {lag_year - 1}
                    GROUP BY fv.ticker
                ),
                totalassets_ttm AS (
                    SELECT 
                        fv.ticker,
                        SUM(fv.value / 1e9) as totalassets_ttm
                    FROM fundamental_values fv
                    WHERE fv.ticker IN ('{ticker_list}')
                    AND fv.item_id = 2
                    AND fv.statement_type = 'BS'
                    AND fv.year <= {lag_year}
                    AND fv.year >= {lag_year - 1}
                    GROUP BY fv.ticker
                )
                SELECT 
                    np.ticker,
                    np.netprofit_ttm,
                    ta.totalassets_ttm,
                    CASE 
                        WHEN ta.totalassets_ttm > 0 THEN np.netprofit_ttm / ta.totalassets_ttm 
                        ELSE NULL 
                    END as roaa
                FROM netprofit_ttm np
                LEFT JOIN totalassets_ttm ta ON np.ticker = ta.ticker
                WHERE np.netprofit_ttm > 0 
                AND ta.totalassets_ttm > 0
            """)
            
            quality_df = pd.read_sql(quality_query, self.engine)
            
            if quality_df.empty:
                return pd.DataFrame()
            
            # Calculate quality score (normalized ROAA)
            quality_df['quality_score'] = (quality_df['roaa'] - quality_df['roaa'].mean()) / quality_df['roaa'].std()
            
            return quality_df
            
        except Exception as e:
            print(f"Error calculating quality factor: {e}")
            return pd.DataFrame()

class ValueFactorEngine(SingleFactorEngine):
    """Value factor engine using P/E ratio."""
    
    def __init__(self, config: dict, price_data: pd.DataFrame, fundamental_data: pd.DataFrame,
                 returns_matrix: pd.DataFrame, benchmark_returns: pd.Series, db_engine):
        super().__init__(config, price_data, fundamental_data, returns_matrix, benchmark_returns, db_engine, "Value")

    def _calculate_single_factor(self, universe: list, analysis_date: pd.Timestamp) -> pd.DataFrame:
        """Calculate value factor using P/E ratio."""
        try:
            lag_days = self.config['factors']['fundamental_lag_days']
            lag_date = analysis_date - pd.Timedelta(days=lag_days)
            lag_year = lag_date.year
            
            ticker_list = "','".join(universe)
            
            # Get fundamental data
            fundamental_query = text(f"""
                WITH netprofit_ttm AS (
                    SELECT 
                        fv.ticker,
                        SUM(fv.value / 1e9) as netprofit_ttm
                    FROM fundamental_values fv
                    WHERE fv.ticker IN ('{ticker_list}')
                    AND fv.item_id = 1
                    AND fv.statement_type = 'PL'
                    AND fv.year <= {lag_year}
                    AND fv.year >= {lag_year - 1}
                    GROUP BY fv.ticker
                )
                SELECT 
                    np.ticker,
                    np.netprofit_ttm
                FROM netprofit_ttm np
                WHERE np.netprofit_ttm > 0
            """)
            
            fundamental_df = pd.read_sql(fundamental_query, self.engine)
            
            if fundamental_df.empty:
                return pd.DataFrame()
            
            # Get market data
            market_ticker_list = "','".join(universe)
            market_query = text(f"""
                SELECT 
                    ticker,
                    market_cap
                FROM vcsc_daily_data_complete
                WHERE trading_date <= :analysis_date
                  AND ticker IN ('{market_ticker_list}')
                ORDER BY ticker, trading_date DESC
            """)
            
            market_df = pd.read_sql(market_query, self.engine, params={'analysis_date': analysis_date})
            
            if market_df.empty:
                return pd.DataFrame()
            
            # Calculate P/E ratio
            value_data = []
            for _, row in fundamental_df.iterrows():
                ticker = row['ticker']
                market_data = market_df[market_df['ticker'] == ticker]
                
                if len(market_data) == 0:
                    continue
                    
                market_cap = market_data.iloc[0]['market_cap']
                net_profit = row['netprofit_ttm'] * 1e9  # Convert back to VND
                
                if net_profit > 0:
                    pe_ratio = market_cap / net_profit
                    value_data.append({
                        'ticker': ticker,
                        'pe_ratio': pe_ratio,
                        'market_cap': market_cap,
                        'net_profit': net_profit
                    })
            
            value_df = pd.DataFrame(value_data)
            
            if value_df.empty:
                return pd.DataFrame()
            
            # Calculate value score (inverse P/E - lower P/E is better)
            value_df['value_score'] = -1 * (value_df['pe_ratio'] - value_df['pe_ratio'].mean()) / value_df['pe_ratio'].std()
            
            return value_df
            
        except Exception as e:
            print(f"Error calculating value factor: {e}")
            return pd.DataFrame()

class MomentumFactorEngine(SingleFactorEngine):
    """Momentum factor engine using multi-horizon momentum."""
    
    def __init__(self, config: dict, price_data: pd.DataFrame, fundamental_data: pd.DataFrame,
                 returns_matrix: pd.DataFrame, benchmark_returns: pd.Series, db_engine):
        super().__init__(config, price_data, fundamental_data, returns_matrix, benchmark_returns, db_engine, "Momentum")

    def _calculate_single_factor(self, universe: list, analysis_date: pd.Timestamp) -> pd.DataFrame:
        """Calculate momentum factor using multi-horizon momentum."""
        try:
            ticker_list = "','".join(universe)
            
            market_query = text(f"""
                SELECT 
                    ticker,
                    trading_date,
                    close_price_adjusted as close
                FROM vcsc_daily_data_complete
                WHERE trading_date <= :analysis_date
                  AND ticker IN ('{ticker_list}')
                ORDER BY ticker, trading_date DESC
            """)
            
            market_df = pd.read_sql(market_query, self.engine, params={'analysis_date': analysis_date})
            
            if market_df.empty:
                return pd.DataFrame()
            
            # Calculate momentum factors
            momentum_data = []
            skip_months = self.config['factors']['skip_months']
            
            for ticker in market_df['ticker'].unique():
                ticker_data = market_df[market_df['ticker'] == ticker].sort_values('trading_date')
                
                if len(ticker_data) < 252 + skip_months:
                    continue
                    
                current_price = ticker_data.iloc[skip_months]['close']
                
                periods = self.config['factors']['momentum_horizons']
                momentum_factors = {'ticker': ticker}
                
                for period in periods:
                    if len(ticker_data) >= period + skip_months:
                        past_price = ticker_data.iloc[period + skip_months - 1]['close']
                        momentum_factors[f'momentum_{period}d'] = (current_price / past_price) - 1
                    else:
                        momentum_factors[f'momentum_{period}d'] = 0
                
                momentum_data.append(momentum_factors)
            
            momentum_df = pd.DataFrame(momentum_data)
            
            if momentum_df.empty:
                return pd.DataFrame()
            
            # Calculate momentum score (equal weighted)
            momentum_columns = [col for col in momentum_df.columns if col.startswith('momentum_')]
            momentum_df['momentum_score'] = momentum_df[momentum_columns].mean(axis=1)
            
            return momentum_df
            
        except Exception as e:
            print(f"Error calculating momentum factor: {e}")
            return pd.DataFrame()

def run_single_factor_analysis(config: dict, price_data: pd.DataFrame, fundamental_data: pd.DataFrame,
                              returns_matrix: pd.DataFrame, benchmark_returns: pd.Series, db_engine,
                              qvm_returns: pd.Series):
    """
    Run all single factor strategies and generate comparison analysis.
    
    Args:
        config: Strategy configuration
        price_data: Price data DataFrame
        fundamental_data: Fundamental data DataFrame
        returns_matrix: Daily returns matrix
        benchmark_returns: Benchmark returns series
        db_engine: Database engine
        qvm_returns: QVM composite strategy returns for comparison
    
    Returns:
        Dictionary containing all strategy returns and comparison metrics
    """
    print("\n" + "="*80)
    print("📊 SINGLE FACTOR STRATEGIES EXECUTION")
    print("="*80)

    try:
        # Quality Factor Strategy
        print("\n🔍 Running Quality Factor Strategy...")
        quality_engine = QualityFactorEngine(
            config=config,
            price_data=price_data,
            fundamental_data=fundamental_data,
            returns_matrix=returns_matrix,
            benchmark_returns=benchmark_returns,
            db_engine=db_engine
        )
        quality_returns, quality_diagnostics = quality_engine.run_backtest()

        # Value Factor Strategy
        print("\n💰 Running Value Factor Strategy...")
        value_engine = ValueFactorEngine(
            config=config,
            price_data=price_data,
            fundamental_data=fundamental_data,
            returns_matrix=returns_matrix,
            benchmark_returns=benchmark_returns,
            db_engine=db_engine
        )
        value_returns, value_diagnostics = value_engine.run_backtest()

        # Momentum Factor Strategy
        print("\n📈 Running Momentum Factor Strategy...")
        momentum_engine = MomentumFactorEngine(
            config=config,
            price_data=price_data,
            fundamental_data=fundamental_data,
            returns_matrix=returns_matrix,
            benchmark_returns=benchmark_returns,
            db_engine=db_engine
        )
        momentum_returns, momentum_diagnostics = momentum_engine.run_backtest()

        # Return results
        results = {
            'quality_returns': quality_returns,
            'quality_diagnostics': quality_diagnostics,
            'value_returns': value_returns,
            'value_diagnostics': value_diagnostics,
            'momentum_returns': momentum_returns,
            'momentum_diagnostics': momentum_diagnostics,
            'qvm_returns': qvm_returns
        }

        print("\n✅ Single factor strategies successfully executed!")
        return results

    except Exception as e:
        print(f"❌ An error occurred during single factor strategy execution: {e}")
        raise

def calculate_performance_metrics(returns: pd.Series, benchmark: pd.Series, periods_per_year: int = 252) -> dict:
    """Calculates a dictionary of institutional performance metrics with corrected alignment."""
    # Align benchmark
    first_trade_date = returns.loc[returns.ne(0)].index.min()
    if pd.isna(first_trade_date):
        return {metric: 0.0 for metric in ['Annualized Return (%)', 'Annualized Volatility (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 'Calmar Ratio', 'Information Ratio', 'Beta']}
    
    aligned_returns = returns.loc[first_trade_date:]
    aligned_benchmark = benchmark.loc[first_trade_date:]

    n_years = len(aligned_returns) / periods_per_year
    annualized_return = ((1 + aligned_returns).prod() ** (1 / n_years) - 1) if n_years > 0 else 0
    annualized_volatility = aligned_returns.std() * np.sqrt(periods_per_year)
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else 0.0
    
    cumulative_returns = (1 + aligned_returns).cumprod()
    max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0.0
    
    excess_returns = aligned_returns - aligned_benchmark
    information_ratio = (excess_returns.mean() * periods_per_year) / (excess_returns.std() * np.sqrt(periods_per_year)) if excess_returns.std() > 0 else 0.0
    beta = aligned_returns.cov(aligned_benchmark) / aligned_benchmark.var() if aligned_benchmark.var() > 0 else 0.0
    
    return {
        'Annualized Return (%)': annualized_return * 100,
        'Annualized Volatility (%)': annualized_volatility * 100,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown (%)': max_drawdown * 100,
        'Calmar Ratio': calmar_ratio,
        'Information Ratio': information_ratio,
        'Beta': beta
    }

def generate_factor_comparison_analysis(results: dict, benchmark_returns: pd.Series):
    """
    Generate comprehensive comparison analysis between single factors and QVM composite.
    
    Args:
        results: Dictionary containing all strategy returns
        benchmark_returns: Benchmark returns series
    """
    print("\n" + "="*80)
    print("📊 SINGLE FACTOR STRATEGIES: PERFORMANCE COMPARISON")
    print("="*80)

    # Calculate metrics for all strategies
    strategies = {
        'QVM Composite': results['qvm_returns'],
        'Quality': results['quality_returns'],
        'Value': results['value_returns'],
        'Momentum': results['momentum_returns']
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
    import matplotlib.pyplot as plt
    
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

    return comparison_df 