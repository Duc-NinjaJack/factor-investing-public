# Single Factor Strategies for QVM Engine v3
import pandas as pd
from sqlalchemy import text

class SingleFactorEngine:
    """Base class for single factor strategies."""
    
    def __init__(self, config, price_data, fundamental_data, returns_matrix, benchmark_returns, db_engine, factor_name):
        self.config = config
        self.engine = db_engine
        self.factor_name = factor_name
        self.daily_returns_matrix = returns_matrix
        self.benchmark_returns = benchmark_returns

    def run_backtest(self):
        """Execute backtest for single factor."""
        print(f"Running {self.factor_name} factor strategy...")
        
        # Generate rebalance dates
        rebalance_dates = self._generate_rebalance_dates()
        daily_holdings = pd.DataFrame(0.0, index=self.daily_returns_matrix.index, columns=self.daily_returns_matrix.columns)
        
        for rebal_date in rebalance_dates:
            # Get universe and calculate factor
            universe = self._get_universe(rebal_date)
            factors_df = self._calculate_factor(universe, rebal_date)
            
            if not factors_df.empty:
                # Construct portfolio
                portfolio = self._construct_portfolio(factors_df)
                if not portfolio.empty:
                    # Apply holdings
                    self._apply_holdings(daily_holdings, portfolio, rebal_date, rebalance_dates)
        
        # Calculate returns
        net_returns = self._calculate_net_returns(daily_holdings)
        return net_returns, pd.DataFrame()

    def _generate_rebalance_dates(self):
        """Generate monthly rebalance dates."""
        all_trading_dates = self.daily_returns_matrix.index
        rebal_dates_calendar = pd.date_range(
            start=self.config['backtest_start_date'],
            end=self.config['backtest_end_date'],
            freq=self.config['rebalance_frequency']
        )
        actual_rebal_dates = [all_trading_dates[all_trading_dates.searchsorted(d, side='left')-1] 
                             for d in rebal_dates_calendar if d >= all_trading_dates.min()]
        return sorted(list(set(actual_rebal_dates)))

    def _get_universe(self, analysis_date):
        """Get liquid universe."""
        lookback_days = self.config['universe']['lookback_days']
        adtv_threshold = self.config['universe']['adtv_threshold_shares']
        min_market_cap = self.config['universe']['min_market_cap_bn'] * 1e9
        
        universe_query = text("""
            SELECT ticker
            FROM vcsc_daily_data_complete
            WHERE trading_date <= :analysis_date
              AND trading_date >= DATE_SUB(:analysis_date, INTERVAL :lookback_days DAY)
            GROUP BY ticker
            HAVING AVG(total_volume) >= :adtv_threshold AND AVG(market_cap) >= :min_market_cap
        """)
        
        universe_df = pd.read_sql(universe_query, self.engine, 
                                 params={'analysis_date': analysis_date, 'lookback_days': lookback_days, 
                                        'adtv_threshold': adtv_threshold, 'min_market_cap': min_market_cap})
        return universe_df['ticker'].tolist()

    def _calculate_factor(self, universe, analysis_date):
        """Calculate factor - to be implemented by subclasses."""
        raise NotImplementedError

    def _construct_portfolio(self, factors_df):
        """Construct portfolio from factor scores."""
        if factors_df.empty:
            return pd.Series(dtype='float64')
        
        # Sort by factor score and select top stocks
        factor_score_col = f'{self.factor_name.lower()}_score'
        if factor_score_col not in factors_df.columns:
            return pd.Series(dtype='float64')
        
        qualified_df = factors_df.sort_values(factor_score_col, ascending=False)
        target_size = self.config['universe']['target_portfolio_size']
        selected_stocks = qualified_df.head(target_size)
        
        if selected_stocks.empty:
            return pd.Series(dtype='float64')
        
        # Equal weight portfolio
        portfolio = pd.Series(1.0 / len(selected_stocks), index=selected_stocks['ticker'])
        return portfolio

    def _apply_holdings(self, daily_holdings, portfolio, rebal_date, rebalance_dates):
        """Apply portfolio holdings to daily holdings matrix."""
        # Find next rebalance date
        current_idx = rebalance_dates.index(rebal_date)
        next_rebal_date = rebalance_dates[current_idx + 1] if current_idx + 1 < len(rebalance_dates) else self.daily_returns_matrix.index.max()
        
        # Apply holdings for the period
        start_period = rebal_date + pd.Timedelta(days=1)
        end_period = next_rebal_date
        holding_dates = self.daily_returns_matrix.index[(self.daily_returns_matrix.index >= start_period) & 
                                                       (self.daily_returns_matrix.index <= end_period)]
        
        daily_holdings.loc[holding_dates] = 0.0
        valid_tickers = portfolio.index.intersection(daily_holdings.columns)
        daily_holdings.loc[holding_dates, valid_tickers] = portfolio[valid_tickers].values

    def _calculate_net_returns(self, daily_holdings):
        """Calculate net returns with transaction costs."""
        holdings_shifted = daily_holdings.shift(1).fillna(0.0)
        gross_returns = (holdings_shifted * self.daily_returns_matrix).sum(axis=1)
        
        # Apply transaction costs
        turnover = (holdings_shifted - holdings_shifted.shift(1)).abs().sum(axis=1) / 2.0
        costs = turnover * (self.config['transaction_cost_bps'] / 10000)
        net_returns = (gross_returns - costs).rename(f'{self.factor_name}_Single_Factor')
        
        return net_returns

class QualityFactorEngine(SingleFactorEngine):
    """Quality factor using ROAA."""
    
    def __init__(self, config, price_data, fundamental_data, returns_matrix, benchmark_returns, db_engine):
        super().__init__(config, price_data, fundamental_data, returns_matrix, benchmark_returns, db_engine, "Quality")

    def _calculate_factor(self, universe, analysis_date):
        """Calculate quality factor using ROAA."""
        try:
            lag_days = self.config['factors']['fundamental_lag_days']
            lag_date = analysis_date - pd.Timedelta(days=lag_days)
            lag_year = lag_date.year
            
            ticker_list = "','".join(universe)
            
            quality_query = text(f"""
                WITH netprofit_ttm AS (
                    SELECT fv.ticker, SUM(fv.value / 1e9) as netprofit_ttm
                    FROM fundamental_values fv
                    WHERE fv.ticker IN ('{ticker_list}')
                    AND fv.item_id = 1 AND fv.statement_type = 'PL'
                    AND fv.year <= {lag_year} AND fv.year >= {lag_year - 1}
                    GROUP BY fv.ticker
                ),
                totalassets_ttm AS (
                    SELECT fv.ticker, SUM(fv.value / 1e9) as totalassets_ttm
                    FROM fundamental_values fv
                    WHERE fv.ticker IN ('{ticker_list}')
                    AND fv.item_id = 2 AND fv.statement_type = 'BS'
                    AND fv.year <= {lag_year} AND fv.year >= {lag_year - 1}
                    GROUP BY fv.ticker
                )
                SELECT np.ticker, np.netprofit_ttm, ta.totalassets_ttm,
                       CASE WHEN ta.totalassets_ttm > 0 THEN np.netprofit_ttm / ta.totalassets_ttm ELSE NULL END as roaa
                FROM netprofit_ttm np
                LEFT JOIN totalassets_ttm ta ON np.ticker = ta.ticker
                WHERE np.netprofit_ttm > 0 AND ta.totalassets_ttm > 0
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
    """Value factor using P/E ratio."""
    
    def __init__(self, config, price_data, fundamental_data, returns_matrix, benchmark_returns, db_engine):
        super().__init__(config, price_data, fundamental_data, returns_matrix, benchmark_returns, db_engine, "Value")

    def _calculate_factor(self, universe, analysis_date):
        """Calculate value factor using P/E ratio."""
        try:
            lag_days = self.config['factors']['fundamental_lag_days']
            lag_date = analysis_date - pd.Timedelta(days=lag_days)
            lag_year = lag_date.year
            
            ticker_list = "','".join(universe)
            
            # Get fundamental data
            fundamental_query = text(f"""
                SELECT fv.ticker, SUM(fv.value / 1e9) as netprofit_ttm
                FROM fundamental_values fv
                WHERE fv.ticker IN ('{ticker_list}')
                AND fv.item_id = 1 AND fv.statement_type = 'PL'
                AND fv.year <= {lag_year} AND fv.year >= {lag_year - 1}
                GROUP BY fv.ticker
                HAVING SUM(fv.value / 1e9) > 0
            """)
            
            fundamental_df = pd.read_sql(fundamental_query, self.engine)
            
            if fundamental_df.empty:
                return pd.DataFrame()
            
            # Get market data
            market_ticker_list = "','".join(universe)
            market_query = text(f"""
                SELECT ticker, market_cap
                FROM vcsc_daily_data_complete
                WHERE trading_date <= :analysis_date AND ticker IN ('{market_ticker_list}')
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
                
                if len(market_data) > 0:
                    market_cap = market_data.iloc[0]['market_cap']
                    net_profit = row['netprofit_ttm'] * 1e9
                    
                    if net_profit > 0:
                        pe_ratio = market_cap / net_profit
                        value_data.append({'ticker': ticker, 'pe_ratio': pe_ratio})
            
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
    """Momentum factor using multi-horizon momentum."""
    
    def __init__(self, config, price_data, fundamental_data, returns_matrix, benchmark_returns, db_engine):
        super().__init__(config, price_data, fundamental_data, returns_matrix, benchmark_returns, db_engine, "Momentum")

    def _calculate_factor(self, universe, analysis_date):
        """Calculate momentum factor using multi-horizon momentum."""
        try:
            ticker_list = "','".join(universe)
            
            market_query = text(f"""
                SELECT ticker, trading_date, close_price_adjusted as close
                FROM vcsc_daily_data_complete
                WHERE trading_date <= :analysis_date AND ticker IN ('{ticker_list}')
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

def run_single_factors(config, price_data, fundamental_data, returns_matrix, benchmark_returns, db_engine, qvm_returns):
    """Run all single factor strategies and return results."""
    print("\n" + "="*80)
    print("📊 SINGLE FACTOR STRATEGIES EXECUTION")
    print("="*80)

    try:
        # Quality Factor
        print("\n🔍 Running Quality Factor Strategy...")
        quality_engine = QualityFactorEngine(config, price_data, fundamental_data, returns_matrix, benchmark_returns, db_engine)
        quality_returns, _ = quality_engine.run_backtest()

        # Value Factor
        print("\n💰 Running Value Factor Strategy...")
        value_engine = ValueFactorEngine(config, price_data, fundamental_data, returns_matrix, benchmark_returns, db_engine)
        value_returns, _ = value_engine.run_backtest()

        # Momentum Factor
        print("\n📈 Running Momentum Factor Strategy...")
        momentum_engine = MomentumFactorEngine(config, price_data, fundamental_data, returns_matrix, benchmark_returns, db_engine)
        momentum_returns, _ = momentum_engine.run_backtest()

        # Return results
        results = {
            'quality_returns': quality_returns,
            'value_returns': value_returns,
            'momentum_returns': momentum_returns,
            'qvm_returns': qvm_returns
        }

        print("\n✅ Single factor strategies successfully executed!")
        return results

    except Exception as e:
        print(f"❌ Error during single factor execution: {e}")
        raise 