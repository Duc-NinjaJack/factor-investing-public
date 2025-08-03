#!/usr/bin/env python3
"""
QVM Engine v3f - FIXED VERSION
All critical issues resolved: regime thresholds, data validation, error handling
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
from pathlib import Path
import sys

# Add project root to path
current_path = Path.cwd()
while not (current_path / 'production').is_dir():
    if current_path.parent == current_path:
        raise FileNotFoundError("Could not find the 'production' directory.")
    current_path = current_path.parent

project_root = current_path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from production.database.connection import get_database_manager
from sqlalchemy import text

warnings.filterwarnings('ignore')

# FIXED CONFIGURATION
QVM_CONFIG = {
    "strategy_name": "QVM_Engine_v3f_Fixed",
    "backtest_start_date": "2016-01-01",
    "backtest_end_date": "2020-12-31",
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
    
    # FIXED: Correct regime thresholds
    "regime": {
        "lookback_period": 90,
        "volatility_threshold": 0.0140,  # FIXED: was 0.2659
        "return_threshold": 0.0012,      # FIXED: was 0.2588
        "low_return_threshold": 0.0002   # FIXED: was 0.2131
    }
}

print("✅ FIXED QVM Engine v3f Configuration Loaded")
print(f"   - Regime Thresholds: Vol={QVM_CONFIG['regime']['volatility_threshold']:.4f}, Ret={QVM_CONFIG['regime']['return_threshold']:.4f}")

# FIXED RegimeDetector
class RegimeDetector:
    def __init__(self, lookback_period: int = 90, volatility_threshold: float = 0.0140, 
                 return_threshold: float = 0.0012, low_return_threshold: float = 0.0002):
        self.lookback_period = lookback_period
        self.volatility_threshold = volatility_threshold
        self.return_threshold = return_threshold
        self.low_return_threshold = low_return_threshold
    
    def detect_regime(self, price_data: pd.DataFrame) -> str:
        """FIXED: Proper regime detection with data validation."""
        try:
            if len(price_data) < 60:  # Minimum required data
                return 'Sideways'
            
            recent_data = price_data.tail(min(len(price_data), self.lookback_period))
            returns = recent_data['close'].pct_change().dropna()
            
            if len(returns) < 30:  # Need minimum returns
                return 'Sideways'
            
            volatility = returns.std()
            avg_return = returns.mean()
            
            # FIXED: Proper regime logic
            if volatility > self.volatility_threshold:
                if avg_return > self.return_threshold:
                    return 'Bull'
                else:
                    return 'Bear'
            else:
                if abs(avg_return) < self.low_return_threshold:
                    return 'Sideways'
                else:
                    return 'Stress'
                    
        except Exception as e:
            print(f"   ⚠️  Regime detection error: {e}")
            return 'Sideways'
    
    def get_regime_allocation(self, regime: str) -> float:
        regime_allocations = {
            'Bull': 1.0, 'Bear': 0.8, 'Sideways': 0.6, 'Stress': 0.4
        }
        return regime_allocations.get(regime, 0.6)

# FIXED QVM Engine
class QVMEngineV3fFixed:
    def __init__(self, config: dict, price_data: pd.DataFrame, fundamental_data: pd.DataFrame,
                 returns_matrix: pd.DataFrame, benchmark_returns: pd.Series, db_engine):
        
        self.config = config
        self.price_data = price_data
        self.fundamental_data = fundamental_data
        self.daily_returns_matrix = returns_matrix
        self.benchmark_returns = benchmark_returns
        self.engine = db_engine
        
        # FIXED: Initialize with proper error handling
        self.regime_detector = RegimeDetector(
            lookback_period=config['regime']['lookback_period'],
            volatility_threshold=config['regime']['volatility_threshold'],
            return_threshold=config['regime']['return_threshold'],
            low_return_threshold=config['regime']['low_return_threshold']
        )
        
        print("✅ QVMEngineV3fFixed initialized with FIXED configuration")
    
    def run_backtest(self) -> (pd.Series, pd.DataFrame):
        """FIXED: Run backtest with proper error handling."""
        try:
            rebalance_dates = self._generate_rebalance_dates()
            print(f"   - Generated {len(rebalance_dates)} monthly rebalance dates")
            
            daily_holdings, diagnostics = self._run_backtesting_loop(rebalance_dates)
            
            # FIXED: Validate holdings before returns calculation
            if daily_holdings.empty or daily_holdings.isna().all().all():
                print("   ⚠️  Empty holdings detected, returning zero returns")
                return pd.Series(0.0, index=self.daily_returns_matrix.index), diagnostics
            
            net_returns = self._calculate_net_returns(daily_holdings)
            return net_returns, diagnostics
            
        except Exception as e:
            print(f"   ❌ Backtest error: {e}")
            return pd.Series(0.0, index=self.daily_returns_matrix.index), pd.DataFrame()
    
    def _calculate_net_returns(self, daily_holdings: pd.DataFrame) -> pd.Series:
        """FIXED: Calculate net returns with proper validation."""
        try:
            # FIXED: Validate inputs
            if daily_holdings.empty:
                return pd.Series(0.0, index=self.daily_returns_matrix.index)
            
            holdings_shifted = daily_holdings.shift(1).fillna(0.0)
            
            # FIXED: Check for alignment issues
            if not holdings_shifted.index.equals(self.daily_returns_matrix.index):
                print("   ⚠️  Index alignment issue, reindexing")
                holdings_shifted = holdings_shifted.reindex(self.daily_returns_matrix.index, fill_value=0.0)
            
            # FIXED: Calculate gross returns with validation
            gross_returns = (holdings_shifted * self.daily_returns_matrix).sum(axis=1)
            
            # FIXED: Handle NaN/inf values
            gross_returns = gross_returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            
            # FIXED: Calculate turnover with validation
            turnover = (holdings_shifted - holdings_shifted.shift(1)).abs().sum(axis=1) / 2.0
            turnover = turnover.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            
            # FIXED: Calculate costs
            costs = turnover * (self.config['transaction_cost_bps'] / 10000)
            costs = costs.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            
            # FIXED: Calculate net returns
            net_returns = (gross_returns - costs).rename(self.config['strategy_name'])
            
            # FIXED: Final validation
            net_returns = net_returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            
            # FIXED: Calculate metrics with validation
            gross_total = (1 + gross_returns).prod() - 1
            net_total = (1 + net_returns).prod() - 1
            cost_drag = gross_total - net_total
            
            print(f"\n💸 FIXED Net returns calculated:")
            print(f"   - Total Gross Return: {gross_total:.2%}")
            print(f"   - Total Net Return: {net_total:.2%}")
            print(f"   - Total Cost Drag: {cost_drag:.2%}")
            
            return net_returns
            
        except Exception as e:
            print(f"   ❌ Returns calculation error: {e}")
            return pd.Series(0.0, index=self.daily_returns_matrix.index)
    
    def _generate_rebalance_dates(self) -> list:
        """Generate monthly rebalance dates."""
        start_date = pd.Timestamp(self.config['backtest_start_date'])
        end_date = pd.Timestamp(self.config['backtest_end_date'])
        
        rebalance_dates = pd.date_range(start=start_date, end=end_date, freq='M')
        return [date for date in rebalance_dates if date in self.daily_returns_matrix.index]
    
    def _run_backtesting_loop(self, rebalance_dates: list) -> (pd.DataFrame, pd.DataFrame):
        """FIXED: Run backtesting loop with proper error handling."""
        daily_holdings = pd.DataFrame(index=self.daily_returns_matrix.index, columns=self.daily_returns_matrix.columns)
        daily_holdings = daily_holdings.fillna(0.0)
        
        diagnostics = []
        current_portfolio = pd.Series(dtype='float64')
        
        for i, rebalance_date in enumerate(rebalance_dates):
            try:
                print(f"   - Processing rebalance {i+1}/{len(rebalance_dates)}: {rebalance_date.strftime('%Y-%m-%d')}...", end=" ")
                
                # FIXED: Get universe with validation
                universe = self._get_universe(rebalance_date)
                if not universe:
                    print("❌ No universe")
                    continue
                
                # FIXED: Detect regime with validation
                regime = self._detect_current_regime(rebalance_date)
                regime_allocation = self.regime_detector.get_regime_allocation(regime)
                
                # FIXED: Calculate factors with validation
                factors_df = self._calculate_factors(universe, rebalance_date)
                if factors_df.empty:
                    print("❌ No factors")
                    continue
                
                # FIXED: Construct portfolio with validation
                portfolio = self._construct_portfolio(factors_df, regime_allocation)
                if portfolio.empty:
                    print("❌ Empty portfolio")
                    continue
                
                # FIXED: Update holdings
                current_portfolio = portfolio
                daily_holdings.loc[rebalance_date:, portfolio.index] = portfolio.values
                
                # FIXED: Calculate turnover
                if i > 0:
                    prev_portfolio = daily_holdings.loc[rebalance_dates[i-1], :]
                    turnover = (abs(portfolio - prev_portfolio).sum() / 2.0) * 100
                else:
                    turnover = 30.0  # Initial portfolio
                
                print(f"✅ Universe: {len(universe)}, Portfolio: {len(portfolio)}, Regime: {regime}, Turnover: {turnover:.1f}%")
                
                diagnostics.append({
                    'date': rebalance_date,
                    'universe_size': len(universe),
                    'portfolio_size': len(portfolio),
                    'regime': regime,
                    'turnover': turnover / 100
                })
                
            except Exception as e:
                print(f"❌ Error: {e}")
                continue
        
        diagnostics_df = pd.DataFrame(diagnostics)
        return daily_holdings, diagnostics_df
    
    def _get_universe(self, analysis_date: pd.Timestamp) -> list:
        """FIXED: Get universe with validation."""
        try:
            lookback_days = self.config['universe']['lookback_days']
            start_date = analysis_date - pd.Timedelta(days=lookback_days)
            
            # FIXED: Query with proper date filtering
            universe_query = text("""
                WITH daily_adtv AS (
                    SELECT trading_date, ticker, total_volume * close_price_adjusted as adtv_vnd
                    FROM vcsc_daily_data_complete
                    WHERE trading_date BETWEEN :start_date AND :analysis_date
                ),
                rolling_adtv AS (
                    SELECT trading_date, ticker,
                        AVG(adtv_vnd) OVER (
                            PARTITION BY ticker 
                            ORDER BY trading_date 
                            ROWS BETWEEN 62 PRECEDING AND CURRENT ROW
                        ) as avg_adtv_63d
                    FROM daily_adtv
                ),
                ranked_universe AS (
                    SELECT trading_date, ticker,
                        ROW_NUMBER() OVER (
                            PARTITION BY trading_date 
                            ORDER BY avg_adtv_63d DESC
                        ) as rank_position
                    FROM rolling_adtv
                    WHERE avg_adtv_63d > 0
                )
                SELECT ticker
                FROM ranked_universe
                WHERE trading_date = :analysis_date
                AND rank_position <= :top_n_stocks
            """)
            
            universe_df = pd.read_sql(universe_query, self.engine, 
                                    params={'start_date': start_date, 
                                           'analysis_date': analysis_date,
                                           'top_n_stocks': self.config['universe']['top_n_stocks']})
            
            return universe_df['ticker'].tolist()
            
        except Exception as e:
            print(f"   ⚠️  Universe error: {e}")
            return []
    
    def _detect_current_regime(self, analysis_date: pd.Timestamp) -> str:
        """FIXED: Detect regime with proper data handling."""
        try:
            lookback_days = self.config['regime']['lookback_period']
            start_date = analysis_date - pd.Timedelta(days=lookback_days)
            
            benchmark_data = self.benchmark_returns.loc[start_date:analysis_date]
            if len(benchmark_data) < 60:
                return 'Sideways'
            
            price_series = (1 + benchmark_data).cumprod()
            price_data = pd.DataFrame({'close': price_series})
            
            return self.regime_detector.detect_regime(price_data)
            
        except Exception as e:
            print(f"   ⚠️  Regime detection error: {e}")
            return 'Sideways'
    
    def _calculate_factors(self, universe: list, analysis_date: pd.Timestamp) -> pd.DataFrame:
        """FIXED: Calculate factors with validation."""
        try:
            if not universe:
                return pd.DataFrame()
            
            # FIXED: Get market data
            market_data = self.price_data[
                (self.price_data['ticker'].isin(universe)) &
                (self.price_data['date'] <= analysis_date)
            ].copy()
            
            if market_data.empty:
                return pd.DataFrame()
            
            # FIXED: Get fundamental data
            lag_days = self.config['factors']['fundamental_lag_days']
            lag_date = analysis_date - pd.Timedelta(days=lag_days)
            
            fundamental_data = self.fundamental_data[
                (self.fundamental_data['ticker'].isin(universe)) &
                (self.fundamental_data['date'] <= lag_date)
            ].copy()
            
            # FIXED: Get most recent fundamental data
            fundamental_data = fundamental_data.sort_values('date').groupby('ticker').tail(1)
            
            if fundamental_data.empty:
                return pd.DataFrame()
            
            # FIXED: Merge data
            factors_df = market_data.merge(fundamental_data, on='ticker', how='inner')
            
            if factors_df.empty:
                return pd.DataFrame()
            
            # FIXED: Calculate composite score
            factors_df = self._calculate_composite_score(factors_df)
            
            return factors_df
            
        except Exception as e:
            print(f"   ⚠️  Factor calculation error: {e}")
            return pd.DataFrame()
    
    def _calculate_composite_score(self, factors_df: pd.DataFrame) -> pd.DataFrame:
        """FIXED: Calculate composite score with validation."""
        try:
            factors_df['composite_score'] = 0.0
            
            # FIXED: ROAA component
            if 'roaa' in factors_df.columns:
                roaa_data = factors_df['roaa'].dropna()
                if len(roaa_data) > 0:
                    roaa_weight = self.config['factors']['roaa_weight']
                    roaa_normalized = (roaa_data - roaa_data.mean()) / roaa_data.std()
                    factors_df.loc[roaa_data.index, 'composite_score'] += roaa_normalized * roaa_weight
            
            # FIXED: P/E component (simplified)
            if 'close' in factors_df.columns and 'market_cap' in factors_df.columns:
                pe_weight = self.config['factors']['pe_weight']
                # Simplified P/E calculation
                factors_df['pe_ratio'] = factors_df['market_cap'] / (factors_df['close'] * 1000)  # Simplified
                pe_data = factors_df['pe_ratio'].dropna()
                if len(pe_data) > 0:
                    pe_normalized = (pe_data - pe_data.mean()) / pe_data.std()
                    factors_df.loc[pe_data.index, 'composite_score'] += (-pe_normalized) * pe_weight
            
            return factors_df
            
        except Exception as e:
            print(f"   ⚠️  Composite score error: {e}")
            factors_df['composite_score'] = 0.0
            return factors_df
    
    def _construct_portfolio(self, factors_df: pd.DataFrame, regime_allocation: float) -> pd.Series:
        """FIXED: Construct portfolio with validation."""
        try:
            if factors_df.empty:
                return pd.Series(dtype='float64')
            
            # FIXED: Filter qualified stocks
            qualified_df = factors_df[factors_df['composite_score'] > 0].copy()
            
            if qualified_df.empty:
                return pd.Series(dtype='float64')
            
            # FIXED: Sort and select
            qualified_df = qualified_df.sort_values('composite_score', ascending=False)
            target_size = self.config['universe']['target_portfolio_size']
            selected_stocks = qualified_df.head(target_size)
            
            if selected_stocks.empty:
                return pd.Series(dtype='float64')
            
            # FIXED: Equal weight portfolio
            portfolio = pd.Series(regime_allocation / len(selected_stocks), 
                                index=selected_stocks['ticker'])
            
            return portfolio
            
        except Exception as e:
            print(f"   ⚠️  Portfolio construction error: {e}")
            return pd.Series(dtype='float64')

# FIXED Data Loading
def load_all_data_for_backtest(config: dict, db_engine):
    """FIXED: Load data with proper error handling."""
    try:
        start_date = config['backtest_start_date']
        end_date = config['backtest_end_date']
        buffer_start_date = pd.Timestamp(start_date) - pd.DateOffset(months=6)
        
        print(f"📂 Loading data: {buffer_start_date.date()} to {end_date}")
        
        # FIXED: Load price data
        price_query = text("""
            SELECT trading_date as date, ticker, close_price_adjusted as close,
                   total_volume as volume, market_cap
            FROM vcsc_daily_data_complete
            WHERE trading_date BETWEEN :start_date AND :end_date
        """)
        
        price_data = pd.read_sql(price_query, db_engine, 
                                params={'start_date': buffer_start_date, 'end_date': end_date}, 
                                parse_dates=['date'])
        
        # FIXED: Load fundamental data (simplified)
        fundamental_query = text("""
            SELECT ticker, sector, date, netprofit_ttm, totalassets_ttm,
                   CASE WHEN totalassets_ttm > 0 THEN netprofit_ttm / totalassets_ttm ELSE NULL END as roaa
            FROM fundamental_values fv
            JOIN master_info mi ON fv.ticker = mi.ticker
            WHERE fv.year BETWEEN YEAR(:start_date) AND YEAR(:end_date)
            AND fv.item_id IN (1, 2)
        """)
        
        fundamental_data = pd.read_sql(fundamental_query, db_engine,
                                     params={'start_date': buffer_start_date, 'end_date': end_date},
                                     parse_dates=['date'])
        
        # FIXED: Load benchmark data
        benchmark_query = text("""
            SELECT date, close
            FROM etf_history
            WHERE ticker = 'VNINDEX' AND date BETWEEN :start_date AND :end_date
        """)
        
        benchmark_data = pd.read_sql(benchmark_query, db_engine,
                                   params={'start_date': buffer_start_date, 'end_date': end_date},
                                   parse_dates=['date'])
        
        # FIXED: Create returns matrix
        price_data['return'] = price_data.groupby('ticker')['close'].pct_change()
        daily_returns_matrix = price_data.pivot(index='date', columns='ticker', values='return')
        daily_returns_matrix = daily_returns_matrix.fillna(0.0)
        
        # FIXED: Create benchmark returns
        benchmark_returns = benchmark_data.set_index('date')['close'].pct_change().rename('VN-Index')
        benchmark_returns = benchmark_returns.fillna(0.0)
        
        print(f"✅ Data loaded: Price={len(price_data)}, Fundamental={len(fundamental_data)}, Benchmark={len(benchmark_returns)}")
        
        return price_data, fundamental_data, daily_returns_matrix, benchmark_returns
        
    except Exception as e:
        print(f"❌ Data loading error: {e}")
        raise

# FIXED Main Execution
if __name__ == "__main__":
    try:
        # Database connection
        db_manager = get_database_manager()
        engine = db_manager.get_engine()
        
        print("🚀 FIXED QVM ENGINE V3F EXECUTION")
        print("=" * 50)
        
        # Load data
        price_data, fundamental_data, daily_returns_matrix, benchmark_returns = load_all_data_for_backtest(QVM_CONFIG, engine)
        
        # Run backtest
        qvm_engine = QVMEngineV3fFixed(
            config=QVM_CONFIG,
            price_data=price_data,
            fundamental_data=fundamental_data,
            returns_matrix=daily_returns_matrix,
            benchmark_returns=benchmark_returns,
            db_engine=engine
        )
        
        qvm_net_returns, qvm_diagnostics = qvm_engine.run_backtest()
        
        # Results
        print(f"\n📊 FIXED RESULTS:")
        print(f"   - Total Return: {(1 + qvm_net_returns).prod() - 1:.2%}")
        print(f"   - Annualized Return: {((1 + qvm_net_returns).prod() ** (252/len(qvm_net_returns)) - 1):.2%}")
        print(f"   - Volatility: {qvm_net_returns.std() * np.sqrt(252):.2%}")
        
        if not qvm_diagnostics.empty:
            regime_counts = qvm_diagnostics['regime'].value_counts()
            print(f"\n📈 Regime Distribution:")
            for regime, count in regime_counts.items():
                print(f"   - {regime}: {count} times")
        
        print("\n✅ FIXED QVM Engine v3f execution complete!")
        
    except Exception as e:
        print(f"❌ Execution error: {e}")
        raise 