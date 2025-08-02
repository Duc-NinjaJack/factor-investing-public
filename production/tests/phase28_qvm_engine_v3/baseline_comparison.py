# ============================================================================
# Phase 28: Baseline Comparison Module
# File: baseline_comparison.py
#
# Objective:
#   To implement the Phase 27 official baseline engine within the Phase 28
#   framework, enabling direct comparison between the baseline and enhanced
#   strategies. This preserves the official baseline v1.0 while allowing
#   validation of Phase 28 improvements.
# ============================================================================

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
from pathlib import Path
import sys

# Database connectivity
from sqlalchemy import create_engine, text

# Import phase28 utilities
try:
    from production.universe.constructors import get_liquid_universe_dataframe
except ImportError:
    # Fallback for testing without full project structure
    def get_liquid_universe_dataframe(date, engine, config):
        """Placeholder function for testing."""
        return pd.DataFrame({'ticker': ['AAA', 'BBB', 'CCC']})

class BaselinePortfolioEngine:
    """
    Phase 27 Official Baseline Engine v1.0
    
    A clean backtesting engine implementing the logic from PortfolioEngine_v3.1.
    This engine includes P0 (turnover) and P1 (hybrid portfolio construction)
    fixes but contains NO risk overlays. It is used to establish the official
    performance baseline of the fully-invested strategy.
    
    This is the immutable reference point against which all Phase 28
    enhancements will be measured.
    """
    
    def __init__(self, config: dict, factor_data: pd.DataFrame, returns_matrix: pd.DataFrame,
                 benchmark_returns: pd.Series, db_engine):
        
        self.config = config
        self.engine = db_engine
        
        # Slice data to the exact backtest window
        start = pd.Timestamp(config['backtest_start_date'])
        end = pd.Timestamp(config['backtest_end_date'])
        
        self.factor_data_raw = factor_data[factor_data['date'].between(start, end)].copy()
        self.daily_returns_matrix = returns_matrix.loc[start:end].copy()
        self.benchmark_returns = benchmark_returns.loc[start:end].copy()
        
        print("✅ BaselinePortfolioEngine (Phase 27) initialized.")
        print(f"   - Strategy: {config['strategy_name']}")
        print(f"   - Period: {self.daily_returns_matrix.index.min().date()} to {self.daily_returns_matrix.index.max().date()}")

    def run_backtest(self) -> (pd.Series, pd.DataFrame):
        """Executes the full backtesting pipeline."""
        print("\n🚀 Starting Phase 27 baseline backtest execution...")
        
        rebalance_dates = self._generate_rebalance_dates()
        daily_holdings, diagnostics = self._run_backtesting_loop(rebalance_dates)
        net_returns = self._calculate_net_returns(daily_holdings)
        
        print("✅ Phase 27 baseline backtest execution complete.")
        return net_returns, diagnostics

    def _generate_rebalance_dates(self) -> list:
        """Generates quarterly rebalance dates based on actual trading days."""
        all_trading_dates = self.daily_returns_matrix.index
        rebal_dates_calendar = pd.date_range(
            start=self.config['backtest_start_date'],
            end=self.config['backtest_end_date'],
            freq=self.config['rebalance_frequency']
        )
        actual_rebal_dates = [all_trading_dates[all_trading_dates.searchsorted(d, side='left')-1] 
                             for d in rebal_dates_calendar if d >= all_trading_dates.min()]
        print(f"   - Generated {len(actual_rebal_dates)} quarterly rebalance dates.")
        return sorted(list(set(actual_rebal_dates)))

    def _run_backtesting_loop(self, rebalance_dates: list) -> (pd.DataFrame, pd.DataFrame):
        """The core loop for portfolio construction at each rebalance date."""
        daily_holdings = pd.DataFrame(0.0, index=self.daily_returns_matrix.index, 
                                    columns=self.daily_returns_matrix.columns)
        diagnostics_log = []
        
        for i, rebal_date in enumerate(rebalance_dates):
            print(f"   - Processing rebalance {i+1}/{len(rebalance_dates)}: {rebal_date.date()}...", end="")
            
            universe_df = get_liquid_universe_dataframe(rebal_date, self.engine, self.config['universe'])
            if universe_df.empty:
                print(" ⚠️ Universe empty. Skipping.")
                continue
            
            factors_on_date = self.factor_data_raw[self.factor_data_raw['date'] == rebal_date]
            liquid_factors = factors_on_date[factors_on_date['ticker'].isin(universe_df['ticker'])].copy()
            
            if len(liquid_factors) < 10:
                print(f" ⚠️ Insufficient stocks ({len(liquid_factors)}). Skipping.")
                continue

            target_portfolio = self._calculate_target_portfolio(liquid_factors)
            if target_portfolio.empty:
                print(" ⚠️ Portfolio empty. Skipping.")
                continue
            
            start_period = rebal_date + pd.Timedelta(days=1)
            end_period = rebalance_dates[i+1] if i + 1 < len(rebalance_dates) else self.daily_returns_matrix.index.max()
            holding_dates = self.daily_returns_matrix.index[
                (self.daily_returns_matrix.index >= start_period) & 
                (self.daily_returns_matrix.index <= end_period)
            ]
            
            daily_holdings.loc[holding_dates] = 0.0
            valid_tickers = target_portfolio.index.intersection(daily_holdings.columns)
            daily_holdings.loc[holding_dates, valid_tickers] = target_portfolio[valid_tickers].values
            
            if i > 0:
                prev_holdings_idx = self.daily_returns_matrix.index.get_loc(rebal_date, method='ffill') - 1
                prev_holdings = daily_holdings.iloc[prev_holdings_idx] if prev_holdings_idx >= 0 else pd.Series(dtype='float64')
            else:
                prev_holdings = pd.Series(dtype='float64')

            # P0 Fix: Corrected turnover calculation
            turnover = (target_portfolio - prev_holdings.reindex(target_portfolio.index).fillna(0)).abs().sum() / 2.0
            
            diagnostics_log.append({
                'date': rebal_date,
                'universe_size': len(universe_df),
                'portfolio_size': len(target_portfolio),
                'turnover': turnover
            })
            print(f" ✅ Universe: {len(universe_df)}, Portfolio: {len(target_portfolio)}, Turnover: {turnover:.1%}")

        return daily_holdings, pd.DataFrame(diagnostics_log).set_index('date')

    def _calculate_target_portfolio(self, factors_df: pd.DataFrame) -> pd.Series:
        """Constructs the portfolio using the hybrid method (P1 Fix)."""
        factors_to_combine = self.config['signal']['factors_to_combine']
        
        # Engineer signals (in this case, none are needed for pure value)
        # Re-normalize and combine
        weighted_scores = []
        for factor, weight in factors_to_combine.items():
            scores = factors_df[factor]
            mean, std = scores.mean(), scores.std()
            if std > 1e-8:  # Z-score safeguard
                weighted_scores.append(((scores - mean) / std) * weight)
        
        if not weighted_scores: 
            return pd.Series(dtype='float64')
        factors_df['final_signal'] = pd.concat(weighted_scores, axis=1).sum(axis=1)
        
        # P1 Fix: Hybrid portfolio construction
        universe_size = len(factors_df)
        if universe_size < 100:
            # Fixed-N for small universes
            portfolio_size = self.config['portfolio']['portfolio_size_small_universe']
            selected_stocks = factors_df.nlargest(portfolio_size, 'final_signal')
        else:
            # Percentile for large universes
            percentile = self.config['portfolio']['selection_percentile']
            score_cutoff = factors_df['final_signal'].quantile(percentile)
            selected_stocks = factors_df[factors_df['final_signal'] >= score_cutoff]
            
        if selected_stocks.empty: 
            return pd.Series(dtype='float64')
        return pd.Series(1.0 / len(selected_stocks), index=selected_stocks['ticker'])

    def _calculate_net_returns(self, daily_holdings: pd.DataFrame) -> pd.Series:
        """Calculates net returns with P0 turnover fix."""
        holdings_shifted = daily_holdings.shift(1).fillna(0.0)
        gross_returns = (holdings_shifted * self.daily_returns_matrix).sum(axis=1)
        
        # P0 Fix: Corrected turnover calculation
        turnover = (holdings_shifted - holdings_shifted.shift(1)).abs().sum(axis=1) / 2.0
        
        costs = turnover * (self.config['transaction_cost_bps'] / 10000)
        net_returns = (gross_returns - costs).rename(self.config['strategy_name'])
        
        print("\n💸 Net returns calculated.")
        print(f"   - Total Gross Return: {(1 + gross_returns).prod() - 1:.2%}")
        print(f"   - Total Net Return: {(1 + net_returns).prod() - 1:.2%}")
        print(f"   - Total Cost Drag: {gross_returns.sum() - net_returns.sum():.2%}")
        
        return net_returns


class BaselineComparisonFramework:
    """
    Framework for comparing Phase 27 baseline with Phase 28 enhanced strategies.
    """
    
    def __init__(self, baseline_config: dict, enhanced_config: dict, db_engine):
        self.baseline_config = baseline_config
        self.enhanced_config = enhanced_config
        self.db_engine = db_engine
        
    def load_baseline_data(self, start_date: str, end_date: str):
        """Load data for baseline strategy (using factor_scores_qvm table)."""
        print("📂 Loading baseline data (factor_scores_qvm)...")
        
        # Add buffer for rolling calculations
        buffer_start_date = pd.Timestamp(start_date) - pd.DateOffset(months=3)
        
        db_params = {
            'start_date': buffer_start_date,
            'end_date': pd.Timestamp(end_date),
            'strategy_version': self.baseline_config['signal']['db_strategy_version']
        }

        # Factor Scores
        factor_query = text("""
            SELECT date, ticker, Quality_Composite, Value_Composite, Momentum_Composite
            FROM factor_scores_qvm
            WHERE date BETWEEN :start_date AND :end_date 
              AND strategy_version = :strategy_version
        """)
        factor_data = pd.read_sql(factor_query, self.db_engine, params=db_params, parse_dates=['date'])
        
        # Price Data
        price_query = text("""
            SELECT date, ticker, close 
            FROM equity_history
            WHERE date BETWEEN :start_date AND :end_date
        """)
        price_data = pd.read_sql(price_query, self.db_engine, params=db_params, parse_dates=['date'])
        
        # Benchmark Data
        benchmark_query = text("""
            SELECT date, close
            FROM etf_history
            WHERE ticker = 'VNINDEX' AND date BETWEEN :start_date AND :end_date
        """)
        benchmark_data = pd.read_sql(benchmark_query, self.db_engine, params=db_params, parse_dates=['date'])
        
        # Prepare data structures
        price_data['return'] = price_data.groupby('ticker')['close'].pct_change()
        daily_returns_matrix = price_data.pivot(index='date', columns='ticker', values='return')
        benchmark_returns = benchmark_data.set_index('date')['close'].pct_change().rename('VN-Index')
        
        return factor_data, daily_returns_matrix, benchmark_returns
    
    def run_comparison(self, start_date: str, end_date: str, enhanced_engine):
        """
        Run side-by-side comparison between baseline and enhanced strategies.
        
        Args:
            start_date: Backtest start date
            end_date: Backtest end date
            enhanced_engine: Phase 28 enhanced engine instance
        """
        print("\n" + "="*80)
        print("🔄 PHASE 27 vs PHASE 28 STRATEGY COMPARISON")
        print("="*80)
        
        # Load baseline data
        baseline_factor_data, daily_returns_matrix, benchmark_returns = self.load_baseline_data(start_date, end_date)
        
        # Run baseline strategy
        print("\n📊 Running Phase 27 Baseline Strategy...")
        baseline_engine = BaselinePortfolioEngine(
            config=self.baseline_config,
            factor_data=baseline_factor_data,
            returns_matrix=daily_returns_matrix,
            benchmark_returns=benchmark_returns,
            db_engine=self.db_engine
        )
        baseline_returns, baseline_diagnostics = baseline_engine.run_backtest()
        
        # Run enhanced strategy
        print("\n📊 Running Phase 28 Enhanced Strategy...")
        enhanced_returns, enhanced_diagnostics = enhanced_engine.run_backtest()
        
        # Generate comparison report
        self._generate_comparison_report(baseline_returns, enhanced_returns, 
                                       baseline_diagnostics, enhanced_diagnostics,
                                       benchmark_returns)
        
        return baseline_returns, enhanced_returns, baseline_diagnostics, enhanced_diagnostics
    
    def _generate_comparison_report(self, baseline_returns, enhanced_returns, 
                                  baseline_diagnostics, enhanced_diagnostics, benchmark_returns):
        """Generate comprehensive comparison report."""
        print("\n" + "="*80)
        print("📈 COMPARISON REPORT: BASELINE vs ENHANCED")
        print("="*80)
        
        # Calculate metrics
        baseline_metrics = self._calculate_metrics(baseline_returns, benchmark_returns)
        enhanced_metrics = self._calculate_metrics(enhanced_returns, benchmark_returns)
        
        # Display comparison table
        comparison_data = []
        for metric in baseline_metrics.keys():
            baseline_val = baseline_metrics[metric]
            enhanced_val = enhanced_metrics[metric]
            improvement = enhanced_val - baseline_val
            improvement_pct = (improvement / baseline_val * 100) if baseline_val != 0 else 0
            
            comparison_data.append({
                'Metric': metric,
                'Baseline': f"{baseline_val:.2f}",
                'Enhanced': f"{enhanced_val:.2f}",
                'Improvement': f"{improvement:+.2f}",
                'Improvement %': f"{improvement_pct:+.1f}%"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print("\n📊 Performance Comparison:")
        print(comparison_df.to_string(index=False))
        
        # Summary
        print(f"\n🎯 Summary:")
        print(f"   - Baseline Annual Return: {baseline_metrics['Annualized Return (%)']:.2f}%")
        print(f"   - Enhanced Annual Return: {enhanced_metrics['Annualized Return (%)']:.2f}%")
        print(f"   - Improvement: {enhanced_metrics['Annualized Return (%)'] - baseline_metrics['Annualized Return (%)']:+.2f}%")
        print(f"   - Baseline Sharpe: {baseline_metrics['Sharpe Ratio']:.2f}")
        print(f"   - Enhanced Sharpe: {enhanced_metrics['Sharpe Ratio']:.2f}")
        print(f"   - Sharpe Improvement: {enhanced_metrics['Sharpe Ratio'] - baseline_metrics['Sharpe Ratio']:+.2f}")
    
    def _calculate_metrics(self, returns: pd.Series, benchmark: pd.Series, periods_per_year: int = 252) -> dict:
        """Calculate performance metrics with corrected benchmark alignment."""
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


# Default baseline configuration (Phase 27 Official Baseline v1.0)
DEFAULT_BASELINE_CONFIG = {
    'strategy_name': 'Official_Baseline_v1.0_Value',
    'backtest_start_date': '2016-03-01',
    'backtest_end_date': '2025-07-28',
    'rebalance_frequency': 'Q',
    'transaction_cost_bps': 20,
    'universe': {
        'min_adtv_vnd': 10_000_000_000,  # 10B VND
        'lookback_days': 63,
        'target_size': 200
    },
    'signal': {
        'db_strategy_version': 'qvm_v2.0_enhanced',
        'factors_to_combine': {
            'Value_Composite': 1.0
        }
    },
    'portfolio': {
        'portfolio_size_small_universe': 20,
        'selection_percentile': 0.2
    }
} 