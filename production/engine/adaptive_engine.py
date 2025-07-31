# production/engine/adaptive_engine.py

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from scipy.optimize import minimize
import yaml
from pathlib import Path
import sys

# Ensure the project root is in the path to find other modules
try:
    project_root = Path(__file__).resolve().parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from production.universe.constructors import get_liquid_universe_dataframe
except (ImportError, NameError):
    print("Warning: Could not import production.universe.constructors. Ensure path is correct.")
    def get_liquid_universe_dataframe(*args, **kwargs):
        return pd.DataFrame()

# This function MUST be at the top level of the module for multiprocessing
def run_optimization_for_period(period_data):
    train_start, train_end, config, factor_data, returns_matrix, benchmark_returns, db_engine_config = period_data
    
    temp_engine = create_engine(f"mysql+pymysql://{db_engine_config['username']}:{db_engine_config['password']}@{db_engine_config['host']}/{db_engine_config['schema_name']}")
    
    factors_to_opt = config['walk_forward']['factors_to_optimize']
    
    class OptimizerBacktester:
        def __init__(self, cfg, fact_data, ret_matrix, db_eng):
            self.config = cfg; self.engine = db_eng
            start = pd.Timestamp(cfg['backtest_start_date']); end = pd.Timestamp(cfg['backtest_end_date'])
            self.factor_data_raw = fact_data[fact_data['date'].between(start, end)].copy()
            self.daily_returns_matrix = ret_matrix.loc[start:end].copy()
            self._universe_cache = {}

        def run(self):
            rebalance_dates = self._generate_rebalance_dates()
            daily_holdings = pd.DataFrame(0.0, index=self.daily_returns_matrix.index, columns=self.daily_returns_matrix.columns)
            for date in rebalance_dates:
                target = self._get_target(date)
                if not target.empty:
                    start_p = date + pd.Timedelta(days=1)
                    end_p = rebalance_dates[rebalance_dates.index(date)+1] if rebalance_dates.index(date)+1 < len(rebalance_dates) else self.daily_returns_matrix.index.max()
                    h_dates = self.daily_returns_matrix.index[(self.daily_returns_matrix.index >= start_p) & (self.daily_returns_matrix.index <= end_p)]
                    valid_tickers = target.index.intersection(daily_holdings.columns)
                    daily_holdings.loc[h_dates, valid_tickers] = target[valid_tickers].values
            h_shifted = daily_holdings.shift(1).fillna(0.0)
            gross = (h_shifted * self.daily_returns_matrix).sum(axis=1)
            turnover = (h_shifted - h_shifted.shift(1)).abs().sum(axis=1) / 2.0
            costs = turnover * (self.config['transaction_cost_bps'] / 10000)
            return gross - costs

        def _generate_rebalance_dates(self):
            all_dates = self.daily_returns_matrix.index
            cal_dates = pd.date_range(start=self.config['backtest_start_date'], end=self.config['backtest_end_date'], freq=self.config['rebalance_frequency'])
            return sorted(list(set([all_dates[all_dates.searchsorted(d, side='left')-1] for d in cal_dates if d >= all_dates.min()])))

        def _get_universe(self, date):
            if date in self._universe_cache: return self._universe_cache[date]
            df = get_liquid_universe_dataframe(date, self.engine, self.config['universe'])
            self._universe_cache[date] = df
            return df

        def _get_target(self, date):
            universe = self._get_universe(date)
            if universe.empty: return pd.Series(dtype='float64')
            factors = self.factor_data_raw[self.factor_data_raw['date'] == date]
            liquid = factors[factors['ticker'].isin(universe['ticker'])].copy()
            if len(liquid) < 10: return pd.Series(dtype='float64')
            liquid['Momentum_Reversal'] = -1 * liquid['Momentum_Composite']
            weights = self.config['signal']['factors_to_combine']
            scores = []
            for factor, weight in weights.items():
                if factor in liquid.columns and weight > 0:
                    s = liquid[factor]; m, s_std = s.mean(), s.std()
                    if s_std > 1e-8: scores.append(((s - m) / s_std) * weight)
            if not scores: return pd.Series(dtype='float64')
            liquid['final_signal'] = pd.concat(scores, axis=1).sum(axis=1)
            selected = liquid.nlargest(self.config['portfolio']['portfolio_size'], 'final_signal')
            if selected.empty: return pd.Series(dtype='float64')
            return pd.Series(1.0 / len(selected), index=selected['ticker'])

    def objective_function(weights):
        temp_config = config.copy()
        temp_config['backtest_start_date'] = train_start.strftime('%Y-%m-%d')
        temp_config['backtest_end_date'] = train_end.strftime('%Y-%m-%d')
        temp_config['signal'] = {'factors_to_combine': dict(zip(factors_to_opt, weights))}
        engine = OptimizerBacktester(temp_config, factor_data, returns_matrix, temp_engine)
        returns = engine.run()
        sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        return -sharpe

    initial_weights = [1.0 / len(factors_to_opt)] * len(factors_to_opt)
    bounds = [config['walk_forward']['bounds'][f] for f in factors_to_opt]
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    result = minimize(objective_function, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 200})
    
    temp_engine.dispose()
    return dict(zip(factors_to_opt, result.x if result.success else [0.25]*4))

class PortfolioEngine_v2_0:
    def __init__(self, config: dict, factor_data: pd.DataFrame, returns_matrix: pd.DataFrame,
                 benchmark_returns: pd.Series, db_engine, universe_cache: dict = None):
        self.config = config; self.engine = db_engine
        start = pd.Timestamp(config['backtest_start_date']); end = pd.Timestamp(config['backtest_end_date'])
        self.factor_data_raw = factor_data
        self.daily_returns_matrix = returns_matrix.loc[start:end].copy()
        self.benchmark_returns = benchmark_returns.loc[start:end].copy()
        self._universe_cache = universe_cache if universe_cache is not None else {}

    def run_backtest(self, mode='iterative') -> (pd.Series, pd.DataFrame, pd.Series):
        rebalance_dates = self._generate_rebalance_dates()
        if mode == 'vectorized':
            daily_holdings, _ = self._run_vectorized_loop(rebalance_dates)
            net_returns = self._calculate_net_returns(daily_holdings)
            return net_returns, pd.DataFrame(), pd.Series()
        else:
            return self._run_daily_iterative_loop(rebalance_dates)

    def _run_vectorized_loop(self, rebalance_dates: list) -> (pd.DataFrame, pd.DataFrame):
        daily_holdings = pd.DataFrame(0.0, index=self.daily_returns_matrix.index, columns=self.daily_returns_matrix.columns)
        for i, rebal_date in enumerate(rebalance_dates):
            target_portfolio = self._get_target_for_date(rebal_date)
            if target_portfolio.empty: continue
            start_period = rebal_date + pd.Timedelta(days=1)
            end_period = rebalance_dates[i+1] if i + 1 < len(rebalance_dates) else self.daily_returns_matrix.index.max()
            holding_dates = self.daily_returns_matrix.index[(self.daily_returns_matrix.index >= start_period) & (self.daily_returns_matrix.index <= end_period)]
            valid_tickers = target_portfolio.index.intersection(daily_holdings.columns)
            daily_holdings.loc[holding_dates, valid_tickers] = target_portfolio[valid_tickers].values
        return daily_holdings, pd.DataFrame()

    def _run_daily_iterative_loop(self, rebalance_dates: list):
        trading_days = self.daily_returns_matrix.index
        target_holdings = pd.Series(dtype='float64'); net_returns = pd.Series(0.0, index=trading_days)
        daily_exposure = pd.Series(1.0, index=trading_days); diagnostics_log = []
        equity_curve = pd.Series(1.0, index=trading_days.insert(0, trading_days[0] - pd.Timedelta(days=1)))
        for i in range(len(trading_days)):
            today = trading_days[i]; yesterday = trading_days[i-1] if i > 0 else today
            if today in rebalance_dates:
                new_target_holdings, diagnostics = self._get_target_for_date(today, get_diagnostics=True)
                if not new_target_holdings.empty:
                    costs = self._calculate_transaction_costs(target_holdings, new_target_holdings)
                    net_returns.loc[today] -= costs
                    diagnostics['turnover'] = costs / (self.config['transaction_cost_bps'] / 10000) if self.config['transaction_cost_bps'] > 0 else 0
                    diagnostics_log.append(diagnostics)
                    target_holdings = new_target_holdings
            prev_holdings = target_holdings.reindex(self.daily_returns_matrix.columns).fillna(0)
            gross_return_today = (prev_holdings * self.daily_returns_matrix.loc[today].fillna(0)).sum()
            exposure_today = self._calculate_daily_exposure(today, gross_return_today, equity_curve.loc[:yesterday])
            daily_exposure.loc[today] = exposure_today
            net_returns.loc[today] += gross_return_today * exposure_today
            equity_curve.loc[today] = equity_curve.loc[yesterday] * (1 + net_returns.loc[today])
        return net_returns, pd.DataFrame(diagnostics_log).set_index('date'), daily_exposure

    def _get_target_for_date(self, date, get_diagnostics=False):
        universe_df = self._get_universe(date)
        if universe_df.empty: return (pd.Series(dtype='float64'), {}) if get_diagnostics else pd.Series(dtype='float64')
        factors_on_date = self.factor_data_raw[self.factor_data_raw['date'] == date]
        liquid_factors = factors_on_date[factors_on_date['ticker'].isin(universe_df['ticker'])].copy()
        if len(liquid_factors) < 10: return (pd.Series(dtype='float64'), {}) if get_diagnostics else pd.Series(dtype='float64')
        liquid_factors['Momentum_Reversal'] = -1 * liquid_factors['Momentum_Composite']
        factor_weights = self.config['signal']['factors_to_combine']
        weighted_scores = []
        for factor, weight in factor_weights.items():
            if factor in liquid_factors.columns and weight > 0:
                scores = liquid_factors[factor]; mean, std = scores.mean(), scores.std()
                if std > 1e-8: weighted_scores.append(((scores - mean) / std) * weight)
        if not weighted_scores: return (pd.Series(dtype='float64'), {}) if get_diagnostics else pd.Series(dtype='float64')
        liquid_factors['final_signal'] = pd.concat(weighted_scores, axis=1).sum(axis=1)
        selected_stocks = liquid_factors.nlargest(self.config['portfolio']['portfolio_size'], 'final_signal')
        if selected_stocks.empty: return (pd.Series(dtype='float64'), {}) if get_diagnostics else pd.Series(dtype='float64')
        target_portfolio = pd.Series(1.0 / len(selected_stocks), index=selected_stocks['ticker'])
        diagnostics = {'date': date, 'universe_size': len(universe_df), 'portfolio_size': len(target_portfolio)}
        return (target_portfolio, diagnostics) if get_diagnostics else target_portfolio

    def _get_universe(self, date):
        if date in self._universe_cache: return self._universe_cache[date]
        df = get_liquid_universe_dataframe(date, self.engine, self.config['universe'])
        self._universe_cache[date] = df
        return df

    def _calculate_transaction_costs(self, prev_holdings: pd.Series, next_holdings: pd.Series) -> float:
        turnover = (next_holdings.reindex(prev_holdings.index).fillna(0) - prev_holdings.reindex(next_holdings.index).fillna(0)).abs().sum() / 2.0
        return turnover * (self.config['transaction_cost_bps'] / 10000)

    def _calculate_daily_exposure(self, today: pd.Timestamp, gross_return_today: float, equity_curve: pd.Series) -> float:
        overlay_cfg = self.config['risk_overlay']
        temp_equity_curve = pd.concat([equity_curve, pd.Series({today: equity_curve.iloc[-1] * (1 + gross_return_today)})])
        recent_returns = temp_equity_curve.pct_change().dropna()
        if len(recent_returns) < 63: vol_exposure = 1.0
        else:
            realized_vol = recent_returns.tail(63).std() * np.sqrt(252)
            vol_exposure = (overlay_cfg['volatility_target'] / realized_vol) if realized_vol > 0 else 1.5
        vol_exposure = np.clip(vol_exposure, 0.3, 1.5)
        vn_index_vol = self.benchmark_returns.loc[:today].tail(63).std() * np.sqrt(252)
        vn_index_cum = (1 + self.benchmark_returns.loc[:today]).cumprod()
        vn_index_dd = (vn_index_cum.iloc[-1] / vn_index_cum.tail(63).max()) - 1
        is_stressed = (vn_index_vol > 0.25) or (vn_index_dd < overlay_cfg['regime_dd_threshold'])
        regime_exposure = 0.5 if is_stressed else 1.0
        strategy_drawdown = (equity_curve.iloc[-1] / equity_curve.cummax().iloc[-1]) - 1
        stop_loss_exposure = 1.0
        if strategy_drawdown < overlay_cfg['stop_loss_threshold']: stop_loss_exposure = overlay_cfg['de_risk_level']
        return min(vol_exposure, regime_exposure, stop_loss_exposure)

    def _generate_rebalance_dates(self) -> list:
        all_trading_dates = self.daily_returns_matrix.index
        rebal_dates_calendar = pd.date_range(start=self.config['backtest_start_date'], end=self.config['backtest_end_date'], freq=self.config['rebalance_frequency'])
        return sorted(list(set([all_trading_dates[all_trading_dates.searchsorted(d, side='left')-1] for d in rebal_dates_calendar if d >= all_trading_dates.min()])))

    def _calculate_net_returns(self, daily_holdings: pd.DataFrame) -> pd.Series:
        holdings_shifted = daily_holdings.shift(1).fillna(0.0)
        gross_returns = (holdings_shifted * self.daily_returns_matrix).sum(axis=1)
        turnover = (holdings_shifted - holdings_shifted.shift(1)).abs().sum(axis=1) / 2.0
        costs = turnover * (self.config['transaction_cost_bps'] / 10000)
        return gross_returns - costs