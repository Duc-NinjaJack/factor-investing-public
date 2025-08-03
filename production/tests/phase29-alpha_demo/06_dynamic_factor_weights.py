# %% [markdown]
# # QVM Engine v3j - Dynamic Factor Weights Strategy
# 
# **Objective:** Enhanced integrated strategy with regime-specific dynamic factor weights.
# This strategy adapts factor exposure based on market regime for optimal risk-adjusted returns.
# 
# **File:** 06_dynamic_factor_weights.py
# 
# **Enhancement:** Regime-specific factor weighting based on factor testing results

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path
import sys
import warnings
import os

warnings.filterwarnings('ignore')

# Add project root to path for Jupyter notebook environment
notebook_dir = Path(os.getcwd())
project_root = notebook_dir.parent.parent.parent  # Go up to factor-investing-public
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(notebook_dir))

# %%
# Import shared components
from components.base_engine import BaseEngine
from components.regime_detector import RegimeDetector
from components.factor_calculator import SectorAwareFactorCalculator

# %% [markdown]
# # CONFIGURATION SETUP

# %%
# Enhanced configuration with dynamic factor weights
QVM_CONFIG = {
    "strategy_name": "QVM_Engine_v3j_Dynamic_Factor_Weights",
    "description": "Enhanced strategy with regime-specific dynamic factor weights",
    "universe": {
        "top_n_stocks": 200,
        "target_portfolio_size": 20,
        "min_universe_size": 5,
    },
    "rebalancing": {
        "frequency": "monthly",
        "skip_months": 1,
    },
    "transaction_costs": {
        "commission": 0.003,  # 30 bps
    },
    "regime_detection": {
        "volatility_threshold": 0.20,
        "correlation_threshold": 0.70,
        "momentum_threshold": 0.05,
        "stress_threshold": 0.30,
    },
    "factors": {
        "momentum_horizons": [21, 63, 126, 252],  # 1M, 3M, 6M, 12M
        "skip_months": 1,
        "fundamental_lag_days": 45,  # 45-day lag for announcement delay
    },
    "dynamic_weights": {
        "bull_market": {
            "roaa_weight": 0.25,
            "pe_weight": 0.20,
            "momentum_weight": 0.45,
            "low_vol_weight": 0.10,
        },
        "bear_market": {
            "roaa_weight": 0.30,
            "pe_weight": 0.25,
            "momentum_weight": 0.15,
            "low_vol_weight": 0.30,
        },
        "sideways_market": {
            "roaa_weight": 0.30,
            "pe_weight": 0.30,
            "momentum_weight": 0.25,
            "low_vol_weight": 0.15,
        },
        "stress_market": {
            "roaa_weight": 0.25,
            "pe_weight": 0.20,
            "momentum_weight": 0.10,
            "low_vol_weight": 0.45,
        }
    }
}

print("🚀 QVM Engine v3j Dynamic Factor Weights Strategy")
print(f"   - Strategy: {QVM_CONFIG['strategy_name']}")
print(f"   - Description: {QVM_CONFIG['description']}")
print("   - Enhancement: Regime-specific dynamic factor weights")
print("   - Factor Adaptation: Bull/Bear/Sideways/Stress market regimes")

# %% [markdown]
# # STRATEGY CLASS DEFINITION

# %%
class QVMEngineV3jDynamicWeights:
    """
    QVM Engine v3j Dynamic Factor Weights Strategy.
    Enhanced strategy with regime-specific dynamic factor weights based on market conditions.
    """
    
    def __init__(self, config: dict, price_data: pd.DataFrame, fundamental_data: pd.DataFrame,
                 returns_matrix: pd.DataFrame, benchmark_returns: pd.Series, db_engine, precomputed_data: dict):
        """
        Initialize the dynamic factor weights strategy.
        
        Args:
            config: Strategy configuration
            price_data: Historical price data
            fundamental_data: Fundamental data
            returns_matrix: Daily returns matrix
            benchmark_returns: Benchmark returns series
            db_engine: Database engine
            precomputed_data: Pre-computed data dictionary
        """
        self.config = config
        self.price_data = price_data
        self.fundamental_data = fundamental_data
        self.daily_returns_matrix = returns_matrix
        self.benchmark_returns = benchmark_returns
        self.db_engine = db_engine
        self.precomputed_data = precomputed_data
        
        # Initialize components
        self.regime_detector = RegimeDetector(config['regime_detection'])
        self.sector_calculator = SectorAwareFactorCalculator(db_engine)
        self.base_engine = BaseEngine(config, db_engine)
        
        # Strategy parameters
        self.target_portfolio_size = config['universe']['target_portfolio_size']
        self.transaction_cost = config['transaction_costs']['commission']
        
        print("✅ QVMEngineV3jDynamicWeights initialized.")
        print(f"   - Dynamic weights: Regime-specific factor adaptation")
        print(f"   - Bull weights: {config['dynamic_weights']['bull_market']}")
        print(f"   - Bear weights: {config['dynamic_weights']['bear_market']}")
        print(f"   - Sideways weights: {config['dynamic_weights']['sideways_market']}")
        print(f"   - Stress weights: {config['dynamic_weights']['stress_market']}")

    def run_backtest(self) -> (pd.Series, pd.DataFrame):
        """Executes the dynamic factor weights strategy backtesting pipeline."""
        print("\n🚀 Starting QVM Engine v3j dynamic factor weights strategy backtest execution...")
        rebalance_dates = self._generate_rebalance_dates()
        daily_holdings, diagnostics = self._run_dynamic_weights_backtesting_loop(rebalance_dates)
        net_returns = self._calculate_net_returns(daily_holdings)
        print("✅ QVM Engine v3j dynamic factor weights strategy backtest execution complete.")
        return net_returns, diagnostics

    def _generate_rebalance_dates(self) -> list:
        """Generate rebalancing dates."""
        start_date = self.daily_returns_matrix.index[0]
        end_date = self.daily_returns_matrix.index[-1]
        
        rebalance_dates = []
        current_date = start_date
        
        while current_date <= end_date:
            rebalance_dates.append(current_date)
            # Move to next month
            if current_date.month == 12:
                current_date = current_date.replace(year=current_date.year + 1, month=1)
            else:
                current_date = current_date.replace(month=current_date.month + 1)
        
        return [date for date in rebalance_dates if date <= end_date]

    def _run_dynamic_weights_backtesting_loop(self, rebalance_dates: list) -> (pd.DataFrame, pd.DataFrame):
        """Dynamic factor weights backtesting loop with regime-specific weight adaptation."""
        daily_holdings = pd.DataFrame(0.0, index=self.daily_returns_matrix.index, columns=self.daily_returns_matrix.columns)
        diagnostics_log = []
        
        previous_portfolio = pd.Series(0.0, index=self.daily_returns_matrix.columns)
        
        for i, rebal_date in enumerate(rebalance_dates):
            print(f"   - Processing rebalance {i+1}/{len(rebalance_dates)}: {rebal_date.date()}...", end="")
            
            # Get universe and regime
            universe = self._get_universe_from_precomputed(rebal_date)
            if len(universe) < 5:
                print(" ⚠️ Universe too small. Skipping.")
                continue
            
            # Detect market regime
            regime = self._detect_regime_at_date(rebal_date)
            regime_allocation = self.regime_detector.get_regime_allocation(regime)
            
            # Get dynamic factor weights based on regime
            factor_weights = self._get_dynamic_factor_weights(regime)
            
            # Get factor data
            factors_df = self._get_factors_from_precomputed(universe, rebal_date)
            if factors_df.empty:
                print(" ⚠️ No factor data. Skipping.")
                continue
            
            # Apply dynamic factor weights
            qualified_df = self._apply_dynamic_factor_weights(factors_df, factor_weights)
            if qualified_df.empty:
                print(" ⚠️ No qualified stocks. Skipping.")
                continue
            
            # Construct portfolio
            target_portfolio = self._construct_dynamic_portfolio(qualified_df, regime_allocation)
            if target_portfolio.empty:
                print(" ⚠️ Portfolio empty. Skipping.")
                continue
            
            # Apply holdings and calculate turnover
            turnover = self._apply_holdings_and_calculate_turnover(
                daily_holdings, target_portfolio, rebal_date, previous_portfolio
            )
            previous_portfolio = target_portfolio.copy()
            
            # Log diagnostics
            diagnostics_log.append({
                'date': rebal_date,
                'universe_size': len(universe),
                'portfolio_size': len(target_portfolio),
                'regime': regime,
                'regime_allocation': regime_allocation,
                'turnover': turnover,
                'roaa_weight': factor_weights['roaa_weight'],
                'pe_weight': factor_weights['pe_weight'],
                'momentum_weight': factor_weights['momentum_weight'],
                'low_vol_weight': factor_weights['low_vol_weight']
            })
            
            print(f" ✅ Universe: {len(universe)}, Portfolio: {len(target_portfolio)}, Regime: {regime}, Turnover: {turnover:.2%}")
            print(f"    Weights: ROAA={factor_weights['roaa_weight']:.2f}, P/E={factor_weights['pe_weight']:.2f}, "
                  f"Momentum={factor_weights['momentum_weight']:.2f}, Low-Vol={factor_weights['low_vol_weight']:.2f}")
        
        if diagnostics_log:
            return daily_holdings, pd.DataFrame(diagnostics_log).set_index('date')
        else:
            return daily_holdings, pd.DataFrame()

    def _detect_regime_at_date(self, analysis_date: pd.Timestamp) -> str:
        """Detect market regime at a specific date."""
        # Use a rolling window for regime detection
        lookback_days = 252  # 1 year
        start_date = analysis_date - timedelta(days=lookback_days)
        
        # Get market data for regime detection
        market_data = self._get_market_data_for_regime_detection(start_date, analysis_date)
        if market_data.empty:
            return 'Sideways'  # Default regime
        
        return self.regime_detector.detect_regime(market_data)

    def _get_market_data_for_regime_detection(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        """Get market data for regime detection."""
        try:
            # Use benchmark returns for regime detection
            market_returns = self.benchmark_returns[start_date:end_date]
            
            # Calculate regime detection metrics
            volatility = market_returns.std() * np.sqrt(252)
            momentum = market_returns.mean() * 252
            correlation = market_returns.rolling(63).corr(market_returns.shift(1)).mean()
            
            return pd.DataFrame({
                'volatility': [volatility],
                'momentum': [momentum],
                'correlation': [correlation]
            })
        except Exception as e:
            print(f"Warning: Error in regime detection: {e}")
            return pd.DataFrame()

    def _get_dynamic_factor_weights(self, regime: str) -> dict:
        """Get dynamic factor weights based on market regime."""
        regime_weights = self.config['dynamic_weights']
        
        if regime == 'Bull':
            return regime_weights['bull_market']
        elif regime == 'Bear':
            return regime_weights['bear_market']
        elif regime == 'Sideways':
            return regime_weights['sideways_market']
        elif regime == 'Stress':
            return regime_weights['stress_market']
        else:
            return regime_weights['sideways_market']  # Default

    def _get_universe_from_precomputed(self, analysis_date: pd.Timestamp) -> list:
        """Get universe from pre-computed data."""
        try:
            universe_data = self.precomputed_data['universe_rankings']
            if analysis_date in universe_data.index:
                return universe_data.loc[analysis_date].dropna().tolist()
            return []
        except Exception as e:
            print(f"Warning: Error getting universe: {e}")
            return []

    def _get_factors_from_precomputed(self, universe: list, analysis_date: pd.Timestamp) -> pd.DataFrame:
        """Get factors from pre-computed data with enhanced factor set."""
        try:
            # Get fundamental factors
            fundamental_factors = self.precomputed_data['fundamental_factors']
            momentum_factors = self.precomputed_data['momentum_factors']
            
            # Filter for analysis date and universe
            fundamental_data = fundamental_factors[fundamental_factors.index == analysis_date]
            momentum_data = momentum_factors[momentum_factors.index == analysis_date]
            
            if fundamental_data.empty or momentum_data.empty:
                return pd.DataFrame()
            
            # Combine factors
            combined_factors = pd.DataFrame()
            
            for ticker in universe:
                if ticker in fundamental_data.columns and ticker in momentum_data.columns:
                    # Get fundamental factors
                    roaa = fundamental_data.loc[analysis_date, f"{ticker}_roaa"]
                    pe_ratio = fundamental_data.loc[analysis_date, f"{ticker}_pe_ratio"]
                    
                    # Get momentum factors
                    momentum_score = momentum_data.loc[analysis_date, f"{ticker}_momentum_score"]
                    
                    # Calculate low-volatility factor (inverse of volatility)
                    if ticker in self.price_data.columns:
                        price_series = self.price_data[ticker].dropna()
                        if len(price_series) >= 252:
                            volatility = price_series.rolling(252).std().iloc[-1] * np.sqrt(252)
                            low_vol_score = 1 / (volatility + 1e-6)  # Avoid division by zero
                        else:
                            low_vol_score = 0.0
                    else:
                        low_vol_score = 0.0
                    
                    # Create factor row
                    factor_row = pd.DataFrame({
                        'ticker': [ticker],
                        'roaa': [roaa],
                        'pe_ratio': [pe_ratio],
                        'momentum_score': [momentum_score],
                        'low_vol_score': [low_vol_score]
                    })
                    
                    combined_factors = pd.concat([combined_factors, factor_row], ignore_index=True)
            
            return combined_factors
            
        except Exception as e:
            print(f"Warning: Error getting factors: {e}")
            return pd.DataFrame()

    def _apply_dynamic_factor_weights(self, factors_df: pd.DataFrame, factor_weights: dict) -> pd.DataFrame:
        """Apply dynamic factor weights to calculate composite scores."""
        try:
            # Normalize factors
            factors_df['roaa_normalized'] = self._normalize_factor(factors_df['roaa'])
            factors_df['pe_normalized'] = self._normalize_factor(-factors_df['pe_ratio'])  # Lower P/E is better
            factors_df['momentum_normalized'] = self._normalize_factor(factors_df['momentum_score'])
            factors_df['low_vol_normalized'] = self._normalize_factor(factors_df['low_vol_score'])
            
            # Calculate weighted composite score
            factors_df['composite_score'] = (
                factor_weights['roaa_weight'] * factors_df['roaa_normalized'] +
                factor_weights['pe_weight'] * factors_df['pe_normalized'] +
                factor_weights['momentum_weight'] * factors_df['momentum_normalized'] +
                factor_weights['low_vol_weight'] * factors_df['low_vol_normalized']
            )
            
            # Apply entry criteria
            qualified_df = factors_df[
                (factors_df['roaa'] > 0) &  # Positive ROAA
                (factors_df['pe_ratio'] > 0) & (factors_df['pe_ratio'] < 50) &  # Reasonable P/E
                (factors_df['momentum_score'] > 0) &  # Positive momentum
                (factors_df['low_vol_score'] > 0)  # Valid low-vol score
            ].copy()
            
            return qualified_df
            
        except Exception as e:
            print(f"Warning: Error applying dynamic weights: {e}")
            return pd.DataFrame()

    def _normalize_factor(self, factor_series: pd.Series) -> pd.Series:
        """Normalize factor to 0-1 range."""
        if factor_series.empty:
            return factor_series
        
        min_val = factor_series.min()
        max_val = factor_series.max()
        
        if max_val == min_val:
            return pd.Series(0.5, index=factor_series.index)
        
        return (factor_series - min_val) / (max_val - min_val)

    def _construct_dynamic_portfolio(self, qualified_df: pd.DataFrame, regime_allocation: float) -> pd.Series:
        """Construct portfolio using dynamic factor weights."""
        try:
            if qualified_df.empty:
                return pd.Series()
            
            # Sort by composite score and select top stocks
            top_stocks = qualified_df.nlargest(self.target_portfolio_size, 'composite_score')
            
            # Create equal-weight portfolio
            portfolio = pd.Series(0.0, index=self.daily_returns_matrix.columns)
            weight_per_stock = regime_allocation / len(top_stocks)
            
            for _, row in top_stocks.iterrows():
                ticker = row['ticker']
                if ticker in portfolio.index:
                    portfolio[ticker] = weight_per_stock
            
            return portfolio
            
        except Exception as e:
            print(f"Warning: Error constructing portfolio: {e}")
            return pd.Series()

    def _apply_holdings_and_calculate_turnover(self, daily_holdings: pd.DataFrame, 
                                             target_portfolio: pd.Series, 
                                             rebal_date: pd.Timestamp,
                                             previous_portfolio: pd.Series) -> float:
        """Apply holdings and calculate turnover."""
        try:
            # Find the next rebalancing date or end of data
            next_rebal_date = None
            for date in daily_holdings.index:
                if date > rebal_date:
                    next_rebal_date = date
                    break
            
            if next_rebal_date is None:
                next_rebal_date = daily_holdings.index[-1]
            
            # Apply holdings for the period
            daily_holdings.loc[rebal_date:next_rebal_date, target_portfolio.index] = target_portfolio.values
            
            # Calculate turnover
            turnover = abs(target_portfolio - previous_portfolio).sum() / 2
            
            return turnover
            
        except Exception as e:
            print(f"Warning: Error applying holdings: {e}")
            return 0.0

    def _calculate_net_returns(self, daily_holdings: pd.DataFrame) -> pd.Series:
        """Calculate net returns including transaction costs."""
        try:
            # Calculate gross returns
            gross_returns = (daily_holdings * self.daily_returns_matrix).sum(axis=1)
            
            # Calculate transaction costs (simplified)
            # In a full implementation, you would track actual trades
            net_returns = gross_returns - self.transaction_cost * 0.01  # Approximate transaction cost
            
            return net_returns
            
        except Exception as e:
            print(f"Warning: Error calculating net returns: {e}")
            return pd.Series(0.0, index=self.daily_returns_matrix.index)

    def generate_comprehensive_tearsheet(self, net_returns: pd.Series, diagnostics: pd.DataFrame) -> dict:
        """Generate comprehensive performance analysis."""
        try:
            # Calculate performance metrics
            metrics = self.base_engine.calculate_performance_metrics(net_returns, self.benchmark_returns)
            
            # Add strategy-specific metrics
            if not diagnostics.empty:
                # Regime distribution
                regime_distribution = diagnostics['regime'].value_counts()
                metrics['regime_distribution'] = regime_distribution.to_dict()
                
                # Average factor weights by regime
                regime_weights = diagnostics.groupby('regime')[['roaa_weight', 'pe_weight', 'momentum_weight', 'low_vol_weight']].mean()
                metrics['regime_weights'] = regime_weights.to_dict()
                
                # Average turnover
                metrics['avg_turnover'] = diagnostics['turnover'].mean()
            
            return metrics
            
        except Exception as e:
            print(f"Warning: Error generating tearsheet: {e}")
            return {}

# %%
# Main execution block
if __name__ == "__main__":
    print("🔧 QVM Engine v3j Dynamic Factor Weights Strategy")
    print("   - Loading data and initializing strategy...")
    
    # Note: This would be integrated with the component comparison framework
    # For standalone execution, you would need to load data and run the strategy
    
    print("✅ Strategy implementation complete.")
    print("   - Use with component_comparison.py for full analysis")
    print("   - Regime-specific factor weights implemented")
    print("   - Enhanced factor set: ROAA, P/E, Momentum, Low-Volatility") 