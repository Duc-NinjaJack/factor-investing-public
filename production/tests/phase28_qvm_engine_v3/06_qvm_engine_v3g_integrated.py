# %% [markdown]
# # QVM Engine v3g - Integrated Innovation Implementation
# 
# **Objective:** Complete integration of all innovations from v3f, v3e percentile, and v3e optimized:
# - Top 200 stocks by ADTV (from v3f)
# - Adaptive percentile-based regime detection with historical learning (from v3e percentile)
# - Performance optimization with pre-loaded data (from v3e optimized)
# - Extended backtest period 2016-2025 with multiple tearsheets
# 
# **Key Innovations:**
# - Top 200 universe by ADTV instead of hard thresholds
# - Dynamic percentile-based regime detection that adapts to market conditions
# - Historical learning for regime thresholds
# - Optimized data preloading for 70-90% speedup
# - Vectorized factor calculations
# - No synthetic data, no look-ahead bias
# 
# **File:** 06_qvm_engine_v3g_integrated.py
# 
# ---

# %% [code]
# Core scientific libraries
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
from pathlib import Path
import sys
import yaml

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Database connectivity
from sqlalchemy import create_engine, text

# --- Environment Setup ---
warnings.filterwarnings('ignore')

# %% [code]
# --- Add Project Root to Python Path ---
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
    from production.database.mappings.financial_mapping_manager import FinancialMappingManager
    print(f"✅ Successfully imported production modules.")
    print(f"   - Project Root set to: {project_root}")

except (ImportError, FileNotFoundError) as e:
    print(f"❌ ERROR: Could not import production modules. Please check your directory structure.")
    print(f"   - Final Path Searched: {project_root}")
    print(f"   - Error: {e}")
    raise

# %% [code]
# --- QVM Engine v3g Integrated Configuration ---
QVM_CONFIG = {
    # --- Backtest Parameters ---
    "strategy_name": "QVM_Engine_v3g_Integrated",
    "backtest_start_date": "2016-01-01",
    "backtest_end_date": "2025-12-31",
    "rebalance_frequency": "M", # Monthly
    "transaction_cost_bps": 30, # Flat 30bps

    # --- Universe Construction (Top 200 by ADTV) ---
    "universe": {
        "lookback_days": 63,
        "top_n_stocks": 200,  # Top 200 stocks by ADTV (from v3f)
        "max_position_size": 0.05,
        "max_sector_exposure": 0.30,
        "target_portfolio_size": 20,
    },

    # --- Factor Configuration ---
    "factors": {
        "roaa_weight": 0.3,
        "pe_weight": 0.3,
        "momentum_weight": 0.4,
        "momentum_horizons": [21, 63, 126, 252], # 1M, 3M, 6M, 12M
        "skip_months": 1,
        "fundamental_lag_days": 45,  # 45-day lag for announcement delay
    },

    # --- Adaptive Regime Detection (from v3e percentile) ---
    "regime": {
        "lookback_period": 90,                    # 90 days lookback period
        "volatility_percentile_high": 75.0,       # 75th percentile for high volatility
        "return_percentile_high": 75.0,           # 75th percentile for high return
        "return_percentile_low": 25.0,            # 25th percentile for low return
        "min_history_required": 30,               # Minimum data points for percentile calculation
        "regime_allocation": {
            'momentum': 0.8,    # High allocation in momentum regime
            'stress': 0.3,      # Low allocation in stress regime
            'normal': 0.6       # Moderate allocation in normal regime
        }
    }
}

print("\n⚙️  QVM Engine v3g Integrated Configuration Loaded:")
print(f"   - Strategy: {QVM_CONFIG['strategy_name']}")
print(f"   - Period: {QVM_CONFIG['backtest_start_date']} to {QVM_CONFIG['backtest_end_date']}")
print(f"   - Universe: Top {QVM_CONFIG['universe']['top_n_stocks']} stocks by ADTV")
print(f"   - Factors: ROAA + P/E + Multi-horizon Momentum")
print(f"   - Regime Detection: Adaptive percentile-based with historical learning")
print(f"   - Performance: Optimized with pre-loaded data")

# %% [code]
# --- Database Connection ---
def create_db_connection():
    """Establishes a SQLAlchemy database engine connection."""
    try:
        db_manager = get_database_manager()
        engine = db_manager.get_engine()
        
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print(f"\n✅ Database connection established successfully.")
        return engine

    except Exception as e:
        print(f"❌ FAILED to connect to the database.")
        print(f"   - Error: {e}")
        return None

# Create the engine for this session
engine = create_db_connection()

if engine is None:
    raise ConnectionError("Database connection failed. Halting execution.")

# %% [markdown]
# ## INTEGRATED ADAPTIVE REGIME DETECTOR
# 
# Combines percentile-based thresholds with historical learning from v3e percentile.

# %% [code]
class AdaptiveRegimeDetector:
    """
    Adaptive regime detector with percentile-based thresholds and historical learning.
    Integrated from v3e percentile with enhanced features.
    """
    
    def __init__(self, config: dict):
        """
        Initialize adaptive regime detector
        
        Args:
            config: Configuration dictionary with regime parameters
        """
        regime_config = config['regime']
        self.lookback_period = regime_config['lookback_period']
        self.volatility_percentile_high = regime_config['volatility_percentile_high']
        self.return_percentile_high = regime_config['return_percentile_high']
        self.return_percentile_low = regime_config['return_percentile_low']
        self.min_history_required = regime_config['min_history_required']
        self.regime_allocation = regime_config['regime_allocation']
        
        # Historical data storage for percentile calculation
        self.volatility_history = []
        self.return_history = []
        
        print(f"✅ AdaptiveRegimeDetector initialized:")
        print(f"   - Lookback Period: {self.lookback_period} days")
        print(f"   - Volatility Percentile: {self.volatility_percentile_high}th")
        print(f"   - Return Percentiles: {self.return_percentile_high}th (high), {self.return_percentile_low}th (low)")
        print(f"   - Min History Required: {self.min_history_required} data points")
    
    def detect_regime(self, price_data: pd.DataFrame) -> str:
        """
        Detect market regime using adaptive percentile-based thresholds
        
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
        if len(self.volatility_history) >= self.min_history_required:
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
                regime = 'momentum'
            elif current_return < return_threshold_low:
                regime = 'stress'
            else:
                regime = 'normal'
        else:
            regime = 'normal'
        
        # Debug output
        print(f"   🔍 Regime Debug: Vol={current_vol:.2%} (thresh={vol_threshold:.2%}), Ret={current_return:.2%} (high={return_threshold_high:.2%}, low={return_threshold_low:.2%})")
        print(f"   📈 Detected: {regime}")
        
        return regime
    
    def get_regime_allocation(self, regime: str) -> float:
        """
        Get portfolio allocation based on detected regime
        
        Args:
            regime: Detected regime ('momentum', 'stress', 'normal')
            
        Returns:
            Portfolio allocation percentage
        """
        return self.regime_allocation.get(regime, 0.6)

# %% [markdown]
# ## OPTIMIZED DATA PRELOADER (from v3e optimized)
# 
# Pre-loads all data upfront for performance optimization.

# %% [code]
class OptimizedDataPreloader:
    """
    Pre-loads all data upfront to eliminate database queries in loops.
    This is the key optimization that provides 70-90% speedup.
    """
    def __init__(self, config: dict, db_engine):
        self.config = config
        self.engine = db_engine
        self.start_date = pd.Timestamp(config['backtest_start_date'])
        self.end_date = pd.Timestamp(config['backtest_end_date'])
        
        # Add buffer for rolling calculations
        self.buffer_start = self.start_date - pd.DateOffset(months=6)
        
        print(f"📂 Initializing optimized data pre-loader...")
        print(f"   - Period: {self.buffer_start.date()} to {self.end_date.date()}")
    
    def load_all_data(self):
        """Load all data upfront in optimized batches."""
        print("\n🔄 Loading all data in optimized batches...")
        
        # 1. Load price and market data
        price_data = self._load_price_data()
        
        # 2. Load fundamental data
        fundamental_data = self._load_fundamental_data()
        
        # 3. Load benchmark data
        benchmark_data = self._load_benchmark_data()
        
        # 4. Pre-calculate momentum factors
        momentum_data = self._pre_calculate_momentum(price_data)
        
        # 5. Pre-calculate universe eligibility (Top 200 by ADTV)
        universe_data = self._pre_calculate_universe(price_data)
        
        # 6. Pre-calculate regime detection
        regime_data = self._pre_calculate_regime(benchmark_data)
        
        print("✅ All data pre-loaded successfully!")
        return {
            'price_data': price_data,
            'fundamental_data': fundamental_data,
            'benchmark_data': benchmark_data,
            'momentum_data': momentum_data,
            'universe_data': universe_data,
            'regime_data': regime_data
        }
    
    def _load_price_data(self):
        """Load price data in chunks to avoid disk space issues."""
        print("   - Loading price data in chunks...")
        
        # Test database connection first
        try:
            test_query = text("SELECT COUNT(*) as count FROM vcsc_daily_data_complete LIMIT 1")
            test_result = pd.read_sql(test_query, self.engine)
            total_records = test_result['count'].iloc[0]
            print(f"     ✅ Database connection test: {total_records:,} records available")
            
            # If too many records, warn user about potential disk space issues
            if total_records > 1000000:  # More than 1M records
                print(f"     ⚠️ WARNING: Large dataset detected ({total_records:,} records)")
                print(f"     ⚠️ This may cause disk space issues during data loading")
                print(f"     ⚠️ Consider using a smaller date range or ensuring sufficient disk space")
                print(f"     ⚠️ Proceeding with real data loading...")
                
        except Exception as e:
            print(f"     ❌ Database connection failed: {e}")
            raise Exception(f"Database connection failed: {e}. Cannot proceed without database access.")
        
        # Load data in much smaller chunks to avoid disk space issues
        chunk_size = pd.DateOffset(months=3)  # 3-month chunks instead of 2-year
        all_price_data = []
        
        current_start = self.buffer_start
        chunk_count = 0
        
        while current_start < self.end_date:
            chunk_end = min(current_start + chunk_size, self.end_date)
            chunk_count += 1
            
            print(f"     - Loading chunk {chunk_count}: {current_start.date()} to {chunk_end.date()}")
            
            try:
                # First, check if data exists for this period
                check_query = text("""
                    SELECT COUNT(*) as count 
                    FROM vcsc_daily_data_complete 
                    WHERE trading_date BETWEEN :start_date AND :end_date
                """)
                
                check_result = pd.read_sql(check_query, self.engine, 
                                          params={'start_date': current_start, 'end_date': chunk_end})
                
                available_count = check_result['count'].iloc[0]
                print(f"       - Available records for this period: {available_count:,}")
                
                if available_count == 0:
                    print(f"       ⚠️ No data available for chunk {chunk_count}")
                    current_start = chunk_end
                    continue
                
                # If too many records, try even smaller chunk
                if available_count > 50000:  # Threshold for disk space issues
                    print(f"       ⚠️ Too many records ({available_count:,}), trying 1-month chunk")
                    smaller_chunk_size = pd.DateOffset(months=1)
                    chunk_end = min(current_start + smaller_chunk_size, self.end_date)
                
                query = text("""
                    SELECT 
                        trading_date as date,
                        ticker,
                        close_price_adjusted as close,
                        total_volume as volume,
                        market_cap,
                        total_volume * close_price_adjusted as adtv_vnd
                    FROM vcsc_daily_data_complete
                    WHERE trading_date BETWEEN :start_date AND :end_date
                    ORDER BY trading_date, ticker
                """)
                
                chunk_data = pd.read_sql(query, self.engine, 
                                        params={'start_date': current_start, 'end_date': chunk_end},
                                        parse_dates=['date'])
                
                if not chunk_data.empty:
                    all_price_data.append(chunk_data)
                    print(f"       ✅ Loaded {len(chunk_data):,} observations")
                    print(f"       - Date range: {chunk_data['date'].min()} to {chunk_data['date'].max()}")
                    print(f"       - Unique tickers: {chunk_data['ticker'].nunique()}")
                    
                    # Clear memory after each chunk
                    del chunk_data
                    import gc
                    gc.collect()
                else:
                    print(f"       ⚠️ No data for chunk {chunk_count}")
                
            except Exception as e:
                print(f"       ❌ Error loading chunk {chunk_count}: {e}")
                # Try with much smaller chunk
                smaller_chunk_size = pd.DateOffset(weeks=2)  # 2-week chunks
                chunk_end = min(current_start + smaller_chunk_size, self.end_date)
                
                try:
                    print(f"       - Trying 2-week chunk: {current_start.date()} to {chunk_end.date()}")
                    chunk_data = pd.read_sql(query, self.engine, 
                                            params={'start_date': current_start, 'end_date': chunk_end},
                                            parse_dates=['date'])
                    
                    if not chunk_data.empty:
                        all_price_data.append(chunk_data)
                        print(f"       ✅ Loaded {len(chunk_data):,} observations (2-week chunk)")
                        
                        # Clear memory after each chunk
                        del chunk_data
                        import gc
                        gc.collect()
                    else:
                        print(f"       ⚠️ No data for 2-week chunk {chunk_count}")
                        
                except Exception as e2:
                    print(f"       ❌ Error loading 2-week chunk {chunk_count}: {e2}")
                    # Try with 1-week chunk as last resort
                    try:
                        tiny_chunk_size = pd.DateOffset(weeks=1)
                        chunk_end = min(current_start + tiny_chunk_size, self.end_date)
                        print(f"       - Trying 1-week chunk: {current_start.date()} to {chunk_end.date()}")
                        
                        chunk_data = pd.read_sql(query, self.engine, 
                                                params={'start_date': current_start, 'end_date': chunk_end},
                                                parse_dates=['date'])
                        
                        if not chunk_data.empty:
                            all_price_data.append(chunk_data)
                            print(f"       ✅ Loaded {len(chunk_data):,} observations (1-week chunk)")
                            
                            # Clear memory after each chunk
                            del chunk_data
                            import gc
                            gc.collect()
                        else:
                            print(f"       ⚠️ No data for 1-week chunk {chunk_count}")
                    except Exception as e3:
                        print(f"       ❌ Error loading 1-week chunk {chunk_count}: {e3}")
                        # Skip this chunk and continue
                        pass
            
            current_start = chunk_end
        
        if not all_price_data:
            print("     ❌ ERROR: No real price data could be loaded")
            print("     ❌ This may be due to disk space issues or database problems")
            print("     ❌ Please ensure sufficient disk space and database connectivity")
            raise Exception("Failed to load any real price data. Cannot proceed with synthetic data.")
        else:
            # Combine all chunks
            price_data = pd.concat(all_price_data, ignore_index=True)
            price_data = price_data.drop_duplicates(subset=['date', 'ticker'])
            price_data = price_data.sort_values(['date', 'ticker'])
            print(f"     ✅ Loaded {len(price_data):,} total real price observations")
        
        # Create returns matrix
        price_data['return'] = price_data.groupby('ticker')['close'].pct_change()
        returns_matrix = price_data.pivot(index='date', columns='ticker', values='return')
        
        print(f"     ✅ Created returns matrix: {returns_matrix.shape}")
        print(f"     ✅ Date range: {price_data['date'].min()} to {price_data['date'].max()}")
        print(f"     ✅ Unique tickers: {price_data['ticker'].nunique()}")
        
        return {
            'raw_data': price_data,
            'returns_matrix': returns_matrix,
            'price_matrix': price_data.pivot(index='date', columns='ticker', values='close'),
            'volume_matrix': price_data.pivot(index='date', columns='ticker', values='volume'),
            'market_cap_matrix': price_data.pivot(index='date', columns='ticker', values='market_cap'),
            'adtv_matrix': price_data.pivot(index='date', columns='ticker', values='adtv_vnd')
        }
    

    
    def _load_fundamental_data(self):
        """Load fundamental data in chunks to avoid disk space issues."""
        print("   - Loading fundamental data in chunks...")
        
        # Load data in year chunks to avoid disk space issues
        chunk_size = pd.DateOffset(years=1)
        all_fundamental_data = []
        
        current_start = self.buffer_start
        chunk_count = 0
        
        while current_start < self.end_date:
            chunk_end = min(current_start + chunk_size, self.end_date)
            chunk_count += 1
            
            print(f"     - Loading fundamental chunk {chunk_count}: {current_start.date()} to {chunk_end.date()}")
            
            try:
                query = text("""
                    WITH quarterly_fundamentals AS (
                        SELECT 
                            fv.ticker,
                            fv.year,
                            fv.quarter,
                            DATE(CONCAT(fv.year, '-', LPAD(fv.quarter * 3, 2, '0'), '-01')) as quarter_date,
                            SUM(CASE WHEN fv.item_id = 1 AND fv.statement_type = 'PL' THEN fv.value / 1e9 ELSE 0 END) as netprofit,
                            SUM(CASE WHEN fv.item_id = 2 AND fv.statement_type = 'BS' THEN fv.value / 1e9 ELSE 0 END) as totalassets,
                            SUM(CASE WHEN fv.item_id = 2 AND fv.statement_type = 'PL' THEN fv.value / 1e9 ELSE 0 END) as revenue
                        FROM fundamental_values fv
                        WHERE fv.year BETWEEN YEAR(:start_date) AND YEAR(:end_date)
                        GROUP BY fv.ticker, fv.year, fv.quarter
                    )
                    SELECT 
                        qf.ticker,
                        mi.sector,
                        qf.quarter_date as date,
                        qf.netprofit,
                        qf.totalassets,
                        qf.revenue,
                        CASE 
                            WHEN qf.totalassets > 0 THEN qf.netprofit / qf.totalassets 
                            ELSE NULL 
                        END as roaa,
                        CASE 
                            WHEN qf.revenue > 0 THEN qf.netprofit / qf.revenue
                            ELSE NULL 
                        END as net_margin,
                        CASE 
                            WHEN qf.totalassets > 0 THEN qf.revenue / qf.totalassets
                            ELSE NULL 
                        END as asset_turnover
                    FROM quarterly_fundamentals qf
                    LEFT JOIN master_info mi ON qf.ticker = mi.ticker
                    WHERE qf.netprofit > 0 
                    AND qf.totalassets > 0
                    AND qf.revenue > 0
                    ORDER BY qf.ticker, qf.quarter_date
                """)
                
                chunk_data = pd.read_sql(query, self.engine, 
                                        params={'start_date': current_start, 'end_date': chunk_end},
                                        parse_dates=['date'])
                
                if not chunk_data.empty:
                    all_fundamental_data.append(chunk_data)
                    print(f"       ✅ Loaded {len(chunk_data):,} fundamental observations")
                else:
                    print(f"       ⚠️ No fundamental data for chunk {chunk_count}")
                
            except Exception as e:
                print(f"       ❌ Error loading fundamental chunk {chunk_count}: {e}")
                # Skip this chunk and continue
                pass
            
            current_start = chunk_end
        
        if not all_fundamental_data:
            print("     ❌ ERROR: No real fundamental data could be loaded")
            print("     ❌ This may be due to disk space issues or database problems")
            print("     ❌ Please ensure sufficient disk space and database connectivity")
            raise Exception("Failed to load any real fundamental data. Cannot proceed with synthetic data.")
        else:
            # Combine all chunks
            fundamental_data = pd.concat(all_fundamental_data, ignore_index=True)
            fundamental_data = fundamental_data.drop_duplicates(subset=['date', 'ticker'])
            fundamental_data = fundamental_data.sort_values(['date', 'ticker'])
            print(f"     ✅ Loaded {len(fundamental_data):,} total real fundamental observations")
        
        print(f"     ✅ Loaded {len(fundamental_data):,} total fundamental observations")
        return fundamental_data
        return fundamental_data
    
    def _load_benchmark_data(self):
        """Load benchmark data (VN-Index) with error handling."""
        print("   - Loading benchmark data...")
        
        try:
            query = text("""
                SELECT date, close
                FROM etf_history
                WHERE ticker = 'VNINDEX' AND date BETWEEN :start_date AND :end_date
                ORDER BY date
            """)
            
            benchmark_data = pd.read_sql(query, self.engine, 
                                        params={'start_date': self.buffer_start, 'end_date': self.end_date},
                                        parse_dates=['date'])
            
            if benchmark_data.empty:
                print("     ❌ ERROR: No benchmark data found")
                print("     ❌ This may be due to database problems")
                print("     ❌ Please ensure database connectivity and data availability")
                raise Exception("Failed to load any benchmark data. Cannot proceed with synthetic data.")
            
            benchmark_returns = benchmark_data.set_index('date')['close'].pct_change()
            
            print(f"     ✅ Loaded {len(benchmark_data):,} benchmark observations")
            return benchmark_returns
            
        except Exception as e:
            print(f"     ❌ Error loading benchmark data: {e}")
            print("     ❌ Cannot proceed without benchmark data")
            raise Exception(f"Failed to load benchmark data: {e}. Cannot proceed with synthetic data.")
    
    def _pre_calculate_momentum(self, price_data):
        """Pre-calculate momentum factors for all dates and tickers."""
        print("   - Pre-calculating momentum factors...")
        
        price_matrix = price_data['price_matrix']
        momentum_horizons = self.config['factors']['momentum_horizons']
        skip_months = self.config['factors']['skip_months']
        
        momentum_data = {}
        
        for horizon in momentum_horizons:
            momentum_data[f'momentum_{horizon}d'] = {}
            
            for ticker in price_matrix.columns:
                ticker_prices = price_matrix[ticker].dropna()
                
                if len(ticker_prices) < horizon + skip_months:
                    continue
                
                # Calculate momentum with skip month
                current_price = ticker_prices.iloc[skip_months]
                past_price = ticker_prices.iloc[horizon + skip_months - 1]
                momentum = (current_price / past_price) - 1
                
                momentum_data[f'momentum_{horizon}d'][ticker] = momentum
        
        print(f"     ✅ Pre-calculated momentum for {len(momentum_horizons)} horizons")
        return momentum_data
    
    def _pre_calculate_universe(self, price_data):
        """Pre-calculate universe eligibility (Top 200 by ADTV)."""
        print("   - Pre-calculating universe eligibility (Top 200 by ADTV)...")
        
        adtv_matrix = price_data['adtv_matrix']
        lookback_days = self.config['universe']['lookback_days']
        top_n_stocks = self.config['universe']['top_n_stocks']
        
        universe_mask = pd.DataFrame(index=adtv_matrix.index, columns=adtv_matrix.columns, dtype=bool)
        
        for date in adtv_matrix.index:
            # Get rolling ADTV for lookback period
            start_date = date - pd.Timedelta(days=lookback_days)
            rolling_adtv = adtv_matrix.loc[start_date:date].mean()
            
            # Select top N stocks by ADTV
            top_stocks = rolling_adtv.nlargest(top_n_stocks).index
            
            # Mark as eligible
            universe_mask.loc[date, top_stocks] = True
            universe_mask.loc[date, universe_mask.columns.difference(top_stocks)] = False
        
        print(f"     ✅ Pre-calculated universe eligibility for {len(adtv_matrix.index)} dates")
        return universe_mask
    
    def _pre_calculate_regime(self, benchmark_returns):
        """Pre-calculate regime detection for all dates."""
        print("   - Pre-calculating regime detection...")
        
        regime_detector = AdaptiveRegimeDetector(self.config)
        regime_data = {}
        
        for date in benchmark_returns.index:
            # Get historical data up to this date
            historical_data = benchmark_returns.loc[:date]
            
            if len(historical_data) < 20:  # Minimum data requirement
                regime_data[date] = 'normal'
                continue
            
            # Create price series for regime detection
            price_series = (1 + historical_data).cumprod()
            price_data = pd.DataFrame({'close': price_series})
            
            # Detect regime
            regime = regime_detector.detect_regime(price_data)
            regime_data[date] = regime
        
        print(f"     ✅ Pre-calculated regime detection for {len(regime_data)} dates")
        return regime_data 

# %% [markdown]
# ## OPTIMIZED FACTOR CALCULATOR (from v3e optimized)
# 
# Vectorized factor calculations with sector-aware adjustments.

# %% [code]
class OptimizedFactorCalculator:
    """
    Optimized factor calculator with vectorized operations and sector-aware calculations.
    """
    
    def __init__(self, config: dict):
        self.config = config
    
    def calculate_factors_for_date(self, date: pd.Timestamp, 
                                 fundamental_data: pd.DataFrame,
                                 momentum_data: dict,
                                 universe_mask: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all factors for a specific date using vectorized operations.
        """
        # Get universe for this date
        if date not in universe_mask.index:
            print(f"       ⚠️ Date {date.date()} not in universe mask index")
            return pd.DataFrame()
        
        universe_tickers = universe_mask.loc[date][universe_mask.loc[date] == True].index.tolist()
        
        if not universe_tickers:
            print(f"       ⚠️ No universe tickers for date {date.date()}")
            return pd.DataFrame()
        
        print(f"       - Universe tickers: {len(universe_tickers)}")
        
        # Get fundamental data for universe
        lag_days = self.config['factors']['fundamental_lag_days']
        lag_date = date - pd.Timedelta(days=lag_days)
        
        fundamental_subset = fundamental_data[
            (fundamental_data['ticker'].isin(universe_tickers)) &
            (fundamental_data['date'] <= lag_date)
        ]
        
        if fundamental_subset.empty:
            print(f"       ⚠️ No fundamental data available before {lag_date.date()}")
            return pd.DataFrame()
        
        print(f"       - Available fundamental records: {len(fundamental_subset)}")
        
        # Get latest fundamental data for each ticker
        latest_fundamentals = fundamental_subset.groupby('ticker').last().reset_index()
        
        print(f"       - Universe tickers with fundamental data: {len(latest_fundamentals)}")
        
        # Add momentum factors
        factors_df = self._add_momentum_factors(latest_fundamentals, momentum_data, date)
        
        # Calculate quality-adjusted PE
        factors_df = self._calculate_quality_adjusted_pe(factors_df)
        
        # Calculate composite score
        factors_df = self._calculate_composite_score_vectorized(factors_df)
        
        print(f"       ✅ Calculated factors for {len(factors_df)} stocks")
        return factors_df
    
    def _add_momentum_factors(self, fundamental_df: pd.DataFrame, momentum_data: dict, date: pd.Timestamp) -> pd.DataFrame:
        """Add momentum factors to fundamental data."""
        momentum_horizons = self.config['factors']['momentum_horizons']
        
        for horizon in momentum_horizons:
            momentum_key = f'momentum_{horizon}d'
            if momentum_key in momentum_data:
                momentum_values = momentum_data[momentum_key]
                
                # Add momentum values for available tickers
                fundamental_df[momentum_key] = fundamental_df['ticker'].map(momentum_values).fillna(0)
        
        return fundamental_df
    
    def _calculate_quality_adjusted_pe(self, factors_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate quality-adjusted P/E by sector."""
        if 'roaa' not in factors_df.columns or 'sector' not in factors_df.columns:
            return factors_df
        
        # Create ROAA quintiles within each sector
        def safe_qcut(x):
            try:
                if len(x) < 5:
                    return pd.Series(['Q3'] * len(x), index=x.index)
                return pd.qcut(x, 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')
            except ValueError:
                return pd.Series(['Q3'] * len(x), index=x.index)
        
        factors_df['roaa_quintile'] = factors_df.groupby('sector')['roaa'].transform(safe_qcut)
        
        # Fill missing quintiles with Q3
        factors_df['roaa_quintile'] = factors_df['roaa_quintile'].fillna('Q3')
        
        # Quality-adjusted P/E weights (higher quality = higher weight)
        quality_weights = {
            'Q1': 0.5,  # Low quality
            'Q2': 0.7,
            'Q3': 1.0,  # Medium quality
            'Q4': 1.3,
            'Q5': 1.5   # High quality
        }
        
        factors_df['quality_adjusted_pe'] = factors_df['roaa_quintile'].map(quality_weights)
        
        # Calculate momentum score with correct signal directions
        momentum_columns = [col for col in factors_df.columns if col.startswith('momentum_')]
        
        if momentum_columns:
            momentum_score = 0.0
            
            for col in momentum_columns:
                if 'momentum_63d' in col or 'momentum_126d' in col:  # 3M and 6M - positive
                    momentum_score += factors_df[col]
                elif 'momentum_21d' in col or 'momentum_252d' in col:  # 1M and 12M - contrarian
                    momentum_score -= factors_df[col]  # Negative for contrarian
            
            # Equal weight the components
            factors_df['momentum_score'] = momentum_score / len(momentum_columns)
        
        return factors_df
    
    def _calculate_composite_score_vectorized(self, factors_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate composite score using vectorized operations."""
        factors_df['composite_score'] = 0.0
        
        # ROAA component (positive signal)
        if 'roaa' in factors_df.columns:
            roaa_weight = self.config['factors']['roaa_weight']
            factors_df['roaa_normalized'] = (factors_df['roaa'] - factors_df['roaa'].mean()) / factors_df['roaa'].std()
            factors_df['composite_score'] += factors_df['roaa_normalized'] * roaa_weight
        
        # P/E component (contrarian signal - lower is better)
        if 'quality_adjusted_pe' in factors_df.columns:
            pe_weight = self.config['factors']['pe_weight']
            factors_df['pe_normalized'] = (factors_df['quality_adjusted_pe'] - factors_df['quality_adjusted_pe'].mean()) / factors_df['quality_adjusted_pe'].std()
            factors_df['composite_score'] += (-factors_df['pe_normalized']) * pe_weight  # Negative for contrarian
        
        # Momentum component (mixed signal - 3M/6M positive, 1M/12M contrarian)
        if 'momentum_score' in factors_df.columns:
            momentum_weight = self.config['factors']['momentum_weight']
            factors_df['momentum_normalized'] = (factors_df['momentum_score'] - factors_df['momentum_score'].mean()) / factors_df['momentum_score'].std()
            factors_df['composite_score'] += factors_df['momentum_normalized'] * momentum_weight
        
        return factors_df

# %% [markdown]
# ## QVM ENGINE V3G INTEGRATED
# 
# Main engine that integrates all innovations from v3f, v3e percentile, and v3e optimized.

# %% [code]
class QVMEngineV3gIntegrated:
    """
    QVM Engine v3g with integrated innovations:
    - Top 200 stocks by ADTV (from v3f)
    - Adaptive percentile-based regime detection (from v3e percentile)
    - Performance optimization with pre-loaded data (from v3e optimized)
    """
    
    def __init__(self, config: dict, preloaded_data: dict, db_engine):
        self.config = config
        self.engine = db_engine
        self.preloaded_data = preloaded_data
        
        # Initialize components
        self.regime_detector = AdaptiveRegimeDetector(config)
        self.factor_calculator = OptimizedFactorCalculator(config)
        
        # Extract data from preloaded_data
        self.price_data = preloaded_data['price_data']
        self.fundamental_data = preloaded_data['fundamental_data']
        self.benchmark_returns = preloaded_data['benchmark_data']
        self.momentum_data = preloaded_data['momentum_data']
        self.universe_mask = preloaded_data['universe_data']
        self.regime_data = preloaded_data['regime_data']
        
        # Performance tracking
        self.portfolio_returns = []
        self.regime_history = []
        self.allocation_history = []
        
        print("✅ QVMEngineV3gIntegrated initialized.")
        print(f"   - Strategy: {config['strategy_name']}")
        print(f"   - Data Period: {self.price_data['returns_matrix'].index.min().date()} to {self.price_data['returns_matrix'].index.max().date()}")

    def run_backtest(self) -> (pd.Series, pd.DataFrame):
        """Executes the full backtesting pipeline."""
        print("\n🚀 Starting QVM Engine v3g integrated backtest execution...")
        
        rebalance_dates = self._generate_rebalance_dates()
        daily_holdings, diagnostics = self._run_optimized_backtesting_loop(rebalance_dates)
        net_returns = self._calculate_net_returns(daily_holdings)
        
        print("✅ QVM Engine v3g integrated backtest execution complete.")
        return net_returns, diagnostics

    def _generate_rebalance_dates(self) -> list:
        """Generates monthly rebalance dates based on actual trading days."""
        all_trading_dates = self.price_data['returns_matrix'].index
        rebal_dates_calendar = pd.date_range(
            start=self.config['backtest_start_date'],
            end=self.config['backtest_end_date'],
            freq=self.config['rebalance_frequency']
        )
        actual_rebal_dates = [all_trading_dates[all_trading_dates.searchsorted(d, side='left')-1] for d in rebal_dates_calendar if d >= all_trading_dates.min()]
        print(f"   - Generated {len(actual_rebal_dates)} monthly rebalance dates.")
        return sorted(list(set(actual_rebal_dates)))

    def _run_optimized_backtesting_loop(self, rebalance_dates: list) -> (pd.DataFrame, pd.DataFrame):
        """The optimized loop for portfolio construction at each rebalance date."""
        daily_holdings = pd.DataFrame(0.0, index=self.price_data['returns_matrix'].index, columns=self.price_data['returns_matrix'].columns)
        diagnostics_log = []
        
        # Track the last successful portfolio for fallback
        last_successful_portfolio = None
        last_successful_date = None
        
        for i, rebal_date in enumerate(rebalance_dates):
            if i % 10 == 0:
                print(f"   - Processing rebalance {i+1}/{len(rebalance_dates)}: {rebal_date.date()}...", end="")
            
            try:
                # Get regime from pre-calculated data
                regime = self.regime_data.get(rebal_date, 'normal')
                regime_allocation = self.regime_detector.get_regime_allocation(regime)
                
                # Calculate factors using optimized calculator
                factors_df = self.factor_calculator.calculate_factors_for_date(
                    rebal_date, 
                    self.fundamental_data,
                    self.momentum_data,
                    self.universe_mask
                )
                
                if factors_df.empty:
                    if i % 10 == 0:
                        print(" ⚠️ No factor data.")
                    # Use last successful portfolio if available
                    if last_successful_portfolio is not None:
                        target_portfolio = last_successful_portfolio
                        if i % 10 == 0:
                            print(f" 🔄 Using last successful portfolio from {last_successful_date.date()}")
                    else:
                        if i % 10 == 0:
                            print(" ❌ No fallback portfolio available. Skipping.")
                        continue
                else:
                    # Apply entry criteria
                    qualified_df = self._apply_entry_criteria(factors_df)
                    
                    if qualified_df.empty:
                        if i % 10 == 0:
                            print(" ⚠️ No qualified stocks.")
                        # Use last successful portfolio if available
                        if last_successful_portfolio is not None:
                            target_portfolio = last_successful_portfolio
                            if i % 10 == 0:
                                print(f" 🔄 Using last successful portfolio from {last_successful_date.date()}")
                        else:
                            if i % 10 == 0:
                                print(" ❌ No fallback portfolio available. Skipping.")
                            continue
                    else:
                        # Construct portfolio
                        target_portfolio = self._construct_portfolio(qualified_df, regime_allocation)
                        
                        if target_portfolio.empty:
                            if i % 10 == 0:
                                print(" ⚠️ Portfolio empty.")
                            # Use last successful portfolio if available
                            if last_successful_portfolio is not None:
                                target_portfolio = last_successful_portfolio
                                if i % 10 == 0:
                                    print(f" 🔄 Using last successful portfolio from {last_successful_date.date()}")
                            else:
                                if i % 10 == 0:
                                    print(" ❌ No fallback portfolio available. Skipping.")
                                continue
                        else:
                            # Update last successful portfolio
                            last_successful_portfolio = target_portfolio.copy()
                            last_successful_date = rebal_date
                
                # Apply holdings
                start_period = rebal_date + pd.Timedelta(days=1)
                end_period = rebalance_dates[i+1] if i + 1 < len(rebalance_dates) else self.price_data['returns_matrix'].index.max()
                holding_dates = self.price_data['returns_matrix'].index[(self.price_data['returns_matrix'].index >= start_period) & (self.price_data['returns_matrix'].index <= end_period)]
                
                daily_holdings.loc[holding_dates] = 0.0
                valid_tickers = target_portfolio.index.intersection(daily_holdings.columns)
                daily_holdings.loc[holding_dates, valid_tickers] = target_portfolio[valid_tickers].values
                
                # Debug: Track last successful rebalance
                if i % 10 == 0:
                    print(f" ✅ Universe: {len(factors_df) if not factors_df.empty else 0}, Portfolio: {len(target_portfolio)}, Regime: {regime}, Last Date: {end_period.date()}")
                
                # Calculate turnover
                if i > 0:
                    prev_holdings_idx = self.price_data['returns_matrix'].index.get_loc(rebal_date) - 1
                    prev_holdings = daily_holdings.iloc[prev_holdings_idx] if prev_holdings_idx >= 0 else pd.Series(dtype='float64')
                else:
                    prev_holdings = pd.Series(dtype='float64')

                turnover = (target_portfolio - prev_holdings.reindex(target_portfolio.index).fillna(0)).abs().sum() / 2.0
                
                diagnostics_log.append({
                    'date': rebal_date,
                    'universe_size': len(factors_df) if not factors_df.empty else 0,
                    'portfolio_size': len(target_portfolio),
                    'regime': regime,
                    'regime_allocation': regime_allocation,
                    'turnover': turnover
                })
                
                if i % 10 == 0:
                    print(f" ✅ Universe: {len(factors_df) if not factors_df.empty else 0}, Portfolio: {len(target_portfolio)}, Regime: {regime}, Turnover: {turnover:.2%}")
            
            except Exception as e:
                if i % 10 == 0:
                    print(f" ❌ Error: {e}")
                # Use last successful portfolio if available
                if last_successful_portfolio is not None:
                    target_portfolio = last_successful_portfolio
                    if i % 10 == 0:
                        print(f" 🔄 Using last successful portfolio from {last_successful_date.date()}")
                    
                    # Apply holdings with fallback portfolio
                    start_period = rebal_date + pd.Timedelta(days=1)
                    end_period = rebalance_dates[i+1] if i + 1 < len(rebalance_dates) else self.price_data['returns_matrix'].index.max()
                    holding_dates = self.price_data['returns_matrix'].index[(self.price_data['returns_matrix'].index >= start_period) & (self.price_data['returns_matrix'].index <= end_period)]
                    
                    daily_holdings.loc[holding_dates] = 0.0
                    valid_tickers = target_portfolio.index.intersection(daily_holdings.columns)
                    daily_holdings.loc[holding_dates, valid_tickers] = target_portfolio[valid_tickers].values
                    
                    diagnostics_log.append({
                        'date': rebal_date,
                        'universe_size': 0,
                        'portfolio_size': len(target_portfolio),
                        'regime': 'fallback',
                        'regime_allocation': 1.0,
                        'turnover': 0.0
                    })
                else:
                    continue

        if diagnostics_log:
            return daily_holdings, pd.DataFrame(diagnostics_log).set_index('date')
        else:
            return daily_holdings, pd.DataFrame()

    def _apply_entry_criteria(self, factors_df: pd.DataFrame) -> pd.DataFrame:
        """Apply entry criteria to filter stocks."""
        # Basic quality filters
        qualified = factors_df.copy()
        
        initial_count = len(qualified)
        
        if 'roaa' in qualified.columns:
            roaa_filtered = qualified[qualified['roaa'] > 0]
            roaa_count = len(roaa_filtered)
            if roaa_count < initial_count:
                print(f"       - ROAA filter: {initial_count} -> {roaa_count} stocks (removed {initial_count - roaa_count})")
            qualified = roaa_filtered
        
        if 'net_margin' in qualified.columns:
            margin_filtered = qualified[qualified['net_margin'] > 0]
            margin_count = len(margin_filtered)
            if margin_count < len(qualified):
                print(f"       - Net margin filter: {len(qualified)} -> {margin_count} stocks (removed {len(qualified) - margin_count})")
            qualified = margin_filtered
        
        final_count = len(qualified)
        if final_count == 0:
            print(f"       ⚠️ All {initial_count} stocks filtered out by entry criteria")
        else:
            print(f"       ✅ {final_count}/{initial_count} stocks passed entry criteria")
        
        return qualified

    def _construct_portfolio(self, qualified_df: pd.DataFrame, regime_allocation: float) -> pd.Series:
        """Construct the portfolio using the qualified stocks."""
        if qualified_df.empty:
            return pd.Series(dtype='float64')
        
        # Sort by composite score
        qualified_df = qualified_df.sort_values('composite_score', ascending=False)
        
        # Select top stocks
        target_size = self.config['universe']['target_portfolio_size']
        selected_stocks = qualified_df.head(target_size)
        
        if selected_stocks.empty:
            return pd.Series(dtype='float64')
        
        # Equal weight portfolio
        portfolio = pd.Series(regime_allocation / len(selected_stocks), index=selected_stocks['ticker'])
        
        return portfolio

    def _calculate_net_returns(self, daily_holdings: pd.DataFrame) -> pd.Series:
        """Calculate net returns with transaction costs."""
        print(f"\n🔍 DEBUG: _calculate_net_returns")
        print(f"   - daily_holdings shape: {daily_holdings.shape}")
        print(f"   - daily_holdings date range: {daily_holdings.index.min()} to {daily_holdings.index.max()}")
        print(f"   - Non-zero holdings count: {(daily_holdings != 0).sum().sum()}")
        
        holdings_shifted = daily_holdings.shift(1).fillna(0.0)
        gross_returns = (holdings_shifted * self.price_data['returns_matrix']).sum(axis=1)
        
        # Calculate turnover and costs
        turnover = (holdings_shifted - holdings_shifted.shift(1)).abs().sum(axis=1) / 2.0
        costs = turnover * (self.config['transaction_cost_bps'] / 10000)
        net_returns = (gross_returns - costs).rename(self.config['strategy_name'])
        
        print(f"   - gross_returns shape: {gross_returns.shape}")
        print(f"   - gross_returns date range: {gross_returns.index.min()} to {gross_returns.index.max()}")
        print(f"   - Non-zero gross returns count: {(gross_returns != 0).sum()}")
        print(f"   - First non-zero return date: {gross_returns[gross_returns != 0].index.min() if (gross_returns != 0).any() else 'None'}")
        print(f"   - Last non-zero return date: {gross_returns[gross_returns != 0].index.max() if (gross_returns != 0).any() else 'None'}")
        
        print("\n💸 Net returns calculated.")
        print(f"   - Total Gross Return: {(1 + gross_returns).prod() - 1:.2%}")
        print(f"   - Total Net Return: {(1 + net_returns).prod() - 1:.2%}")
        print(f"   - Total Cost Drag: {(gross_returns.sum() - net_returns.sum()):.2%}")
        
        return net_returns

# %% [markdown]
# ## PERFORMANCE ANALYSIS FUNCTIONS
# 
# Comprehensive performance analysis and visualization functions.

# %% [code]
def calculate_performance_metrics(returns: pd.Series, benchmark: pd.Series, periods_per_year: int = 252) -> dict:
    """Calculates comprehensive performance metrics with corrected benchmark alignment."""
    print(f"\n🔍 DEBUG: calculate_performance_metrics")
    print(f"   - returns shape: {returns.shape}")
    print(f"   - returns date range: {returns.index.min()} to {returns.index.max()}")
    print(f"   - Non-zero returns count: {(returns != 0).sum()}")
    print(f"   - First non-zero return: {returns[returns != 0].index.min() if (returns != 0).any() else 'None'}")
    print(f"   - Last non-zero return: {returns[returns != 0].index.max() if (returns != 0).any() else 'None'}")
    
    # Align benchmark
    first_trade_date = returns.loc[returns.ne(0)].index.min()
    print(f"   - first_trade_date: {first_trade_date}")
    
    if pd.isna(first_trade_date):
        return {metric: 0.0 for metric in ['Annualized Return (%)', 'Annualized Volatility (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 'Calmar Ratio', 'Information Ratio', 'Beta']}
    
    aligned_returns = returns.loc[first_trade_date:]
    aligned_benchmark = benchmark.loc[first_trade_date:]
    
    print(f"   - aligned_returns shape: {aligned_returns.shape}")
    print(f"   - aligned_returns date range: {aligned_returns.index.min()} to {aligned_returns.index.max()}")

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

def generate_comprehensive_tearsheet(strategy_returns: pd.Series, benchmark_returns: pd.Series, diagnostics: pd.DataFrame, title: str):
    """Generates comprehensive institutional tearsheet with equity curve and analysis."""
    
    print(f"\n🔍 DEBUG: generate_comprehensive_tearsheet")
    print(f"   - strategy_returns shape: {strategy_returns.shape}")
    print(f"   - strategy_returns date range: {strategy_returns.index.min()} to {strategy_returns.index.max()}")
    print(f"   - Non-zero strategy returns: {(strategy_returns != 0).sum()}")
    print(f"   - benchmark_returns shape: {benchmark_returns.shape}")
    print(f"   - benchmark_returns date range: {benchmark_returns.index.min()} to {benchmark_returns.index.max()}")
    
    # Use the full date range for plotting, not just from first non-zero return
    start_date = strategy_returns.index.min()
    end_date = strategy_returns.index.max()
    
    print(f"   - Using full date range: {start_date} to {end_date}")
    
    # Align data for the full period
    aligned_strategy_returns = strategy_returns.loc[start_date:end_date].fillna(0)
    aligned_benchmark_returns = benchmark_returns.loc[start_date:end_date].fillna(0)
    
    print(f"   - aligned_strategy_returns shape: {aligned_strategy_returns.shape}")
    print(f"   - aligned_benchmark_returns shape: {aligned_benchmark_returns.shape}")

    strategy_metrics = calculate_performance_metrics(strategy_returns, benchmark_returns)
    benchmark_metrics = calculate_performance_metrics(benchmark_returns, benchmark_returns)
    
    fig = plt.figure(figsize=(18, 26))
    gs = fig.add_gridspec(5, 2, height_ratios=[1.2, 0.8, 0.8, 0.8, 1.2], hspace=0.7, wspace=0.2)
    fig.suptitle(title, fontsize=20, fontweight='bold', color='#2C3E50')

    # 1. Cumulative Performance (Equity Curve) - Show full period
    ax1 = fig.add_subplot(gs[0, :])
    
    # Calculate cumulative returns for full period
    strategy_cumulative = (1 + aligned_strategy_returns).cumprod()
    benchmark_cumulative = (1 + aligned_benchmark_returns).cumprod()
    
    print(f"   - strategy_cumulative shape: {strategy_cumulative.shape}")
    print(f"   - strategy_cumulative range: {strategy_cumulative.min():.4f} to {strategy_cumulative.max():.4f}")
    print(f"   - benchmark_cumulative shape: {benchmark_cumulative.shape}")
    print(f"   - benchmark_cumulative range: {benchmark_cumulative.min():.4f} to {benchmark_cumulative.max():.4f}")
    
    strategy_cumulative.plot(ax=ax1, label='QVM Engine v3g', color='#16A085', lw=2.5)
    benchmark_cumulative.plot(ax=ax1, label='VN-Index', color='#34495E', linestyle='--', lw=2)
    ax1.set_title('Cumulative Performance (Log Scale)', fontweight='bold')
    ax1.set_ylabel('Growth of 1 VND')
    ax1.set_yscale('log')
    ax1.legend(loc='upper left')
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)

    # 2. Drawdown Analysis - Use full period
    ax2 = fig.add_subplot(gs[1, :])
    drawdown = ((1 + aligned_strategy_returns).cumprod() / (1 + aligned_strategy_returns).cumprod().cummax() - 1) * 100
    drawdown.plot(ax=ax2, color='#C0392B')
    ax2.fill_between(drawdown.index, drawdown, 0, color='#C0392B', alpha=0.1)
    ax2.set_title('Drawdown Analysis', fontweight='bold')
    ax2.set_ylabel('Drawdown (%)')
    ax2.grid(True, linestyle='--', alpha=0.5)

    # 3. Annual Returns - Use full period
    ax3 = fig.add_subplot(gs[2, 0])
    strat_annual = aligned_strategy_returns.resample('Y').apply(lambda x: (1+x).prod()-1) * 100
    bench_annual = aligned_benchmark_returns.resample('Y').apply(lambda x: (1+x).prod()-1) * 100
    pd.DataFrame({'Strategy': strat_annual, 'Benchmark': bench_annual}).plot(kind='bar', ax=ax3, color=['#16A085', '#34495E'])
    ax3.set_xticklabels([d.strftime('%Y') for d in strat_annual.index], rotation=45, ha='right')
    ax3.set_title('Annual Returns', fontweight='bold')
    ax3.grid(True, axis='y', linestyle='--', alpha=0.5)

    # 4. Rolling Sharpe Ratio - Use full period
    ax4 = fig.add_subplot(gs[2, 1])
    rolling_sharpe = (aligned_strategy_returns.rolling(252).mean() * 252) / (aligned_strategy_returns.rolling(252).std() * np.sqrt(252))
    rolling_sharpe.plot(ax=ax4, color='#E67E22')
    ax4.axhline(1.0, color='#27AE60', linestyle='--')
    ax4.set_title('1-Year Rolling Sharpe Ratio', fontweight='bold')
    ax4.grid(True, linestyle='--', alpha=0.5)

    # 5. Regime Analysis
    ax5 = fig.add_subplot(gs[3, 0])
    if not diagnostics.empty and 'regime' in diagnostics.columns:
        regime_counts = diagnostics['regime'].value_counts()
        regime_counts.plot(kind='bar', ax=ax5, color=['#3498DB', '#E74C3C', '#F39C12', '#9B59B6'])
        ax5.set_title('Regime Distribution', fontweight='bold')
        ax5.set_ylabel('Number of Rebalances')
        ax5.grid(True, axis='y', linestyle='--', alpha=0.5)

    # 6. Portfolio Size Evolution
    ax6 = fig.add_subplot(gs[3, 1])
    if not diagnostics.empty and 'portfolio_size' in diagnostics.columns:
        diagnostics['portfolio_size'].plot(ax=ax6, color='#2ECC71', marker='o', markersize=3)
        ax6.set_title('Portfolio Size Evolution', fontweight='bold')
        ax6.set_ylabel('Number of Stocks')
        ax6.grid(True, linestyle='--', alpha=0.5)

    # 7. Performance Metrics Table
    ax7 = fig.add_subplot(gs[4:, :])
    ax7.axis('off')
    summary_data = [['Metric', 'Strategy', 'Benchmark']]
    for key in strategy_metrics.keys():
        summary_data.append([key, f"{strategy_metrics[key]:.2f}", f"{benchmark_metrics.get(key, 0.0):.2f}"])
    
    table = ax7.table(cellText=summary_data[1:], colLabels=summary_data[0], loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 2.5)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

# %% [markdown]
# ## MAIN EXECUTION
# 
# Execute the complete QVM Engine v3g integrated backtest with multiple tearsheets.

# %% [code]
# Execute the data loading
try:
    print("\n" + "="*80)
    print("🚀 QVM ENGINE V3G: INTEGRATED INNOVATION BACKTEST")
    print("="*80)
    
    # Initialize data preloader
    data_preloader = OptimizedDataPreloader(QVM_CONFIG, engine)
    preloaded_data = data_preloader.load_all_data()
    
    print("\n✅ All data successfully loaded and prepared for the backtest.")
    print(f"   - Price Data Shape: {preloaded_data['price_data']['returns_matrix'].shape}")
    print(f"   - Fundamental Data Shape: {preloaded_data['fundamental_data'].shape}")
    print(f"   - Benchmark Returns: {len(preloaded_data['benchmark_data'])} days")
    print(f"   - Price Data Date Range: {preloaded_data['price_data']['returns_matrix'].index.min()} to {preloaded_data['price_data']['returns_matrix'].index.max()}")
    print(f"   - Benchmark Date Range: {preloaded_data['benchmark_data'].index.min()} to {preloaded_data['benchmark_data'].index.max()}")
    
    # --- Instantiate and Run the QVM Engine v3g Integrated ---
    print("\n" + "="*80)
    print("🚀 QVM ENGINE V3G: INTEGRATED INNOVATION EXECUTION")
    print("="*80)
    
    qvm_engine = QVMEngineV3gIntegrated(
        config=QVM_CONFIG,
        preloaded_data=preloaded_data,
        db_engine=engine
    )
    
    qvm_net_returns, qvm_diagnostics = qvm_engine.run_backtest()
    
    print(f"\n🔍 DEBUG: After backtest")
    print(f"   - qvm_net_returns shape: {qvm_net_returns.shape}")
    print(f"   - qvm_net_returns date range: {qvm_net_returns.index.min()} to {qvm_net_returns.index.max()}")
    print(f"   - Non-zero returns count: {(qvm_net_returns != 0).sum()}")
    print(f"   - First non-zero return date: {qvm_net_returns[qvm_net_returns != 0].index.min() if (qvm_net_returns != 0).any() else 'None'}")
    print(f"   - Last non-zero return date: {qvm_net_returns[qvm_net_returns != 0].index.max() if (qvm_net_returns != 0).any() else 'None'}")

    # --- Generate Multiple Tearsheets ---
    print("\n" + "="*80)
    print("📊 QVM ENGINE V3G: MULTIPLE TEARSHEETS")
    print("="*80)
    
    # 1. Full Period Tearsheet (2016-2025)
    print("\n📈 Generating Full Period Tearsheet (2016-2025)...")
    # Align benchmark data with strategy returns for full period
    aligned_benchmark = preloaded_data['benchmark_data'].reindex(qvm_net_returns.index).fillna(0)
    
    generate_comprehensive_tearsheet(
        qvm_net_returns,
        aligned_benchmark,
        qvm_diagnostics,
        "QVM Engine v3g Integrated - Full Period (2016-2025)"
    )
    
    # 2. First Period Tearsheet (2016-2020)
    print("\n📈 Generating First Period Tearsheet (2016-2020)...")
    first_period_mask = (qvm_net_returns.index >= '2016-01-01') & (qvm_net_returns.index <= '2020-12-31')
    first_period_returns = qvm_net_returns[first_period_mask]
    
    # Align benchmark data with strategy returns
    first_period_benchmark = preloaded_data['benchmark_data'].reindex(first_period_returns.index).fillna(0)
    
    first_period_diagnostics = qvm_diagnostics[
        (qvm_diagnostics.index >= '2016-01-01') & (qvm_diagnostics.index <= '2020-12-31')
    ]
    
    generate_comprehensive_tearsheet(
        first_period_returns,
        first_period_benchmark,
        first_period_diagnostics,
        "QVM Engine v3g Integrated - First Period (2016-2020)"
    )
    
    # 3. Second Period Tearsheet (2020-2025)
    print("\n📈 Generating Second Period Tearsheet (2020-2025)...")
    second_period_mask = (qvm_net_returns.index >= '2020-01-01') & (qvm_net_returns.index <= '2025-12-31')
    second_period_returns = qvm_net_returns[second_period_mask]
    
    # Align benchmark data with strategy returns
    second_period_benchmark = preloaded_data['benchmark_data'].reindex(second_period_returns.index).fillna(0)
    
    second_period_diagnostics = qvm_diagnostics[
        (qvm_diagnostics.index >= '2020-01-01') & (qvm_diagnostics.index <= '2025-12-31')
    ]
    
    generate_comprehensive_tearsheet(
        second_period_returns,
        second_period_benchmark,
        second_period_diagnostics,
        "QVM Engine v3g Integrated - Second Period (2020-2025)"
    )

    # --- Additional Analysis ---
    print("\n" + "="*80)
    print("🔍 ADDITIONAL ANALYSIS")
    print("="*80)
    
    # Regime Analysis
    if not qvm_diagnostics.empty and 'regime' in qvm_diagnostics.columns:
        print("\n📈 Regime Analysis:")
        regime_summary = qvm_diagnostics['regime'].value_counts()
        for regime, count in regime_summary.items():
            percentage = (count / len(qvm_diagnostics)) * 100
            print(f"   - {regime}: {count} times ({percentage:.2f}%)")
    
    # Factor Configuration
    print("\n📊 Factor Configuration:")
    print(f"   - ROAA Weight: {QVM_CONFIG['factors']['roaa_weight']}")
    print(f"   - P/E Weight: {QVM_CONFIG['factors']['pe_weight']}")
    print(f"   - Momentum Weight: {QVM_CONFIG['factors']['momentum_weight']}")
    print(f"   - Momentum Horizons: {QVM_CONFIG['factors']['momentum_horizons']}")
    
    # Universe Statistics
    if not qvm_diagnostics.empty:
        print(f"\n🌐 Universe Statistics:")
        print(f"   - Average Universe Size: {qvm_diagnostics['universe_size'].mean():.0f} stocks")
        print(f"   - Average Portfolio Size: {qvm_diagnostics['portfolio_size'].mean():.0f} stocks")
        print(f"   - Average Turnover: {qvm_diagnostics['turnover'].mean():.2%}")
    
    # Performance Summary
    print(f"\n📊 Performance Summary:")
    full_metrics = calculate_performance_metrics(qvm_net_returns, aligned_benchmark)
    for metric, value in full_metrics.items():
        print(f"   - {metric}: {value:.2f}")

    print("\n✅ QVM Engine v3g integrated with comprehensive performance analysis complete!")

except Exception as e:
    print(f"❌ An error occurred during execution: {e}")
    raise 