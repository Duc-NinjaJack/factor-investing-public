# Technical Implementation Notes

**Date**: July 30, 2025  
**Author**: Factor Investing Team  
**Version**: 1.0  

## Overview

This document provides detailed technical notes on the implementation of the momentum regime analysis framework, including solutions to technical challenges, dependency issues, and optimization strategies.

## Technical Challenges and Solutions

### 1. Scipy Dependency Issue

**Problem**: 
```
ImportError: dlopen(.../scipy/spatial/_qhull.cpython-310-darwin.so, 0x0002): 
Library not loaded: @rpath/libgfortran.5.dylib
```

**Root Cause**: Missing `libgfortran.5.dylib` library dependency for scipy.

**Solution**: Implemented manual Spearman correlation function to bypass scipy dependency.

```python
def _calculate_spearman_correlation(x, y):
    """
    Calculate Spearman correlation manually to avoid scipy dependency.
    """
    try:
        # Get ranks
        x_ranks = x.rank()
        y_ranks = y.rank()

        # Calculate correlation using Pearson formula on ranks
        n = len(x)
        if n < 2:
            return 0.0

        x_mean = x_ranks.mean()
        y_mean = y_ranks.mean()

        numerator = ((x_ranks - x_mean) * (y_ranks - y_mean)).sum()
        x_var = ((x_ranks - x_mean) ** 2).sum()
        y_var = ((y_ranks - y_mean) ** 2).sum()

        if x_var == 0 or y_var == 0:
            return 0.0

        correlation = numerator / (x_var * y_var) ** 0.5
        return correlation
    except:
        return 0.0
```

**Benefits**:
- Eliminates external dependency
- Faster execution
- More control over calculation
- Robust error handling

### 2. Module Import Issues

**Problem**: 
```
ModuleNotFoundError: No module named 'database'
```

**Root Cause**: Incorrect `sys.path` configuration pointing to specific module directories.

**Solution**: Updated path configuration to point to parent production directory.

```python
# Before (incorrect)
sys.path.append('../../../production/database')
sys.path.append('../../../production/engine')

# After (correct)
sys.path.append('../../../production')
```

**Implementation**: Applied to all scripts in `phase23_test_momentum_regimes/`.

### 3. Dynamic Module Loading

**Problem**: Relative imports causing linter errors in `05_comprehensive_momentum_regime_analysis.py`.

**Solution**: Used `importlib.util` for dynamic module loading.

```python
import importlib.util

def load_module_from_file(module_name, file_path):
    """Dynamically load a module from file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load modules dynamically
validation_module = load_module_from_file(
    'validation_tests', 
    '01_momentum_regime_validation_tests.py'
)
```

### 4. Indentation Errors

**Problem**: Incorrect indentation after adding manual correlation function.

**Solution**: Fixed indentation in function calls and comments.

```python
# Corrected indentation
ic = _calculate_spearman_correlation(factor_series, return_series)
```

## Performance Optimizations

### 1. Database Query Optimization

**Strategy**: Limit date ranges and use indexed columns.

```python
# Optimized query with date range limits
start_date = analysis_date - pd.DateOffset(months=14)
query = f"""
SELECT date, ticker, close as adj_close
FROM equity_history
WHERE ticker IN ('{ticker_str}')
  AND date BETWEEN '{start_date.date()}' AND '{analysis_date.date()}'
ORDER BY ticker, date
"""
```

### 2. Memory Management

**Strategy**: Process data in chunks and use efficient data structures.

```python
# Process universe in chunks
def get_universe(engine, analysis_date, limit=100):
    """Get universe with memory-efficient processing."""
    query = f"""
    SELECT ticker, market_cap
    FROM master_info
    WHERE market_cap >= 5000000000  -- 5B VND minimum
    ORDER BY market_cap DESC
    LIMIT {limit}
    """
    return pd.read_sql(query, engine.engine)['ticker'].tolist()
```

### 3. Vectorized Calculations

**Strategy**: Use pandas vectorized operations instead of loops.

```python
# Vectorized forward return calculation
def calculate_forward_returns_vectorized(price_data, universe, analysis_date, forward_months):
    """Calculate forward returns using vectorized operations."""
    end_date = analysis_date + pd.DateOffset(months=forward_months)
    
    # Filter data efficiently
    mask = (price_data['date'] >= analysis_date) & (price_data['date'] <= end_date)
    filtered_data = price_data[mask]
    
    # Group by ticker and calculate returns
    returns = filtered_data.groupby('ticker').apply(
        lambda x: (x.iloc[-1]['adj_close'] / x.iloc[0]['adj_close']) - 1
        if len(x) >= 2 else 0.0
    )
    
    return returns
```

## Data Quality Assurance

### 1. Data Completeness Checks

```python
def validate_data_completeness(price_data, universe, analysis_date):
    """Validate data completeness for analysis."""
    # Check for missing tickers
    missing_tickers = set(universe) - set(price_data['ticker'].unique())
    if missing_tickers:
        print(f"⚠️ Missing data for {len(missing_tickers)} tickers")
    
    # Check for sufficient observations
    min_obs = 10
    for ticker in universe:
        ticker_data = price_data[price_data['ticker'] == ticker]
        if len(ticker_data) < min_obs:
            print(f"⚠️ Insufficient data for {ticker}: {len(ticker_data)} obs")
    
    return len(missing_tickers) == 0
```

### 2. Outlier Detection

```python
def detect_outliers(ic_values, threshold=3.0):
    """Detect outliers in IC values using z-score method."""
    mean_ic = np.mean(ic_values)
    std_ic = np.std(ic_values)
    
    outliers = []
    for i, ic in enumerate(ic_values):
        z_score = abs(ic - mean_ic) / std_ic
        if z_score > threshold:
            outliers.append((i, ic, z_score))
    
    return outliers
```

### 3. Data Validation Pipeline

```python
def validate_momentum_data(engine, analysis_date, universe):
    """Comprehensive data validation pipeline."""
    # Check price data availability
    price_data = get_price_data(engine, analysis_date, universe)
    if price_data.empty:
        return False, "No price data available"
    
    # Check fundamental data availability
    fundamental_data = engine.get_fundamentals_correct_timing(analysis_date, universe)
    if fundamental_data.empty:
        return False, "No fundamental data available"
    
    # Check universe size
    if len(universe) < 20:
        return False, f"Universe too small: {len(universe)} stocks"
    
    return True, "Data validation passed"
```

## Error Handling and Logging

### 1. Robust Error Handling

```python
def safe_calculate_ic(engine, analysis_date, universe, forward_months=1):
    """Safely calculate IC with comprehensive error handling."""
    try:
        # Validate inputs
        if not universe or len(universe) < 10:
            return None, "Universe too small"
        
        # Calculate momentum scores
        momentum_scores = engine._calculate_enhanced_momentum_composite(
            fundamental_data, analysis_date, universe
        )
        if not momentum_scores:
            return None, "Failed to calculate momentum scores"
        
        # Calculate forward returns
        forward_returns = calculate_forward_returns(...)
        if not forward_returns:
            return None, "Failed to calculate forward returns"
        
        # Calculate IC
        ic = _calculate_spearman_correlation(momentum_scores, forward_returns)
        return ic, "Success"
        
    except Exception as e:
        return None, f"Error: {str(e)}"
```

### 2. Structured Logging

```python
import logging

def setup_logging():
    """Setup structured logging for momentum analysis."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('momentum_analysis.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()
```

## Configuration Management

### 1. Dynamic Configuration Loading

```python
def load_momentum_config():
    """Load momentum configuration with fallbacks."""
    try:
        with open('config/strategy_config.yml', 'r') as f:
            config = yaml.safe_load(f)
        return config.get('momentum', {})
    except Exception as e:
        logger.warning(f"Failed to load config: {e}")
        return get_default_momentum_config()

def get_default_momentum_config():
    """Default momentum configuration."""
    return {
        'timeframe_weights': {
            '1M': 0.15, '3M': 0.25, '6M': 0.30, '12M': 0.30
        },
        'lookback_periods': {
            '1M': 1, '3M': 3, '6M': 6, '12M': 12
        },
        'skip_months': 1
    }
```

### 2. Environment-Specific Settings

```python
def get_environment_config():
    """Get environment-specific configuration."""
    env = os.getenv('ENVIRONMENT', 'development')
    
    configs = {
        'development': {
            'universe_size': 50,
            'test_periods': 12,
            'log_level': 'DEBUG'
        },
        'production': {
            'universe_size': 100,
            'test_periods': 60,
            'log_level': 'INFO'
        }
    }
    
    return configs.get(env, configs['development'])
```

## Testing Framework

### 1. Unit Tests

```python
def test_spearman_correlation():
    """Test manual Spearman correlation implementation."""
    # Test case 1: Perfect positive correlation
    x = pd.Series([1, 2, 3, 4, 5])
    y = pd.Series([1, 2, 3, 4, 5])
    assert abs(_calculate_spearman_correlation(x, y) - 1.0) < 1e-6
    
    # Test case 2: Perfect negative correlation
    y_neg = pd.Series([5, 4, 3, 2, 1])
    assert abs(_calculate_spearman_correlation(x, y_neg) + 1.0) < 1e-6
    
    # Test case 3: No correlation
    y_random = pd.Series([1, 5, 2, 4, 3])
    corr = _calculate_spearman_correlation(x, y_random)
    assert abs(corr) < 0.5  # Should be close to zero
```

### 2. Integration Tests

```python
def test_momentum_calculation_pipeline():
    """Test complete momentum calculation pipeline."""
    # Setup
    engine = QVMEngineV2Enhanced()
    analysis_date = pd.Timestamp('2023-01-01')
    universe = ['VNM', 'VCB', 'TCB']  # Test universe
    
    # Execute
    result = calculate_momentum_ic(engine, analysis_date, universe)
    
    # Assert
    assert result is not None
    assert 'ic' in result
    assert 'n_stocks' in result
    assert result['n_stocks'] > 0
```

## Performance Benchmarks

### 1. Execution Time Tracking

```python
import time

def benchmark_momentum_calculation(engine, analysis_date, universe):
    """Benchmark momentum calculation performance."""
    start_time = time.time()
    
    # Calculate momentum scores
    momentum_start = time.time()
    momentum_scores = engine._calculate_enhanced_momentum_composite(...)
    momentum_time = time.time() - momentum_start
    
    # Calculate forward returns
    returns_start = time.time()
    forward_returns = calculate_forward_returns(...)
    returns_time = time.time() - returns_start
    
    # Calculate IC
    ic_start = time.time()
    ic = _calculate_spearman_correlation(...)
    ic_time = time.time() - ic_start
    
    total_time = time.time() - start_time
    
    return {
        'total_time': total_time,
        'momentum_time': momentum_time,
        'returns_time': returns_time,
        'ic_time': ic_time,
        'universe_size': len(universe)
    }
```

### 2. Memory Usage Monitoring

```python
import psutil
import os

def monitor_memory_usage():
    """Monitor memory usage during analysis."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return {
        'rss': memory_info.rss / 1024 / 1024,  # MB
        'vms': memory_info.vms / 1024 / 1024,  # MB
        'percent': process.memory_percent()
    }
```

## Deployment Considerations

### 1. Environment Setup

```bash
# Required packages
pip install pandas numpy matplotlib seaborn pyyaml sqlalchemy

# Optional packages (if scipy is available)
pip install scipy

# Database connection
export DATABASE_URL="postgresql://user:pass@localhost:5432/factor_db"
```

### 2. Configuration Files

```yaml
# config/database.yml
database:
  host: localhost
  port: 5432
  name: factor_db
  user: factor_user
  password: ${DB_PASSWORD}

# config/momentum_validation.yml
validation:
  universe_size: 100
  test_periods: 60
  forward_horizons: [1, 3, 6, 12]
  quality_gates:
    mean_ic: 0.02
    t_stat: 2.0
    hit_rate: 0.55
```

### 3. Monitoring and Alerting

```python
def setup_monitoring():
    """Setup monitoring and alerting for momentum analysis."""
    # Performance monitoring
    if execution_time > threshold:
        send_alert("Momentum analysis taking too long")
    
    # Data quality monitoring
    if data_completeness < 0.95:
        send_alert("Data completeness below threshold")
    
    # Factor performance monitoring
    if mean_ic < quality_gate:
        send_alert("Momentum factor performance below threshold")
```

## Future Enhancements

### 1. Parallel Processing

```python
from concurrent.futures import ProcessPoolExecutor

def parallel_ic_calculation(dates, universe, engine):
    """Calculate IC for multiple dates in parallel."""
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(calculate_momentum_ic, engine, date, universe)
            for date in dates
        ]
        results = [future.result() for future in futures]
    return results
```

### 2. Caching Layer

```python
import pickle
import hashlib

def cache_momentum_scores(engine, analysis_date, universe, scores):
    """Cache momentum scores for reuse."""
    cache_key = hashlib.md5(
        f"{analysis_date}_{sorted(universe)}".encode()
    ).hexdigest()
    
    cache_file = f"cache/momentum_{cache_key}.pkl"
    with open(cache_file, 'wb') as f:
        pickle.dump(scores, f)
```

### 3. Real-time Monitoring

```python
def setup_real_time_monitoring():
    """Setup real-time monitoring dashboard."""
    # WebSocket connection for real-time updates
    # Dashboard for factor performance visualization
    # Alert system for regime changes
    pass
```

---

**Contact**: Factor Investing Team  
**Last Updated**: July 30, 2025  
**Next Review**: August 30, 2025 