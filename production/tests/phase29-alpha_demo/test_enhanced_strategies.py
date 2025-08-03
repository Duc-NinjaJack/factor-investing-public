# %% [markdown]
# # Enhanced Strategies Test Script
#
# **Objective:** Test individual enhancement strategies to ensure they can be imported and initialized properly.
#
# **File:** test_enhanced_strategies.py

# %%
import pandas as pd
import numpy as np
import sys
import importlib.util
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# %%
# Strategy mapping for testing
STRATEGY_FILES = {
    'Integrated_Baseline': '04_integrated_strategy.py',
    'Dynamic_Factor_Weights': '06_dynamic_factor_weights.py',
    'Enhanced_Factor_Integration': '07_enhanced_factor_integration.py',
    'Adaptive_Rebalancing': '08_adaptive_rebalancing.py',
    'Risk_Parity_Enhancement': '09_risk_parity_enhancement.py'
}

STRATEGY_CLASSES = {
    'Integrated_Baseline': 'QVMEngineV3jIntegrated',
    'Dynamic_Factor_Weights': 'QVMEngineV3jDynamicWeights',
    'Enhanced_Factor_Integration': 'QVMEngineV3jEnhancedFactors',
    'Adaptive_Rebalancing': 'QVMEngineV3jAdaptiveRebalancing',
    'Risk_Parity_Enhancement': 'QVMEngineV3jRiskParity'
}

print("🧪 Enhanced Strategies Test Script")
print("   - Testing strategy imports and initialization")
print("   - Verifying class definitions and configurations")

# %%
def test_strategy_import(strategy_name: str, strategy_file: str, class_name: str):
    """Test importing a strategy class."""
    try:
        print(f"\n📋 Testing {strategy_name}...")
        
        # Check if file exists
        file_path = Path(__file__).parent / strategy_file
        if not file_path.exists():
            print(f"❌ File not found: {strategy_file}")
            return False
        
        # Import strategy class
        spec = importlib.util.spec_from_file_location("strategy_module", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Check if class exists
        if not hasattr(module, class_name):
            print(f"❌ Class {class_name} not found in {strategy_file}")
            return False
        
        strategy_class = getattr(module, class_name)
        print(f"✅ Successfully imported {class_name} from {strategy_file}")
        
        # Test class initialization (without running backtest)
        print(f"   - Testing class definition and methods...")
        
        # Check if class has required methods
        required_methods = ['__init__', 'run_backtest', 'generate_comprehensive_tearsheet']
        for method in required_methods:
            if not hasattr(strategy_class, method):
                print(f"   ⚠️ Missing method: {method}")
            else:
                print(f"   ✅ Method found: {method}")
        
        # Test configuration loading
        if hasattr(module, 'QVM_CONFIG'):
            config = module.QVM_CONFIG
            print(f"   ✅ Configuration loaded: {config.get('strategy_name', 'Unknown')}")
        else:
            print(f"   ⚠️ No QVM_CONFIG found")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing {strategy_name}: {e}")
        return False

# %%
def test_all_strategies():
    """Test all enhancement strategies."""
    print("\n🚀 Starting strategy tests...")
    
    results = {}
    
    for strategy_name, strategy_file in STRATEGY_FILES.items():
        class_name = STRATEGY_CLASSES[strategy_name]
        success = test_strategy_import(strategy_name, strategy_file, class_name)
        results[strategy_name] = success
    
    # Summary
    print("\n" + "="*60)
    print("STRATEGY TEST RESULTS SUMMARY")
    print("="*60)
    
    successful = sum(results.values())
    total = len(results)
    
    for strategy_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{strategy_name:.<30} {status}")
    
    print(f"\nOverall: {successful}/{total} strategies passed")
    
    if successful == total:
        print("🎉 All strategies ready for comparison!")
    else:
        print("⚠️ Some strategies need attention before comparison.")
    
    return results

# %%
def create_mock_data():
    """Create mock data for testing."""
    print("\n📊 Creating mock data for testing...")
    
    # Mock price data
    dates = pd.date_range('2016-01-01', '2025-12-31', freq='D')
    tickers = ['TICKER1', 'TICKER2', 'TICKER3', 'TICKER4', 'TICKER5']
    
    price_data = pd.DataFrame(
        np.random.randn(len(dates), len(tickers)) * 0.02 + 1.0,
        index=dates,
        columns=tickers
    ).cumprod() * 100
    
    # Mock fundamental data
    fundamental_data = pd.DataFrame({
        'ticker': tickers * 10,
        'year': [2016 + i//len(tickers) for i in range(50)],
        'quarter': [1 + (i % 4) for i in range(50)],
        'roaa': np.random.randn(50) * 0.1 + 0.05,
        'pe_ratio': np.random.randn(50) * 10 + 15
    })
    
    # Mock returns matrix
    returns_matrix = price_data.pct_change().dropna()
    
    # Mock benchmark returns
    benchmark_returns = pd.Series(
        np.random.randn(len(returns_matrix)) * 0.015,
        index=returns_matrix.index
    )
    
    # Mock precomputed data
    precomputed_data = {
        'universe_rankings': pd.DataFrame({
            'TICKER1': [1, 1, 1],
            'TICKER2': [2, 2, 2],
            'TICKER3': [3, 3, 3]
        }, index=pd.date_range('2016-01-01', periods=3)),
        'fundamental_factors': pd.DataFrame({
            'TICKER1_roaa': [0.05, 0.06, 0.07],
            'TICKER1_pe_ratio': [15, 16, 17],
            'TICKER2_roaa': [0.04, 0.05, 0.06],
            'TICKER2_pe_ratio': [18, 19, 20]
        }, index=pd.date_range('2016-01-01', periods=3)),
        'momentum_factors': pd.DataFrame({
            'TICKER1_momentum_score': [0.1, 0.2, 0.3],
            'TICKER2_momentum_score': [0.2, 0.3, 0.4]
        }, index=pd.date_range('2016-01-01', periods=3))
    }
    
    print("✅ Mock data created successfully")
    return price_data, fundamental_data, returns_matrix, benchmark_returns, precomputed_data

# %%
def test_strategy_initialization():
    """Test strategy initialization with mock data."""
    print("\n🔧 Testing strategy initialization...")
    
    # Create mock data
    price_data, fundamental_data, returns_matrix, benchmark_returns, precomputed_data = create_mock_data()
    
    # Mock database engine
    class MockDBEngine:
        def __init__(self):
            pass
    
    db_engine = MockDBEngine()
    
    # Test each strategy
    for strategy_name, strategy_file in STRATEGY_FILES.items():
        try:
            print(f"\n📋 Testing initialization: {strategy_name}")
            
            # Import strategy
            file_path = Path(__file__).parent / strategy_file
            spec = importlib.util.spec_from_file_location("strategy_module", file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            class_name = STRATEGY_CLASSES[strategy_name]
            strategy_class = getattr(module, class_name)
            
            # Create mock config with strategy-specific configurations
            config = {
                "strategy_name": f"QVM_Engine_v3j_{strategy_name}",
                "universe": {"top_n_stocks": 200, "target_portfolio_size": 20},
                "transaction_costs": {"commission": 0.003},
                "regime_detection": {
                    "volatility_threshold": 0.20,
                    "correlation_threshold": 0.70,
                    "momentum_threshold": 0.05,
                    "stress_threshold": 0.30,
                },
                "factors": {"momentum_horizons": [21, 63, 126, 252]},
                "backtest_start_date": "2016-01-01",
                "backtest_end_date": "2025-12-31"
            }
            
            # Add strategy-specific configurations
            if strategy_name == 'Dynamic_Factor_Weights':
                config['dynamic_weights'] = {
                    "bull_market": {"roaa_weight": 0.25, "pe_weight": 0.20, "momentum_weight": 0.45, "low_vol_weight": 0.10},
                    "bear_market": {"roaa_weight": 0.30, "pe_weight": 0.25, "momentum_weight": 0.15, "low_vol_weight": 0.30},
                    "sideways_market": {"roaa_weight": 0.30, "pe_weight": 0.30, "momentum_weight": 0.25, "low_vol_weight": 0.15},
                    "stress_market": {"roaa_weight": 0.25, "pe_weight": 0.20, "momentum_weight": 0.10, "low_vol_weight": 0.45}
                }
            elif strategy_name == 'Enhanced_Factor_Integration':
                config['enhanced_factors'] = {
                    "core_factors": {"roaa_weight": 0.25, "pe_weight": 0.25, "momentum_weight": 0.30},
                    "additional_factors": {"low_vol_weight": 0.15, "piotroski_weight": 0.15, "fcf_yield_weight": 0.15}
                }
            elif strategy_name == 'Adaptive_Rebalancing':
                config['adaptive_rebalancing'] = {
                    "bull_market": {"rebalancing_frequency": "weekly", "days_between_rebalancing": 7, "regime_allocation": 1.0},
                    "bear_market": {"rebalancing_frequency": "monthly", "days_between_rebalancing": 30, "regime_allocation": 0.8},
                    "sideways_market": {"rebalancing_frequency": "biweekly", "days_between_rebalancing": 14, "regime_allocation": 0.6},
                    "stress_market": {"rebalancing_frequency": "quarterly", "days_between_rebalancing": 90, "regime_allocation": 0.4}
                }
            elif strategy_name == 'Risk_Parity_Enhancement':
                config['risk_parity'] = {
                    "target_risk_contribution": 0.25,
                    "risk_lookback_period": 252,
                    "min_factor_weight": 0.05,
                    "max_factor_weight": 0.50,
                    "risk_measure": "volatility",
                    "optimization_method": "equal_risk_contribution"
                }
            
            # Initialize strategy
            strategy_instance = strategy_class(
                config, price_data, fundamental_data,
                returns_matrix, benchmark_returns, 
                db_engine, precomputed_data
            )
            
            print(f"✅ {strategy_name} initialized successfully")
            
        except Exception as e:
            print(f"❌ Error initializing {strategy_name}: {e}")

# %%
# Main execution
if __name__ == "__main__":
    print("🧪 Enhanced Strategies Test Script")
    print("="*50)
    
    # Test imports
    results = test_all_strategies()
    
    # Test initialization if imports are successful
    if sum(results.values()) == len(results):
        print("\n" + "="*50)
        test_strategy_initialization()
    
    print("\n✅ Test script completed!") 