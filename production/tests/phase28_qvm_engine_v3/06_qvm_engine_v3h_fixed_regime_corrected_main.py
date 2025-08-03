#!/usr/bin/env python3
"""
QVM Engine v3h Fixed Regime - CORRECTED MAIN EXECUTION

This file contains the main execution code for the corrected QVM Engine v3h
with proper regime detection thresholds.
"""

# Execute the data loading
try:
    price_data_raw, fundamental_data_raw, daily_returns_matrix, benchmark_returns = load_all_data_for_backtest(QVM_CONFIG, engine)
    print("\n✅ All data successfully loaded and prepared for the backtest.")
    print(f"   - Price Data Shape: {price_data_raw.shape}")
    print(f"   - Fundamental Data Shape: {fundamental_data_raw.shape}")
    print(f"   - Returns Matrix Shape: {daily_returns_matrix.shape}")
    print(f"   - Benchmark Returns: {len(benchmark_returns)} days")
    
    # --- Instantiate and Run the QVM Engine v3h Fixed Regime ---
    print("\n" + "="*80)
    print("🚀 QVM ENGINE V3H: FIXED REGIME (CORRECTED)")
    print("="*80)
    
    qvm_engine = QVMEngineV3fTop200Universe(
        config=QVM_CONFIG,
        price_data=price_data_raw,
        fundamental_data=fundamental_data_raw,
        returns_matrix=daily_returns_matrix,
        benchmark_returns=benchmark_returns,
        db_engine=engine
    )
    
    qvm_net_returns, qvm_diagnostics = qvm_engine.run_backtest()
    
    print(f"\n🔍 DEBUG: After backtest")
    print(f"   - qvm_net_returns shape: {qvm_net_returns.shape}")
    print(f"   - qvm_net_returns date range: {qvm_net_returns.index.min()} to {qvm_net_returns.index.max()}")
    print(f"   - benchmark_returns shape: {benchmark_returns.shape}")
    print(f"   - benchmark_returns date range: {benchmark_returns.index.min()} to {benchmark_returns.index.max()}")
    print(f"   - Non-zero returns count: {(qvm_net_returns != 0).sum()}")
    print(f"   - First non-zero return date: {qvm_net_returns[qvm_net_returns != 0].index.min() if (qvm_net_returns != 0).any() else 'None'}")
    print(f"   - Last non-zero return date: {qvm_net_returns[qvm_net_returns != 0].index.max() if (qvm_net_returns != 0).any() else 'None'}")
    
    # --- Generate Multiple Tearsheets ---
    print("\n" + "="*80)
    print("📊 QVM ENGINE V3H: MULTIPLE TEARSHEETS")
    print("="*80)
    
    # 1. Full Period Tearsheet (2016-2025)
    print("\n📈 Generating Full Period Tearsheet (2016-2025)...")
    generate_comprehensive_tearsheet(
        qvm_net_returns,
        benchmark_returns,
        qvm_diagnostics,
        "QVM Engine v3h Fixed Regime (CORRECTED) - Full Period (2016-2025)"
    )
    
    # 2. First Period Tearsheet (2016-2020)
    print("\n📈 Generating First Period Tearsheet (2016-2020)...")
    first_period_mask = (qvm_net_returns.index >= '2016-01-01') & (qvm_net_returns.index <= '2020-12-31')
    first_period_returns = qvm_net_returns[first_period_mask]
    
    # Align benchmark data with strategy returns for first period
    first_period_benchmark = benchmark_returns.reindex(first_period_returns.index).fillna(0)
    
    first_period_diagnostics = qvm_diagnostics[
        (qvm_diagnostics.index >= '2016-01-01') & (qvm_diagnostics.index <= '2020-12-31')
    ]
    
    generate_comprehensive_tearsheet(
        first_period_returns,
        first_period_benchmark,
        first_period_diagnostics,
        "QVM Engine v3h Fixed Regime (CORRECTED) - First Period (2016-2020)"
    )
    
    # 3. Second Period Tearsheet (2020-2025)
    print("\n📈 Generating Second Period Tearsheet (2020-2025)...")
    second_period_mask = (qvm_net_returns.index >= '2020-01-01') & (qvm_net_returns.index <= '2025-12-31')
    second_period_returns = qvm_net_returns[second_period_mask]
    
    # Align benchmark data with strategy returns for second period
    second_period_benchmark = benchmark_returns.reindex(second_period_returns.index).fillna(0)
    
    second_period_diagnostics = qvm_diagnostics[
        (qvm_diagnostics.index >= '2020-01-01') & (qvm_diagnostics.index <= '2025-12-31')
    ]
    
    generate_comprehensive_tearsheet(
        second_period_returns,
        second_period_benchmark,
        second_period_diagnostics,
        "QVM Engine v3h Fixed Regime (CORRECTED) - Second Period (2020-2025)"
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
    
    print("\n✅ QVM Engine v3h Fixed Regime (CORRECTED) with comprehensive performance analysis complete!")
    
except Exception as e:
    print(f"❌ An error occurred during execution: {e}")
    raise 