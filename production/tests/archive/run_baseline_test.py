#!/usr/bin/env python3
"""
Baseline QVM Engine (v1) - Simple Test Runner
==============================================
Purpose: Validate baseline engine as control group for scientific bake-off
Engine Type: Simple ROAE-based Quality Signal implementation
Status: CONTROL GROUP for signal construction experiment
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging

# Add production engine to path
production_path = Path(__file__).parent.parent
sys.path.append(str(production_path))

# Import baseline engine (v1)
from engine.qvm_engine_v1_baseline import QVMEngineV1Baseline

def run_baseline_validation():
    """Run complete baseline engine validation."""
    
    print("="*70)
    print("BASELINE QVM ENGINE (V1) VALIDATION")
    print("="*70)
    print("🎯 Control Group: Simple ROAE-based Quality Signal")
    print("📋 Hypothesis: ~18% annual return, 1.2 Sharpe ratio")
    print("="*70)
    
    # Test parameters
    TEST_DATE = pd.Timestamp('2025-03-31')
    TEST_UNIVERSE = ['OCB', 'NLG', 'FPT', 'SSI']
    
    EXPECTED_SECTORS = {
        'OCB': 'Banking',
        'NLG': 'Real Estate', 
        'FPT': 'Technology',
        'SSI': 'Securities'
    }
    
    try:
        # Initialize baseline engine
        print("\n🔧 Initializing Baseline QVM Engine (v1)...")
        
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / 'config'
        
        engine = QVMEngineV1Baseline(config_path=str(config_path), log_level='INFO')
        
        print("✅ Baseline engine (v1) initialized successfully")
        print(f"📊 Database: {engine.db_config['host']}/{engine.db_config['schema_name']}")
        print(f"⏱️ Reporting lag: {engine.reporting_lag} days")
        
        # Test 1: Sector Mapping
        print(f"\n🧪 TEST 1: Sector Mapping Validation")
        print("-" * 50)
        
        sector_map = engine.get_sector_mapping()
        test_sectors = sector_map[sector_map['ticker'].isin(TEST_UNIVERSE)]
        
        print(f"Retrieved sectors for test universe:")
        for _, row in test_sectors.iterrows():
            ticker = row['ticker']
            sector = row['sector']
            expected = EXPECTED_SECTORS[ticker]
            status = "✅" if sector == expected else "❌"
            print(f"{status} {ticker}: {sector} (expected: {expected})")
        
        # Test 2: Fundamental Data
        print(f"\n🧪 TEST 2: Fundamental Data Retrieval")
        print("-" * 50)
        
        fundamentals = engine.get_fundamentals_correct_timing(TEST_DATE, TEST_UNIVERSE)
        
        if not fundamentals.empty:
            print(f"✅ Retrieved {len(fundamentals)} fundamental records")
            
            for ticker in TEST_UNIVERSE:
                ticker_data = fundamentals[fundamentals['ticker'] == ticker]
                if not ticker_data.empty:
                    row = ticker_data.iloc[0]
                    sector = row.get('sector', 'Unknown')
                    net_profit = row.get('NetProfit_TTM', 0)
                    total_equity = row.get('AvgTotalEquity', 0)
                    
                    print(f"📊 {ticker} ({sector}):")
                    print(f"   NetProfit_TTM: {net_profit:,.0f}")
                    print(f"   AvgTotalEquity: {total_equity:,.0f}")
                else:
                    print(f"⚠️ {ticker}: No fundamental data")
        else:
            print("❌ No fundamental data retrieved")
        
        # Test 3: Market Data
        print(f"\n🧪 TEST 3: Market Data Retrieval")
        print("-" * 50)
        
        market_data = engine.get_market_data(TEST_DATE, TEST_UNIVERSE)
        
        if not market_data.empty:
            print(f"✅ Retrieved market data for {len(market_data)} tickers")
            
            for _, row in market_data.iterrows():
                ticker = row['ticker']
                market_cap = row.get('market_cap', 0)
                adj_close = row.get('adj_close', 0)
                
                print(f"📈 {ticker}: Market Cap: {market_cap:,.0f}, Close: {adj_close:.2f}")
        else:
            print("❌ No market data retrieved")
        
        # Test 4: QVM Calculation (CRITICAL)
        print(f"\n🧪 TEST 4: QVM Composite Calculation (CRITICAL)")
        print("-" * 50)
        
        qvm_scores = engine.calculate_qvm_composite(TEST_DATE, TEST_UNIVERSE)
        
        if qvm_scores:
            print(f"✅ Calculated QVM scores for {len(qvm_scores)} tickers")
            print("\n📊 BASELINE QVM RESULTS:")
            print("-" * 40)
            
            # Sort by QVM score for ranking
            sorted_scores = sorted(qvm_scores.items(), key=lambda x: x[1], reverse=True)
            
            for rank, (ticker, score) in enumerate(sorted_scores, 1):
                sector = EXPECTED_SECTORS.get(ticker, 'Unknown')
                print(f"{rank}. {ticker} ({sector}): {score:.4f}")
            
            # Validation checks
            non_zero_scores = [score for score in qvm_scores.values() if abs(score) > 0.001]
            reasonable_range = [score for score in qvm_scores.values() if -5 <= score <= 5]
            
            print(f"\n📋 VALIDATION SUMMARY:")
            print(f"   Total scores: {len(qvm_scores)}")
            print(f"   Non-zero scores: {len(non_zero_scores)}")
            print(f"   Reasonable range: {len(reasonable_range)}")
            
            # Success criteria
            success_criteria = [
                len(qvm_scores) == len(TEST_UNIVERSE),
                len(non_zero_scores) >= 2,
                len(reasonable_range) == len(qvm_scores),
                not any(np.isnan(score) for score in qvm_scores.values())
            ]
            
            if all(success_criteria):
                print("\n🎉 BASELINE ENGINE VALIDATION: ✅ PASSED")
                print("🚀 READY FOR BAKE-OFF: Control group validated")
                print("📋 Simple ROAE-based approach confirmed working")
            else:
                print("❌ BASELINE ENGINE VALIDATION: FAILED")
                print(f"   Failed criteria: {success_criteria}")
        else:
            print("❌ No QVM scores calculated")
            
    except Exception as e:
        print(f"❌ VALIDATION ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("="*70)
    return True

if __name__ == "__main__":
    success = run_baseline_validation()
    exit(0 if success else 1)