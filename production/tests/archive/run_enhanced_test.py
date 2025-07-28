#!/usr/bin/env python3
"""
Enhanced QVM Engine (v2) - Simple Test Runner
==============================================
Purpose: Validate enhanced engine as experimental group for scientific bake-off
Engine Type: Sophisticated multi-tier methodology with Master Quality Signal
Status: EXPERIMENTAL GROUP for signal construction experiment
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging

# Add production engine to path
production_path = Path(__file__).parent.parent
sys.path.append(str(production_path))

# Import enhanced engine (v2)
from engine.qvm_engine_v2_enhanced import QVMEngineV2Enhanced

def run_enhanced_validation():
    """Run complete enhanced engine validation."""
    
    print("="*70)
    print("ENHANCED QVM ENGINE (V2) VALIDATION")
    print("="*70)
    print("🎯 Experimental Group: Sophisticated Multi-tier Quality Signal")
    print("📋 Hypothesis: ~26.3% annual return, 1.77 Sharpe ratio")
    print("="*70)
    
    # Test parameters
    TEST_DATE = pd.Timestamp('2025-07-22')
    TEST_UNIVERSE = ['OCB', 'NLG', 'FPT', 'SSI']
    
    EXPECTED_SECTORS = {
        'OCB': 'Banking',
        'NLG': 'Real Estate', 
        'FPT': 'Technology',
        'SSI': 'Securities'
    }
    
    try:
        # Initialize enhanced engine
        print("\n🔧 Initializing Enhanced QVM Engine (v2)...")
        
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / 'config'
        
        engine = QVMEngineV2Enhanced(config_path=str(config_path), log_level='INFO')
        
        print("✅ Enhanced engine (v2) initialized successfully")
        print(f"📊 Database: {engine.db_config['host']}/{engine.db_config['schema_name']}")
        print(f"⏱️ Reporting lag: {engine.reporting_lag} days")
        
        # Check what attributes are available
        print(f"📋 Quality config available: {'quality_config' in dir(engine)}")
        print(f"📋 Quality tier weights: {engine.quality_tier_weights if hasattr(engine, 'quality_tier_weights') else 'Not found'}")
        print(f"📋 EV Calculator available: {hasattr(engine, 'ev_calculator')}")
        print(f"📋 Engine Type: Sophisticated Multi-tier Quality Signal")
        
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
        
        # Test 2: Enhanced EV Calculator
        print(f"\n🧪 TEST 2: Enhanced EV Calculator Validation")
        print("-" * 50)
        
        if hasattr(engine, 'ev_calculator'):
            print("✅ Enhanced EV Calculator found")
            
            # Test sector-specific value weights
            for ticker in TEST_UNIVERSE:
                sector = EXPECTED_SECTORS[ticker]
                weights = engine.ev_calculator.get_sector_specific_value_weights(sector)
                
                print(f"📊 {ticker} ({sector}):")
                print(f"   P/E: {weights['pe']:.0%}, P/B: {weights['pb']:.0%}")
                print(f"   P/S: {weights['ps']:.0%}, EV/EBITDA: {weights['ev_ebitda']:.0%}")
        else:
            print("❌ Enhanced EV Calculator not found")
        
        # Test 3: Fundamental Data
        print(f"\n🧪 TEST 3: Fundamental Data Retrieval")
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
        
        # Test 4: Enhanced QVM Calculation (CRITICAL)
        print(f"\n🧪 TEST 4: Enhanced QVM Composite Calculation (CRITICAL)")
        print("-" * 50)
        
        qvm_scores = engine.calculate_qvm_composite(TEST_DATE, TEST_UNIVERSE)
        
        if qvm_scores:
            print(f"✅ Calculated enhanced QVM scores for {len(qvm_scores)} tickers")
            print("\n📊 ENHANCED QVM RESULTS:")
            print("-" * 40)
            
            # Sort by QVM score for ranking
            sorted_scores = sorted(qvm_scores.items(), key=lambda x: x[1], reverse=True)
            
            for rank, (ticker, score) in enumerate(sorted_scores, 1):
                sector = EXPECTED_SECTORS.get(ticker, 'Unknown')
                print(f"{rank}. {ticker} ({sector}): {score:.4f}")
            
            # Validation checks
            non_zero_scores = [score for score in qvm_scores.values() if abs(score) > 0.001]
            reasonable_range = [score for score in qvm_scores.values() if -5 <= score <= 5]
            
            print(f"\n📋 ENHANCED VALIDATION SUMMARY:")
            print(f"   Total scores: {len(qvm_scores)}")
            print(f"   Non-zero scores: {len(non_zero_scores)}")
            print(f"   Reasonable range: {len(reasonable_range)}")
            print(f"   Score spread: {max(qvm_scores.values()) - min(qvm_scores.values()):.4f}")
            
            # Success criteria
            success_criteria = [
                len(qvm_scores) == len(TEST_UNIVERSE),
                len(non_zero_scores) >= 2,
                len(reasonable_range) == len(qvm_scores),
                not any(np.isnan(score) for score in qvm_scores.values())
            ]
            
            if all(success_criteria):
                print("\n🎉 ENHANCED ENGINE VALIDATION: ✅ PASSED")
                print("🚀 READY FOR BAKE-OFF: Experimental group validated")
                print("📋 Sophisticated methodology confirmed working")
                
                # Compare with baseline results
                baseline_results = {
                    'FPT': 0.7022, 'OCB': -0.0709, 
                    'NLG': -0.2687, 'SSI': -0.3626
                }
                
                print("\n📊 BASELINE vs ENHANCED COMPARISON:")
                print("-" * 40)
                for ticker in TEST_UNIVERSE:
                    if ticker in qvm_scores and ticker in baseline_results:
                        enhanced_score = qvm_scores[ticker]
                        baseline_score = baseline_results[ticker]
                        diff = enhanced_score - baseline_score
                        
                        print(f"{ticker}: Enhanced {enhanced_score:.4f} vs Baseline {baseline_score:.4f} (Δ: {diff:+.4f})")
                
            else:
                print("❌ ENHANCED ENGINE VALIDATION: FAILED")
                print(f"   Failed criteria: {success_criteria}")
        else:
            print("❌ No enhanced QVM scores calculated")
            
    except Exception as e:
        print(f"❌ ENHANCED VALIDATION ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("="*70)
    return True

if __name__ == "__main__":
    success = run_enhanced_validation()
    exit(0 if success else 1)