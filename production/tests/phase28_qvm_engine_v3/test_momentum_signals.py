#!/usr/bin/env python3
"""
Test Momentum Signals Script
============================

This script tests the momentum calculation with correct signal directions:
- 3M and 6M: Positive signals (higher is better)
- 1M and 12M: Contrarian signals (lower is better)
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
current_path = Path.cwd()
while not (current_path / 'production').is_dir():
    if current_path.parent == current_path:
        raise FileNotFoundError("Could not find the 'production' directory.")
    current_path = current_path.parent

project_root = current_path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def calculate_momentum_score(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate multi-horizon momentum score with correct signal directions."""
    momentum_columns = [col for col in data.columns if col.startswith('momentum_')]
    
    if not momentum_columns:
        return data
    
    # Apply correct signal directions:
    # - 3M and 6M: Positive signals (higher is better)
    # - 1M and 12M: Contrarian signals (lower is better)
    momentum_score = 0.0
    
    for col in momentum_columns:
        if 'momentum_63d' in col or 'momentum_126d' in col:  # 3M and 6M - positive
            momentum_score += data[col]
        elif 'momentum_21d' in col or 'momentum_252d' in col:  # 1M and 12M - contrarian
            momentum_score -= data[col]  # Negative for contrarian
    
    # Equal weight the components
    data['momentum_score'] = momentum_score / len(momentum_columns)
    return data

def test_momentum_signals():
    """Test momentum signal calculation with sample data."""
    print("🔍 Testing Momentum Signal Directions")
    print("=" * 50)
    
    # Create sample data with different momentum scenarios
    test_data = pd.DataFrame({
        'ticker': ['STOCK_A', 'STOCK_B', 'STOCK_C', 'STOCK_D'],
        'momentum_21d': [0.05, -0.02, 0.03, -0.01],    # 1M - contrarian (lower is better)
        'momentum_63d': [0.10, 0.15, -0.05, 0.08],     # 3M - positive (higher is better)
        'momentum_126d': [0.20, 0.12, 0.18, -0.03],    # 6M - positive (higher is better)
        'momentum_252d': [0.30, 0.25, 0.22, 0.35]      # 12M - contrarian (lower is better)
    })
    
    print("\n📊 Sample Momentum Data:")
    print(test_data.to_string(index=False))
    
    # Calculate momentum scores
    result = calculate_momentum_score(test_data.copy())
    
    print("\n📈 Momentum Score Calculation:")
    print("-" * 50)
    
    for _, row in result.iterrows():
        ticker = row['ticker']
        momentum_21d = row['momentum_21d']
        momentum_63d = row['momentum_63d']
        momentum_126d = row['momentum_126d']
        momentum_252d = row['momentum_252d']
        momentum_score = row['momentum_score']
        
        # Calculate manually to verify
        manual_score = (
            -momentum_21d +    # 1M contrarian (negative)
            momentum_63d +     # 3M positive
            momentum_126d +    # 6M positive
            -momentum_252d     # 12M contrarian (negative)
        ) / 4
        
        print(f"\n🎯 {ticker}:")
        print(f"   1M (contrarian): {momentum_21d:.3f} → -{momentum_21d:.3f}")
        print(f"   3M (positive):   {momentum_63d:.3f} → +{momentum_63d:.3f}")
        print(f"   6M (positive):   {momentum_126d:.3f} → +{momentum_126d:.3f}")
        print(f"   12M (contrarian): {momentum_252d:.3f} → -{momentum_252d:.3f}")
        print(f"   Final Score: {momentum_score:.3f} (Manual: {manual_score:.3f})")
        
        # Verify calculation
        assert abs(momentum_score - manual_score) < 1e-10, f"Calculation mismatch for {ticker}"
    
    print(f"\n✅ Momentum signal test completed!")
    print(f"   - 3M and 6M: Positive signals (higher momentum = higher score)")
    print(f"   - 1M and 12M: Contrarian signals (lower momentum = higher score)")
    print(f"   - All calculations verified manually")

if __name__ == "__main__":
    test_momentum_signals() 