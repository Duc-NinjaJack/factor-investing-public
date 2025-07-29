#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check Pickle Structure
======================
Component: Data Investigation
Purpose: Check the structure of backtesting pickle files
Author: Duc Nguyen, Principal Quantitative Strategist
Date Created: January 2025
Status: DATA INVESTIGATION

This script checks the structure of backtesting pickle files to understand
how to extract return series and other data.
"""

import pickle
import pandas as pd
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_pickle_structure(filename):
    """Check the structure of a pickle file."""
    logger.info(f"Checking structure of {filename}...")
    
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        logger.info(f"✅ {filename} loaded successfully")
        logger.info(f"   - Type: {type(data)}")
        logger.info(f"   - Keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
        
        if isinstance(data, dict):
            for key, value in data.items():
                logger.info(f"   - {key}: {type(value)}")
                if isinstance(value, dict):
                    logger.info(f"     - Sub-keys: {list(value.keys())}")
                elif isinstance(value, pd.DataFrame):
                    logger.info(f"     - Shape: {value.shape}")
                    logger.info(f"     - Columns: {list(value.columns)}")
                elif isinstance(value, pd.Series):
                    logger.info(f"     - Length: {len(value)}")
                    logger.info(f"     - Index: {type(value.index)}")
        
        return data
        
    except Exception as e:
        logger.error(f"❌ Error loading {filename}: {e}")
        return None

def main():
    """Main execution function."""
    print("🔍 Checking Pickle File Structures")
    print("=" * 40)
    
    # Check simplified backtesting results
    simplified_data = check_pickle_structure('data/simplified_backtesting_comparison_results.pkl')
    print("✅ Simplified backtesting data structure checked")
    
    print("\n" + "="*40 + "\n")
    
    # Check real data backtesting results
    real_data = check_pickle_structure('data/full_backtesting_real_data_results.pkl')
    
    print("\n" + "="*40 + "\n")
    
    # Detailed analysis
    if simplified_data and real_data:
        print("📊 Detailed Structure Analysis:")
        
        # Simplified data analysis
        if 'backtest_results' in simplified_data:
            print("\nSimplified Backtesting Results:")
            backtest_results = simplified_data['backtest_results']
            if isinstance(backtest_results, dict):
                for key, value in backtest_results.items():
                    print(f"  - {key}: {type(value)}")
                    if isinstance(value, dict):
                        print(f"    - Sub-keys: {list(value.keys())}")
        
        # Real data analysis
        if 'backtest_results' in real_data:
            print("\nReal Data Backtesting Results:")
            backtest_results = real_data['backtest_results']
            if isinstance(backtest_results, dict):
                for key, value in backtest_results.items():
                    print(f"  - {key}: {type(value)}")
                    if isinstance(value, dict):
                        print(f"    - Sub-keys: {list(value.keys())}")

if __name__ == "__main__":
    main()