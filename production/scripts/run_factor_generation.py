#!/usr/bin/env python3
"""
================================================================================
QVM Engine v2.1.1 Flat - Historical Factor Generation Script
================================================================================
Purpose:
    Execute historical factor generation using the QVM Engine v2.1.1 Flat
    for VN-Index benchmarking and backtesting validation. This script
    orchestrates the generation process with proper error handling,
    logging, and progress tracking.

Engine Architecture:
    QVM v2.1.1 Flat - 4-pillar architecture with Quality (35%), Value (30%), 
    Momentum (20%), and Defensive (15%) composites using flat methodology.

Author: QVM Engine Team
Date: August 17, 2025
Target Performance: >1.0 Sharpe ratio, <35% max drawdown
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from pathlib import Path
import yaml
from tqdm import tqdm
import time
import warnings
import logging
import sys
import argparse
from datetime import datetime, timedelta
import traceback

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("enhanced_qvm_historical_generation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

def import_enhanced_engine(strategy_version='qvm_v2.1.1_flat'):
    """Import the QVM Engine v2.1.1 Flat from production location."""
    try:
        # Add production engine to path
        production_path = Path(__file__).parent.parent
        sys.path.append(str(production_path))
        
        # Check if hotfix version is requested
        if 'hotfix' in strategy_version:
            # Import HOTFIX F-Score engine that bypasses SQL CTE performance issues
            from engine.qvm_engine_v2_1_1_flat_hotfix import QVMEngineV211Flat
            logger.info("✅ Successfully imported QVM Engine v2.1.1 Flat HOTFIX (F-Score optimized)")
        else:
            # Import standard QVM Engine v2.1.1 Flat
            from engine.qvm_engine_v2_1_1_flat import QVMEngineV211Flat
            logger.info("✅ Successfully imported QVM Engine v2.1.1 Flat")
        
        return QVMEngineV211Flat
        
    except Exception as e:
        logger.error(f"❌ Failed to import QVM Engine v2.1.1 Flat: {e}")
        logger.error("Ensure the engine is located at production/engine/qvm_engine_v2_1_1_flat.py")
        sys.exit(1)

def load_database_config() -> dict:
    """Load database configuration from config files."""
    try:
        config_path = Path(__file__).parent.parent.parent / "config"
        
        # Try to load from database_config.yml or similar
        possible_configs = ['database_config.yml', 'config.yml', 'db_config.yml']
        
        for config_file in possible_configs:
            config_file_path = config_path / config_file
            if config_file_path.exists():
                with open(config_file_path, 'r') as f:
                    config = yaml.safe_load(f)
                logger.info(f"✅ Loaded database configuration from {config_file_path}")
                return config
        
        # If no config file found, use default connection info
        logger.warning("⚠️ No database config file found, using default MySQL connection")
        return {
            'host': 'localhost',
            'port': 3306,
            'database': 'alphabeta',
            'user': 'root'
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to load database configuration: {e}")
        sys.exit(1)

def get_trading_dates(engine, start_date: str, end_date: str) -> list:
    """Get all trading dates in the specified range from equity_history."""
    logger.info(f"📅 Fetching trading dates from {start_date} to {end_date}")
    try:
        query = text("""
        SELECT DISTINCT date 
        FROM equity_history 
        WHERE date BETWEEN :start_date AND :end_date 
        AND volume > 0
        ORDER BY date
        """)
        
        dates_df = pd.read_sql(query, engine, params={'start_date': start_date, 'end_date': end_date}, parse_dates=['date'])
        trading_dates = dates_df['date'].dt.strftime('%Y-%m-%d').tolist()
        
        logger.info(f"✅ Found {len(trading_dates)} trading dates")
        logger.info(f"📊 Date range: {trading_dates[0]} to {trading_dates[-1]}")
        
        return trading_dates
        
    except Exception as e:
        logger.error(f"❌ Failed to fetch trading dates: {e}")
        raise

def get_missing_dates(engine, start_date: str, end_date: str, strategy_version: str) -> list:
    """
    VERSION-AWARE: Find trading dates missing for a specific strategy version.
    Critical for incremental mode to avoid regenerating existing data.
    """
    try:
        # Get all trading dates in range
        all_dates_query = text("""
        SELECT DISTINCT date 
        FROM equity_history 
        WHERE date BETWEEN :start_date AND :end_date 
        ORDER BY date
        """)
        
        # Get existing dates for this version
        existing_dates_query = text("""
        SELECT DISTINCT date 
        FROM factor_scores_qvm 
        WHERE date BETWEEN :start_date AND :end_date 
        AND strategy_version = :strategy_version
        ORDER BY date
        """)
        
        with engine.connect() as conn:
            # Get all possible dates
            all_dates_result = conn.execute(all_dates_query, {
                'start_date': start_date, 
                'end_date': end_date
            })
            all_dates = [row[0] for row in all_dates_result]
            
            # Get existing dates for this version
            existing_dates_result = conn.execute(existing_dates_query, {
                'start_date': start_date, 
                'end_date': end_date,
                'strategy_version': strategy_version
            })
            existing_dates = set(row[0] for row in existing_dates_result)
            
        # Find missing dates
        missing_dates = [date for date in all_dates if date not in existing_dates]
        
        logger.info(f"📊 Date analysis for {strategy_version}:")
        logger.info(f"   Total trading dates: {len(all_dates)}")
        logger.info(f"   Existing dates: {len(existing_dates)}")
        logger.info(f"   Missing dates: {len(missing_dates)}")
        
        return missing_dates
        
    except Exception as e:
        logger.error(f"❌ Failed to find missing dates: {e}")
        return []

def clear_existing_factor_scores(engine, start_date: str, end_date: str, strategy_version: str = 'qvm_v2.0_enhanced'):
    """
    CRITICAL FIX: Version-aware clearing - only clears specified strategy version.
    This prevents accidental deletion of other experimental versions.
    """
    logger.info(f"🧹 Clearing existing factor scores for VERSION {strategy_version} from {start_date} to {end_date}")
    try:
        # FIXED: Version-aware DELETE to prevent cross-contamination
        delete_query = text("""
        DELETE FROM factor_scores_qvm 
        WHERE date BETWEEN :start_date AND :end_date 
        AND strategy_version = :strategy_version
        """)
        
        with engine.begin() as conn:
            result = conn.execute(delete_query, {
                'start_date': start_date, 
                'end_date': end_date,
                'strategy_version': strategy_version
            })
            deleted_count = result.rowcount
            
        logger.info(f"✅ Cleared {deleted_count} records for version {strategy_version} (other versions preserved)")
        
    except Exception as e:
        logger.error(f"❌ Failed to clear existing data: {e}")
        raise

def get_universe(engine) -> list:
    """Get the investment universe from master_info."""
    logger.info("📋 Fetching investment universe...")
    try:
        query = text("SELECT DISTINCT ticker FROM master_info WHERE ticker IS NOT NULL ORDER BY ticker")
        universe_df = pd.read_sql(query, engine)
        universe = universe_df['ticker'].tolist()
        logger.info(f"✅ Found {len(universe)} tickers in universe")
        return universe
    except Exception as e:
        logger.error(f"❌ Failed to fetch universe: {e}")
        raise

def batch_insert_factor_scores(engine, factor_scores: list, strategy_version: str = 'qvm_v2.0_enhanced'):
    """
    Insert factor scores into database with complete component breakdown.
    CRITICAL INFRASTRUCTURE FIX: Now handles individual factor components for institutional transparency.
    """
    if not factor_scores:
        return
        
    try:
        # CRITICAL FIX: Updated query to include ALL columns including enhanced factors
        insert_query = text("""
        INSERT INTO factor_scores_qvm (
            ticker, date, Quality_Composite, Value_Composite, Momentum_Composite, Defensive_Composite, QVM_Composite,
            Low_Volatility_63D, Piotroski_F_Score, FCF_Yield,
            calculation_timestamp, strategy_version
        ) 
        VALUES (
            :ticker, :date, :Quality_Composite, :Value_Composite, :Momentum_Composite, :Defensive_Composite, :QVM_Composite,
            :Low_Volatility_63D, :Piotroski_F_Score, :FCF_Yield,
            NOW(), :strategy_version
        )
        ON DUPLICATE KEY UPDATE
            Quality_Composite = VALUES(Quality_Composite),
            Value_Composite = VALUES(Value_Composite),
            Momentum_Composite = VALUES(Momentum_Composite),
            Defensive_Composite = VALUES(Defensive_Composite),
            QVM_Composite = VALUES(QVM_Composite),
            Low_Volatility_63D = VALUES(Low_Volatility_63D),
            Piotroski_F_Score = VALUES(Piotroski_F_Score),
            FCF_Yield = VALUES(FCF_Yield),
            calculation_timestamp = NOW()
        """)
        
        # CRITICAL FIX: Convert factor_scores format to handle component breakdown with version-aware records
        db_records = []
        for record in factor_scores:
            # Handle both old format (single score) and new format (component breakdown)
            if isinstance(record.get('components', {}), dict):
                # New enhanced format with component breakdown
                components = record['components']
                db_records.append({
                    'ticker': record['ticker'],
                    'date': record['date'],
                    # DEFENSIVE ROUNDING: Ensure precision matches DECIMAL(20,10) schema
                    'Quality_Composite': round(components.get('Quality_Composite', 0.0), 10),
                    'Value_Composite': round(components.get('Value_Composite', 0.0), 10),
                    'Momentum_Composite': round(components.get('Momentum_Composite', 0.0), 10),
                    'Defensive_Composite': round(components.get('Defensive_Composite', 0.0), 10),
                    'QVM_Composite': round(components.get('QVM_Composite', 0.0), 10),
                    # ENHANCED FACTORS: Extract from top-level of record (not components)
                    'Low_Volatility_63D': round(record.get('Low_Volatility_63D', 0.0), 6),
                    'Piotroski_F_Score': int(record.get('Piotroski_F_Score', 0)),
                    'FCF_Yield': round(record.get('FCF_Yield', 0.0), 6),
                    'strategy_version': strategy_version
                })
            else:
                # Fallback for old format - all components as 0.0 except QVM_Composite
                logger.warning(f"Record for {record.get('ticker', 'unknown')} missing component breakdown")
                db_records.append({
                    'ticker': record['ticker'],
                    'date': record['date'],
                    # DEFENSIVE ROUNDING: Ensure precision matches DECIMAL(20,10) schema
                    'Quality_Composite': round(0.0, 10),
                    'Value_Composite': round(0.0, 10),
                    'Momentum_Composite': round(0.0, 10),
                    'Defensive_Composite': round(0.0, 10),
                    'QVM_Composite': round(record.get('qvm_score', 0.0), 10),
                    # ENHANCED FACTORS: Default values for old format
                    'Low_Volatility_63D': round(0.0, 6),
                    'Piotroski_F_Score': 0,
                    'FCF_Yield': round(0.0, 6),
                    'strategy_version': strategy_version
                })
        
        # Execute batch insert within a transaction
        with engine.begin() as conn:
            # Use executemany for efficient batch insertion
            conn.execute(insert_query, db_records)
            
        logger.info(f"✅ Inserted {len(factor_scores)} factor score records with component breakdown")
        
    except Exception as e:
        logger.error(f"❌ Failed to insert factor scores batch: {e}")
        logger.error(f"Sample record structure: {factor_scores[0] if factor_scores else 'No records'}")
        raise

def run_historical_generation(start_date: str, end_date: str, strategy_version: str = 'qvm_v2.0_enhanced', 
                             mode: str = 'incremental', batch_size: int = 30):
    """Run historical factor generation for the specified date range."""
    
    print("🚀 ENHANCED QVM ENGINE v2 - VERSION-AWARE FACTOR GENERATION")
    print("=" * 80)
    print(f"📅 Period: {start_date} to {end_date}")
    print(f"🏷️  Version: {strategy_version}")
    print(f"🎯 Mode: {mode.upper()}")
    print(f"🔧 Engine: QVM Engine v2.1.1 Flat (4-pillar with defensive factors)")
    print("=" * 80)
    
    # Initialize QVM Engine v2.1.1 Flat
    logger.info("🔧 Initializing QVM Engine v2.1.1 Flat...")
    
    try:
        QVMEngineV211Flat = import_enhanced_engine(strategy_version)
        
        # Load config
        config_path = Path(__file__).parent.parent.parent / 'config'
        engine = QVMEngineV211Flat(config_path=str(config_path), log_level='INFO')
        
        logger.info("✅ QVM Engine v2.1.1 Flat initialized successfully")
        logger.info(f"    Database: {engine.db_config['host']}/{engine.db_config['schema_name']}")
        logger.info(f"    Reporting lag: {engine.reporting_lag} days")
        
        # AGENT MEMO: Feature flag implementation exactly as specified
        import os, logging, types, concurrent.futures as cf
        from datetime import datetime
        log = logging.getLogger(__name__)

        def _wrap_with_timeout(fn, timeout_s, fallback):
            def wrapped(*args, **kwargs):
                with cf.ThreadPoolExecutor(max_workers=1) as ex:
                    fut = ex.submit(fn, *args, **kwargs)
                    try:
                        return fut.result(timeout=timeout_s)
                    except cf.TimeoutError:
                        log.error("F-Score DB path timed out after %ss; using vectorized fallback.", timeout_s)
                        return fallback(*args, **kwargs)
                    except Exception:
                        log.exception("F-Score DB path error; using vectorized fallback.")
                        return fallback(*args, **kwargs)
            return wrapped

        impl = os.getenv("F_SCORE_IMPL", "vectorized").lower()
        timeout_s = int(os.getenv("F_SCORE_TIMEOUT_S", "30"))

        if impl in ("vectorized","py","python"):
            from engine.qvm_engine_v2_1_1_flat_fscore_vectorized import install_vectorized_fscore, prime_fscore_cache
            install_vectorized_fscore(engine)
            log.info("F-Score implementation: VECTORIZED (default)")
        elif impl in ("db","sql"):
            from engine.qvm_engine_v2_1_1_flat_fscore_vectorized import (
                _compute_nf_vectorized, _compute_bank_vectorized, _compute_sec_vectorized
            )
            # Wrap original DB methods with timeout+fallback
            engine._get_raw_f_score_non_financial = _wrap_with_timeout(
                engine._get_raw_f_score_non_financial, timeout_s,
                types.MethodType(lambda self, tickers, y, q, d: _compute_nf_vectorized(self, tickers, y, q, d), engine)
            )
            engine._get_raw_f_score_banking = _wrap_with_timeout(
                engine._get_raw_f_score_banking, timeout_s,
                types.MethodType(lambda self, tickers, y, q: _compute_bank_vectorized(self, tickers, y, q), engine)
            )
            engine._get_raw_f_score_securities = _wrap_with_timeout(
                engine._get_raw_f_score_securities, timeout_s,
                types.MethodType(lambda self, tickers, y, q: _compute_sec_vectorized(self, tickers, y, q), engine)
            )
            from engine.qvm_engine_v2_1_1_flat_fscore_vectorized import prime_fscore_cache
            log.info("F-Score implementation: DB with auto-fallback (timeout=%ss)", timeout_s)
        else:
            raise SystemExit(f"Unknown F_SCORE_IMPL={impl}")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize QVM Engine v2.1.1 Flat: {e}")
        logger.error(traceback.format_exc())
        return False
    
    # Get database connection from engine
    try:
        db_engine = engine.engine  # Access the SQLAlchemy engine directly
        logger.info("✅ Database connection accessed successfully")
    except Exception as e:
        logger.error(f"❌ Failed to access database connection: {e}")
        return False
    
    # Get investment universe
    try:
        universe = get_universe(db_engine)
        if not universe:
            logger.error("❌ No tickers found in universe")
            return False
    except Exception as e:
        logger.error(f"❌ Failed to get universe: {e}")
        return False
    
    # VERSION-AWARE PROCESSING: Implement incremental vs refresh logic
    logger.info(f"🎯 MODE: {mode.upper()} for version {strategy_version}")
    
    if mode == 'incremental':
        # INCREMENTAL MODE: Only process missing dates
        logger.info("📊 INCREMENTAL MODE: Finding missing dates for version...")
        try:
            trading_dates = get_missing_dates(db_engine, start_date, end_date, strategy_version)
            if not trading_dates:
                logger.info("✅ No missing dates found - version is up to date!")
                return True
        except Exception as e:
            logger.error(f"❌ Failed to find missing dates: {e}")
            return False
            
    elif mode == 'refresh':
        # REFRESH MODE: Clear existing data and regenerate all dates
        logger.info("🔄 REFRESH MODE: Clearing existing data and regenerating...")
        try:
            # Get all trading dates in range
            trading_dates = get_trading_dates(db_engine, start_date, end_date)
            if not trading_dates:
                logger.error("❌ No trading dates found in specified range")
                return False
                
            # Clear existing data for this version only
            clear_existing_factor_scores(db_engine, start_date, end_date, strategy_version)
            
        except Exception as e:
            logger.error(f"❌ Failed to prepare refresh mode: {e}")
            return False
    
    # Process dates in batches
    total_dates = len(trading_dates)
    batch_count = (total_dates + batch_size - 1) // batch_size
    
    logger.info(f"📊 Processing {total_dates} dates in {batch_count} batches of {batch_size}")
    
    all_factor_scores = []
    successful_dates = 0
    failed_dates = 0
    
    for batch_idx in range(batch_count):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_dates)
        batch_dates = trading_dates[start_idx:end_idx]
        
        logger.info(f"📦 Processing batch {batch_idx + 1}/{batch_count}: {len(batch_dates)} dates")
        logger.info(f"    Date range: {batch_dates[0]} to {batch_dates[-1]}")
        
        batch_factor_scores = []
        
        # Process each date in the batch
        for date_str in tqdm(batch_dates, desc=f"Batch {batch_idx + 1}"):
            try:
                # Convert to pandas Timestamp
                analysis_date = pd.Timestamp(date_str)
                
                # Generate factor scores for this date
                factor_scores_dict = engine.calculate_qvm_composite(analysis_date, universe)
                
                if factor_scores_dict and len(factor_scores_dict) > 0:
                    # CRITICAL FIX: Convert enhanced component structure to database records
                    scores_list = []
                    for ticker, components in factor_scores_dict.items():
                        scores_list.append({
                            'ticker': ticker,
                            'components': components,  # Pass full component breakdown
                            'date': analysis_date.date()
                        })
                    
                    batch_factor_scores.extend(scores_list)
                    successful_dates += 1
                    
                else:
                    logger.warning(f"⚠️ No factor scores generated for {date_str}")
                    failed_dates += 1
                
            except Exception as e:
                logger.error(f"❌ Failed to process {date_str}: {e}")
                failed_dates += 1
                continue
        
        # Insert batch to database (version-aware)
        if batch_factor_scores:
            try:
                batch_insert_factor_scores(db_engine, batch_factor_scores, strategy_version)
                all_factor_scores.extend(batch_factor_scores)
                
            except Exception as e:
                logger.error(f"❌ Failed to insert batch {batch_idx + 1}: {e}")
                return False
        
        # Progress update
        progress = ((batch_idx + 1) / batch_count) * 100
        logger.info(f"📈 Progress: {progress:.1f}% | Success: {successful_dates} | Failed: {failed_dates}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("🏆 HISTORICAL GENERATION COMPLETE")
    print("=" * 80)
    print(f"📊 Total dates processed: {total_dates}")
    print(f"✅ Successful dates: {successful_dates}")
    print(f"❌ Failed dates: {failed_dates}")
    print(f"📈 Success rate: {(successful_dates/total_dates)*100:.1f}%")
    print(f"💾 Total factor scores generated: {len(all_factor_scores)}")
    print("=" * 80)
    
    if successful_dates > 0:
        logger.info("✅ Historical generation completed successfully")
        return True
    else:
        logger.error("❌ Historical generation failed - no successful dates")
        return False

def main():
    """Main execution function with command line argument parsing."""
    parser = argparse.ArgumentParser(description='Enhanced QVM Engine v2 - Version-Aware Factor Generation')
    
    # VERSION-AWARE FRAMEWORK: Core parameters for multi-version safety
    parser.add_argument('--version', default='qvm_v2.1.1_flat_corrected',
                       help='Strategy version identifier (default: qvm_v2.1.1_flat_corrected)')
    parser.add_argument('--mode', choices=['incremental', 'refresh'], default='incremental',
                       help='Generation mode: incremental (append missing dates) or refresh (replace existing)')
    
    # Date range parameters
    parser.add_argument('--start-date', required=True,
                       help='Start date (YYYY-MM-DD format)')
    parser.add_argument('--end-date', required=True,
                       help='End date (YYYY-MM-DD format)')
    
    # Performance parameters
    parser.add_argument('--batch-size', type=int, default=30,
                       help='Number of dates to process in each batch (default: 30)')
    
    args = parser.parse_args()
    
    # Validate dates
    try:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d').strftime('%Y-%m-%d')
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d').strftime('%Y-%m-%d')
    except ValueError:
        logger.error("❌ Invalid date format. Use YYYY-MM-DD format.")
        sys.exit(1)
    
    # Run generation with version-aware parameters
    success = run_historical_generation(
        start_date=start_date, 
        end_date=end_date, 
        strategy_version=args.version,
        mode=args.mode,
        batch_size=args.batch_size
    )
    
    if success:
        print("\n🎉 MISSION ACCOMPLISHED!")
        print("✅ Enhanced QVM Engine v2 historical generation completed successfully")
        print("🚀 Ready for VN-Index benchmarking and backtesting validation")
        sys.exit(0)
    else:
        print("\n❌ MISSION FAILED!")
        print("❌ Historical generation encountered critical errors")
        print("🔍 Check logs for detailed error information")
        sys.exit(1)

if __name__ == "__main__":
    main()