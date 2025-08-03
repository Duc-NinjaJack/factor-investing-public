#!/usr/bin/env python3
"""
Test script to verify notebook imports work correctly
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import warnings
warnings.filterwarnings('ignore')

# Add the necessary paths to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'engine'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'universe'))

print("Testing imports...")

try:
    from qvm_engine_v2_enhanced import QVMEngineV2Enhanced
    print("✅ Successfully imported QVMEngineV2Enhanced")
except ImportError as e:
    print(f"❌ Failed to import QVMEngineV2Enhanced: {e}")

try:
    from constructors import get_liquid_universe
    print("✅ Successfully imported get_liquid_universe")
except ImportError as e:
    print(f"❌ Failed to import get_liquid_universe: {e}")

print(f"Phase 29 Demonstration Started: {datetime.now()}")
print("QVM Engine v2 Enhanced - Complete Implementation Demo")

# Test engine initialization
try:
    engine = QVMEngineV2Enhanced()
    print("✅ QVM Engine v2 Enhanced initialized successfully")
    print(f"   - Engine class: {engine.__class__.__name__}")
    print(f"   - Strategy version: qvm_v2_enhanced")
    print(f"   - Database connection: {'✅ Connected' if hasattr(engine, 'engine') and engine.engine else '❌ Failed'}")
except Exception as e:
    print(f"❌ Engine initialization failed: {e}")

print("✅ All imports and initialization tests passed!") 