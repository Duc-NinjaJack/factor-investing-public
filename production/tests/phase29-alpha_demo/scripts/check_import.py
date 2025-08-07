#!/usr/bin/env python3
"""
Check the actual import path
"""

import sys
import os

# Add the necessary paths to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'engine'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'universe'))

print("Python path:")
for path in sys.path:
    print(f"  {path}")

print("\nTrying to import...")
try:
    import qvm_engine_v2_enhanced
    print(f"✅ Imported qvm_engine_v2_enhanced from: {qvm_engine_v2_enhanced.__file__}")
    
    from qvm_engine_v2_enhanced import QVMEngineV2Enhanced
    print(f"✅ Imported QVMEngineV2Enhanced from: {QVMEngineV2Enhanced.__module__}")
    
    # Check the actual file location
    import inspect
    file_location = inspect.getfile(QVMEngineV2Enhanced)
    print(f"✅ QVMEngineV2Enhanced file location: {file_location}")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
except Exception as e:
    print(f"❌ Other error: {e}") 