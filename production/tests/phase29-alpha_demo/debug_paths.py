#!/usr/bin/env python3
"""
Debug script to check path resolution
"""

from pathlib import Path

# Simulate the engine's path resolution
engine_file = Path("/Users/raymond/Documents/Projects/factor-investing-public/production/engine/qvm_engine_v2_enhanced.py")
print(f"Engine file: {engine_file}")

# Navigate from engine directory to project root
project_root = engine_file.parent.parent.parent
print(f"Project root: {project_root}")

# Check config paths
db_config = project_root / 'config' / 'database.yml'
strategy_config = project_root / 'config' / 'strategy_config.yml'
factor_metadata = project_root / 'config' / 'factor_metadata.yml'

print(f"Database config: {db_config}")
print(f"Strategy config: {strategy_config}")
print(f"Factor metadata: {factor_metadata}")

print(f"Database config exists: {db_config.exists()}")
print(f"Strategy config exists: {strategy_config.exists()}")
print(f"Factor metadata exists: {factor_metadata.exists()}")

# Check what's actually in the config directory
config_dir = project_root / 'config'
print(f"Config directory: {config_dir}")
if config_dir.exists():
    print("Files in config directory:")
    for file in config_dir.glob('*.yml'):
        print(f"  - {file.name}")
else:
    print("Config directory does not exist!") 