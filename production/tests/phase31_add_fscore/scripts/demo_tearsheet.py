#!/usr/bin/env python3
"""Demo script for the comprehensive tearsheet functionality."""
import sys
sys.path.append(".")
from production.tests.phase31_add_fscore.qvm_enhanced import generate_sample_tearsheet_data, calculate_performance_metrics
print("📊 Testing Tearsheet Functionality...")
data = generate_sample_tearsheet_data()
print(f"✅ Generated {len(data[0])} daily returns")
metrics = calculate_performance_metrics(data[0], data[1])
print(f"✅ Calculated {len(metrics)} performance metrics")
print("\\n📊 Performance Metrics:")
for key, value in metrics.items():
    print(f"   {key}: {value:.2f}")
