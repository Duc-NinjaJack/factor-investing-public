#!/usr/bin/env python3
"""
Test tearsheet display functionality
"""

import glob
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Find the latest tearsheet file
tearsheet_files = glob.glob('tearsheet_*.png')
if tearsheet_files:
    latest_tearsheet = max(tearsheet_files, key=os.path.getctime)
    print(f"📊 Latest tearsheet found: {latest_tearsheet}")
    
    # Test loading the image
    try:
        img = mpimg.imread(latest_tearsheet)
        print(f"✅ Image loaded successfully: {img.shape}")
        
        # Test displaying (without showing)
        plt.figure(figsize=(15, 10))
        plt.imshow(img)
        plt.axis('off')
        plt.title('QVM Engine v3j Adaptive Rebalancing FINAL - Performance Tearsheet', fontsize=16, pad=20)
        plt.tight_layout()
        
        # Save a test version
        plt.savefig('test_tearsheet_display.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✅ Tearsheet display test successful!")
        print("📁 Test file saved: test_tearsheet_display.png")
        
    except Exception as e:
        print(f"❌ Error loading tearsheet: {e}")
else:
    print("❌ No tearsheet files found") 