#!/usr/bin/env python3
"""
Add tearsheet display cell to the notebook
"""

import json
import glob
import os

# Read the notebook
with open('12_adaptive_rebalancing_final.ipynb', 'r') as f:
    notebook = json.load(f)

# Find the latest tearsheet file
tearsheet_files = glob.glob('tearsheet_*.png')
if tearsheet_files:
    latest_tearsheet = max(tearsheet_files, key=os.path.getctime)
    print(f"Latest tearsheet: {latest_tearsheet}")
    
    # Create a new cell to display the tearsheet
    tearsheet_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 📊 Strategy Tearsheet\n\n",
            "Below is the comprehensive performance analysis for the QVM Engine v3j Adaptive Rebalancing FINAL strategy:"
        ]
    }
    
    image_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from IPython.display import Image, display\n",
            "import matplotlib.pyplot as plt\n",
            "import matplotlib.image as mpimg\n",
            "import glob\n",
            "import os\n",
            "\n",
            "# Find the latest tearsheet file\n",
            "tearsheet_files = glob.glob('tearsheet_*.png')\n",
            "if tearsheet_files:\n",
            "    latest_tearsheet = max(tearsheet_files, key=os.path.getctime)\n",
            "    print(f'📊 Latest tearsheet found: {latest_tearsheet}')\n",
            "    \n",
            "    # Display the tearsheet\n",
            "    img = mpimg.imread(latest_tearsheet)\n",
            "    plt.figure(figsize=(15, 10))\n",
            "    plt.imshow(img)\n",
            "    plt.axis('off')\n",
            "    plt.title('QVM Engine v3j Adaptive Rebalancing FINAL - Performance Tearsheet', fontsize=16, pad=20)\n",
            "    plt.tight_layout()\n",
            "    plt.show()\n",
            "    \n",
            "    print(f'📊 Tearsheet displayed: {latest_tearsheet}')\n",
            "else:\n",
            "    print('❌ No tearsheet files found. Run the strategy first to generate tearsheet.')"
        ]
    }
    
    # Add cells to notebook
    notebook['cells'].append(tearsheet_cell)
    notebook['cells'].append(image_cell)
    
    # Write back to notebook
    with open('12_adaptive_rebalancing_final.ipynb', 'w') as f:
        json.dump(notebook, f, indent=2)
    
    print("✅ Added tearsheet display cell to notebook")
else:
    print("❌ No tearsheet files found") 