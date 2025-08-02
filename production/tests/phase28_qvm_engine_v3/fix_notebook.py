# ============================================================================
# Fix Notebook - Add Missing RegimeDetector Class
# ============================================================================
# Purpose: Add the missing RegimeDetector class to the notebook

import json

# Read the notebook
with open('28_qvm_engine_v3b.ipynb', 'r') as f:
    notebook = json.load(f)

# Find the cell that contains the QVMEngineV3AdoptedInsights class
regime_detector_code = '''class RegimeDetector:
    """
    Fixed regime detection that addresses data insufficiency issues.
    Based on diagnostic analysis findings.
    """
    def __init__(self, config: dict):
        self.lookback_period = config['regime']['lookback_period']
        self.volatility_threshold = config['regime']['volatility_threshold']
        self.return_threshold = config['regime']['return_threshold']
        self.low_return_threshold = config['regime']['low_return_threshold']
        self.min_data_points = config['regime']['min_data_points']
    
    def detect_regime(self, benchmark_data: pd.Series, analysis_date: pd.Timestamp) -> str:
        """Fixed regime detection with adaptive lookback."""
        
        # Method 1: Use configured lookback period
        start_date = analysis_date - pd.Timedelta(days=self.lookback_period)
        period_data = benchmark_data.loc[start_date:analysis_date]
        
        # Method 2: If insufficient data, extend the lookback period
        if len(period_data) < self.min_data_points:
            extended_days = int(self.lookback_period * 1.5)  # 135 days instead of 90
            start_date = analysis_date - pd.Timedelta(days=extended_days)
            period_data = benchmark_data.loc[start_date:analysis_date]
            
            if len(period_data) < self.min_data_points:
                # If still insufficient, use all available data
                start_date = benchmark_data.index[0]
                period_data = benchmark_data.loc[start_date:analysis_date]
        
        # Calculate metrics
        returns = period_data.dropna()
        if len(returns) < 10:  # Need at least 10 returns
            return 'Sideways'
        
        volatility = returns.std()
        avg_return = returns.mean()
        
        # Apply fixed regime logic
        if volatility > self.volatility_threshold:
            if avg_return > self.return_threshold:
                return 'Bull'
            else:
                return 'Bear'
        else:
            if abs(avg_return) < self.low_return_threshold:
                return 'Sideways'
            else:
                return 'Stress'
    
    def get_regime_allocation(self, regime: str) -> float:
        """Get target allocation based on regime."""
        regime_allocations = {
            'Bull': 1.0,      # Fully invested
            'Bear': 0.8,      # 80% invested
            'Sideways': 0.6,  # 60% invested
            'Stress': 0.4     # 40% invested
        }
        return regime_allocations.get(regime, 0.6)

'''

# Find the cell with QVMEngineV3AdoptedInsights and add RegimeDetector before it
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'class QVMEngineV3AdoptedInsights:' in source:
            # Add RegimeDetector class before QVMEngineV3AdoptedInsights
            new_source = regime_detector_code + '\n' + source
            cell['source'] = new_source.split('\n')
            # Add newline to each line
            cell['source'] = [line + '\n' for line in cell['source']]
            break

# Write the fixed notebook
with open('28_qvm_engine_v3b_fixed.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("✅ Fixed notebook created: 28_qvm_engine_v3b_fixed.ipynb")
print("   - Added missing RegimeDetector class")
print("   - Fixed regime detection logic")
print("   - Added get_regime_allocation method") 