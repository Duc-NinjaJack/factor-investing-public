# Factor Calculator Component
"""
Factor calculation component for QVM strategy variants.
This module contains the factor calculation logic that can be used by different strategies.
"""

import pandas as pd
import numpy as np
from sqlalchemy import text

class SectorAwareFactorCalculator:
    """
    Sector-aware factor calculator with quality-adjusted P/E.
    Based on insights from value_by_sector_and_quality.md.
    """
    def __init__(self, engine):
        self.engine = engine
    
    def calculate_sector_aware_pe(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate quality-adjusted P/E by sector."""
        if 'roaa' not in data.columns or 'sector' not in data.columns:
            return data
        
        # Create ROAA quintiles within each sector
        def safe_qcut(x):
            try:
                if len(x) < 5:
                    return pd.Series(['Q3'] * len(x), index=x.index)
                return pd.qcut(x, 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')
            except ValueError:
                return pd.Series(['Q3'] * len(x), index=x.index)
        
        data['roaa_quintile'] = data.groupby('sector')['roaa'].transform(safe_qcut)
        
        # Fill missing quintiles with Q3
        data['roaa_quintile'] = data['roaa_quintile'].fillna('Q3')
        
        # Quality-adjusted P/E weights (higher quality = higher weight)
        quality_weights = {
            'Q1': 0.5,  # Low quality
            'Q2': 0.7,
            'Q3': 1.0,  # Medium quality
            'Q4': 1.3,
            'Q5': 1.5   # High quality
        }
        
        data['quality_adjusted_pe'] = data['roaa_quintile'].map(quality_weights)
        return data
    
    def calculate_momentum_score(self, data: pd.DataFrame) -> pd.DataFrame:
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

    def calculate_composite_score(self, data: pd.DataFrame, config: dict) -> pd.DataFrame:
        """Calculate composite score combining all factors."""
        data['composite_score'] = 0.0
        
        # ROAA component (positive signal)
        if 'roaa' in data.columns:
            roaa_weight = config['factors']['roaa_weight']
            data['roaa_normalized'] = (data['roaa'] - data['roaa'].mean()) / data['roaa'].std()
            data['composite_score'] += data['roaa_normalized'] * roaa_weight
        
        # P/E component (contrarian signal - lower is better)
        if 'quality_adjusted_pe' in data.columns:
            pe_weight = config['factors']['pe_weight']
            data['pe_normalized'] = (data['quality_adjusted_pe'] - data['quality_adjusted_pe'].mean()) / data['quality_adjusted_pe'].std()
            data['composite_score'] += (-data['pe_normalized']) * pe_weight  # Negative for contrarian
        
        # Momentum component (mixed signal - 3M/6M positive, 1M/12M contrarian)
        if 'momentum_score' in data.columns:
            momentum_weight = config['factors']['momentum_weight']
            data['momentum_normalized'] = (data['momentum_score'] - data['momentum_score'].mean()) / data['momentum_score'].std()
            data['composite_score'] += data['momentum_normalized'] * momentum_weight
        
        return data

    def apply_entry_criteria(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply entry criteria to filter stocks."""
        qualified = data.copy()
        
        if 'roaa' in qualified.columns:
            qualified = qualified[qualified['roaa'] > 0]  # Positive ROAA
        
        if 'net_margin' in qualified.columns:
            qualified = qualified[qualified['net_margin'] > 0]  # Positive net margin
        
        return qualified 