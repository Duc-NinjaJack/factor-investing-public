"""
Market Intelligence Configuration
=================================
Standalone configuration for market intelligence module.
Uses read-only database access and doesn't modify existing configs.
"""

import os
from pathlib import Path
import yaml

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MARKET_INTEL_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = MARKET_INTEL_ROOT / "reports"
TEMPLATES_DIR = MARKET_INTEL_ROOT / "templates"

# Create directories if they don't exist
REPORTS_DIR.mkdir(exist_ok=True)

# Database configuration (read-only access)
def get_db_config():
    """Load database configuration from existing YAML files"""
    db_yaml_path = PROJECT_ROOT / 'config' / 'database.yml'
    with open(db_yaml_path, 'r') as f:
        db_yaml = yaml.safe_load(f)
    return db_yaml.get('production', db_yaml.get('development'))

# Market Intelligence specific settings
MARKET_INTEL_CONFIG = {
    # Report generation settings
    'reports': {
        'daily_alpha_pulse': {
            'generation_time': '07:30',  # ICT
            'output_formats': ['html', 'pdf'],
            'email_enabled': False,
            'retention_days': 30
        },
        'weekly_strategic_review': {
            'generation_day': 'Friday',
            'generation_time': '16:00',  # ICT
            'output_formats': ['html', 'pdf'],
            'email_enabled': False,
            'retention_days': 90
        }
    },
    
    # Risk thresholds
    'risk_thresholds': {
        'volatility_spike': 2.0,  # Standard deviations
        'correlation_breakdown': 0.8,
        'liquidity_warning': 0.5,  # Relative to 20-day average
        'concentration_limit': 0.15  # Max single stock weight
    },
    
    # Signal generation parameters
    'signals': {
        'mean_reversion': {
            'lookback_days': 20,
            'z_score_threshold': 2.0,
            'min_sharpe': 1.0
        },
        'momentum_breakout': {
            'lookback_days': 52 * 5,  # 52 weeks
            'volume_confirmation': 1.5  # Times average volume
        },
        'factor_rotation': {
            'rebalance_threshold': 0.1,  # 10% weight change
            'min_factor_spread': 0.5  # Minimum factor return spread
        }
    },
    
    # Factor analysis settings
    'factors': {
        'universe': 'liquid_top200',  # Use liquid universe
        'lookback_periods': [20, 60, 252],  # Days
        'correlation_window': 60,
        'outlier_threshold': 3.0  # Z-score
    },
    
    # Visualization settings
    'visualization': {
        'color_scheme': {
            'positive': '#2E7D32',  # Green
            'negative': '#C62828',  # Red
            'neutral': '#757575',   # Gray
            'primary': '#1976D2',   # Blue
            'secondary': '#F57C00'  # Orange
        },
        'chart_height': 400,
        'chart_width': 800
    }
}

# Table mappings (read-only access to existing tables)
TABLE_MAPPINGS = {
    'prices': 'equity_history',
    'market_data': 'vcsc_daily_data_complete',
    'fundamentals': 'fundamental_values',
    'factors': 'factor_scores_qvm',
    'foreign_flows': 'vcsc_foreign_flow_summary',
    'company_info': 'master_info'
}

# Report templates
REPORT_TEMPLATES = {
    'daily_alpha_pulse': 'daily_dashboard.html',
    'weekly_strategic_review': 'weekly_report.html'
}