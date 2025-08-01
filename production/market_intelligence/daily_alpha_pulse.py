#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Daily Alpha Pulse Dashboard
==========================
Generates daily quantitative market intelligence for Vietnam equity market.

This module provides pre-market insights including:
- Market overview and breadth
- Factor performance monitoring  
- Risk metrics and regime analysis
- Trading signals and opportunities

Author: Duc Nguyen
Date: July 31, 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io

# Add project root to path for imports
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from production.market_intelligence.components.data_loader import MarketDataLoader
from production.market_intelligence.config import MARKET_INTEL_CONFIG, REPORTS_DIR, TEMPLATES_DIR

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DailyAlphaPulse:
    """Daily Alpha Pulse Dashboard Generator"""
    
    def __init__(self):
        """Initialize the dashboard generator"""
        self.config = MARKET_INTEL_CONFIG
        self.reports_dir = REPORTS_DIR
        self.templates_dir = TEMPLATES_DIR
        self.colors = self.config['visualization']['color_scheme']
        
    def generate_dashboard(self, output_date: Optional[datetime] = None) -> Dict:
        """Generate complete daily alpha pulse dashboard"""
        if output_date is None:
            output_date = datetime.now()
            
        logger.info(f"Generating Daily Alpha Pulse for {output_date.strftime('%Y-%m-%d')}")
        
        dashboard_data = {}
        
        try:
            with MarketDataLoader() as loader:
                # Get latest trading date
                latest_date = loader.get_latest_trading_date()
                
                # Core data collection
                dashboard_data['metadata'] = {
                    'generation_time': output_date,
                    'data_date': latest_date,
                    'is_current': (output_date.date() - latest_date.date()).days <= 1
                }
                
                # Market Overview
                logger.info("Collecting market overview data...")
                dashboard_data['market_overview'] = loader.get_market_overview(latest_date)
                
                # Sector Performance
                logger.info("Analyzing sector performance...")
                dashboard_data['sector_performance'] = loader.get_sector_performance(latest_date)
                
                # Factor Performance
                logger.info("Calculating factor performance...")
                dashboard_data['factor_performance'] = loader.get_factor_performance(latest_date)
                
                # Foreign Flows
                logger.info("Analyzing foreign flows...")
                dashboard_data['foreign_flows'] = loader.get_foreign_flow_summary(latest_date)
                
                # Risk Metrics
                logger.info("Computing risk metrics...")
                dashboard_data['risk_metrics'] = loader.get_risk_metrics(latest_date)
                
                # Trading Signals
                logger.info("Generating trading signals...")
                dashboard_data['signals'] = loader.get_top_signals(latest_date)
                
                # Generate visualizations
                logger.info("Creating visualizations...")
                dashboard_data['charts'] = self._generate_charts(dashboard_data)
                
                # Generate HTML report
                html_report = self._generate_html_report(dashboard_data)
                
                # Save reports
                report_filename = f"daily_alpha_pulse_{output_date.strftime('%Y%m%d')}"
                html_path = self.reports_dir / f"{report_filename}.html"
                
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(html_report)
                    
                logger.info(f"Dashboard saved to: {html_path}")
                
                return {
                    'success': True,
                    'report_path': str(html_path),
                    'data': dashboard_data
                }
                
        except Exception as e:
            logger.error(f"Error generating dashboard: {e}")
            return {
                'success': False,
                'error': str(e),
                'data': dashboard_data
            }
            
    def _generate_charts(self, data: Dict) -> Dict:
        """Generate all charts for the dashboard"""
        charts = {}
        
        # Market Breadth Chart
        charts['market_breadth'] = self._create_market_breadth_chart(data['market_overview'])
        
        # Sector Performance Chart
        if not data['sector_performance'].empty:
            charts['sector_performance'] = self._create_sector_performance_chart(data['sector_performance'])
        
        # Factor Performance Chart
        if not data['factor_performance'].empty:
            charts['factor_performance'] = self._create_factor_performance_chart(data['factor_performance'])
        
        # Foreign Flow Chart
        if not data['foreign_flows'].empty:
            charts['foreign_flows'] = self._create_foreign_flow_chart(data['foreign_flows'])
            
        return charts
        
    def _create_market_breadth_chart(self, market_data: Dict) -> str:
        """Create market breadth visualization"""
        advances = market_data['advances']
        declines = market_data['declines']
        unchanged = market_data['unchanged']
        
        # Create pie chart
        fig = go.Figure(data=[go.Pie(
            labels=['Advances', 'Declines', 'Unchanged'],
            values=[advances, declines, unchanged],
            marker_colors=[self.colors['positive'], self.colors['negative'], self.colors['neutral']],
            textinfo='label+percent',
            textposition='inside'
        )])
        
        fig.update_layout(
            title=f"Market Breadth ({market_data['date'].strftime('%Y-%m-%d')})",
            font=dict(size=12),
            height=400,
            showlegend=True
        )
        
        return fig.to_html(div_id="market_breadth_chart", include_plotlyjs='inline')
        
    def _create_sector_performance_chart(self, sector_df: pd.DataFrame) -> str:
        """Create sector performance bar chart"""
        fig = go.Figure()
        
        colors = [self.colors['positive'] if x > 0 else self.colors['negative'] 
                 for x in sector_df['avg_return']]
        
        fig.add_trace(go.Bar(
            x=sector_df['avg_return'],
            y=sector_df['sector'],
            orientation='h',
            marker_color=colors,
            text=[f"{x:.2f}%" for x in sector_df['avg_return']],
            textposition='outside'
        ))
        
        fig.update_layout(
            title="Sector Performance (Daily Returns)",
            xaxis_title="Return (%)",
            yaxis_title="Sector",
            height=max(400, len(sector_df) * 25),
            font=dict(size=10)
        )
        
        return fig.to_html(div_id="sector_performance_chart", include_plotlyjs='inline')
        
    def _create_factor_performance_chart(self, factor_df: pd.DataFrame) -> str:
        """Create factor performance chart"""
        fig = go.Figure()
        
        colors = [self.colors['positive'] if x > 0 else self.colors['negative'] 
                 for x in factor_df['long_short_return']]
        
        fig.add_trace(go.Bar(
            x=factor_df['factor'],
            y=factor_df['long_short_return'],
            marker_color=colors,
            text=[f"{x:.2f}%" if pd.notna(x) else "N/A" for x in factor_df['long_short_return']],
            textposition='outside'
        ))
        
        fig.update_layout(
            title="Factor Performance (Long-Short Returns)",
            xaxis_title="Factor",
            yaxis_title="Return (%)",
            height=400,
            font=dict(size=12)
        )
        
        return fig.to_html(div_id="factor_performance_chart", include_plotlyjs='inline')
        
    def _create_foreign_flow_chart(self, flow_df: pd.DataFrame) -> str:
        """Create foreign flow trend chart"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=flow_df['date'],
            y=flow_df['total_net_value'] / 1e9,  # Convert to billions
            mode='lines+markers',
            name='Net Foreign Flow',
            line=dict(color=self.colors['primary'], width=2),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title="Foreign Flow Trend (Last 5 Days)",
            xaxis_title="Date",
            yaxis_title="Net Flow (Billion VND)",
            height=400,
            font=dict(size=12),
            hovermode='x unified'
        )
        
        return fig.to_html(div_id="foreign_flow_chart", include_plotlyjs='inline')
        
    def _generate_html_report(self, data: Dict) -> str:
        """Generate HTML report"""
        # Simple HTML template (can be enhanced with Jinja2 later)
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Daily Alpha Pulse - {data['metadata']['generation_time'].strftime('%Y-%m-%d')}</title>
    <style>
        body {{ 
            font-family: 'Segoe UI', Arial, sans-serif; 
            margin: 20px; 
            background-color: #f5f5f5; 
        }}
        .header {{ 
            background: linear-gradient(135deg, #1976D2, #42A5F5);
            color: white; 
            padding: 30px; 
            border-radius: 10px; 
            margin-bottom: 20px;
            text-align: center;
        }}
        .header h1 {{ margin: 0; font-size: 2.5em; }}
        .header p {{ margin: 10px 0 0 0; font-size: 1.2em; opacity: 0.9; }}
        .section {{ 
            background: white; 
            padding: 20px; 
            margin: 20px 0; 
            border-radius: 10px; 
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .section h2 {{ 
            color: #1976D2; 
            border-bottom: 2px solid #E3F2FD; 
            padding-bottom: 10px; 
        }}
        .metric-grid {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
            gap: 15px; 
            margin: 20px 0; 
        }}
        .metric-card {{ 
            background: #f8f9fa; 
            padding: 15px; 
            border-radius: 8px; 
            text-align: center;
            border-left: 4px solid #1976D2;
        }}
        .metric-value {{ 
            font-size: 2em; 
            font-weight: bold; 
            margin: 5px 0; 
        }}
        .positive {{ color: #2E7D32; }}
        .negative {{ color: #C62828; }}
        .neutral {{ color: #757575; }}
        .chart-container {{ margin: 20px 0; }}
        .signals-list {{ 
            background: #f8f9fa; 
            padding: 15px; 
            border-radius: 8px; 
        }}
        .signal-item {{ 
            background: white; 
            padding: 10px; 
            margin: 10px 0; 
            border-radius: 5px; 
            border-left: 4px solid #1976D2;
        }}
        .timestamp {{ 
            text-align: center; 
            color: #666; 
            font-size: 0.9em; 
            margin-top: 30px; 
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Daily Alpha Pulse</h1>
        <p>Vietnam Market Intelligence • {data['metadata']['generation_time'].strftime('%A, %B %d, %Y')}</p>
        <p>Data as of: {data['metadata']['data_date'].strftime('%Y-%m-%d')}</p>
    </div>
    
    <div class="section">
        <h2>🏢 Market Overview</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value positive">{data['market_overview']['advances']}</div>
                <div>Advances</div>
            </div>
            <div class="metric-card">
                <div class="metric-value negative">{data['market_overview']['declines']}</div>
                <div>Declines</div>
            </div>
            <div class="metric-card">
                <div class="metric-value neutral">{data['market_overview']['unchanged']}</div>
                <div>Unchanged</div>
            </div>
            <div class="metric-card">
                <div class="metric-value {'positive' if data['market_overview']['avg_change'] > 0 else 'negative'}">{data['market_overview']['avg_change']:.2f}%</div>
                <div>Avg Change</div>
            </div>
            <div class="metric-card">
                <div class="metric-value {'positive' if data['market_overview']['volume_ratio'] > 1 else 'negative'}">{data['market_overview']['volume_ratio']:.2f}x</div>
                <div>Volume vs 20D Avg</div>
            </div>
        </div>
        
        {data['charts'].get('market_breadth', '') if 'charts' in data else ''}
    </div>
    
    <div class="section">
        <h2>🎯 Factor Performance</h2>
        {self._generate_factor_summary_html(data['factor_performance'])}
        {data['charts'].get('factor_performance', '') if 'charts' in data else ''}
    </div>
    
    <div class="section">
        <h2>🏭 Sector Performance</h2>
        {data['charts'].get('sector_performance', '') if 'charts' in data else ''}
    </div>
    
    <div class="section">
        <h2>🌍 Foreign Flows</h2>
        {data['charts'].get('foreign_flows', '') if 'charts' in data else ''}
    </div>
    
    <div class="section">
        <h2>⚠️ Risk Metrics</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value">{data['risk_metrics']['current_volatility']:.1f}%</div>
                <div>Current Volatility (Ann.)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{data['risk_metrics']['volatility_percentile']:.0f}%</div>
                <div>Vol Percentile</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{data['risk_metrics']['average_correlation']:.2f}</div>
                <div>Avg Correlation</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{data['risk_metrics']['volatility_regime']}</div>
                <div>Vol Regime</div>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>🚀 Top Trading Signals</h2>
        <div class="signals-list">
            {self._generate_signals_html(data['signals'])}
        </div>
    </div>
    
    <div class="timestamp">
        Generated at {data['metadata']['generation_time'].strftime('%Y-%m-%d %H:%M:%S')} ICT
    </div>
</body>
</html>
        """
        
        return html_template
        
    def _generate_factor_summary_html(self, factor_df: pd.DataFrame) -> str:
        """Generate factor performance summary HTML"""
        if factor_df.empty:
            return "<p>No factor performance data available.</p>"
            
        html = '<div class="metric-grid">'
        
        for _, row in factor_df.iterrows():
            long_short = row['long_short_return']
            if pd.isna(long_short):
                continue
                
            performance_class = 'positive' if long_short > 0 else 'negative'
            
            html += f'''
            <div class="metric-card">
                <div class="metric-value {performance_class}">{long_short:.2f}%</div>
                <div>{row['factor']} Long-Short</div>
            </div>
            '''
            
        html += '</div>'
        return html
        
    def _generate_signals_html(self, signals: List[Dict]) -> str:
        """Generate trading signals HTML"""
        if not signals:
            return "<p>No signals generated for today.</p>"
            
        html = ""
        for i, signal in enumerate(signals[:5], 1):  # Top 5 signals
            signal_class = 'positive' if signal['signal'] == 'Long' else 'negative'
            
            html += f'''
            <div class="signal-item">
                <strong>{i}. {signal['signal']} {signal['ticker']}</strong> 
                <span class="{signal_class}">({signal['signal']})</span><br>
                <small>
                    Type: {signal['type'].replace('_', ' ').title()} | 
                    Z-Score: {signal.get('z_score', 'N/A')} |
                    Current: {signal.get('current_price', 'N/A')}
                </small>
            </div>
            '''
            
        return html


def main():
    """Command line interface for generating daily alpha pulse"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Daily Alpha Pulse Dashboard')
    parser.add_argument('--date', type=str, help='Output date (YYYY-MM-DD)', default=None)
    parser.add_argument('--output-dir', type=str, help='Output directory', default=None)
    
    args = parser.parse_args()
    
    # Parse date if provided
    output_date = None
    if args.date:
        try:
            output_date = datetime.strptime(args.date, '%Y-%m-%d')
        except ValueError:
            print("Invalid date format. Use YYYY-MM-DD")
            return
            
    # Initialize and generate dashboard
    dashboard = DailyAlphaPulse()
    result = dashboard.generate_dashboard(output_date)
    
    if result['success']:
        print(f"✅ Daily Alpha Pulse generated successfully!")
        print(f"📄 Report saved to: {result['report_path']}")
    else:
        print(f"❌ Error generating dashboard: {result['error']}")


if __name__ == "__main__":
    main()