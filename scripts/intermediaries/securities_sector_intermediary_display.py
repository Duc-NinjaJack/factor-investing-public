#!/usr/bin/env python3
"""
Securities Sector Intermediary Display Script
============================================
Display calculated securities intermediary values for analysis and validation.

Author: Duc Nguyen (Aureus Sigma Capital)
Date: July 23, 2025 - Updated for OperatingExpenses_TTM validation
"""

import sys
import pandas as pd
import pymysql
import yaml
from pathlib import Path
from datetime import datetime
import argparse

# Setup project path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

def get_db_connection():
    """Get database connection"""
    config_path = project_root / 'config' / 'database.yml'
    with open(config_path, 'r') as f:
        db_config = yaml.safe_load(f)['production']
    
    return pymysql.connect(
        host=db_config['host'],
        user=db_config['username'],
        password=db_config['password'],
        database=db_config['schema_name'],
        charset='utf8mb4'
    )

def display_securities_intermediaries(ticker, save_to_file=False):
    """Display intermediary values for a securities ticker"""
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get intermediary data
    cursor.execute("""
        SELECT * FROM intermediary_calculations_securities_cleaned
        WHERE ticker = %s
        ORDER BY year DESC, quarter DESC
        LIMIT 4
    """, [ticker])
    
    columns = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()
    
    if not rows:
        print(f"No intermediary data found for {ticker}")
        return
    
    # Convert to DataFrame and reverse order (latest quarter on right)
    df = pd.DataFrame(rows, columns=columns)
    df = df.iloc[::-1].reset_index(drop=True)
    
    # Format output
    output_lines = []
    output_lines.append(f"# {ticker} Intermediary Values - Securities Sector")
    output_lines.append("")
    output_lines.append(f"**Source**: intermediary_calculations_securities_cleaned table")
    output_lines.append(f"**🎯 VALIDATION**: Testing fixed OperatingExpenses_TTM calculation")
    output_lines.append(f"**Purpose**: Display calculated securities intermediary values for last 4 quarters")
    output_lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output_lines.append("")
    output_lines.append(f"**✅ SUCCESS**: Displaying {len(df)} quarters of securities intermediary calculations")
    output_lines.append("")
    
    # Revenue streams section
    output_lines.append("## I. Securities Revenue Streams - TTM Values 💰")
    output_lines.append("")
    output_lines.append("| Metric - VNDbn | " + " | ".join([f"{row['year']}Q{row['quarter']}" for _, row in df.iterrows()]) + " |")
    output_lines.append("|---|" + "---|" * len(df))
    
    revenue_metrics = [
        ('**TRADING INCOME**', None),
        ('&nbsp;&nbsp;Trading Gains (FVTPL)', 'TradingGainFVTPL_TTM'),
        ('&nbsp;&nbsp;Trading Gains (HTM)', 'TradingGainHTM_TTM'),
        ('&nbsp;&nbsp;Trading Gains (Loans)', 'TradingGainLoans_TTM'),
        ('&nbsp;&nbsp;Trading Gains (AFS)', 'TradingGainAFS_TTM'),
        ('&nbsp;&nbsp;Trading Gains (Derivatives)', 'TradingGainDerivatives_TTM'),
        ('**Net Trading Income**', 'NetTradingIncome_TTM'),
        (' ', None),
        ('**SECURITIES SERVICES**', None),
        ('&nbsp;&nbsp;Brokerage Revenue', 'BrokerageRevenue_TTM'),
        ('&nbsp;&nbsp;Advisory Revenue', 'AdvisoryRevenue_TTM'),
        ('&nbsp;&nbsp;Custody Service Revenue', 'CustodyServiceRevenue_TTM'),
        ('&nbsp;&nbsp;Underwriting Revenue', 'UnderwritingRevenue_TTM'),
        ('&nbsp;&nbsp;Other Operating Income', 'OtherOperatingIncome_TTM'),
        ('**Total Securities Services**', 'TotalSecuritiesServices_TTM'),
        (' ', None),
        ('**Total Operating Revenue**', 'TotalOperatingRevenue_TTM'),
        (' ', None),
        ('**OPERATING EXPENSES (FIXED)**', None),
        ('&nbsp;&nbsp;Brokerage Expenses', 'BrokerageExpenses_TTM'),
        ('&nbsp;&nbsp;Underwriting Expenses', 'UnderwritingExpenses_TTM'),
        ('&nbsp;&nbsp;Advisory Expenses', 'AdvisoryExpenses_TTM'),
        ('&nbsp;&nbsp;Custody Service Expenses', 'CustodyServiceExpenses_TTM'),
        ('&nbsp;&nbsp;Management Expenses', 'ManagementExpenses_TTM'),
        ('&nbsp;&nbsp;Other Operating Expenses', 'OtherOperatingExpenses_TTM'),
        ('**Operating Expenses (TOTAL)**', 'OperatingExpenses_TTM'),
        ('**Operating Result**', 'OperatingResult_TTM'),
        ('**Profit Before Tax**', 'ProfitBeforeTax_TTM'),
        ('Income Tax Expense', 'IncomeTaxExpense_TTM'),
        ('**Profit After Tax**', 'NetProfit_TTM')
    ]
    
    for label, col in revenue_metrics:
        if col is None:
            output_lines.append("| " + " | " * (len(df) + 1))
        else:
            values = []
            for _, row in df.iterrows():
                val = row.get(col)
                if pd.notna(val) and val != 0:
                    values.append(f"{float(val)/1e9:,.0f}")
                else:
                    values.append("-")
            output_lines.append(f"| {label} | " + " | ".join(values) + " |")
    
    # Balance sheet averages
    output_lines.append("")
    output_lines.append("## II. Balance Sheet - 5-Point Averages 📊")
    output_lines.append("")
    output_lines.append("| Metric - VNDbn | " + " | ".join([f"{row['year']}Q{row['quarter']}" for _, row in df.iterrows()]) + " |")
    output_lines.append("|---|" + "---|" * len(df))
    
    bs_metrics = [
        ('**Total Assets**', 'AvgTotalAssets'),
        ('Financial Assets', 'AvgFinancialAssets'),
        ('Financial Assets FVTPL', 'AvgFinancialAssetsFVTPL'),
        ('Cash & Equivalents', 'AvgCashAndCashEquivalents'),
        (' ', None),
        ('**Total Equity**', 'AvgTotalEquity'),
        ('Charter Capital', 'AvgCharterCapital'),
        ('Retained Earnings', 'AvgRetainedEarnings')
    ]
    
    for label, col in bs_metrics:
        if col is None:
            output_lines.append("| " + " | " * (len(df) + 1))
        else:
            values = []
            for _, row in df.iterrows():
                val = row.get(col)
                if pd.notna(val) and val != 0:
                    values.append(f"{float(val)/1e9:,.0f}")
                else:
                    values.append("-")
            output_lines.append(f"| {label} | " + " | ".join(values) + " |")
    
    # Securities-specific metrics
    output_lines.append("")
    output_lines.append("## III. Securities-Specific Metrics 🏦")
    output_lines.append("")
    output_lines.append("| Metric | " + " | ".join([f"{row['year']}Q{row['quarter']}" for _, row in df.iterrows()]) + " |")
    output_lines.append("|---|" + "---|" * len(df))
    
    specific_metrics = [
        ('**Brokerage Ratio (%)**', 'BrokerageRatio'),
        ('Advisory Ratio (%)', 'AdvisoryRatio'),
        ('Custody Ratio (%)', 'CustodyRatio'),
        ('Trading Ratio (%)', 'TradingRatio'),
        (' ', None),
        ('**ROAA (%)**', 'ROAA'),
        ('**ROAE (%)**', 'ROAE'),
        ('Equity Ratio (%)', 'EquityRatio'),
        ('Leverage Ratio (x)', 'LeverageRatio')
    ]
    
    for label, col in specific_metrics:
        if col is None:
            output_lines.append("| " + " | " * (len(df) + 1))
        else:
            values = []
            for _, row in df.iterrows():
                val = row.get(col)
                if pd.notna(val) and val != 0:
                    values.append(f"{float(val):.1f}")
                else:
                    values.append("-")
            output_lines.append(f"| {label} | " + " | ".join(values) + " |")
    
    # Data quality section
    output_lines.append("")
    output_lines.append("## IV. Data Quality Metadata 📊")
    output_lines.append("")
    output_lines.append("| Metric | " + " | ".join([f"{row['year']}Q{row['quarter']}" for _, row in df.iterrows()]) + " |")
    output_lines.append("|---|" + "---|" * len(df))
    
    quality_metrics = [
        ('Quarters Available (TTM)', 'quarters_available_ttm'),
        ('Has Full TTM (4Q)', 'has_full_ttm'),
        ('Avg Points Used', 'avg_points_used'),
        ('Has Full 5-Point Avg', 'has_full_avg'),
        ('Data Quality Score', 'data_quality_score')
    ]
    
    for label, col in quality_metrics:
        values = []
        for _, row in df.iterrows():
            val = row.get(col)
            if col in ['has_full_ttm', 'has_full_avg']:
                values.append("✅" if val else "❌")
            elif col == 'data_quality_score':
                values.append(f"{float(val):.0f}%" if pd.notna(val) else "-")
            else:
                values.append(str(val) if pd.notna(val) else "-")
        output_lines.append(f"| {label} | " + " | ".join(values) + " |")
    
    # Summary
    output_lines.append("")
    output_lines.append("---")
    output_lines.append("")
    output_lines.append("## Summary")
    output_lines.append("")
    output_lines.append(f"**🎯 SECURITIES INTERMEDIARY CALCULATIONS: ✅ COMPLETE FOR {ticker}**")
    output_lines.append("")
    
    # Key insights
    latest = df.iloc[-1]
    if pd.notna(latest.get('BrokerageRatio')):
        output_lines.append("### Key Insights:")
        output_lines.append(f"1. **Revenue Mix**: Brokerage {float(latest['BrokerageRatio']):.1f}%, Trading {float(latest.get('TradingRatio', 0)):.1f}%")
        output_lines.append(f"2. **Profitability**: ROAA {float(latest.get('ROAA', 0)):.2f}%, ROAE {float(latest.get('ROAE', 0)):.2f}%")
        output_lines.append(f"3. **Capital**: Equity Ratio {float(latest.get('EquityRatio', 0)):.1f}%, Leverage {float(latest.get('LeverageRatio', 0)):.1f}x")
    
    output_lines.append("")
    output_lines.append("**Next Step**: Use these pre-computed securities intermediaries for Phase 3 factor calculations")
    output_lines.append("")
    output_lines.append("---")
    output_lines.append("")
    output_lines.append("**Document Owner**: Phase 2 Securities Infrastructure Validation")
    output_lines.append("**Related Scripts**: `scripts/intermediaries/securities_sector_intermediary_calculator.py`")
    
    # Output
    output_text = "\n".join(output_lines)
    print(output_text)
    
    # Save to file if requested
    if save_to_file:
        output_dir = project_root / 'docs' / '4_validation_and_quality' / 'intermediaries' / 'securities'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"{ticker}_securities_intermediary_values.md"
        with open(output_file, 'w') as f:
            f.write(output_text)
        
        print(f"\n✅ Saved to: {output_file}")
    
    conn.close()

def main():
    parser = argparse.ArgumentParser(description='Display securities intermediary values')
    parser.add_argument('ticker', help='Securities ticker to display')
    parser.add_argument('--save', action='store_true', help='Save output to markdown file')
    
    args = parser.parse_args()
    
    display_securities_intermediaries(args.ticker.upper(), args.save)

if __name__ == "__main__":
    main()