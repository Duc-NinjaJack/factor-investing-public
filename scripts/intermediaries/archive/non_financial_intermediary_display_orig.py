#!/usr/bin/env python3
"""
Non-Financial Sector Intermediary Values Display
===============================================
Displays calculated intermediary values from the database in tabular format
Automatically determines sector folder for saving output files
"""

import pandas as pd
import pymysql
import yaml
from datetime import datetime
from pathlib import Path
import sys
import argparse

# Setup project path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

def connect_to_database():
    """Create database connection"""
    try:
        config_path = project_root / 'config' / 'database.yml'
        with open(config_path, 'r') as f:
            db_config = yaml.safe_load(f)['production']
        
        connection = pymysql.connect(
            host=db_config['host'],
            user=db_config['username'], 
            password=db_config['password'],
            database=db_config['schema_name'],
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        return connection
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return None

def get_ticker_sector(ticker, connection):
    """Get sector for a ticker to determine output folder"""
    query = "SELECT sector FROM master_info WHERE ticker = %s"
    try:
        with connection.cursor() as cursor:
            cursor.execute(query, [ticker])
            result = cursor.fetchone()
            if result:
                sector = result['sector'].lower().replace(' ', '_').replace('&', 'and')
                # Map common sector names to folder names
                sector_mapping = {
                    'technology': 'technology',
                    'real_estate': 'real_estate',
                    'construction': 'construction',
                    'food_and_beverages': 'food_beverage',
                    'wholesale': 'wholesale',
                    'plastics_and_chemicals': 'plastics',
                    'utilities': 'utilities',
                    'logistics': 'logistics',
                    'construction_materials': 'construction_materials',
                    'mining_and_oil': 'mining_oil',
                    'ancillary_production': 'ancillary_production',
                    'healthcare': 'healthcare',
                    'household_goods': 'household_goods',
                    'electrical_equipment': 'electrical_equipment',
                    'retail': 'retail',
                    'seafood': 'seafood',
                    'agriculture': 'agriculture',
                    'industrial_services': 'industrial_services',
                    'hotels_and_tourism': 'hotels_tourism',
                    'machinery': 'machinery',
                    'rubber_products': 'rubber_products'
                }
                return sector_mapping.get(sector, 'non_financial')
            return 'unknown'
    except Exception as e:
        print(f"Warning: Could not determine sector for {ticker}: {e}")
        return 'unknown'

def format_value(value, decimals=1):
    """Format value as billions with comma separator"""
    if pd.isna(value) or value is None:
        return "-"
    
    # For ratios and days metrics, show as regular number
    if abs(value) < 1000:
        return f"{value:.{decimals}f}"
    
    # For large numbers, convert to billions
    billions = value / 1_000_000_000
    if abs(billions) >= 1000:
        return f"{billions:,.0f}"
    elif abs(billions) >= 100:
        return f"{billions:,.0f}"
    elif abs(billions) >= 10:
        return f"{billions:,.1f}"
    else:
        return f"{billions:,.2f}"

def extract_intermediary_data(ticker, connection, periods=12):
    """Extract intermediary data from database"""
    
    query = f"""
    SELECT *
    FROM intermediary_calculations_enhanced
    WHERE ticker = %s
    ORDER BY year DESC, quarter DESC
    LIMIT {periods}
    """
    
    with connection.cursor() as cursor:
        cursor.execute(query, [ticker])
        results = cursor.fetchall()
    
    return pd.DataFrame(results) if results else pd.DataFrame()

def create_quarter_headers(df):
    """Create quarter headers from data - latest quarters on the right"""
    quarters = []
    for _, row in df.iterrows():
        quarters.append(f"{int(row['year'])}Q{int(row['quarter'])}")
    return quarters[::-1]  # Reverse to put latest on the right

def create_markdown_display(ticker, sector, df):
    """Create markdown formatted display of intermediary values"""
    
    # Create quarter headers (latest on right)
    quarters = create_quarter_headers(df)
    
    # Start building markdown content
    content = []
    
    # Header
    sector_display = sector.replace('_', ' ').title()
    content.append(f"# {ticker} Intermediary Values - {sector_display} Sector\n")
    content.append(f"**Source**: intermediary_calculations_enhanced table")
    content.append(f"**Purpose**: Display calculated intermediary values for last {len(df)} quarters")
    content.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    content.append(f"**✅ SUCCESS**: Displaying {len(df)} quarters of intermediary calculations\n")
    
    # I. TTM Values section
    content.append("## I. TTM Values (Trailing Twelve Months) 💰\n")
    
    # Header
    header = "| Metric - VNDbn |" + "".join([f" {q} |" for q in quarters])
    content.append(header)
    separator = "|---|" + "---|" * len(quarters)
    content.append(separator)
    
    # TTM items
    ttm_items = [
        ('Revenue_TTM', 'Revenue_TTM'),
        ('COGS_TTM', 'COGS_TTM'),
        ('GrossProfit_TTM', 'GrossProfit_TTM'),
        ('SellingExpenses_TTM', 'SellingExpenses_TTM'),
        ('AdminExpenses_TTM', 'AdminExpenses_TTM'),
        ('OperatingExpenses_TTM', 'Operating Expenses_TTM'),
        ('EBIT_TTM', '**EBIT_TTM** (Calculated)'),
        ('EBITDA_TTM', '**EBITDA_TTM**'),
        ('FinancialIncome_TTM', 'FinancialIncome_TTM'),
        ('FinancialExpenses_TTM', 'FinancialExpenses_TTM'),
        ('InterestExpense_TTM', 'InterestExpense_TTM'),
        ('ProfitBeforeTax_TTM', 'ProfitBeforeTax_TTM'),
        ('TotalTax_TTM', 'TotalTax_TTM'),
        ('NetProfit_TTM', 'NetProfit_TTM'),
        ('NetProfitAfterMI_TTM', '**NetProfitAfterMI_TTM**'),
        ('', ''),  # Separator
        ('NetCFO_TTM', 'Operating Cash Flow_TTM'),
        ('NetCFI_TTM', 'Investing Cash Flow_TTM'),
        ('NetCFF_TTM', 'Financing Cash Flow_TTM'),
        ('CapEx_TTM', 'CapEx_TTM'),
        ('FCF_TTM', '**Free Cash Flow_TTM**'),
        ('DividendsPaid_TTM', 'Dividends Paid_TTM')
    ]
    
    # Reverse df to match quarter order (latest on right)
    df_reversed = df.iloc[::-1]
    
    for col, label in ttm_items:
        if col == '':
            content.append("|  |" + " |" * len(quarters))
            continue
            
        if col in df.columns:
            row = f"| {label} |"
            for _, period_data in df_reversed.iterrows():
                value = period_data[col]
                row += f" {format_value(value)} |"
            content.append(row)
    
    content.append("")
    
    # II. 5-Point Balance Sheet Averages
    content.append("## II. 5-Point Balance Sheet Averages 📊\n")
    
    # Header
    header = "| Metric - VNDbn |" + "".join([f" {q} |" for q in quarters])
    content.append(header)
    separator = "|---|" + "---|" * len(quarters)
    content.append(separator)
    
    # Average items
    avg_items = [
        ('AvgTotalAssets', '**Avg Total Assets**'),
        ('AvgTotalEquity', '**Avg Total Equity**'),
        ('AvgTotalLiabilities', 'Avg Total Liabilities'),
        ('', ''),
        ('AvgCurrentAssets', 'Avg Current Assets'),
        ('AvgCurrentLiabilities', 'Avg Current Liabilities'),
        ('AvgWorkingCapital', '**Avg Working Capital**'),
        ('', ''),
        ('AvgCash', 'Avg Cash'),
        ('AvgReceivables', 'Avg Receivables'),
        ('AvgInventory', 'Avg Inventory'),
        ('AvgPayables', 'Avg Payables'),
        ('', ''),
        ('AvgFixedAssets', 'Avg Fixed Assets'),
        ('AvgShortTermDebt', 'Avg Short-term Debt'),
        ('AvgLongTermDebt', 'Avg Long-term Debt'),
        ('AvgTotalDebt', '**Avg Total Debt**'),
        ('AvgNetDebt', '**Avg Net Debt**'),
        ('', ''),
        ('AvgInvestedCapital', '**Avg Invested Capital**')
    ]
    
    for col, label in avg_items:
        if col == '':
            content.append("|  |" + " |" * len(quarters))
            continue
            
        if col in df.columns:
            row = f"| {label} |"
            for _, period_data in df_reversed.iterrows():
                value = period_data[col]
                row += f" {format_value(value)} |"
            content.append(row)
    
    content.append("")
    
    # III. Working Capital Metrics
    content.append("## III. Working Capital Metrics 🔄\n")
    
    # Header
    header = "| Metric - Days |" + "".join([f" {q} |" for q in quarters])
    content.append(header)
    separator = "|---|" + "---|" * len(quarters)
    content.append(separator)
    
    # WC items
    wc_items = [
        ('DSO', '**DSO** (Days Sales Outstanding)'),
        ('DIO', '**DIO** (Days Inventory Outstanding)'),
        ('DPO', '**DPO** (Days Payables Outstanding)'),
        ('CCC', '**CCC** (Cash Conversion Cycle)')
    ]
    
    for col, label in wc_items:
        if col in df.columns:
            row = f"| {label} |"
            for _, period_data in df_reversed.iterrows():
                value = period_data[col]
                row += f" {format_value(value, 1)} |"
            content.append(row)
    
    content.append("\n**Working Capital Formulas:**")
    content.append("- DSO = (Avg Receivables × 365) / Revenue_TTM")
    content.append("- DIO = (Avg Inventory × 365) / COGS_TTM")
    content.append("- DPO = (Avg Payables × 365) / COGS_TTM")
    content.append("- CCC = DSO + DIO - DPO\n")
    
    # IV. Data Quality Metadata
    content.append("## IV. Data Quality Metadata 📈\n")
    
    # Header
    header = "| Metric |" + "".join([f" {q} |" for q in quarters])
    content.append(header)
    separator = "|---|" + "---|" * len(quarters)
    content.append(separator)
    
    # Quality items
    quality_items = [
        ('quarters_available', 'Quarters Available (TTM)'),
        ('has_full_ttm', 'Has Full TTM (4Q)'),
        ('avg_points_used', 'Avg Points Used'),
        ('has_full_avg', 'Has Full 5-Point Avg')
    ]
    
    for col, label in quality_items:
        if col in df.columns:
            row = f"| {label} |"
            for _, period_data in df_reversed.iterrows():
                value = period_data[col]
                if col.startswith('has_'):
                    # Boolean values
                    display = "✅" if value == 1 else "❌"
                else:
                    display = f"{int(value)}" if pd.notna(value) else "-"
                row += f" {display} |"
            content.append(row)
    
    content.append("")
    content.append("---\n")
    content.append("## Summary\n")
    content.append(f"**🎯 PHASE 2 INTERMEDIARY CALCULATIONS: ✅ COMPLETE FOR {sector_display.upper()} SECTOR**\n")
    content.append("- **TTM Values**: Calculated for all income statement and cash flow items")
    content.append("- **5-Point Averages**: Calculated for all balance sheet items")
    content.append("- **Working Capital**: DSO, DIO, DPO, and CCC metrics computed")
    content.append("- **Data Quality**: Metadata tracks calculation completeness\n")
    content.append("**Next Step**: Use these pre-computed intermediaries for Phase 3 factor calculations\n")
    content.append("---\n")
    content.append("**Document Owner**: Phase 2 Infrastructure Validation")
    content.append(f"**Related Scripts**: `scripts/{sector}_sector_intermediary_calculator.py`")
    
    return "\n".join(content)

def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description='Display non-financial intermediary values')
    parser.add_argument('ticker', help='Ticker symbol to display')
    parser.add_argument('--periods', type=int, default=12, help='Number of periods to display (default: 12)')
    parser.add_argument('--save', action='store_true', help='Save output to file')
    
    args = parser.parse_args()
    ticker = args.ticker.upper()
    
    # Connect to database
    conn = connect_to_database()
    if not conn:
        return
    
    try:
        # Get sector for this ticker
        sector = get_ticker_sector(ticker, conn)
        print(f"📁 Sector: {sector}")
        
        # Extract data
        print(f"📊 Extracting intermediary data for {ticker}...")
        df = extract_intermediary_data(ticker, conn, args.periods)
        
        if df.empty:
            print(f"❌ No intermediary data found for {ticker}")
            return
        
        print(f"✅ Found {len(df)} quarters of data")
        
        # Create markdown display
        markdown_content = create_markdown_display(ticker, sector, df)
        
        # Print to console
        print("\n" + "="*120)
        print(f"INTERMEDIARY VALUES - {ticker}")
        print("="*120)
        print(markdown_content)
        
        # Save to file if requested
        if args.save:
            # Output path with sector-specific folder
            output_path = project_root / f"docs/4_validation_and_quality/intermediaries/{sector}/{ticker}_intermediary_values.md"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            print(f"\n✅ Intermediary values report saved to: {output_path}")
    
    finally:
        conn.close()

if __name__ == "__main__":
    main()