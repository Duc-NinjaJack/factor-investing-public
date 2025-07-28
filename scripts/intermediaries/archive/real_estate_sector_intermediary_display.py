#!/usr/bin/env python3
"""
Real Estate Sector Intermediary Values Display
=============================================
Displays calculated intermediary values from the database in tabular format
Based on tech sector display script
"""

import pandas as pd
import pymysql
import yaml
from datetime import datetime
from pathlib import Path
import sys

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

def format_value(value, decimals=1, is_days=False):
    """Format value as billions with comma separator"""
    if pd.isna(value) or value is None:
        return "-"
    
    # For days metrics, always show as regular number (not billions)
    if is_days:
        return f"{value:.{decimals}f}"
    
    # For ratios and small metrics, show as regular number
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

def extract_intermediary_data(ticker, connection):
    """Extract intermediary data from database"""
    
    query = """
    SELECT *
    FROM intermediary_calculations_enhanced
    WHERE ticker = %s
    ORDER BY year DESC, quarter DESC
    LIMIT 4
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

def print_ttm_section(df, quarters, output_file):
    """Print TTM values section"""
    content = []
    content.append(f"## I. TTM Values (Trailing Twelve Months) 🏢\n")
    
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
    section_text = "\n".join(content)
    output_file.write(section_text + "\n")

def print_averages_section(df, quarters, output_file):
    """Print 5-point averages section"""
    content = []
    content.append(f"## II. 5-Point Balance Sheet Averages 🏗️\n")
    
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
    
    # Reverse df to match quarter order (latest on right)
    df_reversed = df.iloc[::-1]
    
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
    section_text = "\n".join(content)
    output_file.write(section_text + "\n")

def print_working_capital_section(df, quarters, output_file):
    """Print working capital metrics section"""
    content = []
    content.append(f"## III. Working Capital Metrics 🔄\n")
    
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
    
    # Reverse df to match quarter order (latest on right)
    df_reversed = df.iloc[::-1]
    
    for col, label in wc_items:
        if col in df.columns:
            row = f"| {label} |"
            for _, period_data in df_reversed.iterrows():
                value = period_data[col]
                row += f" {format_value(value, 1, is_days=True)} |"
            content.append(row)
    
    content.append("\n**Working Capital Formulas:**")
    content.append("- DSO = (Avg Receivables × 365) / Revenue_TTM")
    content.append("- DIO = (Avg Inventory × 365) / COGS_TTM")
    content.append("- DPO = (Avg Payables × 365) / COGS_TTM")
    content.append("- CCC = DSO + DIO - DPO\n")
    
    section_text = "\n".join(content)
    output_file.write(section_text + "\n")

def print_data_quality_section(df, quarters, output_file):
    """Print data quality metadata section"""
    content = []
    content.append(f"## IV. Data Quality Metadata 📈\n")
    
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
    
    # Reverse df to match quarter order (latest on right)
    df_reversed = df.iloc[::-1]
    
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
    section_text = "\n".join(content)
    output_file.write(section_text + "\n")

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

def main():
    """Main execution"""
    import sys
    
    # Get ticker from command line argument
    ticker = sys.argv[1] if len(sys.argv) > 1 else 'NLG'
    
    # Connect to database first to get sector
    conn = connect_to_database()
    if not conn:
        return
    
    # Get sector for this ticker
    sector = get_ticker_sector(ticker, conn)
    
    # Output path with sector-specific folder
    output_path = project_root / f"docs/4_validation_and_quality/intermediaries/{sector}/{ticker}_intermediary_values.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract data
    df = extract_intermediary_data(ticker, conn)
    
    if df.empty:
        print(f"❌ No intermediary data found for {ticker}")
        return
    
    # Create quarter headers (latest on right)
    quarters = create_quarter_headers(df)
    
    # Write to markdown file
    with open(output_path, 'w', encoding='utf-8') as f:
        # Header - use actual sector name
        sector_display = sector.replace('_', ' ').title()
        f.write(f"# {ticker} Intermediary Values - {sector_display} Sector\n\n")
        f.write(f"**Source**: intermediary_calculations_enhanced table\n")
        f.write(f"**Purpose**: Display calculated intermediary values for last 4 quarters\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"**✅ SUCCESS**: Displaying {len(df)} quarters of intermediary calculations\n\n")
        
        # Write all sections
        print_ttm_section(df, quarters, f)
        print_averages_section(df, quarters, f)
        print_working_capital_section(df, quarters, f)
        print_data_quality_section(df, quarters, f)
        
        f.write("---\n\n")
        f.write("## Summary\n\n")
        f.write("**🎯 PHASE 2 INTERMEDIARY CALCULATIONS: ✅ COMPLETE FOR REAL ESTATE SECTOR**\n\n")
        f.write(f"- **TTM Values**: Calculated for all income statement and cash flow items\n")
        f.write(f"- **5-Point Averages**: Calculated for all balance sheet items\n")
        f.write(f"- **Working Capital**: DSO, DIO, DPO, and CCC metrics computed\n")
        f.write(f"- **Data Quality**: Metadata tracks calculation completeness\n\n")
        
        f.write("**Next Step**: Use these pre-computed intermediaries for Phase 3 factor calculations\n\n")
        
        f.write("---\n\n")
        f.write(f"**Document Owner**: Phase 2 Infrastructure Validation\n")
        f.write(f"**Related Scripts**: `scripts/real_estate_sector_intermediary_calculator.py`\n")
    
    conn.close()
    
    print(f"✅ Intermediary values report saved to: {output_path}")
    print(f"📊 Displaying calculated values for last 4 quarters")

if __name__ == "__main__":
    main()