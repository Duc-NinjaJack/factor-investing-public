#!/usr/bin/env python3
"""
Banking Sector Intermediary Values Display
=========================================
Displays calculated banking intermediary values from the database in tabular format
Similar to fundamental validation reports but with banking-specific metrics
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

def format_value(value, decimals=1):
    """Format value as billions with comma separator"""
    if pd.isna(value) or value is None:
        return "-"
    
    # Convert to float if string
    if isinstance(value, str):
        try:
            value = float(value)
        except:
            return value
    
    # For ratios, percentages, and days metrics, show as regular number
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

def extract_banking_intermediary_data(ticker, connection, periods=12):
    """Extract banking intermediary values for a specific ticker"""
    
    query = f"""
    SELECT * FROM intermediary_calculations_banking
    WHERE ticker = %s
    ORDER BY year DESC, quarter DESC
    LIMIT {periods}
    """
    
    try:
        with connection.cursor() as cursor:
            cursor.execute(query, (ticker,))
            results = cursor.fetchall()
            
        if not results:
            print(f"No intermediary data found for {ticker}")
            return None
            
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Sort chronologically for display
        df = df.sort_values(['year', 'quarter'])
        return df
        
    except Exception as e:
        print(f"❌ Error extracting data for {ticker}: {e}")
        return None

def create_banking_intermediary_display(ticker, df):
    """Create formatted display of banking intermediary values"""
    
    # Create quarter labels
    quarters = []
    for _, row in df.iterrows():
        quarters.append(f"Q{int(row['quarter'])}/{int(row['year'])}")
    
    # Create sections of data
    sections = []
    
    # 1. Banking Income Statement TTM Metrics
    income_ttm_items = [
        ('Net Interest Income (NII)', 'NII_TTM'),
        ('Interest Income', 'InterestIncome_TTM'),
        ('Interest Expense', 'InterestExpense_TTM'),
        ('Net Fee Income', 'NetFeeIncome_TTM'),
        ('Trading Income', 'TradingIncome_TTM'),
        ('Other Income', 'OtherIncome_TTM'),
        ('Total Operating Income', 'TotalOperatingIncome_TTM'),
        ('Operating Expenses', 'OperatingExpenses_TTM'),
        ('Operating Profit', 'OperatingProfit_TTM'),
        ('Credit Provisions', 'CreditProvisions_TTM'),
        ('Pre-Tax Profit', 'ProfitBeforeTax_TTM'),
        ('Net Profit', 'NetProfit_TTM')
    ]
    
    income_data = []
    for label, col in income_ttm_items:
        if col in df.columns:
            values = [format_value(df.iloc[i][col]) for i in range(len(df))]
            income_data.append([label] + values)
    
    sections.append({
        'title': 'BANKING INCOME STATEMENT - TTM VALUES (Billions VND)',
        'data': income_data
    })
    
    # 2. Banking Balance Sheet Averages
    balance_avg_items = [
        ('Total Assets', 'AvgTotalAssets'),
        ('Gross Loans', 'AvgGrossLoans'),
        ('Net Loans', 'AvgNetLoans'),
        ('Customer Deposits', 'AvgCustomerDeposits'),
        ('Total Equity', 'AvgTotalEquity'),
        ('Cash & Equivalents', 'AvgCash'),
        ('Investment Securities', 'AvgInvestmentSecurities'),
        ('Trading Securities', 'AvgTradingSecurities')
    ]
    
    balance_data = []
    for label, col in balance_avg_items:
        if col in df.columns:
            values = [format_value(df.iloc[i][col]) for i in range(len(df))]
            balance_data.append([label] + values)
    
    sections.append({
        'title': 'BANKING BALANCE SHEET - 5-POINT AVERAGES (Billions VND)',
        'data': balance_data
    })
    
    # 3. Banking-Specific Metrics
    banking_metrics = [
        ('Net Interest Margin (%)', 'NIM', 2),
        ('Loan-to-Deposit Ratio (%)', 'LDR', 1),
        ('Cost of Credit (%)', 'Cost_of_Credit', 2),
        ('Cost-to-Income Ratio (%)', 'Cost_Income_Ratio', 1),
        ('CAR Proxy (%)', 'CAR_Proxy', 1),
        ('ROAA (%)', 'ROAA', 2),
        ('ROAE (%)', 'ROAE', 1),
        ('Fee Income Ratio (%)', 'Fee_Income_Ratio', 1),
        ('Non-Interest Income Ratio (%)', 'NonInterest_Income_Ratio', 1)
    ]
    
    metrics_data = []
    for label, col, decimals in banking_metrics:
        if col in df.columns:
            values = [format_value(df.iloc[i][col], decimals) for i in range(len(df))]
            metrics_data.append([label] + values)
    
    sections.append({
        'title': 'BANKING-SPECIFIC METRICS',
        'data': metrics_data
    })
    
    # 4. Data Quality Indicators
    quality_items = [
        ('Quarters Available (TTM)', 'quarters_available_ttm', 0),
        ('Full TTM Data', 'has_full_ttm', 0),
        ('Avg Points Used', 'avg_points_used', 0),
        ('Full Avg Data', 'has_full_avg', 0),
        ('Data Quality Score (%)', 'data_quality_score', 0)
    ]
    
    quality_data = []
    for label, col, decimals in quality_items:
        if col in df.columns:
            if col in ['has_full_ttm', 'has_full_avg']:
                values = ['Yes' if df.iloc[i][col] else 'No' for i in range(len(df))]
            else:
                values = [format_value(df.iloc[i][col], decimals) for i in range(len(df))]
            quality_data.append([label] + values)
    
    sections.append({
        'title': 'DATA QUALITY INDICATORS',
        'data': quality_data
    })
    
    # Create the display
    display_lines = []
    display_lines.append(f"\n{'='*120}")
    display_lines.append(f"BANKING INTERMEDIARY VALUES - {ticker}")
    display_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    display_lines.append(f"{'='*120}\n")
    
    # Add period headers
    period_headers = ['Item'] + quarters
    
    for section in sections:
        display_lines.append(f"\n{section['title']}")
        display_lines.append('-' * 120)
        
        # Create DataFrame for nice tabular display
        section_df = pd.DataFrame(section['data'], columns=period_headers)
        display_lines.append(section_df.to_string(index=False))
    
    display_lines.append(f"\n{'='*120}")
    
    return '\n'.join(display_lines)

def save_display_to_file(ticker, display_content):
    """Save the display content to a markdown file with proper formatting"""
    # Create output directory if it doesn't exist
    output_dir = project_root / 'docs' / '4_validation_and_quality' / 'intermediaries' / 'banking'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create filename
    filename = f"{ticker}_banking_intermediary_values.md"
    filepath = output_dir / filename
    
    # Format content for markdown with code blocks
    markdown_content = f"""# Banking Intermediary Values - {ticker}

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
Complete banking intermediary calculations including TTM values, 5-point averages, and derived metrics.

## Raw Data Output

```
{display_content}
```

---
*Generated by Banking Sector Intermediary Display Script*
"""
    
    # Save to file
    with open(filepath, 'w') as f:
        f.write(markdown_content)
    
    print(f"✅ Display saved to: {filepath}")
    return filepath

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Display banking intermediary values')
    parser.add_argument('ticker', help='Banking ticker symbol (e.g., VCB, TCB, OCB)')
    parser.add_argument('--periods', type=int, default=12, help='Number of periods to display (default: 12)')
    parser.add_argument('--save', action='store_true', help='Save output to file')
    
    args = parser.parse_args()
    ticker = args.ticker.upper()
    
    # Connect to database
    connection = connect_to_database()
    if not connection:
        return
    
    try:
        # Extract data
        print(f"\n📊 Extracting banking intermediary data for {ticker}...")
        df = extract_banking_intermediary_data(ticker, connection, args.periods)
        
        if df is None or df.empty:
            print(f"❌ No data found for {ticker}")
            return
        
        # Create display
        display_content = create_banking_intermediary_display(ticker, df)
        
        # Print to console
        print(display_content)
        
        # Save to file if requested
        if args.save:
            save_display_to_file(ticker, display_content)
    
    finally:
        connection.close()

if __name__ == "__main__":
    main()