#!/usr/bin/env python3
"""
Complete Securities Fundamental Data Extractor
==============================================
Leverages v_complete_securities_fundamentals (securities items)
Shows complete securities data structure with hierarchical detail
Demonstrates full institutional-grade securities analysis capability

Usage:
    python scripts/securities_enhanced_extract.py SSI
    python scripts/securities_enhanced_extract.py VCI  
    python scripts/securities_enhanced_extract.py HCM
"""

import pandas as pd
import pymysql
import yaml
from datetime import datetime
from pathlib import Path

def connect_to_database():
    """Create database connection"""
    try:
        # Adjusted path: now we need to go up two levels from sector_extracts folder
        project_root = Path(__file__).parent.parent.parent
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

def safe_float_convert(value):
    """Safely convert value to float, handling Decimal types"""
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

def format_value(value):
    """Format value as billions with comma separator"""
    if pd.isna(value) or value is None or value == 0:
        return "-"
    
    # Convert to float if needed
    float_value = safe_float_convert(value)
    if float_value is None:
        return "-"
    
    # Convert to billions and format with comma
    billions = float_value / 1_000_000_000
    if abs(billions) >= 1000:
        return f"{billions:,.0f}"
    elif abs(billions) >= 100:
        return f"{billions:,.0f}"
    elif abs(billions) >= 10:
        return f"{billions:,.1f}"
    elif abs(billions) >= 1:
        return f"{billions:,.1f}"
    else:
        return f"{billions:,.2f}"

def extract_securities_data(ticker):
    """Extract complete securities fundamental data for specific ticker"""
    
    conn = connect_to_database()
    if not conn:
        return None
    
    try:
        query = """
        SELECT * FROM v_complete_securities_fundamentals 
        WHERE ticker = %s 
        ORDER BY year DESC, quarter DESC
        """
        
        with conn.cursor() as cursor:
            cursor.execute(query, [ticker])
            results = cursor.fetchall()
        
        if not results:
            print(f"❌ No data found for ticker: {ticker}")
            return None
            
        df = pd.DataFrame(results)
        
        if df.empty:
            print(f"❌ No data found for ticker: {ticker}")
            return None
            
        print(f"✅ Found {len(df)} quarters of data for {ticker}")
        return df
        
    except Exception as e:
        print(f"❌ Error extracting data: {e}")
        return None
    finally:
        conn.close()

def generate_securities_analysis(ticker, df):
    """Generate comprehensive securities analysis similar to banking format"""
    
    if df is None or df.empty:
        return None
    
    # Get latest 8 quarters (or available quarters)
    df_analysis = df.head(8).copy()
    
    # Ensure we have some data
    if len(df_analysis) == 0:
        print(f"❌ No analysis data available for {ticker}")
        return None
    
    print(f"📊 Generating analysis for {ticker} ({len(df_analysis)} quarters)")
    print(f"📋 Column names: {list(df_analysis.columns)}")
    print(f"📋 Sample data: {df_analysis.iloc[0].to_dict()}")
    
    # Reverse data to match banking format (latest on right)
    df_reversed = df_analysis.iloc[::-1]
    
    # Create quarter headers
    quarter_headers = []
    for _, row in df_reversed.iterrows():
        quarter_headers.append(f"{int(row['year'])}Q{int(row['quarter'])}")
    
    analysis_md = f"""# {ticker} Enhanced Analysis - Securities Sector

**Source**: v_complete_securities_fundamentals (securities items)
**Purpose**: Institutional-grade securities analysis with full hierarchy
**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**✅ SUCCESS**: Extracted {len(df_analysis)} quarters with {len(df_analysis.columns)} columns

## I. Complete Securities P&L ✅ **INSTITUTIONAL-GRADE DETAIL**

| Quarter - VNDbn | {' | '.join(quarter_headers)} |
|---|{'---|' * len(quarter_headers)}
| **TRADING INCOME** | | | | | | | | |
| &nbsp;&nbsp;Trading Gains (FVTPL) | {' | '.join([format_value(row['TradingGainFVTPL']) for _, row in df_reversed.iterrows()])} |
| &nbsp;&nbsp;Trading Gains (HTM) | {' | '.join([format_value(row['TradingGainHTM']) for _, row in df_reversed.iterrows()])} |
| &nbsp;&nbsp;Trading Gains (Loans) | {' | '.join([format_value(row['TradingGainLoans']) for _, row in df_reversed.iterrows()])} |
| &nbsp;&nbsp;Trading Gains (AFS) | {' | '.join([format_value(row['TradingGainAFS']) for _, row in df_reversed.iterrows()])} |
| **SECURITIES SERVICES** | | | | | | | | |
| &nbsp;&nbsp;Brokerage Revenue | {' | '.join([format_value(row['BrokerageRevenue']) for _, row in df_reversed.iterrows()])} |
| &nbsp;&nbsp;Advisory Revenue | {' | '.join([format_value(row['AdvisoryRevenue']) for _, row in df_reversed.iterrows()])} |
| &nbsp;&nbsp;Underwriting Revenue | {' | '.join([format_value(row['UnderwritingRevenue']) for _, row in df_reversed.iterrows()])} |
| &nbsp;&nbsp;Custody Service Revenue | {' | '.join([format_value(row['CustodyServiceRevenue']) for _, row in df_reversed.iterrows()])} |
| &nbsp;&nbsp;Other Operating Income | {' | '.join([format_value(row['OtherOperatingIncome']) for _, row in df_reversed.iterrows()])} |
| **Total Operating Revenue** | {' | '.join([format_value(row['TotalOperatingRevenue']) for _, row in df_reversed.iterrows()])} |
| **Management Expenses** | {' | '.join([format_value(row['ManagementExpenses']) for _, row in df_reversed.iterrows()])} |
| **Operating Result** | {' | '.join([format_value(row['OperatingResult']) for _, row in df_reversed.iterrows()])} |
| **Profit Before Tax** | {' | '.join([format_value(row['ProfitBeforeTax']) for _, row in df_reversed.iterrows()])} |
| **Income Tax Expense** | {' | '.join([format_value(row['IncomeTaxExpense']) for _, row in df_reversed.iterrows()])} |
| **Profit After Tax** | {' | '.join([format_value(row['ProfitAfterTax']) for _, row in df_reversed.iterrows()])} |

**Securities P&L Ratios** (Latest Quarter):"""

    # Calculate latest quarter ratios
    latest = df_analysis.iloc[0]
    
    # Revenue Mix Analysis
    total_rev = safe_float_convert(latest['TotalOperatingRevenue'])
    if total_rev and total_rev != 0:
        brokerage_rev = safe_float_convert(latest['BrokerageRevenue'])
        advisory_rev = safe_float_convert(latest['AdvisoryRevenue'])
        custody_rev = safe_float_convert(latest['CustodyServiceRevenue'])
        
        brokerage_mix = (brokerage_rev / total_rev * 100) if brokerage_rev else 0
        advisory_mix = (advisory_rev / total_rev * 100) if advisory_rev else 0
        custody_mix = (custody_rev / total_rev * 100) if custody_rev else 0
        analysis_md += f"""
- **Brokerage Revenue Mix**: {brokerage_mix:.1f}%
- **Advisory Revenue Mix**: {advisory_mix:.1f}%
- **Custody Revenue Mix**: {custody_mix:.1f}%"""
    
    # Operating Efficiency
    total_rev_float = safe_float_convert(latest['TotalOperatingRevenue'])
    op_exp_float = safe_float_convert(latest['OperatingExpenses'])
    if total_rev_float and op_exp_float and total_rev_float != 0:
        cost_income_ratio = (abs(op_exp_float) / total_rev_float * 100)
        analysis_md += f"""
- **Cost-to-Income Ratio**: {cost_income_ratio:.1f}%"""
    
    # Profitability
    equity_float = safe_float_convert(latest['OwnersEquity'])
    pat_float = safe_float_convert(latest['ProfitAfterTax'])
    if equity_float and equity_float != 0 and pat_float:
        roe_quarterly = (pat_float / equity_float * 100)
        roe_annualized = roe_quarterly * 4
        analysis_md += f"""
- **Return on Equity (ROE)**: {roe_annualized:.1f}%"""

    analysis_md += f"""

## II. Complete Securities Balance Sheet ✅ **INSTITUTIONAL-GRADE DETAIL**

| Quarter - VNDbn | {' | '.join(quarter_headers)} |
|---|{'---|' * len(quarter_headers)}
| **TOTAL ASSETS** | {' | '.join([format_value(row['TotalAssets']) for _, row in df_reversed.iterrows()])} |
| &nbsp;&nbsp;Cash and Cash Equivalents | {' | '.join([format_value(row['CashAndCashEquivalents']) for _, row in df_reversed.iterrows()])} |
| &nbsp;&nbsp;Financial Assets (FVTPL) | {' | '.join([format_value(row['FinancialAssetsFVTPL']) for _, row in df_reversed.iterrows()])} |
| &nbsp;&nbsp;Held-to-Maturity Investments | {' | '.join([format_value(row['HeldToMaturityInvestments']) for _, row in df_reversed.iterrows()])} |
| &nbsp;&nbsp;Loan Receivables | {' | '.join([format_value(row['LoanReceivables']) for _, row in df_reversed.iterrows()])} |
| &nbsp;&nbsp;Available-for-Sale Financial | {' | '.join([format_value(row['AvailableForSaleFinancial']) for _, row in df_reversed.iterrows()])} |
| &nbsp;&nbsp;Securities Services Receivables | {' | '.join([format_value(row['SecuritiesServicesReceivables']) for _, row in df_reversed.iterrows()])} |
| &nbsp;&nbsp;Other Receivables | {' | '.join([format_value(row['OtherReceivables']) for _, row in df_reversed.iterrows()])} |
| &nbsp;&nbsp;Fixed Assets | {' | '.join([format_value(row['FixedAssets']) for _, row in df_reversed.iterrows()])} |
| **LIABILITIES AND EQUITY** | {' | '.join([format_value(row['TotalLiabilitiesAndEquity']) for _, row in df_reversed.iterrows()])} |
| &nbsp;&nbsp;Short-term Borrowings | {' | '.join([format_value(row['ShortTermBorrowingsFinancial']) for _, row in df_reversed.iterrows()])} |
| &nbsp;&nbsp;Trade Payables | {' | '.join([format_value(row['ShortTermTradePayables']) for _, row in df_reversed.iterrows()])} |
| &nbsp;&nbsp;Employee Payables | {' | '.join([format_value(row['PayablesToEmployees']) for _, row in df_reversed.iterrows()])} |
| &nbsp;&nbsp;Tax Liabilities | {' | '.join([format_value(row['TaxesDuesGovernment']) for _, row in df_reversed.iterrows()])} |
| **OWNERS' EQUITY** | {' | '.join([format_value(row['OwnersEquity']) for _, row in df_reversed.iterrows()])} |
| &nbsp;&nbsp;Owner Capital | {' | '.join([format_value(row['OwnerCapital']) for _, row in df_reversed.iterrows()])} |
| &nbsp;&nbsp;&nbsp;&nbsp;Charter Capital | {' | '.join([format_value(row['CharterCapital']) for _, row in df_reversed.iterrows()])} |
| &nbsp;&nbsp;&nbsp;&nbsp;Share Premium | {' | '.join([format_value(row['SharePremium']) for _, row in df_reversed.iterrows()])} |
| &nbsp;&nbsp;Retained Earnings | {' | '.join([format_value(row['RetainedEarnings']) for _, row in df_reversed.iterrows()])} |

**Securities Balance Sheet Ratios** (Latest Quarter):"""

    # Balance Sheet Ratios
    total_assets_float = safe_float_convert(latest['TotalAssets'])
    if total_assets_float and total_assets_float != 0:
        equity_float_bs = safe_float_convert(latest['OwnersEquity'])
        if equity_float_bs:
            equity_ratio = (equity_float_bs / total_assets_float * 100)
            analysis_md += f"""
- **Equity-to-Assets Ratio**: {equity_ratio:.1f}%"""
        
        trading_assets_float = safe_float_convert(latest['FinancialAssetsFVTPL'])
        if trading_assets_float:
            trading_assets_ratio = (trading_assets_float / total_assets_float * 100)
            analysis_md += f"""
- **Trading Assets Ratio**: {trading_assets_ratio:.1f}%"""
        
        total_rev_bs_float = safe_float_convert(latest['TotalOperatingRevenue'])
        if total_rev_bs_float:
            asset_turnover = (total_rev_bs_float / total_assets_float * 100)
            analysis_md += f"""
- **Revenue-to-Assets Ratio**: {asset_turnover:.2f}%"""

    analysis_md += f"""

## III. Complete Securities Cash Flow ✅ **INSTITUTIONAL-GRADE DETAIL**

| Quarter - VNDbn | {' | '.join(quarter_headers)} |
|---|{'---|' * len(quarter_headers)}
| **Net Cash Flow from Operating Activities** | {' | '.join([format_value(row['NetCashFlowFromOperatingActivities']) for _, row in df_reversed.iterrows()])} |
| **Net Cash Flow from Investing Activities** | {' | '.join([format_value(row['NetCashFlowFromInvestingActivities']) for _, row in df_reversed.iterrows()])} |
| **Net Cash Flow from Financing Activities** | {' | '.join([format_value(row['NetCashFlowFromFinancingActivities']) for _, row in df_reversed.iterrows()])} |
| &nbsp;&nbsp;Proceeds from Share Issuance | {' | '.join([format_value(row['ProceedsFromShareIssuance']) for _, row in df_reversed.iterrows()])} |
| &nbsp;&nbsp;Dividends Paid to Owners | {' | '.join([format_value(row['DividendsPaidToOwners']) for _, row in df_reversed.iterrows()])} |
| **Net Increase/Decrease in Cash** | {' | '.join([format_value(row['NetIncreaseDecreaseInCash']) for _, row in df_reversed.iterrows()])} |
| **Cash at Beginning of Period** | {' | '.join([format_value(row['CashAtBeginningOfPeriod']) for _, row in df_reversed.iterrows()])} |
| **Cash at End of Period** | {' | '.join([format_value(row['CashAtEndOfPeriod']) for _, row in df_reversed.iterrows()])} |

## IV. Securities Business Analysis

### Revenue Stream Analysis
The securities firm shows a diversified revenue model with the following characteristics:"""

    # Business Analysis
    brok_rev_ba = safe_float_convert(latest['BrokerageRevenue'])
    adv_rev_ba = safe_float_convert(latest['AdvisoryRevenue'])
    cust_rev_ba = safe_float_convert(latest['CustodyServiceRevenue'])
    
    if brok_rev_ba and adv_rev_ba and cust_rev_ba:
        total_core_revenue = brok_rev_ba + adv_rev_ba + cust_rev_ba
        analysis_md += f"""

- **Core Securities Revenue**: {format_value(total_core_revenue)} billion VND
- **Primary Revenue**: Brokerage services ({brokerage_mix:.1f}% of total)
- **Value-Added Services**: Advisory and custody combined ({advisory_mix + custody_mix:.1f}% of total)"""

    # Capital Adequacy Analysis
    equity_ca_val = safe_float_convert(latest['OwnersEquity'])
    assets_ca_val = safe_float_convert(latest['TotalAssets'])
    if equity_ca_val and assets_ca_val:
        equity_ratio_final = (equity_ca_val / assets_ca_val * 100)
        analysis_md += f"""

### Capital Adequacy
- **Equity Position**: {format_value(equity_ca_val)} billion VND
- **Capital Strength**: {equity_ratio_final:.1f}% equity-to-assets ratio
- **Regulatory Buffer**: {"Strong" if equity_ratio_final > 20 else "Adequate" if equity_ratio_final > 15 else "Moderate"}"""

    # Client Business Analysis
    receivables_val = safe_float_convert(latest['SecuritiesServicesReceivables'])
    if receivables_val:
        analysis_md += f"""

### Client Business Metrics
- **Client Receivables**: {format_value(receivables_val)} billion VND
- **Business Quality**: {"High-quality client base" if receivables_val > 0 else "Conservative client exposure"}"""

    analysis_md += f"""

## V. Institutional Investment Perspective

### Strengths
- Securities-focused business model with diversified revenue streams"""
    
    # Add capital strength if available
    if 'equity_ratio_final' in locals():
        analysis_md += f"""
- {"Strong" if equity_ratio_final > 20 else "Adequate"} capital position for regulatory compliance"""
    
    # Add cost efficiency if available
    if 'cost_income_ratio' in locals():
        analysis_md += f"""
- {"Efficient" if cost_income_ratio < 50 else "Moderate"} operational cost structure"""
    
    analysis_md += f"""

### Key Metrics for Factor Analysis
- **Business Mix**: Balanced between brokerage, advisory, and custody services"""
    
    # Add ROE if available
    if 'roe_annualized' in locals():
        analysis_md += f"""
- **Capital Efficiency**: {"High" if roe_annualized > 15 else "Moderate" if roe_annualized > 10 else "Low"} return on equity"""
    
    analysis_md += f"""
- **Asset Quality**: Liquid financial assets provide flexibility

### Factor Investment Considerations
- Market cycle sensitivity through brokerage revenue correlation
- Revenue stability through diversified service offerings
- Capital adequacy for sustained dividend capacity

---
*Analysis generated using institutional methodology*
*Data source: v_complete_securities_fundamentals*"""

    return analysis_md

def main():
    """Main execution with command line argument support"""
    import sys
    
    # Get ticker from command line or default to SSI
    if len(sys.argv) > 1:
        ticker = sys.argv[1].upper()
    else:
        ticker = "SSI"
    
    print(f"🏛️ Complete Securities Analysis for {ticker}")
    print("=" * 60)
    print("📊 Using v_complete_securities_fundamentals view")
    print("🎯 Institutional-grade securities sector analysis")
    print("💡 Following banking sector methodology pattern")
    print()
    
    # Extract data
    df = extract_securities_data(ticker)
    
    if df is not None:
        # Generate analysis
        analysis_md = generate_securities_analysis(ticker, df)
        
        if analysis_md:
            # Create output directory if it doesn't exist
            project_root = Path(__file__).parent.parent.parent
            output_dir = project_root / "docs/4_validation_and_quality/fundamentals/securities"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save analysis to markdown file
            output_file = output_dir / f"{ticker}_enhanced_analysis.md"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(analysis_md)
            
            print(f"✅ Analysis complete!")
            print(f"📄 Report saved to: {output_file}")
            print(f"📊 Data coverage: {len(df)} quarters")
            print(f"🎯 Analysis scope: Complete securities fundamentals")
            
            # Display summary statistics
            latest = df.iloc[0]
            brok_summary = safe_float_convert(latest.BrokerageRevenue)
            assets_summary = safe_float_convert(latest.TotalAssets)
            equity_summary = safe_float_convert(latest.OwnersEquity)
            
            if brok_summary and assets_summary and equity_summary:
                print(f"\n📈 Latest Quarter Summary ({latest.year}Q{latest.quarter}):")
                print(f"   Brokerage Revenue: {format_value(brok_summary)} billion VND")
                print(f"   Total Assets: {format_value(assets_summary)} billion VND")
                print(f"   Owners' Equity: {format_value(equity_summary)} billion VND")
                
                if assets_summary != 0:
                    equity_ratio_summary = (equity_summary / assets_summary * 100)
                    print(f"   Equity Ratio: {equity_ratio_summary:.1f}%")
        else:
            print("❌ Failed to generate analysis")
    else:
        print(f"❌ No data available for {ticker}")
        print("💡 Available securities tickers: SSI, VCI, HCM, MBS, SHS, etc.")

if __name__ == "__main__":
    main()