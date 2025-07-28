#!/usr/bin/env python3
"""
Tech Sector Intermediary Calculator - CORRECTED
=============================================
Phase 2: Pure intermediary calculations ONLY (not factors yet)

Intermediaries are:
- TTM sums (Revenue_TTM, COGS_TTM, etc.)  
- 5-point averages (AvgAssets_5Point, etc.)
- Working capital metrics (DSO_Enhanced, etc.)

Factors will be calculated in Phase 3 using these intermediaries.
"""

import sys
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
import logging
import pymysql
import yaml

# Setup project path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TechSectorIntermediaryCalculator:
    """Pure Intermediary Calculator - Phase 2 Only"""
    
    def __init__(self):
        self.db_connection = self._get_db_connection()
    
    def __del__(self):
        if hasattr(self, 'db_connection') and self.db_connection:
            self.db_connection.close()
    
    def _get_db_connection(self):
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
            logger.error(f"Database connection failed: {e}")
            raise
    
    def get_tech_fundamental_data(self, ticker: str) -> pd.DataFrame:
        """Get tech fundamental data - need 8-9 quarters for calculations"""
        query = """
        SELECT 
            ticker, year, quarter,
            -- Income Statement (Flow items for TTM)
            NetRevenue, COGS, GrossProfit, SellingExpenses, AdminExpenses,
            FinancialIncome, FinancialExpenses, InterestExpenses,
            ProfitBeforeTax, CurrentIncomeTax, TotalIncomeTax,
            NetProfit, NetProfitAfterMI, EBIT,
            -- Balance Sheet (Stock items for 5-point averaging)
            TotalAssets, CurrentAssets, CashAndCashEquivalents,
            AccountsReceivable, Inventory, FixedAssets,
            TotalLiabilities, CurrentLiabilities, AccountsPayable,
            ShortTermDebt, LongTermDebt, TotalEquity,
            CharterCapital, RetainedEarnings,
            -- Cash Flow (Flow items for TTM) - FINAL CORRECTION
            NetCFO, NetCFI, NetCFF, DepreciationAmortization, CapEx,
            DividendsPaid, ShareIssuanceProceeds, ShareRepurchase
        FROM v_comprehensive_fundamental_items
        WHERE ticker = %s 
        AND year BETWEEN 2022 AND 2025
        ORDER BY year, quarter
        """
        
        try:
            with self.db_connection.cursor() as cursor:
                cursor.execute(query, [ticker])
                results = cursor.fetchall()
                
                if not results:
                    logger.warning(f"No data found for {ticker}")
                    return pd.DataFrame()
                
                result_df = pd.DataFrame(results)
                logger.info(f"Retrieved {len(result_df)} quarters of data for {ticker}")
                return result_df
                
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return pd.DataFrame()
    
    def calculate_quarterly_intermediaries(self, ticker: str) -> dict:
        """Calculate intermediaries for each of the last 4 quarters"""
        data = self.get_tech_fundamental_data(ticker)
        if data.empty:
            return {}
        
        # Target last 4 quarters 
        target_quarters = [(2025, 1), (2024, 4), (2024, 3), (2024, 2)]
        results = {}
        
        for target_year, target_quarter in target_quarters:
            logger.info(f"Calculating intermediaries for {ticker} {target_year}Q{target_quarter}")
            
            # Calculate pure intermediaries only
            ttm_intermediaries = self.calculate_ttm_intermediaries(data, target_year, target_quarter)
            avg_intermediaries = self.calculate_5point_intermediaries(data, target_year, target_quarter)
            wc_intermediaries = self.calculate_working_capital_intermediaries(ttm_intermediaries, avg_intermediaries)
            
            # Combine intermediaries
            quarter_result = {
                'ticker': ticker,
                'year': target_year,
                'quarter': target_quarter,
                'calc_date': datetime.now().strftime('%Y-%m-%d'),
                **ttm_intermediaries,
                **avg_intermediaries,
                **wc_intermediaries
            }
            
            results[f"{target_year}Q{target_quarter}"] = quarter_result
            
            # Count intermediaries (exclude metadata)
            intermediary_count = len([k for k, v in quarter_result.items() 
                                    if k not in ['ticker', 'year', 'quarter', 'calc_date'] 
                                    and isinstance(v, (int, float)) and not pd.isna(v)])
            
            logger.info(f"✅ {intermediary_count} intermediaries calculated for {ticker} {target_year}Q{target_quarter}")
        
        return results
    
    def calculate_ttm_intermediaries(self, data: pd.DataFrame, target_year: int, target_quarter: int) -> dict:
        """Calculate TTM intermediaries - sum flows over 4 quarters"""
        ttm_quarters = self._get_ttm_quarters(target_year, target_quarter)
        data['period_key'] = data['year'].astype(str) + 'Q' + data['quarter'].astype(str)
        ttm_data = data[data['period_key'].isin([f"{y}Q{q}" for y, q in ttm_quarters])]
        
        if len(ttm_data) < 4:
            logger.warning(f"Insufficient TTM data: {len(ttm_data)} quarters")
            return {}
        
        ttm_intermediaries = {}
        
        # TTM Flow Items - FINAL CORRECTION
        ttm_items = [
            'NetRevenue', 'COGS', 'GrossProfit', 'SellingExpenses', 'AdminExpenses',
            'FinancialIncome', 'FinancialExpenses', 'InterestExpenses', 
            'ProfitBeforeTax', 'CurrentIncomeTax', 'TotalIncomeTax',
            'NetProfit', 'NetProfitAfterMI',
            'NetCFO', 'NetCFI', 'NetCFF', 'DepreciationAmortization', 'CapEx',
            'DividendsPaid', 'ShareIssuanceProceeds', 'ShareRepurchase'
        ]
        
        for item in ttm_items:
            if item in ttm_data.columns:
                values = pd.to_numeric(ttm_data[item], errors='coerce').fillna(0)
                ttm_intermediaries[f"{item}_TTM"] = values.sum() if not values.isna().all() else 0.0
        
        # Vietnamese EBIT intermediary
        if all(f"{col}_TTM" in ttm_intermediaries for col in ['GrossProfit', 'SellingExpenses', 'AdminExpenses']):
            ttm_intermediaries['EBIT_TTM'] = (
                ttm_intermediaries['GrossProfit_TTM'] - 
                ttm_intermediaries['SellingExpenses_TTM'] - 
                ttm_intermediaries['AdminExpenses_TTM']
            )
        
        # EBITDA intermediary
        if 'EBIT_TTM' in ttm_intermediaries and 'DepreciationAmortization_TTM' in ttm_intermediaries:
            ttm_intermediaries['EBITDA_TTM'] = ttm_intermediaries['EBIT_TTM'] + ttm_intermediaries['DepreciationAmortization_TTM']
        
        # FCF intermediary
        if 'NetCFO_TTM' in ttm_intermediaries and 'CapEx_TTM' in ttm_intermediaries:
            ttm_intermediaries['FCF_TTM'] = ttm_intermediaries['NetCFO_TTM'] + ttm_intermediaries['CapEx_TTM']
        
        logger.info(f"Calculated {len(ttm_intermediaries)} TTM intermediaries")
        return ttm_intermediaries
    
    def calculate_5point_intermediaries(self, data: pd.DataFrame, target_year: int, target_quarter: int) -> dict:
        """Calculate 5-point balance sheet intermediaries - average stocks over 5 quarters"""
        avg_quarters = self._get_5point_quarters(target_year, target_quarter)
        data['period_key'] = data['year'].astype(str) + 'Q' + data['quarter'].astype(str)
        avg_data = data[data['period_key'].isin([f"{y}Q{q}" for y, q in avg_quarters])]
        
        if len(avg_data) < 5:
            logger.warning(f"Insufficient 5-point data: {len(avg_data)} quarters")
            return {}
        
        avg_intermediaries = {}
        
        # 5-Point Balance Sheet Items
        balance_sheet_items = [
            'TotalAssets', 'CurrentAssets', 'CashAndCashEquivalents',
            'AccountsReceivable', 'Inventory', 'FixedAssets',
            'TotalLiabilities', 'CurrentLiabilities', 'AccountsPayable',
            'ShortTermDebt', 'LongTermDebt', 'TotalEquity',
            'CharterCapital', 'RetainedEarnings'
        ]
        
        for item in balance_sheet_items:
            if item in avg_data.columns:
                values = pd.to_numeric(avg_data[item], errors='coerce').fillna(0)
                if not values.isna().all():
                    avg_intermediaries[f"Avg{item}_5Point"] = values.mean()
        
        # Derived intermediaries
        if 'AvgShortTermDebt_5Point' in avg_intermediaries and 'AvgLongTermDebt_5Point' in avg_intermediaries:
            avg_intermediaries['AvgTotalDebt_5Point'] = avg_intermediaries['AvgShortTermDebt_5Point'] + avg_intermediaries['AvgLongTermDebt_5Point']
        
        if 'AvgTotalDebt_5Point' in avg_intermediaries and 'AvgCashAndCashEquivalents_5Point' in avg_intermediaries:
            avg_intermediaries['AvgNetDebt_5Point'] = avg_intermediaries['AvgTotalDebt_5Point'] - avg_intermediaries['AvgCashAndCashEquivalents_5Point']
        
        if 'AvgCurrentAssets_5Point' in avg_intermediaries and 'AvgCurrentLiabilities_5Point' in avg_intermediaries:
            avg_intermediaries['AvgWorkingCapital_5Point'] = avg_intermediaries['AvgCurrentAssets_5Point'] - avg_intermediaries['AvgCurrentLiabilities_5Point']
        
        logger.info(f"Calculated {len(avg_intermediaries)} 5-point intermediaries")
        return avg_intermediaries
    
    def calculate_working_capital_intermediaries(self, ttm_intermediaries: dict, avg_intermediaries: dict) -> dict:
        """Calculate enhanced working capital intermediaries"""
        wc_intermediaries = {}
        
        # DSO Enhanced
        if 'AvgAccountsReceivable_5Point' in avg_intermediaries and 'NetRevenue_TTM' in ttm_intermediaries:
            if ttm_intermediaries['NetRevenue_TTM'] != 0:
                wc_intermediaries['DSO_Enhanced'] = (avg_intermediaries['AvgAccountsReceivable_5Point'] * 365) / ttm_intermediaries['NetRevenue_TTM']
        
        # DIO Enhanced
        if 'AvgInventory_5Point' in avg_intermediaries and 'COGS_TTM' in ttm_intermediaries:
            if ttm_intermediaries['COGS_TTM'] != 0:
                wc_intermediaries['DIO_Enhanced'] = (avg_intermediaries['AvgInventory_5Point'] * 365) / abs(ttm_intermediaries['COGS_TTM'])
        
        # DPO Enhanced
        if 'AvgAccountsPayable_5Point' in avg_intermediaries and 'COGS_TTM' in ttm_intermediaries:
            if ttm_intermediaries['COGS_TTM'] != 0:
                wc_intermediaries['DPO_Enhanced'] = (avg_intermediaries['AvgAccountsPayable_5Point'] * 365) / abs(ttm_intermediaries['COGS_TTM'])
        
        # CCC Enhanced
        if all(metric in wc_intermediaries for metric in ['DSO_Enhanced', 'DIO_Enhanced', 'DPO_Enhanced']):
            wc_intermediaries['CCC_Enhanced'] = wc_intermediaries['DSO_Enhanced'] + wc_intermediaries['DIO_Enhanced'] - wc_intermediaries['DPO_Enhanced']
        
        logger.info(f"Calculated {len(wc_intermediaries)} working capital intermediaries")
        return wc_intermediaries
    
    def _get_ttm_quarters(self, year: int, quarter: int) -> list:
        """Get 4 quarters for TTM"""
        quarters = []
        for i in range(4):
            q = quarter - i
            y = year
            if q <= 0:
                q += 4
                y -= 1
            quarters.append((y, q))
        return quarters[::-1]
    
    def _get_5point_quarters(self, year: int, quarter: int) -> list:
        """Get 5 quarters for 5-point averaging"""
        quarters = []
        for i in range(5):
            q = quarter - i
            y = year
            if q <= 0:
                q += 4
                y -= 1
            quarters.append((y, q))
        return quarters[::-1]
    
    def generate_report(self, ticker: str) -> str:
        """Generate quarterly intermediary validation report"""
        results = self.calculate_quarterly_intermediaries(ticker)
        
        if not results:
            return "No data available"
        
        report = f"""# {ticker} Tech Intermediary Validation - FINAL CORRECTED
**Phase:** 2 - Pure Intermediaries Only
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- **Quarters:** {len(results)} processed
- **Focus:** Pure intermediaries only (factors in Phase 3)
- **Results:** Quarter-by-quarter breakdown

"""
        
        for quarter_key in sorted(results.keys(), reverse=True):
            quarter_result = results[quarter_key]
            
            ttm_count = len([k for k in quarter_result.keys() if k.endswith('_TTM')])
            avg_count = len([k for k in quarter_result.keys() if k.endswith('_5Point')])
            wc_count = len([k for k in quarter_result.keys() if 'Enhanced' in k])
            
            report += f"""
## {quarter_key} - {ttm_count + avg_count + wc_count} Intermediaries

### TTM Intermediaries ({ttm_count})
| Name | Value (VND) | Type |
|------|-------------|------|
"""
            for key, value in sorted(quarter_result.items()):
                if key.endswith('_TTM') and isinstance(value, (int, float)):
                    report += f"| {key} | {value:,.0f} | TTM Sum |\n"
            
            report += f"""
### 5-Point Averages ({avg_count})
| Name | Value (VND) | Type |
|------|-------------|------|
"""
            for key, value in sorted(quarter_result.items()):
                if key.endswith('_5Point') and isinstance(value, (int, float)):
                    report += f"| {key} | {value:,.0f} | 5Q Average |\n"
            
            report += f"""
### Working Capital ({wc_count})
| Name | Value | Type |
|------|-------|------|
"""
            for key, value in sorted(quarter_result.items()):
                if 'Enhanced' in key and isinstance(value, (int, float)):
                    unit = "Days" if key.endswith('Enhanced') else ""
                    report += f"| {key} | {value:.2f} {unit} | Enhanced |\n"
        
        report += "\n✅ **Phase 2 Complete** - Ready for Phase 3 factor calculations"
        return report

def main():
    parser = argparse.ArgumentParser(description='Tech Sector Pure Intermediaries')
    parser.add_argument('--ticker', default='FPT', help='Ticker to process')
    args = parser.parse_args()
    
    calculator = TechSectorIntermediaryCalculator()
    report = calculator.generate_report(args.ticker)
    
    # Save report - CORRECTED PATH
    output_path = project_root / f"docs/1_governance/phase2_workstream2/validation_outputs/technology_sector/{args.ticker}_intermediary_validation.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"✅ Report saved to correct path: {output_path}")
    print("📊 Quarter-by-quarter intermediaries calculated")
    print("🔬 Pure intermediaries only (not factors)")
    print("\n" + "="*50)
    print(report[:1000] + "...")

if __name__ == "__main__":
    main() 