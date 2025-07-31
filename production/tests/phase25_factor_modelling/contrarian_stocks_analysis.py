"""
Contrarian Stocks Analysis
==========================
Analyze which specific stocks in Agriculture and Mining & Oil sectors
are showing strong contrarian behavior in P/E and P/B factors.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime, timedelta

# Add the parent directory to the path to import database modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'database'))
from connection import get_engine

class ContrarianStocksAnalyzer:
    def __init__(self):
        self.engine = get_engine()
        
    def get_contrarian_stocks_data(self, analysis_date: datetime) -> pd.DataFrame:
        """Get detailed data for Agriculture and Mining & Oil stocks."""
        # Get tickers for Agriculture and Mining & Oil sectors
        sectors_query = """
        SELECT ticker, sector FROM master_info
        WHERE sector IN ('Agriculture', 'Mining & Oil')
        ORDER BY sector, ticker
        """
        
        sectors_df = pd.read_sql(sectors_query, self.engine)
        print(f"Found {len(sectors_df)} stocks in Agriculture and Mining & Oil sectors")
        print(sectors_df.groupby('sector').size())
        
        all_stocks_data = []
        
        for _, row in sectors_df.iterrows():
            ticker = row['ticker']
            sector = row['sector']
            
            try:
                # Get fundamental data
                fundamental_data = self.get_fundamental_data(ticker, analysis_date)
                
                # Get market data
                market_data = self.get_market_data(ticker, analysis_date)
                
                # Get forward returns
                forward_return = self.get_forward_return(ticker, analysis_date)
                
                if fundamental_data is not None and market_data is not None:
                    stock_data = {
                        'ticker': ticker,
                        'sector': sector,
                        'roaa': fundamental_data['roaa'],
                        'roae': fundamental_data['roae'],
                        'pe_ratio': fundamental_data['pe_ratio'],
                        'pb_ratio': fundamental_data['pb_ratio'],
                        'pe_score': fundamental_data['pe_score'],
                        'pb_score': fundamental_data['pb_score'],
                        'market_cap': market_data['market_cap'],
                        'close_price': market_data['close_price'],
                        'total_shares': market_data['total_shares'],
                        'forward_return': forward_return,
                        'net_profit_ttm': fundamental_data['net_profit_ttm'],
                        'total_equity': fundamental_data['total_equity'],
                        'total_assets': fundamental_data['total_assets']
                    }
                    all_stocks_data.append(stock_data)
                    
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
                continue
        
        return pd.DataFrame(all_stocks_data)
    
    def get_fundamental_data(self, ticker: str, analysis_date: datetime) -> dict:
        """Get fundamental data for a specific ticker."""
        # Try enhanced table first
        fundamental_query = """
        SELECT 
            NetProfit_TTM / AvgTotalAssets as roaa,
            NetProfit_TTM / AvgTotalEquity as roae,
            NetProfit_TTM,
            AvgTotalEquity,
            AvgTotalAssets
        FROM intermediary_calculations_enhanced
        WHERE ticker = %s AND quarter = (
            SELECT MAX(quarter) 
            FROM intermediary_calculations_enhanced 
            WHERE quarter <= %s
        )
        """
        
        fundamental_data = pd.read_sql(fundamental_query, self.engine, params=(ticker, analysis_date))
        
        if len(fundamental_data) == 0:
            # Try securities table
            fundamental_query = """
            SELECT 
                NetProfit_TTM / AvgTotalAssets as roaa,
                NetProfit_TTM / AvgTotalEquity as roae,
                NetProfit_TTM,
                AvgTotalEquity,
                AvgTotalAssets
            FROM intermediary_calculations_securities_cleaned
            WHERE ticker = %s AND quarter = (
                SELECT MAX(quarter) 
                FROM intermediary_calculations_securities_cleaned 
                WHERE quarter <= %s
            )
            """
            fundamental_data = pd.read_sql(fundamental_query, self.engine, params=(ticker, analysis_date))
        
        if len(fundamental_data) == 0:
            return None
        
        row = fundamental_data.iloc[0]
        
        # Get market data for value calculations
        market_cap_query = """
        SELECT trading_date, close_price, market_cap, total_shares
        FROM vcsc_daily_data_complete
        WHERE ticker = %s AND trading_date <= %s
        ORDER BY trading_date DESC
        LIMIT 1
        """
        
        market_data = pd.read_sql(market_cap_query, self.engine, params=(ticker, analysis_date))
        
        if len(market_data) == 0:
            return None
        
        market_cap = market_data.iloc[0]['market_cap']
        
        # Calculate value ratios
        pe_ratio = 0
        pb_ratio = 0
        pe_score = 0
        pb_score = 0
        
        if pd.notna(row['NetProfit_TTM']) and row['NetProfit_TTM'] > 0 and market_cap > 0:
            pe_ratio = market_cap / row['NetProfit_TTM']
            pe_score = 1 / pe_ratio if pe_ratio > 0 else 0
        
        if pd.notna(row['AvgTotalEquity']) and row['AvgTotalEquity'] > 0 and market_cap > 0:
            pb_ratio = market_cap / row['AvgTotalEquity']
            pb_score = 1 / pb_ratio if pb_ratio > 0 else 0
        
        return {
            'roaa': row['roaa'],
            'roae': row['roae'],
            'pe_ratio': pe_ratio,
            'pb_ratio': pb_ratio,
            'pe_score': pe_score,
            'pb_score': pb_score,
            'net_profit_ttm': row['NetProfit_TTM'],
            'total_equity': row['AvgTotalEquity'],
            'total_assets': row['AvgTotalAssets']
        }
    
    def get_market_data(self, ticker: str, analysis_date: datetime) -> dict:
        """Get market data for a specific ticker."""
        market_query = """
        SELECT trading_date, close_price, market_cap, total_shares
        FROM vcsc_daily_data_complete
        WHERE ticker = %s AND trading_date <= %s
        ORDER BY trading_date DESC
        LIMIT 1
        """
        
        market_data = pd.read_sql(market_query, self.engine, params=(ticker, analysis_date))
        
        if len(market_data) == 0:
            return None
        
        row = market_data.iloc[0]
        return {
            'market_cap': row['market_cap'],
            'close_price': row['close_price'],
            'total_shares': row['total_shares']
        }
    
    def get_forward_return(self, ticker: str, analysis_date: datetime) -> float:
        """Get forward return for a specific ticker."""
        try:
            current_date = analysis_date
            future_date = current_date + timedelta(days=30)
            
            current_query = """
            SELECT close_price 
            FROM vcsc_daily_data_complete 
            WHERE ticker = %s AND trading_date = %s
            """
            current_result = pd.read_sql(current_query, self.engine, params=(ticker, current_date))
            
            future_query = """
            SELECT close_price, trading_date
            FROM vcsc_daily_data_complete 
            WHERE ticker = %s AND trading_date >= %s
            ORDER BY trading_date ASC
            LIMIT 1
            """
            future_result = pd.read_sql(future_query, self.engine, params=(ticker, future_date))
            
            if len(current_result) > 0 and len(future_result) > 0:
                current_price = current_result.iloc[0]['close_price']
                future_price = future_result.iloc[0]['close_price']
                
                if current_price > 0:
                    forward_return = (future_price - current_price) / current_price
                else:
                    forward_return = 0
            else:
                forward_return = 0
                
            return forward_return
            
        except Exception as e:
            print(f"Error calculating forward return for {ticker}: {e}")
            return 0
    
    def analyze_contrarian_stocks(self, data: pd.DataFrame):
        """Analyze which stocks are showing contrarian behavior."""
        print("\n" + "="*100)
        print("CONTRARIAN STOCKS ANALYSIS: AGRICULTURE AND MINING & OIL")
        print("="*100)
        
        # Calculate correlations for each stock
        stock_analysis = []
        
        for _, stock in data.iterrows():
            # Calculate individual stock ICs (using the stock's own data point)
            # For individual stocks, we can't calculate correlation with just one point
            # Instead, we'll analyze the characteristics of contrarian stocks
            
            stock_analysis.append({
                'ticker': stock['ticker'],
                'sector': stock['sector'],
                'roaa': stock['roaa'],
                'roae': stock['roae'],
                'pe_ratio': stock['pe_ratio'],
                'pb_ratio': stock['pb_ratio'],
                'pe_score': stock['pe_score'],
                'pb_score': stock['pb_score'],
                'market_cap': stock['market_cap'],
                'forward_return': stock['forward_return'],
                'net_profit_ttm': stock['net_profit_ttm'],
                'total_equity': stock['total_equity'],
                'total_assets': stock['total_assets']
            })
        
        stock_df = pd.DataFrame(stock_analysis)
        
        # Analyze by sector
        for sector in ['Agriculture', 'Mining & Oil']:
            sector_data = stock_df[stock_df['sector'] == sector].copy()
            
            if len(sector_data) == 0:
                continue
            
            print(f"\n{sector.upper()} SECTOR ANALYSIS:")
            print(f"Total stocks: {len(sector_data)}")
            print(f"Mean ROAA: {sector_data['roaa'].mean():.4f}")
            print(f"Mean ROAE: {sector_data['roae'].mean():.4f}")
            print(f"Mean P/E Ratio: {sector_data['pe_ratio'].mean():.2f}")
            print(f"Mean P/B Ratio: {sector_data['pb_ratio'].mean():.2f}")
            print(f"Mean P/E Score: {sector_data['pe_score'].mean():.4f}")
            print(f"Mean P/B Score: {sector_data['pb_score'].mean():.4f}")
            print(f"Mean Forward Return: {sector_data['forward_return'].mean():.4f}")
            
            # Find most contrarian stocks (high P/E score but negative return, or low P/E score but positive return)
            sector_data['pe_contrarian_score'] = -sector_data['pe_score'] * sector_data['forward_return']
            sector_data['pb_contrarian_score'] = -sector_data['pb_score'] * sector_data['forward_return']
            
            # Sort by contrarian scores
            pe_contrarian = sector_data.nlargest(5, 'pe_contrarian_score')
            pb_contrarian = sector_data.nlargest(5, 'pb_contrarian_score')
            
            print(f"\nTop 5 Most Contrarian P/E Stocks in {sector}:")
            print(f"{'Ticker':<10} {'P/E Score':<10} {'Forward Return':<15} {'Contrarian Score':<15} {'ROAA':<10} {'Market Cap':<15}")
            print("-" * 85)
            
            for _, stock in pe_contrarian.iterrows():
                print(f"{stock['ticker']:<10} {stock['pe_score']:<10.4f} {stock['forward_return']:<15.4f} "
                      f"{stock['pe_contrarian_score']:<15.4f} {stock['roaa']:<10.4f} {stock['market_cap']:<15.0f}")
            
            print(f"\nTop 5 Most Contrarian P/B Stocks in {sector}:")
            print(f"{'Ticker':<10} {'P/B Score':<10} {'Forward Return':<15} {'Contrarian Score':<15} {'ROAA':<10} {'Market Cap':<15}")
            print("-" * 85)
            
            for _, stock in pb_contrarian.iterrows():
                print(f"{stock['ticker']:<10} {stock['pb_score']:<10.4f} {stock['forward_return']:<15.4f} "
                      f"{stock['pb_contrarian_score']:<15.4f} {stock['roaa']:<10.4f} {stock['market_cap']:<15.0f}")
            
            # Analyze characteristics of contrarian stocks
            print(f"\nCharacteristics of Contrarian Stocks in {sector}:")
            
            # High P/E score, negative return (overvalued)
            overvalued = sector_data[(sector_data['pe_score'] > sector_data['pe_score'].median()) & 
                                   (sector_data['forward_return'] < 0)]
            
            # Low P/E score, positive return (undervalued)
            undervalued = sector_data[(sector_data['pe_score'] < sector_data['pe_score'].median()) & 
                                    (sector_data['forward_return'] > 0)]
            
            print(f"Overvalued stocks (high P/E, negative return): {len(overvalued)}")
            if len(overvalued) > 0:
                print(f"  Mean ROAA: {overvalued['roaa'].mean():.4f}")
                print(f"  Mean Market Cap: {overvalued['market_cap'].mean():.0f}")
                print(f"  Tickers: {', '.join(overvalued['ticker'].tolist())}")
            
            print(f"Undervalued stocks (low P/E, positive return): {len(undervalued)}")
            if len(undervalued) > 0:
                print(f"  Mean ROAA: {undervalued['roaa'].mean():.4f}")
                print(f"  Mean Market Cap: {undervalued['market_cap'].mean():.0f}")
                print(f"  Tickers: {', '.join(undervalued['ticker'].tolist())}")
        
        return stock_df
    
    def create_contrarian_visualizations(self, data: pd.DataFrame):
        """Create visualizations for contrarian stocks analysis."""
        plt.style.use('seaborn-v0_8')
        
        # Create multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Contrarian Stocks Analysis: Agriculture and Mining & Oil Sectors', fontsize=16, fontweight='bold')
        
        # 1. P/E Score vs Forward Return by Sector
        ax1 = axes[0, 0]
        for sector in ['Agriculture', 'Mining & Oil']:
            sector_data = data[data['sector'] == sector]
            ax1.scatter(sector_data['pe_score'], sector_data['forward_return'], 
                       label=sector, alpha=0.7, s=100)
        
        ax1.set_xlabel('P/E Score')
        ax1.set_ylabel('Forward Return')
        ax1.set_title('P/E Score vs Forward Return')
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax1.legend()
        
        # Add trend line
        z = np.polyfit(data['pe_score'], data['forward_return'], 1)
        p = np.poly1d(z)
        ax1.plot(data['pe_score'], p(data['pe_score']), "r--", alpha=0.8)
        
        # 2. P/B Score vs Forward Return by Sector
        ax2 = axes[0, 1]
        for sector in ['Agriculture', 'Mining & Oil']:
            sector_data = data[data['sector'] == sector]
            ax2.scatter(sector_data['pb_score'], sector_data['forward_return'], 
                       label=sector, alpha=0.7, s=100)
        
        ax2.set_xlabel('P/B Score')
        ax2.set_ylabel('Forward Return')
        ax2.set_title('P/B Score vs Forward Return')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax2.legend()
        
        # Add trend line
        z = np.polyfit(data['pb_score'], data['forward_return'], 1)
        p = np.poly1d(z)
        ax2.plot(data['pb_score'], p(data['pb_score']), "r--", alpha=0.8)
        
        # 3. ROAA vs Forward Return by Sector
        ax3 = axes[1, 0]
        for sector in ['Agriculture', 'Mining & Oil']:
            sector_data = data[data['sector'] == sector]
            ax3.scatter(sector_data['roaa'], sector_data['forward_return'], 
                       label=sector, alpha=0.7, s=100)
        
        ax3.set_xlabel('ROAA')
        ax3.set_ylabel('Forward Return')
        ax3.set_title('ROAA vs Forward Return')
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax3.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax3.legend()
        
        # Add trend line
        z = np.polyfit(data['roaa'], data['forward_return'], 1)
        p = np.poly1d(z)
        ax3.plot(data['roaa'], p(data['roaa']), "r--", alpha=0.8)
        
        # 4. Market Cap vs Forward Return by Sector
        ax4 = axes[1, 1]
        for sector in ['Agriculture', 'Mining & Oil']:
            sector_data = data[data['sector'] == sector]
            ax4.scatter(sector_data['market_cap'] / 1e9, sector_data['forward_return'], 
                       label=sector, alpha=0.7, s=100)
        
        ax4.set_xlabel('Market Cap (Billions VND)')
        ax4.set_ylabel('Forward Return')
        ax4.set_title('Market Cap vs Forward Return')
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax4.legend()
        
        # Add trend line
        z = np.polyfit(data['market_cap'] / 1e9, data['forward_return'], 1)
        p = np.poly1d(z)
        ax4.plot(data['market_cap'] / 1e9, p(data['market_cap'] / 1e9), "r--", alpha=0.8)
        
        plt.tight_layout()
        plt.savefig('contrarian_stocks_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_contrarian_analysis(self, analysis_date: datetime = None):
        """Run comprehensive contrarian stocks analysis."""
        if analysis_date is None:
            analysis_date = datetime(2024, 12, 18)
        
        print(f"Running contrarian stocks analysis for {analysis_date}")
        
        # Get detailed data for Agriculture and Mining & Oil stocks
        data = self.get_contrarian_stocks_data(analysis_date)
        
        if len(data) == 0:
            print("No data found for Agriculture and Mining & Oil stocks")
            return
        
        print(f"Total stocks analyzed: {len(data)}")
        print("Sector breakdown:")
        print(data['sector'].value_counts())
        
        # Analyze contrarian stocks
        stock_df = self.analyze_contrarian_stocks(data)
        
        # Create visualizations
        self.create_contrarian_visualizations(data)
        
        return {
            'stock_data': stock_df,
            'raw_data': data
        }


def main():
    """Main execution function."""
    analyzer = ContrarianStocksAnalyzer()
    
    # Run analysis for date with sufficient historical data
    analysis_date = datetime(2024, 12, 18)
    
    results = analyzer.run_contrarian_analysis(analysis_date)
    
    if results:
        print("\nContrarian stocks analysis completed successfully!")
        print("Results saved to: contrarian_stocks_analysis.png")
    else:
        print("Contrarian stocks analysis failed")


if __name__ == "__main__":
    main() 