"""
Data Loader Component
====================
Read-only data access for market intelligence reports.
Does not modify any existing data or tables.
"""

import pandas as pd
import mysql.connector
from datetime import datetime, timedelta
import logging
from typing import Optional, List, Dict, Tuple
import numpy as np

# Add project root to path for imports
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from production.market_intelligence.config import get_db_config, TABLE_MAPPINGS

logger = logging.getLogger(__name__)


class MarketDataLoader:
    """Read-only data loader for market intelligence"""
    
    def __init__(self):
        """Initialize with database connection"""
        self.db_config = get_db_config()
        self._conn = None
        
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()
        
    def connect(self):
        """Establish database connection"""
        try:
            self._conn = mysql.connector.connect(
                host=self.db_config['host'],
                database=self.db_config['schema_name'],
                user=self.db_config['username'],
                password=self.db_config['password']
            )
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
            
    def disconnect(self):
        """Close database connection"""
        if self._conn:
            self._conn.close()
            logger.info("Database connection closed")
            
    def get_latest_trading_date(self) -> datetime:
        """Get the most recent trading date with data"""
        query = f"""
        SELECT MAX(date) as latest_date
        FROM {TABLE_MAPPINGS['prices']}
        """
        df = pd.read_sql(query, self._conn)
        return pd.to_datetime(df['latest_date'].iloc[0])
        
    def get_market_overview(self, date: Optional[datetime] = None) -> Dict:
        """Get market overview statistics"""
        if date is None:
            date = self.get_latest_trading_date()
            
        # Get market breadth - MySQL compatible version
        query = f"""
        WITH latest_dates AS (
            SELECT DISTINCT date 
            FROM {TABLE_MAPPINGS['prices']} 
            WHERE date <= %s 
            ORDER BY date DESC 
            LIMIT 2
        ),
        price_changes AS (
            SELECT 
                eh.ticker,
                eh.close as current_close,
                LAG(eh.close) OVER (PARTITION BY eh.ticker ORDER BY eh.date) as prev_close
            FROM {TABLE_MAPPINGS['prices']} eh
            INNER JOIN latest_dates ld ON eh.date = ld.date
        )
        SELECT 
            COUNT(CASE WHEN current_close > prev_close THEN 1 END) as advances,
            COUNT(CASE WHEN current_close < prev_close THEN 1 END) as declines,
            COUNT(CASE WHEN current_close = prev_close THEN 1 END) as unchanged,
            AVG((current_close - prev_close) / prev_close * 100) as avg_change
        FROM price_changes
        WHERE prev_close IS NOT NULL
        """
        
        breadth_df = pd.read_sql(query, self._conn, params=[date])
        
        # Get volume analysis - simplified version for now
        volume_query = f"""
        SELECT 
            SUM(total_volume) as total_volume,
            AVG(total_volume) as avg_20d_volume,
            1.0 as volume_ratio
        FROM {TABLE_MAPPINGS['market_data']} v
        WHERE v.trading_date = %s
        """
        
        volume_df = pd.read_sql(volume_query, self._conn, params=[date])
        
        return {
            'date': date,
            'advances': int(breadth_df['advances'].iloc[0]),
            'declines': int(breadth_df['declines'].iloc[0]),
            'unchanged': int(breadth_df['unchanged'].iloc[0]),
            'avg_change': float(breadth_df['avg_change'].iloc[0]),
            'total_volume': float(volume_df['total_volume'].iloc[0]) if len(volume_df) > 0 else 0,
            'volume_ratio': float(volume_df['volume_ratio'].iloc[0]) if len(volume_df) > 0 else 1.0
        }
        
    def get_sector_performance(self, date: Optional[datetime] = None, lookback_days: int = 1) -> pd.DataFrame:
        """Get sector performance statistics"""
        if date is None:
            date = self.get_latest_trading_date()
            
        query = f"""
        WITH sector_returns AS (
            SELECT 
                mi.sector,
                eh.ticker,
                eh.close / LAG(eh.close, %s) OVER (PARTITION BY eh.ticker ORDER BY eh.date) - 1 as return_pct
            FROM {TABLE_MAPPINGS['prices']} eh
            JOIN {TABLE_MAPPINGS['company_info']} mi ON eh.ticker = mi.ticker
            WHERE eh.date = %s
        )
        SELECT 
            sector,
            COUNT(ticker) as stock_count,
            AVG(return_pct) * 100 as avg_return,
            STDDEV(return_pct) * 100 as return_std,
            MIN(return_pct) * 100 as min_return,
            MAX(return_pct) * 100 as max_return
        FROM sector_returns
        WHERE return_pct IS NOT NULL
        GROUP BY sector
        ORDER BY avg_return DESC
        """
        
        return pd.read_sql(query, self._conn, params=[lookback_days, date])
        
    def get_factor_performance(self, date: Optional[datetime] = None, lookback_days: int = 1) -> pd.DataFrame:
        """Get factor performance for Q, V, M"""
        if date is None:
            date = self.get_latest_trading_date()
            
        # Get factor quintile returns
        query = f"""
        WITH factor_quintiles AS (
            SELECT 
                f.ticker,
                f.date,
                f.quality_score,
                f.value_score,
                f.momentum_score,
                NTILE(5) OVER (PARTITION BY f.date ORDER BY f.quality_score DESC) as q_quintile,
                NTILE(5) OVER (PARTITION BY f.date ORDER BY f.value_score DESC) as v_quintile,
                NTILE(5) OVER (PARTITION BY f.date ORDER BY f.momentum_score DESC) as m_quintile
            FROM {TABLE_MAPPINGS['factors']} f
            WHERE f.date = DATE_SUB(%s, INTERVAL %s DAY)
                AND f.version = 'v2_enhanced'
        ),
        returns AS (
            SELECT 
                r1.ticker,
                (r2.close / r1.close - 1) * 100 as return_pct
            FROM {TABLE_MAPPINGS['prices']} r1
            JOIN {TABLE_MAPPINGS['prices']} r2 
                ON r1.ticker = r2.ticker 
                AND r2.date = %s
            WHERE r1.date = DATE_SUB(%s, INTERVAL %s DAY)
        )
        SELECT 
            'Quality' as factor,
            AVG(CASE WHEN fq.q_quintile = 1 THEN r.return_pct END) -
            AVG(CASE WHEN fq.q_quintile = 5 THEN r.return_pct END) as long_short_return,
            AVG(CASE WHEN fq.q_quintile = 1 THEN r.return_pct END) as top_quintile_return,
            AVG(CASE WHEN fq.q_quintile = 5 THEN r.return_pct END) as bottom_quintile_return
        FROM factor_quintiles fq
        JOIN returns r ON fq.ticker = r.ticker
        
        UNION ALL
        
        SELECT 
            'Value' as factor,
            AVG(CASE WHEN fq.v_quintile = 1 THEN r.return_pct END) -
            AVG(CASE WHEN fq.v_quintile = 5 THEN r.return_pct END) as long_short_return,
            AVG(CASE WHEN fq.v_quintile = 1 THEN r.return_pct END) as top_quintile_return,
            AVG(CASE WHEN fq.v_quintile = 5 THEN r.return_pct END) as bottom_quintile_return
        FROM factor_quintiles fq
        JOIN returns r ON fq.ticker = r.ticker
        
        UNION ALL
        
        SELECT 
            'Momentum' as factor,
            AVG(CASE WHEN fq.m_quintile = 1 THEN r.return_pct END) -
            AVG(CASE WHEN fq.m_quintile = 5 THEN r.return_pct END) as long_short_return,
            AVG(CASE WHEN fq.m_quintile = 1 THEN r.return_pct END) as top_quintile_return,
            AVG(CASE WHEN fq.m_quintile = 5 THEN r.return_pct END) as bottom_quintile_return
        FROM factor_quintiles fq
        JOIN returns r ON fq.ticker = r.ticker
        """
        
        return pd.read_sql(query, self._conn, params=[date, lookback_days, date, date, lookback_days])
        
    def get_foreign_flow_summary(self, date: Optional[datetime] = None, lookback_days: int = 5) -> pd.DataFrame:
        """Get foreign flow summary"""
        if date is None:
            date = self.get_latest_trading_date()
            
        query = f"""
        SELECT 
            date,
            SUM(buy_volume) as total_buy_volume,
            SUM(sell_volume) as total_sell_volume,
            SUM(net_volume) as total_net_volume,
            SUM(buy_value) as total_buy_value,
            SUM(sell_value) as total_sell_value,
            SUM(net_value) as total_net_value
        FROM {TABLE_MAPPINGS['foreign_flows']}
        WHERE date > DATE_SUB(%s, INTERVAL %s DAY)
            AND date <= %s
        GROUP BY date
        ORDER BY date DESC
        """
        
        return pd.read_sql(query, self._conn, params=[date, lookback_days, date])
        
    def get_risk_metrics(self, date: Optional[datetime] = None, lookback_days: int = 252) -> Dict:
        """Calculate market risk metrics"""
        if date is None:
            date = self.get_latest_trading_date()
            
        # Get VN-Index returns for volatility calculation
        query = f"""
        SELECT 
            date,
            close
        FROM {TABLE_MAPPINGS['prices']}
        WHERE ticker = 'VN-INDEX'
            AND date > DATE_SUB(%s, INTERVAL %s DAY)
            AND date <= %s
        ORDER BY date
        """
        
        vnindex_df = pd.read_sql(query, self._conn, params=[date, lookback_days, date])
        
        if len(vnindex_df) > 20:
            vnindex_df['returns'] = vnindex_df['close'].pct_change()
            
            # Calculate volatility metrics
            current_vol = vnindex_df['returns'].tail(20).std() * np.sqrt(252) * 100
            historical_vol = vnindex_df['returns'].std() * np.sqrt(252) * 100
            vol_percentile = (vnindex_df['returns'].rolling(20).std().rank(pct=True).iloc[-1] * 100)
            
            # Get correlation matrix for top stocks
            top_stocks_query = f"""
            SELECT ticker
            FROM {TABLE_MAPPINGS['market_data']}
            WHERE date = %s
            ORDER BY market_cap DESC
            LIMIT 20
            """
            
            top_stocks = pd.read_sql(top_stocks_query, self._conn, params=[date])['ticker'].tolist()
            
            if top_stocks:
                # Get returns for correlation calculation
                returns_query = f"""
                SELECT 
                    date,
                    ticker,
                    close
                FROM {TABLE_MAPPINGS['prices']}
                WHERE ticker IN ({','.join(['%s'] * len(top_stocks))})
                    AND date > DATE_SUB(%s, INTERVAL 60 DAY)
                    AND date <= %s
                ORDER BY ticker, date
                """
                
                params = top_stocks + [date, date]
                returns_df = pd.read_sql(returns_query, self._conn, params=params)
                
                # Pivot and calculate correlation
                returns_pivot = returns_df.pivot(index='date', columns='ticker', values='close').pct_change()
                current_corr = returns_pivot.tail(20).corr().values
                avg_correlation = current_corr[np.triu_indices_from(current_corr, k=1)].mean()
            else:
                avg_correlation = 0.5
                
        else:
            current_vol = 20.0
            historical_vol = 20.0
            vol_percentile = 50.0
            avg_correlation = 0.5
            
        return {
            'current_volatility': current_vol,
            'historical_volatility': historical_vol,
            'volatility_percentile': vol_percentile,
            'average_correlation': avg_correlation,
            'volatility_regime': 'High' if vol_percentile > 80 else 'Normal' if vol_percentile > 20 else 'Low'
        }
        
    def get_top_signals(self, date: Optional[datetime] = None, signal_type: str = 'all') -> List[Dict]:
        """Get top trading signals"""
        if date is None:
            date = self.get_latest_trading_date()
            
        signals = []
        
        # Mean reversion signals
        if signal_type in ['all', 'mean_reversion']:
            mr_query = f"""
            WITH price_stats AS (
                SELECT 
                    ticker,
                    close,
                    AVG(close) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) as ma20,
                    STDDEV(close) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) as std20
                FROM {TABLE_MAPPINGS['prices']}
                WHERE date <= %s
                    AND date > DATE_SUB(%s, INTERVAL 30 DAY)
            ),
            latest_prices AS (
                SELECT 
                    ps.ticker,
                    ps.close,
                    ps.ma20,
                    ps.std20,
                    (ps.close - ps.ma20) / ps.std20 as z_score,
                    v.volume,
                    v.avg_volume_20d
                FROM price_stats ps
                JOIN {TABLE_MAPPINGS['market_data']} v ON ps.ticker = v.ticker
                WHERE v.date = %s
                    AND ps.close = (SELECT close FROM {TABLE_MAPPINGS['prices']} WHERE ticker = ps.ticker AND date = %s)
            )
            SELECT 
                ticker,
                close,
                ma20,
                z_score,
                volume / avg_volume_20d as volume_ratio
            FROM latest_prices
            WHERE ABS(z_score) > 2.0
                AND volume > avg_volume_20d * 0.5
            ORDER BY ABS(z_score) DESC
            LIMIT 5
            """
            
            mr_df = pd.read_sql(mr_query, self._conn, params=[date, date, date, date])
            
            for _, row in mr_df.iterrows():
                signals.append({
                    'type': 'mean_reversion',
                    'ticker': row['ticker'],
                    'signal': 'Long' if row['z_score'] < -2 else 'Short',
                    'z_score': round(row['z_score'], 2),
                    'current_price': row['close'],
                    'target_price': row['ma20']
                })
                
        return signals[:10]  # Return top 10 signals
        
    def get_liquid_universe(self, date: Optional[datetime] = None, top_n: int = 200) -> List[str]:
        """Get liquid universe tickers"""
        if date is None:
            date = self.get_latest_trading_date()
            
        query = f"""
        SELECT 
            ticker,
            market_cap,
            avg_volume_20d * close as avg_daily_value
        FROM {TABLE_MAPPINGS['market_data']}
        WHERE date = %s
            AND market_cap > 0
            AND avg_volume_20d > 0
        ORDER BY avg_daily_value DESC
        LIMIT %s
        """
        
        df = pd.read_sql(query, self._conn, params=[date, top_n])
        return df['ticker'].tolist()