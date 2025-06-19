# core/data/storage.py

import sqlite3
import pandas as pd
from pathlib import Path
from configs.settings import Config
from typing import Union, Optional

class DataStorage:
    """统一数据存储管理器"""
    
    def __init__(self):
        self.db_path = Config.DATA_DIR / 'market_data.db'
        self._init_db()
        
    def _init_db(self):
        """初始化数据库表结构"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS candles (
                    symbol TEXT,
                    timeframe TEXT,
                    open_time DATETIME,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    PRIMARY KEY (symbol, timeframe, open_time)
                )
            """)
    
    def save_candles(self, symbol: str, timeframe: str, df: pd.DataFrame):
        """保存K线数据"""
        if df.empty:
            return
            
        df = df.copy()
        df['symbol'] = symbol
        df['timeframe'] = timeframe
        df['open_time'] = df.index
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # 启用事务和WAL模式提高并发性
                conn.execute("PRAGMA journal_mode=WAL")
                with conn:
                    # 使用参数化查询防止SQL注入
                    existing = pd.read_sql(
                        "SELECT 1 FROM candles WHERE symbol=? AND timeframe=? AND open_time=? LIMIT 1",
                        conn,
                        params=(symbol, timeframe, df.index[-1]),
                    )
                
                    if not existing.empty:
                        logger.warning(f"数据已存在: {symbol}-{timeframe}-{df.index[-1]}")
                        return
                
                    # 批量插入
                    df.to_sql('candles', conn, if_exists='append', index=False, 
                             method='multi', chunksize=1000)
        except sqlite3.Error as e:
            logger.error(f"保存数据失败: {e}")
            raise
    
    def load_candles(self, symbol: str, timeframe: str, 
                    start: Optional[str] = None, 
                    end: Optional[str] = None) -> pd.DataFrame:
        """加载K线数据"""
        query = f"""
            SELECT open_time, open, high, low, close, volume 
            FROM candles 
            WHERE symbol='{symbol}' AND timeframe='{timeframe}'
        """
        
        if start:
            query += f" AND open_time >= '{start}'"
        if end:
            query += f" AND open_time <= '{end}'"
            
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql(query, conn, parse_dates=['open_time'])
            
        if not df.empty:
            df.set_index('open_time', inplace=True)
            df.sort_index(inplace=True)
            
        return df
