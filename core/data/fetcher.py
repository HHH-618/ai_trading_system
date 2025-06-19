# core/data/fetcher.py

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from configs.settings import Config
from configs.constants import TimeConstants
from utils.logger import setup_logger

logger = setup_logger('data_fetcher')

class DataFetcher:
    """多源数据获取器，支持MT5和API数据源"""
    
    def __init__(self):
        self._init_mt5()
        self.cache = {}
        self.symbol_info = {}
        
    def _init_mt5(self) -> bool:
        """初始化MT5连接"""
        if not mt5.initialize():
            logger.error(f"MT5初始化失败: {mt5.last_error()}")
            return False
        logger.info("MT5连接成功")
        return True
        
    def get_historical_data(self, symbol: str, timeframe: str, 
                          bars: int = 1000, from_date: datetime = None) -> Optional[pd.DataFrame]:
        """
        获取历史K线数据
        :param symbol: 交易品种
        :param timeframe: 时间框架
        :param bars: 获取的K线数量
        :param from_date: 开始日期
        """
        try:
            # 检查缓存
            cache_key = f"{symbol}_{timeframe}"
            if cache_key in self.cache:
                return self.cache[cache_key]
                
            # 确定时间范围
            if from_date is None:
                rates = mt5.copy_rates_from_pos(symbol, self._parse_timeframe(timeframe), 0, bars)
            else:
                rates = mt5.copy_rates_range(symbol, self._parse_timeframe(timeframe), from_date, datetime.now())
                
            if rates is None:
                logger.error(f"获取{symbol}数据失败: {mt5.last_error()}")
                return None
                
            # 转换为DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # 缓存数据
            self.cache[cache_key] = df
            return df
            
        except Exception as e:
            logger.error(f"获取历史数据异常: {str(e)}")
            return None
            
    def get_realtime_tick(self, symbol: str) -> Optional[Dict]:
        """获取实时tick数据"""
        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return None
                
            return {
                'time': pd.to_datetime(tick.time, unit='s'),
                'bid': tick.bid,
                'ask': tick.ask,
                'last': tick.last,
                'volume': tick.volume
            }
        except Exception as e:
            logger.error(f"获取tick数据异常: {str(e)}")
            return None
            
    def get_market_depth(self, symbol: str) -> Optional[Dict]:
        """获取市场深度数据"""
        try:
            depth = mt5.market_book_get(symbol)
            if depth is None:
                return None
                
            return {
                'bids': [(item.price, item.volume) for item in depth.bids],
                'asks': [(item.price, item.volume) for item in depth.asks]
            }
        except Exception as e:
            logger.error(f"获取市场深度异常: {str(e)}")
            return None
            
    def _parse_timeframe(self, timeframe: str) -> int:
        """将字符串时间帧转换为MT5常量"""
        tf_mapping = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1
        }
        return tf_mapping.get(timeframe, mt5.TIMEFRAME_H1)
        
    def shutdown(self):
        """关闭连接"""
        mt5.shutdown()
        logger.info("MT5连接已关闭")

# 单例模式
data_fetcher = DataFetcher()
