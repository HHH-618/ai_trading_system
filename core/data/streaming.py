# core/data/streaming.py

import pandas as pd
import numpy as np
from collections import deque
from threading import Thread, Lock
from queue import Queue
from core.utils.helpers import safe_divide

class DataStreamer:
    """实时数据流处理器，支持多时间帧合成"""
    
    def __init__(self, symbol: str, timeframes: list):
        self.symbol = symbol
        self.timeframes = timeframes
        self.buffers = {tf: deque(maxlen=1000) for tf in timeframes}
        self.locks = {tf: Lock() for tf in timeframes}
        self.data_queue = Queue()
        self._stop_event = False
        
    def start(self):
        """启动数据流线程"""
        Thread(target=self._run, daemon=True).start()
        
    def _run(self):
        """主处理循环"""
        while not self._stop_event:
            try:
                # 从API/MT5获取原始tick数据
                raw_ticks = self._fetch_ticks()  
                
                # 多时间帧处理
                for tf in self.timeframes:
                    self._process_timeframe(raw_ticks, tf)
                    
            except Exception as e:
                print(f"Data streaming error: {str(e)}")
                time.sleep(1)
    
    def _process_timeframe(self, ticks: list, timeframe: str):
        """将tick数据合成指定时间帧的K线"""
        with self.locks[timeframe]:
            for tick in ticks:
                if not self.buffers[timeframe]:
                    # 新建K线
                    new_candle = self._create_candle(tick)
                    self.buffers[timeframe].append(new_candle)
                else:
                    last_candle = self.buffers[timeframe][-1]
                    if self._should_close_candle(last_candle, tick, timeframe):
                        # 闭合当前K线并新建
                        closed_candle = self._close_candle(last_candle, tick)
                        self.data_queue.put((timeframe, closed_candle))
                        new_candle = self._create_candle(tick)
                        self.buffers[timeframe].append(new_candle)
                    else:
                        # 更新当前K线
                        self._update_candle(last_candle, tick)
    
    def _should_close_candle(self, candle: dict, tick: dict, timeframe: str) -> bool:
        """判断是否应该闭合当前K线"""
        timeframe_minutes = {
            'M1': 1, 'M5': 5, 'M15': 15, 
            'H1': 60, 'H4': 240, 'D1': 1440
        }
        candle_duration = (tick['time'] - candle['open_time']).total_seconds() / 60
        return candle_duration >= timeframe_minutes.get(timeframe, 5)
    
    def get_latest(self, timeframe: str, n: int = 1) -> list:
        """获取最新n根K线"""
        with self.locks[timeframe]:
            return list(self.buffers[timeframe])[-n:]
