# core/strategies/rl/environment.py

import logging
import numpy as np
import pandas as pd
from typing import Tuple
from utils.logger import setup_logger

logger = setup_logger('trading_env')

class TradingEnvironment:
    """DQN训练环境"""
    
    def __init__(self, data: pd.DataFrame, window_size: int = 60):
        self.data = data
        self.window_size = window_size
        self.current_step = window_size
        self.max_steps = len(data) - window_size - 1
        
    def reset(self) -> np.ndarray:
        """重置环境"""
        self.current_step = self.window_size
        return self._get_observation(self.current_step)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        执行动作
        返回: (next_state, reward, done, info)
        """
        logger.info(f"执行步骤 {self.current_step}/{self.max_steps}, 动作: {action}")
        if self.current_step >= self.max_steps:
            raise IndexError("Episode已经结束")
            
        # 执行动作 (0=持有, 1=买入, 2=卖出)
        current_price = self.data.iloc[self.current_step]['close']
        next_price = self.data.iloc[self.current_step + 1]['close']
        
        # 计算回报
        if action == 1:  # 买入
            reward = (next_price - current_price) / current_price
        elif action == 2:  # 卖出
            reward = (current_price - next_price) / current_price
        else:  # 持有
            reward = -0.001  # 小惩罚鼓励交易
            
        # 更新状态
        self.current_step += 1
        next_state = self._get_observation(self.current_step)
        done = self.current_step >= self.max_steps
        
        # 返回结果
        return next_state, reward, done, {
            'current_price': current_price,
            'action': action
        }
    
    def _get_observation(self, step: int) -> np.ndarray:
        """获取当前观察状态"""
        start_idx = step - self.window_size
        end_idx = step
        return self.data.iloc[start_idx:end_idx].values
