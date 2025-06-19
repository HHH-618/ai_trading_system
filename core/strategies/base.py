# core/strategies/base.py

from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional, Tuple
from configs.settings import Config
from utils.logger import setup_logger

logger = setup_logger('strategy_base')

class BaseStrategy(ABC):
    """
    策略基类，定义所有交易策略的通用接口
    """
    
    def __init__(self):
        self.name = self.__class__.__name__
        self.required_indicators = []  # 子类应设置所需指标
        self.min_data_length = 60      # 最小数据长度
        
    @abstractmethod
    def generate_signal(self, df: pd.DataFrame) -> Tuple[Optional[str], Optional[float]]:
        """
        生成交易信号
        返回: (action, confidence)
        - action: 'buy', 'sell' 或 None
        - confidence: 0-1之间的置信度
        """
        pass
        
    def calculate_position_size(self, price: float, risk_per_trade: float = 0.01) -> float:
        """
        基于风险的仓位计算
        :param price: 当前价格
        :param risk_per_trade: 单笔交易风险比例
        """
        account_info = self._get_account_info()
        if account_info is None:
            return Config.MIN_VOLUME
            
        risk_amount = account_info['balance'] * risk_per_trade
        stop_loss_pips = price * Config.STOP_LOSS_PCT
        volume = risk_amount / (stop_loss_pips * 10)  # 假设1手每点10美元
        
        return max(Config.MIN_VOLUME, min(round(volume, 2), Config.MAX_VOLUME))
        
    def get_stop_loss_take_profit(self, price: float, action: str) -> Tuple[float, float]:
        """
        计算动态止损止盈水平
        :param price: 入场价格
        :param action: 'buy'或'sell'
        """
        if action == 'buy':
            sl = price * (1 - Config.STOP_LOSS_PCT)
            tp = price * (1 + Config.TAKE_PROFIT_PCT)
        else:
            sl = price * (1 + Config.STOP_LOSS_PCT)
            tp = price * (1 - Config.TAKE_PROFIT_PCT)
            
        return sl, tp
        
    def _get_account_info(self) -> Optional[dict]:
        """获取账户信息(模拟实现)"""
        # 实际实现应连接交易平台API
        return {
            'balance': 10000,
            'equity': 10500,
            'margin': 3000
        }
        
    def _validate_data(self, df: pd.DataFrame) -> bool:
        """验证数据是否满足策略要求"""
        if df is None or len(df) < self.min_data_length:
            logger.warning(f"数据不足，需要至少{self.min_data_length}条")
            return False
            
        missing_cols = [col for col in self.required_indicators if col not in df.columns]
        if missing_cols:
            logger.warning(f"缺少必要指标: {missing_cols}")
            return False
            
        return True
